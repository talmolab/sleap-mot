"""Pipeline runner for sleap-mot tracking pipelines.

Loads YAML config files, instantiates tracker layers, and runs
multi-layer tracking pipelines with explanation capture.
"""

import inspect
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sleap_io as sio
import yaml

from sleap_mot.io.slpt import SLPTFile
from sleap_mot.tracking.instance_explanations import InstanceExplanationStore

# Registry mapping short names to tracker classes.
# Imports are deferred to avoid circular imports and heavy import cost at module level.
TRACKER_REGISTRY: Dict[str, type] = {}

_REGISTRY_LOADED = False


def _ensure_registry():
    """Populate TRACKER_REGISTRY on first use."""
    global _REGISTRY_LOADED
    if _REGISTRY_LOADED:
        return
    _REGISTRY_LOADED = True

    from sleap_mot.tracking.online_tracking import (
        MotionTracker,
        DirectionalMotionTracker,
        FacingConsistencyTracker,
        GeneralOnlineTracker,
        TrackletStitcher,
    )
    from sleap_mot.tracking.feature_tracking.tail_tracking import TailFeatureTracker
    from sleap_mot.tracking.feature_tracking.RFID_tracking import (
        RFIDFeatureTracker,
        CoordinateRFIDTracker,
    )
    from sleap_mot.tracking.feature_tracking.visual_patch_tracking import (
        VisualPatchTracker,
    )

    TRACKER_REGISTRY.update(
        {
            "MotionTracker": MotionTracker,
            "DirectionalMotionTracker": DirectionalMotionTracker,
            "FacingConsistencyTracker": FacingConsistencyTracker,
            "GeneralOnlineTracker": GeneralOnlineTracker,
            "TrackletStitcher": TrackletStitcher,
            "TailFeatureTracker": TailFeatureTracker,
            "RFIDFeatureTracker": RFIDFeatureTracker,
            "CoordinateRFIDTracker": CoordinateRFIDTracker,
            "VisualPatchTracker": VisualPatchTracker,
        }
    )


def _interpolate_variables(value: Any, variables: Dict[str, str]) -> Any:
    """Replace ${input.video} / ${input.labels} in string values.

    Args:
        value: A param value (string, list, dict, or primitive).
        variables: Mapping like {"input.video": "/path/to/video.mp4"}.

    Returns:
        Value with interpolated strings.
    """
    if isinstance(value, str):
        for var_name, var_value in variables.items():
            value = value.replace(f"${{{var_name}}}", str(var_value))
        return value
    if isinstance(value, list):
        return [_interpolate_variables(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate_variables(v, variables) for k, v in value.items()}
    return value


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> dict:
    """Load and validate a YAML pipeline config.

    Args:
        path: Path to the YAML config file.
        overrides: Optional dict with keys like "labels", "video", "output",
            "save_slp" to override top-level config values.

    Returns:
        Validated config dict with variable interpolation applied.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is missing required fields.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Config file is empty")

    # Validate required top-level keys
    if "input" not in config or "labels" not in config.get("input", {}):
        raise ValueError("Config must specify input.labels")
    if "pipeline" not in config or not config["pipeline"]:
        raise ValueError("Config must specify pipeline with at least one layer")

    # Apply CLI overrides
    if overrides:
        if overrides.get("labels"):
            config["input"]["labels"] = overrides["labels"]
        if overrides.get("video"):
            config.setdefault("input", {})["video"] = overrides["video"]
        if overrides.get("output"):
            config.setdefault("output", {})["path"] = overrides["output"]
        if overrides.get("save_slp") is not None:
            config.setdefault("output", {})["save_slp"] = overrides["save_slp"]

    # Build variable map for interpolation
    variables = {
        "input.labels": config["input"]["labels"],
    }
    if config["input"].get("video"):
        variables["input.video"] = config["input"]["video"]

    # Interpolate variables in pipeline params
    for layer_cfg in config["pipeline"]:
        if "params" in layer_cfg:
            layer_cfg["params"] = _interpolate_variables(
                layer_cfg["params"], variables
            )

    # Validate layer names against registry
    _ensure_registry()
    for i, layer_cfg in enumerate(config["pipeline"]):
        layer_name = layer_cfg.get("layer")
        if not layer_name:
            raise ValueError(f"Pipeline layer {i} missing 'layer' field")
        if layer_name not in TRACKER_REGISTRY:
            available = ", ".join(sorted(TRACKER_REGISTRY.keys()))
            raise ValueError(
                f"Unknown tracker '{layer_name}' in pipeline layer {i}. "
                f"Available trackers: {available}"
            )

    # Set output defaults
    config.setdefault("output", {})
    if "path" not in config["output"]:
        # Default output path based on input
        input_path = Path(config["input"]["labels"])
        config["output"]["path"] = str(
            input_path.parent / (input_path.stem + "_tracked.slpt")
        )
    config["output"].setdefault("save_slp", False)

    return config


def split_params(tracker_cls: type, params: Dict[str, Any]) -> Tuple[dict, dict]:
    """Split params into constructor args vs track() args.

    Inspects the tracker class __init__ signature to determine which
    params are constructor arguments. The rest go to .track().

    Args:
        tracker_cls: The tracker class.
        params: Combined parameter dict from config.

    Returns:
        (constructor_args, track_args) tuple.
    """
    init_sig = inspect.signature(tracker_cls.__init__)
    init_params = set(init_sig.parameters.keys()) - {"self", "kwargs"}
    # Also check MRO for inherited __init__ params
    for cls in tracker_cls.__mro__:
        if cls is object:
            continue
        if "__init__" in cls.__dict__:
            sig = inspect.signature(cls.__init__)
            init_params |= set(sig.parameters.keys()) - {"self", "kwargs"}

    constructor_args = {k: v for k, v in params.items() if k in init_params}
    track_args = {k: v for k, v in params.items() if k not in init_params}
    return constructor_args, track_args


def _track_accepts_param(tracker_cls: type, param_name: str) -> bool:
    """Check if a tracker's .track() method accepts a given parameter.

    Args:
        tracker_cls: The tracker class.
        param_name: Parameter name to check.

    Returns:
        True if .track() accepts this parameter (explicitly or via **kwargs).
    """
    # Walk MRO to find the actual track method
    for cls in tracker_cls.__mro__:
        if "track" in cls.__dict__:
            sig = inspect.signature(cls.track)
            params = sig.parameters
            # Accepts explicitly
            if param_name in params:
                return True
            # Accepts via **kwargs
            for p in params.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    return True
            return False
    return False


def run_pipeline(config: dict) -> Path:
    """Run a full tracking pipeline from a validated config.

    Args:
        config: Validated config dict (from load_config).

    Returns:
        Path to the output .slpt file.
    """
    _ensure_registry()

    # 1. Load labels
    labels_path = config["input"]["labels"]
    if not Path(labels_path).exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    print(f"Loading labels: {labels_path}")
    labels = sio.load_slp(labels_path)
    print(f"  {len(labels.labeled_frames)} frames, "
          f"{len(labels.tracks)} tracks")

    # Load video if specified
    video_path = config["input"].get("video")
    if video_path and not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 2. Create shared explanation store
    store = InstanceExplanationStore()

    # 3. Run each layer
    total_start = time.time()
    for i, layer_cfg in enumerate(config["pipeline"]):
        layer_type = layer_cfg["layer"]
        layer_name = layer_cfg.get("name", layer_type)
        priority = layer_cfg.get("priority")
        params = dict(layer_cfg.get("params", {}))

        print(f"\n[{i + 1}/{len(config['pipeline'])}] {layer_name} ({layer_type})")

        tracker_cls = TRACKER_REGISTRY[layer_type]

        # Inject priority and name into params if not already there
        if priority is not None:
            params.setdefault("priority", priority)
        params.setdefault("name", layer_name)

        # Split into constructor vs track args
        constructor_args, track_args = split_params(tracker_cls, params)

        # Auto-inject video_path into track_args if available and accepted
        if (
            video_path
            and "video_path" not in track_args
            and _track_accepts_param(tracker_cls, "video_path")
        ):
            track_args["video_path"] = video_path

        # Instantiate tracker
        tracker = tracker_cls(**constructor_args)

        # Set explanation store
        tracker.explanation_store = store

        # Run tracking
        layer_start = time.time()
        labels = tracker.track(labels, **track_args) or labels
        elapsed = time.time() - layer_start
        print(f"  Done in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\nPipeline complete in {total_elapsed:.1f}s")

    # 4. Save .slpt
    output_path = Path(config["output"]["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving: {output_path}")
    slpt = SLPTFile.from_labels(
        labels,
        explanation_store=store,
        source_path=config["input"]["labels"],
    )

    # Record pipeline layers in metadata
    for i, layer_cfg in enumerate(config["pipeline"]):
        layer_type = layer_cfg["layer"]
        layer_name = layer_cfg.get("name", layer_type)
        priority = layer_cfg.get("priority", 0)
        slpt.add_layer_to_pipeline(
            name=layer_name,
            class_name=f"sleap_mot.{layer_type}",
            priority=priority,
            config=layer_cfg.get("params", {}),
        )

    slpt.save(output_path)

    stats = slpt.get_statistics()
    print(f"  {stats.get('n_frames', '?')} frames, "
          f"{stats.get('n_explanations', '?')} explanations")

    # 5. Optionally save .slp
    if config["output"].get("save_slp"):
        slp_path = output_path.with_suffix(".slp")
        print(f"Saving SLP: {slp_path}")
        slpt_labels = slpt.to_labels()
        sio.save_slp(slpt_labels, slp_path)

    return output_path
