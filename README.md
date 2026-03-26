# sleap-mot
Standalone multi-object tracking for the SLEAP ecosystem.

## Installation

For development, use one of the following syntaxes:
```
conda env create -f environment.yml
```
```
pip install -e .[dev]
```
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more information on development.

## CLI Usage

After installation, run multi-layer tracking pipelines from a YAML config:

```bash
sleap-mot run config.yaml
```

### CLI Options

```bash
sleap-mot run config.yaml \
  --labels path/to/predictions.slp \   # override input.labels
  --video path/to/video.mp4 \          # override input.video
  --output path/to/output.slpt \       # override output.path
  --save-slp                           # also save a plain .slp file
```

### YAML Config Format

```yaml
input:
  labels: "path/to/predictions.slp"
  video: "path/to/video.mp4"           # optional, used by trackers that need it

output:
  path: "path/to/output.slpt"          # always saves .slpt
  save_slp: false                       # optionally also save a plain .slp

pipeline:
  - layer: MotionTracker
    priority: 5
    name: "MotionTrackletGenerator"
    params:
      long_kde_path: "kde_long.joblib"
      short_kde_path: "kde_short.joblib"
      iou_threshold: 0.1
      max_match_distance: 30

  - layer: TailFeatureTracker
    priority: 10
    name: "TailReID"
    params:
      tail_nodes: ["Tail_0", "Tail_1", "TailTip"]
      n_clusters: 8
      video_path: "${input.video}"      # reference top-level input
```

### Available Trackers

| Name | Description |
|------|-------------|
| `MotionTracker` | KDE-based motion prediction |
| `DirectionalMotionTracker` | Motion tracker with pose-based facing direction |
| `FacingConsistencyTracker` | Motion tracker with facing direction consistency |
| `GeneralOnlineTracker` | Configurable tracker with multiple feature/scoring options |
| `TrackletStitcher` | Merges fragmented tracklets based on spatial/temporal continuity |
| `TailFeatureTracker` | Tail segment pattern clustering for identity matching |
| `RFIDFeatureTracker` | RFID heatmap-based tracking |
| `CoordinateRFIDTracker` | Coordinate-based RFID tracking |
| `VisualPatchTracker` | Visual patch re-identification |

### Variable Interpolation

Use `${input.video}` and `${input.labels}` in layer params to reference top-level paths without duplication.

### Parameter Routing

Each layer's `params` are automatically split between the tracker constructor and its `.track()` method based on signature inspection. Parameters matching `__init__` arguments go to the constructor; the rest are passed to `.track()`. If `input.video` is set and a tracker's `.track()` accepts `video_path`, it is injected automatically.

## Tracking Architecture

sleap-mot provides a layered tracking architecture where different tracking methods can be combined with priority-based conflict resolution.

### Class Hierarchy

```
TrackingLayer (Abstract Base)
├── FeatureTracker (for probability-based trackers)
│   ├── RFIDFeatureTracker (heatmap-based)
│   ├── CoordinateRFIDTracker (coordinate-based, ~95% accuracy)
│   ├── FurColorFeatureTracker
│   ├── TailTattooFeatureTracker
│   └── TailFeatureTracker (tail segment patterns, ~97% accuracy)
└── OnlineTrackingLayer (for frame-by-frame trackers)
    ├── MotionTracker (KDE-based motion prediction)
    │   └── DirectionalMotionTracker (with pose-based facing direction)
    └── TrackletStitcher (merges fragmented tracklets)
```

### Explanation System Hierarchy

```
BaseDecisionRecord (Abstract Base)
├── MotionDecisionRecord
│   └── DirectionalMotionDecisionRecord
├── RFIDDecisionRecord
├── RFIDNoMatchBroadcast
├── PropagationRecord
├── StitchDecisionRecord
├── SwitchRepairDecisionRecord (identity switch detection/repair)
└── TailDecisionRecord (tail segment pattern clustering)

Supporting Data Structures:
├── IdentityVote (single vote from a feature tracker)
├── TrackletSegment (temporal segment with consistent identity)
├── DetectedSwitch (detected identity switch point)
└── TrackletIdentityAnalysis (full analysis results for a tracklet)
```

### Key Concepts

- **Priority-based conflict resolution**: Each tracker has a priority level (e.g., RFID=10, FurColor=8, Centroid=5). Higher priority trackers can override lower priority assignments.
- **Tracklet-based assignment**: Feature trackers can use existing tracklets (connected sequences of instances) for majority voting when assigning identities.
- **Bidirectional propagation**: Identity changes propagate both forward and backward through connected frames.

## Priority-Based Inter-Layer Conflict Resolution

sleap-mot provides sophisticated inter-layer conflict resolution that allows trackers to work together without conflicts. Understanding the distinction between **renaming** and **slicing** operations is crucial.

### Two Types of Track Operations

#### 1. Renaming a Tracklet (ALWAYS allowed for temporary tracks)

Renaming changes a tracklet's identity across ALL frames where it appears. For temporary tracklets (those created by motion trackers), this is **ALWAYS allowed** regardless of priority.

```python
# Example: RFID tracker (priority=10) renames a motion tracklet (priority=15)
# Motion tracker creates: tracklet_1 spanning frames 1-10
# RFID tracker renames entire tracklet to "RFID_A"
# Result: ALL frames 1-10 now have identity "RFID_A"
```

This is the typical workflow:
1. Motion tracker creates temporary tracklets with high confidence
2. RFID tracker identifies which animal each tracklet belongs to
3. RFID tracker renames the entire tracklet to the animal's identity

**API**: Use `rename_track_globally()` for rename operations.

#### 2. Slicing a Tracklet (Priority-controlled)

Slicing breaks a tracklet into separate identities at certain frames. This is **subject to priority rules** - lower priority cannot slice higher priority tracklets.

```python
# Example: Motion tracker (priority=15) creates tracklet_1 spanning frames 1-10
# A lower-priority tracker (priority=10) wants to:
#   - Assign frames 1-5 to "Animal_A"
#   - Assign frames 6-10 to "Animal_B"
# This is BLOCKED because it would slice a higher-priority tracklet
```

**API**: Use `assign_with_priority_resolution()` with `propagate_to_tracklet=False` for slice operations.

### Operation Summary

| Operation | Temporary Track | Global Track |
|-----------|-----------------|--------------|
| Rename (propagate to ALL frames) | Always allowed | Priority-controlled |
| Slice (different identities) | Priority-controlled | Priority-controlled |

### Using the Priority Resolution API

#### rename_track_globally()

Use this when you want to change a tracklet's identity across ALL frames:

```python
# In a custom tracker
success = self.rename_track_globally(
    labels=labels,
    old_track_name="tracklet_1",
    new_track_name="Animal_A",
    reason="RFID identification",
)
# Returns True if rename succeeded, False if blocked by priority
```

#### assign_with_priority_resolution()

Use this for individual instance assignments with proper conflict resolution:

```python
# Assign with propagation to entire tracklet (RENAME operation)
success = self.assign_with_priority_resolution(
    labels=labels,
    frame_idx=100,
    instance_idx=0,
    new_track_name="Animal_A",
    reason="RFID match",
    propagate_to_tracklet=True,  # Will rename entire tracklet
)

# Assign without propagation (SLICE operation)
success = self.assign_with_priority_resolution(
    labels=labels,
    frame_idx=100,
    instance_idx=0,
    new_track_name="Animal_B",
    reason="Override assignment",
    propagate_to_tracklet=False,  # Only this instance, subject to priority
)
```

## Implementing New Tracking Layers

To implement a custom tracking layer, follow these requirements:

### Requirements

1. **Inherit from `TrackingLayer`** (or a subclass like `OnlineTrackingLayer` or `FeatureTracker`)

2. **Use priority-based assignment methods** instead of direct track assignment:
   - `assign_with_priority_resolution()` for individual instance assignments
   - `rename_track_globally()` for tracklet renaming operations

3. **Set appropriate priority level** based on tracker confidence:

   | Priority | Tracker Type | Examples |
   |----------|--------------|----------|
   | 10 | Feature-based (high confidence) | RFID, visual markers |
   | 8 | Feature-based (medium confidence) | Fur color, tail tattoo |
   | 6 | Tracklet stitching | TrackletStitcher |
   | 5 | Motion-based tracklet generation | MotionTracker, DirectionalMotionTracker |
   | 1 | Fallback/default | Simple centroid tracker |

4. **Implement `get_config()`** returning all parameters for pipeline reconstruction

### Example: Custom Tracker

```python
from sleap_mot.tracking.base import TrackingLayer, TrackContext
from typing import Optional, Dict, Any
import sleap_io as sio

class MyCustomTracker(TrackingLayer):
    def __init__(
        self,
        priority: int = 7,
        name: str = "MyCustomTracker",
        my_threshold: float = 0.5,
    ):
        super().__init__(priority=priority, name=name, temporary=False)
        self.my_threshold = my_threshold

    def track(self, labels: sio.Labels, **kwargs) -> sio.Labels:
        # Build frame mapping for efficient lookup
        frame_map = {lf.frame_idx: lf for lf in labels.labeled_frames}

        for lf in labels.labeled_frames:
            for inst_idx, inst in enumerate(lf.instances):
                # Your tracking logic here...
                new_identity = self._compute_identity(inst)

                if new_identity is not None:
                    # Use priority-based assignment
                    success = self.assign_with_priority_resolution(
                        labels=labels,
                        frame_idx=lf.frame_idx,
                        instance_idx=inst_idx,
                        new_track_name=new_identity,
                        reason=f"Custom tracker match (score={score:.3f})",
                        propagate_to_tracklet=True,  # Propagate to entire tracklet
                    )

                    if not success:
                        # Assignment was blocked by higher priority
                        pass

        return labels

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "my_threshold": self.my_threshold,
        })
        return config

    # Implement required abstract methods...
    def get_track_context(self, labels, frame_idx, track):
        # ... implementation
        pass

    def has_track_in_frame(self, labels, frame_idx, track) -> bool:
        # ... implementation
        pass

    # etc.
```

### Best Practices

1. **Use rename for tracklet identification**: When identifying which animal a tracklet belongs to, use `rename_track_globally()` to ensure the identity propagates to all frames.

2. **Use slice sparingly**: Only use `propagate_to_tracklet=False` when you specifically need to override just one instance.

3. **Log explanations**: Set up an `InstanceExplanationStore` and log decisions for debugging:

   ```python
   if self._explanation_store is not None:
       record = self._create_decision_record(...)
       self._explanation_store.add(record)
   ```

4. **Handle blocked assignments gracefully**: When `assign_with_priority_resolution()` returns `False`, the assignment was blocked by a higher-priority tracker. Your tracker should handle this case appropriately.

## Feature Tracking

### RFID Tracking

The `RFIDFeatureTracker` uses RFID ping data and spatial heatmaps to assign animal identities.

#### Usage

```python
from sleap_mot.tracking.feature_tracking.RFID_tracking import RFIDFeatureTracker
import sleap_io as sio

# Create tracker
tracker = RFIDFeatureTracker(priority=10)

# Step 1: Generate heatmaps (one-time setup per experiment)
tracker.generate_heatmaps(
    rfid_pings_path="path/to/rfid_pings.csv",
    slp_paths=["path/to/labels1.slp", "path/to/labels2.slp"],
    video_paths=["path/to/video1.avi", "path/to/video2.avi"],
    output_path="rfid_heatmaps.h5",
    camera_filter="Oryx",  # Optional: filter by camera
    body_nodes=["Nose", "Head", "Upper_back", "Lower_back", "Tailbase"],
)

# Step 2: Run tracking (can reuse heatmaps)
# Option A: Use heatmaps from generate_heatmaps()
labels = sio.load_file("path/to/labels.slp")
result = tracker.track(
    labels=labels,
    rfid_pings_path="path/to/rfid_pings.csv",
    output_path="tracked_output.slp",  # Optional: save results
    window_size=5,  # Frame window for matching RFID pings
)

# Option B: Use pre-generated heatmaps
tracker.heatmaps_path = "path/to/existing_heatmaps.h5"
result = tracker.track(
    labels=labels,
    rfid_pings_path="path/to/rfid_pings.csv",
)
```

#### RFID Pings CSV Format

The RFID pings CSV should contain columns:
- `frame_number`: Video frame where ping occurred
- `unitLabel`: RFID reader/unit identifier
- `IdRFID`: Animal RFID tag identifier
- `Camera`: Camera name (for filtering)
- `eventDuration`: Duration of RFID detection event (ms)

#### How It Works

1. **Heatmap Generation**: For each RFID unit, creates a spatial probability map based on where animals were detected during RFID pings
2. **Probability Assignment**: For each RFID ping, looks up instance positions in a window around the ping frame and assigns probabilities from heatmaps
3. **Track Assignment**:
   - With existing tracklets: Uses majority voting across all pings in a tracklet
   - Without tracklets: Assigns highest-probability identity per frame, resolving conflicts

### Coordinate-Based RFID Tracking

The `CoordinateRFIDTracker` uses direct coordinate matching between RFID ping locations and instance centroids. This approach achieves significantly higher accuracy (~95%) compared to heatmap-based methods (~31%).

#### Usage

```python
from sleap_mot.tracking.feature_tracking.RFID_tracking import CoordinateRFIDTracker
import sleap_io as sio

# Create tracker
tracker = CoordinateRFIDTracker(priority=10, max_distance_from_rfid=100.0)

# Load labels
labels = sio.load_file("path/to/labels.slp")

# Run tracking
result = tracker.track(
    labels=labels,
    rfid_pings_path="path/to/rfid_pings.csv",
    window_size=5,  # Search window around each ping
    output_path="tracked_output.slp",  # Optional: save results
)
```

#### With Explanation Store

```python
from sleap_mot.tracking import InstanceExplanationStore, CoordinateRFIDTracker

# Create tracker with explanation store
store = InstanceExplanationStore()
tracker = CoordinateRFIDTracker(priority=10)
tracker.explanation_store = store

# Run tracking
result = tracker.track(labels=labels, rfid_pings_path="rfid_pings.csv")

# Query explanations
stats = store.get_statistics()
print(f"Total decisions: {stats['total_records']}")
print(f"Trackers: {stats['trackers']}")
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `priority` | 10 | Priority level for conflict resolution (higher = more authoritative) |
| `max_distance_from_rfid` | 100.0 | Maximum distance (px) between RFID ping and instance for valid match |
| `window_size` | 1 | Number of frames to search around each RFID ping |
| `clear_existing_tracks` | True | Whether to clear existing tracks before assignment |
| `detect_switches` | False | Enable identity switch detection and repair |
| `min_votes_for_identity` | 2 | Minimum votes required on each side of a detected switch |
| `switch_confidence_threshold` | 0.8 | Confidence threshold to confirm a switch |
| `split_on_switch` | True | Whether to split tracklets at detected switch points |
| `require_partner_validation` | False | Only split/swap if partner tracklet confirms the switch |
| `frame_tolerance` | 10 | Max frame difference to consider switches related |

#### Required CSV Columns

The RFID pings CSV must contain:
- `frame_number`: Video frame where ping occurred
- `IdRFID`: Animal RFID tag identifier
- `center.x`: X coordinate of RFID ping location
- `center.y`: Y coordinate of RFID ping location

#### Operating Modes

1. **With Tracklets (Primary)**: When tracklets exist, matches RFID pings to tracklets and propagates the assignment to all instances in the tracklet.

2. **Without Tracklets (Fallback)**: Matches RFID pings directly to individual instances.

#### When to Use Coordinate vs Heatmap Approach

| Approach | Use When |
|----------|----------|
| **CoordinateRFIDTracker** | RFID pings have reliable X/Y coordinates; higher accuracy needed; simpler setup |
| **RFIDFeatureTracker** | X/Y coordinates unavailable; spatial probability patterns useful; existing heatmap data |

#### Identity Switch Detection

The `CoordinateRFIDTracker` can detect and repair identity switches that occur when motion trackers incorrectly swap the identities of two animals. This is enabled with `detect_switches=True`.

**Problem this solves**: When two animals get close (e.g., during social interaction), motion trackers may swap their identities. Without switch detection, a single close-but-wrong RFID match can override many correct matches, propagating the wrong identity to an entire tracklet.

**How it works**:

1. **Vote Collection**: Instead of using only the closest RFID match per tracklet, collects ALL matches within `max_distance_from_rfid`. Each match becomes a "vote" for that identity.

2. **Consistency Analysis**: For each tracklet, analyzes whether votes are temporally consistent or show a switch pattern (e.g., identity A dominates frames 0-855, identity B dominates frames 856+).

3. **Switch Detection**: If votes show a clear temporal boundary between two identities, detects a switch point.

4. **Cross-Validation**: Looks for complementary switches (tracklet A: X→Y, tracklet B: Y→X at similar frames) which strongly indicate a motion tracker ID swap.

5. **Repair Actions**:
   - **Swap**: For cross-validated pairs, swaps post-switch segments between tracklets
   - **Split**: For single-tracklet switches, splits at the switch point
   - **Majority Vote**: Falls back to assigning the dominant identity

**Usage with switch detection**:

```python
from sleap_mot.tracking.feature_tracking.RFID_tracking import CoordinateRFIDTracker
from sleap_mot.tracking import InstanceExplanationStore

# Create tracker with switch detection enabled
tracker = CoordinateRFIDTracker(
    priority=10,
    max_distance_from_rfid=100.0,
    min_votes_for_identity=2,        # Require 2+ votes on each side
    switch_confidence_threshold=0.8,  # 80% confidence to confirm switch
    split_on_switch=True,             # Split tracklets at detected switches
    require_partner_validation=False, # Split even without cross-validation
)

# Optional: attach explanation store for debugging
store = InstanceExplanationStore()
tracker.explanation_store = store

# Run tracking with switch detection
result = tracker.track(
    labels=labels,
    rfid_pings_path="rfid_pings.csv",
    window_size=5,
    detect_switches=True,  # Enable switch detection
    frame_tolerance=10,    # Max frame gap for cross-validation
)

# Export explanations to see switch detection decisions
store.save_json("rfid_switch_detection.json")
```

**When to use switch detection**:

| Scenario | Recommendation |
|----------|----------------|
| Animals frequently interact/cross paths | Enable: `detect_switches=True` |
| Motion tracker has known ID swap issues | Enable: `detect_switches=True` |
| Clean, well-separated tracklets | Disable: faster, simpler |
| Few RFID pings (sparse data) | Disable: not enough votes for detection |

### Fur Color Tracking

The `FurColorFeatureTracker` uses fur color patterns for identity assignment.

```python
from sleap_mot.tracking.feature_tracking.base import FurColorFeatureTracker

tracker = FurColorFeatureTracker()
result = tracker.track(
    labels=labels,
    video_path="path/to/video.avi",
    output_path="tracked_output.slp",
    method="bbox",  # or "motion"
    n_bins=4,
    patch_size=5,
    max_instances=8,
)
```

### Tail Tattoo Tracking

The `TailTattooFeatureTracker` uses tail marking patterns for identity assignment.

```python
from sleap_mot.tracking.feature_tracking.base import TailTattooFeatureTracker

tracker = TailTattooFeatureTracker()
result = tracker.track(
    labels=labels,
    video_path="path/to/video.avi",
    output_path="tracked_output.slp",
    tail_nodes=["tti", "t0", "t1", "t2"],
    max_instances=3,
)
```

### Tail Feature Tracking

The `TailFeatureTracker` uses tail segment patterns for identity matching. It extracts visual features from tail segments, clusters them using PCA + k-means, and maps clusters to track identities using configurable mapping methods.

This tracker achieved ~96.9% accuracy on HCM test data with 3 segments at 30px threshold.

#### Usage

```python
from sleap_mot.tracking.feature_tracking.tail_tracking import TailFeatureTracker
import sleap_io as sio

# Create tracker
tracker = TailFeatureTracker(
    tail_nodes=["Tail_0", "Tail_1", "Tail_2", "TailTip"],
    min_node_distance=30.0,  # Minimum pixel distance between tail nodes
    mapping_method="trust_first",  # or "backfill", "majority_vote"
    priority=10,
)

# Load labels
labels = sio.load_file("predictions.slp")

# Run tracking (video loaded from labels automatically)
result = tracker.track(
    labels=labels,
    output_path="tracked_output.slp",  # Optional: save results
)
```

#### With Explanation Store

Track decisions can be logged for debugging and analysis:

```python
from sleap_mot.tracking import InstanceExplanationStore

# Create tracker with explanation store
store = InstanceExplanationStore()
tracker = TailFeatureTracker(
    tail_nodes=["Tail_0", "Tail_1", "Tail_2", "TailTip"],
    mapping_method="trust_first",
)
tracker.explanation_store = store

# Run tracking
result = tracker.track(labels=labels)

# Query decisions
records = store.get_records(frame_idx=100, instance_idx=0)
for r in records:
    print(f"Cluster {r.cluster_id} -> {r.assigned_track_id}")
    print(f"Checkpoint: {r.is_checkpoint_frame}")
    print(f"Reasons: {r.reasons}")

# Save explanations for debugging
store.save_json("tail_tracking_explanations.json")
```

#### With Identity Propagation

When animals swap identities during close encounters, the tracker can propagate corrections back to the switch point:

```python
tracker = TailFeatureTracker(
    tail_nodes=["Tail_0", "Tail_1", "Tail_2", "TailTip"],
    mapping_method="trust_first",
    propagate_identity=True,  # Enable proximity-based propagation
    proximity_method="pose_centroid",  # or "bbox_centroid", "bbox_iou"
    proximity_threshold=150.0,  # Distance threshold for switch detection
)

result = tracker.track(labels=labels)
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tail_nodes` | **Required** | Node names for tail segments (e.g., `["Tail_0", "Tail_1", "Tail_2", "TailTip"]`) |
| `n_clusters` | Auto | Number of k-means clusters (auto-detected from tracks if None) |
| `min_node_distance` | Auto | Min distance between consecutive tail nodes (px) |
| `max_segment_angle` | None | Max angle between segments (degrees); None = disabled |
| `n_pca_components` | 30 | PCA components before k-means |
| `require_all_tails` | False | Require all instances have valid tails to process frame |
| `mapping_method` | "trust_first" | One of: "trust_first", "backfill", "majority_vote" |
| `propagate_identity` | False | Propagate corrections backward to proximity switch point |
| `proximity_method` | "pose_centroid" | Method for proximity detection |
| `proximity_threshold` | 150.0 | Distance (px) or IOU threshold for switch detection |
| `priority` | 10 | Priority for conflict resolution (higher = more authoritative) |

#### Mapping Methods

| Method | Description |
|--------|-------------|
| **trust_first** | Use first complete checkpoint frame to build cluster→track mapping, then apply to all frames |
| **backfill** | Same as trust_first, plus correct frames before the first checkpoint |
| **majority_vote** | Vote across all checkpoints; cluster maps to most common track |

#### Quality Filters

A tail is considered valid if:
1. All specified `tail_nodes` are present with valid (non-NaN) coordinates
2. Distance between each consecutive pair of nodes >= `min_node_distance`
3. If `max_segment_angle` is set: angle between consecutive segments <= threshold

#### Cluster Filtering

When multiple instances share the same k-means cluster:

| `require_all_tails` | Behavior |
|---------------------|----------|
| `False` | Remove only the duplicate instances; keep unique clusters |
| `True` | Discard the entire frame if any instances share a cluster |

## Tracklet Stitching

The `TrackletStitcher` merges fragmented tracklets based on spatial/temporal continuity. It runs after initial tracking and stitches tracks that are clearly the same animal but got disconnected due to conservative tracking thresholds.

### Usage

```python
from sleap_mot.tracking.online_tracking import TrackletStitcher
from sleap_mot.tracking import InstanceExplanationStore
import sleap_io as sio

# Load pre-tracked labels
labels = sio.load_file("tracked_labels.slp")

# Create stitcher with explanation store
store = InstanceExplanationStore()
stitcher = TrackletStitcher(
    max_gap_frames=10,      # Max frames between track end and start
    max_distance=50.0,      # Max centroid distance (pixels)
    use_facing_consistency=True,  # Check facing direction
    facing_threshold=60.0,  # Max facing change (degrees)
)
stitcher.explanation_store = store

# Run stitching
labels = stitcher.track(labels)

# Get statistics
stats = stitcher.get_stitch_statistics()
print(f"Stitches: {stats['total_stitches']}, Rounds: {stats['rounds']}")

# Save explanations
store.save_json("stitch_explanations.json")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_gap_frames` | 10 | Max frames between track end and start |
| `max_distance` | 50.0 | Max centroid distance (pixels) for stitching |
| `use_facing_consistency` | True | Check facing direction consistency |
| `facing_threshold` | 60.0 | Max facing direction change (degrees) |
| `min_tracklet_length` | 1 | Min length of tracklets to consider |

### How It Works

1. **Build tracklet index**: For each tracklet, record first/last frame, centroid, and facing direction
2. **Find stitch candidates**: For each pair (A, B) where B starts after A ends:
   - Check temporal gap <= `max_gap_frames`
   - Check spatial distance <= `max_distance`
   - Check facing change <= `facing_threshold` (if enabled)
3. **Optimal matching**: Use Hungarian algorithm for one-to-one assignment
4. **Apply stitches**: Rename target tracklet instances to source tracklet
5. **Iterate**: Repeat until no more stitches possible

### Typical Results

| Configuration | Tracklet Reduction | Avg Length Increase |
|---------------|-------------------|---------------------|
| Conservative (50px, facing=True) | ~63% | ~3x |
| Aggressive (200px, facing=False) | ~91% | ~10x |

**Note**: Aggressive settings can cause ID switches when animals cross paths. Use conservative settings for higher accuracy.

## Explanation Systems

sleap-mot provides two complementary explanation systems for understanding and debugging tracking decisions.

### Overview: Two Explanation Systems

| Aspect | `instance_explanations.py` | `explanations.py` |
|--------|---------------------------|-------------------|
| **Purpose** | Comprehensive logging for offline analysis | Real-time explanation generation |
| **Primary Use** | Batch analysis, metrics, debugging | ID Switch Viewer debugging tool |
| **Storage** | `InstanceExplanationStore` (queryable) | Embedded in `TrackContext.track_history` |
| **Query Method** | By `(frame_idx, instance_idx)` or tracker name | Via track history traversal |
| **Output Format** | Dataclass records (`*DecisionRecord`) | `SwitchExplanation` objects |
| **Serialization** | Dedicated JSON format with statistics | Part of track_history JSON |
| **Completeness** | Logs ALL decisions (matches, skips, propagations) | Logs decisions as they occur |
| **Status** | **Recommended** - newer, more flexible | Legacy - still used for metrics integration |

### When to Use Each

**Use `InstanceExplanationStore` (recommended) when you need to:**
- Analyze all tracking decisions after tracking completes
- Query decisions by frame, instance, or tracker type
- Generate statistics about tracking behavior
- Save/load explanations for later analysis
- Debug why specific instances got specific assignments

**Use `ExplanationGenerator` classes when you need to:**
- Generate explanations during tracking for real-time display
- Integrate with the ID Switch Viewer tool
- Work with the existing `SwitchExplanation` type from `sleap_mot.metrics.types`

### Instance Explanation System (Recommended)

The `InstanceExplanationStore` provides a comprehensive per-instance explanation system that logs ALL tracker interactions, making it easy to debug and understand tracking decisions.

#### Key Features

- **Instance-centric**: Explanations are indexed by `(frame_idx, instance_idx)` for easy lookup
- **Tracker-specific records**: Each tracker type has its own dataclass with relevant fields
- **Complete logging**: Logs matches, new tracks, skipped assignments, and propagation events
- **JSON serialization**: Save and load explanations for offline analysis

#### Usage

```python
from sleap_mot.tracking import InstanceExplanationStore
from sleap_mot.tracking.online_tracking.motion_tracker import DirectionalMotionTracker

# Create shared store
store = InstanceExplanationStore()

# Create tracker with store
tracker = DirectionalMotionTracker.for_tracklets(long_kde_path="model.joblib")
tracker.explanation_store = store

# Run tracking
labels = tracker.track(labels)

# Query decisions for a specific frame and instance
records = store.get_records(frame_idx=100, instance_idx=0)
for r in records:
    print(f"Decision: {r.decision_type.value} -> {r.assigned_track_id}")
    print(f"Summary: {r.summary}")
    for cs in r.candidate_scores:
        print(f"  Candidate {cs.candidate_id}: {cs.score:.3f} - {cs.rejection_reason}")

# Query all decisions in a frame
for inst_idx, records in store.get_records_for_frame(100).items():
    print(f"Instance {inst_idx}: {len(records)} decisions")

# Query by tracker
motion_records = store.get_records_by_tracker("DirectionalMotionTracker")

# Get statistics
stats = store.get_statistics()
print(f"Total records: {stats['total_records']}")
print(f"Trackers: {stats['trackers']}")
print(f"Decision types: {stats['decision_types']}")

# Save to JSON
store.save_json("explanations.json")

# Load from JSON
loaded_store = InstanceExplanationStore.load_json("explanations.json")
```

#### Record Types

All explanation records inherit from `BaseDecisionRecord` and share these **standard fields**:

| Field | Type | Description |
|-------|------|-------------|
| `frame_idx` | int | Frame index where decision was made |
| `instance_idx` | int | Instance index within the frame |
| `tracker_name` | str | Name of the tracker |
| `tracker_priority` | int | Priority level for conflict resolution |
| `decision_type` | str | "matched", "new_track", "no_match", "skipped", or "propagated" |
| `assigned_track_id` | str | Track ID assigned (None if no assignment) |
| `previous_track_id` | str | Previous track ID if any |
| `summary` | str | Human-readable summary |
| `reasons` | List[str] | List of reasons explaining the decision |

Each tracker adds **method-specific fields**:

| Record Type | Description | Method-Specific Fields |
|-------------|-------------|----------------------|
| `MotionDecisionRecord` | Motion-based tracker decisions | `thresholds`, `instance_centroid`, `candidate_scores`, `winning_track_id`, `kde_model_used`, `motion_probability`, `threshold_checks`, `association_score`, `score_type` |
| `DirectionalMotionDecisionRecord` | Directional motion tracker | Extends Motion + `facing_direction`, `alignment_angle`, `is_stagnant`, `directional_multiplier`, `backward_rejected` |
| `RFIDDecisionRecord` | RFID-based tracker decisions | `rfid_ping_present`, `candidate_rfids`, `winning_rfid_id`, `winning_probability`, `is_propagation`, `propagation_source_frame`, `heatmap_probability`, `instance_centroid` |
| `RFIDNoMatchBroadcast` | Unmatched RFID pings | `rfid_id`, `no_match_reason`, `probability_at_this_instance`, `max_probability_found`, `rfid_unit_label` |
| `PropagationRecord` | Identity propagation events | `source_frame_idx`, `propagation_direction`, `conflict_existed`, `conflict_resolution_method`, `propagated_track_id` |
| `StitchDecisionRecord` | Tracklet stitching decisions | `source_tracklet_id`, `target_tracklet_id`, `temporal_gap`, `spatial_distance`, `facing_angle_change`, `source_last_centroid`, `target_first_centroid`, `stitch_round` |
| `SwitchRepairDecisionRecord` | Identity switch detection and repair | `switch_frame`, `identity_before`, `identity_after`, `votes_before`, `votes_after`, `switch_confidence`, `partner_tracklet`, `partner_validated`, `repair_action`, `original_tracklet`, `resulting_tracks` |
| `TailDecisionRecord` | Tail segment pattern clustering | `cluster_id`, `mapping_method`, `is_checkpoint_frame`, `barcode_valid`, `propagation_source_frame`, `proximity_frame`, `instance_centroid`, `cluster_confidence` |

#### JSON Output Format

```json
{
  "version": "2.0",
  "statistics": {
    "total_records": 15000,
    "trackers": {"MotionTracker": 5000, "RFIDTracker": 10000},
    "decision_types": {"matched": 8000, "new_track": 5000, "propagated": 2000}
  },
  "records": {
    "0": {
      "0": [
        {
          "record_type": "DirectionalMotionDecisionRecord",
          "tracker_name": "DirectionalMotionTracker",
          "decision_type": "matched",
          "assigned_track_id": "tracklet_1",
          "candidate_scores": [
            {"candidate_id": "tracklet_1", "score": 0.85, "passed_thresholds": true},
            {"candidate_id": "tracklet_2", "score": 0.42, "rejection_reason": "Lower score"}
          ],
          "facing_direction": [0.8, 0.6],
          "alignment_angle": 23.5
        }
      ]
    }
  }
}
```

### Explanation Generators (Legacy)

The `ExplanationGenerator` classes in `explanations.py` provide real-time explanation generation during tracking. These are used by the ID Switch Viewer debugging tool and integrate with the `SwitchExplanation` type from `sleap_mot.metrics.types`.

#### Class Hierarchy

```
ExplanationGenerator (Abstract Base)
├── MotionExplanationGenerator
│   └── DirectionalMotionExplanationGenerator
└── FeatureExplanationGenerator
    └── RFIDExplanationGenerator
```

#### Usage

```python
from sleap_mot.tracking.explanations import DirectionalMotionExplanationGenerator

# Create generator with thresholds
generator = DirectionalMotionExplanationGenerator(
    max_match_distance=150.0,
    min_probability_threshold=0.001,
    backward_rejection_angle=120.0,
    stagnant_threshold=15.0,
)

# Generate explanation for a match decision
explanation = generator.explain_match(
    track_id="track_1",
    instance_idx=0,
    score=0.85,
    all_scores={"track_1": 0.85, "track_2": 0.42},
    distance=45.2,
    facing_direction=(0.8, 0.6),
    alignment_angle=23.5,
)

print(explanation.summary)  # "Matched with score 0.8500 (distance: 45.20px)"
print(explanation.reasons)  # List of human-readable reasons
```

#### Key Methods

| Method | Description |
|--------|-------------|
| `explain_match()` | Explain why a track was matched to an instance |
| `explain_no_match()` | Explain why a track was not continued |
| `explain_new_track()` | Explain why a new track was created |

#### Output: SwitchExplanation

The generators return `SwitchExplanation` objects with:
- `decision`: Type of decision ("matched", "track_broken", "new_track")
- `summary`: Human-readable summary
- `reasons`: List of detailed reasons
- `score`: Association score (if applicable)
- `competing_scores`: Scores for all candidates
- `threshold_checks`: Dict of threshold check results
- `motion_context`: Motion-specific context (centroids, distances)
- `feature_context`: Feature-specific context (RFID, fur color)

## SLPT File Format

The `.slpt` file format is a tracking-aware archive format that preserves all tracking metadata across save/load cycles. This enables multi-layer tracking pipelines with proper conflict resolution.

### What is SLPT?

A `.slpt` file is a ZIP archive containing:
- `labels.slp` - Standard SLEAP labels file
- `track_contexts.json` - TrackContext metadata (priorities, history) for all instances
- `explanations.json` - InstanceExplanationStore decision records
- `pipeline.json` - Tracking pipeline metadata (which layers were applied, in what order)

### Why Use SLPT?

When running multi-layer tracking pipelines (e.g., motion tracking → RFID tracking → stitching), standard `.slp` files lose important metadata:
- **Priority information** - Which tracker made each assignment
- **Track history** - How assignments changed over time
- **Explanations** - Why each decision was made
- **Pipeline state** - Which layers have been applied

SLPT preserves all this information, enabling:
- Proper conflict resolution when higher-priority trackers override lower-priority ones
- Debugging and analysis of tracking decisions
- Reproducible multi-stage pipelines

### Basic Usage

```python
from sleap_mot.io import SLPTFile
import sleap_io as sio

# Create SLPT from existing labels
labels = sio.load_file("predictions.slp")
slpt = SLPTFile.from_labels(labels)

# Save to .slpt file
slpt.save("tracked.slpt")

# Load existing .slpt file
slpt = SLPTFile.load("tracked.slpt")

# Get labels with TrackContext restored
labels = slpt.to_labels()

# Export to standard .slp (loses metadata)
slpt.export_slp("tracked.slp")
```

### Multi-Layer Pipeline with SLPT

All trackers support `track_slpt()` for pipeline-aware tracking:

```python
from sleap_mot.io import SLPTFile
from sleap_mot.tracking.online_tracking.motion_tracker import FacingConsistencyTracker
from sleap_mot.tracking.online_tracking.tracklet_stitcher import TrackletStitcher
from sleap_mot.tracking.feature_tracking.RFID_tracking import CoordinateRFIDTracker
import sleap_io as sio

# Start with predictions
labels = sio.load_file("predictions.slp")
slpt = SLPTFile.from_labels(labels, source_path="predictions.slp")

# Layer 1: Generate tracklets with motion tracker (priority=5)
motion_tracker = FacingConsistencyTracker.for_tracklets(
    long_kde_path="motion_model.joblib",
    priority=5,
    name="TrackletGenerator"
)
slpt = motion_tracker.track_slpt(slpt)

# Layer 2: Assign RFID identities (priority=10, overrides motion)
rfid_tracker = CoordinateRFIDTracker(priority=10)
slpt = rfid_tracker.track_slpt(slpt, rfid_pings_path="rfid_pings.csv")

# Layer 3: Stitch remaining tracklets (priority=6)
stitcher = TrackletStitcher(priority=6, max_gap_frames=20)
slpt = stitcher.track_slpt(slpt)

# Save complete pipeline state
slpt.save("final_tracked.slpt")

# Check pipeline history
for layer in slpt.get_pipeline_history():
    print(f"{layer.order}: {layer.name} (priority={layer.priority})")
```

### Tracker Configuration with `get_config()`

All trackers provide a `get_config()` method that returns their configuration as a dictionary. This is used by SLPT for pipeline metadata and enables tracker reconstruction.

```python
from sleap_mot.tracking.online_tracking.motion_tracker import DirectionalMotionTracker
from sleap_mot.tracking.feature_tracking.RFID_tracking import CoordinateRFIDTracker

# Get configuration from any tracker
tracker = DirectionalMotionTracker.for_tracklets(long_kde_path="model.joblib")
config = tracker.get_config()
print(config)
# {
#   'priority': 5,
#   'name': 'DirectionalTrackletGenerator',
#   'temporary': False,
#   'max_gap': 1,
#   'matching_method': 'hungarian',
#   'long_kde_path': 'model.joblib',
#   'kde_threshold': 40.0,
#   'forward_boost': 1.0,
#   ...
# }

# Reconstruct tracker from config
new_tracker = DirectionalMotionTracker(**config)
```

### SLPT API Reference

#### SLPTFile Methods

| Method | Description |
|--------|-------------|
| `from_labels(labels, explanation_store, source_path)` | Create SLPT from Labels object |
| `load(path)` | Load existing .slpt file |
| `save(path)` | Save to .slpt file |
| `to_labels()` | Get Labels with TrackContext restored |
| `export_slp(path)` | Export to standard .slp (loses metadata) |
| `get_pipeline_history()` | Get list of applied tracking layers |
| `get_track_context(frame_idx, instance_idx)` | Get TrackContext for specific instance |
| `get_explanations_for_frame(frame_idx)` | Get all explanations for a frame |
| `get_statistics()` | Get file statistics |
| `merge_explanations(store)` | Add explanations from InstanceExplanationStore |

#### Tracker Methods

All trackers inherit these methods from `TrackingLayer`:

| Method | Description |
|--------|-------------|
| `get_config()` | Get tracker configuration as dict |
| `track_slpt(slpt, **kwargs)` | Track using SLPT, preserving all metadata |
| `track(labels, **kwargs)` | Standard tracking (no metadata preservation) |
| `export_explanations_json(path)` | Export explanation store to JSON for debugging |

### Exporting Explanations for Debugging

All trackers provide an `export_explanations_json()` method for exporting tracking decisions to JSON files. This is useful for debugging and analyzing why specific tracking decisions were made.

```python
from sleap_mot.tracking import InstanceExplanationStore
from sleap_mot.tracking.online_tracking.motion_tracker import DirectionalMotionTracker

# Create tracker with explanation store
store = InstanceExplanationStore()
tracker = DirectionalMotionTracker.for_tracklets(long_kde_path="model.joblib")
tracker.explanation_store = store

# Run tracking
labels = tracker.track(labels)

# Export explanations to JSON
tracker.export_explanations_json("debug_explanations.json")

# Or with .slp path (creates *_explanations.json automatically)
tracker.export_explanations_json("tracked_output.slp")
# Creates: tracked_output_explanations.json
```

The exported JSON contains:
- **version**: Format version ("2.0")
- **statistics**: Summary counts (total records, by tracker, by decision type)
- **records**: Nested by `frame_idx` → `instance_idx` → list of decision records

Each record includes the decision type, assigned track, candidate scores, and tracker-specific fields (see [Record Types](#record-types) for details).

## Module Structure

```
sleap_mot/
├── io/
│   ├── __init__.py                # Module exports (SLPTFile)
│   └── slpt.py                    # SLPT file format (SLPTFile, TrackContextData, PipelineLayerRecord)
├── tracking/
│   ├── base.py                    # TrackingLayer, TrackContext, ConflictResolutionState
│   ├── instance_explanations.py   # Per-instance explanation storage (InstanceExplanationStore, *DecisionRecord)
│   ├── explanations.py            # Real-time explanation generators (*ExplanationGenerator → SwitchExplanation)
│   ├── feature_tracking/
│   │   ├── base.py                # FeatureTracker base class
│   │   ├── RFID_tracking.py       # RFIDFeatureTracker, CoordinateRFIDTracker
│   │   └── tail_tracking.py       # TailFeatureTracker (tail segment patterns)
│   └── online_tracking/
│       ├── base.py                # OnlineTrackingLayer
│       ├── motion_tracker.py      # MotionTracker, DirectionalMotionTracker, FacingConsistencyTracker
│       ├── general_tracker.py     # GeneralOnlineTracker
│       └── tracklet_stitcher.py   # TrackletStitcher
├── feature_tracker.py             # Legacy feature tracker (standalone)
└── utils.py                       # Utility functions
```

## Known Issues

1. **H5 file descriptor error on network filesystems**: When using HDF5 files on network-mounted drives (NFS, SMB), you may encounter file descriptor errors. **Workaround**: Use local filesystem (e.g., `/tmp`) for H5 files.

