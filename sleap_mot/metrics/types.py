"""Data types for metrics package."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import pandas as pd


@dataclass
class SwitchExplanation:
    """Detailed explanation of why a track assignment was made or not made.

    This dataclass provides granular information about tracking decisions,
    including threshold checks, scoring context, and tracker-specific metadata.
    Used by the ID Switch Viewer for detailed debugging of tracking behavior.

    Attributes:
        decision: The type of decision made. One of:
            - "matched": Instance matched to existing track
            - "new_track": New track created
            - "track_broken": Track terminated due to threshold failure
            - "reassigned": Track reassigned to different identity
        summary: One-line human-readable summary of the decision.
        reasons: List of specific explanations for the decision.
        score: Association score when applicable.
        competing_scores: Dict mapping instance indices to their scores
            for comparison (e.g., {"0": 0.85, "1": 0.71}).
        threshold_checks: Dict of threshold checks with structure:
            {"threshold_name": {"value": float, "threshold": float, "passed": bool}}
        motion_context: Motion-related context for motion-based trackers.
            May include distance, direction, alignment_angle, is_stagnant, etc.
        feature_context: Feature-related context for feature-based trackers.
            May include rfid_probability, rfid_unit, voting_result, etc.
    """

    # Core fields (always present)
    decision: str  # "matched", "new_track", "track_broken", "reassigned"
    summary: str  # One-line human-readable summary

    # Detailed reasons (list of specific explanations)
    reasons: List[str] = field(default_factory=list)

    # Scoring context (when applicable)
    score: Optional[float] = None
    competing_scores: Optional[Dict[str, float]] = None  # {instance_id: score}

    # Threshold checks (tracker-specific, all optional)
    # Example: {"max_distance": {"value": 133.58, "threshold": 120.0, "passed": False}}
    threshold_checks: Optional[Dict[str, Dict[str, Any]]] = None

    # Motion context (for motion-based trackers)
    # Example: {"distance": 45.2, "direction": "forward", "alignment_angle": 15.3}
    motion_context: Optional[Dict[str, Any]] = None

    # Feature context (for feature-based trackers like RFID)
    # Example: {"rfid_probability": 0.92, "rfid_unit": "antenna_1", "voting_result": {...}}
    feature_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export.

        Returns:
            Dictionary representation with all non-None fields.
        """
        result = {
            "decision": self.decision,
            "summary": self.summary,
            "reasons": self.reasons,
        }

        if self.score is not None:
            result["score"] = self.score
        if self.competing_scores is not None:
            result["competing_scores"] = self.competing_scores
        if self.threshold_checks is not None:
            result["threshold_checks"] = self.threshold_checks
        if self.motion_context is not None:
            result["motion_context"] = self.motion_context
        if self.feature_context is not None:
            result["feature_context"] = self.feature_context

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwitchExplanation":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with explanation fields.

        Returns:
            SwitchExplanation instance.
        """
        return cls(
            decision=data.get("decision", "unknown"),
            summary=data.get("summary", ""),
            reasons=data.get("reasons", []),
            score=data.get("score"),
            competing_scores=data.get("competing_scores"),
            threshold_checks=data.get("threshold_checks"),
            motion_context=data.get("motion_context"),
            feature_context=data.get("feature_context"),
        )


@dataclass
class TrackHistoryEntry:
    """Structured entry for track assignment history.

    Attributes:
        layer_name: Name of the tracking layer that made this assignment.
        old_track_name: Previous track name (None if first assignment).
        new_track_name: New track name assigned.
        frame_idx: Frame index where change occurred.
        reason: Human-readable explanation for the change.
        conflict_resolved: Whether this resolved a conflict.
        propagated_from_frame: Source frame if this was propagated (None otherwise).
        explanation: Optional detailed explanation with threshold checks and context.
    """

    layer_name: str
    old_track_name: Optional[str]
    new_track_name: str
    frame_idx: int
    reason: str
    conflict_resolved: bool = False
    propagated_from_frame: Optional[int] = None
    explanation: Optional[SwitchExplanation] = None

    @classmethod
    def from_dict(cls, d: dict) -> "TrackHistoryEntry":
        """Create from legacy dict format.

        Args:
            d: Dictionary with history entry fields.

        Returns:
            TrackHistoryEntry instance.
        """
        # Handle explanation field if present
        explanation = None
        if d.get("explanation"):
            explanation = SwitchExplanation.from_dict(d["explanation"])

        return cls(
            layer_name=d.get("layer_name", "unknown"),
            old_track_name=d.get("old_track_name"),
            new_track_name=d.get("new_track_name", ""),
            frame_idx=d.get("frame_idx", -1),
            reason=d.get("reason", ""),
            conflict_resolved=d.get("conflict_resolved", False),
            propagated_from_frame=d.get("propagated_from_frame"),
            explanation=explanation,
        )

    def to_dict(self) -> dict:
        """Convert to dict for serialization.

        Returns:
            Dictionary representation of the entry.
        """
        result = {
            "layer_name": self.layer_name,
            "old_track_name": self.old_track_name,
            "new_track_name": self.new_track_name,
            "frame_idx": self.frame_idx,
            "reason": self.reason,
            "conflict_resolved": self.conflict_resolved,
            "propagated_from_frame": self.propagated_from_frame,
        }

        # Include explanation if present
        if self.explanation is not None:
            result["explanation"] = self.explanation.to_dict()

        return result


@dataclass
class MetricResult:
    """Container for computed metrics with multiple output formats.

    Attributes:
        name: Metric name (e.g., "accuracy", "fragmentation").
        value: Primary scalar value.
        per_frame: Optional per-frame breakdown.
        per_track: Optional per-track breakdown.
        per_instance: Optional per-instance breakdown.
        metadata: Additional context (e.g., parameters used).
    """

    name: str
    value: float
    per_frame: Optional[Dict[int, float]] = None
    per_track: Optional[Dict[str, Any]] = None
    per_instance: Optional[List[Dict]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict.

        Returns:
            Dictionary representation of the metric result.
        """
        return {
            "name": self.name,
            "value": self.value,
            "per_frame": self.per_frame,
            "per_track": self.per_track,
            "per_instance": self.per_instance,
            "metadata": self.metadata,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame.

        Returns the most granular breakdown available as a DataFrame.

        Returns:
            DataFrame with metric data.
        """
        if self.per_instance is not None:
            return pd.DataFrame(self.per_instance)
        elif self.per_frame is not None:
            return pd.DataFrame(
                [
                    {"frame_idx": k, "value": v}
                    for k, v in sorted(self.per_frame.items())
                ]
            )
        elif self.per_track is not None:
            rows = []
            for track, val in self.per_track.items():
                if isinstance(val, dict):
                    rows.append({"track": track, **val})
                else:
                    rows.append({"track": track, "value": val})
            return pd.DataFrame(rows)
        else:
            return pd.DataFrame([{"name": self.name, "value": self.value}])
