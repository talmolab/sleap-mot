"""Data types for metrics package."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import pandas as pd


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
    """

    layer_name: str
    old_track_name: Optional[str]
    new_track_name: str
    frame_idx: int
    reason: str
    conflict_resolved: bool = False
    propagated_from_frame: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "TrackHistoryEntry":
        """Create from legacy dict format.

        Args:
            d: Dictionary with history entry fields.

        Returns:
            TrackHistoryEntry instance.
        """
        return cls(
            layer_name=d.get("layer_name", "unknown"),
            old_track_name=d.get("old_track_name"),
            new_track_name=d.get("new_track_name", ""),
            frame_idx=d.get("frame_idx", -1),
            reason=d.get("reason", ""),
            conflict_resolved=d.get("conflict_resolved", False),
            propagated_from_frame=d.get("propagated_from_frame"),
        )

    def to_dict(self) -> dict:
        """Convert to dict for serialization.

        Returns:
            Dictionary representation of the entry.
        """
        return {
            "layer_name": self.layer_name,
            "old_track_name": self.old_track_name,
            "new_track_name": self.new_track_name,
            "frame_idx": self.frame_idx,
            "reason": self.reason,
            "conflict_resolved": self.conflict_resolved,
            "propagated_from_frame": self.propagated_from_frame,
        }


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
