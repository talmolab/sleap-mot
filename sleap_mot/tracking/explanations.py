"""Explanation generators for tracking decisions.

This module provides classes for generating detailed explanations of tracking
decisions, including threshold checks, scoring context, and tracker-specific
metadata. These explanations are used by the ID Switch Viewer for debugging.

The explanation system follows a hierarchical design:
1. ExplanationGenerator - Abstract base class
2. MotionExplanationGenerator - For motion-based trackers
3. DirectionalMotionExplanationGenerator - Extends motion with directional info
4. FeatureExplanationGenerator - For feature-based trackers
5. RFIDExplanationGenerator - Specific to RFID tracking
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from sleap_mot.metrics.types import SwitchExplanation


class ExplanationGenerator(ABC):
    """Abstract base class for generating switch explanations.

    Subclasses implement tracker-specific explanation logic for different
    types of tracking decisions: matches, no-matches, and new tracks.
    """

    @abstractmethod
    def explain_match(
        self,
        track_id: str,
        instance_idx: int,
        score: float,
        all_scores: Dict[str, float],
        **context
    ) -> SwitchExplanation:
        """Explain why a match was made.

        Args:
            track_id: The track ID that was matched.
            instance_idx: Index of the matched instance in the frame.
            score: Association score for this match.
            all_scores: Dict mapping track_id to scores for all candidates.
            **context: Additional tracker-specific context.

        Returns:
            SwitchExplanation with match details.
        """
        pass

    @abstractmethod
    def explain_no_match(
        self,
        track_id: str,
        instance_idx: int,
        reason_code: str,
        **context
    ) -> SwitchExplanation:
        """Explain why a track was not continued.

        Args:
            track_id: The track ID that was not continued.
            instance_idx: Index of the best candidate instance.
            reason_code: Code indicating failure reason:
                - "threshold_failure": Failed one or more thresholds
                - "better_match_exists": Another instance was a better match
                - "no_candidates": No valid candidates in frame
            **context: Additional tracker-specific context.

        Returns:
            SwitchExplanation with no-match details.
        """
        pass

    @abstractmethod
    def explain_new_track(
        self,
        track_id: str,
        instance_idx: int,
        **context
    ) -> SwitchExplanation:
        """Explain why a new track was created.

        Args:
            track_id: The new track ID created.
            instance_idx: Index of the instance assigned to the new track.
            **context: Additional tracker-specific context.

        Returns:
            SwitchExplanation with new track details.
        """
        pass


class MotionExplanationGenerator(ExplanationGenerator):
    """Explanation generator for motion-based trackers.

    Generates explanations that include motion-specific context like
    distances, probabilities, and threshold checks.

    Attributes:
        max_match_distance: Maximum allowed distance for a match.
        min_probability_threshold: Minimum probability to continue a track.
        proximity_threshold: Minimum distance to other instances.
        iou_threshold: Maximum IoU overlap with other instances.
    """

    def __init__(
        self,
        max_match_distance: Optional[float] = None,
        min_probability_threshold: Optional[float] = None,
        proximity_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ):
        """Initialize the motion explanation generator.

        Args:
            max_match_distance: Maximum distance threshold.
            min_probability_threshold: Minimum probability threshold.
            proximity_threshold: Minimum distance to other instances.
            iou_threshold: Maximum IoU threshold.
        """
        self.max_match_distance = max_match_distance
        self.min_probability_threshold = min_probability_threshold
        self.proximity_threshold = proximity_threshold
        self.iou_threshold = iou_threshold

    def explain_match(
        self,
        track_id: str,
        instance_idx: int,
        score: float,
        all_scores: Dict[str, float],
        **context
    ) -> SwitchExplanation:
        """Explain why a motion-based match was made."""
        distance = context.get("distance", 0.0)
        frame_gap = context.get("frame_gap", 1)

        reasons = []
        threshold_checks = {}

        # Build summary
        summary = f"Matched with score {score:.4f}"
        if distance > 0:
            summary += f" (distance: {distance:.2f}px)"

        reasons.append(f"Best match score: {score:.4f}")

        # Check distance threshold if applicable
        if self.max_match_distance is not None and distance > 0:
            passed = distance < self.max_match_distance
            threshold_checks["max_distance"] = {
                "value": distance,
                "threshold": self.max_match_distance,
                "passed": passed,
            }
            if passed:
                reasons.append(
                    f"Distance check passed: {distance:.2f} < {self.max_match_distance}"
                )

        # Check probability threshold if applicable
        if self.min_probability_threshold is not None:
            passed = score >= self.min_probability_threshold
            threshold_checks["probability"] = {
                "value": score,
                "threshold": self.min_probability_threshold,
                "passed": passed,
            }
            if passed:
                reasons.append(
                    f"Probability check passed: {score:.4f} >= {self.min_probability_threshold}"
                )

        # Build competing scores
        competing_scores = {str(k): v for k, v in all_scores.items()}

        # Build motion context
        motion_context = {
            "distance": distance,
            "frame_gap": frame_gap,
        }
        if "centroid_prev" in context:
            motion_context["centroid_prev"] = context["centroid_prev"]
        if "centroid_curr" in context:
            motion_context["centroid_curr"] = context["centroid_curr"]

        return SwitchExplanation(
            decision="matched",
            summary=summary,
            reasons=reasons,
            score=score,
            competing_scores=competing_scores,
            threshold_checks=threshold_checks if threshold_checks else None,
            motion_context=motion_context,
        )

    def explain_no_match(
        self,
        track_id: str,
        instance_idx: int,
        reason_code: str,
        **context
    ) -> SwitchExplanation:
        """Explain why a motion-based track was not continued."""
        reasons = []
        threshold_checks = {}

        distance = context.get("distance", 0.0)
        score = context.get("score", 0.0)
        proximity_distance = context.get("proximity_distance")
        iou_value = context.get("iou_value")
        all_scores = context.get("all_scores", {})

        # Check each threshold and build explanation
        summary = "Track broken: "

        if reason_code == "threshold_failure":
            # Check max distance
            if self.max_match_distance is not None:
                passed = distance < self.max_match_distance
                threshold_checks["max_distance"] = {
                    "value": distance,
                    "threshold": self.max_match_distance,
                    "passed": passed,
                }
                if not passed:
                    reason = f"Failed max distance threshold: {distance:.2f} >= {self.max_match_distance}"
                    reasons.append(reason)
                    if not summary.endswith(": "):
                        summary = reason

            # Check probability
            if self.min_probability_threshold is not None:
                passed = score >= self.min_probability_threshold
                threshold_checks["probability"] = {
                    "value": score,
                    "threshold": self.min_probability_threshold,
                    "passed": passed,
                }
                if not passed:
                    reason = f"Failed probability threshold: {score:.4f} < {self.min_probability_threshold}"
                    reasons.append(reason)
                    if summary == "Track broken: ":
                        summary = reason

            # Check proximity
            if self.proximity_threshold is not None and proximity_distance is not None:
                passed = proximity_distance >= self.proximity_threshold
                threshold_checks["proximity"] = {
                    "value": proximity_distance,
                    "threshold": self.proximity_threshold,
                    "passed": passed,
                }
                if not passed:
                    reason = f"Failed proximity threshold: {proximity_distance:.2f} < {self.proximity_threshold}"
                    reasons.append(reason)
                    if summary == "Track broken: ":
                        summary = reason

            # Check IoU
            if self.iou_threshold is not None and iou_value is not None:
                passed = iou_value <= self.iou_threshold
                threshold_checks["iou"] = {
                    "value": iou_value,
                    "threshold": self.iou_threshold,
                    "passed": passed,
                }
                if not passed:
                    reason = f"Failed IoU threshold: {iou_value:.4f} > {self.iou_threshold}"
                    reasons.append(reason)
                    if summary == "Track broken: ":
                        summary = reason

        elif reason_code == "better_match_exists":
            better_instance = context.get("better_instance_idx")
            better_score = context.get("better_score", 0.0)
            summary = f"Better match exists: instance {better_instance} has score {better_score:.4f}"
            reasons.append(
                f"Instance {better_instance} has higher score ({better_score:.4f}) than instance {instance_idx} ({score:.4f})"
            )

        elif reason_code == "no_candidates":
            summary = "No valid candidates in frame"
            reasons.append("No instances available for matching in this frame")

        else:
            summary = f"Track broken: {reason_code}"
            reasons.append(reason_code)

        # If summary is still the prefix, use first reason
        if summary == "Track broken: " and reasons:
            summary = reasons[0]

        # Build motion context
        motion_context = {
            "distance": distance,
        }
        if "centroid_prev" in context:
            motion_context["centroid_prev"] = context["centroid_prev"]
        if "centroid_curr" in context:
            motion_context["centroid_curr"] = context["centroid_curr"]

        return SwitchExplanation(
            decision="track_broken",
            summary=summary,
            reasons=reasons,
            score=score,
            competing_scores={str(k): v for k, v in all_scores.items()} if all_scores else None,
            threshold_checks=threshold_checks if threshold_checks else None,
            motion_context=motion_context,
        )

    def explain_new_track(
        self,
        track_id: str,
        instance_idx: int,
        **context
    ) -> SwitchExplanation:
        """Explain why a new track was created."""
        reason = context.get("reason", "unmatched_instance")

        if reason == "no_active_tracks":
            summary = "New track (no active tracks)"
            reasons = ["No active tracks to match against"]
        elif reason == "unmatched_instance":
            summary = "New track (unmatched instance)"
            reasons = ["Instance could not be matched to any existing track"]
        else:
            summary = f"New track ({reason})"
            reasons = [reason]

        motion_context = {}
        if "centroid" in context:
            motion_context["centroid"] = context["centroid"]

        return SwitchExplanation(
            decision="new_track",
            summary=summary,
            reasons=reasons,
            motion_context=motion_context if motion_context else None,
        )


class DirectionalMotionExplanationGenerator(MotionExplanationGenerator):
    """Extended explanation generator with directional motion info.

    Adds facing direction, alignment angle, and directional penalties
    to the explanations generated by MotionExplanationGenerator.

    Attributes:
        backward_rejection_angle: Angle above which movement is rejected.
        stagnant_threshold: Distance below which directional penalty is skipped.
    """

    def __init__(
        self,
        max_match_distance: Optional[float] = None,
        min_probability_threshold: Optional[float] = None,
        proximity_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        backward_rejection_angle: float = 120.0,
        stagnant_threshold: float = 15.0,
        kde_threshold: float = 40.0,
    ):
        """Initialize the directional motion explanation generator.

        Args:
            max_match_distance: Maximum distance threshold.
            min_probability_threshold: Minimum probability threshold.
            proximity_threshold: Minimum distance to other instances.
            iou_threshold: Maximum IoU threshold.
            backward_rejection_angle: Angle above which movement is rejected.
            stagnant_threshold: Distance below which directional penalty skipped.
            kde_threshold: Movement threshold for KDE vs distance scoring.
        """
        super().__init__(
            max_match_distance=max_match_distance,
            min_probability_threshold=min_probability_threshold,
            proximity_threshold=proximity_threshold,
            iou_threshold=iou_threshold,
        )
        self.backward_rejection_angle = backward_rejection_angle
        self.stagnant_threshold = stagnant_threshold
        self.kde_threshold = kde_threshold

    def explain_match(
        self,
        track_id: str,
        instance_idx: int,
        score: float,
        all_scores: Dict[str, float],
        **context
    ) -> SwitchExplanation:
        """Explain a directional motion match with alignment info."""
        explanation = super().explain_match(
            track_id, instance_idx, score, all_scores, **context
        )

        # Add directional context
        facing_direction = context.get("facing_direction")
        alignment_angle = context.get("alignment_angle")
        is_stagnant = context.get("is_stagnant", False)
        directional_multiplier = context.get("directional_multiplier", 1.0)
        base_score = context.get("base_score")
        distance = context.get("distance", 0.0)

        if explanation.motion_context is None:
            explanation.motion_context = {}

        if facing_direction is not None:
            if isinstance(facing_direction, np.ndarray):
                facing_direction = facing_direction.tolist()
            explanation.motion_context["facing_direction"] = facing_direction

        if alignment_angle is not None:
            explanation.motion_context["alignment_angle"] = alignment_angle
            # Add explanation based on alignment
            if alignment_angle < 45:
                explanation.reasons.append(
                    f"Movement aligned with facing direction ({alignment_angle:.1f}°)"
                )
            elif alignment_angle > 135:
                explanation.reasons.append(
                    f"Movement against facing direction ({alignment_angle:.1f}°, penalized)"
                )

        explanation.motion_context["is_stagnant"] = is_stagnant
        if is_stagnant:
            explanation.reasons.append(
                f"Stagnant movement ({distance:.2f}px < {self.stagnant_threshold}px), no directional penalty"
            )

        if directional_multiplier != 1.0:
            explanation.motion_context["directional_multiplier"] = directional_multiplier

        if base_score is not None:
            explanation.motion_context["base_score"] = base_score
            explanation.motion_context["final_score"] = score

        # Add threshold info for directional checks
        if explanation.threshold_checks is None:
            explanation.threshold_checks = {}

        if distance >= self.kde_threshold:
            explanation.threshold_checks["kde_threshold"] = {
                "value": distance,
                "threshold": self.kde_threshold,
                "passed": True,
                "description": "Using KDE scoring (distance >= threshold)",
            }
        else:
            explanation.threshold_checks["kde_threshold"] = {
                "value": distance,
                "threshold": self.kde_threshold,
                "passed": False,
                "description": "Using distance-based scoring (distance < threshold)",
            }

        return explanation

    def explain_no_match(
        self,
        track_id: str,
        instance_idx: int,
        reason_code: str,
        **context
    ) -> SwitchExplanation:
        """Explain why a directional motion track was not continued."""
        # Check for backward rejection first
        alignment_angle = context.get("alignment_angle")
        if reason_code == "backward_rejection" or (
            alignment_angle is not None and alignment_angle > self.backward_rejection_angle
        ):
            reasons = [
                f"Movement rejected: alignment angle {alignment_angle:.1f}° > {self.backward_rejection_angle}° threshold"
            ]
            summary = f"Backward movement rejected ({alignment_angle:.1f}°)"

            motion_context = {
                "alignment_angle": alignment_angle,
                "is_backward": True,
            }
            if "facing_direction" in context:
                facing_dir = context["facing_direction"]
                if isinstance(facing_dir, np.ndarray):
                    facing_dir = facing_dir.tolist()
                motion_context["facing_direction"] = facing_dir
            if "distance" in context:
                motion_context["distance"] = context["distance"]

            threshold_checks = {
                "backward_rejection": {
                    "value": alignment_angle,
                    "threshold": self.backward_rejection_angle,
                    "passed": False,
                }
            }

            return SwitchExplanation(
                decision="track_broken",
                summary=summary,
                reasons=reasons,
                score=context.get("score", 0.0),
                threshold_checks=threshold_checks,
                motion_context=motion_context,
            )

        # Fall back to parent implementation for other cases
        explanation = super().explain_no_match(
            track_id, instance_idx, reason_code, **context
        )

        # Add directional context
        if alignment_angle is not None:
            if explanation.motion_context is None:
                explanation.motion_context = {}
            explanation.motion_context["alignment_angle"] = alignment_angle

        facing_direction = context.get("facing_direction")
        if facing_direction is not None:
            if explanation.motion_context is None:
                explanation.motion_context = {}
            if isinstance(facing_direction, np.ndarray):
                facing_direction = facing_direction.tolist()
            explanation.motion_context["facing_direction"] = facing_direction

        return explanation


class FeatureExplanationGenerator(ExplanationGenerator):
    """Explanation generator for feature-based trackers.

    Base class for trackers that use features like RFID, fur color,
    or other identifying characteristics.
    """

    def explain_match(
        self,
        track_id: str,
        instance_idx: int,
        score: float,
        all_scores: Dict[str, float],
        **context
    ) -> SwitchExplanation:
        """Explain a feature-based match."""
        feature_type = context.get("feature_type", "feature")
        summary = f"Matched by {feature_type} with confidence {score:.4f}"
        reasons = [f"Best {feature_type} match: {score:.4f}"]

        return SwitchExplanation(
            decision="matched",
            summary=summary,
            reasons=reasons,
            score=score,
            competing_scores={str(k): v for k, v in all_scores.items()},
            feature_context=context.get("feature_context"),
        )

    def explain_no_match(
        self,
        track_id: str,
        instance_idx: int,
        reason_code: str,
        **context
    ) -> SwitchExplanation:
        """Explain why a feature-based match was not made."""
        feature_type = context.get("feature_type", "feature")
        score = context.get("score", 0.0)

        if reason_code == "threshold_failure":
            summary = f"Failed {feature_type} confidence threshold"
            reasons = [f"{feature_type.capitalize()} confidence {score:.4f} below threshold"]
        elif reason_code == "ambiguous":
            summary = f"Ambiguous {feature_type} match"
            reasons = [f"Multiple possible {feature_type} matches with similar scores"]
        else:
            summary = f"No {feature_type} match: {reason_code}"
            reasons = [reason_code]

        return SwitchExplanation(
            decision="track_broken",
            summary=summary,
            reasons=reasons,
            score=score,
            feature_context=context.get("feature_context"),
        )

    def explain_new_track(
        self,
        track_id: str,
        instance_idx: int,
        **context
    ) -> SwitchExplanation:
        """Explain a new feature-based track."""
        feature_type = context.get("feature_type", "feature")
        reason = context.get("reason", "unmatched")

        if reason == "no_feature_data":
            summary = f"New track (no {feature_type} data)"
            reasons = [f"No {feature_type} data available for this instance"]
        else:
            summary = f"New track ({reason})"
            reasons = [reason]

        return SwitchExplanation(
            decision="new_track",
            summary=summary,
            reasons=reasons,
            feature_context=context.get("feature_context"),
        )


class RFIDExplanationGenerator(FeatureExplanationGenerator):
    """Explanation generator specifically for RFID tracking.

    Generates explanations that include RFID-specific context like
    antenna location, ping timing, and voting results.
    """

    def __init__(self, min_probability_threshold: Optional[float] = None):
        """Initialize the RFID explanation generator.

        Args:
            min_probability_threshold: Minimum RFID probability threshold.
        """
        self.min_probability_threshold = min_probability_threshold

    def explain_match(
        self,
        track_id: str,
        instance_idx: int,
        score: float,
        all_scores: Dict[str, float],
        **context
    ) -> SwitchExplanation:
        """Explain an RFID-based match."""
        rfid_unit = context.get("rfid_unit", "unknown")
        voting_result = context.get("voting_result")
        heatmap_probability = context.get("heatmap_probability")

        summary = f"RFID assignment with probability {score:.2f}"
        reasons = []

        if rfid_unit:
            reasons.append(f"RFID ping from unit '{rfid_unit}' matched with probability {score:.2f}")

        if voting_result:
            vote_count = voting_result.get("vote_count", 0)
            total_votes = voting_result.get("total_votes", 0)
            reasons.append(f"Majority vote: {vote_count}/{total_votes} pings matched this identity")

        if heatmap_probability:
            reasons.append(f"Heatmap probability at instance location: {heatmap_probability:.4f}")

        threshold_checks = {}
        if self.min_probability_threshold is not None:
            threshold_checks["rfid_probability"] = {
                "value": score,
                "threshold": self.min_probability_threshold,
                "passed": score >= self.min_probability_threshold,
            }

        feature_context = {
            "rfid_probability": score,
            "rfid_unit": rfid_unit,
        }
        if voting_result:
            feature_context["voting_result"] = voting_result
        if heatmap_probability:
            feature_context["heatmap_probability"] = heatmap_probability
        if "rfid_id" in context:
            feature_context["rfid_id"] = context["rfid_id"]

        return SwitchExplanation(
            decision="matched",
            summary=summary,
            reasons=reasons,
            score=score,
            competing_scores={str(k): v for k, v in all_scores.items()} if all_scores else None,
            threshold_checks=threshold_checks if threshold_checks else None,
            feature_context=feature_context,
        )

    def explain_no_match(
        self,
        track_id: str,
        instance_idx: int,
        reason_code: str,
        **context
    ) -> SwitchExplanation:
        """Explain why an RFID match was not made."""
        score = context.get("score", 0.0)
        rfid_unit = context.get("rfid_unit")

        reasons = []
        threshold_checks = {}

        if reason_code == "threshold_failure":
            if self.min_probability_threshold is not None:
                threshold_checks["rfid_probability"] = {
                    "value": score,
                    "threshold": self.min_probability_threshold,
                    "passed": False,
                }
            summary = f"RFID probability {score:.2f} below threshold"
            reasons.append(f"RFID probability {score:.2f} < {self.min_probability_threshold}")

        elif reason_code == "no_ping_data":
            summary = "No RFID ping data for this frame window"
            reasons.append("No RFID pings detected within the search window")

        elif reason_code == "voting_tie":
            summary = "RFID voting resulted in tie"
            voting_result = context.get("voting_result", {})
            reasons.append(f"Multiple RFID identities tied in voting: {voting_result}")

        elif reason_code == "conflict":
            conflicting_id = context.get("conflicting_id")
            summary = f"RFID conflict with {conflicting_id}"
            reasons.append(f"Another instance has higher probability for RFID {conflicting_id}")

        else:
            summary = f"RFID match failed: {reason_code}"
            reasons.append(reason_code)

        feature_context = {
            "rfid_probability": score,
        }
        if rfid_unit:
            feature_context["rfid_unit"] = rfid_unit
        if "voting_result" in context:
            feature_context["voting_result"] = context["voting_result"]

        return SwitchExplanation(
            decision="track_broken",
            summary=summary,
            reasons=reasons,
            score=score,
            threshold_checks=threshold_checks if threshold_checks else None,
            feature_context=feature_context,
        )

    def explain_new_track(
        self,
        track_id: str,
        instance_idx: int,
        **context
    ) -> SwitchExplanation:
        """Explain a new RFID-based track."""
        reason = context.get("reason", "unmatched")
        rfid_id = context.get("rfid_id")
        probability = context.get("probability", 0.0)

        if reason == "first_assignment":
            summary = f"Initial RFID assignment: {rfid_id}"
            reasons = [f"First RFID assignment for identity {rfid_id} with probability {probability:.2f}"]
        elif reason == "no_tracklet":
            summary = "New track (no existing tracklet)"
            reasons = ["Creating new track - no existing tracklet to assign RFID to"]
        else:
            summary = f"New RFID track ({reason})"
            reasons = [reason]

        feature_context = {}
        if rfid_id:
            feature_context["rfid_id"] = rfid_id
        if probability > 0:
            feature_context["rfid_probability"] = probability
        if "voting_result" in context:
            feature_context["voting_result"] = context["voting_result"]

        return SwitchExplanation(
            decision="new_track",
            summary=summary,
            reasons=reasons,
            score=probability if probability > 0 else None,
            feature_context=feature_context if feature_context else None,
        )
