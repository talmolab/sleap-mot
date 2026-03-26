"""Visual patch-based re-identification tracker.

Extracts rotated visual patches centered on each instance, builds averaged
templates for each known global identity, and assigns unassigned tracklets
to the closest matching template using cosine similarity.

This tracker is designed as a final layer to assign remaining tracklets
after motion tracking, stitching, and RFID/tail feature assignment.
"""

import cv2
import numpy as np
import sleap_io as sio
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

from sleap_mot.tracking.base import TrackContext
from sleap_mot.tracking.feature_tracking.base import FeatureTracker
from sleap_mot.tracking.instance_explanations import (
    InstanceExplanationStore,
    CandidateScore,
    DecisionType,
)
from sleap_mot.utils import get_centroid, get_keypoints


class VisualPatchTracker(FeatureTracker):
    """Re-identify instances using whole-body visual patch templates.

    For each known global identity (e.g., RFID IDs), builds an averaged
    visual template from frames where that identity is confidently assigned.
    Then for each unassigned tracklet, extracts patches and assigns to the
    closest template by cosine similarity.

    Patches are extracted as square crops centered on the instance centroid,
    rotated so the nose-to-tail axis is horizontal (nose pointing left).

    Attributes:
        global_ids: Set of known global identity names.
        patch_size: Size of extracted square patches (pixels).
        video_path: Path to the video file for frame extraction.
        min_cosine_similarity: Minimum similarity to accept a match.
    """

    def __init__(
        self,
        global_ids: set,
        video_path: str,
        patch_size: int = 50,
        min_cosine_similarity: float = 0.5,
        nose_node_idx: int = 0,
        tail_node_idx: int = -1,
        priority: int = 20,
        name: str = "VisualPatchTracker",
    ):
        """Initialize the visual patch tracker.

        Args:
            global_ids: Set of known global identity names to match against.
            video_path: Path to the video file for extracting patches.
            patch_size: Size of the square patch to extract (default 50px).
            min_cosine_similarity: Minimum cosine similarity for a valid
                match. Tracklets below this threshold are not assigned.
            nose_node_idx: Index of the nose keypoint in the skeleton
                (used for rotation alignment). Default 0.
            tail_node_idx: Index of the tail keypoint in the skeleton
                (used for rotation alignment). Default -1 (last node).
            priority: Priority level for conflict resolution.
            name: Name of this tracking layer.
        """
        super().__init__(priority=priority, name=name)
        self.global_ids = set(global_ids)
        self.video_path = video_path
        self.patch_size = patch_size
        self.min_cosine_similarity = min_cosine_similarity
        self.nose_node_idx = nose_node_idx
        self.tail_node_idx = tail_node_idx

        # Internal state
        self._templates: Dict[str, np.ndarray] = {}  # global_id -> averaged patch
        self._frame_idx_to_lf: Dict[int, sio.LabeledFrame] = {}

    # =========================================================================
    # Patch Extraction
    # =========================================================================

    def _extract_patch(
        self, instance: sio.PredictedInstance, image: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract a rotated square patch centered on the instance.

        The patch is rotated so the nose-to-tail axis is horizontal with
        the nose pointing left, normalizing orientation across frames.

        Args:
            instance: Instance to extract patch from.
            image: Video frame (H, W) or (H, W, 3).

        Returns:
            Grayscale patch of shape (patch_size, patch_size), or None if
            extraction fails (missing keypoints, out of bounds, etc.).
        """
        centroid = get_centroid(instance)
        keypoints = get_keypoints(instance)

        if centroid is None or image is None or keypoints is None:
            return None
        if len(keypoints) < 2:
            return None

        cx, cy = int(centroid[0]), int(centroid[1])

        # Get nose and tail for rotation
        nose = keypoints[self.nose_node_idx]
        tail_idx = self.tail_node_idx if self.tail_node_idx >= 0 else len(keypoints) + self.tail_node_idx
        if tail_idx >= len(keypoints):
            return None
        tail = keypoints[tail_idx]

        if np.any(np.isnan(nose)) or np.any(np.isnan(tail)):
            return None

        # Compute rotation angle (nose pointing left = 180 degrees)
        angle = np.degrees(np.arctan2(nose[1] - tail[1], nose[0] - tail[0]))

        # Rotate image around centroid
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((cx, cy), angle - 180, 1)
        rotated = cv2.warpAffine(image, M, (w, h))

        # Extract patch
        half = self.patch_size // 2
        x1, x2 = cx - half, cx + half
        y1, y2 = cy - half, cy + half

        # Create output patch
        if len(image.shape) == 3:
            patch = np.zeros((self.patch_size, self.patch_size, image.shape[2]), dtype=np.uint8)
        else:
            patch = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)

        # Compute valid source/destination regions
        src_x1, src_x2 = max(0, x1), min(w, x2)
        src_y1, src_y2 = max(0, y1), min(h, y2)
        dst_x1 = max(0, -x1)
        dst_x2 = self.patch_size - max(0, x2 - w)
        dst_y1 = max(0, -y1)
        dst_y2 = self.patch_size - max(0, y2 - h)

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return None

        patch[dst_y1:dst_y2, dst_x1:dst_x2] = rotated[src_y1:src_y2, src_x1:src_x2]

        # Convert to grayscale
        if len(patch.shape) == 3:
            patch = np.mean(patch, axis=2).astype(np.uint8)

        return patch

    # =========================================================================
    # Template Building
    # =========================================================================

    def _build_templates(self, labels: sio.Labels) -> Dict[str, np.ndarray]:
        """Build averaged visual templates for each known global identity.

        Iterates through all frames, extracts patches for instances with
        global IDs, and averages them per identity.

        Args:
            labels: SLEAP Labels object with tracked instances.

        Returns:
            Dict mapping global_id to averaged grayscale patch.
        """
        import imageio.v3 as iio

        print(f"  Building visual templates from {self.video_path}...")
        patches_by_id: Dict[str, List[np.ndarray]] = {
            gid: [] for gid in self.global_ids
        }

        reader = iio.imiter(self.video_path)

        for lf in labels.labeled_frames:
            try:
                image = next(reader)
            except StopIteration:
                break

            for inst in lf.instances:
                if inst.track is None:
                    continue

                track_name = (
                    inst.track.name
                    if isinstance(inst.track, TrackContext)
                    else inst.track.name
                )

                if track_name not in self.global_ids:
                    continue

                patch = self._extract_patch(inst, image)
                if patch is not None:
                    patches_by_id[track_name].append(patch.astype(np.float32))

        # Average patches per identity
        templates = {}
        for gid, patches in patches_by_id.items():
            if len(patches) > 0:
                templates[gid] = np.mean(np.stack(patches), axis=0)
                print(f"    {gid}: template from {len(patches)} patches")
            else:
                print(f"    {gid}: no patches found (skipping)")

        return templates

    # =========================================================================
    # Tracklet Matching
    # =========================================================================

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two flattened arrays."""
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

    def _match_tracklet(
        self,
        labels: sio.Labels,
        tracklet_name: str,
        templates: Dict[str, np.ndarray],
        rfid_frame_sets: Dict[str, set],
    ) -> Optional[Tuple[str, float]]:
        """Find the best matching global identity for a tracklet.

        Extracts patches for all instances in the tracklet, computes mean
        cosine similarity against each template, and returns the best
        non-conflicting match.

        Args:
            labels: SLEAP Labels object.
            tracklet_name: Name of the tracklet to match.
            templates: Dict of global_id -> averaged template.
            rfid_frame_sets: Dict of global_id -> set of frames occupied.

        Returns:
            (best_global_id, similarity) or None if no valid match.
        """
        import imageio.v3 as iio

        tracklet_frames = self.get_tracklet_frames(labels, tracklet_name)
        if not tracklet_frames:
            return None

        tracklet_frame_set = {f for f, _ in tracklet_frames}

        # Extract patches for this tracklet
        # We need video frames — use a targeted read
        frame_indices = sorted(set(f for f, _ in tracklet_frames))

        # Read only needed frames from video
        patches = []
        reader = iio.imiter(self.video_path)
        current_frame = 0
        frame_set = set(frame_indices)

        for lf in labels.labeled_frames:
            if lf.frame_idx > max(frame_indices):
                break
            try:
                image = next(reader)
            except StopIteration:
                break

            if lf.frame_idx not in frame_set:
                continue

            for inst_frame, inst_idx in tracklet_frames:
                if inst_frame == lf.frame_idx:
                    inst = lf.instances[inst_idx]
                    patch = self._extract_patch(inst, image)
                    if patch is not None:
                        patches.append(patch.astype(np.float32))

        if not patches:
            return None

        # Compute mean similarity to each template
        scores = []
        for gid, template in templates.items():
            similarities = [
                self._cosine_similarity(p, template) for p in patches
            ]
            mean_sim = np.mean(similarities)

            # Check for frame conflicts
            overlap = tracklet_frame_set & rfid_frame_sets.get(gid, set())
            has_conflict = len(overlap) > 0

            scores.append((gid, mean_sim, has_conflict, len(overlap)))

        # Sort by similarity (descending), prefer non-conflicting
        scores.sort(key=lambda x: (-x[1]))

        # Return best match above threshold
        for gid, sim, has_conflict, n_overlap in scores:
            if sim >= self.min_cosine_similarity:
                return (gid, sim, has_conflict, n_overlap)

        return None

    # =========================================================================
    # Main Track Method
    # =========================================================================

    def track(self, labels: sio.Labels, **kwargs) -> sio.Labels:
        """Run visual patch re-identification on all unassigned tracklets.

        1. Build averaged visual templates for each known global identity
        2. Find all unassigned tracklets
        3. Match each tracklet to the closest template
        4. Assign using per-frame assignment (with conflict handling)

        Args:
            labels: SLEAP Labels object with existing track assignments.

        Returns:
            Labels with updated track assignments.
        """
        self._build_frame_mapping(labels)

        # Step 1: Build templates
        self._templates = self._build_templates(labels)

        if not self._templates:
            print("  No templates built — nothing to match against.")
            return labels

        # Step 2: Find unassigned tracklets
        all_tracks = set()
        for lf in labels.labeled_frames:
            for inst in lf.instances:
                if inst.track:
                    n = (
                        inst.track.name
                        if isinstance(inst.track, TrackContext)
                        else inst.track.name
                    )
                    all_tracks.add(n)

        unassigned = [t for t in all_tracks if t not in self.global_ids]
        print(f"  Global ID tracks: {len(all_tracks & self.global_ids)}")
        print(f"  Unassigned tracklets: {len(unassigned)}")

        # Step 3: Build RFID frame occupancy index
        rfid_frame_sets = {gid: set() for gid in self.global_ids}
        for lf in labels.labeled_frames:
            for inst in lf.instances:
                if inst.track is None:
                    continue
                n = (
                    inst.track.name
                    if isinstance(inst.track, TrackContext)
                    else inst.track.name
                )
                if n in self.global_ids:
                    rfid_frame_sets[n].add(lf.frame_idx)

        # Step 4: Match and assign each tracklet
        frame_lookup = {lf.frame_idx: lf for lf in labels.labeled_frames}
        assigned = 0
        invalidated_total = 0
        skipped = 0

        for tracklet_name in sorted(unassigned):
            result = self._match_tracklet(
                labels, tracklet_name, self._templates, rfid_frame_sets
            )

            if result is None:
                skipped += 1
                continue

            best_gid, similarity, has_conflict, n_overlap = result

            # Assign per-frame to avoid duplicates
            tracklet_frames = self.get_tracklet_frames(labels, tracklet_name)
            tracklet_frame_set = {f for f, _ in tracklet_frames}
            overlap_frames = tracklet_frame_set & rfid_frame_sets.get(best_gid, set())
            new_track = sio.Track(name=best_gid)

            invalidated = 0
            for frame_idx, inst_idx in tracklet_frames:
                lf = frame_lookup[frame_idx]
                inst = lf.instances[inst_idx]

                if frame_idx in overlap_frames:
                    # Target ID already exists in this frame — invalidate
                    if isinstance(inst.track, TrackContext):
                        inst.track.valid = False
                        inst.track.add_history_entry(
                            layer_name=self.name,
                            old_track_name=tracklet_name,
                            new_track_name=tracklet_name,
                            frame_idx=frame_idx,
                            reason=f"Invalidated: {best_gid} already present (cosine sim {similarity:.3f})",
                        )
                    invalidated += 1
                else:
                    # Safe to assign
                    old_history = []
                    if isinstance(inst.track, TrackContext):
                        old_history = inst.track.track_history.copy()

                    new_ctx = TrackContext(
                        priority=self.priority,
                        track=new_track,
                        name=best_gid,
                        temporary_track=False,
                        valid=True,
                        track_history=old_history,
                    )
                    new_ctx.add_history_entry(
                        layer_name=self.name,
                        old_track_name=tracklet_name,
                        new_track_name=best_gid,
                        frame_idx=frame_idx,
                        reason=f"Visual patch match: cosine sim {similarity:.3f}",
                    )
                    inst.track = new_ctx

            # Update frame occupancy
            rfid_frame_sets[best_gid].update(tracklet_frame_set - overlap_frames)

            assigned += 1
            invalidated_total += invalidated

            if assigned % 50 == 0:
                print(f"    Assigned {assigned} tracklets so far...")

        print(f"  Assigned: {assigned} tracklets")
        print(f"  Invalidated (overlap): {invalidated_total} instances")
        print(f"  Skipped: {skipped} tracklets")

        return labels

    def _build_frame_mapping(self, labels: sio.Labels):
        """Build frame index to LabeledFrame mapping."""
        self._frame_idx_to_lf = {
            lf.frame_idx: lf for lf in labels.labeled_frames
        }
