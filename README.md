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

## Tracking Architecture

sleap-mot provides a layered tracking architecture where different tracking methods can be combined with priority-based conflict resolution.

### Class Hierarchy

```
TrackingLayer (Abstract Base)
├── FeatureTracker (for probability-based trackers)
│   ├── RFIDFeatureTracker
│   ├── FurColorFeatureTracker
│   └── TailTattooFeatureTracker
└── OnlineTrackingLayer (for frame-by-frame trackers)
    └── CentroidTracker
```

### Key Concepts

- **Priority-based conflict resolution**: Each tracker has a priority level (e.g., RFID=10, FurColor=8, Centroid=5). Higher priority trackers can override lower priority assignments.
- **Tracklet-based assignment**: Feature trackers can use existing tracklets (connected sequences of instances) for majority voting when assigning identities.
- **Bidirectional propagation**: Identity changes propagate both forward and backward through connected frames.

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

## Module Structure

```
sleap_mot/
├── tracking/
│   ├── base.py                    # TrackingLayer, TrackContext, ConflictResolutionState
│   ├── feature_tracking/
│   │   ├── base.py                # FeatureTracker base class
│   │   └── RFID_tracking.py       # RFIDFeatureTracker
│   └── online_tracking/
│       └── base.py                # OnlineTrackingLayer
├── feature_tracker.py             # Legacy feature tracker (standalone)
└── utils.py                       # Utility functions
```

## Known Issues

1. **H5 file descriptor error on network filesystems**: When using HDF5 files on network-mounted drives (NFS, SMB), you may encounter file descriptor errors. **Workaround**: Use local filesystem (e.g., `/tmp`) for H5 files.

