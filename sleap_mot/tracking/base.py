import sleap_io as sio
from abc import ABC, abstractmethod

class IdTrackObject:
    def __init__(self, priority: int or None, track: sio.Track):
        self.priority = priority
        self.track = track
        self.temporary_track = False
        self.valid = True

    def check_priority(self, priority: int):
        if self.priority < priority:
            return True
        return False

    def is_valid(self):
        return self.valid


class IdTrackLayer(ABC):
    def convert_tracks_to_objects(self, labels: sio.Labels, priority: int = None):
        for lf in labels.labeled_frames:
            for inst in lf.instances:
                inst.track = IdTrackObject(priority, inst.track)

    def convert_objects_to_tracks(self, labels: sio.Labels):
        for lf in labels.labeled_frames:
            instances = []
            for inst in lf.instances:
                if inst.track.is_valid():
                    inst.track = inst.track.track
                    instances.append(inst)
            lf.instances = instances

    def clear_tracks(self, labels: sio.Labels):
        for lf in labels.labeled_frames:
            for inst in lf.instances:
                inst.track = None

    @abstractmethod
    def track(self, labels: sio.Labels, priority: int):
        pass
