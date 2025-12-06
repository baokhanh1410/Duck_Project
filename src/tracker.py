import supervision as sv
import numpy as np

class Tracker():
    def __init__(self):
        self.tracker = sv.ByteTrack(
            lost_track_buffer=30,
            track_activation_threshold=0.3,
            minimum_matching_threshold = 0.7
        )
    def update(self, frame, bboxes):
        if not bboxes:
            tracked_detections = self.tracker.update(sv.Detections.empty())
            return []

        xyxy = np.array([box[0] for box in bboxes])
        conf = np.array([box[1] for box in bboxes])
        cls = np.array([box[2] for box in bboxes])

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=cls
        )
        # Output Detections: (Bboxex=[],conf=[],cls=[])
        tracked_detections = self.tracker.update_with_detections(detections)
        results = []
        if tracked_detections.tracker_id is not None:
            for bbox, track_id, label in zip(
            tracked_detections.xyxy,
            tracked_detections.tracker_id,
            tracked_detections.class_id
            ):
                bbox = tuple(map(int, bbox))
                track_id = int(track_id)
                label = int(label)
                results.append((bbox, track_id, label))
        return results

