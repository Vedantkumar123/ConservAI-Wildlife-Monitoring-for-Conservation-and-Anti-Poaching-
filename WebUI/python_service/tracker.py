import math
from datetime import datetime, timedelta

class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.next_id_counter = {} # Keep a separate counter for each class
        self.iou_threshold = 0.4  # If IoU is > 40%, it's the same object
        self.disappear_timeout = timedelta(seconds=15) # Forget object if not seen for 15s

    def _calculate_iou(self, boxA, boxB):
        # Determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0

        # Compute the area of both bounding boxes
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Compute the IoU
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def update(self, detections):
        """
        Takes a list of new detections (each as {'box': [x1,y1,x2,y2], 'label': 'tiger', ...})
        Returns the detections with unique IDs.
        """
        current_time = datetime.now()
        updated_detections = []

        if not detections:
            self._cleanup_old_tracks(current_time)
            return []

        # Try to match new detections with existing tracked objects
        for det in detections:
            best_match_id = None
            highest_iou = 0

            for obj_id, obj_data in self.tracked_objects.items():
                # Only compare objects of the same class
                if obj_data['label'] == det['label']:
                    iou = self._calculate_iou(det['box'], obj_data['box'])
                    if iou > highest_iou:
                        highest_iou = iou
                        best_match_id = obj_id

            if highest_iou > self.iou_threshold:
                # It's a match, update the existing object
                self.tracked_objects[best_match_id]['box'] = det['box']
                self.tracked_objects[best_match_id]['last_seen'] = current_time
                self.tracked_objects[best_match_id]['confidence'] = det['confidence']
                det['unique_id'] = best_match_id
            else:
                # It's a new object, assign a new ID
                label = det['label']
                # Get the next ID for this specific label
                current_id_num = self.next_id_counter.get(label, 0) + 1
                self.next_id_counter[label] = current_id_num
                
                new_id = f"{label}-{current_id_num}"
                self.tracked_objects[new_id] = {
                    'box': det['box'],
                    'label': det['label'],
                    'confidence': det['confidence'],
                    'last_seen': current_time
                }
                det['unique_id'] = new_id
            
            updated_detections.append(det)

        self._cleanup_old_tracks(current_time)
        return updated_detections

    def _cleanup_old_tracks(self, current_time):
        # Remove objects that haven't been seen for a while
        disappeared_ids = []
        for obj_id, obj_data in self.tracked_objects.items():
            if current_time - obj_data['last_seen'] > self.disappear_timeout:
                disappeared_ids.append(obj_id)
        
        for obj_id in disappeared_ids:
            print(f"Removing stale track: {obj_id}")
            del self.tracked_objects[obj_id]

    def get_active_counts(self):
        counts = {}
        for obj_data in self.tracked_objects.values():
            label = obj_data['label']
            counts[label] = counts.get(label, 0) + 1
        return counts