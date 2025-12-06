from ultralytics import YOLO
import cv2

class Detector():
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame, conf_threshold=0.1):
        results = self.model(frame, conf=conf_threshold, verbose=False) 
        bboxes = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                bboxes.append([[x1, y1, x2, y2], conf, cls])
        return bboxes