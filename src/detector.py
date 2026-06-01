from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image):
        return self.model(image)