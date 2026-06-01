import cv2
from src.detector import PlateDetector
from src.ocr import PlateOCR

detector = PlateDetector("models/yolov8/best.pt")
default_ocr_engine = PlateOCR()

def process_image(img, ocr_engine=None):
    if ocr_engine is None:
        ocr_engine = default_ocr_engine

    h, w = img.shape[:2]
    results = detector.detect(img)
    outputs = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Clip bounding box coordinates to image boundaries
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        # Skip empty crops
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        text, ocr_conf = ocr_engine.read_text(crop)

        outputs.append({
            "bbox": (x1, y1, x2, y2),
            "text": text,
            "det_conf": conf,
            "ocr_conf": ocr_conf,
            "crop": crop
        })

    return outputs