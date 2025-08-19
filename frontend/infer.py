from ultralytics import YOLO
import json
import logging
import os
import cv2

# Silence Ultralytics logs and other non-critical outputs
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Load model
model = YOLO(r"C:\Users\laksh\Desktop\Projects\card-detection\Playing-Cards-Detection-with-YoloV8\yolov8s_playing_cards.pt")

# Inference source
image_path = r"C:\Users\laksh\Desktop\selection-American-playing-cards-jack-queen-ace.webp"

# Run inference without verbose console logs and without saving images
results = model.predict(source=image_path, save=False, verbose=False)

# Convert results to JSON-friendly structure
detections = []
for r in results:
    boxes = getattr(r, "boxes", None)
    names = getattr(r, "names", {})
    img_path = getattr(r, "path", image_path)
    if boxes is None:
        continue
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].tolist()
        conf = float(boxes.conf[i])
        cls_id = int(boxes.cls[i])
        detections.append({
            "image": img_path,
            "class_id": cls_id,
            "class_name": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id),
            "confidence": conf,
            "box": {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
        })

output = {"detections": detections}

# Draw overlays with OpenCV and save annotated image (no console output)
try:
    image = cv2.imread(image_path)
    if image is not None and len(detections) > 0:
        for det in detections:
            x1 = int(det["box"]["x1"])
            y1 = int(det["box"]["y1"])
            x2 = int(det["box"]["x2"])
            y2 = int(det["box"]["y2"])
            label = f"{det['class_name']} {det['confidence']:.2f}"

            # Box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label background
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text = max(y1, text_h + 4)
            cv2.rectangle(image, (x1, y_text - text_h - 4), (x1 + text_w + 2, y_text), (0, 255, 0), -1)
            # Label text
            cv2.putText(image, label, (x1 + 1, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        annotated_path = os.path.splitext(image_path)[0] + "_annotated.jpg"
        cv2.imwrite(annotated_path, image)
except Exception:
    # Keep silent by design; avoid printing/logging
    pass

# Save JSON log next to the image
json_path = os.path.splitext(image_path)[0] + "_detections.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# Print only the JSON to stdout
print(json.dumps(output, ensure_ascii=False))