from ultralytics import YOLO
import json
import logging
import os
import cv2

# Silence Ultralytics logs and other non-critical outputs
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Load model
model = YOLO(r"C:\Users\laksh\Desktop\Projects\card-detection\Playing-Cards-Detection-with-YoloV8\yolov8s_playing_cards.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use default webcam (usually index 0)

if not cap.isOpened():
    print(json.dumps({"error": "Could not open webcam"}, ensure_ascii=False))
    exit()

print(json.dumps({"status": "Webcam started - Press 'q' to quit"}, ensure_ascii=False))

try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        results = model.predict(source=frame, save=False, verbose=False)

        # Process detections
        detections = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            names = getattr(r, "names", {})
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                detections.append({
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id),
                    "confidence": conf,
                    "box": {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
                })

        # Draw overlays on frame
        for det in detections:
            x1 = int(det["box"]["x1"])
            y1 = int(det["box"]["y1"])
            x2 = int(det["box"]["x2"])
            y2 = int(det["box"]["y2"])
            label = f"{det['class_name']} {det['confidence']:.2f}"

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label background
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text = max(y1, text_h + 4)
            cv2.rectangle(frame, (x1, y_text - text_h - 4), (x1 + text_w + 2, y_text), (0, 255, 0), -1)
            # Label text
            cv2.putText(frame, label, (x1 + 1, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Card Detection', frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print(json.dumps({"status": "Webcam stopped"}, ensure_ascii=False))