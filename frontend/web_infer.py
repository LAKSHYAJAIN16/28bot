from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import json
import logging
import base64
import numpy as np
import threading
import time
from io import BytesIO
from PIL import Image

# Silence Ultralytics logs
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

app = Flask(__name__)

# Load model
model = YOLO(r"C:\Users\laksh\Desktop\Projects\card-detection\Playing-Cards-Detection-with-YoloV8\yolov8s_playing_cards.pt")

# Global variable to store latest detections
latest_detections = []
detection_lock = threading.Lock()

def process_frame(frame_data):
    """Process a frame and return detections"""
    global latest_detections
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(frame_data.split(',')[1])
        image = Image.open(BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run inference
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
        
        # Update latest detections
        with detection_lock:
            latest_detections = detections
            
    except Exception as e:
        print(f"Error processing frame: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    """Endpoint to receive frame data from phone"""
    try:
        data = request.get_json()
        frame_data = data.get('frame')
        
        if frame_data:
            # Process frame in a separate thread to avoid blocking
            thread = threading.Thread(target=process_frame, args=(frame_data,))
            thread.daemon = True
            thread.start()
            
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "No frame data received"})
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_detections')
def get_detections():
    """Endpoint to get latest detections"""
    with detection_lock:
        return jsonify({"detections": latest_detections})

if __name__ == '__main__':
    print(json.dumps({"status": "Starting web server on http://0.0.0.0:5000"}, ensure_ascii=False))
    print(json.dumps({"status": "Access from phone using your computer's IP address"}, ensure_ascii=False))
    app.run(host='0.0.0.0', port=5000, debug=False)
