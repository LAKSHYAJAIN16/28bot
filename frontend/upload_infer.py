from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import json
import logging
import base64
import numpy as np
import os
from io import BytesIO
from PIL import Image
from werkzeug.utils import secure_filename

# Silence Ultralytics logs
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model
model = YOLO(r"C:\Users\laksh\Desktop\Projects\card-detection\Playing-Cards-Detection-with-YoloV8\yolov8s_playing_cards.pt")

# Create uploads directory
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def process_image(image_data):
    """Process an image and return detections"""
    try:
        # Convert to OpenCV format
        if isinstance(image_data, str):
            # Handle base64 data
            image_data = base64.b64decode(image_data.split(',')[1])
        
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
        
        return detections
            
    except Exception as e:
        print(f"Error processing image: {e}")
        return []

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"})
        
        if file:
            # Read file data
            file_data = file.read()
            
            # Process the image
            detections = process_image(file_data)
            
            return jsonify({
                "status": "success",
                "detections": detections,
                "filename": secure_filename(file.filename)
            })
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/process_base64', methods=['POST'])
def process_base64():
    """Handle base64 image data"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"status": "error", "message": "No image data received"})
        
        # Process the image
        detections = process_image(image_data)
        
        return jsonify({
            "status": "success",
            "detections": detections
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    print(json.dumps({"status": "Starting upload server on http://0.0.0.0:5000"}, ensure_ascii=False))
    print(json.dumps({"status": "Access from phone using your computer's IP address"}, ensure_ascii=False))
    app.run(host='0.0.0.0', port=5000, debug=False)
