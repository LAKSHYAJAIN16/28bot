from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import json
import logging
import base64
import numpy as np
import threading
import time
import ssl
import os
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

def create_self_signed_cert():
    """Create a self-signed certificate for HTTPS"""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from datetime import datetime, timedelta
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Card Detection"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress("127.0.0.1"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Save certificate and key
        with open("cert.pem", "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        with open("key.pem", "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        return "cert.pem", "key.pem"
        
    except ImportError:
        print("cryptography library not found. Installing...")
        os.system("pip install cryptography")
        return create_self_signed_cert()

if __name__ == '__main__':
    print(json.dumps({"status": "Creating HTTPS certificate..."}, ensure_ascii=False))
    
    try:
        cert_file, key_file = create_self_signed_cert()
        print(json.dumps({"status": "HTTPS certificate created successfully"}, ensure_ascii=False))
        print(json.dumps({"status": "Starting HTTPS server on https://localhost:5000"}, ensure_ascii=False))
        print(json.dumps({"status": "Access from phone using https://YOUR_COMPUTER_IP:5000"}, ensure_ascii=False))
        print(json.dumps({"status": "Note: You may need to accept the security warning in your browser"}, ensure_ascii=False))
        
        # Run with HTTPS
        app.run(host='0.0.0.0', port=5000, ssl_context=(cert_file, key_file), debug=False)
        
    except Exception as e:
        print(json.dumps({"error": f"Failed to create HTTPS certificate: {e}"}, ensure_ascii=False))
        print(json.dumps({"status": "Falling back to HTTP (camera may not work)"}, ensure_ascii=False))
        app.run(host='0.0.0.0', port=5000, debug=False)
