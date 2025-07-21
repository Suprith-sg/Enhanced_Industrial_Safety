from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import supervision as sv
import os
import logging
import threading
import queue

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Configurable model path
MODEL_PATH = os.getenv("MODEL_PATH", r"E:\project\runs yolov11\detect\train\weights\best.pt")
model = YOLO(MODEL_PATH)

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Use a thread-safe queue for processing frames
frame_queue = queue.Queue(maxsize=5)
processing_lock = threading.Lock()

def process_frame_worker():
    while True:
        try:
            # Get frame from queue with timeout
            img_np = frame_queue.get(timeout=1)
            
            with processing_lock:
                input_size = (640, 480)
                resized_img = cv2.resize(img_np, input_size)

                # Perform detection
                results = model(resized_img, conf=0.6, iou=0.5)[0]
                
                # Scale bounding boxes back to original image dimensions
                scale_x = img_np.shape[1] / input_size[0]
                scale_y = img_np.shape[0] / input_size[1]
                
                # Create Detections object
                detections = sv.Detections(
                    xyxy=results.boxes.xyxy.cpu().numpy(),
                    confidence=results.boxes.conf.cpu().numpy(),
                    class_id=results.boxes.cls.cpu().numpy().astype(int)
                )

                # Scale bounding box coordinates
                detections.xyxy[:, [0, 2]] *= scale_x
                detections.xyxy[:, [1, 3]] *= scale_y

                # Access labels correctly
                labels = [results.names[int(class_id)] for class_id in detections.class_id]

                # Detailed violation tracking
                person_detected = False
                no_hardhat = False
                no_safety_vest = False

                for label in labels:
                    if label == 'Person':
                        person_detected = True
                    elif label == 'NO-Hardhat':
                        no_hardhat = True
                    elif label == 'NO-Safety Vest':
                        no_safety_vest = True

                # Trigger alarm if person is detected without proper PPE
                alert_needed = person_detected and (no_hardhat or no_safety_vest)

                # Annotate the image with bounding boxes and labels
                annotated_image = bounding_box_annotator.annotate(scene=img_np, detections=detections)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

                # Encode annotated image to base64
                _, buffer = cv2.imencode('.jpg', annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                encoded_image = base64.b64encode(buffer).decode('utf-8')

                return {
                    'image': f'data:image/jpeg;base64,{encoded_image}',
                    'alert': alert_needed,
                    'violations': labels,
                    'person_detected': person_detected,
                    'no_hardhat': no_hardhat,
                    'no_safety_vest': no_safety_vest
                }
        
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Frame processing error: {e}")
            return None

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Get the image from the request
    image_data = request.json['image']
    
    # Remove the data URL prefix if present
    if image_data.startswith('data:image/jpeg;base64,'):
        image_data = image_data.split(',')[1]
    
    # Decode base64 image
    image_bytes = base64.b64decode(image_data)
    
    # Convert to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Add frame to processing queue
    try:
        frame_queue.put_nowait(img_np)
    except queue.Full:
        # If queue is full, remove oldest frame
        try:
            frame_queue.get_nowait()
            frame_queue.put_nowait(img_np)
        except queue.Empty:
            pass

    # Process frame
    result = process_frame_worker()
    
    return jsonify(result) if result else jsonify({'error': 'Processing failed'})

@app.route('/')
def home():
    return render_template('test 1.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Start processing thread
if __name__ == '__main__':
    try:
        # Ensure you have a 'static' folder with an 'alarm.mp3' file
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Error starting the application: {e}")