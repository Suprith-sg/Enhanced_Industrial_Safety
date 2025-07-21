from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import supervision as sv
from concurrent.futures import ThreadPoolExecutor
import os
import logging
from threading import Lock

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Configurable model path
MODEL_PATH = os.getenv("MODEL_PATH", r"C:\Users\B\Desktop\PPE\test-final\test-final\templates\best.pt")
model = YOLO(MODEL_PATH)

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
executor = ThreadPoolExecutor(max_workers=2)
processing_lock = Lock()


def process_image(img_np):
    input_size = (640, 480)
    resized_img = cv2.resize(img_np, input_size)

    results = model(resized_img, conf=0.6, iou=0.5)[0]

    scale_x = img_np.shape[1] / input_size[0]
    scale_y = img_np.shape[0] / input_size[1]
    detections = sv.Detections.from_ultralytics(results)
    detections.xyxy[:, [0, 2]] *= scale_x
    detections.xyxy[:, [1, 3]] *= scale_y

    # Check for specific labels
    alert_needed = any(label in ['no-vest', 'no-helmet'] for label in detections.labels)

    annotated_image = bounding_box_annotator.annotate(scene=img_np, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    _, buffer = cv2.imencode('.jpg', annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buffer).decode('utf-8'), alert_needed


@app.route('/process_frame', methods=['POST'])
def process_image(img_np):
    input_size = (640, 480)  # Resize input for faster processing
    resized_img = cv2.resize(img_np, input_size)

    # Perform detection
    results = model(resized_img, conf=0.6, iou=0.5)[0]

    # Scale bounding boxes back to original image dimensions
    scale_x = img_np.shape[1] / input_size[0]
    scale_y = img_np.shape[0] / input_size[1]
    detections = sv.Detections.from_ultralytics(results)

    # Scale bounding box coordinates
    detections.xyxy[:, [0, 2]] *= scale_x
    detections.xyxy[:, [1, 3]] *= scale_y

    # Access labels correctly
    labels = [results.names[int(label)] for label in detections.labels]

    # Check if the detected labels contain 'no-vest' or 'no-helmet'
    alert_needed = any(label in ['no-vest', 'no-helmet'] for label in labels)

    # Annotate the image with bounding boxes and labels
    annotated_image = bounding_box_annotator.annotate(scene=img_np, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Encode annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buffer).decode('utf-8'), alert_needed





@app.route('/')
def home():
    return render_template('test 1.html')


@app.route('/index')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        executor.shutdown()
