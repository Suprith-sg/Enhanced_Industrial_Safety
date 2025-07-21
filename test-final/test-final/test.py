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
    # Increase the resolution for better output quality
    input_size = (640, 480)  # High resolution
    resized_img = cv2.resize(img_np, input_size)

    results = model(resized_img, conf=0.6, iou=0.5)[0]

    # Scale detections back to original dimensions
    scale_x = img_np.shape[1] / input_size[0]
    scale_y = img_np.shape[0] / input_size[1]
    detections = sv.Detections.from_ultralytics(results)
    detections.xyxy[:, [0, 2]] *= scale_x
    detections.xyxy[:, [1, 3]] *= scale_y

    annotated_image = bounding_box_annotator.annotate(scene=img_np, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Higher quality encoding
    _, buffer = cv2.imencode('.jpg', annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        if processing_lock.locked():
            return jsonify({'image': None})  # Skip if still processing

        data = request.get_json()
        image_data = data['image']
        img_str = base64.b64decode(image_data.split(',')[1])
        img_np = cv2.imdecode(np.frombuffer(img_str, np.uint8), cv2.IMREAD_COLOR)

        with processing_lock:
            annotated_image_str = process_image(img_np)
        return jsonify({'image': f'data:image/jpeg;base64,{annotated_image_str}'})
    except Exception as e:
        logging.error("Error processing frame: %s", str(e))
        return jsonify({'error': str(e)})


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
