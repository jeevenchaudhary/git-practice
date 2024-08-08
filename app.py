import os
import cv2
import numpy as np
import time
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layer_indices = net.getUnconnectedOutLayers()
    if isinstance(output_layer_indices[0], np.ndarray):
        output_layer_indices = output_layer_indices.flatten()
    output_layers = [layer_names[i - 1] for i in output_layer_indices]
    return net, output_layers

def detect_objects(image_path):
    net, output_layers = load_yolo_model()
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image from {image_path}")
        return None

    height, width, channels = image.shape
    real_width_cm = 21.0  # cm
    real_height_cm = 29.7  # cm
    px_to_cm_width = real_width_cm / width
    px_to_cm_height = real_height_cm / height

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if isinstance(indexes, tuple) or len(indexes) == 0:
        print("No objects detected.")
        return None

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        w_cm = w * px_to_cm_width
        h_cm = h * px_to_cm_height
        text = f"{label} ({confidence:.2f}), {w_cm:.2f}cm x {h_cm:.2f}cm"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
    cv2.imwrite(output_path, image)
    
    if not os.path.exists(output_path):
        print(f"Failed to save output image to {output_path}")
    
    return output_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
            file.save(filename)
            output_path = detect_objects(filename)
            if output_path:
                return render_template('index.html', input_image=filename, output_image=output_path)
    return render_template('index.html', input_image=None, output_image=None)

@app.route('/capture', methods=['POST'])
def capture_image():
    # cap = cv2.VideoCapture(0)  # 0 for default camera

    # # Give the camera some time to warm up
    # time.sleep(2)

    # if not cap.isOpened():
    #     return 'Camera could not be opened. Please check the connection and try again.'

    # ret, frame = cap.read()
    # cap.release()

    # if not ret:
    #     return 'Failed to capture image from camera. Please ensure the camera is properly connected and accessible.'

    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
    cv2.imwrite(filename, frame)
    
    # Check if the file was saved successfully
    if not os.path.exists(filename):
        return 'Failed to save the captured image.'

    output_path = detect_objects(filename)
    if output_path:
        return render_template('index.html', input_image=filename, output_image=output_path)
    
    return render_template('index.html', input_image=None, output_image=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
