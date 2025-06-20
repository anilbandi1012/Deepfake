import os
import cv2
import numpy as np
import tensorflow as tf

model = None

# Build absolute paths to the model files in ../models/
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "../models")
prototxt_path = os.path.join(model_dir, "deploy.prototxt")
weights_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

# Load OpenCV DNN face detection model
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

def load_model_once(model_path):
    global model
    if model is None:
        model = tf.keras.models.load_model(model_path)

def preprocess_image(face):
    face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
    face = face / 255.0
    return np.expand_dims(face, axis=0)

def process_image(image, model_path, conf_threshold=0.5):
    load_model_once(model_path)
    h, w = image.shape[:2]
    output = image.copy()

    # Prepare blob for DNN face detector
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Add padding to the face box
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            face = image[y1:y2, x1:x2]

            # Skip very small faces
            if face.shape[0] < 50 or face.shape[1] < 50:
                continue

            processed = preprocess_image(face)
            pred = model.predict(processed, verbose=0)[0][0]
            label = "Deepfake" if pred > 0.5 else "Real"
            color = (0, 0, 255) if label == "Deepfake" else (0, 255, 0)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            results.append(label)

    return output, results

def extract_faces(image, face_net, confidence_threshold=0.5):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            if face.size > 0:
                faces.append(face)
    return faces

def draw_bounding_boxes(image, bounding_boxes, labels):
    """
    Draw bounding boxes on the image.

    Args:
    - image: The input image (numpy array).
    - bounding_boxes: List of bounding box coordinates [(x1, y1, x2, y2)].
    - labels: List of labels ("Deepfake" or "Real") corresponding to bounding boxes.

    Returns:
    - image with drawn bounding boxes and labels.
    """
    for (x1, y1, x2, y2), label in zip(bounding_boxes, labels):
        color = (0, 0, 255) if label == "Deepfake" else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return image
