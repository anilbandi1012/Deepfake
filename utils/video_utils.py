import os
import cv2
import numpy as np
import tensorflow as tf
import tempfile

model = None

# Load face detection model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "../models")
prototxt_path = os.path.join(model_dir, "deploy.prototxt")
weights_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

def load_model_once(model_path):
    global model
    if model is None:
        model = tf.keras.models.load_model(model_path)

def preprocess_image(face):
    face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
    face = face / 255.0
    return np.expand_dims(face, axis=0)

def process_video(video_path, model_path, conf_threshold=0.5):
    load_model_once(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    # Use MP4 format for better web compatibility
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_output.close()  # Close the file handle so OpenCV can write to it
    
    # Use H.264 codec for better browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or try 'H264', 'XVID'
    out = cv2.VideoWriter(
        temp_output.name,
        fourcc,
        fps,
        (width, height)
    )
    
    if not out.isOpened():
        # Fallback to XVID if mp4v doesn't work
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            temp_output.name,
            fourcc,
            fps,
            (width, height)
        )

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1

        # Create blob for face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0), swapRB=False, crop=False)
        face_net.setInput(blob)
        detections = face_net.forward()

        # Process each detected face
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                (h, w) = frame.shape[:2]
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # Add padding around face
                pad = 20
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

                # Extract face region
                face = frame[y1:y2, x1:x2]
                if face.shape[0] < 50 or face.shape[1] < 50:
                    continue

                # Predict if face is deepfake
                try:
                    processed = preprocess_image(face)
                    pred = model.predict(processed, verbose=0)[0][0]
                    label = "Deepfake" if pred > 0.5 else "Real"
                    color = (0, 0, 255) if label == "Deepfake" else (0, 255, 0)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label with confidence
                    label_text = f"{label} ({pred:.2f})"
                    
                    # Calculate text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Draw background rectangle for text
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                                (x1 + text_width, y1), color, -1)
                    
                    # Draw text
                    cv2.putText(frame, label_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                
                except Exception as e:
                    print(f"Error processing face in frame {frame_count}: {e}")
                    continue

        # Write frame to output video
        out.write(frame)

    # Release everything
    cap.release()
    out.release()
    
    # Verify the output file was created successfully
    if not os.path.exists(temp_output.name) or os.path.getsize(temp_output.name) == 0:
        raise Exception("Failed to create output video file")
    
    print(f"Processed {frame_count} frames, output size: {os.path.getsize(temp_output.name)} bytes")
    return temp_output.name