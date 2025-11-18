import tensorflow as tf
from google.colab import drive
import numpy as np
import os
import cv2

# --- 1. Mount Drive and Load Model ---
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Load the model you saved
print("Loading your saved model...")
model_path = '/content/drive/MyDrive/fused_model_checkpoint_FULL_FINAL.keras'
fused_model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# --- 2. Redefine Constants and Helper Functions ---
IMG_SIZE = 128
SPATIAL_SEQ_LENGTH = 20
TEMPORAL_SEQ_LENGTH = 10

# We get the class names from your permanent NPY folder on Drive
NPY_DATA_DIR = "/content/drive/MyDrive/UCF101_Sports_NPY_Full"
class_names = sorted(os.listdir(NPY_DATA_DIR))
print(f"Loaded {len(class_names)} classes.")

# Redefine the extraction functions (must match what you trained on)
def extract_frames_spatial(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: return None
    frame_indices = np.linspace(0, total_frames - 1, SPATIAL_SEQ_LENGTH, dtype=int)
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        else:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
        frames.append(frame)
    cap.release()
    return np.array(frames)

def extract_frames_temporal(video_path):
    frames, flow_fields = [], []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < TEMPORAL_SEQ_LENGTH + 1: return None
    start_frame = (total_frames - (TEMPORAL_SEQ_LENGTH + 1)) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(TEMPORAL_SEQ_LENGTH + 1):
        ret, frame = cap.read()
        if not ret: return None
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    cap.release()
    for i in range(TEMPORAL_SEQ_LENGTH):
        prev_frame, next_frame = frames[i], frames[i+1]
        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_normalized = cv2.normalize(flow, None, 0, 1, cv2.NORM_MINMAX)
        flow_fields.append(flow_normalized)
    return np.array(flow_fields)

# Redefine the prediction function
def predict_action(video_path, model):
    spatial_data = extract_frames_spatial(video_path)
    if spatial_data is None:
        print("Error processing spatial stream.")
        return None, None
    spatial_data = np.expand_dims(spatial_data, axis=0)

    temporal_data = extract_frames_temporal(video_path)
    if temporal_data is None:
        print("Error processing temporal stream (video too short?)")
        return None, None
    temporal_data = np.expand_dims(temporal_data, axis=0)

    prediction_array = model.predict({
        'spatial_input': spatial_data,
        'temporal_input': temporal_data
    })

    predicted_index = np.argmax(prediction_array)
    predicted_class = class_names[predicted_index]
    confidence = prediction_array[0][predicted_index] * 100

    return predicted_class, confidence

LOCAL_VIDEO_FILENAME = "test_video.mp4"

if not os.path.exists(LOCAL_VIDEO_FILENAME):
    print("\n--- 🛑 ERROR: Video loading failed! ---")
else:
    print("\nLoading complete!")
    print(f"Predicting action for: {LOCAL_VIDEO_FILENAME}")
    predicted_action, confidence_score = predict_action(LOCAL_VIDEO_FILENAME, fused_model)

    if predicted_action:
        print("\n--- 🥁 PREDICTION 🥁 ---")
        print(f"Action: {predicted_action}")
        print(f"Confidence: {confidence_score:.2f}%")
