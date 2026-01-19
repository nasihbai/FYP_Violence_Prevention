import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf
import threading
import h5py
import json
import os
import time

# ===== VIDEO SOURCE CONFIGURATION =====
# Option 1: Use local MP4 file path
# VIDEO_PATH = "path/to/your/video.mp4"

# Option 2: Use MP4 file URL (works with direct MP4 links)
#VIDEO_PATH = "https://example.com/your-video.mp4"

# Option 3: Use RTSP stream (IP camera or streaming server)
# VIDEO_PATH = "rtsp://username:password@192.168.1.100:554/stream"
# VIDEO_PATH = "rtsp://192.168.1.100:554/stream1"

# Option 4: Use camera device (original behavior)
VIDEO_PATH = 0 # or 1, 6, etc. for camera device index

# RTSP Configuration (optional, for better RTSP performance)
RTSP_BUFFER_SIZE = 1  # Reduce latency for RTSP streams
RTSP_TRANSPORT = "tcp"  # Use TCP for RTSP (more reliable than UDP)

# Performance Configuration
VERBOSE_PREDICTION = False  # Set to True to see prediction percentages
# =====================================

# Check if VIDEO_PATH is a local file, URL, RTSP stream, or camera device
is_rtsp = False
is_stream = False

if isinstance(VIDEO_PATH, str):
    if VIDEO_PATH.startswith("rtsp://"):
        # RTSP stream
        print(f"Loading RTSP stream: {VIDEO_PATH}")
        is_rtsp = True
        is_stream = True

        # Set RTSP transport protocol
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{RTSP_TRANSPORT}"

        cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)

        # Configure buffer size for lower latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, RTSP_BUFFER_SIZE)

    elif VIDEO_PATH.startswith("http://") or VIDEO_PATH.startswith("https://"):
        # HTTP/HTTPS URL
        print(f"Loading video from URL: {VIDEO_PATH}")
        is_stream = True
        cap = cv2.VideoCapture(VIDEO_PATH)
    else:
        # Local file - verify it exists
        if not os.path.exists(VIDEO_PATH):
            raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")
        print(f"Loading video from local file: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
else:
    # Camera device index
    print(f"Using camera device: {VIDEO_PATH}")
    is_stream = True
    cap = cv2.VideoCapture(VIDEO_PATH)

# Verify video opened successfully
if not cap.isOpened():
    raise ValueError(f"Failed to open video source: {VIDEO_PATH}")

# Print video information
print(f"Video opened successfully!")
if not is_stream:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info - FPS: {fps}, Frames: {frame_count}, Resolution: {width}x{height}")

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

custom_objects = {
    'Orthogonal': tf.keras.initializers.Orthogonal
}

with h5py.File("lstm-model.h5", 'r') as f:
    model_config = f.attrs.get('model_config')
    model_config = json.loads(model_config)  

    for layer in model_config['config']['layers']:
        if 'time_major' in layer['config']:
            del layer['config']['time_major']

    model_json = json.dumps(model_config)

    model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)

    weights_group = f['model_weights']
    for layer in model.layers:
        layer_name = layer.name
        if layer_name in weights_group:
            weight_names = weights_group[layer_name].attrs['weight_names']
            layer_weights = [weights_group[layer_name][weight_name] for weight_name in weight_names]
            layer.set_weights(layer_weights)

lm_list = []
label = "neutral"
neutral_label = "neutral"

def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for lm in results.pose_landmarks.landmark:
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0) if label == neutral_label else (0, 0, 255)
    thickness = 2
    lineType = 2
    cv2.putText(img, str(label),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)

    # Silent prediction (suppress TensorFlow verbose output)
    result = model.predict(lm_list, verbose=0)

    if VERBOSE_PREDICTION:
        percentage_result = result * 100
        print(f"Model prediction result: {percentage_result}")

    if result[0][0] > 0.5:
        label = "violent"
    else:
        label = "neutral"
    return str(label)

i = 0
warm_up_frames = 60

while True:
    ret, frame = cap.read()

    # Handle end of video or stream failure
    if not ret:
        if is_stream:
            # For streams (RTSP, camera, live URLs), connection lost - try to reconnect
            print("Stream connection lost. Attempting to reconnect...")
            cap.release()

            # Wait before reconnecting
            time.sleep(2)

            # Try to reconnect
            if is_rtsp:
                cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, RTSP_BUFFER_SIZE)
            else:
                cap = cv2.VideoCapture(VIDEO_PATH)

            if not cap.isOpened():
                print("Failed to reconnect. Exiting...")
                break

            print("Reconnected successfully!")
            i = 0  # Reset frame counter
            lm_list = []  # Clear landmark list
            label = "neutral"  # Reset label
            continue
        else:
            # For local video files - loop back to start
            print("End of video reached. Looping back to start...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            i = 0  # Reset frame counter
            lm_list = []  # Clear landmark list
            label = "neutral"  # Reset label
            continue

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    i += 1
    if i > warm_up_frames:
        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            if len(lm_list) == 20:
                # Create a copy to avoid threading issues
                lm_list_copy = lm_list.copy()
                t1 = threading.Thread(target=detect, args=(model, lm_list_copy))
                t1.start()
                lm_list = []

            # Draw bounding box around detected person
            x_coordinate = []
            y_coordinate = []
            for lm in results.pose_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_coordinate.append(cx)
                y_coordinate.append(cy)

            # Safety check: ensure coordinates exist before drawing
            if x_coordinate and y_coordinate:
                cv2.rectangle(frame,
                                (min(x_coordinate), max(y_coordinate)),
                                (max(x_coordinate), min(y_coordinate) - 25),
                                (0, 255, 0),
                                1)

            frame = draw_landmark_on_image(mpDraw, results, frame)

        frame = draw_class_on_image(label, frame)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()
