import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load YOLO Models for pose estimation and object detection
pose_model = YOLO("D:/Streamlit_project/yolo11n-pose.pt")  # Pose estimation model
object_model = YOLO("D:/Streamlit_project/best.pt")  # Object detection model

st.title("YOLO Pose & Object Detection")

# File uploader for video input
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

# Function to calculate angles between three keypoints
def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b  # Vector from b to a
    bc = c - b  # Vector from b to c
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)  # Cosine similarity formula
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))  # Convert to degrees
    return angle

# Function to classify the detected pose based on keypoints
def classify_pose(keypoints):
    if keypoints is None or len(keypoints) < 9:
        return "Unknown"
    
    # Extract keypoints for major body parts
    nose, left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle = keypoints[:9]
    
    # Check if required keypoints exist
    keypoint_check = all(kp is not None for kp in [left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle])
    if not keypoint_check:
        return "Unknown"

    # Calculate important angles for classification
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    torso_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    arm_angle = calculate_angle(left_shoulder, nose, right_shoulder)

    if None in [left_knee_angle, right_knee_angle, torso_angle, arm_angle]:
        return "Unknown"

    # Calculate stride length and knee difference for movement detection
    stride_length = abs(left_ankle[0] - right_ankle[0])
    knee_difference = abs(left_knee[1] - right_knee[1])

    # Classify movements based on angle values
    if torso_angle < 45:
        return "Bending"
    elif left_knee_angle < 100 or right_knee_angle < 100:
        if stride_length > 50 and knee_difference > 30:
            return "Running"
        return "Walking"
    elif left_knee_angle > 160 and right_knee_angle > 160:
        return "Standing"
    elif nose[1] > left_hip[1] and nose[1] > right_hip[1]:
        return "Lying on Floor"
    elif arm_angle > 120:
        return "Arm Raising"
    elif left_hip[1] < left_knee[1] and right_hip[1] < right_knee[1]:
        return "Jumping"
    return "Person"

# Process video if uploaded
if uploaded_file:
    # Save uploaded file to a temporary location
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_file.read())
    temp_video.close()

    cap = cv2.VideoCapture(temp_video.name)
    frame_placeholder = st.empty()

    # Set up video writer to save processed output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
    output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Extract FPS of input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Extract width
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Extract height
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    process_video = st.button("Start Processing Video")

    if process_video:
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 5 != 0:
                continue  # Process every 5th frame for efficiency

            # Perform object detection
            object_results = object_model.predict(source=frame, conf=0.4)

            # Perform pose estimation
            pose_results = pose_model.predict(source=frame, conf=0.4)

            # Draw bounding boxes for detected objects
            for result in object_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = object_model.names[int(box.cls[0])]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Draw pose estimation keypoints and classify action
            for result in pose_results:
                for box, keypoints in zip(result.boxes, result.keypoints.xy):
                    keypoints = [tuple(map(int, kp)) for kp in keypoints]
                    action = classify_pose(keypoints)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, action, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Convert frame to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_placeholder.image(frame_pil, use_column_width=True)

            # Save processed frame to output video
            out.write(frame)

        cap.release()
        out.release()

        # Provide download button for processed video
        with open(output_path, "rb") as f:
            st.download_button(label="Download Processed Video", data=f, file_name="processed_video.mp4", mime="video/mp4")
