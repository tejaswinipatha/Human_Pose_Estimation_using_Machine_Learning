import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import numpy as np

# Mediapipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function for pose detection
def detect_pose(image, pose):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        )
    return image, results.pose_landmarks

# Streamlit app
st.markdown("<h1 style='text-align: center; color: blue;'>Human Pose Estimation</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: left;'>Select Input Method</h2>", unsafe_allow_html=True)

st.text('Make sure you have a clear image with all the parts clearly visible')
thres = st.slider('Threshold for detecting the key points', min_value=0, value=20, max_value=100, step=5)
thres = thres / 100

input_type = st.radio("Choose Input Method", ["Image", "Webcam", "Video"], index=0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    if input_type == "Image":
        st.markdown("<h3 style='text-align: left; color: green;'>Upload an Image</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image, landmarks = detect_pose(image, pose)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
            if landmarks:
                st.success("Pose landmarks detected!")

    elif input_type == "Webcam":
        st.markdown("<h3 style='text-align: center; color: green;'>Webcam Input</h3>", unsafe_allow_html=True)
        st.write("Press 'Start Webcam' to begin pose detection.")
        run_webcam = st.button("Start Webcam")
        if run_webcam:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video")
                    break
                frame, _ = detect_pose(frame, pose)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()

    elif input_type == "Video":
        st.markdown("<h3 style='text-align: center; color: green;'>Upload a Video</h3>", unsafe_allow_html=True)
        uploaded_video = st.file_uploader("", type=["mp4", "mov", "avi"])
        if uploaded_video:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_video.read())
            cap = cv2.VideoCapture(temp_file.name)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame, _ = detect_pose(frame, pose)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            cap.release()
            st.success("Video Processing Completed!")
