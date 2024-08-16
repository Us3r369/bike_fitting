import streamlit as st
import cv2
import time
import mediapipe as mp
from render_annotated_video import frames_to_video
import json
from preprocess_and_measure_frames import  extract_keypoints, calculate_angle_three_points, calculate_angle_two_points, annotate_angles, process_video_and_save_keypoints_and_angles, create_angle_overview
input_folder = "./files/input_files/"
video_name = 'input_video.mp4'
video_path = "./files/input_files/input_video.mp4"
output_folder = "./files/output_files/"
output_video_path = f'{output_folder}/output_video.mov'
frame_rate = 30  # Adjust based on your video's frame rate


st.title("Bike Fitting Application")

# Step 1: Video Upload
uploaded_file = st.file_uploader("Upload your cycling video", type=["mp4", "mov", "avi"])

# Step 2: Target Range Inputs
st.sidebar.header("Define Target Ranges (in degrees)")
knee_angle_target = st.sidebar.slider("Knee Angle", min_value=0, max_value=180, value=(30, 40))
elbow_angle_target = st.sidebar.slider("Elbow Angle", min_value=0, max_value=180, value=(10, 20))
shoulder_angle_target = st.sidebar.slider("Shoulder Angle", min_value=0, max_value=180, value=(30, 50))
hip_angle_target = st.sidebar.slider("Hip Angle", min_value=0, max_value=180, value=(40, 50))
ankle_angle_target = st.sidebar.slider("Ankle Angle", min_value=0, max_value=180, value=(5, 15))
back_gradient_target = st.sidebar.slider("Back Gradient", min_value=-90, max_value=90, value=(10, 30))
thigh_gradient_target = st.sidebar.slider("Thigh Gradient", min_value=-90, max_value=90, value=(15, 25))
#Button to save the target ranges
if st.sidebar.button("Save Target Ranges"):
    target_ranges = {
        "knee_angle": knee_angle_target,
        "elbow_angle": elbow_angle_target,
        "shoulder_angle": shoulder_angle_target,
        "hip_angle": hip_angle_target,
        "ankle_angle": ankle_angle_target,
        "back_gradient": back_gradient_target,
        "thigh_gradient": thigh_gradient_target
    }
    with open(f'{input_folder}/target_ranges.json', 'w') as f:
        json.dump(target_ranges, f)
    st.sidebar.success("Target ranges saved successfully!")

tab1, tab2 = st.tabs(["Analysis", "Explanation"])

# Step 3: Start Analysis Button
with tab1:
    if st.button("Start Analysis"):
        if uploaded_file is not None:
            # Save uploaded file to the input folder
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.8)
            
        angles = process_video_and_save_keypoints_and_angles(video_path, output_folder)
        angle_stats = create_angle_overview(angles)
        #save angles and overview
        with open(f'{output_folder}/angles_all.json', 'w') as f:
            json.dump(angles, f)
        angle_stats.to_csv(f'{output_folder}/angle_stats.csv', index=False)
        #Render video
        frames_to_video(output_folder, output_video_path, frame_rate)

        st.header("Analysis Results")
        #Step 4: Display Results
        video_file = open("./files/output_files/output_video.mov", "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)

        #st.write(analysis_results)  # Assumes this is a DataFrame or dictionary of results
    else:
        st.error("Please upload a video file.")

with tab2:
    st.write("Please see the image below for an explanation of the different angles.")
    #display picture
    st.image("./files/in_app_information/measurement_explanation.png", use_column_width=True)