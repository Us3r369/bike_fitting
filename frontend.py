from bike_fit.bike_fit import Bike_Fit
from bike_fit.video_processor import VideoProcessor
import json
import streamlit as st
import mediapipe as mp

input_folder = "./files/input_files/"
video_name = 'input_video.mp4'
video_path = "./files/input_files/input_video.mp4"
output_folder = "./files/output_files/"
output_video_path = f'{output_folder}/output_video.mov'
frame_rate = 30  # Adjust based on your video's frame rate

with open(f'{input_folder}target_ranges.json', 'r') as f:
    target_angles = json.load(f)

bike_fit = Bike_Fit(target_angles)

st.title("Bike Fitting Application")
uploaded_file = st.file_uploader("Upload your cycling video", type=["mp4", "mov", "avi"])

# Sidebar to define target ranges
st.sidebar.header("Define Target Ranges (in degrees)")
knee_angle_target = st.sidebar.slider(
    "Knee Angle", min_value=0, max_value=180, value=(target_angles['knee_angle'][0], target_angles['knee_angle'][1]))
elbow_angle_target = st.sidebar.slider(
    "Elbow Angle", min_value=0, max_value=180, value=(target_angles['elbow_angle'][0], target_angles['elbow_angle'][1]))
shoulder_angle_target = st.sidebar.slider(
    "Shoulder Angle", min_value=0, max_value=180, value=(target_angles['shoulder_angle'][0], target_angles['shoulder_angle'][1]))
hip_angle_target = st.sidebar.slider(
    "Hip Angle", min_value=0, max_value=180, value=(target_angles['hip_angle'][0], target_angles['hip_angle'][1]))
ankle_angle_target = st.sidebar.slider(
    "Ankle Angle", min_value=0, max_value=180, value=(target_angles['ankle_angle'][0], target_angles['ankle_angle'][1]))
back_gradient_target = st.sidebar.slider(
    "Back Gradient", min_value=-90, max_value=90, value=(target_angles['back_gradient'][0], target_angles['back_gradient'][1]))
thigh_gradient_target = st.sidebar.slider(
    "Thigh Gradient", min_value=-90, max_value=90, value=(target_angles['thigh_gradient'][0], target_angles['thigh_gradient'][1]))

# Update the bike_fit object with the new target ranges
bike_fit.set_target_angle("knee_angle", knee_angle_target)
bike_fit.set_target_angle("elbow_angle", elbow_angle_target)
bike_fit.set_target_angle("shoulder_angle", shoulder_angle_target)
bike_fit.set_target_angle("hip_angle", hip_angle_target)
bike_fit.set_target_angle("ankle_angle", ankle_angle_target)
bike_fit.set_target_angle("back_gradient", back_gradient_target)
bike_fit.set_target_angle("thigh_gradient", thigh_gradient_target)

# Button to save the target ranges
if st.sidebar.button("Save Target Ranges"):
    bike_fit.save_target_angles(f'{input_folder}/target_ranges.json')
    st.sidebar.success("Target ranges saved successfully!")

# Step 3: Start Analysis Button
if st.button("Start Analysis"):
    if uploaded_file is not None:
        # Save uploaded file to the input folder
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Initialize the VideoProcessor with the input video
    video_processor = VideoProcessor(video_path, output_video_path)

    # Process the video and get the actual angles
    video_processor.process_video()


    # Update the BikeFit object with the actual angles
    #bike_fit.update_actual_angle(actual_angles)

    # Compare the target angles with the actual angles
    #comparison_results = bike_fit.compare_angles()