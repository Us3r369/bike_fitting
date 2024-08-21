from bike_fit.bike_fit import Bike_Fit
from bike_fit.video_processor import VideoProcessor
import json
import streamlit as st
import pandas as pd
import mediapipe as mp
import logging
logging.basicConfig(level=logging.INFO)

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
    with open(f'{input_folder}target_ranges.json', 'w') as f:
        json.dump(bike_fit.target_angles, f)

# Step 3: Start Analysis Button
if st.button("Start Analysis"):
    if uploaded_file is not None:
        # Save uploaded file to the input folder
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.8)
    # Initialize the VideoProcessor with the input video
    video_processor = VideoProcessor(video_path, output_video_path)

    # Process the video and get the actual angles
    video_angles = video_processor.process_video()

    # Convert the dictionary into a Pandas DataFrame
    video_angles_df = pd.DataFrame.from_dict(video_angles.actual_angles, orient='index', columns=['Min Angle', 'Max Angle'])

    #save video angles to json file
    with open(f'{output_folder}/video_angles.json', 'w') as f:
        json.dump(video_angles.actual_angles, f)

    st.header("Analysis Results")
    #Step 4: Display Results
    video_file = open("./files/output_files/output_video.mov", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)
    bike_fit.set_actual_angles(video_angles.actual_angles)
    comparison = bike_fit.compare_angles()
    # Convert the comparison results to a DataFrame
    data = []

    for angle_name, result in comparison.items():
        status = result.get('status', 'N/A')
        issues = ', '.join(result.get('issues', [])) if 'issues' in result else 'None'
        min_deviation = result.get('deviation', {}).get('min_deviation', 'N/A')
        max_deviation = result.get('deviation', {}).get('max_deviation', 'N/A')
        
        data.append({
            'Angle': angle_name,
            'Status': status,
            'Issues': issues,
            'Min Deviation': min_deviation,
            'Max Deviation': max_deviation
        })

    comparison_df = pd.DataFrame(data)
    #join video_angles_df and comparison_df on angle name
    evaluation_df = pd.merge(video_angles_df, comparison_df, left_index=True, right_on='Angle')
    evaluation_df.set_index('Angle', inplace=True)
    st.write("The actual angle and evaluation of your position is:")
    st.table(evaluation_df)