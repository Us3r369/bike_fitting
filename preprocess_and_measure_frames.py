import cv2
import mediapipe as mp
import numpy as np
import os
from render_annotated_video import frames_to_video
import json
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.")

video_name = 'seatpost_4_1'
video_path = f'./{video_name}.mov'
output_folder = f'processed_files_{video_name}'
output_video_path = f'./{output_folder}/{video_name}_annotated.mp4'
frame_rate = 30  # Adjust based on your video's frame rate

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.8)

def extract_keypoints(landmarks):
    keypoints = {}
    keypoints['left_shoulder'] = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                  landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    keypoints['left_elbow'] = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                               landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y]
    keypoints['left_wrist'] = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                               landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y]
    keypoints['left_hip'] = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                             landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y]
    keypoints['left_knee'] = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                              landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y]
    keypoints['left_ankle'] = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                               landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y]
    keypoints['left_foot'] = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                               landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]
    return keypoints

def calculate_angle_three_points(point1, point2, point3):
    # Convert points to numpy arrays
    A = np.array(point1)
    B = np.array(point2)
    C = np.array(point3)
    
    # Vectors AB and BC
    AB = B - A
    BC = C - B
    
    # Dot product of AB and BC
    dot_product = np.dot(AB, BC)
    
    # Magnitudes of AB and BC
    mag_AB = np.linalg.norm(AB)
    mag_BC = np.linalg.norm(BC)
    
    # Cosine of the angle
    cos_theta = dot_product / (mag_AB * mag_BC)
    
    # Angle in radians
    angle_rad = np.arccos(cos_theta)
    
    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def calculate_angle_two_points(point1, point2):
    # Unpack the points
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the differences in the x and y coordinates
    delta_x = x2 - x1
    delta_y = y2 - y1

    # Calculate the angle in radians
    angle_rad = np.arctan2(delta_y, delta_x)

    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    # Normalize the angle to be between 0 and 180 degrees
    #if angle_deg < 0:
    #    angle_deg += 180

    return angle_deg

def annotate_angles(frame, keypoints, angles):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    line_type = 2

    # Define the angle ranges for each joint
    #Source of truth: https://docs.velogicfit.com/portal/en/kb/articles/triathlon-metrics
    angle_ranges = {
        'knee_angle': (71,140),
        'elbow_angle': (83, 86),#upper should be 86 - only changing for debugging
        'shoulder_angle': (81, 84),
        'hip_angle': (44, 84),
        'ankle_angle': (87, 113),
        'back_gradient': (1, 14),
        'thigh_gradient': (0, 10)
    }

    def get_font_color(angle_name, angle_value):
        min_angle, max_angle = angle_ranges[angle_name]
        if min_angle <= angle_value <= max_angle:
            return (0, 255, 0)  # Green for within range
        else:
            return (0, 0, 255)  # Red for out of range

    #ToDo: Make this a loop!!
    # Annotate knee angle
    knee_pos = keypoints['left_knee']
    knee_pos = (int(knee_pos[0] * frame.shape[1]), int(knee_pos[1] * frame.shape[0]))
    knee_color = get_font_color('knee_angle', angles['knee_angle'])
    cv2.putText(frame, f'{angles["knee_angle"]:.2f}', knee_pos, font, font_scale, knee_color, line_type)

    # Annotate elbow angle
    elbow_pos = keypoints['left_elbow']
    elbow_pos = (int(elbow_pos[0] * frame.shape[1]), int(elbow_pos[1] * frame.shape[0]))
    elbow_color = get_font_color('elbow_angle', angles['elbow_angle'])
    cv2.putText(frame, f'{angles["elbow_angle"]:.2f}', elbow_pos, font, font_scale, elbow_color, line_type)

    # Annotate shoulder angle
    shoulder_pos = keypoints['left_shoulder']
    shoulder_pos = (int(shoulder_pos[0] * frame.shape[1]), int(shoulder_pos[1] * frame.shape[0]))
    shoulder_color = get_font_color('shoulder_angle', angles['shoulder_angle'])
    cv2.putText(frame, f'{angles["shoulder_angle"]:.2f}', shoulder_pos, font, font_scale, shoulder_color, line_type)

    # Annotate hip angle
    hip_pos = keypoints['left_hip']
    hip_pos = (int(hip_pos[0] * frame.shape[1]), int(hip_pos[1] * frame.shape[0]))
    hip_color = get_font_color('hip_angle', angles['hip_angle'])
    cv2.putText(frame, f'{angles["hip_angle"]:.2f}', hip_pos, font, font_scale, hip_color, line_type)

    # Annotate ankle angle
    ankle_pos = keypoints['left_ankle']
    ankle_pos = (int(ankle_pos[0] * frame.shape[1]), int(ankle_pos[1] * frame.shape[0]))
    ankle_color = get_font_color('ankle_angle', angles['ankle_angle'])
    cv2.putText(frame, f'{angles["ankle_angle"]:.2f}', ankle_pos, font, font_scale, ankle_color, line_type)

    #Todo make this a loop
    # Annotate slope - back
    back_slope_pos = keypoints['left_shoulder']
    back_slope_text_pos = (int(back_slope_pos[0] * frame.shape[1]), int(back_slope_pos[1] * frame.shape[0]) - 30)
    slope_color = (255, 0, 0)  # Blue color for slope
    cv2.putText(frame, f'{angles["back_gradient"]:.2f}', back_slope_text_pos, font, font_scale, slope_color, line_type)

    # Annotate slope - thigh
    thigh_slope_pos = keypoints['left_hip']
    slope_text_pos = (int(thigh_slope_pos[0] * frame.shape[1]), int(thigh_slope_pos[1] * frame.shape[0]) - 30)
    slope_color = (255, 0, 0)  # Blue color for slope
    cv2.putText(frame, f'{angles["thigh_gradient"]:.2f}', slope_text_pos, font, font_scale, slope_color, line_type)

def process_video_and_save_keypoints_and_angles(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    angles_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = extract_keypoints(results.pose_landmarks)
            
            # Calculate relevant angles
            knee_angle = calculate_angle_three_points(keypoints['left_hip'], keypoints['left_knee'], keypoints['left_ankle'])
            elbow_angle = calculate_angle_three_points(keypoints['left_shoulder'], keypoints['left_elbow'], keypoints['left_wrist'])
            shoulder_angle = calculate_angle_three_points(keypoints['left_hip'], keypoints['left_shoulder'], keypoints['left_elbow'])
            hip_angle = calculate_angle_three_points(keypoints['left_shoulder'], keypoints['left_hip'], keypoints['left_knee'])
            ankle_angle = calculate_angle_three_points(keypoints['left_knee'], keypoints['left_ankle'], keypoints['left_foot'])
            back_gradient = calculate_angle_two_points(keypoints['left_shoulder'], keypoints['left_hip'])
            thigh_gradient = calculate_angle_two_points(keypoints['left_hip'], keypoints['left_knee'])

            angles = {
                'knee_angle': knee_angle,
                'elbow_angle': elbow_angle,
                'shoulder_angle': shoulder_angle,
                'hip_angle': hip_angle,
                'ankle_angle': ankle_angle,
                'back_gradient': back_gradient,
                'thigh_gradient': thigh_gradient
            }

            angles_list.append({
                'frame': frame_count,
                'angles': angles
            })

            # Draw the pose annotation on the image.
            annotated_image = frame.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Annotate the angles
            annotate_angles(annotated_image, keypoints, angles)
            
            # Save the annotated image.
            cv2.imwrite(os.path.join(output_folder, f'frame_{frame_count:04d}.png'), annotated_image)
        
        frame_count += 1

    cap.release()
    return angles_list

def create_angle_overview(angles):
    # Initialize dictionaries to store the highest and lowest values for each angle
    angle_stats = {}

    # Iterate through each frame and update the highest and lowest values for each angle
    for entry in angles:
        angles = entry['angles']
        for angle, value in angles.items():
            if angle not in angle_stats:
                angle_stats[angle] = {'min': value, 'max': value}
            else:
                if value > angle_stats[angle]['max']:
                    angle_stats[angle]['max'] = value
                if value < angle_stats[angle]['min']:
                    angle_stats[angle]['min'] = value

    # Convert the result to dataframe
    angle_stats_df = pd.DataFrame(angle_stats)
    angle_stats_df.transpose().to_csv(f'{output_folder}/angle_stats.csv')
    return angle_stats_df.transpose()
    

if __name__ == '__main__':
    angles = process_video_and_save_keypoints_and_angles(video_path, output_folder)
    angle_stats = create_angle_overview(angles)
    #save angles as json
    with open(f'{output_folder}/angles_all.json', 'w') as f:
        json.dump(angles, f)
    frames_to_video(output_folder, output_video_path, frame_rate)

    print(f"Video saved as {output_video_path}\n")
    print(f"Here are the angle stats:\n{angle_stats.head(10)}")