import cv2
import mediapipe as mp
import numpy as np
from bike_fit.bike_fit import Bike_Fit
import logging
import warnings
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.")

class VideoProcessor:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.pose = mp.solutions.pose.Pose(static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils

    def _calculate_angle_three_points(self,point1, point2, point3):
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

    def _calculate_angle_two_points(self,point1, point2):
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

    def _extract_keypoints(self,landmarks):
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
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

    def process_video(self):
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.8)
        # Open the input video
        cap = cv2.VideoCapture(self.video_path)
        bike_fit = Bike_Fit()
        logging.info(f"The default bike angles are: {bike_fit.actual_angles}")
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Define the codec and create a VideoWriter object to save the annotated video
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'XVID' for .avi files
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Perform pose estimation
            results = pose.process(image_rgb)
            
            # Draw pose annotations on the frame
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                keypoints = self._extract_keypoints(results.pose_landmarks)
            
                # Calculate relevant angles
                knee_angle = self._calculate_angle_three_points(keypoints['left_hip'], keypoints['left_knee'], keypoints['left_ankle'])
                elbow_angle = self._calculate_angle_three_points(keypoints['left_shoulder'], keypoints['left_elbow'], keypoints['left_wrist'])
                shoulder_angle = self._calculate_angle_three_points(keypoints['left_hip'], keypoints['left_shoulder'], keypoints['left_elbow'])
                hip_angle = self._calculate_angle_three_points(keypoints['left_shoulder'], keypoints['left_hip'], keypoints['left_knee'])
                ankle_angle = self._calculate_angle_three_points(keypoints['left_knee'], keypoints['left_ankle'], keypoints['left_foot'])
                back_gradient = self._calculate_angle_two_points(keypoints['left_shoulder'], keypoints['left_hip'])
                thigh_gradient = self._calculate_angle_two_points(keypoints['left_hip'], keypoints['left_knee'])
                #summarize the angles
                frame_angles = {"knee_angle": knee_angle, "elbow_angle": elbow_angle, "shoulder_angle": shoulder_angle, "hip_angle": hip_angle, "ankle_angle": ankle_angle, "back_gradient": back_gradient, "thigh_gradient": thigh_gradient} 
                #annotate the angles
                self.annotate_angles(frame, keypoints, frame_angles)
                #If angle of current frame smaller than first corresponding, or larger than second corresponding, append to actual_angles
                for angle in frame_angles:
                    bike_fit.update_actual_angles(angle, frame_angles[angle])
            # Write the annotated frame to the output video
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return bike_fit

    def annotate_angles(self,frame, keypoints, angles):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 0, 0)
        line_type = 2

        # Annotate knee angle
        knee_pos = keypoints['left_knee']
        knee_pos = (int(knee_pos[0] * frame.shape[1]), int(knee_pos[1] * frame.shape[0]))
        cv2.putText(frame, f'{angles["knee_angle"]:.2f}', knee_pos, font, font_scale, font_color, line_type)

        # Annotate elbow angle
        elbow_pos = keypoints['left_elbow']
        elbow_pos = (int(elbow_pos[0] * frame.shape[1]), int(elbow_pos[1] * frame.shape[0]))
        cv2.putText(frame, f'{angles["elbow_angle"]:.2f}', elbow_pos, font, font_scale, font_color, line_type)

        # Annotate shoulder angle
        shoulder_pos = keypoints['left_shoulder']
        shoulder_pos = (int(shoulder_pos[0] * frame.shape[1]), int(shoulder_pos[1] * frame.shape[0]))
        cv2.putText(frame, f'{angles["shoulder_angle"]:.2f}', shoulder_pos, font, font_scale, font_color, line_type)

        # Annotate hip angle
        hip_pos = keypoints['left_hip']
        hip_pos = (int(hip_pos[0] * frame.shape[1]), int(hip_pos[1] * frame.shape[0]))
        cv2.putText(frame, f'{angles["hip_angle"]:.2f}', hip_pos, font, font_scale, font_color, line_type)

        # Annotate ankle angle
        ankle_pos = keypoints['left_ankle']
        ankle_pos = (int(ankle_pos[0] * frame.shape[1]), int(ankle_pos[1] * frame.shape[0]))
        cv2.putText(frame, f'{angles["ankle_angle"]:.2f}', ankle_pos, font, font_scale, font_color, line_type)

        #Todo make this a loop
        # Annotate slope - back
        back_slope_pos = keypoints['left_shoulder']
        back_slope_text_pos = (int(back_slope_pos[0] * frame.shape[1]), int(back_slope_pos[1] * frame.shape[0]) - 30)
        cv2.putText(frame, f'{angles["back_gradient"]:.2f}', back_slope_text_pos, font, font_scale, font_color, line_type)

        # Annotate slope - thigh
        thigh_slope_pos = keypoints['left_hip']
        slope_text_pos = (int(thigh_slope_pos[0] * frame.shape[1]), int(thigh_slope_pos[1] * frame.shape[0]) - 30)
        cv2.putText(frame, f'{angles["thigh_gradient"]:.2f}', slope_text_pos, font, font_scale, font_color, line_type)

    def __del__(self):
        # Release resources
        self.pose.close()