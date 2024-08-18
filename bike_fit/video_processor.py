import cv2
import mediapipe as mp
import numpy as np

class VideoProcessor:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.pose = mp.solutions.pose.Pose(static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils

    def process_video(self):
        # Open the input video
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Define the codec and create a VideoWriter object to save the annotated video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Perform pose estimation
            results = self.pose.process(image_rgb)
            
            # Draw pose annotations on the frame
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            
            # Write the annotated frame to the output video
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def __del__(self):
        # Release resources
        self.pose.close()