import cv2
import os

def frames_to_video(frames_folder, output_video_path, frame_rate=30):
    # Get the list of frames
    frames = [f for f in sorted(os.listdir(frames_folder)) if f.endswith('.png')]
    
    # Read the first frame to get the video dimensions
    first_frame = cv2.imread(os.path.join(frames_folder, frames[0]))
    height, width, layers = first_frame.shape
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'XVID' for .avi files
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # Write each frame to the video, then delete the frame
    for frame_name in frames:
        frame = cv2.imread(os.path.join(frames_folder, frame_name))
        video_writer.write(frame)
        os.remove(os.path.join(frames_folder, frame_name))
    
    # Release the video writer
    video_writer.release()

if __name__ == '__main__':
    # Example usage
    frames_folder = './annotated_frames'
    output_video_path = 'annnotated_video.mp4'
    frame_rate = 30  # Adjust based on your video's frame rate

    frames_to_video(frames_folder, output_video_path, frame_rate)

    print(f"Video saved as {output_video_path}")