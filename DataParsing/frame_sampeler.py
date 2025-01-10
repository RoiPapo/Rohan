from moviepy.editor import VideoFileClip
from PIL import Image
import os

def extract_frames(video_path, output_directory, fps=5, resolution=(1920, 1080)):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load the video clip
    clip = VideoFileClip(video_path)

    # Calculate the duration of each frame
    duration_per_frame = 1.0 / fps

    # Iterate over each frame and save it as an image
    for i, frame in enumerate(clip.iter_frames(fps=fps, dtype='uint8'), start=1):
        # Construct the file name for the frame
        filename = os.path.join(output_directory, f"frame-{i:04d}.png")

        # Save the frame as an image
        clip = Image.fromarray(frame)
        clip.save(filename)

    # Close the clip
    clip.close()
