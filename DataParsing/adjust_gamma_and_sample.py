import glob
import ffmpeg
import os


def extract_frames(video_path, output_dir, letter, adjust_gamma=False, gamma=1, fps=1/4, resolution='1920x1080'):
    """
    Extracts frames from a video, optionally adjusting gamma, at a specified fps and resolution.

    Args:
    video_path (str): Path to the input video.
    output_dir (str): Directory to save the extracted frames.
    letter (str): Prefix letter for the output frame filenames.
    adjust_gamma (bool): Flag to determine if gamma adjustment should be applied.
    gamma (float): Gamma value for adjustment (only used if adjust_gamma is True).
    fps (int): Frames per second for frame extraction.
    resolution (str): Resolution for the extracted frames, format 'widthxheight'.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Filename handling
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_output_path = os.path.join(output_dir, f"{letter}{base_name}_frame-%04d.png")
    
    # Prepare the ffmpeg input
    input_video = ffmpeg.input(video_path)
    
    # Apply gamma adjustment if the flag is set
    if adjust_gamma:
        input_video = input_video.filter('eq', gamma=gamma)
    
    # Extract frames
    (
        input_video
        .filter('fps', fps=fps)
        .output(frame_output_path, video_bitrate='5000k', s=resolution)
        .run(overwrite_output=True)
    )

