import os

# Path to the directory containing the text files
directory = '/data/home/roipapo/HandsDetection/imgs/leg_frames_sample/labels'

# List all files in the directory
files = os.listdir(directory)

# Iterate through each file in the directory
for filename in files:
    if filename.endswith('.txt'):
        # Rename only text files
        new_filename = filename.split('_png')[0] + '.txt'
        # Create the full path for the old and new filenames
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_path, new_path)

print("File names have been updated.")