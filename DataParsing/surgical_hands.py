import glob
import os
import shutil
# Path to the folder containing nested folders representing videos
videos_folder = "USE YOUR SOURCE PATH"

# Path to the data folder where you want to store all frames
data_folder = "USE YOUR SOURCE PATH"
annotations_path = "USE YOUR SOURCE PATH"

def get_id_from_filename(s: str):
    return s.split('/')[-1]


def delete_differance(A_folder, B_folder):
    """
    A-B: if the groups are not the same we delete the extras
    @return: list of names to delete
    """
    set_A = glob.glob(A_folder)
    set_B = glob.glob(B_folder)
    print(len(set_A), len(set_B))
    photo_IDS_A = set([get_id_from_filename(os.path.basename(s)[:-4]) for s in set_A])
    photo_IDS_B = set([get_id_from_filename(os.path.basename(s)[:-4]) for s in set_B])
    photo_IDS_A = {word[1:] if word.startswith('n') else word for word in photo_IDS_A}
    photo_IDS_B = {word[1:] if word.startswith('n') else word for word in photo_IDS_B}
    items_to_delete = photo_IDS_A.difference(photo_IDS_B)
    print(f"{len(items_to_delete)} items to be delete which is {len(items_to_delete)/len(set_A)} ")
    # exit()
    for src in set_A:
        if any(file in src for file in items_to_delete):
            os.remove(src)
            print(src, "were removed")
    print(len(items_to_delete), " in total were removed")



def parse_annotations(anot_path):
    import json

    # Load the JSON data from the file
    with open(anot_path, 'r') as file:
        data = json.load(file)

    # Create an empty dictionary to store image IDs and their bounding boxes
    image_boxes_dict = {}
    image_metadata_dict={}

    # Iterate through the data and populate the dictionary
    for video in data.values():
        for metadata in video['images']:
            image_id = metadata['id']
            frame_width = metadata['frame_width']
            frame_height = metadata['frame_height']
            if image_id in image_metadata_dict:
                # If the image_id exists, append the bbox to its list of bounding boxes
                image_metadata_dict[image_id].append([frame_width,frame_height])
            else:
                # If the image_id doesn't exist, create a new list with the bbox
                image_metadata_dict[image_id] = [frame_width,frame_height]
        for annotation in video['annotations']:
            image_id = annotation['image_id']
            bbox = [0]+ pascal_voc_to_yolo(*(annotation['bbox']+ image_metadata_dict[image_id]))
        
            # Check if the image_id already exists in the dictionary
            if image_id in image_boxes_dict:
                # If the image_id exists, append the bbox to its list of bounding boxes
                image_boxes_dict[image_id].append(bbox)
            else:
                # If the image_id doesn't exist, create a new list with the bbox
                image_boxes_dict[image_id] = [bbox]


    # Print the resulting dictionary
    print(image_boxes_dict)

    # Loop over the dictionary
    for image_id, boxes_list in image_boxes_dict.items():
        # Create a new text file with the name of the key
        file_name = f"{image_id}.txt"
        with open(file_name, 'w') as txt_file:
            # Write the elements of each list with space separator in a new line
            for bbox in boxes_list:
                # Convert each element to string and join them with space separator
                bbox_str = ' '.join(map(str, bbox))
                # Write the bbox string to the text file
                txt_file.write(f"{bbox_str}\n")
    print("DONE !")
def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1) / (2 * image_w)), ((y2 + y1) / (2 * image_h)), (x2 - x1) / image_w, (y2 - y1) / image_h]

def copy_images():
    # Create the data folder if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Traverse through the nested folders
    for root, dirs, files in os.walk(videos_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):  # Assuming images are in jpg or png format
                video_name = os.path.basename(root)  # Name of the video folder
                frame_name = file  # Name of the frame/image
                new_name = f"{video_name}_{frame_name}"  # New name for the image

                # Source path of the image
                source_path = os.path.join(root, file)

                # Destination path where the image will be moved
                destination_path = os.path.join(data_folder, new_name)

                # Move the image to the data folder with the new name
                shutil.copy(source_path, destination_path)

    print("All frames have been moved to the data folder.")



def group_files_by_video(directory):
    """
    Groups files in a directory by [NAME_OF_VIDEO] and moves them into corresponding folders.

    Args:
    - directory (str): Path to the directory containing the files.

    Returns:
    - None
    """

    # Create a dictionary to hold file paths based on video names
    video_files = {}

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Split the filename into name and frame number
        name, frame_number = filename.rsplit('_', 1)

        # Extract the video name
        video_name = name

        # Check if the video name already exists in the dictionary
        if video_name in video_files:
            # Append the file path to the list of files for this video name
            video_files[video_name].append(filename)
        else:
            # Create a new list with this file path
            video_files[video_name] = [filename]

    # Create folders for each video name and move files into corresponding folders
    for video_name, files in video_files.items():
        # Create a folder for the video name if it doesn't exist
        video_folder = os.path.join(directory, video_name)
        os.makedirs(video_folder, exist_ok=True)

        # Move files into the video folder
        for filename in files:
            src = os.path.join(directory, filename)
            dst = os.path.join(video_folder, filename)
            shutil.move(src, dst)

