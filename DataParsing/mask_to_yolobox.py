import glob
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw


def delete_n_from_filenames(folder_path):
    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file starts with 'n'
        if filename.startswith('n'):
            new_filename = filename[1:]  # Remove the first character

            # Construct the new file path
            new_file_path = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(file_path, new_file_path)


def draw_bounding_box_yolo(image, bbox_yolo_all):
    # Get image dimensions
    image_height, image_width, _ = image.shape
    if len(bbox_yolo_all) !=2: 
        bbox_yolo_all=[bbox_yolo_all]

    for bbox_yolo in bbox_yolo_all:

        # Extract class and bounding box coordinates
        class_id = int(bbox_yolo[0])
        bbox_x, bbox_y, bbox_w, bbox_h = bbox_yolo[1:]

        # Convert YOLO bounding box to absolute coordinates
        bbox_x = int(bbox_x * image_width)
        bbox_y = int(bbox_y * image_height)
        bbox_w = int(bbox_w * image_width)
        bbox_h = int(bbox_h * image_height)

        # Calculate bounding box coordinates
        bbox_xmin = bbox_x - bbox_w // 2
        bbox_ymin = bbox_y - bbox_h // 2
        bbox_xmax = bbox_xmin + bbox_w
        bbox_ymax = bbox_ymin + bbox_h

        # Draw bounding box on the image
        color = (0, 255, 0)  # Green color for the bounding box
        thickness = 2
        cv2.rectangle(image, (bbox_xmin, bbox_ymin), (bbox_xmax, bbox_ymax), color, thickness)

        # Add class label to the bounding box
        label = f"Class: {class_id}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(image, (bbox_xmin, bbox_ymin - label_size[1] - 10),
                    (bbox_xmin + label_size[0], bbox_ymin - 10), color, cv2.FILLED)
        cv2.putText(image, label, (bbox_xmin, bbox_ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    return image


def get_coordinates(img: np.ndarray):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    # w, h = ymax - ymin, xmax - xmin
    # shape = [(40, 40), (w - 10, h - 10)]

    return xmin, ymin, xmax, ymax


def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1) / (2 * image_w)), ((y2 + y1) / (2 * image_h)), (x2 - x1) / image_w, (y2 - y1) / image_h]

