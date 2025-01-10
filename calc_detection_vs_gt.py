import os
import numpy as np
import cv2

def load_bboxes(file_path):
    with open(file_path, 'r') as f:
        bboxes = []
        for line in f:
            bbox = list(map(float, line.strip().split()))
            bboxes.append(bbox)
        return np.array(bboxes)

def load_dataset(images_folder, labels_folder, preds_folder):
    data = []
    for image_file in os.listdir(images_folder):
        image_id = os.path.splitext(image_file)[0]
        label_file = os.path.join(labels_folder, f"{image_id}.txt")
        pred_file = os.path.join(preds_folder, f"{image_id}.txt")

        if os.path.exists(label_file) and os.path.exists(pred_file):
            gt_bboxes = load_bboxes(label_file)
            pred_bboxes = load_bboxes(pred_file)
            data.append((image_id, gt_bboxes, pred_bboxes))
    return data

#PATHS -> user input
images_folder = 'INSERT SOURCE'
labels_folder = 'INSERT SOURCE'
preds_folder = 'INSERT SOURCE'





data = load_dataset(images_folder, labels_folder, preds_folder)

from sklearn.metrics import precision_recall_curve
from collections import defaultdict

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    
    return iou

def match_bboxes(gt_bboxes, pred_bboxes, iou_threshold=0.5):
    gt_matched = [False] * len(gt_bboxes)
    pred_matched = [False] * len(pred_bboxes)
    
    ious = []
    for i, gt in enumerate(gt_bboxes):
        for j, pred in enumerate(pred_bboxes):
            iou = compute_iou(gt, pred)
            if iou >= iou_threshold:
                ious.append((iou, i, j))
    
    ious = sorted(ious, reverse=True, key=lambda x: x[0])
    
    tp, fp, fn = 0, 0, 0
    for iou, i, j in ious:
        if not gt_matched[i] and not pred_matched[j]:
            tp += 1
            gt_matched[i] = True
            pred_matched[j] = True
    
    fp = len(pred_bboxes) - tp
    fn = len(gt_bboxes) - tp
    
    return tp, fp, fn

ious = defaultdict(list)
tp, fp, fn = 0, 0, 0

for image_id, gt_bboxes, pred_bboxes in data:
    t, f, n = match_bboxes(gt_bboxes, pred_bboxes)
    tp += t
    fp += f
    fn += n

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"Precision: {precision}")
print(f"Recall: {recall}")



def calculate_ap(precision, recall):
    precision = np.concatenate(([0.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    
    indices = np.where(recall[1:] != recall[:-1])[0]
    
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    
    return ap

def compute_map(data):
    all_precisions = []
    all_recalls = []
    
    for image_id, gt_bboxes, pred_bboxes in data:
        if len(pred_bboxes) == 0:
            continue
        
        pred_bboxes = sorted(pred_bboxes, key=lambda x: -x[4])
        tp, fp, fn = match_bboxes(gt_bboxes, pred_bboxes)
        
        precisions, recalls = [], []
        for i in range(len(pred_bboxes)):
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        
        all_precisions.append(precisions)
        all_recalls.append(recalls)
    
    map50 = 0
    for precisions, recalls in zip(all_precisions, all_recalls):
        map50 += calculate_ap(precisions, recalls)
    
    map50 /= len(all_precisions)
    return map50

map50 = compute_map(data)
print(f"mAP50: {map50}")



def apply_bounding_boxes(images_folder, bboxes_folder, output_folder='WILOR_leg'):
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # List all image files in the images folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        # Read the image
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to read {image_path}")
            continue
        
        # Get the corresponding bounding box file
        bbox_file = os.path.join(bboxes_folder, os.path.splitext(image_file)[0] + '.txt')
        
        if not os.path.exists(bbox_file):
            print(f"Bounding box file {bbox_file} does not exist for image {image_file}")
            with open(bbox_file, 'w') as f:
                print(f"Created empty file: {bbox_file}")
            # This creates an empty file
            continue
        
        # Read the bounding boxes from the file
        with open(bbox_file, 'r') as f:
            bboxes = f.readlines()
        
        height, width, _ = image.shape
        
        for bbox in bboxes:
            # YOLO format: class x_center y_center width height (all normalized)
            bbox_data = bbox.strip().split()
            class_id = int(bbox_data[0])
            x_center, y_center, box_width, box_height = map(float, bbox_data[1:])
            
            # Convert to pixel coordinates
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height
            
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)
            
            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Optionally, you can add the class label
            label = str(class_id)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save the resulting image to the output folder
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)
        print(f"Saved {output_path}")
apply_bounding_boxes(images_folder, preds_folder)