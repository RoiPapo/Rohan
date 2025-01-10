import cv2
import glob
from ultralytics import YOLO
import os
import torch
import numpy as np

# Load a model
model = YOLO("/rohan/experiments/yydhrcho_autumn-sweep-1/Pipline/detector_10/weights/best.pt")  # use cycle 10 weights

video_folder = "USER INPUT: ADD SOURCE TO a VIDEO FOLDER"
vids = glob.glob(f'{video_folder}/**/*')



for id,vid in enumerate(vids):
    results = model(source=vid,save=True, conf=0.3, device=[1])
    folder_name= 'ENTER OUTPUT FOLDER SOURCE' 
    for frame_idx,res in enumerate(results):
            if len(results[frame_idx].boxes.xywhn) > 0:
                np.savetxt(f"{folder_name}/labels/{results[frame_idx].path.split('/')[-1][:-4]}.txt", results[frame_idx].boxes.xywhn.cpu().numpy(),  fmt='%f', delimiter=' ')
                cv2.imwrite(f"{folder_name}/images/{results[frame_idx].path.split('/')[-1][:-4]}.jpg",results[frame_idx].orig_img)

