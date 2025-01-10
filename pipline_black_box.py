from ultralytics import YOLO
import glob
import numpy as np
import cv2
import os
import yaml
import shutil
from natsort import natsorted
import torch
from ultralytics.yolo.utils.metrics import box_iou
from collections import defaultdict
import copy
import wandb
wandb.login()


def convert_to_regular_dict(d):
    return {k: v for k, v in d.items()}


def delete_bad_preds_huristic(root_folder_path, distance_threshold, huristic_type, track_hist=None, track_ids=None):
    # Initialize an empty list to store the files to remove
    train_folder_path = root_folder_path + '/labels'
    val_folder_path = train_folder_path.replace("Retrain", "Reval")
    img_path = root_folder_path + '/images'
    val_img_folder_path = img_path.replace("Retrain", "Reval")
    files_to_remove = []
    dict_id_box = defaultdict(list)
    all_files = natsorted(os.listdir(train_folder_path))
    counter = 0
    # Iterate over the files in chunks of 10
    if huristic_type == "follow":
        for id, amount in convert_to_regular_dict(track_hist).items():
            if amount < 5 and id in track_ids:
                counter += 1
                files_to_remove.append(track_ids[id][0])

        print(f"{counter} deleted which are {counter/len(track_hist.keys())} %")

    if huristic_type == "working_zone":
        for i in range(0, len(all_files), 10):
            # Collect bounding box coordinates from the current chunk of files
            bounding_boxes = []
            for j in range(i, i + 10):
                if len(all_files) > j:
                    filename = all_files[j]

                if filename.endswith(".txt"):
                    coordinates = np.array(np.loadtxt(os.path.join(
                        train_folder_path, filename)), dtype=np.float32)
                    if coordinates.ndim == 1:
                        bounding_boxes.append((coordinates[1:], filename))
                        dict_id_box[filename].append(coordinates[1:])
                    else:
                        for bbox in coordinates:
                            bounding_boxes.append((bbox[1:], filename))
                            dict_id_box[filename].append(bbox[1:])

            bboxes_no_id = [a[0] for a in bounding_boxes]
            ids_of_10_files = [a[1] for a in bounding_boxes]

            # Calculate the centroid for the current chunk of bounding boxes
            centroid = np.mean(bboxes_no_id, axis=0)

            # Calculate the distances between the bounding boxes and the centroid
            distances = np.linalg.norm(bboxes_no_id - centroid, axis=1)

            # Add the files containing bounding boxes far from the centroid to the list to remove
            for j, distance in enumerate(distances):
                if distance > distance_threshold:
                    files_to_remove.append(ids_of_10_files[j])

    # Remove the files containing bounding boxes far from the centroid
    for filename in set(files_to_remove):
        try:
            os.remove(os.path.join(train_folder_path, filename))
            os.remove(os.path.join(img_path, filename[:-3] + "jpg"))
            print(f" deleting {os.path.join(train_folder_path, filename)}")
        except:
            os.remove(os.path.join(val_folder_path, filename))
            os.remove(os.path.join(val_img_folder_path, filename[:-3] + "jpg"))
            print(f" deleting {os.path.join(val_folder_path, filename)}")

    print(len(files_to_remove), "were removed using the huristic")


def leading_zero_bugfix(directory):
    # Process all text files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            if not os.path.isfile(file_path):
                print(f"File not found: {file_path}")
                return

            # Open the file for reading and writing
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Modify the lines
            modified_lines = [
                '0' + line[8:] if line.startswith('0.000000') else line for line in lines]
            with open(file_path, 'w') as file:
                file.writelines(modified_lines)


def find_most_recent_train_folder(directory):
    most_recent_folder = None
    most_recent_time = 0

    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if "train" in dir_name:
                folder_path = os.path.join(root, dir_name)
                folder_time = os.path.getctime(folder_path)

                if folder_time > most_recent_time:
                    most_recent_time = folder_time
                    most_recent_folder = folder_path

    return os.path.abspath(most_recent_folder)


def generate_yaml_file(output_file, run_name):
    source = f"/experiments/{str(run_name)}"
    data = {
        "path": os.getcwd() + source,
        "train": "Retrain",
        "val": "Reval",
        "names": {
            0: "hand"
        }
    }
    yaml_source = f"{os.getcwd()+source}/{output_file}"
    with open(yaml_source, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    return os.path.abspath(yaml_source)


def setup_retrain_folders(folder_name, run_name):
    folder_name = os.path.join("experiments", str(run_name), folder_name)
    labels_folder = "labels"
    images_folder = "images"
    early_prediction_folder = "early_predictions"

    if os.path.exists(folder_name):
        # Remove the "labels" and "images" folders and their contents
        labels_path = os.path.join(folder_name, labels_folder)
        images_path = os.path.join(folder_name, images_folder)
        early_prediction_path = os.path.join(
            folder_name, early_prediction_folder)
        if os.path.exists(labels_path):
            shutil.rmtree(labels_path)
        if os.path.exists(images_path):
            shutil.rmtree(images_path)
        if os.path.exists(early_prediction_path):
            shutil.rmtree(early_prediction_path)
    else:
        os.mkdir(folder_name)

    # Recreate "labels" and "images" folders
    labels_path = os.path.join(folder_name, labels_folder)
    images_path = os.path.join(folder_name, images_folder)
    early_prediction_path = os.path.join(folder_name, early_prediction_folder)

    os.mkdir(labels_path)
    os.mkdir(images_path)
    os.mkdir(early_prediction_path)

    return os.path.abspath(folder_name)


def setup_experiment_folder(run_name):
    folder_path = os.path.join("experiments", str(run_name))
    os.makedirs(folder_path, exist_ok=True)


def main():
    wandb.init()
    DEVICE = 0
    config = wandb.config
    never_seen_data_loc = "SOURCE TO YOUR FOLDER"  # test data
    conf_persentage = config.conf_persentage
    huristic_flag = config.huristic_flag
    # load a Trained model
    model = YOLO("rohan_pretrained.pt")

    training_cycles = 10
    run_name= f"{sweep_id}_{wandb.run.name}"
    setup_experiment_folder(run_name)
    for train_cycle in range(training_cycles):
        track_hist = defaultdict(int)
        track_ids = defaultdict(list)
        re_train_folder = setup_retrain_folders("Retrain", run_name)
        re_val_folder = setup_retrain_folders("Reval", run_name)
        print("initial Predicting...")
        tracking_model = copy.deepcopy(model)
        results = tracking_model.track(source=never_seen_data_loc, save=False, conf=conf_persentage, device=[
            DEVICE], tracker="botsort.yaml")
        for frame_idx, res in enumerate(results):
            if len(results[frame_idx].boxes.xywhn) > 0:
                ids = results[frame_idx].boxes.id
                if ids is not None:
                    for id in ids:
                        track_hist[int(id.item())] += 1
                        track_ids[int(id.item())].append(
                            f"{results[frame_idx].path.split('/')[-1][:-4]}_frame{str(frame_idx)}.txt")

                labels_matrix_with_zero_class = np.hstack((np.zeros((results[frame_idx].boxes.xywhn.cpu(
                ).numpy().shape[0], 1)), results[frame_idx].boxes.xywhn.cpu().numpy()))
                labels_matrix_with_zero_class[0][0] = int(0)
                if frame_idx % 4 == 0:
                    np.savetxt(f"{re_train_folder}/labels/{results[frame_idx].path.split('/')[-1][:-4]}_frame{str(frame_idx)}.txt",
                               labels_matrix_with_zero_class, fmt='%f', delimiter=' ')
                    cv2.imwrite(
                        f"{re_train_folder}/images/{results[frame_idx].path.split('/')[-1][:-4]}_frame{str(frame_idx)}.jpg", results[frame_idx].orig_img)
                    cv2.imwrite(
                        f"{re_train_folder}/early_predictions/{results[frame_idx].path.split('/')[-1][:-4]}_frame{str(frame_idx)}_box.jpg", results[frame_idx].plot())

                else:
                    np.savetxt(f"{re_val_folder}/labels/{results[frame_idx].path.split('/')[-1][:-4]}_frame{str(frame_idx)}.txt",
                               labels_matrix_with_zero_class, fmt='%f', delimiter=' ')
                    cv2.imwrite(
                        f"{re_val_folder}/images/{results[frame_idx].path.split('/')[-1][:-4]}_frame{str(frame_idx)}.jpg", results[frame_idx].orig_img)
                    cv2.imwrite(
                        f"{re_val_folder}/early_predictions/{results[frame_idx].path.split('/')[-1][:-4]}_frame{str(frame_idx)}_box.jpg", results[frame_idx].plot())

        leading_zero_bugfix(
            f"{re_val_folder}/labels")
        leading_zero_bugfix(
            f"{re_train_folder}/labels")
        if huristic_flag is not None:
            delete_bad_preds_huristic(re_train_folder, 0.3, huristic_type=huristic_flag,
                                      track_hist=track_hist, track_ids=track_ids)
        print("Re-Train Set is ready...")
        print("creating training Yaml...")
        # generate_yaml_file(never_seen_data_loc.split("/HandsDetection/")[-1].split("/images")[0],"Retrain.yaml")
        yaml_source = generate_yaml_file("Retrain.yaml", run_name)
        print("finetuning based on previous predictions for 5 epoches...")

        model.train(data=yaml_source, device=DEVICE, epochs=5,
                    project=f"experiments/{str(run_name)}/Pipline", name=f"detector_{train_cycle}")  # Fintune the model
        metrics = model.val(
            data=f"USE YOUR EXPERIMENT YOLO YAML", save=False)

        wandb.log({'Precision': metrics.results_dict['metrics/precision(B)'],
                  'Recall': metrics.results_dict['metrics/recall(B)'],
                   'mAP50': metrics.results_dict['metrics/mAP50(B)'],
                   'Total': (metrics.results_dict['metrics/precision(B)'] + metrics.results_dict['metrics/mAP50(B)'] + metrics.results_dict['metrics/recall(B)']) / 3})
        model = YOLO(f"experiments/{str(run_name)}/Pipline/detector_{train_cycle}/weights/best.pt")
    print("predicting using the fintuned model...")
    results = model(source=never_seen_data_loc, save=True, conf=0.25, device=DEVICE) 


if __name__ == '__main__':
    if os.path.exists("experiments"):
        print("cleaning for new experiment..")
        # shutil.rmtree("experiments")


    sweep_config = {
        'method': 'grid',
        'parameters': {
            'hypers': {
                'values': ['no_flip.yaml']},
            'huristic_flag': {
                'values': ["follow","working_zone"]},
            'conf_persentage': {
                'values': [0.25]
            },
        }
    }    
    sweep_id = wandb.sweep(sweep_config, project="RoHan")
    wandb.agent(sweep_id, function=main)
