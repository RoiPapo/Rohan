import os
import glob
import json
import tqdm
import shutil


def get_id_from_filename(s: str):
    return s.split('/')[-1]


def duplicate_with_prefix(folder_path):
    prefix = "blue_gloves_"  # The prefix to add to each file name
    all_files = glob.glob(folder_path+'/*')
    for file_name in all_files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            if len(os.path.basename(file_name).split('_')) ==1:
                # print(os.path.basename(file_name))
                new_file_name = prefix + os.path.basename(file_name)
                new_file_path = os.path.join(folder_path, new_file_name)
                print (new_file_path)
                shutil.copyfile(file_path, new_file_path)



def delete_all_from_number(json_src, foldersrc):
    """
    the json contains the numbers that we want to see on the set
    the folder contains all the files, we want to delete numbers that are not on folder
    """    
    with open(json_src, 'r') as json_file:
        num_list = json.load(json_file)
    count =0
    files = glob.glob(foldersrc+'/*')
    pbar = tqdm.tqdm(total=len(files))
    for file in files:
          pbar.update(1)
          file_id= os.path.basename(file).split('_')[-1][:-4]
          if str(file_id) not in num_list:
              count+=1
              
              print(f'{os.path.basename(file)} will be deleted')
            #   os.remove(file) # dangerous
    print (f'{count} were deleted which are' , count/len(files), '%')


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
    exit()
    for src in set_A:
        if any(file in src for file in items_to_delete):
            # os.remove(src)
            print(src, "were removed")
    print(len(items_to_delete), " in total were removed")

    # return items_to_delete


def add_prefix_to_files(source_folder, prefix):
    """
    Adds a prefix to each file name in the given source folder.

    Args:
        source_folder (str): Path to the source folder.
        prefix (str): Prefix string to be added to file names.
    """
    # Iterate through all items in the source folder
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)  # Get the full path of the item
        if os.path.isfile(item_path):  # Check if item is a file
            # Extract the file name and file extension
            file_name, file_extension = os.path.splitext(item)

            # Create the new file name with the prefix added
            new_file_name = prefix + file_name + file_extension

            # Construct the new path for the file with the prefix added
            new_item_path = os.path.join(source_folder, new_file_name)

            # Rename the file with the prefix added
            os.rename(item_path, new_item_path)
            print(f"Renamed {item} to {new_file_name}")


def get_nested_files(root_directory):
    import os

    file_sources = []

    for dirpath, dirnames, filenames in os.walk(root_directory):
        for file in filenames:
            file_sources.append(os.path.join(dirpath, file))
    return file_sources


def get_subfolder_paths(root_folder):
    """
    Retrieves the paths of all subfolders under the given root folder.

    Args:
        root_folder (str): Path to the root folder.

    Returns:
        list: A list of subfolder paths.
    """
    subfolder_paths = []
    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path):
            subfolder_paths.append(item_path)
    return subfolder_paths


def count_numer_of_files():
    import os
    for root_directory in get_subfolder_paths("USE YOUR PATH"):
        file_count = 0

        for dirpath, dirnames, filenames in os.walk(root_directory):
            for file in filenames:
                # if file.endswith('.png') or file.endswith('.jpg'):
                   file_count += 1

        print("Total number of files in", root_directory, "is", file_count)

def transfer_files(sources,output):
    import shutil

    count_numer_of_files()
    for kind in sources:
        kind_name = kind.split('/')[-1]
        all_files = get_nested_files(kind)
        for source in all_files:
            loc= source.replace(kind_name, output,1)
            shutil.copy2(source,loc ) # complete target filename given




def split_train_to_train_val(source_folder):
    import random
    # Set the paths to the train and test directories
    train_dir = os.path.join(source_folder, 'train')
    test_dir = os.path.join(source_folder, 'test')

    # Create test directories if they don't exist
    os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)

    # Get the list of image files in the train directory
    label_files = os.listdir(os.path.join(train_dir, 'labels'))

    # Calculate the number of files to move (20% of the total)
    num_files_to_move = int(len(label_files) * 0.2)

    # Randomly select the files to move
    files_to_move = random.sample(label_files, num_files_to_move)

    # Move the selected files from the train directory to the test directory
    for filename in files_to_move:
        if filename.startswith("blue"):
            continue
        # Move image file
        shutil.move(
            os.path.join(train_dir, 'labels', filename),
            os.path.join(test_dir, 'labels', filename)
        )


        # Move label file
        label_filename = os.path.splitext(filename)[0] + '.jpg'
        shutil.move(
            os.path.join(train_dir, 'images', label_filename),
            os.path.join(test_dir, 'images', label_filename)
        )
