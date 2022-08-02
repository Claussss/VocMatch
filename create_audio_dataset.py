# -----------------------------------------------------------
# Splits all the files from the specified directory to train, valid, and test datasets using the specified percentages.
# -----------------------------------------------------------


import os
from tqdm import tqdm
import shutil
import numpy as np

path_to_result = r'D:\Education\Projects\FiZam\data\audio_dataset'
path_to_files = r'D:\Education\Projects\FiZam\data\sections'
# Train, valid, test percentages
# Has to sum up to 1
split_percentages = {'train': 0.8,
                     'valid': 0.1,
                     'test': 0.1}

if __name__ == '__main__':
    file_names = os.listdir(path_to_files)
    files_num = len(file_names)
    indexes = np.arange(files_num)
    # Shuffle indexes inplace
    np.random.shuffle(indexes)

    start_indx = 0
    for dir_name, split_percentage in split_percentages.items():
        print(dir_name)
        full_result_path = os.path.join(path_to_result, dir_name)
        if not os.path.exists(full_result_path):
            os.mkdir(full_result_path)

        num_of_items = int(files_num * split_percentage)
        for i in tqdm(range(start_indx, start_indx + num_of_items)):
            file_name = file_names[indexes[i]]
            old_full_path_to_file = os.path.join(path_to_files, file_name)
            new_full_path_to_file = os.path.join(full_result_path, file_name)
            shutil.move(old_full_path_to_file, new_full_path_to_file)

        start_indx += num_of_items

    shutil.rmtree(path_to_files)
