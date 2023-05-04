import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, '../utils/')


class WINDOWEDValidationDatasetImages(Dataset):

    def __init__(self, absolute_path_data_folder, folder_file_mapping_path):
        self.data_folder = absolute_path_data_folder
        self.folder_file_mapping = pd.read_csv(folder_file_mapping_path)
        self.transform = torchvision.transforms.ToTensor()

    def __len__(self):
        folder = glob.glob(os.path.join(self.data_folder, "**/*.json"), recursive=True)
        return len(folder)

    def __getitem__(self, index):
        folder, num_windows = self._get_window_folder_and_num_windows(index)
        label = self._get_audio_common_label(folder)
        full_folder_path = os.path.join(self.data_folder, str(folder))
        window_list = self._get_audio_windows(full_folder_path, num_windows)

        return window_list, np.array(label)

    def _get_audio_windows(self, full_folder_path, num_windows):
        window_list = []
        for i in range(num_windows):
            path = os.path.join(full_folder_path, f'W{i + 1}.png')
            img = Image.open(path).convert('RGB')
            image = self.transform(img)
            window_list.append(image)
            img.close()

        return window_list

    def _get_window_folder_and_num_windows(self, index):
        folder = self.folder_file_mapping.iloc[index][1]
        num_windows = self.folder_file_mapping.iloc[index][2]

        return folder, num_windows

    def _get_audio_common_label(self, folder_index):
        file = os.path.join(self.data_folder, str(folder_index), str(folder_index) + ".json")
        with open(file, 'r') as openfile:
            description = json.load(openfile)
        return list(description['label'].values())
