import torchvision.transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import json
import sys
import pandas as pd
import numpy as np

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


if __name__ == "__main__":
    ABSOLUTE_PATH_DATA_FOLDER = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/WINDOWED_Validation_Data'
    FOLDER_FILE_MAPPING_PATH = os.path.join(ABSOLUTE_PATH_DATA_FOLDER, 'folder_file_mapping.csv')

    ds = WINDOWEDValidationDatasetImages(
        ABSOLUTE_PATH_DATA_FOLDER,
        FOLDER_FILE_MAPPING_PATH
    )

    print(f'There are {len(ds)} samples')
    signal, label = ds[-1]
    class_names = ['tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']

    import matplotlib.pyplot as plt
    import numpy as np

    indexes = np.where(np.array(label) == 1)[0]
    title = [class_names[i] for i in indexes]
    print(signal[0].shape)
    print(title)

    # plt.imsave('dada.png', signal[0])
    # print(min(signal[0]))
    # print(max(signal[0]))

    import cv2

    cv2.imshow('image', signal[1].permute(1, 2, 0).numpy())
    cv2.waitKey(0)
    # plt.imshow(signal[0])
    # plt.title(title)
    # plt.show()

    # a = 1
