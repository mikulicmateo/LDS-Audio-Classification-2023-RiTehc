import glob
import json
import os
import sys

import numpy as np
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, '../utils/')


class MIXEDDatasetImages(Dataset):

    def __init__(self, absolute_path_data_folder):
        self.data_folder = absolute_path_data_folder
        self.transform = torchvision.transforms.ToTensor()

    def __len__(self):
        folder = glob.glob(os.path.join(self.data_folder, "**/*.png"), recursive=True)
        return len(folder)

    def __getitem__(self, index):
        index = index + 1
        path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)

        img = Image.open(path).convert('RGB')
        img = self.transform(img)

        return img, np.array(label)

    def _get_audio_sample_path(self, index):
        return os.path.join(self.data_folder, str(index), str(index) + ".png")

    def _get_audio_sample_label(self, index):
        file = os.path.join(self.data_folder, str(index), str(index) + ".json")
        with open(file, 'r') as openfile:
            description = json.load(openfile)
        return list(description['label'].values())
