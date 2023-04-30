import torchvision.transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import json
import sys

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

        img = Image.open(path)
        img = self.transform(img)

        return img, label

    def _get_audio_sample_path(self, index):
        return os.path.join(self.data_folder, str(index), str(index) + ".png")

    def _get_audio_sample_label(self, index):
        file = os.path.join(self.data_folder, str(index), str(index) + ".json")
        with open(file, 'r') as openfile:
            description = json.load(openfile)
        return list(description['label'].values())


if __name__ == "__main__":
    ABSOLUTE_PATH_DATA_FOLDER = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/MIXED_Training_Data'


    ds = MIXEDDataset(
        ABSOLUTE_PATH_DATA_FOLDER
    )

    print(f'There are {len(ds)} samples')
    signal, label = ds[0]
    class_names = ['tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']

    import matplotlib.pyplot as plt
    import numpy as np

    indexes = np.where(np.array(label) == 1)[0]
    title = [class_names[i] for i in indexes]
    print(signal.shape)
    print(title)

    # # plt.imsave('dada.png', signal[0])
    # print(min(signal[0]))
    # print(max(signal[0]))

    # print(signal[0])ž
    import cv2
    cv2.imshow('image', signal.permute(1, 2, 0).numpy())
    cv2.waitKey(0)

    plt.imshow(signal)
    plt.title(title)
    plt.show()

    # a = 1
