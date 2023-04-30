import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '../data/')

from data.MIXEDDataset import MIXEDDataset
from data.WINDOWEDValidationDataset import WINDOWEDValidationDataset


def create_data_loader(train_data, batch_size, num_workers, shuffle):
    dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataloader

def generate_training_images(dataloader, save_path, image_format):
    images = tqdm(dataloader, leave=False)

    for i, (image, _) in enumerate(images):
        os.chdir(os.path.join(save_path, str(i+1)))
        plt.imsave(f"{i+1}.{image_format}", image[0][0])


def generate_validation_images(dataloader, save_path, image_format):
    files = tqdm(dataloader, leave=False)

    for i, (images, _) in enumerate(files):
        os.chdir(os.path.join(save_path, str(i + 1)))
        for j, window in enumerate(images):
            plt.imsave(f"W{j + 1}.{image_format}", window[0][0])


if __name__ == "__main__":

    BATCH_SIZE = 1
    ABSOLUTE_PATH_DATA_FOLDER = '/home/dominik/Work/Lumen Datascience/LDS-Audio-Classification-2023-RiTehc/MIXED_Training_Data'
    NEW_SAMPLERATE = 22050  # TODO
    NEW_CHANNELS = 1
    MAX_NUM_SAMPLES = 66150  # TODO
    SHIFT_PERCENT = 0.1
    N_MELS = 64  # height of spec
    N_FFT = 1024
    MAX_MASK_PERCENT = 0.1
    N_FREQ_MASKS = 2
    N_TIME_MASKS = 2
    MAX_MIXES = 5
    MAX_DECIBEL = 80
    HOP_LEN = 517  # width of spec = Total number of samples / hop_length
    NUM_WORKERS = 10

    ds = MIXEDDataset(
        ABSOLUTE_PATH_DATA_FOLDER,
        NEW_SAMPLERATE,
        NEW_CHANNELS,
        MAX_NUM_SAMPLES,
        SHIFT_PERCENT,
        N_MELS,
        N_FFT,
        MAX_MASK_PERCENT,
        N_FREQ_MASKS,
        N_TIME_MASKS,
        MAX_MIXES,
        MAX_DECIBEL,
        HOP_LEN
    )

    ABSOLUTE_PATH_VAL_DATA_FOLDER = '/home/dominik/Work/Lumen Datascience/LDS-Audio-Classification-2023-RiTehc/WINDOWED_Validation_Data'
    FOLDER_FILE_MAPPING_PATH = os.path.join(ABSOLUTE_PATH_VAL_DATA_FOLDER, 'folder_file_mapping.csv')

    vds = WINDOWEDValidationDataset(
        ABSOLUTE_PATH_VAL_DATA_FOLDER,
        FOLDER_FILE_MAPPING_PATH,
        NEW_SAMPLERATE,
        NEW_CHANNELS,
        MAX_NUM_SAMPLES,
        N_MELS,
        N_FFT,
        MAX_DECIBEL,
        HOP_LEN
    )

    validation_dataloader = create_data_loader(vds, BATCH_SIZE, NUM_WORKERS, False)
    train_dataloader = create_data_loader(ds, BATCH_SIZE, NUM_WORKERS, False)

    generate_training_images(train_dataloader, ABSOLUTE_PATH_DATA_FOLDER, "png")
    generate_validation_images(validation_dataloader, ABSOLUTE_PATH_VAL_DATA_FOLDER, "png")
