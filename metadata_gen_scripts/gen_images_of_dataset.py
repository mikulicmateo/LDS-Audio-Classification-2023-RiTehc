import json
import os

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.MIXEDDataset import MIXEDDataset
from data.WINDOWEDValidationDataset import WINDOWEDValidationDataset


def create_data_loader(train_data, batch_size, num_workers, shuffle):
    dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataloader


def generate_training_images(dataloader, save_path, image_format):
    images = tqdm(dataloader, leave=False)

    for i, (image, _) in enumerate(images):
        os.chdir(os.path.join(save_path, str(i + 1)))
        plt.imsave(f"{i + 1}.{image_format}", image[0][0])
        if i == 0:
            print(f"Image Shape: {image[0][0].shape}")


def generate_validation_images(dataloader, save_path, image_format):
    files = tqdm(dataloader, leave=False)

    for i, (images, _) in enumerate(files):
        os.chdir(os.path.join(save_path, str(i + 1)))
        for j, window in enumerate(images):
            plt.imsave(f"W{j + 1}.{image_format}", window[0][0])


def main():
    project_path = os.path.dirname(os.getcwd())
    config_file_path = os.path.join(project_path, "config.json")
    mixed_dataset_path = os.path.join(project_path, "MIXED_Training_Data")
    windowed_validation_dataset_path = os.path.join(project_path, "WINDOWED_Validation_Data")
    folder_file_mapping_path = os.path.join(windowed_validation_dataset_path, 'folder_file_mapping.csv')

    with open(config_file_path, "r") as config_file:
        config_dict = json.load(config_file)

    ds = MIXEDDataset(
        mixed_dataset_path,
        new_samplerate=config_dict["NEW_SAMPLERATE"],
        new_channels=config_dict["NEW_CHANNELS"],
        max_num_samples=config_dict["MAX_NUM_SAMPLES"],
        shift_limit=config_dict["SHIFT_PERCENT"],
        n_mels=config_dict["N_MELS"],
        n_fft=config_dict["N_FFT"],
        mask_percent=config_dict["MAX_MASK_PERCENT"],
        n_freq_masks=config_dict["N_FREQ_MASKS"],
        n_time_masks=config_dict["N_TIME_MASKS"],
        max_mixes=config_dict["MAX_MIXES"],
        db_max=config_dict["MAX_DECIBEL"],
        hop_len=config_dict["HOP_LEN"]
    )

    vds = WINDOWEDValidationDataset(
        windowed_validation_dataset_path,
        folder_file_mapping_path,
        new_samplerate=config_dict["NEW_SAMPLERATE"],
        new_channels=config_dict["NEW_CHANNELS"],
        max_num_samples=config_dict["MAX_NUM_SAMPLES"],
        n_mels=config_dict["N_MELS"],
        n_fft=config_dict["N_FFT"],
        db_max=config_dict["MAX_DECIBEL"],
        hop_len=config_dict["HOP_LEN"]
    )

    validation_dataloader = create_data_loader(vds, 1, config_dict["NUM_WORKERS"], False)
    train_dataloader = create_data_loader(ds, 1, config_dict["NUM_WORKERS"], False)

    generate_training_images(train_dataloader, mixed_dataset_path, "png")
    generate_validation_images(validation_dataloader, windowed_validation_dataset_path, "png")


if __name__ == "__main__":
    main()
