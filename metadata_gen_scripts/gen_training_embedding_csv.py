import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '../data/')

from data.MIXEDDataset import MIXEDDataset
from model.Encoder import Encoder
from model.UNet_original import UNet


def load_unet_state(unet_path):
    print("LOADING MODEL")
    unet = UNet(padding=1)
    unet_dict = torch.load(unet_path)

    unet.load_state_dict(unet_dict['model_state'])
    unet.to(device)
    print(f"LOADED MODEL, epoch {unet_dict['epoch']}"
          + f", time {unet_dict['time']}")

    return unet


def load_encoder_state(encoder_path):
    print("LOADING MODEL")
    encoder = Encoder()
    encoder_dict = torch.load(encoder_path)
    encoder.load_state_dict(encoder_dict['model_state'])

    print(f"LOADED MODEL, epoch {encoder_dict['epoch']}"
          + f", time {encoder_dict['time']}")

    return encoder


def create_data_loader(train_data, batch_size, num_workers, shuffle):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return train_dataloader


def generate_embeddings_encoder(encoder, device, dataloader, save_path):
    encoder.eval()
    os.chdir(save_path)
    loop = tqdm(dataloader, leave=False)

    first = True
    for i, (image_batch, label) in enumerate(loop):
        data = []
        image_batch = image_batch.to(device)
        embedding = encoder(image_batch)
        temp = [i, embedding.cpu().detach().numpy(), [int(l) for l in label]]
        data.append(temp)

        df = pd.DataFrame(data, columns=["index", "embedding", "label"])
        if first:
            first = False
            df.to_csv('unet_embeddings.csv', mode='w', header=True, index=False)
        else:
            df.to_csv('unet_embeddings.csv', mode='a', header=False, index=False)


def generate_embeddings_unet(unet, device, dataloader, save_path):
    unet.eval()
    os.chdir(save_path)
    loop = tqdm(dataloader, leave=False)

    first = True
    for i, (image_batch, label) in enumerate(loop):
        data = []
        image_batch = image_batch.to(device)
        embedding = unet(image_batch, True)
        temp = [i, embedding.cpu().detach().numpy(), [int(l) for l in label]]
        data.append(temp)

        df = pd.DataFrame(data, columns=["index", "embedding", "label"])
        if first:
            first = False
            df.to_csv('unet_embeddings.csv', mode='w', header=True, index=False)
        else:
            df.to_csv('unet_embeddings.csv', mode='a', header=False, index=False)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    LOAD_MODEL_TO_TRAIN = True
    BATCH_SIZE = 1
    VAL_BATCH_SIZE = 1
    EPOCHS = 50
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
    MAX_DECIBEL = 105
    HOP_LEN = 517  # width of spec = Total number of samples / hop_length
    NUM_WORKERS = 4
    VAL_STEP = 1
    CHECKPOINT_DATA_COUNT = 25_000

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

    train_dataloader = create_data_loader(ds, BATCH_SIZE, NUM_WORKERS, False)
    unet = load_unet_state("/home/dominik/Work/Lumen Datascience/LDS-Audio-Classification-2023-RiTehc/trainer/best-unet.pt")
    #encoder = load_encoder_state(".../Encoder")
    save_path = "/home/dominik/Work/Lumen Datascience/LDS-Audio-Classification-2023-RiTehc/MIXED_Training_Data/"
    generate_embeddings_unet(unet, device, train_dataloader, save_path)
    #generate_embeddings_encoder(encoder, device, train_dataloader, save_path)

