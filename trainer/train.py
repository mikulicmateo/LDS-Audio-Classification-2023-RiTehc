import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import numpy as np
import sys

sys.path.insert(0, '../data/')

from data.MIXEDDataset import MIXEDDataset
from model.Encoder import Encoder
from model.Decoder import Decoder


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    train_loss = []

    for image_batch, _ in dataloader:
        image_batch = image_batch.to(device)

        encoded_data = encoder(image_batch)
        decoded_data = decoder(encoded_data)

        loss = loss_fn(decoded_data, image_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('\t partial train loss (single batch): %f' % loss.data)
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def train(encoder, decoder, data_loader, loss_fn, optimizer, device, epochs):
    print('Going training!')
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        mean = train_epoch(encoder, decoder, device, data_loader, loss_fn, optimizer)
        print(f'BATCH MEAN = {mean}')
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    BATCH_SIZE = 5
    EPOCHS = 1
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
    HOP_LEN = None  # width of spec = Total number of samples / hop_length

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

    train_data_loader = create_data_loader(ds, BATCH_SIZE)
    encoder = Encoder(4, 0)
    decoder = Decoder(4, 0)
    encoder.to(device)
    decoder.to(device)
    loss_fn = torch.nn.MSELoss()
    lr = 0.001

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)
    train(encoder,decoder,train_data_loader, loss_fn, optim, device, EPOCHS)
    torch.save(encoder.state_dict(), "/home/dominik/Work/Lumen Datascience/LDS-Audio-Classification-2023-RiTehc/model/encoder.pt")
    torch.save(decoder.state_dict(), "/home/dominik/Work/Lumen Datascience/LDS-Audio-Classification-2023-RiTehc/model/decoder.pt")