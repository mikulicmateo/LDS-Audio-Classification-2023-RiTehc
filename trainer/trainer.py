import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import numpy as np
import sys
from tqdm import tqdm
import os

sys.path.insert(0, '../data/')

from data.MIXEDDataset import MIXEDDataset
from data.WINDOWEDValidationDataset import WINDOWEDValidationDataset
from model.Encoder import Encoder
from model.Decoder import Decoder


def create_data_loader(train_data, batch_size, num_workers):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    return train_dataloader


def create_model_state_dict(model, train_loss, val_loss, optimizer, epoch):
    model_state = {
        'time': str(datetime.datetime.now()),
        'model_state': model.state_dict(),
        'model_name': type(encoder).__name__,
        'optimizer_state': optimizer.state_dict(),
        'optimizer_name': type(optimizer).__name__,
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss
    }

    return model_state


def save_models(encoder, decoder, train_loss, val_loss, optimizer, epoch, best):
    encoder_state = create_model_state_dict(encoder, train_loss, val_loss, optimizer, epoch)
    decoder_state = create_model_state_dict(decoder, train_loss, val_loss, optimizer, epoch)

    torch.save(encoder_state, 'last-encoder.pt')
    torch.save(decoder_state, 'last-decoder.pt')

    if best:
        torch.save(encoder_state, 'best-encoder.pt')
        torch.save(decoder_state, 'best-decoder.pt')


def val_epoch(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():

        out = []
        label = []
        for image_batch, _ in tqdm(dataloader, leave=False):
            for windowed_batch in image_batch:
                windowed_batch = windowed_batch.to(device)

                encoded_data = encoder(windowed_batch)
                decoded_data = decoder(encoded_data)

                out.append(decoded_data.cpu())
                label.append(windowed_batch.cpu())

        out = torch.cat(out)
        label = torch.cat(label)
        # Evaluate global loss
        val_loss = loss_fn(out, label)

    return val_loss


def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    train_loss = []
    loop = tqdm(dataloader, leave=False)
    for image_batch, _ in loop:
        image_batch = image_batch.to(device)

        encoded_data = encoder(image_batch)
        decoded_data = decoder(encoded_data)

        loss = loss_fn(decoded_data, image_batch)
        train_loss.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

    return np.mean(train_loss)


def train(encoder, decoder, train_loader, val_loader, loss_fn, optimizer, device, epochs, val_step):
    print('Going training!')
    best_val_loss = float('inf')
    best_epoch = 1
    for epoch in range(1, epochs + 1):

        print(f"Epoch {epoch}")

        train_loss = train_epoch(encoder, decoder, device, train_loader, loss_fn, optimizer)
        print(f"Train Loss: {train_loss}")

        if epoch == 1 or epoch % val_step == 0:
            val_loss = val_epoch(encoder, decoder, device, val_loader, loss_fn)
            print(f"Validation Loss: {val_loss}")
            if epoch == 1 or val_loss < best_val_loss:
                save_models(encoder, decoder, train_loss, val_loss, optimizer, epoch, best=True)
                best_val_loss = val_loss
                best_epoch = epoch

        if epoch - best_epoch > val_step * 3:
            print(f"Early stopping at epoch: {epoch}")
            break
        print("---------------------------")

    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    BATCH_SIZE = 32
    VAL_BATCH_SIZE = 1
    EPOCHS = 50
    ABSOLUTE_PATH_DATA_FOLDER = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/MIXED_Training_Data'
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

    ABSOLUTE_PATH_VAL_DATA_FOLDER = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/WINDOWED_Validation_Data'
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

    train_data_loader = create_data_loader(ds, BATCH_SIZE, NUM_WORKERS)
    val_data_loader = create_data_loader(vds, VAL_BATCH_SIZE, NUM_WORKERS)
    encoder = Encoder()
    decoder = Decoder()
    encoder.to(device)
    decoder.to(device)
    loss_fn = torch.nn.MSELoss()
    lr = 0.0001

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=1e-05)
    train(encoder, decoder, train_data_loader, val_data_loader, loss_fn, optim, device, EPOCHS, VAL_STEP)
