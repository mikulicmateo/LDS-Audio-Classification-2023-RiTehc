import datetime
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '../data/')

from data.MIXEDDataset import MIXEDDataset
from data.WINDOWEDValidationDataset import WINDOWEDValidationDataset
from model.Encoder import Encoder
from model.Decoder import Decoder


def create_data_loader(train_data, batch_size, num_workers, shuffle):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return train_dataloader


def create_random_dataset_with_checkpoints(checkpoint_data_count, full_dataset):
    dataset = []
    for i in range((len(full_dataset) // checkpoint_data_count)):
        dataset.append(checkpoint_data_count)

    return torch.utils.data.random_split(full_dataset, dataset)


def create_dataloaders_for_subsetet_data(subsets, batch_size, num_workers, shuffle):
    data_loaders = []
    for subset in subsets:
        data_loaders.append(create_data_loader(subset, batch_size, num_workers, shuffle=shuffle))
    return data_loaders


def load_model_optimizer(model_path, lr):
    print("LOADING OPTIMIZER")
    encoder = Encoder()
    decoder = Decoder()
    model_dict = torch.load(model_path)

    optimizer_params = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optimizer = torch.optim.AdamW(optimizer_params, lr=lr, weight_decay=1e-05)
    optimizer.load_state_dict(model_dict['optimizer_state'])

    print(f"LOADED OPTIMIZER, train_loss {model_dict['train_loss']}"
          + f", val_loss {model_dict['val_loss']}")

    return optimizer


def load_model_state(encoder_path, decoder_path):
    print("LOADING MODEL")
    encoder = Encoder()
    decoder = Decoder()
    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)
    encoder.load_state_dict(encoder_dict['model_state'])
    decoder.load_state_dict(decoder_dict['model_state'])

    print(f"LOADED MODEL, epoch {encoder_dict['epoch']}"
          + f", time {encoder_dict['time']}")

    return encoder, decoder, encoder_dict['epoch']


def load_model_for_further_training(encoder_path, decoder_path, lr):
    encoder, decoder, trained_epochs = load_model_state(encoder_path, decoder_path)

    encoder.to(device)
    decoder.to(device)

    optimizer = load_model_optimizer(encoder_path, lr)

    return encoder, decoder, trained_epochs, optimizer


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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def train(encoder, decoder, train_loader, val_loader, loss_fn, optimizer, device, epochs, val_step, start_epoch=1):
    print('Going training!')
    best_val_loss = float('inf')
    val_epoch_start = 1
    best_epoch = val_epoch_start
    early_stop = False

    for epoch in range(start_epoch, epochs + 1):

        print(f"Epoch {epoch}")

        for i in range(1, 5):
            train_loss = train_epoch(encoder, decoder, device, train_loader[i - 1], loss_fn, optimizer)
            print(f"Checkpoint {i} Training Loss: {train_loss}")

            if epoch >= val_epoch_start:
                if epoch == val_epoch_start or epoch % val_step == 0:
                    val_loss = val_epoch(encoder, decoder, device, val_loader, loss_fn)
                    print(f"Checkpoint {i} Validation Loss: {val_loss}")
                    if epoch == val_epoch_start or val_loss < best_val_loss:
                        save_models(encoder, decoder, train_loss, val_loss, optimizer, epoch, best=True)
                        best_val_loss = val_loss
                        best_epoch = epoch

                if epoch - best_epoch > val_step * 3:
                    print(f"Early stopping at epoch {epoch}, checkpoint: {i}")
                    early_stop = True
                    break

        print("---------------------------")

        if early_stop:
            break

    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    LOAD_MODEL_TO_TRAIN = False
    BATCH_SIZE = 32
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
    test_dataset = torch.utils.data.Subset(vds, range(800))
    val_dataset = torch.utils.data.Subset(vds, range(800, len(vds)))

    train_dataset = create_random_dataset_with_checkpoints(CHECKPOINT_DATA_COUNT, ds)
    train_data_loader = create_dataloaders_for_subsetet_data(train_dataset, BATCH_SIZE, NUM_WORKERS, shuffle=True)

    val_data_loader = create_data_loader(val_dataset, VAL_BATCH_SIZE, NUM_WORKERS, shuffle=True)
    test_data_loader = create_data_loader(test_dataset, VAL_BATCH_SIZE, NUM_WORKERS, shuffle=True)

    loss_fn = torch.nn.MSELoss()
    lr = 0.0001
    trained_epochs = 0

    if LOAD_MODEL_TO_TRAIN:
        encoder, decoder, trained_epochs, optim = load_model_for_further_training(
            "/home/dominik/Work/Lumen Datascience/LDS-Audio-Classification-2023-RiTehc/trainer/best-encoder.pt",
            "/home/dominik/Work/Lumen Datascience/LDS-Audio-Classification-2023-RiTehc/trainer/best-decoder.pt",
            lr)
    else:
        encoder = Encoder()
        decoder = Decoder()
        encoder.to(device)
        decoder.to(device)

        params_to_optimize = [
            {'params': encoder.parameters()},
            {'params': decoder.parameters()}
        ]

        optim = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=1e-05)

    train(encoder, decoder, train_data_loader, val_data_loader, loss_fn, optim, device, EPOCHS + trained_epochs,
          VAL_STEP, trained_epochs + 1)
