import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import numpy as np
import sys
from tqdm import tqdm

sys.path.insert(0, '../data/')

from data.MIXEDDataset import MIXEDDataset
from data.IRMASValidationDataset import IRMASValidationDataset
from model.Encoder import Encoder
from model.Decoder import Decoder


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def val_epoch(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():

        out = []
        label = []
        for image_batch in tqdm(dataloader, leave=False):

            image_batch = image_batch.to(device)

            encoded_data, indices_first, indices_second = encoder(image_batch)
            decoded_data = decoder(encoded_data, indices_first, indices_second)

            out.append(decoded_data.cpu())
            #with open("validation_loss.txt", "w") as f:
            #    f.write(f"{decoded_data.cpu().numpy()}\n")
            label.append(image_batch.cpu())
            break

        #with open("validation_loss.txt") as f:
        #    for line in f.readlines():
        #       out.append(float(line.strip()))

        #print(label)
        out = torch.cat(out)
        label = torch.cat(label)
        # Evaluate global loss
        val_loss = loss_fn(out, label)

    return val_loss


def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    train_loss = []
    f = open("training_loss.txt", "w")

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

        # print('\t partial train loss (single batch): %f' % loss.data)
        f.write(f"{loss.detach().cpu().numpy()}\n")

    f.close()

    with open("training_loss.txt") as f:
        for line in f.readlines():
            train_loss.append(float(line.strip()))

    return np.mean(train_loss)


def train(encoder, decoder, train_loader, val_loader, loss_fn, optimizer, device, epochs):
    print('Going training!')
    losses = {'train_loss': [], 'val_loss': []}
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        print(f"Train Loss: {train_epoch(encoder, decoder, device, train_loader, loss_fn, optimizer)}")
        torch.save(encoder.state_dict(), f"encoder{i+1}.pt")
        torch.save(decoder.state_dict(), f"decoder{i+1}.pt")
        #losses['train_loss'].append(train_loss)
        #losses['val_loss'].append(val_loss)
        print("---------------------------")
    #print(f"Validation Loss: {val_epoch(encoder, decoder, device, val_loader, loss_fn)}")
    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    BATCH_SIZE = 100
    EPOCHS = 16
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

    ANNOTATIONS_FILE = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/IRMAS_Validation_Data/validation_annotation_file.csv'
    PROJECT_DIR = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc'

    vds = IRMASValidationDataset(
        ANNOTATIONS_FILE,
        PROJECT_DIR,
    )

    train_data_loader = create_data_loader(ds, BATCH_SIZE)
    val_data_loader = create_data_loader(vds, BATCH_SIZE)
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
    train(encoder, decoder, train_data_loader, val_data_loader, loss_fn, optim, device, EPOCHS)
