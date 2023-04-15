import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from data.WINDOWEDValidationDataset import WINDOWEDValidationDataset

import os
import sys
sys.path.insert(0, '../model/')

from trainer import trainer


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

                for i in range(len(decoded_data)):
                    plt.imshow(windowed_batch[i][0].cpu().detach().numpy())
                    plt.title("IN")
                    plt.show()
                    plt.imshow(encoded_data[i][0].cpu().detach().numpy())
                    plt.title("EMBEDDING 1")
                    plt.show()
                    plt.imshow(encoded_data[i][1].cpu().detach().numpy())
                    plt.title("EMBEDDING 2")
                    plt.show()
                    plt.imshow(decoded_data[i][0].cpu().detach().numpy())
                    plt.title("OUT")
                    plt.show()

                out.append(decoded_data.cpu())
                label.append(windowed_batch.cpu())
            break

        out = torch.cat(out)
        label = torch.cat(label)
        #Evaluate global loss
        val_loss = loss_fn(out, label)

    return val_loss

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
    CHECKPOINT_DATA_COUNT = 25_000


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

    test_dataset = torch.utils.data.Subset(vds, range(800))
    val_dataset = torch.utils.data.Subset(vds, range(800, len(vds)))

    val_data_loader = trainer.create_data_loader(val_dataset, VAL_BATCH_SIZE, NUM_WORKERS, shuffle=False)
    test_data_loader = trainer.create_data_loader(test_dataset, VAL_BATCH_SIZE, NUM_WORKERS, shuffle=False)

    loss_fn = torch.nn.MSELoss()
    lr = 0.0001
    trained_epochs = 0

    encoder, decoder, trained_epochs, optim = trainer.load_model_for_further_training(
        "/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/trainer/best-encoder.pt",
        "/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/trainer/best-decoder.pt",
        lr,
        device)

    #val_epoch(encoder, decoder, device, val_data_loader, loss_fn)
    val_epoch(encoder, decoder, device, test_data_loader, loss_fn)