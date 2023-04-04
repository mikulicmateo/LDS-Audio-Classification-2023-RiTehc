import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from data.IRMASValidationDataset import IRMASValidationDataset
from model.Encoder import Encoder
from model.Decoder import Decoder


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def get_model_for_eval():
    encoder = Encoder()
    decoder = Decoder()
    encoder.load_state_dict(torch.load("/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/trainer/encoderC-final.pt"))
    decoder.load_state_dict(torch.load("/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/trainer/decoderC-final.pt"))

    return encoder, decoder


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
            for i in range(len(decoded_data)):
                plt.imshow(image_batch[i][0].cpu().detach().numpy())
                plt.show()
                plt.imshow(encoded_data[i][0].cpu().detach().numpy())
                plt.show()
                print(encoded_data[i][0].cpu().detach().numpy())
                plt.imshow(decoded_data[i][0].cpu().detach().numpy())
                plt.show()

            out.append(decoded_data.cpu())
            label.append(image_batch.cpu())
            break

        out = torch.cat(out)
        label = torch.cat(label)
        # Evaluate global loss
        val_loss = loss_fn(out, label)

    return val_loss.data


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    ANNOTATIONS_FILE = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/IRMAS_Validation_Data/validation_annotation_file.csv'
    PROJECT_DIR = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc'

    vds = IRMASValidationDataset(
        ANNOTATIONS_FILE,
        PROJECT_DIR,
    )

    BATCH_SIZE = 100
    val_data_loader = create_data_loader(vds, BATCH_SIZE)
    loss_fn = torch.nn.MSELoss()
    encoder, decoder = get_model_for_eval()
    encoder.to(device)
    decoder.to(device)
    val_loss = val_epoch(encoder, decoder, device, val_data_loader, loss_fn)
    print(f'val loss {val_loss}')


