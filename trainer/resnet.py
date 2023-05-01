import torch.optim
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import os
from data.MIXEDDatasetImages import MIXEDDatasetImages
from data.WINDOWEDValidationDatasetImages import WINDOWEDValidationDatasetImages
from tqdm import tqdm
import numpy as np
import datetime
from torch.utils.data import DataLoader


class ResnetModel(nn.Module):

    def __init__(self, pretrained_model):
        super().__init__()

        self.model = pretrained_model
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU()
        )

        self.last_layer = nn.Sequential(
            nn.Linear(in_features=512, out_features=11),
            # nn.Sigmoid()
        )

    def forward(self, x, get_embedding=False):
        x = model(x)
        if get_embedding:
            return x

        x = self.last_layer(x)
        return x


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

def create_model_state_dict(model, train_loss, val_loss, optimizer, epoch):
    model_state = {
        'time': str(datetime.datetime.now()),
        'model_state': model.state_dict(),
        'model_name': type(model).__name__,
        'optimizer_state': optimizer.state_dict(),
        'optimizer_name': type(optimizer).__name__,
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss
    }

    return model_state
def save_models(model, train_loss, val_loss, optimizer, epoch, best):
    model_state = create_model_state_dict(model, train_loss, val_loss, optimizer, epoch)

    torch.save(model_state, 'last-resnet.pt')

    if best:
        torch.save(model_state, 'best-resnet.pt')

def train_epoch(model, device, dataloader, loss_fn, optimizer):
    model.train()
    train_loss = []
    loop = tqdm(dataloader, leave=False)
    for image_batch, labels in loop:
        image_batch = image_batch.to(device)
        labels = labels.to(device)

        predicted_data = model(image_batch)

        loss = loss_fn(predicted_data, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def val_epoch(model, device, dataloader, loss_fn):
    model.eval()
    with torch.no_grad():

        out = []
        label = []
        for image_batch, labels in tqdm(dataloader, leave=False):
            labels = labels.to(device)
            for windowed_batch in image_batch:
                windowed_batch = windowed_batch.to(device)

                predicted_data = model(windowed_batch)

                # for i in range(len(decoded_data)):
                #     plt.imshow(windowed_batch[i][0].cpu().detach().numpy())
                #     plt.show()
                #     plt.imshow(encoded_data[i][0].cpu().detach().numpy())
                #     plt.show()
                #     print(encoded_data[i][0].cpu().detach().numpy())
                #     plt.imshow(decoded_data[i][0].cpu().detach().numpy())
                #     plt.show()

                out.append(predicted_data.cpu())
                label.append(labels.float().cpu())

        out = torch.cat(out)
        label = torch.cat(label)
        # Evaluate global loss
        val_loss = loss_fn(out, label)

    return val_loss


def train(model, train_loader, val_loader, loss_fn, optimizer, device, epochs, val_step, start_epoch=1):
    print('Going training!')
    best_val_loss = float('inf')
    val_epoch_start = 1
    best_epoch = val_epoch_start
    early_stop = False
    checkpoints_num = len(train_loader)

    for epoch in range(start_epoch, epochs + 1):

        print(f"Epoch {epoch}")

        for i in range(1, checkpoints_num + 1):
            train_loss = train_epoch(model, device, train_loader[i - 1], loss_fn, optimizer)
            print(f"Checkpoint {i} Training Loss: {train_loss}")

            if epoch >= val_epoch_start:
                if epoch == val_epoch_start or epoch % val_step == 0:
                    val_loss = 0
                    for k in range(len(val_loader)):
                        new_val_loss = val_epoch(model, device, val_loader[k], loss_fn)
                        if new_val_loss > val_loss:
                            val_loss = new_val_loss
                    print(f"Checkpoint {i} Validation Loss: {val_loss}")
                    if epoch == val_epoch_start or val_loss < best_val_loss:
                        save_models(model, train_loss, val_loss, optimizer, epoch, best=True)
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


if __name__ == '__main__':
    BATCH_SIZE = 32
    ABSOLUTE_PATH_DATA_FOLDER = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/MIXED_Training_Data'
    CHECKPOINT_DATA_COUNT = 25_000
    VAL_STEP = 1
    VAL_BATCH_SIZE = 1
    EPOCHS = 200
    NUM_WORKERS = 6


    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    my_resnet = ResnetModel(model)
    my_resnet.to(device)
    params_to_optimize = [
        {'params': my_resnet.parameters()}
    ]

    lr = 0.01
    optim = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=1e-05)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = MIXEDDatasetImages(
        ABSOLUTE_PATH_DATA_FOLDER
    )

    ABSOLUTE_PATH_VAL_DATA_FOLDER = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/WINDOWED_Validation_Data'
    FOLDER_FILE_MAPPING_PATH = os.path.join(ABSOLUTE_PATH_VAL_DATA_FOLDER, 'folder_file_mapping.csv')

    vds = WINDOWEDValidationDatasetImages(
        ABSOLUTE_PATH_VAL_DATA_FOLDER,
        FOLDER_FILE_MAPPING_PATH
    )

    trained_epochs=0

    test_dataset = torch.utils.data.Subset(vds, range(794))
    val_dataset = torch.utils.data.Subset(vds, range(794, len(vds)))

    train_dataset = create_random_dataset_with_checkpoints(CHECKPOINT_DATA_COUNT, ds)
    train_data_loader = create_dataloaders_for_subsetet_data(train_dataset, BATCH_SIZE, NUM_WORKERS, shuffle=True)

    val_dataset = create_random_dataset_with_checkpoints(520, val_dataset)
    val_data_loader = create_dataloaders_for_subsetet_data(val_dataset, VAL_BATCH_SIZE, NUM_WORKERS, shuffle=True)
    test_data_loader = create_data_loader(test_dataset, VAL_BATCH_SIZE, NUM_WORKERS, shuffle=True)

    train(my_resnet, train_data_loader, val_data_loader, loss_fn, optim, device, EPOCHS + trained_epochs,
          VAL_STEP, trained_epochs + 1)
