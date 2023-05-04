import json
import os

import pandas as pd
import torch
from torchvision.models import resnet34, ResNet34_Weights
from tqdm import tqdm

from data.MIXEDDatasetImages import MIXEDDatasetImages
from data.WINDOWEDValidationDatasetImages import WINDOWEDValidationDatasetImages
from trainer.ResnetTrainer import load_resnet, create_data_loader


def generate_embeddings_resnet(model, device, dataloader, save_path):
    model.eval()
    os.chdir(save_path)
    loop = tqdm(dataloader, leave=False)

    data = []
    for i, (image_batch, label) in enumerate(loop):
        image_batch = image_batch.to(device)
        embedding = model(image_batch)
        label = label.cpu().detach().numpy()[0]

        temp = [embedding.cpu().detach().numpy()[0], [int(l) for l in label]]
        data.append(temp)

    df = pd.DataFrame(data, columns=["embedding", "label"])
    df.to_pickle('embeddings-512-resnet34.pickle')


def generate_windowed_embeddings_resnet(model, device, dataloader, save_path):
    model.eval()
    os.chdir(save_path)

    loop = tqdm(dataloader, leave=False)

    data = []
    for image_batch, label in loop:
        embeddings = []
        label = label.cpu().detach().numpy()[0]
        for windowed_batch in image_batch:
            windowed_batch = windowed_batch.to(device)
            embedding = model(windowed_batch)

            embeddings.append(embedding.cpu().detach().numpy()[0])

        data.append([embeddings, [int(l) for l in label]])

    df = pd.DataFrame(data, columns=['embeddings', 'labels'])
    df.to_pickle('embeddings-windowed-512-resnet34.pickle')


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    project_path = os.path.dirname(os.getcwd())
    config_file_path = os.path.join(project_path, "config.json")
    mixed_dataset_path = os.path.join(project_path, "MIXED_Training_Data")
    save_path = os.path.join(project_path, "data")
    windowed_validation_dataset_path = os.path.join(project_path, "WINDOWED_Validation_Data")
    folder_file_mapping_path = os.path.join(windowed_validation_dataset_path, 'folder_file_mapping.csv')
    resnet_path = os.path.join(project_path, "model/best-ResNet.pt")

    ds = MIXEDDatasetImages(
        mixed_dataset_path
    )

    vds = WINDOWEDValidationDatasetImages(
        windowed_validation_dataset_path,
        folder_file_mapping_path
    )

    with open(config_file_path, "r") as config_file:
        config_dict = json.load(config_file)

    train_dataloader = create_data_loader(ds, 1, config_dict["NUM_WORKERS"], False)
    validation_dataloader = create_data_loader(vds, 1, config_dict["NUM_WORKERS"], False)
    pretrained = resnet34(weights=ResNet34_Weights.DEFAULT)
    resnet, _, _ = load_resnet(resnet_path, pretrained, None)

    generate_embeddings_resnet(resnet, device, train_dataloader, save_path)
    generate_windowed_embeddings_resnet(resnet, device, validation_dataloader, save_path)


if __name__ == "__main__":
    main()
