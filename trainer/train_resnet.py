import torch.optim
from torchvision.models import resnet34, ResNet34_Weights
from torch import nn
import os
from data.MIXEDDatasetImages import MIXEDDatasetImages
from data.WINDOWEDValidationDatasetImages import WINDOWEDValidationDatasetImages
import json
from torch.utils.data import DataLoader
from model.TunedResnetModel import TunedResnetModel
from ResnetTrainer import ResNetTrainer
from ResnetTrainer import load_resnet


def main():
    project_path = os.path.dirname(os.getcwd())
    config_file_path = os.path.join(project_path, "config.json")
    mixed_dataset_path = os.path.join(project_path, "MIXED_Uniform_Training_Data")
    windowed_validation_dataset_path = os.path.join(project_path, "WINDOWED_Validation_Data")
    folder_file_mapping_path = os.path.join(windowed_validation_dataset_path, 'folder_file_mapping.csv')
    resnet_path = os.path.join(project_path, "model/best-ResNet.pt")

    with open(config_file_path, "r") as config_file:
        config_dict = json.load(config_file)

    ds = MIXEDDatasetImages(
        mixed_dataset_path
    )

    vds = WINDOWEDValidationDatasetImages(
        windowed_validation_dataset_path,
        folder_file_mapping_path
    )

    trainer = ResNetTrainer(
        training_data=ds,
        validation_data=torch.utils.data.Subset(vds, range(794, len(vds))),
        test_data=torch.utils.data.Subset(vds, range(794)),
        training_batch_size=config_dict["TRAINING_BATCH_SIZE"],
        num_workers=config_dict["NUM_WORKERS"],
        shuffle=config_dict["SHUFFLE"],
        training_checkpoint_data_count=config_dict["TRAINING_CHECKPOINT_DATA_COUNT"],
        validation_checkpoint_data_count=config_dict["VALIDATION_SPLIT_DATA_COUNT"],
        loss_fn=nn.BCEWithLogitsLoss(),
        epochs_to_train=config_dict["EPOCHS_TO_TRAIN"]
    )

    if config_dict["TRAIN_HEAD_FIRST"]:
        resnet = TunedResnetModel(resnet34(weights=ResNet34_Weights.DEFAULT), freeze=True)
        optimizer = torch.optim.AdamW(resnet.parameters(), lr=config_dict["LEARNING_RATE"], weight_decay=1e-05)

        trainer.set_model(resnet, optimizer)

        trainer.train()

        if config_dict["TEST_MODEL"]:
            trainer.test_model()

    resnet = TunedResnetModel(resnet34(weights=ResNet34_Weights.DEFAULT), freeze=False)
    optimizer = torch.optim.AdamW(resnet.parameters(), lr=config_dict["LEARNING_RATE"], weight_decay=1e-05)

    resnet, optimizer, model_epoch = load_resnet(resnet_path, resnet, optimizer)

    trainer.set_model(resnet, optimizer)
    trainer.start_epoch = model_epoch

    trainer.train()

    if config_dict["TEST_MODEL"]:
        trainer.test_model()


if __name__ == '__main__':
    main()
