import datetime
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_resnet(model_path, model, optimizer):
    model_dict = torch.load(model_path)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.load_state_dict(model_dict['model_state'])
    model.to(device)
    model_epoch = model_dict['epoch']
    optimizer.load_state_dict(model_dict['optimizer_state'])

    return model, optimizer, model_epoch


class ResNetTrainer:

    def __init__(self, training_data, validation_data, test_data, training_batch_size, num_workers, shuffle,
                 training_checkpoint_data_count, validation_checkpoint_data_count, loss_fn,
                 epochs_to_train, optimizer=None, resnet=None, start_epoch=1):

        self.training_dataloader = self.create_dataloaders_for_subset_data(
            self.create_random_dataset_with_checkpoints(training_checkpoint_data_count, training_data),
            training_batch_size,
            num_workers,
            shuffle
        )

        self.validation_dataloader = self.create_dataloaders_for_subset_data(
            self.create_random_dataset_with_checkpoints(validation_checkpoint_data_count, validation_data),
            1,
            num_workers,
            shuffle
        )

        self.test_dataloader = self.create_data_loader(test_data, 1, num_workers, shuffle)

        self.loss_fn = loss_fn
        self.epochs_to_train = epochs_to_train
        self.start_epoch = start_epoch

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.device = device
        print(f"Using {device}")

        if resnet is None:
            self.resnet = resnet
        else:
            self.resnet = resnet.to(device)

        self.optimizer = optimizer

    def set_resnet(self, resnet):
        self.resnet = resnet.to(self.device)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_model(self, resnet, optimizer):
        self.set_resnet(resnet)
        self.set_optimizer(optimizer)

    def create_data_loader(self, data, batch_size, num_workers, shuffle):
        dataloader = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        return dataloader

    def create_random_dataset_with_checkpoints(self, checkpoint_data_count, full_dataset):
        dataset = []
        for i in range((len(full_dataset) // checkpoint_data_count)):
            dataset.append(checkpoint_data_count)

        return torch.utils.data.random_split(full_dataset, dataset)

    def create_dataloaders_for_subset_data(self, subsets, batch_size, num_workers, shuffle):
        data_loaders = []
        for subset in subsets:
            data_loaders.append(self.create_data_loader(subset, batch_size, num_workers, shuffle=shuffle))
        return data_loaders

    def create_model_state_dict(self, epoch, train_loss, val_loss):
        model_state = {
            'time': str(datetime.datetime.now()),
            'model_state': self.resnet.state_dict(),
            'model_name': type(self.resnet).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        }

        return model_state

    def save_model(self, train_loss, val_loss, epoch, best):
        model_state = self.create_model_state_dict(epoch, train_loss, val_loss)
        save_path = os.path.join(os.path.dirname(os.getcwd()), "model")

        torch.save(model_state, os.path.join(save_path, "last-ResNet.pt"))
        if best:
            torch.save(model_state, os.path.join(save_path, "best-ResNet.pt"))

    def train_epoch(self, dataloader_index):
        self.resnet.train()
        train_loss = []
        loop = tqdm(self.training_dataloader[dataloader_index], leave=False)
        for image_batch, labels in loop:
            image_batch = image_batch.to(self.device)
            labels = labels.to(self.device)

            predicted_data = self.resnet(image_batch)

            loss = self.loss_fn(predicted_data, labels.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loop.set_postfix(loss=loss.item())
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    def validate_epoch(self, dataloader_index):
        self.resnet.eval()

        with torch.no_grad():
            val_losses = []
            loop = tqdm(self.validation_dataloader[dataloader_index], leave=False)
            for image_batch, labels in loop:
                labels = labels.to(self.device)
                for windowed_batch in image_batch:
                    windowed_batch = windowed_batch.to(self.device)

                    predicted_data = self.resnet(windowed_batch)
                    loss = self.loss_fn(predicted_data, labels.float())
                    val_losses.append(loss.detach().cpu().numpy())
                    loop.set_postfix(loss=loss.item())

            # Evaluate global loss
        return np.mean(val_losses)

    def test_model(self):
        self.resnet.eval()

        with torch.no_grad():
            val_losses = []
            loop = tqdm(self.test_dataloader, leave=False)
            for image_batch, labels in loop:
                labels = labels.to(self.device)
                for windowed_batch in image_batch:
                    windowed_batch = windowed_batch.to(self.device)

                    predicted_data = self.resnet(windowed_batch)
                    loss = self.loss_fn(predicted_data, labels.float())
                    val_losses.append(loss.detach().cpu().numpy())
                    loop.set_postfix(loss=loss.item())

            # Evaluate global loss
        return np.mean(val_losses)

    def train(self, val_step=1):
        if self.resnet is None or self.optimizer is None:
            print("Resnet and Optimizer not initialized")
            return

        print('Going training!')
        best_val_loss = float('inf')
        best_epoch = 0
        early_stop = False
        checkpoints_num = len(self.training_dataloader)

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs_to_train):

            print(f"Epoch {epoch}:")

            for i in range(1, checkpoints_num + 1):
                train_loss = self.train_epoch(i-1)
                print(f"Checkpoint {i} Training Loss: {train_loss}")

                if epoch % val_step == 0:
                    val_loss = 0

                    for k in range(len(self.validation_dataloader)):
                        new_val_loss = self.validate_epoch(k)
                        if new_val_loss > val_loss:
                            val_loss = new_val_loss

                    print(f"Checkpoint {i} Validation Loss: {val_loss}")

                    self.save_model(train_loss, val_loss, epoch, best=False)
                    if val_loss < best_val_loss:
                        self.save_model(train_loss, val_loss, epoch, best=True)
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
