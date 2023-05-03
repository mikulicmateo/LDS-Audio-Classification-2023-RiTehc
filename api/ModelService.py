import pandas as pd
import numpy as np
from model.AnnoyANN import AnnoyANN
from torchvision.models import resnet34, ResNet34_Weights
import torchvision
import torch
from model.TunedResnetModel import TunedResnetModel
from PIL import Image


class ModelService:
    def __init__(self, vector_size=256):
        self.vector_size = vector_size
        self.labels = self.fc_training_embedding_labels('data/labels.pickle')
        self.tree = AnnoyANN(self.vector_size, self.labels, path='model/angular-resnet34-256-t30.ann', metric='angular',
                             prefault=True)
        pretrained = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.model = self.load_tuned_resnet_state('model/best-resnet-34.pt', pretrained)
        self.model.eval()
        self.transform = torchvision.transforms.ToTensor()
        self.class_names = ['tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']

    def classify_audio(self, spectrograms):
        embeddings = self.get_embeddings(spectrograms)
        final_predictions = self.get_prediction(embeddings)
        predictions_dict = dict(zip(self.class_names, final_predictions))
        return predictions_dict

    def get_embeddings(self, spectrograms):
        embeddings = []
        for spectrogram in spectrograms:
            spectrogram = Image.open(spectrogram).convert('RGB')
            spectrogram = self.transform(spectrogram).unsqueeze(0)
            embedding = self.model(spectrogram).detach().numpy()[0]
            embeddings.append(embedding)
        return embeddings

    def fc_training_embedding_labels(self, data_path):
        data = pd.read_pickle(data_path)
        labels = data[['label']].to_numpy()
        labels = np.array([labels[i][0] for i in range(len(labels))])

        return labels

    def get_prediction(self, vectors, k=10, threshold_p=0.5):
        sample_prediction = [0 for _ in self.class_names]
        window_results = []
        for j, window in enumerate(vectors):
            window_prediction = [0 for _ in self.class_names]
            query_results, query_distances = self.tree.query(window, k=k)

            query_results = np.array(query_results)
            weights = self.calculate_weights(np.array(query_distances))
            weighted_results = [1.0 * query_result for weight, query_result in zip(weights, query_results)]

            for result in weighted_results:
                window_prediction = np.add(window_prediction, result)

            window_prediction /= k
            window_results.append(window_prediction)

        for result in window_results:
            sample_prediction = np.add(sample_prediction, result)

        sample_prediction /= len(vectors)
        sample_prediction = [1 if x > threshold_p else 0 for x in sample_prediction]
        return sample_prediction

    def calculate_weights(self, dists):
        dists[dists == 0.0] = 0.00001  # division by 0 handle for floats
        dists[dists == 0] = 1  # division by 0 handle for integers
        weights = 1.0 / dists
        return weights / np.sum(weights)

    def load_tuned_resnet_state(self, model_path, pretrained_model):
        model = TunedResnetModel(pretrained_model, freeze=False, get_embedding=True)
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict['model_state'])
        model.to('cpu')
        return model
