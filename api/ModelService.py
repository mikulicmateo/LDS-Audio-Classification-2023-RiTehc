import matplotlib.pyplot as plt


class ModelService:
    def __init__(self):
        # load models
        # self.encoder = torch.load('models/Encoder.pt')
        # self.encoder.eval()

        self.class_names = ['tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']

    def classify_audio(self, spectrograms):
        final_predictions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for spectrogram in spectrograms:
            plt.imshow(spectrogram[0])
            plt.show()
            # encode
            # classify
            # sum labels
        # return classification
        predictions_dict = dict(zip(self.class_names, final_predictions))
        return predictions_dict
