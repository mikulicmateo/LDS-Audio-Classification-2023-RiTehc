from torch import nn

class Autoencoder(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.decoder.forward(x)
        return x