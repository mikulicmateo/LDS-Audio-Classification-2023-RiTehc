import torch
from torch import nn


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, padding):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 128 * 60 * 126),
            nn.LeakyReLU(True)
        )

        self.decoder_conv = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(128, 60, 126)),
            # nn.ConvTranspose2d(1024, 512, 3, stride=1, padding=padding),
            # nn.LeakyReLU(True),
            # nn.MaxUnpool2d(2),
            # nn.ConvTranspose2d(512, 256, 3, stride=1, padding=padding),
            # nn.LeakyReLU(True),
            # nn.MaxUnpool2d(2),
            # nn.ConvTranspose2d(256, 128, 3, stride=1, padding=padding),
            # nn.LeakyReLU(True),
            # nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=padding),
            nn.LeakyReLU(True),
            # nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(64, 1, 3, stride=1, padding=padding),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        #print('first ', x.shape)
        x = self.decoder_lin(x)
        #print('out of declin', x.shape)
        #x = self.unflatten(x)
        #print('out of unf', x.shape)
        x = self.decoder_conv(x)
        #print('out of deccnn', x.shape)
        x = torch.sigmoid(x)
        #print('out of sigm', x.shape)
        return x
