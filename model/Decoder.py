import torch
from torch import nn


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, padding):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 245 * 50 * 1024),
            nn.LeakyReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(1024, 245, 50))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=padding),
            nn.LeakyReLU(True),
            nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(512, 256, 2, stride=2, padding=padding),
            nn.LeakyReLU(True),
            nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(256, 128, 2, stride=2, padding=padding),
            nn.LeakyReLU(True),
            nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(128, 64, 2, stride=2, padding=padding),
            nn.LeakyReLU(True),
            nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(64, 1, 2, stride=2, padding=padding),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
