import torch
from torch import nn
import torch.nn.functional as F

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.decoder_b1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256,kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
        )

        self.reverse_pool1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
        )

        self.decoder_b2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128,kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64,kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
        )

        self.reverse_pool2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
        )

        self.decoder_b3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32,kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1,kernel_size=3, stride=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.decoder_b1(x)
        x = self.reverse_pool1(x)

        x = self.decoder_b2(x)
        x = self.reverse_pool2(x)

        x = self.decoder_b3(x)
        return x
