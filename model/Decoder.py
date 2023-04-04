import torch
from torch import nn
import torch.nn.functional as F

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.decoder_b1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
        )

        self.decoder_max_unpool = nn.MaxUnpool2d(2)

        self.decoder_normalise = nn.BatchNorm2d(256)

        self.decoder_b2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, indices_first, indices_second):
        x = F.max_unpool2d(x, indices_second, 2, output_size=[26, 59])
        x = self.decoder_normalise(x)
        x = self.decoder_b1(x)
        x = F.max_unpool2d(x, indices_first, 2)
        x = self.decoder_b2(x)
        return x
