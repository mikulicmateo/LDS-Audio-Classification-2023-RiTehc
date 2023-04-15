from torch import nn


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.decoder_b11 = nn.Sequential(
            nn.ConvTranspose2d(2, 32, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2)
        )

        self.decoder_b12 = nn.Sequential(
            nn.ConvTranspose2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.decoder_b13 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
        )

        self.decoder_b14 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.decoder_b15 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
        )

        self.decoder_b16 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.decoder_b11(x)
        # print(x.shape)
        x = self.decoder_b12(x)
        # print(x.shape)
        x = self.decoder_b13(x)
        # print(x.shape)
        x = self.decoder_b14(x)
        # print(x.shape)
        x = self.decoder_b15(x)
        # print(x.shape)
        x = self.decoder_b16(x)
        # print(x.shape)

        return x
