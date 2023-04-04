from torch import nn


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder_b1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
        )

        self.encoder_max_pool = nn.MaxPool2d(2)

        self.encoder_normalise = nn.BatchNorm2d(64)

        self.encoder_b2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
        )

        self.encoder_b3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.encoder_b1(x)
        x = self.encoder_max_pool(x)

        x = self.encoder_normalise(x)
        x = self.encoder_b2(x)
        x = self.encoder_max_pool(x)

        x = self.encoder_b3(x)

        return x
