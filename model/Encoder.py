from torch import nn


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder_b1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.encoder_b2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 2, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2),
        )

        self.flatten = nn.Flatten()
        #
        self.encoder_fc = nn.Sequential(
            nn.Linear(in_features=2*16*32, out_features=2*16*32),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.encoder_b1(x)
        # print(x.shape)
        x = self.encoder_b2(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.encoder_fc(x)
        return x
