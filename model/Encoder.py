from torch import nn


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, padding):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, stride=1, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, stride=1, padding=padding),
            nn.LeakyReLU(inplace=True)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(245 * 50 * 1024, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, encoded_space_dim)
        )

        # ### Convolutional section
        # self.encoder_cnn = nn.Sequential(
        #     nn.Conv2d(1, 8, 3, stride=2, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 16, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(True),
        #     nn.Conv2d(16, 32, 3, stride=2, padding=0),
        #     nn.ReLU(True)
        # )
        #
        # ### Flatten layer
        # self.flatten = nn.Flatten(start_dim=1)
        # ### Linear section
        # self.encoder_lin = nn.Sequential(
        #     nn.Linear(3 * 3 * 32, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, encoded_space_dim)
        # )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
