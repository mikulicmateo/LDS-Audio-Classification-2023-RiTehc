from torch import nn


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, padding):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=padding),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, return_indices=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=padding),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2),
            # nn.Conv2d(128, 256, 3, stride=1, padding=padding),
            # nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2),
            # nn.Conv2d(256, 512, 3, stride=1, padding=padding),
            # nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2),
            # nn.Conv2d(512, 1024, 3, stride=1, padding=padding),
            # nn.LeakyReLU(inplace=True)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(128 * 60 * 126, 128),#128 * 60 * 255, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        print('out of enccnn',x.shape)
        x = self.flatten(x)
        print('after flatten', x.shape)
        x = self.encoder_lin(x)
        print('after enclin', x.shape)
        return x
