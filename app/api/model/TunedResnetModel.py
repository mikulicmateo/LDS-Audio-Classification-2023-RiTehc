from torch import nn


class TunedResnetModel(nn.Module):

    def __init__(self, pretrained_model, freeze=True, get_embedding=False):
        super().__init__()
        self.get_embedding = get_embedding
        self.model = pretrained_model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
        )

        self.last_layer = nn.Sequential(
            nn.Linear(in_features=256, out_features=11),
        )

    def forward(self, x):
        x = self.model(x)
        if self.get_embedding:
            return x

        x = self.last_layer(x)
        return x
