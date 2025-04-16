import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        latent_size = 4 * 28 * 28
        self.classifier = nn.Linear(latent_size, num_classes)

    def forward(self, x):
        x = self.classifier(x)
        return x
