import torch.nn as nn


__all__ = ['convnet']


class ConvNet(nn.Module):

    def __init__(self, num_classes=10, **kwargs):
        
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Flatten(), 
            nn.Linear(9*9*128, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
        )

    def forward(self, x, bfgs=False):
        x = self.features(x)
        return x


def convnet(**kwargs):
    model = ConvNet(**kwargs)
    return model

