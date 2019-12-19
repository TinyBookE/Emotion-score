import torch
import torch.nn as nn
from torch.nn.modules.flatten import Flatten
from torchvision import models

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features
        net = [nn.AdaptiveAvgPool2d(output_size=(7,7)),
                    Flatten(),
                    nn.Linear(25088, 4096),
                    nn.ReLU(True), nn.Dropout(0.5, False),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True), nn.Dropout(0.5, False),
                    nn.Linear(4096, 1)]
        self.net = nn.Sequential(*net)
        self.loss = nn.MSELoss()

    def get_optimizer(self, lr=0.0002):
        return torch.optim.Adam(list(self.net.parameters()), lr=lr)

    def forward(self, x, y = None):
        features = self.vgg19(x)
        out = self.net(features)
        out = torch.flatten(out)
        if y is not None:
            loss = self.loss(out, y)
            return out, loss
        return out, 0


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout=False):
        super(ResnetBlock, self).__init__()
        block = [nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                 nn.BatchNorm2d(dim), nn.ReLU(True)]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                  nn.BatchNorm2d(dim), nn.ReLU(True)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)
