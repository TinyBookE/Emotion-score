import torch
import torch.nn as nn
from torch.nn.modules.flatten import Flatten
from torchvision import models
import os
import numpy as np

resnet = models.resnet50()
print(resnet)