import torch
import torch.nn as nn
from torch.nn.modules.flatten import Flatten
from torchvision import models
import os
import numpy as np



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


save_dir = './checkpoints'


def loadModel(emotion):
    iter_file = os.path.join(save_dir, 'iter_%s.txt'%(emotion))
    model_file = os.path.join(save_dir, 'net_%s_latest.pth'%(emotion))
    txt = np.loadtxt(iter_file)
    start_epoch = int(txt[0])
    total_step = int(txt[1])
    model = Model().cuda()
    model.load_state_dict(torch.load(model_file))

    return model, start_epoch, total_step


def saveModel(model, emotion, epoch, total_step = 0):
    save_file = os.path.join(save_dir, 'net_%s_%s.pth' % (emotion, epoch + 1))
    torch.save(model.cpu().state_dict(), save_file)
    print('save model in {}'.format(save_file))
    save_file = os.path.join(save_dir, 'net_%s_latest.pth' % (emotion))
    torch.save(model.cpu().state_dict(), save_file)
    iter_path = os.path.join(save_dir, 'iter_%s.txt' % (emotion))
    np.savetxt(iter_path, (epoch + 1, total_step), fmt='%d')
    print('----------------')
    model.cuda()