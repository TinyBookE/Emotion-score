import torch
import torch.nn as nn
from torch.nn.modules.flatten import Flatten
from torchvision import models
import os
import numpy as np

class VGG(nn.Module):
    def __init__(self, isTrain = True):
        super(VGG, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features
        net = [nn.Conv2d(512, 512, 3, padding=1),
               nn.Conv2d(512, 512, 1),
               nn.AdaptiveAvgPool2d(output_size=(7,7)),
               Flatten(),
               nn.Linear(25088, 4096),
               nn.ReLU(True), nn.Dropout(0.5, False),
               nn.Linear(4096, 4096),
               nn.ReLU(True), nn.Dropout(0.5, False),
               nn.Linear(4096, 2048),
               nn.ReLU(True), nn.Dropout(0.5, False),
               nn.Linear(2048, 1)]
        self.net = nn.Sequential(*net)
        if isTrain:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.MSELoss(reduction='none')

    def get_optimizer(self, lr=0.0002, adjust = False):
        if adjust:
            parms = list(self.vgg19.parameters()) + list(self.net.parameters())
        else:
            parms = list(self.net.parameters())
        return torch.optim.Adam(parms, lr=lr)

    def forward(self, x, y = None):
        features = self.vgg19(x)
        out = self.net(features)
        out = torch.flatten(out)
        if y is not None:
            loss = self.loss(out, y)
            return out, loss
        return out, 0


class ResNet(nn.Module):
    def __init__(self, isTrain = True):
        super(ResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        block = [nn.AdaptiveAvgPool2d(output_size=(2,2)),
                 Flatten(),
                 nn.Linear(8192, 4096),
                 nn.ReLU(True), nn.Dropout(0.5, False),
                 nn.Linear(4096, 2048),
                 nn.ReLU(True), nn.Dropout(0.5, False),
                 nn.Linear(2048, 1)]
        self.net = nn.Sequential(*block)
        if isTrain:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.MSELoss(reduction='none')

    def forward(self, x, y = None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.net(x)
        out = torch.flatten(out)
        if y is not None:
            loss = self.loss(out, y)
            return out, loss
        return out, 0

    def get_optimizer(self, lr=0.0002, adjust = False):
        if adjust:
            params = []
            params += self.conv1.parameters()
            params += self.bn1.parameters()
            params += self.relu.parameters()
            params += self.maxpool.parameters()
            params += self.layer1.parameters()
            params += self.layer2.parameters()
            params += self.layer3.parameters()
            params += self.layer4.parameters()
            params += self.net.parameters()
        else:
            params = self.net.parameters()
        return torch.optim.Adam(params, lr=lr)


default_save_dir = './checkpoints'


def loadModel(emotion, model_type = 'VGG', save_dir = None, isTrain = True):
    if save_dir is None:
        save_dir = default_save_dir

    iter_file = os.path.join(save_dir, 'iter_%s.txt'%(emotion))
    model_file = os.path.join(save_dir, 'net_%s_latest.pth'%(emotion))
    txt = np.loadtxt(iter_file)
    start_epoch = int(txt[0])
    total_step = int(txt[1])
    if model_type == 'VGG':
        model = VGG(isTrain).cuda()
    elif model_type == 'ResNet':
        model = ResNet(isTrain).cuda()
    else:
        raise('Unknown model type')
    model.load_state_dict(torch.load(model_file))

    return model, start_epoch, total_step


def saveModel(model, emotion, epoch, total_step = 0, save_dir = None):
    if save_dir is None:
        save_dir = default_save_dir

    save_file = os.path.join(save_dir, 'net_%s_%s.pth' % (emotion, epoch + 1))
    torch.save(model.cpu().state_dict(), save_file)
    print('save model in {}'.format(save_file))
    save_file = os.path.join(save_dir, 'net_%s_latest.pth' % (emotion))
    torch.save(model.cpu().state_dict(), save_file)
    iter_path = os.path.join(save_dir, 'iter_%s.txt' % (emotion))
    np.savetxt(iter_path, (epoch + 1, total_step), fmt='%d')
    print('----------------')
    model.cuda()

if __name__ == "__main__":
    print(models.resnet50())