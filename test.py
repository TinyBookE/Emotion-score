import torch
from torch.utils.data import DataLoader
from Dataset import CustomData
from Model import loadModel
from torch.autograd import Variable
import numpy as np
import os
from utils.utils import log

def test_VGG(emotion):
    csv_file = os.path.join('./data/test', '%s.csv'%emotion)
    custom_data = CustomData('./data/test/img_files', csv_file,isTrain=False)
    dataset = DataLoader(custom_data, batch_size=1, shuffle=False)
    model, _, _ = loadModel(emotion, model_type= 'VGG', save_dir= './checkpoints/VGG19', isTrain=False)
    with torch.no_grad():
        losses = []
        for data in dataset:
            result, loss = model(Variable(data['img']).cuda(), Variable(data['label']).float().cuda())
            loss = loss.cpu().numpy()
            losses.extend(loss)

        losses = np.array(losses)
        losses = np.sqrt(losses)

        test_result = emotion + ': %f' % (np.sum(losses) / len(losses))
        print(len(losses))
        print(test_result)
        log('./results/test_loss.txt', test_result+'\n')

def test_ResNet(emotion):
    csv_file = os.path.join('./data/test', '%s.csv' % emotion)
    custom_data = CustomData('./data/test/img_files', csv_file, isTrain=False)
    dataset = DataLoader(custom_data, batch_size=1, shuffle=False)
    model, _, _ = loadModel(emotion, model_type= 'ResNet', save_dir= './checkpoints/ResNet50', isTrain=False)
    with torch.no_grad():
        losses = []
        for data in dataset:
            result, loss = model(Variable(data['img']).cuda(), Variable(data['label']).float().cuda())
            loss = loss.cpu().numpy()
            losses.extend(loss)

        losses = np.array(losses)
        losses = np.sqrt(losses)

        test_result = emotion + ': %f' % (np.sum(losses)/len(losses))
        print(len(losses))
        print(test_result)
        log('./results/test_loss.txt', test_result + '\n')


test_list = ['beautiful', 'boring', 'depressing', 'lively', 'safety', 'wealthy']


log('./results/test_loss.txt', '------VGG-------\n')
for emotion in test_list:
    test_VGG(emotion)

log('./results/test_loss.txt', '-----ResNet-----\n')
for emotion in test_list:
    test_ResNet(emotion)

