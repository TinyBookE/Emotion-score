import torch
from torch.utils.data import DataLoader
from Dataset import CustomData
from Model import loadModel
from torch.autograd import Variable
import numpy as np
import os

def test_VGG(emotion):
    csv_file = os.path.join('./data/test', '%s.csv'%emotion)
    custom_data = CustomData(csv_file, './data/test/img_files')
    dataset = DataLoader(custom_data, shuffle=False)
    model, _, _ = loadModel(emotion, './checkpoints/VGG19')
    with torch.no_grad():
        losses = []
        for data in dataset:
            result, loss = model(Variable(data['img']).cuda(), Variable(data['label']).float().cuda())
            losses.append(loss.cpu().numpy())
        test_result = emotion + ': %f'%np.average(losses)
        print(test_result)
        log('./results/test_loss.txt', test_result)
        log('./results/test_loss.txt', '----------------')


def log(file, str):
    with open(file, 'a+') as f:
        f.write(str)


test_list = ['beautiful', 'boring', 'depressing', 'lively', 'safety', 'wealthy']

for emotion in test_list:
    test_VGG(emotion)
