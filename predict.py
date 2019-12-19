import torch
from torch.utils.data import DataLoader
from Dataset import CustomData
from Model import Model
from torch.autograd import Variable
import numpy as np

cpt_file = './checkpoints/net_beautiful_latest.pth'

custom_data = CustomData('./data/test/beautiful.csv', './data/test/img_files')
dataset = DataLoader(custom_data, shuffle=False)
data_size = len(custom_data)

with torch.no_grad():
    model = Model().cuda()
    model.load_state_dict(torch.load(cpt_file))
    losses = []
    for data in dataset:
        result, loss = model(Variable(data['img']).cuda(), Variable(data['label']).float().cuda())
        losses.append(loss.cpu().numpy())
    print(np.average(losses))