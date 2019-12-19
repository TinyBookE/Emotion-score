import torch
from torch.utils.data import DataLoader
from Dataset import CustomData
from Model import Model, loadModel, saveModel
from torch.autograd import Variable
import datetime
import os
import numpy as np

batch_size = 8
all_epoch = 100
save_delta = 20
init_lr = 0.0002


def train(emotion, load = False):
    csv_file = os.path.join('./data/train', emotion + '.csv')
    custom_data = CustomData(csv_file, './data/train/img_files')
    dataset = DataLoader(custom_data, batch_size=batch_size, shuffle=True)

    if load:
        model, start_epoch, total_step = loadModel(emotion)
        print('continue to train. start from epoch %s'%start_epoch)
    else:
        total_step = 0
        start_epoch = 0
        model = Model().cuda()
    print(list(model.children()))

    lr = init_lr - (all_epoch - start_epoch) * init_lr/100.0
    for epoch in range(start_epoch, all_epoch):

        if epoch > 20:
            lr_dec = init_lr/100.0
            lr = lr - lr_dec

        losses = []

        for data in dataset:
            total_step += 1
            result, loss = model(Variable(data['img']).cuda(), Variable(data['label']).float().cuda())
            optimizer = model.get_optimizer(lr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())

        t = datetime.datetime.now()
        print('current time: ', t)
        print('epoch: ', epoch+1)
        print('loss: ', np.average(losses))
        print('----------------')
        losses.clear()

        if (epoch+1) % save_delta == 0:
            saveModel(model, emotion, epoch, total_step)


train('beautiful', True)
