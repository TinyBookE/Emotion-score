from torch.utils.data import DataLoader
from Dataset import CustomData
from Model import VGG, loadModel, saveModel, ResNet
from torch.autograd import Variable
import datetime
import os
import numpy as np

batch_size = 8
all_epoch = 120
save_delta = 10
init_lr = 0.0002
lr_decay = 20
adjust = 100


def train_VGG(emotion, load = False):
    csv_file = os.path.join('./data/train', emotion + '.csv')
    custom_data = CustomData(csv_file, './data/train/img_files')
    dataset = DataLoader(custom_data, batch_size=batch_size, shuffle=True)

    if load:
        model, start_epoch, total_step = loadModel(emotion, './checkpoints/VGG19')
        print('continue to train. start from epoch %s'%start_epoch)
    else:
        total_step = 0
        start_epoch = 0
        model = VGG().cuda()
    print(list(model.children()))

    if start_epoch > lr_decay:
        lr = init_lr - (start_epoch - lr_decay) * init_lr / 100.0
    else:
        lr = init_lr

    for epoch in range(start_epoch, all_epoch):

        if epoch > lr_decay:
            lr_dec = init_lr/100.0
            lr = lr - lr_dec

        losses = []

        for data in dataset:
            total_step += 1
            result, loss = model(Variable(data['img']).cuda(), Variable(data['label']).float().cuda())
            if(epoch < adjust):
                optimizer = model.get_optimizer(lr)
            else:
                optimizer = model.get_optimizer(lr, True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())

        t = datetime.datetime.now()
        str = 'current time: {}\nepoch: {}\nloss: {}\n----------------\n'\
            .format(t, epoch, np.average(losses))
        print(str)
        log('checkpoints/VGG19/log_%s.txt'%emotion, str)
        losses.clear()

        if (epoch+1) % save_delta == 0:
            saveModel(model, emotion, epoch, total_step, './checkpoints/VGG19')


def train_ResNet(emotion, load = False):
    csv_file = os.path.join('./data/train', emotion + '.csv')
    custom_data = CustomData(csv_file, './data/train/img_files')
    dataset = DataLoader(custom_data, batch_size=batch_size, shuffle=True)

    if load:
        model, start_epoch, total_step = loadModel(emotion, './checkpoints/ResNet50')
        print('continue to train. start from epoch %s'%start_epoch)
    else:
        total_step = 0
        start_epoch = 0
        model = ResNet().cuda()
    print(list(model.children()))

    if start_epoch > lr_decay:
        lr = init_lr - (start_epoch - lr_decay) * init_lr / 100.0
    else:
        lr = init_lr

    for epoch in range(start_epoch, all_epoch):

        if epoch > lr_decay:
            lr_dec = init_lr/100.0
            lr = lr - lr_dec

        losses = []

        for data in dataset:
            total_step += 1
            result, loss = model(Variable(data['img']).cuda(), Variable(data['label']).float().cuda())
            if(epoch < adjust):
                optimizer = model.get_optimizer(lr)
            else:
                optimizer =model.get_optimizer(lr, True)
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
            saveModel(model, emotion, epoch, total_step, './checkpoints/ResNet50')


def log(file, str):
    with open(file, 'a+') as f:
        f.write(str)

train_list = ['beautiful', 'boring', 'depressing', 'lively', 'safety', 'wealthy']

train_VGG('wealthy')
'''
for name in train_list:
    train_VGG(name)
'''