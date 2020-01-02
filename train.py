from torch.utils.data import DataLoader
import torch
from Dataset import CustomData
from Model import VGG, loadModel, saveModel, ResNet
from torch.autograd import Variable
import datetime
import os
import numpy as np
from utils.utils import log

batch_size = 16
all_epoch = 180
save_delta = 20
init_lr = 0.0002
lr_decay = 80
adjust = 100


def train_VGG(emotion, load = False):
    csv_file = os.path.join('./data/train', emotion + '.csv')
    train_data = CustomData('./data/train/img_files', csv_file)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = CustomData('./data/test/img_files', csv_file, isTrain=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size)

    if load:
        model, start_epoch, total_step = loadModel(emotion, model_type= 'VGG', save_dir= './checkpoints/VGG19')
        print('continue to train. start from epoch %s'%start_epoch)
    else:
        total_step = 0
        start_epoch = 0
        model = VGG().cuda()
    print(list(model.children()))

    # if continue train, learn rate decay
    if start_epoch > lr_decay:
        lr = init_lr - (start_epoch - lr_decay) * init_lr / 100.0
    else:
        lr = init_lr

    for epoch in range(start_epoch, all_epoch):

        # learn rate decay with the epoch increasing
        if epoch > lr_decay and epoch - lr_decay < 100:
            lr_dec = init_lr/100.0
            lr = lr - lr_dec

        train_losses = []
        test_losses = []

        for data in train_dataset:
            total_step += 1
            result, loss = model(Variable(data['img']).cuda(), Variable(data['label']).float().cuda())
            if(epoch < adjust):
                optimizer = model.get_optimizer(lr)
            else:
                optimizer = model.get_optimizer(lr, True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().numpy())

        with torch.no_grad():
            for data in test_dataset:
                result, loss = model(Variable(data['img']).cuda(), Variable(data['label']).float().cuda())
                test_losses.append(loss.cpu().numpy())

        t = datetime.datetime.now()
        str = 'current time: {}\nepoch: {}\ntrain_loss: {}\ntest_loss: {}\n----------------\n'\
            .format(t, epoch, np.average(train_losses), np.average(test_losses))
        print(str)
        log('checkpoints/VGG19/log_%s.txt'%emotion, str)

        train_losses.clear()
        test_losses.clear()

        if (epoch+1) % save_delta == 0:
            saveModel(model, emotion, epoch, total_step, './checkpoints/VGG19')


def train_ResNet(emotion, load = False):
    csv_file = os.path.join('./data/train', emotion + '.csv')
    train_data = CustomData('./data/train/img_files', csv_file)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = CustomData('./data/test/img_files', csv_file, isTrain=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size)

    if load:
        model, start_epoch, total_step = loadModel(emotion, model_type= 'ResNet', save_dir= './checkpoints/ResNet50')
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

        train_losses = []
        test_losses = []
        for data in train_dataset:
            total_step += 1
            result, loss = model(Variable(data['img']).cuda(), Variable(data['label']).float().cuda())
            if(epoch < adjust):
                optimizer = model.get_optimizer(lr)
            else:
                optimizer = model.get_optimizer(lr, True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().numpy())

        with torch.no_grad():
            for data in test_dataset:
                result, loss = model(Variable(data['img']).cuda(), Variable(data['label']).float().cuda())
                test_losses.append(loss.cpu().numpy())

        t = datetime.datetime.now()
        str = 'current time: {}\nepoch: {}\ntrain_loss: {}\ntest_loss: {}\n----------------\n' \
            .format(t, epoch, np.average(train_losses), np.average(test_losses))
        print(str)
        log('checkpoints/ResNet50/log_%s.txt' % emotion, str)

        train_losses.clear()
        test_losses.clear()

        if (epoch+1) % save_delta == 0:
            saveModel(model, emotion, epoch, total_step, save_dir= './checkpoints/ResNet50')


train_list = ['boring', 'depressing', 'lively', 'safety', 'wealthy']


for name in train_list:
    train_ResNet(name)
