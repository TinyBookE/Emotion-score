import torch
from torch.utils.data import DataLoader
from Dataset import CustomData
from Model import Model
from torch.autograd import Variable
import datetime
import os
import numpy as np

batch_size = 8
all_epoch = 100
start_epoch = 0
save_delta = 20
total_step = 0
save_dir = './checkpoints'
init_lr = 0.0002

custom_data = CustomData('./data/train/beautiful.csv', './data/train/img_files')
dataset = DataLoader(custom_data, batch_size=batch_size, shuffle=True)
data_size = len(custom_data)

model = Model().cuda()
print(list(model.children()))

lr = init_lr
for epoch in range(start_epoch, all_epoch):

    if epoch > 20:
        lr_dec = init_lr/100.0
        lr = lr -lr_dec

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
        save_file = os.path.join(save_dir, 'net_beautiful_%s.pth'%(epoch+1))
        torch.save(model.cpu().state_dict(), save_file)
        print('save model in {}'.format(save_file))
        save_file = os.path.join(save_dir, 'net_beautiful_latest.pth')
        torch.save(model.cpu().state_dict(), save_file)
        iter_path = os.path.join(save_dir, 'iter_beautiful.txt')
        np.savetxt(iter_path, (epoch+1, total_step), fmt='%d')
        print('----------------')
        model.cuda()

