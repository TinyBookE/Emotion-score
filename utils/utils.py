import os
from PIL import Image
import random
from matplotlib import pyplot as plt
import numpy as np

def log(file, str):
    with open(file, 'a+') as f:
        f.write(str)

def delDuplicatedFromTestDir(train_dir, test_dir):
    test_imgs = os.listdir(test_dir)
    for img in test_imgs:
        file = os.path.join(train_dir, img)
        if os.path.exists(file):
            os.remove(os.path.join(test_dir, img))

def crop(from_dir, to_dir, left, upper, right, lower):
    imgs = os.listdir(from_dir)
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)
    for img in imgs:
        load_path = os.path.join(from_dir, img)
        save_path = os.path.join(to_dir, img)
        file = Image.open(load_path)
        cropped = file.crop((left, upper, right, lower))
        cropped.save(save_path)

def getFileName(file):
    file_path, full_name = os.path.split(file)
    name, ext = os.path.splitext(full_name)
    return name

def randomPick(m, n):
    idx = random.sample(range(m), n)
    return idx

def rotate(from_dir, to_dir, angle):
    imgs = os.listdir(from_dir)
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)
    for img in imgs:
        load_path = os.path.join(from_dir, img)
        save_path = os.path.join(to_dir, img)
        file = Image.open(load_path)
        rotated = file.rotate(angle)
        rotated.save(save_path)


def showLoss(input_file, to_file, emotion):
    input_file = input_file.format(emotion)
    to_file = to_file.format(emotion)
    train_loss = []
    test_loss = []
    count = 0
    with open(input_file) as f:
        try:
            while f.readable():
                check = f.readline()
                if check == "":
                    break
                f.readline()
                train = f.readline().split(' ')
                train_loss.append(float(train[1]))
                test = f.readline().split(' ')
                test_loss.append(float(test[1]))
                f.readline()
                count += 1
        except Exception as e:
            print(e)
        finally:
            x = np.linspace(0, count - 1, count)
            plt.title(emotion)
            plt.plot(x, train_loss, 'r-', label='train_loss')
            plt.plot(x, test_loss, 'b-', label='test_loss')
            plt.legend(loc='best')
            plt.xlabel('epoch')
            plt.ylabel('MSELoss')
            plt.savefig(to_file)
            plt.show()
            plt.close('all')

if __name__ == "__main__":
    '''
    input_path = '../checkpoints/ResNet50/log_{}.txt'
    output_path = '../results/loss/log_{}.png'
    emotion_list = ['beautiful', 'boring', 'depressing', 'lively', 'safety', 'wealthy']
    for e in emotion_list:
        showLoss(input_path, output_path, e)
    '''
    # rotate('../data/imgs', '../data/imgs_rotated', 270)
