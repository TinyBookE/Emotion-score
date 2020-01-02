import os
from PIL import Image
import random

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


if __name__ == "__main__":
    rotate('../data/imgs_2_cropped', '../data/imgs_2', 270)