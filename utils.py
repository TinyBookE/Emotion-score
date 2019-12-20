import os

def log(file, str):
    with open(file, 'a+') as f:
        f.write(str)

def delDuplicatedFromTestDir(train_dir, test_dir):
    test_imgs = os.listdir(test_dir)
    for img in test_imgs:
        file = os.path.join(train_dir, img)
        if os.path.exists(file):
            os.remove(os.path.join(test_dir, img))

def getFileName(file):
    file_path, full_name = os.path.split(file)
    name, ext = os.path.splitext(full_name)
    return name

delDuplicatedFromTestDir('./data/train/img_files', './data/test/img_files')