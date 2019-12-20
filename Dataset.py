from torch.utils.data import Dataset
import torchvision
import pandas as pd
import os
from PIL import Image
from utils import getFileName

class CustomData(Dataset):
    def __init__(self, img_dir, score_file = None):
        self.img = []
        self.label = []

        if score_file is not None:
            df = pd.read_csv(score_file, index_col=False)
            for i in range(len(df)):
                img_name = df.iloc[i, 0]
                img_file = os.path.join(img_dir, img_name+'.png')
                if os.path.exists(img_file):
                    self.img.append(img_file)
                    self.label.append(df.iloc[i, 1])
        else:
            imgs = os.listdir(img_dir)
            for img in imgs:
                file = os.path.join(img_dir, img)
                self.img.append(file)

        self.norm = torchvision.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        img_file = self.img[item]
        name = getFileName(img_file)
        img = Image.open(img_file)
        img = img.convert('RGB')
        img = self.transform(img).float()
        img = self.norm(img)
        if len(self.label) > 0:
            label = self.label[item]
        else:
            label = 0
        return {'img': img, 'label': label, 'name': name}


