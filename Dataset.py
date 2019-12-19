from torch.utils.data import Dataset
import torchvision
import pandas as pd
import os
from PIL import Image


class CustomData(Dataset):
    def __init__(self, score_file, img_dir):
        df = pd.read_csv(score_file, index_col=False)
        self.img = []
        self.label = []
        for i in range(len(df)):
            img_name = df.iloc[i, 0]
            img_file = os.path.join(img_dir, img_name+'.png')
            if(os.path.exists(img_file)):
                self.img.append(img_file)
                self.label.append(df.iloc[i, 1])
        self.norm = torchvision.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((240,240)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img_file = self.img[item]
        label = self.label[item]
        img = Image.open(img_file)
        img = img.convert('RGB')
        img = self.transform(img).float()
        img = self.norm(img)
        return {'img': img, 'label': label}

