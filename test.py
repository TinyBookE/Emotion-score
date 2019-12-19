import torchvision.models
from PIL import Image
from skimage import io

vgg = torchvision.models.vgg19()

print(vgg)
img = Image.open('data/train/img_files/113.814407_30.744719_0_0.png')
print(img)
img = img.convert('RGB')
print(img)