import os
from glob import glob
import numpy as np

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import torchvision.transforms.functional as F
from pymatting import *

from lib import U2NET_full
from lib.utils.oom import free_up_memory
import cv2


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def load_samples(folder_path='./dataset/demo'):
    assert os.path.isdir(folder_path), f'Unable to open {folder_path}'
    samples = glob(os.path.join(folder_path, f'*.jpg'))
    return samples

def square_pad(image, fill=255):# 图像变成方块，短边padding,fill为填充颜色
    w, h = image.size
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp)
    return F.pad(image, padding, fill, 'constant')

def get_transform():
    transforms = []
    # transforms.append(Resize(440)) # TBD: keep aspect ratio
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[.5,.5,.5],
                                  std=[.5,.5,.5]))
    return Compose(transforms)
# 1 开始设置------------------------------------------------
print('start...')
device = 'cuda'
samples = load_samples()
model_select='checkpoint.pth'
sample_select = (samples[0])

device = 'cpu'
checkpoint = torch.load(f'./checkpoints/{model_select}', map_location=device)
model = U2NET_full().to(device=device)

if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(checkpoint)

image = Image.open(sample_select).convert('RGB')
bg = Image.open('./dataset/demo/bj.jpg').convert('RGB')

image = square_pad(image, 0)
image = image.resize((600, 600), Image.ANTIALIAS)
bg = square_pad(bg, 0)
bg = bg.resize((600, 600), Image.ANTIALIAS)

cv2.imshow('1 origin',cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
cv2.waitKey(1)
cv2.imshow('2 bg',cv2.cvtColor(np.asarray(bg), cv2.COLOR_RGB2BGR))
cv2.waitKey(1)



ddd = convert(cv2.cvtColor(np.asarray(bg), cv2.COLOR_RGB2BGR), 0, 255, np.uint8)


cv2.imwrite('ddd.jpg',ddd)

# from torchstat import stat
# import torchvision.models as models
# # model = models.resnet152()
# stat(model, (7, 3, 32, 64))



transforms = get_transform()

# 2 ai工作------------------------------------------------
model.eval()
with torch.no_grad():
    x = transforms(image)
    x = x.to(device=device).unsqueeze(dim=0)
    y_hat, _ = model(x)

    # alpha_image = y_hat.mul(255)
    # alpha_image = Image.fromarray(alpha_image.squeeze().cpu().detach().numpy()).convert('L')    # RGB转灰度
    # alpha_image.show()
    # 得到掩码图片alpha_image

image =  np.asarray(image)


# ##--V
# opencvImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# opencvMask = np.asarray(alpha_image)

# imagecv = cv2.add(opencvImage, np.zeros(np.shape(opencvImage), dtype=np.uint8), mask=opencvMask)
# cv2.imshow('d2',imagecv)
# cv2.waitKey(1)
# ##--^

background = np.zeros(image.shape)
background[:, :] = [0, 177 / 255, 64 / 255]
# background[:, :] = [0 / 255, 0 / 255, 0 / 255]
cv2.imshow('3 background',background)
cv2.waitKey(1)

alpha = y_hat.squeeze().cpu().detach()
alpha = np.asarray(alpha)
# alpha = (alpha * 255).astype(np.uint8)
imagealpha = alpha.astype(np.float32) / 255      # 归一且numpy的图片


# lll = np.asarray(imagelll)
cv2.imshow('4 alpha',alpha)
cv2.waitKey(1)


foreground = estimate_foreground_ml(
    image/255.0, alpha) # , n_big_iterations=1, n_small_iterations=1, regularization=10e-10

bg =  np.asarray(bg)

bg = bg.astype(np.float32)/ 255 


new_image = blend(foreground, bg, alpha)
cv2.imshow('5 foreground',cv2.cvtColor(foreground.astype(np.float32), cv2.COLOR_RGB2BGR))
cv2.waitKey(1)

new_image = new_image.astype(np.float32)
new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)



new_image = convert(new_image, 0, 255, np.uint8)


cv2.imwrite('lll.jpg',new_image)
cv2.imshow('6',new_image)
cv2.waitKey(0)





# imgf = Image.fromarray(foreground, mode='RGB')
# imgf.save('dd.png')

# img = Image.fromarray(new_image, mode='RGBA')
# img.save('ddd.png')



del y_hat
free_up_memory()
