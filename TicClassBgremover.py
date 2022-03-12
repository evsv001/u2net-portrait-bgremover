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

# import matplotlib.pyplot as plt # plt 用于显示图片
# plt.switch_backend('tkagg')

class bgremover:

    def convert(self, img, target_type_min, target_type_max, target_type):
        imin = img.min()
        imax = img.max()

        a = (target_type_max - target_type_min) / (imax - imin)
        b = target_type_max - a * imax
        new_img = (a * img + b).astype(target_type)
        return new_img

    def square_pad(self, image, fill=255):  # 图像变成方块，短边padding,fill为填充颜色
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, fill, 'constant')

# transforms接受的类型是PIL Image类型的，如果要将numpy转换成PIL格式，那么要求数据类型是dtype=uint8, range[0, 255] 
# 同时传入的numpy数组的类型为(H, W, C),即(高，宽，通道)，如果C=1，就不用写了
# pil_image = transforms.ToPILImage()(x)

    def get_transform(self,):
        transforms = []
        # transforms.append(Resize(440)) # TBD: keep aspect ratio
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[.5, .5, .5],
                                    std=[.5, .5, .5]))
        return Compose(transforms)

    def load_model(self, file_model='./checkpoints/checkpoint.pth',device='cuda'):
        checkpoint = torch.load(file_model, map_location=device)
        model = U2NET_full().to(device=device)

        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        print(model)
    # 统计1
            
        # from torchstat import stat
        # stat(model, (3, 128, 128))

    # 统计2
        # model是我们在pytorch定义的神经网络层
        # model.parameters()取出这个model所有的权重参数
        para = sum([np.prod(list(p.size())) for p in model.parameters()])
        # 下面的type_size是4，因为我们的参数是float32也就是4B，4个字节
        print('\tModel {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
        
        return model

    def load_img(self, file_img='./dataset/demo/2.jpg', file_bg='./dataset/demo/bj.jpg',size=1024):
        PILimage = Image.open(file_img).convert('RGB')
        PILbg = Image.open(file_bg).convert('RGB')

        PILimage = self.square_pad(PILimage, 0)
        PILimage = PILimage.resize((size, size), Image.ANTIALIAS)
        PILbg = self.square_pad(PILbg, 0)
        PILbg = PILbg.resize((size, size), Image.ANTIALIAS)

        return PILimage, PILbg

    def showPil2CV2(self, img, winname='1'):
        cv2.imshow(winname, cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def ai(self,PILimage, model, device='cpu'):

 
                # 2 ai工作------------------------------------------------
        transforms = self.get_transform()
        model.eval()
        with torch.no_grad():   #pytorch 需要 PIL
            x = transforms(PILimage)
            x = x.to(device=device).unsqueeze(dim=0)
            y_hat, _ = model(x)
        # del y_hat
        free_up_memory()
        return y_hat
        
    # 将PILimage图片产生其人物mask
    def ai_mask(self, y_hat):
        alphaNumpy = y_hat.squeeze().cpu().detach().numpy()
        # .cpu()函数作用是将数据从GPU上复制到memory上，相对应的函数是cuda()
        # .detach() 返回一个new Tensor，只不过不再有梯度。
        
        # alphas = (alphaNumpy * 255).astype(np.uint8)
        # cv2.imwrite('alphas.jpg', alphas)
        # cv2.imshow('4 alpha', alphas)
        # cv2.waitKey(1)
      
        del y_hat
        
        alphaPil = Image.fromarray((alphaNumpy*255).astype('uint8')).convert('RGB')#
        # alphaPil.save('alphaPil.jpg')
        return alphaNumpy,alphaPil
    
    def ai_blend(self, PILimage, BgPILimg,alpha):
    
        foreground = estimate_foreground_ml(    # 本模型需要float32的numpy array
            np.asarray(PILimage)/255.0, alpha)  # , n_big_iterations=1, n_small_iterations=1, regularization=10e-10

        bgArrayfloat32 = np.asarray(BgPILimg).astype(np.float32) / 255

        new_image = blend(foreground, bgArrayfloat32, alpha)

        # cv2.imshow('5 foreground', cv2.cvtColor(
        #     foreground.astype(np.float32), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)

        # new_image = new_image.astype(np.float32)
        # new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        # new_image = self.convert(new_image, 0, 255, np.uint8)

        # cv2.imwrite('new_image.jpg', new_image)
        # cv2.imshow('6', new_image)
        # cv2.waitKey(3000)
        
        foregroundPil=Image.fromarray((foreground*255).astype('uint8')).convert('RGB')# 
        blendPil=Image.fromarray((new_image*255).astype('uint8')).convert('RGB')# 
       
        return foregroundPil, blendPil

    def ai_zhengjian(self, PILimage,alpha,color=[0,1,0]):
    
        foreground = estimate_foreground_ml(    # 本模型需要float32的numpy array
            np.asarray(PILimage)/255.0, alpha)  # , n_big_iterations=1, n_small_iterations=1, regularization=10e-10
        background = np.zeros(foreground.shape)
        background[:, :] = color
        new_image = blend(foreground, background, alpha)
        zhenjiangPil=Image.fromarray((new_image*255).astype('uint8')).convert('RGB')# 
       
        return zhenjiangPil
    
    def ai_removing(self, PILimage, alpha): 
        image = np.asarray(PILimage)/255.0
        trimap = alpha

        if image.shape[:2] != trimap.shape[:2]:
            raise ValueError("Input image and trimap must have same size")

        alpha = estimate_alpha_cf(image, trimap)
        foreground = estimate_foreground_ml(image, alpha)
        cutout = stack_images(foreground, alpha)

        save_image(r'ooo.png', cutout)
        return cutout
# 1 开始设置------------------------------------------------

# br=bgremover()
# print('start...')
# model = br.load_model()
# fimg, bimg = br.load_img(file_img='./dataset/demo/3.jpg')
# # br.showPil2CV2( fimg,'1')
# # br.showPil2CV2(bimg,'2')
# print('ai ......')
# y=br.ai(fimg, model, device='cpu')
# print('mask ......')
# mask,_=br.ai_mask(y) 
# print('ai blending ...')
# fPil,blendPil=br.ai_blend(fimg, bimg,mask)
# print('ai_removing ...')

# cutout=br.ai_removing(fimg, mask)
# print('showing ...')

# #结果展示
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码

# plt.subplot(231)
# plt.imshow(fimg)
# plt.title('原图')
# #不显示坐标轴
# plt.axis('off')

# plt.subplot(232)
# plt.imshow(bimg)
# plt.title('背景')
# #不显示坐标轴
# plt.axis('off')

# plt.subplot(233)
# plt.imshow(fPil)
# plt.title('前景')
# #不显示坐标轴
# plt.axis('off')
 
# #子图2
# plt.subplot(234)
# plt.imshow(blendPil)
# plt.title('合并')
# plt.axis('off')

#  #子图2
# plt.subplot(235)
# plt.imshow(cutout)
# plt.title('抠图')
# plt.axis('off')
 
# # #设置子图默认的间距
# plt.tight_layout()
# #显示图像
# plt.show()
# print('ok')