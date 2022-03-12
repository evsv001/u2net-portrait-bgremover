# import cv2
# import  numpy as np
# import PIL
# from PIL import Image

# def blend_two_images():
#     img1 = Image.open( "bridge.png ")
#     img1 = img1.convert('RGBA')
 
#     img2 = Image.open( "birds.png ")
#     img2 = img2.convert('RGBA')
    
#     img = Image.blend(img1, img2, 0.3)
#     img.show()
#     img.save( "blend.png")
 
#     return

# def trans2non(a):
#     b = a.convert('RGBA')
#     L, H = b.size
#     color_0 = b.getpixel((0,0))
#     for h in range(H):
#         for l in range(L):
#             dot = (l,h)
#             color_1 = b.getpixel(dot)
#             # if color_1 == color_0:
#             color_1 = color_1[:-1] + (color_1[0],)
#             b.putpixel(dot,color_1)
#     return b 



# new_mask = Image.open("alphas.jpg")

# new_mask = trans2non(new_mask)
# new_mask.save('new_mask.png')

import cv2
from PIL import Image
import numpy as np
 
 
class UnsupportedFormat(Exception):
 def __init__(self, input_type):
  self.t = input_type
 
 def __str__(self):
  return "不支持'{}'模式的转换，请使用为图片地址(path)、PIL.Image(pil)或OpenCV(cv2)模式".format(self.t)
 
 
class MatteMatting():
 def __init__(self, original_graph, mask_graph, input_type='path'):
  """
  将输入的图片经过蒙版转化为透明图构造函数
  :param original_graph:输入的图片地址、PIL格式、CV2格式
  :param mask_graph:蒙版的图片地址、PIL格式、CV2格式
  :param input_type:输入的类型，有path：图片地址、pil：pil类型、cv2类型
  """
  if input_type == 'path':
   self.img1 = cv2.imread(original_graph)
   self.img2 = cv2.imread(mask_graph)
  elif input_type == 'pil':
   self.img1 = self.__image_to_opencv(original_graph)
   self.img2 = self.__image_to_opencv(mask_graph)
  elif input_type == 'cv2':
   self.img1 = original_graph
   self.img2 = mask_graph
  else:
   raise UnsupportedFormat(input_type)
 
 @staticmethod
 def __transparent_back(img):
  """
  :param img: 传入图片地址
  :return: 返回替换白色后的透明图
  """
  img = img.convert('RGBA')
  L, H = img.size
  color_0 = (255, 255, 255, 255) # 要替换的颜色
  for h in range(H):
   for l in range(L):
    dot = (l, h)
    color_1 = img.getpixel(dot)
    if color_1 == color_0:
     color_1 = color_1[:-1] + (0,)
     img.putpixel(dot, color_1)
  return img
 
 def save_image(self, path, mask_flip=False):
  """
  用于保存透明图
  :param path: 保存位置
  :param mask_flip: 蒙版翻转，将蒙版的黑白颜色翻转;True翻转;False不使用翻转
  """
  if mask_flip:
   img2 = cv2.bitwise_not(self.img2) # 黑白翻转
  image = cv2.add(self.img1, img2)
  image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # OpenCV转换成PIL.Image格式
  img = self.__transparent_back(image)
  img.save(path)
 
 @staticmethod
 def __image_to_opencv(image):
  """
  PIL.Image转换成OpenCV格式
  """
  img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
  return img

mm = MatteMatting("new_image.jpg", "new_mask.png")
mm.save_image("output.png", mask_flip=True) # mask_flip是指蒙版翻转，即把白色的变成黑色的，黑色的变成白色的


 
