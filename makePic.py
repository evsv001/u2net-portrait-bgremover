# coding: utf-8
# !/usr/bin/python
"""
@File       :   颜色空间转换.py
@Author     :   jiaming
@Modify Time:   2020/1/31 11:58    
@Contact    :   https://blog.csdn.net/weixin_39541632
@Version    :   1.0
@Desciption :   你将学习如何对图像进行颜色空间转换，比如从 BGR 到灰度图，或者从 BGR 到 HSV
                创建一个程序用来从一幅图片中获取某个特定颜色的物体
                cv2.cvtColor()、cv2.inRange()
"""
from importlib.util import decode_source
import os
import sys
import numpy as np
import cv2
import pprint
from matplotlib import pyplot as plt

rawPath = os.path.abspath(__file__)
currentFile = os.path.basename(sys.argv[0])
dataPath = rawPath[:rawPath.find(currentFile)] + r'static\\'


img_bg = cv2.imread('ddd.jpg', cv2.IMREAD_UNCHANGED)

# img_bg = cv2.resize(img_bg, h , w, interpolation=cv2.INTER_LINEAR)

# 获取每一帧
frame = cv2.imread('lll.jpg', cv2.IMREAD_UNCHANGED)


# 转换到 HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 设定蓝色的阈值
lower_blue = np.array([30, 100, 100])
upper_blue = np.array([80, 255, 255])

# 根据阈值构建掩模
mask = cv2.inRange(hsv, lower_blue, upper_blue)


res_bg = cv2.bitwise_and(img_bg, img_bg, mask=mask)
mask = cv2.bitwise_not(mask)

# 对原图像和掩模进行位运算

res = cv2.bitwise_and(frame, frame, mask=mask)

res = cv2.add(res_bg, res)

# 显示图像
cv2.imshow('frame', frame)
cv2.imshow('mask_bg', mask)
cv2.imshow('res', res)
cv2.imshow('res_bg', res_bg)
k = cv2.waitKey(0)
if k == 27:
    cv2.imwrite('bbb.jpg', frame)



"""
虽然还会有噪音，但是这是最简单的追踪物体的方法了。
"""
