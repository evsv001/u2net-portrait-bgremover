import TicClassBgremover
import matplotlib.pyplot as plt # plt 用于显示图片
plt.switch_backend('tkagg')

print('start...')
br=TicClassBgremover.bgremover()
device='cuda'
model = br.load_model(device=device)
fimg, bimg = br.load_img(file_img='./dataset/demo/sxt.jpg',file_bg='./dataset/demo/bj.jpg',size=256)
# br.showPil2CV2( fimg,'1')
# br.showPil2CV2(bimg,'2')
print('ai ......')
y=br.ai(fimg, model, device=device)
print('mask ......')
mask,_=br.ai_mask(y) 
print('ai blending ...')
fPil,blendPil=br.ai_blend(fimg, bimg,mask)
print('ai zhenjian ...')
zhengjianPil=br.ai_zhengjian(fimg, mask,color=[1,0,0])
print('ai_removing ...')

cutout=br.ai_removing(fimg, mask)
print('showing ...')

#结果展示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码

plt.subplot(231)
plt.imshow(fimg)
plt.title('原图')
#不显示坐标轴
plt.axis('off')

plt.subplot(232)
plt.imshow(bimg)
plt.title('背景')
#不显示坐标轴
plt.axis('off')


plt.subplot(233)
plt.imshow(fPil)
plt.title('前景')
#不显示坐标轴
plt.axis('off')
 
#子图2
plt.subplot(234)
plt.imshow(blendPil)
plt.title('合并')
plt.axis('off')

 #子图2
plt.subplot(235)
plt.imshow(cutout)
plt.title('抠图')
plt.axis('off')
 
 #子图2
plt.subplot(236)
plt.imshow(zhengjianPil)
plt.title('证件')
plt.axis('off')
# #设置子图默认的间距
plt.tight_layout()
#显示图像
plt.show()
print('ok')