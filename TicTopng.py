import TicClassBgremover
import matplotlib.pyplot as plt # plt 用于显示图片
plt.switch_backend('tkagg')

print('start...')
br=TicClassBgremover.bgremover()
device='cpu'
model = br.load_model(device=device)
fimg = br.load_PILimg(file_img='./dataset/demo/lsj.jpg',size=1024)
print('ai ......')
y=br.ai(fimg, model, device=device)
print('mask ......')
mask,_=br.ai_mask(y) 
print('ai_removing ...')
cutout=br.ai_removing(fimg, mask)
print('showing ...')

#结果展示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码

 #子图
plt.subplot(111)
plt.imshow(cutout)
plt.title('抠图')
plt.axis('off')
 

# #设置子图默认的间距
plt.tight_layout()
#显示图像
plt.show()
print('ok')