from scipy.fftpack import fft
import TicClassBgremover
import cv2
import numpy as np
from PIL import Image

br=TicClassBgremover.bgremover()
device='cuda'

print('start...')
size=256
model = br.load_model(device=device)
fimg, bimg = br.load_img(file_img='./dataset/demo/3.jpg',size=size)
print('mask ......')
PILbg = br.square_pad(bimg, 0)
PILbg = PILbg.resize((size, size), Image.ANTIALIAS)
cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read() #读取
    # img = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
    ff=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    PILimage = br.square_pad(ff, 0)
    PILimage = PILimage.resize((size, size), Image.ANTIALIAS)
    
    y=br.ai(PILimage, model, device=device)
    pp,p=br.ai_mask(y)
        
    fPil,blendPil=br.ai_blend(PILimage, PILbg,pp)
    imgp = cv2.cvtColor(np.asarray(blendPil), cv2.COLOR_RGB2BGR)

    cv2.imshow("capture", imgp) #显示
    if cv2.waitKey(10) & 0xff == ord('q'): #按q退出
        break
    
    
