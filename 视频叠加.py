import cv2 as cv
from scipy.fftpack import fft
import TicClassBgremover
import numpy as np
from PIL import Image

br=TicClassBgremover.bgremover()
device='cuda'

print('start...')
size=256
model = br.load_model(device=device)

def multy(path1, path2):
    cap1 = cv.VideoCapture(path1)
    cap2 = cv.VideoCapture(path2)
    output = 'output.mp4'
    height = int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT))
    weight = int(cap1.get(cv.CAP_PROP_FRAME_WIDTH))
    fps = int(cap1.get(cv.CAP_PROP_FPS))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 设置格式
    videowriter = cv.VideoWriter(output, fourcc, fps, (weight, height))
    print(height, weight, fps)
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1 is True and ret2 is True :
            # img = cv.cvtColor(np.asarray(frame), cv.COLOR_RGB2BGR)

            PILbg = Image.fromarray(cv.cvtColor(frame1,cv.COLOR_BGR2RGB))
            # PILbg = PILbg.resize((size, size), Image.ANTIALIAS)
            
            PILimage = Image.fromarray(cv.cvtColor(frame2,cv.COLOR_BGR2RGB))
            # PILimage = PILimage.resize((size, size), Image.ANTIALIAS)

            y=br.ai(PILimage, model, device=device)
            pp,p=br.ai_mask(y)
                
            fPil,blendPil=br.ai_blend(PILimage, PILbg,pp)
            imgp = cv.cvtColor(np.asarray(blendPil), cv.COLOR_RGB2BGR)
            videowriter.write(imgp)
            cv.imshow("capture", imgp) #显示
            if cv.waitKey(30) & 0xff == ord('q'): #按q退出
                break
        else:
            break

    videowriter.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    path1 = 'zy.mp4'
    path2 = "tic.mp4"
    multy(path1, path2)

