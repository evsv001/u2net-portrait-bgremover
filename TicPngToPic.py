"""将一个透明图像粘贴到另一幅图像"""
from PIL import Image

overlay = Image.open('dataset/demo/zy.png')
overlay = overlay.rotate(10) #旋转180度

base = Image.open('out.png')

bands = list(overlay.split())

if len(bands) == 4:

# Assuming alpha is the last band

    bands[3] = bands[3].point(lambda x: x*0.99)

    overlay = Image.merge(overlay.mode, bands)
    # overlay = overlay.rotate(10) #旋转180度

    base.paste(overlay, (0, 0), overlay)
    base.paste(overlay, (1024, 600), overlay)
    base.save('result.png')
    # base.open()
