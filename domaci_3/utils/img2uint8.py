import numpy as np
from domaci_3.utils.img2double import img2double

def img2uint8(img):
    img_double = img2double(img)
    img_uint8 = img_double * 255.0
    return img_uint8.astype(np.uint8)
