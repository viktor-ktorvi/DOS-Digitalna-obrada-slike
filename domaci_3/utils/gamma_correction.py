import numpy as np
from domaci_3.utils.img2double import img2double


def gamma_correction(img, gamma):
    img_double = img2double(img)
    img_double = img_double ** (1 / gamma)
    return np.clip(img_double, 0, 1)
