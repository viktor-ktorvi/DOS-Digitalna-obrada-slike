import numpy as np


def img2double(img):
    return img / np.amax(img)
