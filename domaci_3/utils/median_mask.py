import numpy as np
from domaci_3.utils.img2uint8 import img2uint8
from skimage import filters
from skimage.morphology import disk


def median_mask(mask, radius):
    mask_uinnt8 = img2uint8(mask)
    median_mask = filters.median(mask_uinnt8, disk(radius), mode='mirror')

    median_mask = median_mask / 255.0
    return median_mask.astype(np.uint8)
