# %% importi
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.filters import threshold_otsu
import skvideo.io
import time
import os
from skimage.color import rgb2lab, lab2rgb
from numpy.lib.stride_tricks import as_strided
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import cv2
import imageio
from skimage import filters
from skimage.morphology import disk
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

from domaci_3.utils.distance import distance_einsum
from domaci_3.utils.segmentation_mask import segment_by_sample_with_resizing
from domaci_3.utils.Sample import Sample
from domaci_3.utils.division import get_mass_division
from domaci_3.utils.img2double import img2double
from domaci_3.utils.gamma_correction import gamma_correction
from domaci_3.utils.median_mask import median_mask
from domaci_3.utils.segmentation import segment_frame

# %% Ucitavanje

test_frames = []
file_name = 'test'
extension = '.jpg'

for i in range(1, 7):
    img = io.imread('sekvence/test_frames/' + file_name + str(i) + extension)
    test_frames.append(img)

figsize = (10, 7)
fontsize = 20

videodata = skvideo.io.vread("sekvence/video_road.mp4")
time_flag = True
# %%
rand_int = np.random.randint(0, videodata.shape[0] - 1)
print(rand_int)
img = videodata[545]
plt.figure(figsize=figsize)
plt.imshow(img)
plt.show()
# %% Yellow line mask
yellow = (200, 170, 0)
yellowish = (255, 255, 140)
mask_yellow = cv2.inRange(img, yellow, yellowish)
result = cv2.bitwise_and(img, img, mask=mask_yellow)
plt.figure(figsize=figsize)
plt.subplot(1, 2, 1)
plt.imshow(mask_yellow, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title("Zuta traka")
plt.show()
# zuta traka (r,g,b) - (r > 225, g > 200, b < 100)
# bela sve vece od 225?
# %% White line mask
white = (255, 255, 245)
whiteish = (240, 220, 210)
mask_white = cv2.inRange(img, whiteish, white)
result = cv2.bitwise_and(img, img, mask=mask_white)
plt.figure(figsize=figsize)
plt.subplot(1, 2, 1)
plt.imshow(mask_white, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title("Bela traka")
plt.show()
# %% Merged masks
merged_mask = mask_white | mask_yellow
merged_mask_result = cv2.bitwise_and(img, img, mask=merged_mask)
plt.figure(figsize=figsize)
plt.subplot(1, 2, 1)
plt.imshow(merged_mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(merged_mask_result)
plt.title("Spojene maske")
plt.show()
# %% Closing
kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
closing = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel_closing)
closing_result = cv2.bitwise_and(img, img, mask=closing)
plt.figure(figsize=figsize)
plt.subplot(1, 2, 1)
plt.imshow(closing, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(closing_result)
plt.title("Closing")
plt.show()
# %% Dilation
# kernel_dilation = np.ones((7, 7), np.uint8)
kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
dilation = cv2.dilate(closing, kernel_dilation, iterations=1)
dilation_result = cv2.bitwise_and(img, img, mask=dilation)
plt.figure(figsize=figsize)
plt.subplot(1, 2, 1)
plt.imshow(dilation, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(dilation_result)
plt.title("Dilation")
plt.show()


do_video = True
whole_video_flag = True
print_percent_flag = True
num_frames = 200
if do_video:

    if whole_video_flag:
        frames_range = videodata.shape[0]
    else:
        frames_range = num_frames

    new_video = np.zeros_like(videodata[:frames_range])

    if time_flag:
        start = time.perf_counter()
    for i in range(frames_range):
        mask_yellow = cv2.inRange(videodata[i], yellow, yellowish)
        mask_white = cv2.inRange(videodata[i], whiteish, white)
        merged_mask = mask_white | mask_yellow
        closing = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel_closing)
        dilation = cv2.dilate(closing, kernel_dilation, iterations=1)

        new_video[i, :, :, :] = cv2.bitwise_and(videodata[i], videodata[i], mask=dilation)
        if i % 50 == 0 and print_percent_flag:
            print("%3.1f" % ((i + 1) / frames_range * 100), " %", end="\r")

    if time_flag:
        print("Time video processing: " + "%0.4f" % (time.perf_counter() - start) + " sec")

    if time_flag:
        start = time.perf_counter()
    imageio.mimwrite(os.getcwd() + '\\sekvence\\out_vid.mp4', new_video, fps=24.0)
    if time_flag:
        print("Time video saving: " + "%0.4f" % (time.perf_counter() - start) + " sec")
