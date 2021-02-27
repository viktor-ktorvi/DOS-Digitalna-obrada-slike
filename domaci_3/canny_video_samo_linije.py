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
from skimage.color import rgb2lab, lab2rgb, rgb2gray

from domaci_3.utils.array_4d import make_4d_array_custom
from domaci_3.utils.distance import distance_einsum
from domaci_3.utils.segmentation_mask import segment_by_sample_with_resizing
from domaci_3.utils.Sample import Sample
from domaci_3.utils.division import get_mass_division
from domaci_3.utils.img2double import img2double
from domaci_3.utils.gamma_correction import gamma_correction
from domaci_3.utils.median_mask import median_mask
from domaci_3.utils.segmentation import segment_frame

from domaci_3.utils.new_utils.segment_lanes import segment_lanes
from domaci_3.utils.new_utils.canny_edge_detection import canny_edge_detection

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
# %%

img = videodata[545]
plt.figure(figsize=figsize)
plt.imshow(img)
plt.title("Input img")
plt.show()
# %% Segmentacija
dilation, closing, merged_mask, mask_white, mask_yellow = segment_lanes(img)
# %% Canny
threshold_low = 0.3
threshold_high = 0.6
time_flag = True
if time_flag:
    start = time.perf_counter()
edges, sobel_ampsqr, sobel_angle, sobel_angle_discrete = canny_edge_detection(dilation, threshold_low, threshold_high)
if time_flag:
    print("Time canny: " + "%0.4f" % (time.perf_counter() - start) + " sec")
plt.figure(figsize=figsize)
plt.imshow(edges, cmap='gray')
plt.title("edges")
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

    new_video = np.zeros((videodata.shape[0], videodata.shape[1], videodata.shape[2]))

    if time_flag:
        start = time.perf_counter()
    for i in range(frames_range):
        dilation, _, _, _, _ = segment_lanes(videodata[i])
        edges, _, _, _ = canny_edge_detection(dilation, threshold_low,
                                              threshold_high)

        new_video[i, :, :] = edges * 255
        new_video[i, :, :] = new_video[i, :, :].astype(np.uint8)
        if i % 50 == 0 and print_percent_flag:
            print("%3.1f" % ((i + 1) / frames_range * 100), " %", end="\r")

    if time_flag:
        print("Time video processing: " + "%0.4f" % (time.perf_counter() - start) + " sec")

    if time_flag:
        start = time.perf_counter()
    imageio.mimwrite(os.getcwd() + '\\sekvence\\out_vid_canny.mp4', new_video, fps=24.0)
    if time_flag:
        print("Time video saving: " + "%0.4f" % (time.perf_counter() - start) + " sec")
