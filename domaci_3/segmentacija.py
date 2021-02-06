# %% importi
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.filters import threshold_otsu
import skvideo.io
import time
import os
from numpy.lib.stride_tricks import as_strided
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import cv2
import imageio
from skimage import filters
from skimage.morphology import disk

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
    test_frames.append(io.imread('sekvence/test_frames/' + file_name + str(i) + extension))

figsize = (10, 7)
fontsize = 20

# %% Sample
road_sample_height = (500, 600)
road_sample_width = (420, 920)

road = Sample(test_frames[3], road_sample_height, road_sample_height, threshold=15)
# %% Image
img = test_frames[2]

plt.figure(figsize=figsize)
plt.imshow(img)
plt.title("Input image")
plt.show()
# %% Hyperparameters

img_stride = 20
median_radius = 10
sample = road
num_bins = 500
gamma = 10
thresh_bonus = 1.1
a_lowpass = -0.9
plot_mid_result = True
plot_end_result = True
time_flag = True
do_video = True
whole_video_flag = True
print_percent_flag = True
num_frames = 200

# %% Statistical distance
if time_flag:
    start = time.perf_counter()
dist = distance_einsum(img[::img_stride, ::img_stride, :], sample.sigma_inv, sample.M)
dist_gamma_corrected = gamma_correction(dist, gamma)
if time_flag:
    print("Time gamma: " + "%0.4f" % (time.perf_counter() - start) + " sec")
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(dist, cmap='gray')
    plt.title("Statistical distance")
    plt.show()

if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(dist_gamma_corrected, cmap='gray')
    plt.title("Statistical distance - gamma corrected")
    plt.show()

hist_f, bin_edges = np.histogram(dist, bins=num_bins, range=(0.0, np.amax(dist)))

if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.plot(bin_edges[0:-1], hist_f)

    plt.title("Statistical distance histogram")
    plt.show()

if time_flag:
    start = time.perf_counter()
hist_f, bin_edges = np.histogram(dist_gamma_corrected, bins=num_bins, range=(0.0, np.amax(dist_gamma_corrected)))
if time_flag:
    print("Time hist: " + "%0.4f" % (time.perf_counter() - start) + " sec")
# %% threshold Otsu
if time_flag:
    start = time.perf_counter()
thresh = bin_edges[-1]
for i in range(2):
    thresh = threshold_otsu(dist_gamma_corrected[dist_gamma_corrected < thresh])
thresh *= thresh_bonus
if time_flag:
    print("Time otsu: " + "%0.4f" % (time.perf_counter() - start) + " sec")
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.plot(bin_edges[0:-1], hist_f)

    plt.vlines(thresh, np.amin(hist_f), np.amax(hist_f), colors='r')
    plt.title("Statistical distance - gamma corrected histogram")
    plt.show()

# %% Binarization
if time_flag:
    start = time.perf_counter()
binary_dist_gamma = dist_gamma_corrected < thresh
binary_dist_gamma = binary_dist_gamma.astype(np.uint8)
if time_flag:
    print("Time binary: " + "%0.4f" % (time.perf_counter() - start) + " sec")

if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(binary_dist_gamma, cmap='gray')
    plt.title("Statistical distance - gamma corrected - binary")
    plt.show()

# %% Spatial median filter
if time_flag:
    start = time.perf_counter()
filtered_mask = median_mask(binary_dist_gamma, median_radius)
if time_flag:
    print("Time median: " + "%0.4f" % (time.perf_counter() - start) + " sec")
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(filtered_mask, cmap='gray')
    plt.title("Median filtered mask")
    plt.show()

# %% Temporal lowpass filter
if time_flag:
    start = time.perf_counter()
prev_mask = filtered_mask
filtered_mask = -a_lowpass * prev_mask + (1 + a_lowpass) * filtered_mask
if time_flag:
    print("Time lowpass: " + "%0.4f" % (time.perf_counter() - start) + " sec")

# %% Resizing
if time_flag:
    start = time.perf_counter()
resized_mask = np.broadcast_to(filtered_mask[:, None, :, None],
                               (filtered_mask.shape[0], img_stride, filtered_mask.shape[1], img_stride)).reshape(
    filtered_mask.shape[0] * img_stride, filtered_mask.shape[1] * img_stride)

resized_mask = resized_mask[0:img.shape[0], 0:img.shape[1]]
if time_flag:
    print("Time resizing: " + "%0.4f" % (time.perf_counter() - start) + " sec")
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(resized_mask, cmap='gray')
    plt.title("Resized mask")
    plt.show()

# %% Whole segmentation
if time_flag:
    start = time.perf_counter()
segmented_image, _, _ = segment_frame(img=img,
                                      sample=road,
                                      img_stride=img_stride,
                                      gamma=gamma,
                                      num_bins=num_bins,
                                      thresh_bonus=thresh_bonus,
                                      median_radius=median_radius,
                                      a_lowpass=a_lowpass)
if time_flag:
    end = time.perf_counter()
    print("Time segmentation: " + "%0.4f" % (end - start) + " sec")
    estimated_time_per_frame = end - start
if plot_end_result:
    plt.figure(figsize=figsize)
    plt.imshow(segmented_image)
    plt.title("Segmented image")
    plt.show()

# %% Video processing
if do_video:
    if time_flag:
        start = time.perf_counter()
    videodata = skvideo.io.vread("sekvence/video_road.mp4")

    if whole_video_flag:
        frames_range = videodata.shape[0]
    else:
        frames_range = num_frames

    new_video = np.zeros_like(videodata[:frames_range])
    prev_mask = 0
    if time_flag:
        print("Time video loading: " + "%0.4f" % (time.perf_counter() - start) + " sec")
    if time_flag:
        print("Estimated time for video: " + "%0.4f" % (estimated_time_per_frame * frames_range) +" sec")
    if time_flag:
        start = time.perf_counter()
    for i in range(frames_range):
        new_video[i, :, :, :], _, prev_mask = segment_frame(img=videodata[i],
                                                            sample=road,
                                                            img_stride=img_stride,
                                                            gamma=gamma,
                                                            num_bins=num_bins,
                                                            thresh_bonus=thresh_bonus,
                                                            median_radius=median_radius,
                                                            a_lowpass=a_lowpass,
                                                            prev_mask=prev_mask,
                                                            cnt=i)
        if i % 50 == 0 and print_percent_flag:
            print("%3.1f" % ((i + 1) / frames_range * 100), " %", end="\r")

    if time_flag:
        print("Time video processing: " + "%0.4f" % (time.perf_counter() - start) + " sec")

    if time_flag:
        start = time.perf_counter()
    imageio.mimwrite(os.getcwd() + '\\sekvence\\out_vid.mp4', new_video, fps=24.0)
    if time_flag:
        print("Time video saving: " + "%0.4f" % (time.perf_counter() - start) + " sec")
# TODO Zasto je sporije nego sto treba?
