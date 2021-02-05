# %% importi
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.filters import threshold_otsu
import skvideo.io
import time
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
img_double = img2double(img)

plt.figure(figsize=figsize)
plt.imshow(img_double)
plt.title("Input image")
plt.show()
# %% Hyperparameters

img_step = 10
median_radius = 7
sample = road
num_bins = 500
gamma = 10  # TODO mozda je gamma preveliko
thresh_bonus = 1.05
plot_mid_result = True
time_flag = True
# %% Statistical distance

dist = distance_einsum(img_double[::img_step, ::img_step, :], sample.sigma_inv, sample.M)
if time_flag:
    start = time.time()
dist_gamma_corrected = gamma_correction(dist, gamma)
if time_flag:
    print("Time gamma: " + "%0.4f" % (time.time() - start) + " sec")

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

# Gamma corrected distance
hist_f, bin_edges = np.histogram(dist_gamma_corrected, bins=num_bins, range=(0.0, np.amax(dist_gamma_corrected)))
# double threshold_otsu

if time_flag:
    start = time.time()
thresh = bin_edges[-1]
for i in range(2):
    thresh = threshold_otsu(dist_gamma_corrected[dist_gamma_corrected < thresh])
thresh *= thresh_bonus
if time_flag:
    print("Time otsu: " + "%0.4f" % (time.time() - start) + " sec")
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.plot(bin_edges[0:-1], hist_f)

    plt.vlines(thresh, np.amin(hist_f), np.amax(hist_f), colors='r')
    plt.title("Statistical distance - gamma corrected histogram")
    plt.show()

if time_flag:
    start = time.time()
binary_dist_gamma = dist_gamma_corrected < thresh
binary_dist_gamma = binary_dist_gamma.astype(np.uint8)
if time_flag:
    print("Time binary: " + "%0.4f" % (time.time() - start) + " sec")

if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(binary_dist_gamma, cmap='gray')
    plt.title("Statistical distance - gamma corrected - binary")
    plt.show()

if time_flag:
    start = time.time()
filtered_mask = median_mask(binary_dist_gamma, median_radius)
if time_flag:
    print("Time median: " + "%0.4f" % (time.time() - start) + " sec")
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(filtered_mask, cmap='gray')
    plt.title("Median filtered mask")
    plt.show()

if time_flag:
    start = time.time()
resized_mask = np.broadcast_to(filtered_mask[:, None, :, None],
                               (filtered_mask.shape[0], img_step, filtered_mask.shape[1], img_step)).reshape(
    filtered_mask.shape[0] * img_step, filtered_mask.shape[1] * img_step)

resized_mask = resized_mask[0:img_double.shape[0], 0:img_double.shape[1]]
if time_flag:
    print("Time resizing: " + "%0.4f" % (time.time() - start) + " sec")
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(resized_mask, cmap='gray')
    plt.title("Resized mask")
    plt.show()

# TODO
#  create function out of all of this
#  test on video
#  add low pass temporal filtering