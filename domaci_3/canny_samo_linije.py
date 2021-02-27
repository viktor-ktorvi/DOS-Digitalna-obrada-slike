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

dilation_result = cv2.bitwise_and(img, img, mask=dilation)

canny_input = dilation
plt.figure(figsize=figsize)
plt.imshow(canny_input)
plt.title("canny_input")
plt.show()

# %% Preksacemo filtriranje posto je maska
kernel_size = 5
img_filtered = canny_input
# canny_input = rgb2gray(canny_input)
img_filtered = cv2.GaussianBlur(canny_input, (kernel_size, kernel_size), 0)
# cudna stvar! Kod potisikvanja lokalnih nemaksimuma desi se da je mnogo piksela u susedstvu jednako pa se ne potisnu,
# ovaj maleni sum ih ucini nejednakim !!!
# img_filtered = img_filtered + np.random.randn(*img_filtered.shape) * 1e-9
# plt.figure(figsize=figsize)
# plt.imshow(img_filtered, cmap='gray')
# plt.title("img_filtered")
# plt.show()
# %%
sobel_kernel = 3
sobelx = cv2.Sobel(img_filtered, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
sobely = cv2.Sobel(img_filtered, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

sobel_ampsqr = sobelx ** 2 + sobely ** 2
sobel_ampsqr = sobel_ampsqr / np.amax(sobel_ampsqr)
edges = sobel_ampsqr
sobel_angle = np.arctan2(sobely, sobelx) / np.pi * 180

bins = np.array([-200, -157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5, 200])
# digitize ce vratiti od 1 do 9 pa -5 je od -4 do 4 pa puta 45 je od -180 do 180 diskretno
sobel_angle_discrete = (np.digitize(sobel_angle, bins) - 5) * 45
sobel_angle_discrete[sobel_angle_discrete == -180] = 0
sobel_angle_discrete[sobel_angle_discrete == 180] = 0
sobel_angle_discrete[sobel_angle_discrete == -135] = 45
sobel_angle_discrete[sobel_angle_discrete == -90] = 90
sobel_angle_discrete[sobel_angle_discrete == -45] = 135

# malo je bzvz ispalo 90 su horizontalne ivice, 135 su kao u levo 45, 45 je u desno 45, 0 je vertikalna ivica

plt.figure(figsize=figsize)
plt.imshow(sobel_ampsqr, cmap='gray')
plt.title("Sobel x^2 + y^2")
plt.show()

plt.figure(figsize=figsize)
plt.imshow(sobel_angle_discrete, cmap='gray')
plt.title("Sobel atan2 y / x diskretno")
plt.show()
# %% Potiskivanje nelokalnih maksimuma
row_idxs, col_idxs = np.nonzero(edges)

for cnt in range(len(row_idxs)):
    i = row_idxs[cnt]
    j = col_idxs[cnt]
    # provera da nismo kod ivica
    if i == 0 or j == 0 or i == edges.shape[0] - 1 or j == edges.shape[1] - 1:
        edges[i, j] = 0
        continue

    # vertikalna ivica -> horizontalno susedstvo
    if sobel_angle_discrete[i, j] == 0:
        if edges[i, j] != max(edges[i, j - 1], edges[i, j], edges[i, j + 1]):
            edges[i, j] = 0

    # ivica u desno po 45 -> susedstvo u levo po 45
    if sobel_angle_discrete[i, j] == 45:
        if edges[i, j] != max(edges[i - 1, j - 1], edges[i, j], edges[i + 1, j + 1]):
            edges[i, j] = 0

    # horizontalna ivica -> vertikalno susedstvo
    if sobel_angle_discrete[i, j] == 90:
        if edges[i, j] != max(edges[i - 1, j], edges[i, j], edges[i + 1, j]):
            edges[i, j] = 0

    # ivica u levo po 45 -> susedstvo u desno po 45
    if sobel_angle_discrete[i, j] == 135:
        if edges[i, j] != max(edges[i + 1, j - 1], edges[i, j], edges[i - 1, j + 1]):
            edges[i, j] = 0

plt.figure(figsize=figsize)
plt.imshow(edges, cmap='gray')
plt.title("Potisnuti lokalni ne maksimumi")
plt.show()

threshold_low = 0.3
threshold_high = 0.6

edges[edges > threshold_high] = 1
edges[edges < threshold_low] = 0

row_idxs, col_idxs = np.nonzero(edges)

for cnt in range(len(row_idxs)):
    i = row_idxs[cnt]
    j = col_idxs[cnt]
    # provera da nismo kod ivica
    if i == 0 or j == 0 or i == edges.shape[0] - 1 or j == edges.shape[1] - 1:
        edges[i, j] = 0
        continue

    if edges[i, j] == 1:
        continue

    neighbourhood = edges[i - 1:i + 2, j - 1:j + 2]
    if neighbourhood[neighbourhood == 1].size > 0:
        edges[i, j] = 1
    else:
        edges[i, j] = 0

plt.figure(figsize=figsize)
plt.imshow(edges, cmap='gray')
plt.title("edges")
plt.show()