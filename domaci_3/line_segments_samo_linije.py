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
from domaci_3.utils.get_hough_lines import get_hough_lines
from domaci_3.utils.MyLine import MyLine, make_lines, LineSegment, lines_are_touching, dist

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

edges, sobel_ampsqr, sobel_angle, sobel_angle_discrete = canny_edge_detection(dilation, threshold_low, threshold_high)

plt.figure(figsize=figsize)
plt.imshow(edges, cmap='gray')
plt.title("edges")
plt.show()
# %%
num_points = 360
theta_limits_left = [np.pi / 6, np.pi / 3]
theta_limits_right = [-np.pi / 3, -np.pi / 6]

left_line_params = get_hough_lines(edges, theta_limits_left, num_points)
right_line_params = get_hough_lines(edges, theta_limits_right, num_points)

left_lines = make_lines(*left_line_params)
right_lines = make_lines(*right_line_params)

plt.figure(figsize=figsize)
plt.imshow(edges, cmap='gray')
for i in range(len(left_lines)):
    left_lines[i].plotLine(color='r')
    right_lines[i].plotLine(color='b')
plt.title("lines")
plt.show()

tolerancy = 3
min_size = 30
max_gaps = 35

line = right_lines[0]
r = tolerancy
img_padded = np.pad(edges, ((r, r), (r, r)),
                    mode="constant")
image_4d = make_4d_array_custom(img_padded, edges.shape[0], edges.shape[1], 2 * r + 1,
                                2 * r + 1)
points_j = []
points_i = []
for j in range(edges.shape[1]):
    y = line.getYatX(j)
    i = round(y)
    if i < 0 or i > edges.shape[0] - 1:
        continue

    if np.count_nonzero(image_4d[i, j]) > 0:
        plt.plot(j, i, 'co')
        points_j.append(j)
        points_i.append(i)

segments = []
for cnt in range(len(points_j)):
    if cnt == 0:
        seg = LineSegment(x_start=points_j[0], x_end=points_j[0], y_start=points_i[0], y_end=points_i[0])

    if seg.dist2point(points_j[cnt], points_i[cnt]) < max_gaps:
        seg.addPoint(points_j[cnt], points_i[cnt])
    else:
        if seg.length > min_size:
            segments.append(seg)
        seg = LineSegment(x_start=points_j[cnt], x_end=points_j[cnt], y_start=points_i[cnt], y_end=points_i[cnt])

if seg.length > min_size:
    segments.append(seg)
seg = []

for seg in segments:
    seg.plotSegment(color='orange', linewidth=3)
