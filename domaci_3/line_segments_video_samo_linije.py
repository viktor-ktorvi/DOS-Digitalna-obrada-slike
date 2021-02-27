# %% importi
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.filters import threshold_otsu
import skvideo.io
import time
from PIL import Image, ImageDraw

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
from domaci_3.utils.new_utils.get_lines import get_lines
from domaci_3.utils.new_utils.get_line_segments import get_line_segments
from domaci_3.utils.new_utils.merge_segments import merge_segments

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
img = test_frames[5]
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
num_points = 100
theta_limits_left = [np.pi / 4, 3 * np.pi / 8]
theta_limits_right = [- 3 * np.pi / 8, -np.pi / 4]
time_flag = True
if time_flag:
    start = time.perf_counter()
lines, left_lines, right_lines = get_lines(edges, theta_limits_left, theta_limits_right, num_points)
if time_flag:
    end = time.perf_counter()
    print("Time Hough lines: " + "%0.4f" % (end - start) + " sec")

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
if time_flag:
    start = time.perf_counter()
segments, all_points_j, all_points_i = get_line_segments(edges, lines, min_size, max_gaps, tolerancy)
if time_flag:
    end = time.perf_counter()
    print("Time segments: " + "%0.4f" % (end - start) + " sec")
for cnt in range(len(all_points_j)):
    plt.plot(all_points_j[cnt], all_points_i[cnt], 'co')

right_segments = []
left_segments = []
for seg in segments:
    if np.sign(seg.slope) == np.sign(right_lines[0].slope):
        right_segments.append(seg)
    elif np.sign(seg.slope) == np.sign(left_lines[0].slope):
        left_segments.append(seg)

for seg in segments:
    seg.plotSegment(color='orange', linewidth=3)

if time_flag:
    start = time.perf_counter()
stacked_img = np.stack((edges,) * 3, axis=-1)
im = Image.fromarray(np.uint8(stacked_img * 255))
draw = ImageDraw.Draw(im)

right_line_coords = merge_segments(right_segments)
left_line_coords = merge_segments(left_segments)

draw.line(right_line_coords, fill=(0, 255, 0, 255), width=9)
draw.line(left_line_coords, fill=(0, 255, 0, 255), width=9)

img_lines_drawn = np.asarray(im)
if time_flag:
    end = time.perf_counter()
    print("Time line drawing: " + "%0.4f" % (end - start) + " sec")

plt.figure(figsize=figsize)
plt.imshow(img_lines_drawn)
plt.title("img_lines_drawn")
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
        dilation, _, _, _, _ = segment_lanes(videodata[i])
        edges, _, _, _ = canny_edge_detection(dilation, threshold_low,
                                              threshold_high)

        lines, _, _ = get_lines(edges, theta_limits_left, theta_limits_right, num_points)
        segments, _, _ = get_line_segments(edges, lines, min_size, max_gaps, tolerancy)

        right_segments = []
        left_segments = []
        for seg in segments:
            if np.sign(seg.slope) == np.sign(right_lines[0].slope):
                right_segments.append(seg)
            elif np.sign(seg.slope) == np.sign(left_lines[0].slope):
                left_segments.append(seg)

        # stacked_img = np.stack((edges,) * 3, axis=-1)
        # im = Image.fromarray(np.uint8(stacked_img * 255))
        im = Image.fromarray(videodata[i])
        draw = ImageDraw.Draw(im)

        right_line_coords = merge_segments(right_segments)
        left_line_coords = merge_segments(left_segments)

        draw.line(right_line_coords, fill=(0, 255, 0, 255), width=9)
        draw.line(left_line_coords, fill=(0, 255, 0, 255), width=9)

        new_video[i, :, :, :] = np.asarray(im)
        if i % 50 == 0 and print_percent_flag:
            print("%3.1f" % ((i + 1) / frames_range * 100), " %", end="\r")

    if time_flag:
        print("Time video processing: " + "%0.4f" % (time.perf_counter() - start) + " sec")

    if time_flag:
        start = time.perf_counter()
    imageio.mimwrite(os.getcwd() + '\\sekvence\\out_vid_lines.mp4', new_video, fps=24.0)
    if time_flag:
        print("Time video saving: " + "%0.4f" % (time.perf_counter() - start) + " sec")
