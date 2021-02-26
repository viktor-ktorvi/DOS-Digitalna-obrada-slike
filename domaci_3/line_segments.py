from skimage import io
import time
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from matplotlib import pyplot as plt
import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks

from domaci_3.utils.Sample import Sample
from domaci_3.utils.segmentation import segment_frame
from domaci_3.utils.filter_gauss import filter_gauss
from domaci_3.utils.edge_orientation import Orientation, get_orientation
from domaci_3.utils.nonlocal_maxima_suppression import nonlocal_maxima_suppression
from domaci_3.utils.array_4d import make_4d_array_custom
from domaci_3.utils.canny_edge_detection import canny_edge_detection
from domaci_3.utils.get_hough_lines import get_hough_lines
from domaci_3.utils.MyLine import MyLine, make_lines, LineSegment, lines_are_touching

# %%

test_frames = []
file_name = 'test'
extension = '.jpg'

for i in range(1, 7):
    img = io.imread('sekvence/test_frames/' + file_name + str(i) + extension)
    test_frames.append(img)

figsize = (10, 7)
fontsize = 20
# %%
img_stride = 20
median_radius = 10
num_bins = 500
gamma = 10
thresh_bonus = 1.06
a_lowpass = -0.9
lab_flag = False
plot_mid_result = False
plot_end_result = True
time_flag = True

# %% Sample
road_sample_height = (500, 600)
road_sample_width = (420, 920)

sample_img = test_frames[3]
if lab_flag:
    sample_img = rgb2lab(sample_img)
road = Sample(sample_img, road_sample_height, road_sample_height, threshold=15)
sample = road
# %% Segmentation
img = test_frames[5]
if time_flag:
    start = time.perf_counter()
segmented_image, mask, _ = segment_frame(img=img,
                                         sample=road,
                                         lab_flag=lab_flag,
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
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(segmented_image)
    plt.title("Segmented image")
    plt.show()
# TODO mozda mogu da napravim pozadinu maske sive boje - tipa 127 kako gradijenti na ivici ne bi bili ogromni
# %% Canny
t_high = 0.005
t_low = 0.001
kernel_size = 3
sigma = (kernel_size - 1) / 6
if time_flag:
    start = time.perf_counter()
my_canny, sobel_angle = canny_edge_detection(img_in=segmented_image, sigma=sigma, threshold_low=t_low,
                                             threshold_high=t_high)
if time_flag:
    end = time.perf_counter()
    print("Time my Canny: " + "%0.4f" % (end - start) + " sec")
if plot_end_result:
    plt.figure(figsize=figsize)
    plt.imshow(my_canny, cmap='gray')
    plt.title("My Canny")
    plt.show()

# %%
if time_flag:
    start = time.perf_counter()
image = my_canny
num_points = 360
theta_limits_right = [-np.pi / 3, -np.pi / 6]
theta_limits_left = [np.pi / 6, np.pi / 3]

xx, yy, slope = get_hough_lines(image, theta_limits_left, num_points)
left_lines = make_lines(xx, yy, slope)

xx, yy, slope = get_hough_lines(image, theta_limits_right, num_points)
right_lines = make_lines(xx, yy, slope)

if time_flag:
    end = time.perf_counter()
    print("Time Hough lines: " + "%0.4f" % (end - start) + " sec")
if plot_end_result:
    plt.figure(figsize=figsize)
    plt.imshow(my_canny, cmap='gray')

    for i in range(len(left_lines)):
        left_lines[i].plotLine(color='r')
        right_lines[i].plotLine(color='b')

    plt.title("My Canny - linije")
    plt.show()

line = right_lines[0]

tolerancy = 1
min_size = 10
max_gaps = 5
r = tolerancy
img_padded = np.pad(image, ((r, r), (r, r)),
                    mode="constant")
image_4d = make_4d_array_custom(img_padded, image.shape[0], image.shape[1], 2 * r + 1,
                                2 * r + 1)
sum_of_pixels = np.einsum('ijkl->ij', image_4d)

currently_in_line = False
first_segment_flag = True

all_segments = []
for j in range(image.shape[1]):
    y = line.getYatX(j)
    i = round(y)
    if i < 0 or i > image.shape[0] - 1:
        continue

    if sum_of_pixels[i, j] > 0 and 20 < np.rad2deg(sobel_angle[i, j] - np.pi / 2) < 60:
        print(sobel_angle[i, j], i, j)

        if not currently_in_line:
            j_start = j
            i_start = i
            currently_in_line = True
    else:
        if currently_in_line:
            if not first_segment_flag:
                not_segment = LineSegment(j_end, j_start, i_end, i_start)
                all_segments.append(not_segment)
            j_end = j
            i_end = i
            currently_in_line = False
            first_segment_flag = False

            segment = LineSegment(j_start, j_end, i_start, i_end)
            all_segments.append(segment)

segments_of_all_sizes = []

# Prvo popunjavamo rupe
for i in range(len(all_segments)):
    # segment
    if i % 2 == 0:
        if i + 1 != len(all_segments) and all_segments[i + 1] is not None:
            if all_segments[i + 1].length < max_gaps:
                all_segments[i].y_end = all_segments[i + 1].y_end
                all_segments[i].x_end = all_segments[i + 1].x_end

                all_segments[i].calcLength()
                all_segments[i + 1] = None

        if i - 1 != -1 and all_segments[i - 1] is not None:
            if all_segments[i - 1].length < max_gaps:
                all_segments[i].y_start = all_segments[i - 1].y_start
                all_segments[i].x_start = all_segments[i - 1].x_start

                all_segments[i].calcLength()
                all_segments[i - 1] = None

        segments_of_all_sizes.append(all_segments[i])
    # not segment
    else:
        if all_segments[i] is not None:
            all_segments[i].plotSegment(color='orange', linewidth=3)

valid_segments = []
# izbacujemo male segmente
for i in range(len(segments_of_all_sizes)):
    if segments_of_all_sizes[i].length < min_size:
        if i != 0:
            if lines_are_touching(segments_of_all_sizes[i], segments_of_all_sizes[i - 1]):
                # TODO mozda proveriti zbir duzina?
                valid_segments.append(segments_of_all_sizes[i])
        if i + 1 != len(segments_of_all_sizes):
            if lines_are_touching(segments_of_all_sizes[i], segments_of_all_sizes[i + 1]):
                # TODO mozda proveriti zbir duzina?
                valid_segments.append(segments_of_all_sizes[i])
    else:
        valid_segments.append(segments_of_all_sizes[i])

for i in range(len(valid_segments)):
    valid_segments[i].plotSegment(color='cyan', linewidth=3)

# TODO Dakle odradjeno ja prepoznavanje linija, testirano na jednoj slici samo za jednu stranu jedne linije
#  Pritom canny prag ne vidi linije u mnogim slikama, dosta posla ima, mozda algoritam nije bas robustan, videcemo,
#  treba srediti dosta, spakovati u funkcije, proveriti rezultate, tempirati, i ako ostne vremena, ubrzati