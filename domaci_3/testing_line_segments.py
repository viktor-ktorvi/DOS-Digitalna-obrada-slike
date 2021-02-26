from skimage import io
import time
from skimage.color import rgb2lab, lab2rgb, rgb2gray, rgb2hsv
from matplotlib import pyplot as plt
import cv2
import numpy as np
from domaci_3.utils.Sample import Sample
from domaci_3.utils.segmentation import segment_frame
from domaci_3.utils.filter_gauss import filter_gauss
from domaci_3.utils.edge_orientation import Orientation, get_orientation
from domaci_3.utils.nonlocal_maxima_suppression import nonlocal_maxima_suppression
from domaci_3.utils.array_4d import make_4d_array_custom
from domaci_3.utils.canny_edge_detection import canny_edge_detection
from domaci_3.utils.MyLine import MyLine, make_lines, LineSegment, lines_are_touching
from domaci_3.utils.get_hough_lines import get_hough_lines
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

t_high = 0.005
t_low = 0.001
kernel_size = 3
sigma = (kernel_size - 1) / 6

tolerancy = 1
min_size = 10
max_gaps = 5
r = tolerancy
# %% Sample
road_sample_height = (500, 600)
road_sample_width = (420, 920)

sample_img = test_frames[3]
if lab_flag:
    sample_img = rgb2lab(sample_img)
road = Sample(sample_img, road_sample_height, road_sample_height, threshold=15)
sample = road
# %%
for cnt in range(len(test_frames)):
    img = test_frames[cnt]
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
        start = time.perf_counter()
    # img_in = rgb2hsv(segmented_image)
    img_in = segmented_image
    # img_in = img_in[:, :, 2]
    my_canny, _ = canny_edge_detection(img_in=img_in, sigma=sigma, threshold_low=t_low, threshold_high=t_high, rgb_flag=True)
    if time_flag:
        end = time.perf_counter()
        print("Time my Canny: " + "%0.4f" % (end - start) + " sec")
    if plot_end_result:
        plt.figure(figsize=figsize)
        plt.imshow(img, cmap='gray')
        plt.title("Test frame " + "%d" % cnt)
        plt.show()
    if plot_end_result:
        plt.figure(figsize=figsize)
        plt.imshow(my_canny, cmap='gray')
        plt.title("My Canny test frame " + "%d" % cnt)
        plt.show()

    #%%
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