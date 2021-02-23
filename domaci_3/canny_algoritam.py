from skimage import io
import time
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from matplotlib import pyplot as plt
import cv2
import numpy as np
from domaci_3.utils.Sample import Sample
from domaci_3.utils.segmentation import segment_frame
from domaci_3.utils.filter_gauss import filter_gauss

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
plot_mid_result = True
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
if time_flag:
    start = time.perf_counter()
segmented_image, _, _ = segment_frame(img=img,
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
if plot_end_result:
    plt.figure(figsize=figsize)
    plt.imshow(segmented_image)
    plt.title("Segmented image")
    plt.show()
# %% Gauss filter
seg_img_gray = segmented_image
if lab_flag:
    seg_img_gray = lab2rgb(seg_img_gray)
seg_img_gray = rgb2gray(seg_img_gray)

kernel_size = 11
if time_flag:
    start = time.perf_counter()
img_filtered = cv2.GaussianBlur(seg_img_gray, (kernel_size, kernel_size), 0)
if time_flag:
    end = time.perf_counter()
    print("Time filtering: " + "%0.4f" % (end - start) + " sec")
if plot_end_result:
    plt.figure(figsize=figsize)
    plt.imshow(img_filtered, cmap='gray')
    plt.title("Filtered image")
    plt.show()

if time_flag:
    start = time.perf_counter()
sobelx = cv2.Sobel(img_filtered, cv2.CV_64F, 1, 0, ksize=kernel_size)
sobely = cv2.Sobel(img_filtered, cv2.CV_64F, 0, 1, ksize=kernel_size)

sobel_ampsqr = sobelx ** 2 + sobely ** 2
sobel_angle = np.arctan2(sobely, sobelx)
if time_flag:
    end = time.perf_counter()
    print("Time sobel: " + "%0.4f" % (end - start) + " sec")

if plot_end_result:
    plt.figure(figsize=figsize)
    plt.imshow(sobelx, cmap='gray')
    plt.title("Sobel X")
    plt.show()
if plot_end_result:
    plt.figure(figsize=figsize)
    plt.imshow(sobely, cmap='gray')
    plt.title("Sobel Y")
    plt.show()
if plot_end_result:
    plt.figure(figsize=figsize)
    plt.imshow(sobel_ampsqr, cmap='gray')
    plt.title("Sobel x^2 + y^2")
    plt.show()
if plot_end_result:
    plt.figure(figsize=figsize)
    plt.imshow(sobel_angle, cmap='gray')
    plt.title("Sobel atan2 y / x")
    plt.show()

