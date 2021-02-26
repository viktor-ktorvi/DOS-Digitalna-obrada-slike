from skimage import io
import time
from skimage.color import rgb2lab, lab2rgb, rgb2gray
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

t_high = 0.02
t_low = 0.001

# %% Sample
road_sample_height = (500, 600)
road_sample_width = (420, 920)

sample_img = test_frames[3]
if lab_flag:
    sample_img = rgb2lab(sample_img)
road = Sample(sample_img, road_sample_height, road_sample_height, threshold=15)
sample = road
# %% Segmentation
img = test_frames[0]
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

# %% Gauss filter
# TODO Da li neki drugi kolor sistem moze bolje od grayscale (HSV?, LAB?, tipa samo H, ili samo L)
seg_img_gray = segmented_image
if lab_flag:
    seg_img_gray = lab2rgb(seg_img_gray)
seg_img_gray = rgb2gray(seg_img_gray)

kernel_size = 3
if time_flag:
    start = time.perf_counter()
img_filtered = cv2.GaussianBlur(seg_img_gray, (kernel_size, kernel_size), 0)
if time_flag:
    end = time.perf_counter()
    print("Time filtering: " + "%0.4f" % (end - start) + " sec")
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(img_filtered, cmap='gray')
    plt.title("Filtered image")
    plt.show()

if time_flag:
    start = time.perf_counter()
sobelx = cv2.Sobel(img_filtered, cv2.CV_64F, 1, 0, ksize=kernel_size)
sobely = cv2.Sobel(img_filtered, cv2.CV_64F, 0, 1, ksize=kernel_size)

sobel_ampsqr = sobelx ** 2 + sobely ** 2
sobel_ampsqr = sobel_ampsqr / np.amax(sobel_ampsqr)
sobel_angle = np.arctan2(sobely, sobelx)
if time_flag:
    end = time.perf_counter()
    print("Time sobel: " + "%0.4f" % (end - start) + " sec")

if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(sobelx, cmap='gray')
    plt.title("Sobel X")
    plt.show()
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(sobely, cmap='gray')
    plt.title("Sobel Y")
    plt.show()
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(sobel_ampsqr, cmap='gray')
    plt.title("Sobel x^2 + y^2")
    plt.show()
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(sobel_angle, cmap='gray')
    plt.title("Sobel atan2 y / x")
    plt.show()

# %% Local maxima
# TODO Ubrzati kako znas i umes
if time_flag:
    start = time.perf_counter()
sobel_nonlocal_suppressed = nonlocal_maxima_suppression(sobel_ampsqr, sobel_angle)
if time_flag:
    end = time.perf_counter()
    print("Time local maxima: " + "%0.4f" % (end - start) + " sec")
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(sobel_nonlocal_suppressed, cmap='gray')
    plt.title("Sobel x^2 + y^2 - local maxima")
    plt.show()

# %% Weak edges
if time_flag:
    start = time.perf_counter()

sobel_thresholded = sobel_nonlocal_suppressed
sobel_thresholded[sobel_thresholded > t_high] = 1
sobel_thresholded[sobel_thresholded < t_low] = 0

r = 1
img_padded = np.pad(sobel_thresholded, ((r, r), (r, r)),
                    mode="constant")
image_4d = make_4d_array_custom(img_padded, sobel_thresholded.shape[0], sobel_thresholded.shape[1], 2 * r + 1,
                                2 * r + 1)
sum_of_pixels = np.einsum('ijkl->ij', image_4d)

weak_edges_connected = sum_of_pixels > 1
weak_edges_connected[sobel_thresholded < t_low] = 0
weak_edges_connected = weak_edges_connected.astype(np.uint8)
if time_flag:
    end = time.perf_counter()
    print("Time thresh: " + "%0.4f" % (end - start) + " sec")
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(sobel_thresholded, cmap='gray')
    plt.title("Sobel thresholded")
    plt.show()
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(sum_of_pixels, cmap='gray')
    plt.title("Sum of pixels")
    plt.show()
if plot_mid_result:
    plt.figure(figsize=figsize)
    plt.imshow(weak_edges_connected, cmap='gray')
    plt.title("Weak edges connected")
    plt.show()
# %% Built in Canny
if time_flag:
    start = time.perf_counter()
edges = cv2.Canny(segmented_image, 100, 200)
if time_flag:
    end = time.perf_counter()
    print("Time Canny opencv: " + "%0.4f" % (end - start) + " sec")
if plot_end_result:
    plt.figure(figsize=figsize)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny opencv")
    plt.show()
# %% Funkcija
sigma = (kernel_size - 1) / 6

if time_flag:
    start = time.perf_counter()
my_canny, _ = canny_edge_detection(img_in=segmented_image, sigma=sigma, threshold_low=t_low, threshold_high=t_high)
if time_flag:
    end = time.perf_counter()
    print("Time my Canny: " + "%0.4f" % (end - start) + " sec")
if plot_end_result:
    plt.figure(figsize=figsize)
    plt.imshow(my_canny, cmap='gray')
    plt.title("My Canny")
    plt.show()
#%%
if time_flag:
    start = time.perf_counter()
y_idxs, x_idxs = np.nonzero(sobel_ampsqr)
if time_flag:
    end = time.perf_counter()
    print("Time nonezero: " + "%0.4f" % (end - start) + " sec")