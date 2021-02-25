import cv2
from skimage.color import rgb2gray
import numpy as np

from domaci_3.utils.nonlocal_maxima_suppression import nonlocal_maxima_suppression
from domaci_3.utils.array_4d import make_4d_array_custom


def canny_edge_detection(img_in, sigma, threshold_low, threshold_high):
    gray_image = rgb2gray(img_in)

    gauss_radius = round(3 * sigma)
    kernel_size = 2 * gauss_radius + 1
    img_filtered = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)

    sobelx = cv2.Sobel(img_filtered, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(img_filtered, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sobel_ampsqr = sobelx ** 2 + sobely ** 2
    sobel_ampsqr = sobel_ampsqr / np.amax(sobel_ampsqr)
    sobel_angle = np.arctan2(sobely, sobelx)

    sobel_nonlocal_suppressed = nonlocal_maxima_suppression(sobel_ampsqr, sobel_angle)

    sobel_thresholded = sobel_nonlocal_suppressed
    sobel_thresholded[sobel_thresholded > threshold_high] = 1
    sobel_thresholded[sobel_thresholded < threshold_low] = 0

    r = 1
    img_padded = np.pad(sobel_thresholded, ((r, r), (r, r)),
                        mode="constant")
    image_4d = make_4d_array_custom(img_padded, sobel_thresholded.shape[0], sobel_thresholded.shape[1], 2 * r + 1,
                                    2 * r + 1)
    sum_of_pixels = np.einsum('ijkl->ij', image_4d)

    weak_edges_connected = sum_of_pixels > 1
    weak_edges_connected[sobel_thresholded < threshold_low] = 0
    weak_edges_connected = weak_edges_connected.astype(np.uint8)

    return weak_edges_connected, sobel_angle
