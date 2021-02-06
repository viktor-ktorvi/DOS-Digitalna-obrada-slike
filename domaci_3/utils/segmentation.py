import numpy as np
from skimage.filters import threshold_otsu
from domaci_3.utils.img2double import img2double
from domaci_3.utils.distance import distance_einsum
from domaci_3.utils.gamma_correction import gamma_correction
from domaci_3.utils.median_mask import median_mask


def segment_frame(img, sample, img_stride, gamma, num_bins, thresh_bonus, median_radius, a_lowpass, prev_mask=0, cnt=0):

    dist = distance_einsum(img[::img_stride, ::img_stride, :], sample.sigma_inv, sample.M)
    dist_gamma_corrected = gamma_correction(dist, gamma)

    hist_f, bin_edges = np.histogram(dist_gamma_corrected, bins=num_bins, range=(0.0, np.amax(dist_gamma_corrected)))
    thresh = bin_edges[-1]
    for i in range(2):
        thresh = threshold_otsu(dist_gamma_corrected[dist_gamma_corrected < thresh])
    thresh *= thresh_bonus

    binary_dist_gamma = dist_gamma_corrected < thresh
    binary_dist_gamma = binary_dist_gamma.astype(np.uint8)
    filtered_mask = median_mask(binary_dist_gamma, median_radius)

    if cnt != 0:
        filtered_mask = -a_lowpass * prev_mask + (1 + a_lowpass) * filtered_mask

    binary_mask = filtered_mask > 0.5
    resized_mask = np.broadcast_to(binary_mask[:, None, :, None],
                                   (binary_mask.shape[0], img_stride, binary_mask.shape[1], img_stride)).reshape(
        binary_mask.shape[0] * img_stride, binary_mask.shape[1] * img_stride)
    resized_mask = resized_mask[0:img.shape[0], 0:img.shape[1]]

    return img * resized_mask[:, :, np.newaxis], resized_mask, filtered_mask
