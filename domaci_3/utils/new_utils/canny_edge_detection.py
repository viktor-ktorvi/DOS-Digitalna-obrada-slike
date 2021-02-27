import numpy as np
import cv2


def canny_edge_detection(canny_input, threshold_low, threshold_high):
    gauss_kernel_size = 5
    img_filtered = cv2.GaussianBlur(canny_input, (gauss_kernel_size, gauss_kernel_size), 0)

    sobel_kernel_size = 3
    sobelx = cv2.Sobel(img_filtered, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    sobely = cv2.Sobel(img_filtered, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)

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

    edges[edges > threshold_high] = 1
    edges[edges < threshold_low] = 0

    row_idxs, col_idxs = np.nonzero(edges)

    # slabe ivice
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

    return edges, sobel_ampsqr, sobel_angle, sobel_angle_discrete
