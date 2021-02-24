from domaci_3.utils.edge_orientation import *


def nonlocal_maxima_suppression(sobel_ampsqr, sobel_angle, mask):
    for i in range(sobel_ampsqr.shape[0]):
        for j in range(sobel_ampsqr.shape[1]):
            # if i == 438 and j == 1071:
            #     print('OVDE')
            # if mask[i, j] == 0:
            #     continue
            if i == 0 or j == 0 or i == sobel_ampsqr.shape[0] - 1 or j == sobel_ampsqr.shape[1] - 1:
                sobel_ampsqr[i, j] = 0
                continue
            angle_orientation = get_orientation(sobel_angle[i, j])

            if angle_orientation == Orientation.vertical:
                if sobel_ampsqr[i, j] != max(sobel_ampsqr[i + 1, j], sobel_ampsqr[i, j], sobel_ampsqr[i - 1, j]):
                    sobel_ampsqr[i, j] = 0
            elif angle_orientation == Orientation.horizontal:
                if sobel_ampsqr[i, j] != max(sobel_ampsqr[i, j + 1], sobel_ampsqr[i, j], sobel_ampsqr[i, j - 1]):
                    sobel_ampsqr[i, j] = 0
            elif angle_orientation == Orientation.minus45:
                if sobel_ampsqr[i, j] != max(sobel_ampsqr[i + 1, j + 1], sobel_ampsqr[i, j],
                                             sobel_ampsqr[i - 1, j - 1]):
                    sobel_ampsqr[i, j] = 0
            elif angle_orientation == Orientation.plus45:
                if sobel_ampsqr[i, j] != max(sobel_ampsqr[i - 1, j + 1], sobel_ampsqr[i, j],
                                             sobel_ampsqr[i + 1, j - 1]):
                    sobel_ampsqr[i, j] = 0

    return sobel_ampsqr
