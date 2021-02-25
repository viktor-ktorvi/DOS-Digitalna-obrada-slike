import numpy as np
from skimage.transform import hough_line, hough_line_peaks


def get_hough_lines(image, theta_limits, num_points):
    tested_angles = np.linspace(theta_limits[0], theta_limits[1], num_points, endpoint=False)
    h, theta, d = hough_line(image, theta=tested_angles)

    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()

    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]

    x0 = np.zeros(2)
    y0 = np.zeros(2)
    slope = np.zeros(2)
    i = 0
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=2)):
        (x0[i], y0[i]) = dist * np.array([np.cos(angle), np.sin(angle)])
        slope[i] = np.tan(angle + np.pi / 2)
        i += 1

    return x0, y0, slope
