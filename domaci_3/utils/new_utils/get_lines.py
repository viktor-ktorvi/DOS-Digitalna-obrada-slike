from domaci_3.utils.get_hough_lines import get_hough_lines
from domaci_3.utils.MyLine import make_lines


def get_lines(edges, theta_limits_left, theta_limits_right, num_points):
    left_line_params = get_hough_lines(edges, theta_limits_left, num_points)
    right_line_params = get_hough_lines(edges, theta_limits_right, num_points)

    left_lines = make_lines(*left_line_params)
    right_lines = make_lines(*right_line_params)
    lines = right_lines + left_lines

    return lines, left_lines, right_lines
