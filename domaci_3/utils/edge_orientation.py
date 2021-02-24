from enum import Enum
import numpy as np


class Orientation(Enum):
    horizontal = 1
    vertical = 2
    plus45 = 3
    minus45 = 4


def get_orientation(angle):
    if angle < -np.pi or angle > np.pi:
        raise Exception("Angle must be between -PI and +Pi")

    # Horizontal
    if -np.pi / 8 <= angle < np.pi / 8:
        return Orientation.horizontal
    if angle < -7 * np.pi / 8:
        return Orientation.horizontal
    if 7 * np.pi / 8 <= angle:
        return Orientation.horizontal

    # minus 45
    if np.pi / 8 < angle <= 3 * np.pi / 8:
        return Orientation.minus45
    if - 7 * np.pi / 8 < angle <= -5 * np.pi / 8:
        return Orientation.minus45

    # Vertical
    if 3 * np.pi / 8 < angle <= 5 * np.pi / 8:
        return Orientation.vertical
    if - 5 * np.pi / 8 < angle <= - 3 * np.pi / 8:
        return Orientation.vertical

    # plus 45
    if 5 * np.pi / 8 < angle <= 7 * np.pi / 8:
        return Orientation.plus45
    if -3 * np.pi / 8 < angle <= - np.pi / 8:
        return Orientation.plus45


# angle = np.linspace(-180, 180, 200)
#
# for i in range(len(angle)):
#     print("%3.2f" % angle[i], "\t\t", get_orientation(angle[i] / 180 * np.pi))
