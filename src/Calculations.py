import numpy as np


def calculate_angle(a: list, b: list, c: list) -> float:
    """
    Calculate the angle between three points.

    :param a: First point
    :param b: Midpoint
    :param c: Endpoint
    :return: Angle
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
