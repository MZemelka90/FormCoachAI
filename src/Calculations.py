import numpy as np
import math


def calculate_three_point_angle(a: list, b: list, c: list) -> float:
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


def calculate_two_point_angle(a, b):
    # Differences
    delta_x = b[0] - a[0]
    delta_y = b[1] - a[1]

    # Angle in radians
    angle_radians = math.atan2(delta_y, delta_x)

    # Convert to degrees
    angle_degrees = math.degrees(angle_radians)
    return round(angle_degrees)


def calculate_distance(a, b):
    return abs(math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2))