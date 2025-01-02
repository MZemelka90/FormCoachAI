import math


def calculate_three_point_angle(a: tuple, b: tuple, c: tuple):
    """
    Calculate the angle (in degrees) formed by the points A, B, and C.
    The angle is at point B with line segments BA and BC.

    Args:
        a (tuple): Coordinates of point A (x1, y1)
        b (tuple): Coordinates of point B (x2, y2)
        c (tuple): Coordinates of point C (x3, y3)

    Returns:
        float: The angle in degrees.
    """
    # Calculate vectors BA and BC
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    # Calculate the dot product and magnitudes of BA and BC
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if magnitude_ba == 0 or magnitude_bc == 0:
        raise ValueError("One or both vectors have zero length; cannot determine angle.")

    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_ba * magnitude_bc)

    # Clamp the value to avoid math domain errors due to floating-point precision
    cos_theta = max(-1, min(1, cos_theta))

    # Calculate the angle in radians and then convert to degrees
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


def calculate_two_point_angle(a, b):
    """
    Calculate the angle between two points

    :param a: First point
    :param b: Second point
    :return: Angle
    """
    if a[0] == b[0] and a[1] == b[1]:
        return 0

    # Differences
    delta_x = b[0] - a[0]
    delta_y = b[1] - a[1]

    # Angle in radians
    angle_radians = math.atan2(delta_y, delta_x)

    # Convert to degrees
    angle_degrees = math.degrees(angle_radians)
    return round(angle_degrees)


def calculate_distance(a, b):
    """
    Calculate the distance between two points.
    """
    return abs(math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2))
