import unittest
from src.Calculations import calculate_distance, calculate_two_point_angle, calculate_three_point_angle


class TestCalculateThreePointAngle(unittest.TestCase):

    def test_right_angle(self):
        a = (0, 1)
        b = (0, 0)
        c = (1, 0)
        self.assertAlmostEqual(calculate_three_point_angle(a, b, c), 90.0, places=5)

    def test_straight_line(self):
        a = (0, 0)
        b = (1, 0)
        c = (2, 0)
        self.assertAlmostEqual(calculate_three_point_angle(a, b, c), 180.0, places=5)

    def test_obtuse_angle(self):
        a = (-1, 0)
        b = (0, 0)
        c = (1, 1)
        self.assertAlmostEqual(calculate_three_point_angle(a, b, c), 135.0, places=5)

    def test_collinear_points(self):
        a = (0, 0)
        b = (1, 1)
        c = (2, 2)
        self.assertAlmostEqual(calculate_three_point_angle(a, b, c), 180.0, places=5)

    def test_zero_vector(self):
        a = (1, 1)
        b = (1, 1)
        c = (2, 2)
        with self.assertRaises(ValueError):
            calculate_three_point_angle(a, b, c)


class TestCalculateTwoPointAngle(unittest.TestCase):

    def test_horizontal_line(self):
        a = (0, 0)
        b = (1, 0)
        self.assertAlmostEqual(calculate_two_point_angle(a, b), 0.0, places=5)

    def test_vertical_line(self):
        a = (0, 0)
        b = (0, 1)
        self.assertAlmostEqual(calculate_two_point_angle(a, b), 90.0, places=5)

    def test_obtuse_angle(self):
        a = (0, 0)
        b = (1, 1)
        self.assertAlmostEqual(calculate_two_point_angle(a, b), 45.0, places=5)

    def test_zero_vector(self):
        a = (0, 0)
        b = (0, 0)
        self.assertAlmostEqual(calculate_two_point_angle(a, b), 0.0, places=5)


class TestCalculateDistance(unittest.TestCase):

    def test_horizontal_distance(self):
        a = (0, 0)
        b = (1, 0)
        self.assertAlmostEqual(calculate_distance(a, b), 1.0, places=5)

    def test_vertical_distance(self):
        a = (0, 0)
        b = (0, 1)
        self.assertAlmostEqual(calculate_distance(a, b), 1.0, places=5)

    def test_diagonal_distance(self):
        a = (0, 0)
        b = (1, 1)
        self.assertAlmostEqual(calculate_distance(a, b), 1.4142135623730951, places=5)

    def test_negative_coordinates(self):
        a = (-1, 0)
        b = (1, 0)
        self.assertAlmostEqual(calculate_distance(a, b), 2.0, places=5)

    def test_zero_distance(self):
        a = (1, 1)
        b = (1, 1)
        self.assertAlmostEqual(calculate_distance(a, b), 0.0, places=5)

    def test_all_negative_coordinates(self):
        a = (-1, -1)
        b = (-2, -2)
        self.assertAlmostEqual(calculate_distance(a, b), 1.4142135623730951, places=5)


if __name__ == '__main__':
    unittest.main()
