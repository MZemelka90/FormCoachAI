import unittest
import numpy as np
from src.MovementDrawings import SquatDrawings
from collections import namedtuple

# Create a mock Mediapipe landmark structure
Landmark = namedtuple('Landmark', ['x', 'y'])


class TestSquatDrawings(unittest.TestCase):
    def setUp(self):
        # Create a blank test image
        self.width, self.height = 800, 600
        self.test_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.drawer = SquatDrawings(self.test_image, self.width, self.height)

    def test_draw_side_angle_squat(self):
        # Test side angle drawing
        self.drawer.draw_side_angle_squat(
            camera_angle="Side",
            hip=(200, 300),
            knee=(250, 400),
            ankle=(300, 500),
            shoulder=(150, 200),
            feet=(300, 550),
            hip_angle=45.0,
            knee_angle=90.0,
            shin_angle=135.0,
            scale=1.0
        )
        # Validate that the markers and text are drawn
        self.assertFalse(np.array_equal(self.test_image, np.zeros((self.height, self.width, 3), dtype=np.uint8)))

    def test_draw_back_angle_squat(self):
        # Test back angle drawing
        hips = [Landmark(0.3, 0.5), Landmark(0.7, 0.5)]
        shoulders = [Landmark(0.3, 0.3), Landmark(0.7, 0.3)]
        self.drawer.draw_back_angle_squat(
            camera_angle="Back",
            hips=hips,
            shoulders=shoulders,
            hip_angle=30.0,
            hip_shift_angle=60.0
        )
        # Validate that the markers and text are drawn
        self.assertFalse(np.array_equal(self.test_image, np.zeros((self.height, self.width, 3), dtype=np.uint8)))

    def test_draw_bar_path(self):
        # Test bar path drawing
        bar_path = [(100, 100), (150, 150), (200, 200), (250, 250)]
        self.drawer.draw_bar_path(bar_path)
        # Validate that lines are drawn
        self.assertFalse(np.array_equal(self.test_image, np.zeros((self.height, self.width, 3), dtype=np.uint8)))

    def test_draw_balance(self):
        # Test balance drawing
        shoulder = (200, 150)
        feet = (200, 550)
        scale = 1.0
        self.drawer.draw_balance(shoulder, feet, scale)
        # Validate that markers and lines are drawn
        self.assertFalse(np.array_equal(self.test_image, np.zeros((self.height, self.width, 3), dtype=np.uint8)))

if __name__ == "__main__":
    unittest.main()
