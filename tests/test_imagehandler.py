import unittest
import cv2
import numpy as np
from unittest.mock import MagicMock, patch
from src.ImageHandler import FrameHandler


class TestFrameHandler(unittest.TestCase):
    @patch('cv2.VideoCapture')
    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    def setUp(self, mock_resizeWindow, mock_namedWindow, mock_VideoCapture):
        """
        Setup for the FrameHandler tests.
        """
        mock_VideoCapture.return_value.read.return_value = (True, np.zeros((480, 640, 3), np.uint8))
        mock_VideoCapture.return_value.get = MagicMock(side_effect=[640, 480])
        self.frame_handler = FrameHandler("test.mp4", "TestWindow", scale=0.5)

    def test_initialization(self):
        """
        Test initialization of the FrameHandler class.
        """
        self.assertEqual(self.frame_handler.width, int(640 * 0.5))
        self.assertEqual(self.frame_handler.height, int(480 * 0.5))
        self.assertIsNotNone(self.frame_handler.pose)

    def test_get_blank_image_dimensions(self):
        """
        Test blank image dimension calculation.
        """
        blank_height, blank_width = self.frame_handler._get_blank_image_dimensions()
        self.assertEqual(blank_height, self.frame_handler.height // 3)
        self.assertEqual(blank_width, self.frame_handler.width // 3)

    def test_create_blank_image_with_border(self):
        """
        Test creation of a blank image with a border.
        """
        border_size = 5
        blank_image = self.frame_handler._create_blank_image_with_border(border_size)
        expected_height = (self.frame_handler.height // 3)
        expected_width = (self.frame_handler.width // 3)

        self.assertEqual(blank_image.shape[0], expected_height)
        self.assertEqual(blank_image.shape[1], expected_width)
        self.assertTrue(np.all(blank_image[:border_size, :, :] == 255))
        self.assertTrue(np.all(blank_image[-border_size:, :, :] == 255))
        self.assertTrue(np.all(blank_image[:, :border_size, :] == 255))
        self.assertTrue(np.all(blank_image[:, -border_size:, :] == 255))

    @patch('src.ImageHandler.SquatDrawings')
    def test_add_images_to_frame(self, MockSquatDrawings):
        """
        Test adding images to the frame.
        """
        frame = np.zeros((self.frame_handler.height, self.frame_handler.width, 3), np.uint8)
        args = ("Side Angle", [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 45, 90, 30, [(10, 10)])

        MockSquatDrawings.return_value.draw_side_angle_squat = MagicMock()
        MockSquatDrawings.return_value.draw_bar_path = MagicMock()
        MockSquatDrawings.return_value.draw_balance = MagicMock()

        updated_frame = self.frame_handler.add_images_to_frame(frame, args)
        self.assertEqual(updated_frame.shape[1], self.frame_handler.width + self.frame_handler._get_blank_image_dimensions()[1])

    def test_get_position_coordinates(self):
        """
        Test position coordinates calculation.
        """
        image_height = 100
        self.assertEqual(self.frame_handler._get_position_coordinates("Top", image_height), (0, 100))
        self.assertEqual(self.frame_handler._get_position_coordinates("Middle", image_height),
                         (self.frame_handler.height // 2 - image_height // 2,
                          self.frame_handler.height // 2 + image_height // 2))
        self.assertEqual(self.frame_handler._get_position_coordinates("Bottom", image_height),
                         (self.frame_handler.height - image_height, self.frame_handler.height))

    def test_add_positioned_image(self):
        """
        Test adding a positioned image to the frame.
        """
        frame = np.zeros((self.frame_handler.height, self.frame_handler.width + 100, 3), np.uint8)
        image = np.zeros((100, 100, 3), np.uint8)
        position = "Top"
        args = ("Side Angle", [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 45, 90, 30, [(10, 10)])

        y_start, y_end = self.frame_handler._get_position_coordinates(position, image.shape[0])
        x_start, x_end = self.frame_handler.width, self.frame_handler.width + image.shape[1]

        updated_frame = self.frame_handler._add_positioned_image(frame, image, position, args)
        self.assertTrue(np.array_equal(updated_frame[y_start:y_end, x_start:x_end], image))

    def test_draw_squat_info(self):
        """
        Test drawing squat information.
        """
        image = np.zeros((100, 100, 3), np.uint8)
        drawings = MagicMock()
        position = "Top"
        args = ("Side Angle", [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 45, 90, 30, [(10, 10)])

        self.frame_handler._draw_squat_info(drawings, position, args)
        drawings.draw_side_angle_squat.assert_called_once()

if __name__ == '__main__':
    unittest.main()