import unittest
from unittest.mock import MagicMock
from src.MovementPatterns import JointCoordinates, SquatPose
import mediapipe as mp


class TestJointCoordinates(unittest.TestCase):
    def setUp(self):
        self.mock_landmarks = {
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value: MagicMock(x=0.1, y=0.2, visibility=0.9),
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value: MagicMock(x=0.3, y=0.2, visibility=0.8),
            mp.solutions.pose.PoseLandmark.LEFT_HIP.value: MagicMock(x=0.1, y=0.5, visibility=0.7),
            mp.solutions.pose.PoseLandmark.RIGHT_HIP.value: MagicMock(x=0.3, y=0.5, visibility=0.6),
            mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value: MagicMock(x=0.15, y=0.9, visibility=0.8),
            mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value: MagicMock(x=0.35, y=0.9, visibility=0.7),
            mp.solutions.pose.PoseLandmark.LEFT_HEEL.value: MagicMock(x=0.15, y=0.85, visibility=0.9),
            mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value: MagicMock(x=0.35, y=0.85, visibility=0.9),
        }
        self.video_width = 640
        self.video_height = 480

    def test_coordinates_initialization(self):
        coordinates = JointCoordinates(
            left_shoulder=self.mock_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
            right_shoulder=self.mock_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
            left_hip=self.mock_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
            right_hip=self.mock_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
            left_elbow=self.mock_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
            right_elbow=self.mock_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
            left_knee=self.mock_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value],
            right_knee=self.mock_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
            left_ankle=self.mock_landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value],
            right_ankle=self.mock_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value],
            left_foot=self.mock_landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value],
            right_foot=self.mock_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value],
            left_heel=self.mock_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL.value],
            right_heel=self.mock_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value],
            video_width=self.video_width,
            video_height=self.video_height,
        )
        self.assertEqual(coordinates.left_shoulder, [64, 96])
        self.assertEqual(coordinates.right_shoulder, [192, 96])
        self.assertEqual(coordinates.shoulder_midpoint, [128, 96])
        self.assertEqual(coordinates.hip_midpoint, [128, 240])


class TestSquatPose(unittest.TestCase):
    def setUp(self):
        self.mock_landmarks = {
            i: MagicMock(x=0.1 * i, y=0.2 * i, visibility=0.8 - 0.05 * i)
            for i in range(33)  # Simulating all 33 PoseLandmarks
        }
        self.video_width = 640
        self.video_height = 480
        self.pose = SquatPose(self.mock_landmarks, self.video_width, self.video_height)

    def test_visibility_difference(self):
        left_shoulder = self.mock_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = self.mock_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        self.assertTrue(SquatPose.visibility_difference((left_shoulder, right_shoulder)))

    def test_check_visibility(self):
        angle = self.pose.check_visibility()
        self.assertIn(angle, ["Back Angle", "Side Angle"])

    def test_check_which_side_is_visible(self):
        side = self.pose.check_which_side_is_visible()
        self.assertIn(side, ["Left", "Right"])

    def test_get_side_coordinates(self):
        side = "Left"
        coords = self.pose.get_side_coordinates(side)
        self.assertEqual(len(coords), 6)


if __name__ == "__main__":
    unittest.main()
