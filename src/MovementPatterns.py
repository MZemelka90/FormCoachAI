from typing import Tuple, List
import mediapipe as mp
from dataclasses import dataclass

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose


@dataclass
class JointCoordinates:
    left_shoulder: List
    right_shoulder: List
    left_hip: List
    right_hip: List
    left_elbow: List
    right_elbow: List
    left_knee: List
    right_knee: List
    left_ankle: List
    right_ankle: List
    left_foot: List
    right_foot: List
    left_heel: List
    right_heel: List
    video_width: int
    video_height: int

    @property
    def shoulder_midpoint(self):
        return [int((self.left_shoulder[0] + self.right_shoulder[0]) / 2),
                int((self.left_shoulder[1] + self.right_shoulder[1]) / 2)]

    @property
    def hip_midpoint(self):
        return [int((self.left_hip[0] + self.right_hip[0]) / 2),
                int((self.left_hip[1] + self.right_hip[1]) / 2)]

    @property
    def left_foot_midpoint(self):
        return [int((self.left_foot[0] + self.left_heel[0]) / 2),
                int((self.left_foot[1] + self.left_heel[1]) / 2)]

    @property
    def right_foot_midpoint(self):
        return [int((self.right_foot[0] + self.right_heel[0]) / 2),
                int((self.right_foot[1] + self.right_heel[1]) / 2)]

    def __post_init__(self):
        self.left_shoulder = [int(self.left_shoulder.x * self.video_width),
                              int(self.left_shoulder.y * self.video_height)]
        self.right_shoulder = [int(self.right_shoulder.x * self.video_width),
                               int(self.right_shoulder.y * self.video_height)]
        self.left_hip = [int(self.left_hip.x * self.video_width),
                         int(self.left_hip.y * self.video_height)]
        self.right_hip = [int(self.right_hip.x * self.video_width),
                          int(self.right_hip.y * self.video_height)]
        self.left_elbow = [int(self.left_elbow.x * self.video_width),
                           int(self.left_elbow.y * self.video_height)]
        self.right_elbow = [int(self.right_elbow.x * self.video_width),
                            int(self.right_elbow.y * self.video_height)]
        self.left_knee = [int(self.left_knee.x * self.video_width),
                          int(self.left_knee.y * self.video_height)]
        self.right_knee = [int(self.right_knee.x * self.video_width),
                           int(self.right_knee.y * self.video_height)]
        self.left_ankle = [int(self.left_ankle.x * self.video_width),
                           int(self.left_ankle.y * self.video_height)]
        self.right_ankle = [int(self.right_ankle.x * self.video_width),
                            int(self.right_ankle.y * self.video_height)]
        self.left_foot = [int(self.left_foot.x * self.video_width),
                          int(self.left_foot.y * self.video_height)]
        self.right_foot = [int(self.right_foot.x * self.video_width),
                           int(self.right_foot.y * self.video_height)]
        self.left_heel = [int(self.left_heel.x * self.video_width),
                          int(self.left_heel.y * self.video_height)]
        self.right_heel = [int(self.right_heel.x * self.video_width),
                           int(self.right_heel.y * self.video_height)]


class SquatPose:
    """
    Class to represent a squat pose.
    """

    def __init__(self, landmarks, video_width: int, video_height: int):
        self.landmarks = landmarks

        self.shoulders = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        self.hips = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
        self.elbows = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        self.knees = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        self.ankles = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        self.feet = (landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value],
                     landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])
        self.heels = (landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value],
                      landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value])

        self.video_width = video_width
        self.video_height = video_height

        self.coordinates = JointCoordinates(self.shoulders[0], self.shoulders[1],
                                            self.hips[0], self.hips[1],
                                            self.elbows[0], self.elbows[1],
                                            self.knees[0], self.knees[1],
                                            self.ankles[0], self.ankles[1],
                                            self.feet[0], self.feet[1],
                                            self.heels[0], self.heels[1],
                                            self.video_width,
                                            self.video_height)

    @staticmethod
    def visibility_difference(relevant_landmarks: Tuple[mp_pose.PoseLandmark, mp_pose.PoseLandmark],
                              threshold: float = 0.2) -> bool:
        """Calculate the difference in visibility between two landmarks"""
        return abs(relevant_landmarks[0].visibility - relevant_landmarks[1].visibility) < threshold

    def check_visibility(self, threshold: float = 0.2) -> str:
        """Check visibility differences for all landmarks"""
        visibility = {
            'shoulders': self.visibility_difference(self.shoulders, threshold),
            'hips': self.visibility_difference(self.hips, threshold),
            'elbows': self.visibility_difference(self.elbows, threshold),
            'knees': self.visibility_difference(self.knees, threshold),
            'ankles': self.visibility_difference(self.ankles, threshold),
        }
        if all(visibility.values()):
            return "Back Angle"
        else:
            return "Side Angle"

    def check_which_side_is_visible(self) -> str:
        """Check which side is visible"""
        right_side_joints = [
            self.landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            self.landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            self.landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            self.landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            self.landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        ]
        left_side_joints = [
            self.landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            self.landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            self.landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            self.landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            self.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        ]
        right_side_visibility = sum(joint.visibility for joint in right_side_joints)
        left_side_visibility = sum(joint.visibility for joint in left_side_joints)

        if right_side_visibility > left_side_visibility:
            return "Right"
        elif left_side_visibility > right_side_visibility:
            return "Left"
        else:
            raise ValueError("Cannot determine which side is visible")

    def get_side_coordinates(self, side: str) -> tuple:
        """
        Get the coordinates of the hip, knee, and ankle for a specific side

        Args:
            side (str): The side to get the coordinates for

        Returns:
            tuple: The coordinates of the hip, knee, and ankle
        """
        if side == "Left":
            index = 0
        elif side == "Right":
            index = 1
        else:
            raise ValueError("Invalid side")

        return ([getattr(self.hips[index], coord) for coord in ['x', 'y']],
                [getattr(self.knees[index], coord) for coord in ['x', 'y']],
                [getattr(self.ankles[index], coord) for coord in ['x', 'y']],
                [getattr(self.shoulders[index], coord) for coord in ['x', 'y']],
                [getattr(self.feet[index], coord) for coord in ['x', 'y']])
