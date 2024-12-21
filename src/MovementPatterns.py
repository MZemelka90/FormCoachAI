from typing import Tuple, List
import mediapipe as mp

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose


class SquatPose:
    """
    Class to represent a squat pose.
    """
    def __init__(self, landmarks):
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

    @staticmethod
    def right_side_connections() -> List[Tuple[mp_pose.PoseLandmark, mp_pose.PoseLandmark]]:
        """Get the right side connections"""
        return [
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL),
            (mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
        ]

    @staticmethod
    def left_side_connections() -> List[Tuple[mp_pose.PoseLandmark, mp_pose.PoseLandmark]]:
        """Get the left side connections"""
        return [
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL),
            (mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
        ]

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
