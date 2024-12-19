import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


@dataclass
class SquatPose:
    """ Dataclass to store relevant squat pose landmarks """
    shoulders: tuple[mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]
    hips: tuple[mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]
    elbows: tuple[mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]
    knees: tuple[mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE]
    ankles: tuple[mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.LEFT_ANKLE]

    def visibility_difference(self, landmarks: Tuple[mp_pose.PoseLandmark, mp_pose.PoseLandmark], threshold: float = 0.2) -> bool:
        """Calculate the difference in visibility between two landmarks"""
        return abs(landmarks[0].visibility - landmarks[1].visibility) < threshold

    def check_visibility(self, threshold: float = 0.2) -> dict:
        """Check visibility differences for all landmarks"""
        return {
            'shoulders': self.visibility_difference(self.shoulders, threshold),
            'hips': self.visibility_difference(self.hips, threshold),
            'elbows': self.visibility_difference(self.elbows, threshold),
            'knees': self.visibility_difference(self.knees, threshold),
            'ankles': self.visibility_difference(self.ankles, threshold),
        }


def determine_angle(landmarks: list[mp_pose.PoseLandmark]) -> str:
    squat_pose = SquatPose(
        shoulders=(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]),
        hips=(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]),
        elbows=(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]),
        knees=(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]),
        ankles=(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
    )
    if all(squat_pose.check_visibility(0.2).values()):
        return "Back Angle"
    else:
        return "Side Angle"


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Open video feed (0 for webcam or provide video file path)
#cap = cv2.VideoCapture(r"C:\Users\Lenovo\OneDrive\Dokumente\FormCoachAI\Data\Squat_Test_Side_2.mp4")
cap = cv2.VideoCapture(r"C:\Users\Lenovo\OneDrive\Dokumente\FormCoachAI\Data\Squat_Test_Video_Back.mp4")


# Get the video resolution
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a window with the same resolution as the video
cv2.namedWindow('FormCoachAI', cv2.WINDOW_NORMAL)
cv2.resizeWindow('FormCoachAI', video_width, video_height)

# Set display resolution
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process pose
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark
        video_angle = determine_angle(landmarks)

        # Display video angle
        cv2.putText(image, f"Angle: {video_angle}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Get coordinates
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculate the angle
        angle = calculate_angle(hip, knee, ankle)

        # Display feedback
        feedback = "Good Form" if 80 <= angle <= 120 else "Adjust Form"
        # Display feedback and angle
        cv2.putText(image, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Knee Angle: {int(angle)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
    except Exception:
        pass

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the image
    cv2.imshow('FormCoachAI', image)

    # Exit with 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
