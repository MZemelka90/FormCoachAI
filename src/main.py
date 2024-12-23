import numpy as np

import cv2
import mediapipe as mp
from MovementPatterns import SquatPose
from Calculations import calculate_three_point_angle, calculate_two_point_angle, calculate_distance
from MovementDrawings import SquatDrawings

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open video feed (0 for webcam or provide video file path)
# cap = cv2.VideoCapture(r"C:\Users\Lenovo\OneDrive\Dokumente\FormCoachAI\Data\Squat_Test_Side_2.mp4")
cap = cv2.VideoCapture(r"C:\Users\Lenovo\OneDrive\Dokumente\FormCoachAI\Data\Squat_Test_Video_Back.mp4")

# Get the video resolution
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a window with the same resolution as the video
cv2.namedWindow('FormCoachAI', cv2.WINDOW_NORMAL)
cv2.resizeWindow('FormCoachAI', video_width, video_height)

bar_path = []
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
    if not results.pose_landmarks:
        continue

    landmarks = results.pose_landmarks.landmark
    squat_pose = SquatPose(landmarks, video_width, video_height)
    squat_drawings = SquatDrawings(image, video_width, video_height)

    video_angle = squat_pose.check_visibility(0.2)
    if video_angle == "Side Angle":
        filmed_side = squat_pose.check_which_side_is_visible()
        if filmed_side == "Left":
            hip, knee, ankle, shoulder, feet = squat_pose.get_side_coordinates("Left")

        else:
            hip, knee, ankle, shoulder, feet = squat_pose.get_side_coordinates("Right")

        # Calculate the angles
        knee_angle = calculate_three_point_angle(hip, knee, ankle)
        hip_angle = calculate_three_point_angle(shoulder, hip, knee)
        shin_angle = calculate_three_point_angle(knee, ankle, feet)
        squat_drawings.draw_side_angle_squat(video_angle, hip, knee, ankle, shoulder, knee_angle, hip_angle, shin_angle)

        shoulder_middle = squat_pose.coordinates.shoulder_midpoint
        left_foot_midpoint = squat_pose.coordinates.left_foot_midpoint
        right_foot_midpoint = squat_pose.coordinates.right_foot_midpoint
        feet_middle = (left_foot_midpoint[0] + right_foot_midpoint[0]) // 2, (
                left_foot_midpoint[1] + right_foot_midpoint[1]) // 2
        balance_dist = calculate_distance(shoulder_middle, feet_middle)
        cv2.drawMarker(image, shoulder_middle, (0, 0, 255), markerType=cv2.MARKER_SQUARE, markerSize=10, thickness=2)
        cv2.drawMarker(image, feet_middle, (0, 0, 255), markerType=cv2.MARKER_SQUARE, markerSize=10, thickness=2)

        cv2.line(image, shoulder_middle, [shoulder_middle[0], feet_middle[1]], (0, 0, 255), 2)
        cv2.line(image, feet_middle, [feet_middle[0], shoulder_middle[1]], (0, 0, 255), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=120, maxRadius=200)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                distance_to_shoulder = calculate_distance(circle, shoulder_middle)
                if distance_to_shoulder > 150:
                    continue
                center = (circle[0], circle[1])
                bar_path.append(center)
                cv2.circle(image, center, 5, (0, 255, 0), -1)

        # Draw the path
        for i in range(1, len(bar_path)):
            cv2.line(image, bar_path[i - 1], bar_path[i], (255, 0, 0), 2)

    elif video_angle == "Back Angle":
        shoulders = squat_pose.shoulders
        hips = squat_pose.hips
        shoulder_middle = squat_pose.coordinates.shoulder_midpoint
        hip_middle = squat_pose.coordinates.hip_midpoint
        hip_shift_angle = calculate_two_point_angle(shoulder_middle, hip_middle)
        hip_angle = calculate_two_point_angle([squat_pose.coordinates.left_hip[0], squat_pose.coordinates.left_hip[1]],
                                              [squat_pose.coordinates.right_hip[0], squat_pose.coordinates.right_hip[1]])
        squat_drawings.draw_back_angle_squat(video_angle,
                                             shoulders=shoulders,
                                             hips=hips,
                                             hip_shift_angle=hip_shift_angle,
                                             hip_angle=hip_angle
                                             )

    # Display the image
    cv2.imshow('FormCoachAI', image)

    # Exit with 'q'
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
