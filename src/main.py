import cv2
import mediapipe as mp
from MovementPatterns import SquatPose
from Calculations import calculate_three_point_angle, calculate_two_point_angle
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
    squat_pose = SquatPose(landmarks)
    squat_drawings = SquatDrawings(image, video_width, video_height)

    video_angle = squat_pose.check_visibility(0.2)
    if video_angle == "Side Angle":
        filmed_side = squat_pose.check_which_side_is_visible()
        if filmed_side == "Left":
            connections = squat_pose.left_side_connections()
            hip, knee, ankle, shoulder, feet = squat_pose.get_side_coordinates("Left")

        else:
            connections = squat_pose.right_side_connections()
            hip, knee, ankle, shoulder, feet = squat_pose.get_side_coordinates("Right")

        # Calculate the angles
        knee_angle = calculate_three_point_angle(hip, knee, ankle)
        hip_angle = calculate_three_point_angle(shoulder, hip, knee)
        shin_angle = calculate_three_point_angle(knee, ankle, feet)
        squat_drawings.draw_side_angle_squat(video_angle, hip, knee, ankle, shoulder, knee_angle, hip_angle, shin_angle)

    elif video_angle == "Back Angle":
        shoulders = squat_pose.shoulders
        hips = squat_pose.hips
        squat_drawings.draw_back_angle_squat(video_angle, shoulders=shoulders, hips=hips)

    # Display the image
    cv2.imshow('FormCoachAI', image)

    # Exit with 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
