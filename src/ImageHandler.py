import cv2
import numpy as np
import mediapipe as mp
from src.MovementPatterns import SquatPose
from src.MovementDrawings import SquatDrawings
from src.Calculations import calculate_three_point_angle, calculate_two_point_angle


class FrameHandler:
    def __init__(self, file_path: str, window_name: str, scale: float = 1):
        """
        Initialize the FrameHandler class.
        :param file_path: The path to the video file
        :param window_name: The name of the window
        :param scale: The scale of the image
        """
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        self.scale = scale

        # Video dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

        # Initialize window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width, self.height)

        # Initialize Mediapipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def add_images_to_frame(self, frame: np.ndarray, args: tuple) -> np.ndarray:
        """
        Adds three vertically stacked blank images with information to the right of the frame.

        :param frame: The original frame.
        :param args: Additional arguments for squat analysis.
        :return: Updated frame with additional images.
        """
        new_width = self.width + self._get_blank_image_dimensions()[1]
        new_frame = np.zeros((self.height, new_width, 3), np.uint8)
        new_frame[:, :self.width] = frame

        positions = ["Top", "Middle", "Bottom"]
        for position in positions:
            blank_image = self._create_blank_image_with_border(1)
            new_frame = self._add_positioned_image(new_frame, blank_image, position, args)

        return new_frame

    def _create_blank_image_with_border(self, border_size: int) -> np.ndarray:
        """
        Creates a blank image with a white border.

        :param border_size: Width of the white border.
        :return: Blank image with border.
        """
        blank_height, blank_width = self._get_blank_image_dimensions()
        blank_image = np.zeros((blank_height - 2 * border_size, blank_width - 2 * border_size, 3), np.uint8)

        return cv2.copyMakeBorder(
            blank_image,
            top=border_size, bottom=border_size,
            left=border_size, right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)  # White border
        )

    def _get_blank_image_dimensions(self) -> tuple:
        """
        Computes dimensions for blank images based on frame dimensions.

        :return: Dimensions (height, width) of blank image.
        """
        return self.height // 3, self.width // 3

    def _add_positioned_image(self, frame: np.ndarray, image: np.ndarray, position: str, args: tuple) -> np.ndarray:
        """
        Adds a blank image with information at a specific position.

        :param frame: Original frame.
        :param image: Image to add.
        :param position: Position (Top, Middle, Bottom).
        :param args: Additional arguments for squat analysis.
        :return: Updated frame.
        """
        y_start, y_end = self._get_position_coordinates(position, image.shape[0])
        x_start, x_end = self.width, self.width + image.shape[1]

        squat_drawings = SquatDrawings(image=image, height=self.height, width=self.width)
        self._draw_squat_info(squat_drawings, position, args)

        frame[y_start:y_end, x_start:x_end] = image
        return frame

    def _get_position_coordinates(self, position: str, image_height: int) -> tuple:
        """
        Computes Y-coordinates for a specific position.

        :param position: Position (Top, Middle, Bottom).
        :param image_height: Height of the image.
        :return: Y-coordinate range.
        """
        match position:
            case "Top":
                return 0, image_height
            case "Middle":
                y_start = (self.height - image_height) // 2
                return y_start, y_start + image_height
            case "Bottom":
                return self.height - image_height, self.height

    def _draw_squat_info(self, drawings: SquatDrawings, position: str, args: tuple):
        """
        Draws squat information on the provided blank image.

        :param drawings: SquatDrawings instance.
        :param position: Position (Top, Middle, Bottom).
        :param args: Additional arguments for drawing.
        """
        match position:
            case "Top":
                drawings.draw_side_angle_squat(
                    camera_angle=args[0], hip=args[1], knee=args[2], ankle=args[3],
                    shoulder=args[4], feet=args[5], knee_angle=args[6],
                    hip_angle=args[7], shin_angle=args[8], scale=0.3
                )
            case "Middle":
                drawings.draw_bar_path(args[9])
            case "Bottom":
                drawings.draw_balance(args[4], args[5], scale=0.3)

    def get_barbell_coordinates(self, frame: np.ndarray) -> tuple:
        """
        Identify and return the coordinates of the barbell by detecting the weight plates.

        This method uses circle detection (HoughCircles) to identify the weight plates,
        which are assumed to be circular objects in the frame.

        :return: A tuple (x, y) representing the coordinates of the barbell's center, or None if not found.
        """
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        blurred_frame = cv2.GaussianBlur(gray_frame, (9, 9), 2)

        # Use HoughCircles to detect circular objects (weight plates)
        circles = cv2.HoughCircles(
            blurred_frame,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=160,
            maxRadius=180
        )

        if circles is not None:
            # Convert circle parameters to integers
            circles = np.round(circles[0, :]).astype(int)
            cv2.circle(frame, (circles[0][0], circles[0][1]), circles[0][2], (0, 255, 0), 2)

            barbell_coords = (circles[0][0], circles[0][1])
            return barbell_coords

        # If no weight plates are detected, return None
        return None

    def run_video_analysis(self):
        """
        Runs video analysis and displays processed frames.
        """
        bar_path = []

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (self.width, self.height))
            results = self.pose.process(frame)

            if not results.pose_landmarks:
                continue
            landmarks = results.pose_landmarks.landmark
            squat_pose = SquatPose(landmarks, self.width, self.height)
            video_angle = squat_pose.check_visibility(0.2)

            if video_angle == "Side Angle":
                filmed_side = squat_pose.check_which_side_is_visible()
                side_coords = squat_pose.get_side_coordinates(filmed_side)

                knee_angle = calculate_three_point_angle(side_coords[0], side_coords[1], side_coords[2])
                hip_angle = calculate_three_point_angle(side_coords[3], side_coords[0], side_coords[1])
                shin_angle = calculate_three_point_angle(side_coords[1], side_coords[2], side_coords[4])

                # bar_path.append((int(side_coords[3][0] * 0.3), int(side_coords[3][1] * 0.3)))
                bar_coords = self.get_barbell_coordinates(frame)
                if bar_coords:
                    cv2.circle(frame, (int(bar_coords[0] / 3), int(bar_coords[1] / 3)), 10, (0, 0, 255), -1)
                    bar_path.append((int(bar_coords[0] / 3), int(bar_coords[1] / 3)))
                args = (
                    video_angle, side_coords[0], side_coords[1], side_coords[2],
                    side_coords[3], side_coords[4], knee_angle,
                    hip_angle, shin_angle, bar_path
                )

                frame = self.add_images_to_frame(frame, args)

            elif video_angle == "Back Angle":
                squat_drawings = SquatDrawings(image=frame, height=self.height, width=self.width)
                hips = squat_pose.hips
                shoulders = squat_pose.shoulders
                hip_angle = calculate_two_point_angle(
                    squat_pose.coordinates.left_hip,
                    squat_pose.coordinates.right_hip
                )
                hip_shift_angle = calculate_two_point_angle(
                    squat_pose.get_shoulder_midpoint(),
                    squat_pose.get_hip_midpoint()
                )
                squat_drawings.draw_back_angle_squat(
                    camera_angle=video_angle,
                    hips=hips,
                    shoulders=shoulders,
                    hip_angle=hip_angle,
                    hip_shift_angle=hip_shift_angle
                )

            cv2.imshow("FormCoachAI", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
