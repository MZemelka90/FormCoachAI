import cv2
import numpy as np

COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (128, 128, 128)
}


class SquatDrawings:
    def __init__(self, image: np.ndarray, width: int, height: int):
        self.width = width
        self.height = height
        self.image = image

        # Calculate font scale and line thickness based on image size
        self.font_scale = min(self.width, self.height) / 1500
        self.line_thickness = int(min(self.width, self.height) / 500)

        self.header_text_x = 30
        self.header_text_y = int(self.font_scale * 50)

    def draw_side_angle_squat(self,
                              camera_angle: str,
                              hip: tuple[int, int],
                              knee: tuple[int, int],
                              ankle: tuple[int, int],
                              shoulder: tuple[int, int],
                              feet: tuple[int, int],
                              hip_angle: float,
                              knee_angle: float,
                              shin_angle: float,
                              scale: float
                              ):
        # Draw the image
        cv2.putText(self.image,
                    f"Camera Angle: {camera_angle}",
                    (self.header_text_x, self.header_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    COLORS["white"],
                    self.line_thickness,
                    cv2.LINE_AA)

        marker_lst = [hip, knee, ankle, shoulder, feet]
        for marker in marker_lst:
            cv2.drawMarker(self.image,
                           (int(marker[0] * scale), int(marker[1] * scale)),
                           COLORS["red"],
                           markerType=cv2.MARKER_SQUARE,
                           markerSize=10,
                           thickness=2
                           )

        # Draw lines between the hip, knee, shoulder, and ankle
        cv2.line(self.image,
                 (int(hip[0] * scale), int(hip[1] * scale)),
                 (int(shoulder[0] * scale), int(shoulder[1] * scale)),
                 COLORS["green"], self.line_thickness)

        cv2.line(self.image,
                 (int(knee[0] * scale), int(knee[1] * scale)),
                 (int(hip[0] * scale), int(hip[1] * scale)),
                 COLORS["green"], self.line_thickness)

        cv2.line(self.image,
                 (int(ankle[0] * scale), int(ankle[1] * scale)),
                 (int(knee[0] * scale), int(knee[1] * scale)), COLORS["green"], self.line_thickness)

        cv2.line(self.image,
                 (int(ankle[0] * scale), int(ankle[1] * scale)),
                 (int(feet[0] * scale), int(feet[1] * scale)), COLORS["green"], self.line_thickness)

        joint_lst = [knee, hip, ankle]
        for i, angle in enumerate([knee_angle, hip_angle, shin_angle]):
            cv2.putText(self.image,
                        f"{round(angle, 2)}deg",
                        (int(joint_lst[i][0] * scale + 40),
                         int(joint_lst[i][1] * scale + 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        COLORS["white"],
                        self.line_thickness,
                        cv2.LINE_AA)

    def draw_back_angle_squat(self,
                              camera_angle: str,
                              hips: tuple,
                              shoulders: tuple,
                              hip_angle: float,
                              hip_shift_angle: float
                              ):
        # Draw the image
        cv2.putText(self.image,
                    f"Camera Angle: {camera_angle}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    COLORS["white"],
                    self.line_thickness,
                    cv2.LINE_AA)

        cv2.drawMarker(self.image,
                       (int(hips[0].x * self.width), int(hips[0].y * self.height)),
                       (0, 255, 0),
                       markerType=cv2.MARKER_SQUARE,
                       markerSize=10,
                       thickness=2
                       )
        cv2.drawMarker(self.image,
                       (int(hips[1].x * self.width),
                        int(hips[1].y * self.height)),
                       (0, 255, 0),
                       markerType=cv2.MARKER_SQUARE,
                       markerSize=10,
                       thickness=2
                       )
        cv2.line(self.image,
                 (int(hips[0].x * self.width), int(hips[0].y * self.height)),
                 (int(hips[1].x * self.width), int(hips[1].y * self.height)),
                 (0, 255, 0), self.line_thickness)
        hip_middle = [(int(hips[0].x * self.width) + int(hips[1].x * self.width)) // 2,
                      (int(hips[0].y * self.height) + int(hips[1].y * self.height)) // 2]
        shoulder_middle = [(int(shoulders[0].x * self.width) + int(shoulders[1].x * self.width)) // 2,
                           (int(shoulders[0].y * self.height) + int(shoulders[1].y * self.height)) // 2]

        cv2.drawMarker(self.image, hip_middle, (0, 0, 255), markerType=cv2.MARKER_SQUARE, markerSize=10, thickness=2)
        cv2.drawMarker(self.image, shoulder_middle, (0, 0, 255), markerType=cv2.MARKER_SQUARE, markerSize=10, thickness=2)

        cv2.putText(self.image,
                    f"Hip horiz. Angle: {hip_angle} deg",
                    (hip_middle[0] - 100, hip_middle[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 0), self.line_thickness, cv2.LINE_AA)

        cv2.putText(self.image,
                    f"Hip Shift Angle: {hip_shift_angle - 90} deg",
                    (hip_middle[0] - 100, hip_middle[1] + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 0), self.line_thickness, cv2.LINE_AA)
        cv2.line(self.image, shoulder_middle, hip_middle, (0, 255, 0), 2)

    def draw_bar_path(self, bar_path: list):
        cv2.putText(self.image,
                    "Bar Path",
                    (self.header_text_x, self.header_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (255, 255, 255),
                    self.line_thickness,
                    cv2.LINE_AA)

        for i in range(1, len(bar_path)):
            cv2.line(self.image, bar_path[i - 1], bar_path[i], (255, 255, 255), self.line_thickness)

    def draw_balance(self, shoulder: tuple, feet: tuple, scale: float) -> None:
        cv2.putText(self.image,
                    "Balance Points",
                    (self.header_text_x, self.header_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (255, 255, 255),
                    self.line_thickness,
                    cv2.LINE_AA)

        cv2.drawMarker(self.image,
                       (int(shoulder[0] * scale),
                        int(shoulder[1] * scale)),
                       (0, 0, 255),
                       markerType=cv2.MARKER_SQUARE,
                       markerSize=10,
                       thickness=2
                       )

        cv2.putText(self.image,
                    "Shoulder",
                    (int(shoulder[0] * scale) + 20, int(shoulder[1] * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (255, 255, 255),
                    self.line_thickness,
                    cv2.LINE_AA)

        cv2.drawMarker(self.image,
                       (int(feet[0] * scale),
                        int(feet[1] * scale)),
                       (0, 0, 255),
                       markerType=cv2.MARKER_SQUARE,
                       markerSize=10,
                       thickness=2
                       )
        cv2.putText(self.image,
                    "Feet",
                    (int(feet[0] * scale) + 20, int(feet[1] * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (255, 255, 255),
                    self.line_thickness,
                    cv2.LINE_AA)

        cv2.line(self.image,
                 (int(shoulder[0] * scale), int(shoulder[1] * scale)),
                 (int(shoulder[0] * scale), int(shoulder[1] * scale) + 200),
                 (0, 0, 255),
                 self.line_thickness
                 )
        cv2.line(self.image,
                 (int(feet[0] * scale), int(feet[1] * scale)),
                 (int(feet[0] * scale), int(feet[1] * scale) - 200),
                 (0, 0, 255),
                 self.line_thickness
                 )
