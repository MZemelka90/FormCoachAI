import cv2

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
    def __init__(self, image: cv2.Mat, width: int, height: int):
        self.width = width
        self.height = height
        self.image = image

    def draw_side_angle_squat(self,
                              camera_angle: str,
                              hip: tuple[int, int],
                              knee: tuple[int, int],
                              ankle: tuple[int, int],
                              shoulder: tuple[int, int],
                              hip_angle: float,
                              knee_angle: float,
                              shin_angle: float
                              ):
        # Draw the image
        cv2.putText(self.image,
                    f"Camera Angle: {camera_angle}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    COLORS["blue"],
                    2,
                    cv2.LINE_AA)

        marker_lst = [hip, knee, ankle, shoulder]
        for marker in marker_lst:
            cv2.drawMarker(self.image,
                           (marker[0], marker[1]),
                           COLORS["red"],
                           markerType=cv2.MARKER_SQUARE,
                           markerSize=10,
                           thickness=2
                           )

        # Draw lines between the hip, knee, shoulder, and ankle
        cv2.line(self.image, (hip[0], hip[1]), (knee[0], knee[1]), COLORS["green"], 2)

        cv2.line(self.image, (knee[0], knee[1]), (ankle[0], ankle[1]), COLORS["green"], 2)

        cv2.line(self.image, (shoulder[0], shoulder[1]), (hip[0], hip[1]), COLORS["green"], 2)

        joint_lst = [knee, hip, ankle]
        for i, angle in enumerate([knee_angle, hip_angle, shin_angle]):
            cv2.putText(self.image,
                        f"{round(angle, 2)}deg",
                        (joint_lst[i][0] + 10,
                         joint_lst[i][1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        COLORS["green"],
                        2,
                        cv2.LINE_AA)

    def draw_back_angle_squat(self,
                              camera_angle: str,
                              hips: tuple,
                              shoulders: tuple,
                              hip_angle: float,
                              hip_shift_angle: float
                              ):
        cv2.putText(self.image,
                    f"Camera Angle: {camera_angle}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                    )
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
                 (0, 255, 0), 2)
        hip_middle = [(int(hips[0].x * self.width) + int(hips[1].x * self.width)) // 2,
                      (int(hips[0].y * self.height) + int(hips[1].y * self.height)) // 2]
        shoulder_middle = [(int(shoulders[0].x * self.width) + int(shoulders[1].x * self.width)) // 2,
                           (int(shoulders[0].y * self.height) + int(shoulders[1].y * self.height)) // 2]

        cv2.drawMarker(self.image, hip_middle, (0, 0, 255), markerType=cv2.MARKER_SQUARE, markerSize=10, thickness=2)
        cv2.drawMarker(self.image, shoulder_middle, (0, 0, 255), markerType=cv2.MARKER_SQUARE, markerSize=10, thickness=2)

        cv2.putText(self.image,
                    f"Hip horiz. Angle: {hip_angle} deg",
                    (hip_middle[0] - 100, hip_middle[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(self.image,
                    f"Hip Shift Angle: {hip_shift_angle - 90} deg",
                    (hip_middle[0] - 100, hip_middle[1] + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(self.image, shoulder_middle, hip_middle, (0, 255, 0), 2)
