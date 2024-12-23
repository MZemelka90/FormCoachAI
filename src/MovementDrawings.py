import cv2
from Calculations import calculate_two_point_angle


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
        cv2.putText(self.image, f"Camera Angle: {camera_angle}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)

        # Draw hip marker
        cv2.drawMarker(self.image,
                       (int(hip[0] * self.width), int(hip[1] * self.height)),
                       (0, 255, 0),
                       markerType=cv2.MARKER_SQUARE,
                       markerSize=10,
                       thickness=2
                       )
        # Draw knee marker
        cv2.drawMarker(self.image,
                       (int(knee[0] * self.width),
                        int(knee[1] * self.height)),
                       (0, 255, 0),
                       markerType=cv2.MARKER_SQUARE,
                       markerSize=10,
                       thickness=2
                       )
        # draw ankle marker
        cv2.drawMarker(self.image,
                       (int(ankle[0] * self.width),
                        int(ankle[1] * self.height)),
                       (0, 255, 0),
                       markerType=cv2.MARKER_SQUARE,
                       markerSize=10,
                       thickness=2
                       )
        # draw shoulder marker
        cv2.drawMarker(self.image,
                       (int(shoulder[0] * self.width),
                        int(shoulder[1] * self.height)),
                       (0, 255, 0),
                       markerType=cv2.MARKER_SQUARE,
                       markerSize=10,
                       thickness=2
                       )

        # Draw lines between the hip, knee, shoulder, and ankle
        cv2.line(self.image, (int(hip[0] * self.width), int(hip[1] * self.height)),
                 (int(knee[0] * self.width), int(knee[1] * self.height)), (0, 255, 0), 2)

        cv2.line(self.image, (int(knee[0] * self.width), int(knee[1] * self.height)),
                 (int(ankle[0] * self.width), int(ankle[1] * self.height)), (0, 255, 0), 2)

        cv2.line(self.image, (int(shoulder[0] * self.width), int(shoulder[1] * self.height)),
                 (int(hip[0] * self.width), int(hip[1] * self.height)), (0, 255, 0), 2)

        # Add angle text to the respective markers
        cv2.putText(self.image,
                    f"Knee Angle: {int(knee_angle)} deg",
                    (int(knee[0] * self.width + 10),
                     int(knee[1] * self.height + 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)

        cv2.putText(self.image,
                    f"Hip Angle: {int(hip_angle)} deg",
                    (int(hip[0] * self.width + 10),
                     int(hip[1] * self.height + 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)

        cv2.putText(self.image,
                    f"Shin Angle: {int(shin_angle)} deg",
                    (int(ankle[0] * self.width + 10),
                     int(ankle[1] * self.height + 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
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
