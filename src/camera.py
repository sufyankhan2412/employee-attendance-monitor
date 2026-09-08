"""
camera.py  (Technical Document, section 22)

Thin wrapper around cv2.VideoCapture.

For local testing, `source=0` opens your laptop's built-in webcam.
Later, to connect real CCTV, replace `source` in config.yaml with the
camera's RTSP/HTTP stream URL, e.g.:
    "rtsp://user:password@192.168.1.50:554/stream1"
No other code needs to change.
"""

import cv2


class Camera:
    def __init__(self, source, camera_id: str = "CAM"):
        self.camera_id = camera_id
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera source '{source}' for camera '{camera_id}'.\n"
                f"If this is your laptop webcam, make sure no other app is using it "
                f"and that this program has camera permission."
            )

    def read_frame(self):
        """Returns a single BGR frame, or None if the read failed."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
