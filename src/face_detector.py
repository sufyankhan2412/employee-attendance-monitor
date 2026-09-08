"""
face_detector.py  (Technical Document, section 19)

Wraps YuNet (face_detection_yunet_2023mar.onnx) using OpenCV's built-in
FaceDetectorYN. Responsibilities, per the design doc:
    - Load YuNet
    - Receive image/frame
    - Detect faces
    - Return bounding boxes
    - Return landmarks (left eye, right eye, nose, left mouth, right mouth)
    - Return confidence score
"""

import os
import cv2
import numpy as np


class FaceDetector:
    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        input_size=(320, 320),
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YuNet model not found at '{model_path}'.\n"
                f"Run 'python download_models.py' first (see README)."
            )

        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        # cv2.FaceDetectorYN_create is available in opencv-contrib-python >= 4.5.4
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            tuple(input_size),
            score_threshold,
            nms_threshold,
            top_k,
        )

    def detect(self, frame: np.ndarray):
        """
        Run detection on a single BGR frame.

        Returns a list of dicts, one per detected face:
            {
                "box": (x, y, w, h),
                "landmarks": {
                    "right_eye": (x, y),
                    "left_eye": (x, y),
                    "nose": (x, y),
                    "right_mouth": (x, y),
                    "left_mouth": (x, y),
                },
                "confidence": float,
                "raw": np.ndarray   # the raw 15-value row (needed for alignment/recognition)
            }
        """
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(frame)

        results = []
        if faces is None:
            return results

        for row in faces:
            x, y, bw, bh = row[0:4]
            # YuNet's raw output order (per OpenCV Zoo docs):
            # [x, y, w, h, right_eye_x, right_eye_y, left_eye_x, left_eye_y,
            #  nose_x, nose_y, right_mouth_x, right_mouth_y,
            #  left_mouth_x, left_mouth_y, score]
            landmarks = {
                "right_eye": (row[4], row[5]),
                "left_eye": (row[6], row[7]),
                "nose": (row[8], row[9]),
                "right_mouth": (row[10], row[11]),
                "left_mouth": (row[12], row[13]),
            }
            confidence = float(row[14])

            results.append(
                {
                    "box": (x, y, bw, bh),
                    "landmarks": landmarks,
                    "confidence": confidence,
                    "raw": row,
                }
            )
        return results
