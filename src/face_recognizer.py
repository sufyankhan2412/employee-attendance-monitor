"""
face_recognizer.py  (Technical Document, sections 4, 5, 20)

Wraps SFace (face_recognition_sface_2021dec.onnx) using OpenCV's built-in
FaceRecognizerSF. Responsibilities:
    - Align the detected face using the 5 facial landmarks (section 4)
    - Generate a face embedding from the aligned face (section 5)
    - Compare embeddings using cosine similarity
    - Return the matching employee identity (or UNKNOWN)
"""

import os
import cv2
import numpy as np


class FaceRecognizer:
    def __init__(self, model_path: str, cosine_threshold: float = 0.363):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SFace model not found at '{model_path}'.\n"
                f"Run 'python download_models.py' first (see README)."
            )

        self.cosine_threshold = cosine_threshold
        self.recognizer = cv2.FaceRecognizerSF.create(model_path, "")

    def align_and_crop(self, frame: np.ndarray, raw_detection_row: np.ndarray) -> np.ndarray:
        """
        Geometric alignment (section 4): uses the 5 landmarks from YuNet's raw
        output row to rotate/crop the face into a normalized 112x112 image,
        exactly as SFace expects.
        """
        return self.recognizer.alignCrop(frame, raw_detection_row)

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """Generate the 128-d SFace embedding from an aligned face crop."""
        feature = self.recognizer.feature(aligned_face)
        return feature.flatten()

    def get_embedding_from_frame(self, frame: np.ndarray, raw_detection_row: np.ndarray) -> np.ndarray:
        """Convenience: align + embed in one call."""
        aligned = self.align_and_crop(frame, raw_detection_row)
        return self.get_embedding(aligned)

    def compare(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """
        Cosine similarity between two embeddings (section 5 / 15).
        Returns a float; higher = more similar. Typical match range ~0.3-1.0.
        """
        a = embedding_a.reshape(1, -1).astype(np.float32)
        b = embedding_b.reshape(1, -1).astype(np.float32)
        return float(self.recognizer.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))

    def identify(self, embedding: np.ndarray, known_embeddings: list):
        """
        Compare one embedding against a list of (employee_id, embedding) pairs
        and return the best match (section 7 / 15).

        Returns: (employee_id_or_None, best_score)
            employee_id is None ("UNKNOWN") if best_score < cosine_threshold.
        """
        best_id = None
        best_score = -1.0

        for employee_id, known_embedding in known_embeddings:
            score = self.compare(embedding, known_embedding)
            if score > best_score:
                best_score = score
                best_id = employee_id

        if best_score >= self.cosine_threshold:
            return best_id, best_score
        return None, best_score
