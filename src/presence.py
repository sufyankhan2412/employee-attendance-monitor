"""
presence.py  (Technical Document, sections 11, 12, 13, 14, 24)

Workplace Presence Verification pipeline:
    Workplace CCTV -> Random Frame Selection -> Face Detection -> Face
    Alignment -> SFace Recognition -> Employee Identification ->
    Presence Result (PRESENT / ABSENT / UNKNOWN / EXCEPTION)
"""

import os
import cv2

from .utils import now_compact, ensure_dir, draw_face_box


class PresenceService:
    def __init__(self, detector, recognizer, database, captured_images_dir: str, camera_id: str):
        self.detector = detector
        self.recognizer = recognizer
        self.db = database
        self.captured_images_dir = captured_images_dir
        self.camera_id = camera_id
        ensure_dir(self.captured_images_dir)

    def verify_frame(self, frame, expected_employee_id: str, known_embeddings):
        """
        Run one presence sample: detect + recognize faces in the frame and
        check whether the expected employee is among them.

        Returns: (status, best_score, matched_employee_id, image_path)
            status in {"PRESENT", "ABSENT", "UNKNOWN", "EXCEPTION"}
        """
        faces = self.detector.detect(frame)

        if not faces:
            status = "EXCEPTION"
            matched_id, score = None, 0.0
        else:
            matched_id, score = None, -1.0
            for face in faces:
                embedding = self.recognizer.get_embedding_from_frame(frame, face["raw"])
                emp_id, s = self.recognizer.identify(embedding, known_embeddings)
                label = f"{emp_id or 'UNKNOWN'} ({s:.2f})"
                draw_face_box(frame, face["box"], label,
                              color=(0, 200, 0) if emp_id else (0, 0, 255))
                if emp_id == expected_employee_id:
                    matched_id, score = emp_id, s
                    break
                if s > score:
                    matched_id, score = emp_id, s

            if matched_id == expected_employee_id:
                status = "PRESENT"
            elif matched_id is None:
                status = "UNKNOWN"
            else:
                status = "ABSENT"

        filename = f"presence_{expected_employee_id}_{now_compact()}.jpg"
        image_path = os.path.join(self.captured_images_dir, filename)
        cv2.imwrite(image_path, frame)

        self.db.log_presence_event(
            employee_id=expected_employee_id,
            camera_id=self.camera_id,
            status=status,
            recognition_score=score if score >= 0 else 0.0,
            image_path=image_path,
        )

        return status, score, matched_id, image_path
