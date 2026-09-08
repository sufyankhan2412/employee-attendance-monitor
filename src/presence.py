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

    def capture_group_presence(self, frame, known_embeddings, roster=None, sample_time=None):
        """
        Multi-employee version of verify_frame, for a workspace shared by
        several people (Technical Document sections 11-14, extended for
        LAMS Phase 2 group workplaces).

        Saves ONE screenshot of the whole workspace and logs ONE
        presence_events row per employee actually recognized in it.
        If `roster` (a list of employee_ids expected at this camera) is
        given, any roster employee who was NOT seen in the frame gets an
        explicit ABSENT row too, so "nobody showed up" is visible in the
        DB rather than just being a missing row.

        Returns: (image_path, list of {employee_id, status, score, box})
        """
        faces = self.detector.detect(frame)
        recognized = []  # employee_ids actually seen this sample

        results = []
        for face in faces:
            embedding = self.recognizer.get_embedding_from_frame(frame, face["raw"])
            employee_id, score = self.recognizer.identify(embedding, known_embeddings)
            label = f"{employee_id or 'UNKNOWN'} ({score:.2f})"
            draw_face_box(frame, face["box"], label,
                          color=(0, 200, 0) if employee_id else (0, 0, 255))

            if employee_id:
                recognized.append(employee_id)
                results.append({"employee_id": employee_id, "status": "PRESENT",
                                 "score": score, "box": face["box"]})
            else:
                # A face was detected but didn't match anyone enrolled -
                # kept as UNKNOWN so you can review image quality/enrollment.
                results.append({"employee_id": None, "status": "UNKNOWN",
                                 "score": score, "box": face["box"]})

        if not faces:
            results.append({"employee_id": None, "status": "EXCEPTION", "score": 0.0, "box": None})

        filename = f"presence_{self.camera_id}_{now_compact()}.jpg"
        image_path = os.path.join(self.captured_images_dir, filename)
        cv2.imwrite(image_path, frame)

        for r in results:
            self.db.log_presence_event(
                employee_id=r["employee_id"] or "UNKNOWN",
                camera_id=self.camera_id,
                status=r["status"],
                recognition_score=max(r["score"], 0.0),
                image_path=image_path,
                sample_time=sample_time,
            )

        if roster:
            for expected_id in roster:
                if expected_id not in recognized:
                    self.db.log_presence_event(
                        employee_id=expected_id,
                        camera_id=self.camera_id,
                        status="ABSENT",
                        recognition_score=0.0,
                        image_path=image_path,
                        sample_time=sample_time,
                    )

        return image_path, results