"""
attendance.py  (Technical Document, sections 8, 9, 10, 23)

Main Gate pipeline:
    Video Frame -> Face Detection -> Face Alignment -> SFace Recognition
    -> Employee Identification -> Attendance Event (ID, time, camera, image, score)

The original document specifies CHECK_OUT only. This build also supports
CHECK_IN at the same gate/camera, auto-toggling per employee per day:
first recognized scan of the day = CHECK_IN, next scan = CHECK_OUT, and
so on. This can also be forced explicitly (see record_event's event_type
argument).
"""

import os
import cv2

from .utils import now_compact, ensure_dir, draw_face_box


class AttendanceService:
    def __init__(self, detector, recognizer, database, captured_images_dir: str, camera_id: str):
        self.detector = detector
        self.recognizer = recognizer
        self.db = database
        self.captured_images_dir = captured_images_dir
        self.camera_id = camera_id
        ensure_dir(self.captured_images_dir)

    def process_frame(self, frame, known_embeddings, draw: bool = True):
        """
        Run the full check-out pipeline on a single frame.
        Returns a list of result dicts (one per face found), and the
        (possibly annotated) frame.
        """
        results = []
        faces = self.detector.detect(frame)

        for face in faces:
            embedding = self.recognizer.get_embedding_from_frame(frame, face["raw"])
            employee_id, score = self.recognizer.identify(embedding, known_embeddings)

            label = f"{employee_id or 'UNKNOWN'} ({score:.2f})"
            if draw:
                draw_face_box(
                    frame,
                    face["box"],
                    label,
                    color=(0, 200, 0) if employee_id else (0, 0, 255),
                )

            results.append(
                {
                    "employee_id": employee_id,
                    "score": score,
                    "box": face["box"],
                    "confidence": face["confidence"],
                }
            )

        return results, frame

    def determine_event_type(self, employee_id: str) -> str:
        """
        Auto-toggle logic: if the employee has no event yet today, this is
        a CHECK_IN. If their last event today was CHECK_IN, this is a
        CHECK_OUT, and vice versa.
        """
        last_type = self.db.get_last_event_type_today(employee_id)
        if last_type is None or last_type == "CHECK_OUT":
            return "CHECK_IN"
        return "CHECK_OUT"

    def record_event(self, frame, employee_id: str, score: float, event_type: str = None):
        """
        Persist an attendance event (CHECK_IN or CHECK_OUT): save the
        captured frame to disk and insert a row into attendance_events
        (section 9 / 10).

        If event_type is not given, it's auto-determined per employee/day
        via determine_event_type().
        """
        if event_type is None:
            event_type = self.determine_event_type(employee_id)

        prefix = "checkin" if event_type == "CHECK_IN" else "checkout"
        filename = f"{prefix}_{employee_id}_{now_compact()}.jpg"
        image_path = os.path.join(self.captured_images_dir, filename)
        cv2.imwrite(image_path, frame)

        self.db.log_attendance_event(
            employee_id=employee_id,
            camera_id=self.camera_id,
            event_type=event_type,
            recognition_score=score,
            image_path=image_path,
        )
        return event_type, image_path

    def record_check_out(self, frame, employee_id: str, score: float):
        """Kept for backward compatibility: forces a CHECK_OUT event."""
        _, image_path = self.record_event(frame, employee_id, score, event_type="CHECK_OUT")
        return image_path

    def record_check_in(self, frame, employee_id: str, score: float):
        """Forces a CHECK_IN event."""
        _, image_path = self.record_event(frame, employee_id, score, event_type="CHECK_IN")
        return image_path