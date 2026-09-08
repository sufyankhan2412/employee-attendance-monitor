"""
main.py

LAMS Face Recognition System - Phase 1 (local testing build)

Interactive CLI so you can try every part of the pipeline described in the
technical document using just your laptop webcam:
    1) Enroll an employee (capture photos with your webcam)
    2) Enroll an employee from an existing folder of photos
    3) Live recognition preview (sanity check that detection+recognition work)
    4) Simulate Main Gate (Check-In / Check-Out)
    5) Simulate Workplace Presence Verification
    6) View employees / recent events
    0) Exit

Run with:  python main.py
(from the project root, i.e. the 'lams' folder, with your venv activated)
"""

import os
import sys
import cv2

from src.utils import load_config, ensure_dir, build_database_url
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.face_database import FaceDatabase
from src.camera import Camera
from src.enrollment import enroll_from_folder
from src.attendance import AttendanceService
from src.presence import PresenceService


class LAMSApp:
    def __init__(self, config_path="config/config.yaml"):
        self.config = load_config(config_path)

        detector_cfg = self.config["detection"]
        self.detector = FaceDetector(
            model_path=self.config["models"]["detector_path"],
            score_threshold=detector_cfg["score_threshold"],
            nms_threshold=detector_cfg["nms_threshold"],
            top_k=detector_cfg["top_k"],
            input_size=detector_cfg["input_size"],
        )

        self.recognizer = FaceRecognizer(
            model_path=self.config["models"]["recognizer_path"],
            cosine_threshold=self.config["recognition"]["cosine_threshold"],
        )

        self.db = FaceDatabase(build_database_url(self.config))

        # Register cameras from config into the DB (idempotent).
        gate = self.config["cameras"]["main_gate"]
        self.db.upsert_camera(gate["camera_id"], camera_name="Main Gate", camera_type="GATE")
        for wp in self.config["cameras"]["workplaces"]:
            self.db.upsert_camera(wp["camera_id"], camera_name=wp["camera_id"], camera_type="WORKPLACE")

        self.dataset_dir = self.config["paths"]["dataset_dir"]
        self.captured_dir = self.config["paths"]["captured_images_dir"]
        ensure_dir(self.dataset_dir)
        ensure_dir(self.captured_dir)

    # ------------------------------------------------------------------ #
    # 1) Enroll via webcam
    # ------------------------------------------------------------------ #
    def enroll_via_webcam(self):
        employee_id = input("Employee ID (e.g. EMP001): ").strip()
        employee_name = input("Employee name: ").strip()
        department = input("Department (optional): ").strip()
        n_images = self.config["enrollment"]["images_per_employee"]

        employee_dir = os.path.join(self.dataset_dir, employee_id)
        ensure_dir(employee_dir)

        print(f"\nOpening webcam. A window will appear.")
        print(f"Press SPACE to capture a photo ({n_images} needed). Press ESC to cancel.\n")

        cam_source = self.config["cameras"]["main_gate"]["source"]
        with Camera(cam_source, camera_id="ENROLL_CAM") as cam:
            captured = 0
            while captured < n_images:
                frame = cam.read_frame()
                if frame is None:
                    print("Could not read from webcam.")
                    break

                preview = frame.copy()
                faces = self.detector.detect(preview)
                for face in faces:
                    x, y, w, h = [int(v) for v in face["box"]]
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(preview, f"Captured: {captured}/{n_images}  (SPACE=capture, ESC=cancel)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("LAMS Enrollment - " + employee_id, preview)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("Cancelled.")
                    break
                elif key == 32:  # SPACE
                    if not faces:
                        print("  No face detected - try again.")
                        continue
                    captured += 1
                    img_path = os.path.join(employee_dir, f"{captured:02d}.jpg")
                    cv2.imwrite(img_path, frame)
                    print(f"  Captured {img_path}")

            cv2.destroyAllWindows()

        if captured == 0:
            print("No images captured. Enrollment aborted.")
            return

        print(f"\nGenerating embeddings from {captured} captured photo(s)...")
        used, skipped = enroll_from_folder(
            self.detector, self.recognizer, self.db,
            self.dataset_dir, employee_id, employee_name, department,
        )
        print(f"Done. {used} embedding(s) stored, {skipped} image(s) skipped.")

    # ------------------------------------------------------------------ #
    # 2) Enroll from existing folder
    # ------------------------------------------------------------------ #
    def enroll_from_existing_folder(self):
        employee_id = input("Employee ID (folder name under dataset/, e.g. EMP001): ").strip()
        employee_name = input("Employee name: ").strip()
        department = input("Department (optional): ").strip()

        try:
            used, skipped = enroll_from_folder(
                self.detector, self.recognizer, self.db,
                self.dataset_dir, employee_id, employee_name, department,
            )
            print(f"\nDone. {used} embedding(s) stored, {skipped} image(s) skipped.")
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print(f"Create the folder 'dataset/{employee_id}/' and put a few clear, "
                  f"front-facing .jpg photos of the employee in it, then try again.")

    # ------------------------------------------------------------------ #
    # 3) Live recognition preview
    # ------------------------------------------------------------------ #
    def live_preview(self):
        known_embeddings = self.db.get_all_embeddings()
        if not known_embeddings:
            print("No employees enrolled yet - recognition will show everyone as UNKNOWN.")

        cam_source = self.config["cameras"]["main_gate"]["source"]
        print("\nOpening webcam for live preview. Press ESC or 'q' to quit.\n")
        with Camera(cam_source, camera_id="PREVIEW_CAM") as cam:
            while True:
                frame = cam.read_frame()
                if frame is None:
                    break

                faces = self.detector.detect(frame)
                for face in faces:
                    embedding = self.recognizer.get_embedding_from_frame(frame, face["raw"])
                    employee_id, score = self.recognizer.identify(embedding, known_embeddings)
                    x, y, w, h = [int(v) for v in face["box"]]
                    label = f"{employee_id or 'UNKNOWN'} ({score:.2f})"
                    color = (0, 200, 0) if employee_id else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, max(0, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.imshow("LAMS Live Preview", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    # 4) Simulate main gate (check-in / check-out)
    # ------------------------------------------------------------------ #
    def simulate_checkout(self):
        known_embeddings = self.db.get_all_embeddings()
        if not known_embeddings:
            print("No employees enrolled yet. Enroll someone first (menu option 1 or 2).")
            return

        gate_cfg = self.config["cameras"]["main_gate"]
        service = AttendanceService(self.detector, self.recognizer, self.db,
                                     self.captured_dir, gate_cfg["camera_id"])

        print("\nOpening main gate webcam.")
        print("  SPACE = log event (auto-detects CHECK_IN vs CHECK_OUT per employee/day)")
        print("  I     = force CHECK_IN   |   O = force CHECK_OUT")
        print("  ESC/q = quit\n")

        with Camera(gate_cfg["source"], camera_id=gate_cfg["camera_id"]) as cam:
            while True:
                frame = cam.read_frame()
                if frame is None:
                    break

                results, annotated = service.process_frame(frame.copy(), known_embeddings)
                cv2.putText(annotated, "SPACE=auto  I=check-in  O=check-out  ESC/q=quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.imshow("LAMS Main Gate - Check-In / Check-Out", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
                elif key in (32, ord("i"), ord("I"), ord("o"), ord("O")):
                    matched = [r for r in results if r["employee_id"]]
                    if not matched:
                        print("  No recognized employee in frame - nothing logged.")
                        continue
                    best = max(matched, key=lambda r: r["score"])

                    forced_type = None
                    if key in (ord("i"), ord("I")):
                        forced_type = "CHECK_IN"
                    elif key in (ord("o"), ord("O")):
                        forced_type = "CHECK_OUT"

                    event_type, image_path = service.record_event(
                        frame, best["employee_id"], best["score"], event_type=forced_type
                    )
                    print(f"  {event_type} logged: {best['employee_id']} "
                          f"(score={best['score']:.2f}) -> {image_path}")

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    # 5) Simulate presence verification
    # ------------------------------------------------------------------ #
    def simulate_presence(self):
        known_embeddings = self.db.get_all_embeddings()
        if not known_embeddings:
            print("No employees enrolled yet. Enroll someone first (menu option 1 or 2).")
            return

        expected_id = input("Expected employee ID at this workplace camera (e.g. EMP001): ").strip()
        wp_cfg = self.config["cameras"]["workplaces"][0]
        service = PresenceService(self.detector, self.recognizer, self.db,
                                   self.captured_dir, wp_cfg["camera_id"])

        print(f"\nOpening workplace webcam ({wp_cfg['camera_id']}). "
              f"Press SPACE to take a presence sample. Press ESC/'q' to quit.\n")

        with Camera(wp_cfg["source"], camera_id=wp_cfg["camera_id"]) as cam:
            while True:
                frame = cam.read_frame()
                if frame is None:
                    break

                preview = frame.copy()
                cv2.putText(preview, "SPACE = sample now | ESC/q = quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("LAMS Workplace - Presence", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
                elif key == 32:
                    status, score, matched_id, image_path = service.verify_frame(
                        frame.copy(), expected_id, known_embeddings
                    )
                    print(f"  Result: {status}  (matched={matched_id}, score={score:.2f}) "
                          f"-> {image_path}")

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    # 6) View data
    # ------------------------------------------------------------------ #
    def view_data(self):
        print("\n--- Employees ---")
        employees = self.db.list_employees()
        if not employees:
            print("(none enrolled yet)")
        for emp_id, name, dept, status, n_emb in employees:
            print(f"  {emp_id:10s} {name:20s} {dept:15s} {status:8s} embeddings={n_emb}")

        print("\n--- Recent Check-Out Events ---")
        events = self.db.recent_attendance_events(10)
        if not events:
            print("(none yet)")
        for e in events:
            event_id, emp_id, cam_id, etype, etime, score, image_path = e
            print(f"  #{event_id} {etime}  {emp_id}  {etype}  score={score:.2f}  {image_path}")

        print("\n--- Recent Presence Events ---")
        events = self.db.recent_presence_events(10)
        if not events:
            print("(none yet)")
        for e in events:
            pid, emp_id, cam_id, stime, status, score, image_path = e
            print(f"  #{pid} {stime}  {emp_id}  {status}  score={score:.2f}  {image_path}")
        print()

    def close(self):
        self.db.close()


MENU = """
==================================================
 LAMS Face Recognition System - Phase 1 (local test)
==================================================
 1) Enroll employee via webcam
 2) Enroll employee from existing dataset/ folder
 3) Live recognition preview
 4) Simulate Main Gate (Check-In / Check-Out)
 5) Simulate Workplace Presence Verification
 6) View employees / recent events
 0) Exit
==================================================
"""


def main():
    try:
        app = LAMSApp()
    except FileNotFoundError as e:
        print(f"\nStartup error: {e}\n")
        sys.exit(1)

    try:
        while True:
            print(MENU)
            choice = input("Choose an option: ").strip()

            if choice == "1":
                app.enroll_via_webcam()
            elif choice == "2":
                app.enroll_from_existing_folder()
            elif choice == "3":
                app.live_preview()
            elif choice == "4":
                app.simulate_checkout()
            elif choice == "5":
                app.simulate_presence()
            elif choice == "6":
                app.view_data()
            elif choice == "0":
                break
            else:
                print("Invalid option, try again.")
    finally:
        app.close()
        print("Goodbye.")


if __name__ == "__main__":
    main()