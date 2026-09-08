"""
run_gate_daemon.py

Continuous main-gate service - this is what actually runs 24/7 against
a real RTSP camera, unlike the manual/menu-driven simulate_checkout() in
main.py (which is still useful for one-off manual testing).

Behaviour (Pakistan time by default, see config.yaml -> timezone):
  - Before gate.checkin_start (default 08:00): idle, no recognition run.
  - 08:00 -> 17:00: any employee recognized for the first time today gets
    a CHECK_IN logged (once per employee per day).
  - 17:00 onward: any employee recognized gets a CHECK_OUT logged (once
    per employee per day). The daemon just keeps running through the
    evening, so it's naturally "active until the last employee leaves" -
    there's nothing to configure for that part, it simply keeps watching.
  - A per-employee cooldown (gate.cooldown_seconds) stops the same person
    lingering at the gate from generating duplicate rows.
  - If the camera read fails (common with flaky RTSP links), it retries
    the connection with a short backoff instead of crashing.

Run as a long-lived process (systemd service, `screen`/`tmux` session, or
Docker container) - see TESTING.md for real-camera setup and a sample
systemd unit.

Usage:
    python run_gate_daemon.py [--config config/config.yaml]
"""

import argparse
import logging
import sys
import time as time_module
from zoneinfo import ZoneInfo

import cv2

from src.utils import load_config, ensure_dir, now_str_tz, today_str_tz, build_database_url
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.face_database import FaceDatabase
from src.camera import Camera
from src.attendance import AttendanceService
from src.scheduler import gate_mode_for_time
from src.scheduler import _parse_hhmm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [gate] %(levelname)s %(message)s",
)
log = logging.getLogger("gate_daemon")


def build_services(config):
    detector_cfg = config["detection"]
    detector = FaceDetector(
        model_path=config["models"]["detector_path"],
        score_threshold=detector_cfg["score_threshold"],
        nms_threshold=detector_cfg["nms_threshold"],
        top_k=detector_cfg["top_k"],
        input_size=detector_cfg["input_size"],
    )
    recognizer = FaceRecognizer(
        model_path=config["models"]["recognizer_path"],
        cosine_threshold=config["recognition"]["cosine_threshold"],
    )
    db = FaceDatabase(build_database_url(config))

    gate_cfg = config["cameras"]["main_gate"]
    db.upsert_camera(gate_cfg["camera_id"], camera_name="Main Gate", camera_type="GATE",
                      stream_url=str(gate_cfg["source"]))

    captured_dir = config["paths"]["captured_images_dir"]
    ensure_dir(captured_dir)

    attendance = AttendanceService(detector, recognizer, db, captured_dir, gate_cfg["camera_id"])
    return detector, recognizer, db, attendance, gate_cfg


def open_camera_with_retry(source, camera_id, max_wait_seconds=60):
    """Keeps trying to (re)connect to a flaky RTSP stream instead of dying."""
    wait = 2
    while True:
        try:
            return Camera(source, camera_id=camera_id)
        except RuntimeError as e:
            log.warning(f"Camera connect failed ({e}); retrying in {wait}s...")
            time_module.sleep(wait)
            wait = min(wait * 2, max_wait_seconds)


def run(config_path: str):
    config = load_config(config_path)
    tz_name = config.get("timezone", "Asia/Karachi")
    tz = ZoneInfo(tz_name)

    gate_daemon_cfg = config.get("gate", {})
    checkin_start = _parse_hhmm(gate_daemon_cfg.get("checkin_start", "08:00"))
    checkout_start = _parse_hhmm(gate_daemon_cfg.get("checkout_start", "17:00"))
    poll_interval = float(gate_daemon_cfg.get("poll_interval_seconds", 1.0))
    cooldown_seconds = float(gate_daemon_cfg.get("cooldown_seconds", 120))

    detector, recognizer, db, attendance, gate_cfg = build_services(config)
    known_embeddings = db.get_all_embeddings()
    if not known_embeddings:
        log.warning("No employees enrolled yet - every face will show as UNKNOWN. "
                    "Run 'python main.py' option 1/2 to enroll people first.")

    last_embeddings_refresh = time_module.monotonic()
    last_event_at = {}  # employee_id -> monotonic() timestamp of last logged event

    cam = open_camera_with_retry(gate_cfg["source"], gate_cfg["camera_id"])
    log.info(f"Gate daemon started. Timezone={tz_name}, "
             f"check-in from {checkin_start}, check-out from {checkout_start}.")

    try:
        while True:
            now = None
            try:
                import datetime as _dt
                now = _dt.datetime.now(tz)
                mode = gate_mode_for_time(now, checkin_start, checkout_start)

                if mode is None:
                    # Before opening hours - idle, don't burn CPU/GPU on recognition.
                    time_module.sleep(min(30, poll_interval * 10))
                    continue

                # Refresh the enrolled-employee list every 5 minutes in case
                # someone was enrolled while the daemon is running.
                if time_module.monotonic() - last_embeddings_refresh > 300:
                    known_embeddings = db.get_all_embeddings()
                    last_embeddings_refresh = time_module.monotonic()

                frame = cam.read_frame()
                if frame is None:
                    log.warning("Lost camera frame - reconnecting...")
                    cam.release()
                    cam = open_camera_with_retry(gate_cfg["source"], gate_cfg["camera_id"])
                    continue

                results, _ = attendance.process_frame(frame.copy(), known_embeddings, draw=False)
                today = today_str_tz(tz_name)

                for r in results:
                    employee_id = r["employee_id"]
                    if not employee_id:
                        continue

                    last_t = last_event_at.get(employee_id, 0)
                    if time_module.monotonic() - last_t < cooldown_seconds:
                        continue  # still in cooldown, ignore repeat sighting

                    if db.has_event_type_today(employee_id, mode, today):
                        # Already checked in (or out) today - don't duplicate.
                        last_event_at[employee_id] = time_module.monotonic()
                        continue

                    event_type, image_path = attendance.record_event(
                        frame, employee_id, r["score"], event_type=mode,
                        event_time=now_str_tz(tz_name),
                    )
                    last_event_at[employee_id] = time_module.monotonic()
                    log.info(f"{event_type} logged for {employee_id} "
                             f"(score={r['score']:.2f}) -> {image_path}")

            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("Unexpected error in gate loop; continuing after short pause.")
                time_module.sleep(2)

            time_module.sleep(poll_interval)

    except KeyboardInterrupt:
        log.info("Gate daemon stopped by user.")
    finally:
        cam.release()
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAMS main-gate check-in/check-out daemon")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    try:
        run(args.config)
    except FileNotFoundError as e:
        print(f"Startup error: {e}")
        sys.exit(1)