"""
run_presence_daemon.py

Continuous workplace-presence service for camera(s) watching a shared
workspace with multiple employees. At startup (and again at local
midnight) it rolls a fresh random schedule for the day using
src.scheduler.generate_daily_presence_schedule():

    09:00-10:30  -> 1 random shot
    10:30-12:00  -> 1 random shot
    12:00-13:00  -> 1 random shot
    13:00-14:00  -> (lunch, no shot)
    14:00-15:30  -> 1 random shot
    15:30-17:00  -> 1 random shot

  = 5 screenshots/day/camera, each at least `min_gap_minutes` (default 40)
  apart, at unpredictable times within its window - all configurable in
  config.yaml under `presence`.

At each scheduled instant, for EVERY workplace camera in config.yaml it
grabs one frame, detects+recognizes ALL faces in it (not just one
expected person), saves the screenshot once, and logs one presence_events
row per employee actually recognized (PRESENT), plus UNKNOWN rows for
any unrecognized face, plus ABSENT rows for anyone on that camera's
optional `roster` list who wasn't seen.

Run as a long-lived process (systemd service, `screen`/`tmux`, or Docker) -
see TESTING.md.

Usage:
    python run_presence_daemon.py [--config config/config.yaml]
"""

import argparse
import logging
import sys
import time as time_module
import datetime as dt
from zoneinfo import ZoneInfo

from src.utils import load_config, ensure_dir, now_str_tz
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.face_database import FaceDatabase
from src.camera import Camera
from src.presence import PresenceService
from src.scheduler import generate_daily_presence_schedule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [presence] %(levelname)s %(message)s",
)
log = logging.getLogger("presence_daemon")


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
    db = FaceDatabase(config["paths"]["database_path"])
    captured_dir = config["paths"]["captured_images_dir"]
    ensure_dir(captured_dir)

    services = {}
    for wp in config["cameras"]["workplaces"]:
        db.upsert_camera(wp["camera_id"], camera_name=wp["camera_id"], camera_type="WORKPLACE",
                          stream_url=str(wp["source"]))
        services[wp["camera_id"]] = {
            "cfg": wp,
            "service": PresenceService(detector, recognizer, db, captured_dir, wp["camera_id"]),
        }
    return detector, recognizer, db, services


def take_one_shot(cam_source, camera_id, service, known_embeddings, roster, tz_name):
    """Open the camera, grab a single frame, run group presence capture, close it.

    Opening per-shot (rather than holding every workplace stream open all
    day) keeps this simple and robust for RTSP links that don't love being
    held idle for 40+ minutes between reads. If your cameras handle
    long-lived connections fine and you have many workplaces, you can
    switch this to persistent connections instead - see TESTING.md.
    """
    try:
        with Camera(cam_source, camera_id=camera_id) as cam:
            frame = cam.read_frame()
            if frame is None:
                log.warning(f"[{camera_id}] could not read a frame - skipping this sample.")
                return
            image_path, results = service.capture_group_presence(
                frame, known_embeddings, roster=roster, sample_time=now_str_tz(tz_name)
            )
            present = [r["employee_id"] for r in results if r["status"] == "PRESENT"]
            unknown_n = sum(1 for r in results if r["status"] == "UNKNOWN")
            log.info(f"[{camera_id}] sample -> present={present} unknown_faces={unknown_n} "
                     f"-> {image_path}")
    except RuntimeError as e:
        log.warning(f"[{camera_id}] camera error, skipping this sample: {e}")


def run(config_path: str):
    config = load_config(config_path)
    tz_name = config.get("timezone", "Asia/Karachi")
    tz = ZoneInfo(tz_name)

    presence_cfg = config.get("presence", {})
    windows_cfg = presence_cfg.get("windows", [])
    min_gap_minutes = int(presence_cfg.get("min_gap_minutes", 40))
    if not windows_cfg:
        log.error("No presence.windows configured - nothing to schedule. Check config.yaml.")
        sys.exit(1)

    detector, recognizer, db, services = build_services(config)
    known_embeddings = db.get_all_embeddings()
    if not known_embeddings:
        log.warning("No employees enrolled yet - every face will show as UNKNOWN.")

    current_day = dt.datetime.now(tz).date()
    schedule = generate_daily_presence_schedule(windows_cfg, min_gap_minutes, tz, day=current_day)
    log.info(f"Presence daemon started. Timezone={tz_name}. Today's schedule:")
    for t in schedule:
        log.info(f"    {t.strftime('%H:%M:%S')}")

    pending = list(schedule)  # times not yet fired today

    try:
        while True:
            now = dt.datetime.now(tz)

            # New day -> roll a fresh random schedule.
            if now.date() != current_day:
                current_day = now.date()
                schedule = generate_daily_presence_schedule(
                    windows_cfg, min_gap_minutes, tz, day=current_day
                )
                pending = list(schedule)
                known_embeddings = db.get_all_embeddings()  # pick up newly enrolled staff too
                log.info(f"New day - rolled new schedule for {current_day}: "
                         f"{[t.strftime('%H:%M:%S') for t in schedule]}")

            # Fire any scheduled shot whose time has arrived.
            while pending and now >= pending[0]:
                fire_time = pending.pop(0)
                log.info(f"Scheduled shot at {fire_time.strftime('%H:%M:%S')} firing now "
                         f"(actual time {now.strftime('%H:%M:%S')}).")
                for camera_id, entry in services.items():
                    roster = entry["cfg"].get("roster") or None
                    take_one_shot(entry["cfg"]["source"], camera_id, entry["service"],
                                  known_embeddings, roster, tz_name)

            time_module.sleep(5)  # fine-grained enough without busy-looping

    except KeyboardInterrupt:
        log.info("Presence daemon stopped by user.")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAMS workplace presence screenshot daemon")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    try:
        run(args.config)
    except FileNotFoundError as e:
        print(f"Startup error: {e}")
        sys.exit(1)