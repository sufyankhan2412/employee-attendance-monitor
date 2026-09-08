"""
utils.py
Small shared helpers used across the project: config loading, timestamps,
drawing helpers. Nothing in here is AI-specific.
"""

import os
import yaml
import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9 fallback
    from backports.zoneinfo import ZoneInfo


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the YAML config file and return it as a dict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. "
            f"Run this script from the project root (the 'lams' folder)."
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def now_str() -> str:
    """Current timestamp as YYYY-MM-DD HH:MM:SS (for DB rows)."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_compact() -> str:
    """Current timestamp as HHMMSS (for filenames)."""
    return datetime.datetime.now().strftime("%H%M%S")


def today_str() -> str:
    return datetime.datetime.now().strftime("%Y%m%d")


# --------------------------------------------------------------------- #
# Timezone-aware helpers (used by the gate/presence daemons)
#
# IMPORTANT: the server this runs on may NOT be set to Pakistan time
# (cloud VMs are very often UTC). Everything that decides "is it 8am
# yet" or "what day is 'today' for this employee's check-in" must go
# through these functions instead of naive datetime.now(), or the
# check-in/check-out windows will silently be wrong by however many
# hours the server's clock is offset from Asia/Karachi.
# --------------------------------------------------------------------- #
def now_tz(tz_name: str = "Asia/Karachi") -> datetime.datetime:
    """Current timezone-aware datetime in the given IANA timezone."""
    return datetime.datetime.now(ZoneInfo(tz_name))


def now_str_tz(tz_name: str = "Asia/Karachi") -> str:
    """Current timestamp in tz_name, formatted for DB storage."""
    return now_tz(tz_name).strftime("%Y-%m-%d %H:%M:%S")


def today_str_tz(tz_name: str = "Asia/Karachi") -> str:
    """Current date (YYYY-MM-DD) in tz_name, for 'has this happened today' checks."""
    return now_tz(tz_name).strftime("%Y-%m-%d")


def build_database_url(config: dict) -> str:
    """
    Returns the SQLAlchemy connection URL to use, so every entry point
    (main.py, run_gate_daemon.py, run_presence_daemon.py) builds it the
    same way.

    Preferred: an explicit config["database"]["url"], e.g.
        postgresql+psycopg2://lams_user:PASSWORD@localhost:5432/lams

    Falls back to a local SQLite file at config["paths"]["database_path"]
    if no database.url is set, so existing configs keep working unchanged.
    """
    db_cfg = config.get("database") or {}
    url = db_cfg.get("url")
    if url:
        return url
    return f"sqlite:///{config['paths']['database_path']}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def draw_face_box(frame, box, label, color=(0, 255, 0)):
    """
    Draw a bounding box + label on a frame (used for on-screen preview
    while testing with the laptop webcam).
    box: (x, y, w, h)
    """
    import cv2

    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        label,
        (x, max(0, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame