"""
utils.py
Small shared helpers used across the project: config loading, timestamps,
drawing helpers. Nothing in here is AI-specific.
"""

import os
import yaml
import datetime


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
