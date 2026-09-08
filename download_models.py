"""
download_models.py

Downloads the two ONNX models this project needs, straight from the
official OpenCV Zoo GitHub repository, into the models/ folder:

    - face_detection_yunet_2023mar.onnx   (~230 KB)
    - face_recognition_sface_2021dec.onnx (~37 MB)

Run this once, after installing requirements.txt:

    python download_models.py

If your network blocks GitHub's LFS media host (media.githubusercontent.com),
see the manual-download instructions printed below on failure.
"""

import os
import sys
import requests

MODELS_DIR = "models"

FILES = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}

# Minimum expected byte size for each file. If a download comes back much
# smaller than this, it's almost certainly an HTML/LFS-pointer page rather
# than the real model, and we should fail loudly instead of saving junk.
MIN_SIZE_BYTES = {
    "face_detection_yunet_2023mar.onnx": 100_000,
    "face_recognition_sface_2021dec.onnx": 5_000_000,
}


def download(name: str, url: str) -> bool:
    dest = os.path.join(MODELS_DIR, name)
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_SIZE_BYTES[name]:
        print(f"[skip] {name} already present.")
        return True

    print(f"[download] {name}")
    print(f"           from {url}")
    try:
        with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            written = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
                        written += len(chunk)
            size_ok = written >= MIN_SIZE_BYTES[name]
            if not size_ok:
                os.remove(dest)
                print(f"[fail] Downloaded file for {name} was only {written} bytes "
                      f"(expected several MB+). Removed the bad file.")
                return False
            print(f"[ok]   Saved {name} ({written/1024:.0f} KB) -> {dest}")
            return True
    except Exception as e:
        print(f"[fail] Could not download {name}: {e}")
        return False


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    all_ok = True
    for name, url in FILES.items():
        ok = download(name, url)
        all_ok = all_ok and ok

    if not all_ok:
        print("\n" + "=" * 70)
        print("One or more models failed to download automatically.")
        print("Manual fallback: download these two files in your browser")
        print("and place them in the 'models/' folder with these exact names:")
        print()
        for name, url in FILES.items():
            print(f"  - {name}")
            print(f"    {url}")
        print()
        print("(Opening that GitHub URL in a browser and clicking 'Download' works")
        print(" even when the direct script download is blocked by a firewall/proxy.)")
        print("=" * 70)
        sys.exit(1)

    print("\nAll models ready.")


if __name__ == "__main__":
    main()
