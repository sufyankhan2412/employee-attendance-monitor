# LAMS Face Recognition System — Phase 1 (Local Test Build)

This is a working implementation of the system described in the
"LAMS Face Recognition System – Technical Document":

- **Face Detection** → YuNet (`face_detection_yunet_2023mar.onnx`)
- **Face Alignment** → geometric alignment using YuNet's 5 landmarks
- **Face Recognition** → SFace (`face_recognition_sface_2021dec.onnx`)
- **Matching** → cosine similarity, threshold `0.363` (configurable)
- **Employee enrollment**, **Check-Out events**, and **Presence
  verification**, all logged to a local SQLite database.

For this first version you test everything using your **laptop's webcam**
instead of real CCTV. Swapping in real CCTV later is a one-line config
change (see "Connecting real CCTV later" at the bottom) — no code changes.

You said you've never used these models before — that's completely fine,
this guide assumes zero prior experience. Follow it top to bottom.

---

## 0. What you need before you start

- **Python 3.9–3.11** installed (3.12 also fine, but opencv-contrib wheels
  are most reliable on 3.9–3.11). Check with:
  ```
  python --version
  ```
  If you don't have Python, download it from https://www.python.org/downloads/
  (on Windows, tick "Add Python to PATH" during install).
- **VS Code** with the Python extension installed.
- A working **laptop webcam** (built-in or USB).
- Internet access, at least for the one-time model download in Step 3.

---

## 1. Unzip and open the project

1. Unzip the file you downloaded — you'll get a folder called `lams/`.
2. Open VS Code → `File > Open Folder...` → select the `lams` folder.
3. Open a terminal inside VS Code: `Terminal > New Terminal`.

Everything below is run from this terminal, **from inside the `lams`
folder** (that's your project root — you should see `main.py` if you run
`ls` / `dir`).

---

## 2. Create a virtual environment and install dependencies

A virtual environment keeps this project's Python packages separate from
everything else on your machine — you don't need to understand it deeply,
just run these commands.

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You'll know the venv is active because your terminal prompt shows `(venv)`
at the start. **Every time you open a new terminal to work on this
project, re-run the `activate` line** before running any Python command.

If VS Code asks "Select Python Interpreter", pick the one inside
`venv` (e.g. `./venv/bin/python` or `.\venv\Scripts\python.exe`).

---

## 3. Download the AI models (one-time)

The two model files (YuNet and SFace) are not stored in this zip because
they're binary files better fetched fresh. Run:

```bash
python download_models.py
```

This downloads:
- `models/face_detection_yunet_2023mar.onnx` (~230 KB) — finds faces in an image
- `models/face_recognition_sface_2021dec.onnx` (~37 MB) — turns a face into a numeric "fingerprint" (embedding)

Both come from the official, open-source **OpenCV Zoo** GitHub repository —
these are the exact models named in the technical document.

**If the script fails** (e.g. your network/firewall blocks GitHub's file
storage), it will print two direct links. Open each one in your browser,
click "Download", and save the files into the `models/` folder using the
exact filenames shown in the script's output.

---

## 4. Understand what's in this project

```
lams/
├── models/                          # the two ONNX models (Step 3)
├── dataset/                         # employee enrollment photos go here
│   └── EMP001/ 01.jpg, 02.jpg, ...
├── src/
│   ├── face_detector.py             # YuNet wrapper (finds faces)
│   ├── face_recognizer.py           # SFace wrapper (aligns + embeds + compares faces)
│   ├── face_database.py             # SQLite storage (employees, embeddings, events)
│   ├── enrollment.py                # turns employee photos into stored embeddings
│   ├── attendance.py                # main-gate check-out pipeline
│   ├── presence.py                  # workplace presence-verification pipeline
│   └── camera.py                    # opens your webcam (or, later, a CCTV stream)
├── outputs/captured_images/         # snapshots saved by check-out / presence events
├── config/config.yaml               # all settings (thresholds, camera sources, paths)
├── database/lams.db                 # created automatically on first run
├── download_models.py               # one-time model downloader (Step 3)
├── main.py                          # <-- you run this
└── requirements.txt
```

You generally never need to edit the `src/` files to *use* the system —
`config/config.yaml` controls the behavior, and `main.py` is the menu you
interact with.

---

## 5. Run the app

```bash
python main.py
```

You'll see a menu:

```
1) Enroll employee via webcam
2) Enroll employee from existing dataset/ folder
3) Live recognition preview
4) Simulate Main Gate Check-Out
5) Simulate Workplace Presence Verification
6) View employees / recent events
0) Exit
```

### Step-by-step first run

**Step A — Enroll yourself as a test employee**
1. Choose `1) Enroll employee via webcam`.
2. Enter an Employee ID (e.g. `EMP001`), your name, and a department.
3. A webcam window opens. Look at the camera and press **SPACE** 5 times
   (from slightly different angles — straight on, slight left, slight
   right, closer, further) to capture 5 photos. Press **ESC** to cancel
   if needed.
4. The app automatically detects your face in each photo, aligns it, and
   stores a numeric "embedding" for you in the database. You'll see
   `[ok] Embedded 01.jpg ...` printed for each photo.

**Step B — Sanity-check recognition works**
1. Choose `3) Live recognition preview`.
2. A window opens showing your webcam feed with a green box around your
   face and your Employee ID + similarity score (e.g. `EMP001 (0.55)`).
   A red box + `UNKNOWN` means the face wasn't matched.
3. Press **ESC** or **q** to close.

**Step C — Simulate a check-out event (main gate)**
1. Choose `4) Simulate Main Gate Check-Out`.
2. With your face visible and recognized (green box), press **SPACE**.
   This logs a `CHECK_OUT` event (your ID, timestamp, camera, recognition
   score) and saves a snapshot to `outputs/captured_images/`.
3. Press **ESC**/`q` to quit.

**Step D — Simulate presence verification (workplace camera)**
1. Choose `5) Simulate Workplace Presence Verification`.
2. Enter the Employee ID you expect to see (e.g. `EMP001`).
3. Press **SPACE** to take a sample. You'll get `PRESENT`, `ABSENT`,
   `UNKNOWN`, or `EXCEPTION` (no face found at all), matching section 13
   of the technical document.

**Step E — Review what got logged**
1. Choose `6) View employees / recent events` to see everything stored
   in the SQLite database so far.

That's the whole Phase 1 pipeline, running end-to-end on your laptop.

---

## 6. Enrolling with existing photos instead of the webcam

If you already have photos of someone (e.g. from your phone):
1. Create a folder `dataset/EMP002/` and put 3–5 clear, front-facing
   `.jpg`/`.png` photos in it (one face per photo, good lighting).
2. Run `python main.py` → option `2) Enroll employee from existing
   dataset/ folder` → enter `EMP002` and their details.

---

## 7. Tuning the recognition threshold

`config/config.yaml` has:
```yaml
recognition:
  cosine_threshold: 0.363
```
This is the **initial development value from the technical document**,
not a final production value. In practice:
- If real matches are being reported as `UNKNOWN`, **lower** this slightly
  (e.g. `0.30`).
- If different people are being confused with each other, **raise** it
  (e.g. `0.40–0.45`).

Change the number, save the file, and re-run `main.py` — no code changes
needed. Re-tune once you have real employee photos and real camera
footage, as the document recommends.

---

## 8. Connecting real CCTV later

Everything above uses your webcam because `config.yaml` has:
```yaml
cameras:
  main_gate:
    source: 0
```
`0` means "default webcam". To use a real IP camera instead, replace it
with the camera's stream URL, e.g.:
```yaml
cameras:
  main_gate:
    source: "rtsp://username:password@192.168.1.50:554/stream1"
```
No other code changes are required — `src/camera.py` passes `source`
straight to OpenCV's `VideoCapture`, which understands RTSP/HTTP streams
the same way it understands a webcam index.

---

## 9. Troubleshooting

- **"Could not open camera source '0'"** — another app (Zoom, Teams,
  browser tab) may be using the webcam; close it and retry. On macOS,
  make sure your terminal/VS Code has camera permission (System Settings
  → Privacy & Security → Camera).
- **`ModuleNotFoundError: No module named 'cv2'`** — your venv isn't
  active, or `pip install -r requirements.txt` didn't finish. Re-run
  Step 2.
- **Model download fails** — see the manual-download fallback printed by
  `download_models.py` (Step 3).
- **Everyone shows as UNKNOWN** — make sure you enrolled at least one
  employee (option 1 or 2) and that lighting is decent; also see the
  threshold tuning note in Section 7 above.
- **`sqlite3.OperationalError: database is locked`** — close any other
  program (e.g. a DB browser) that has `database/lams.db` open.

---

## 10. What's implemented vs. what's next

This matches "Phase 1 Implementation" (section 27 of the technical
document): model loading, detection, alignment, embedding, similarity
comparison, enrollment, a local face database, recognition, and
unknown-person handling — all runnable end-to-end locally.

Not yet built (would come in later phases, per the document's broader
architecture): live/continuous CCTV ingestion for multiple simultaneous
streams, the Admin Dashboard, and automated scheduled random-sampling for
presence checks (right now, presence samples are taken on-demand via the
menu so you can test it manually with your webcam).
