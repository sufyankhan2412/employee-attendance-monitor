"""
enrollment.py  (Technical Document, section 6)

Employee Enrollment pipeline:
    Employee Image -> Face Detection -> Face Alignment -> SFace ->
    Face Embedding -> stored against the employee ID.

Multiple embeddings per employee (one per enrollment image) let the
system represent variations in appearance, per the design doc.
"""

import os
import cv2

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


def enroll_from_folder(detector, recognizer, database, dataset_dir: str, employee_id: str,
                        employee_name: str = "", department: str = ""):
    """
    Enrolls one employee from a folder of images:
        dataset/EMP001/01.jpg, 02.jpg, ...

    Returns: (n_images_used, n_images_skipped)
    """
    employee_dir = os.path.join(dataset_dir, employee_id)
    if not os.path.isdir(employee_dir):
        raise FileNotFoundError(f"No dataset folder found for '{employee_id}' at '{employee_dir}'")

    database.upsert_employee(employee_id, employee_name, department)

    used, skipped = 0, 0
    for fname in sorted(os.listdir(employee_dir)):
        if not fname.lower().endswith(SUPPORTED_EXTENSIONS):
            continue

        img_path = os.path.join(employee_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [skip] Could not read image: {img_path}")
            skipped += 1
            continue

        faces = detector.detect(img)
        if not faces:
            print(f"  [skip] No face detected in: {img_path}")
            skipped += 1
            continue

        # If multiple faces are found in an enrollment photo, use the
        # largest one (most likely to be the intended subject).
        face = max(faces, key=lambda f: f["box"][2] * f["box"][3])

        embedding = recognizer.get_embedding_from_frame(img, face["raw"])
        database.add_embedding(employee_id, embedding)
        used += 1
        print(f"  [ok]   Embedded {fname} (confidence={face['confidence']:.2f})")

    return used, skipped
