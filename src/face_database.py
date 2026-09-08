"""
face_database.py  (Technical Document, sections 6, 17, 21)

SQLite-backed storage matching the database structure in the design doc:
    - employees
    - face_embeddings
    - cameras
    - attendance_events
    - presence_events

Responsibilities:
    - Employee enrollment (store embeddings against an employee ID)
    - Retrieve employee embeddings for the recognizer
    - Record check-out / presence events
"""

import os
import sqlite3
import numpy as np

from .utils import now_str


class FaceDatabase:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def _create_tables(self):
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                employee_id   TEXT PRIMARY KEY,
                employee_name TEXT,
                department    TEXT,
                status        TEXT DEFAULT 'ACTIVE',
                created_at    TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                embedding_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id   TEXT REFERENCES employees(employee_id),
                embedding     BLOB,
                created_at    TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id   TEXT PRIMARY KEY,
                camera_name TEXT,
                camera_type TEXT,
                location    TEXT,
                stream_url  TEXT,
                status      TEXT DEFAULT 'ACTIVE',
                created_at  TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance_events (
                event_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id        TEXT REFERENCES employees(employee_id),
                camera_id          TEXT REFERENCES cameras(camera_id),
                event_type         TEXT,
                event_time         TEXT,
                recognition_score  REAL,
                image_path         TEXT,
                created_at         TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS presence_events (
                presence_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id        TEXT REFERENCES employees(employee_id),
                camera_id          TEXT REFERENCES cameras(camera_id),
                sample_time        TEXT,
                status             TEXT,
                recognition_score  REAL,
                image_path         TEXT,
                created_at         TEXT
            )
        """)

        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Employees / enrollment
    # ------------------------------------------------------------------ #
    def upsert_employee(self, employee_id: str, employee_name: str = "", department: str = ""):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO employees (employee_id, employee_name, department, status, created_at)
            VALUES (?, ?, ?, 'ACTIVE', ?)
            ON CONFLICT(employee_id) DO UPDATE SET
                employee_name = excluded.employee_name,
                department = excluded.department
            """,
            (employee_id, employee_name, department, now_str()),
        )
        self.conn.commit()

    def add_embedding(self, employee_id: str, embedding: np.ndarray):
        blob = embedding.astype(np.float32).tobytes()
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO face_embeddings (employee_id, embedding, created_at) VALUES (?, ?, ?)",
            (employee_id, blob, now_str()),
        )
        self.conn.commit()

    def get_all_embeddings(self):
        """Returns list of (employee_id, np.ndarray) for every stored embedding."""
        cur = self.conn.cursor()
        cur.execute("SELECT employee_id, embedding FROM face_embeddings")
        rows = cur.fetchall()
        result = []
        for employee_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            result.append((employee_id, vec))
        return result

    def list_employees(self):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT e.employee_id, e.employee_name, e.department, e.status,
                   COUNT(f.embedding_id) as n_embeddings
            FROM employees e
            LEFT JOIN face_embeddings f ON e.employee_id = f.employee_id
            GROUP BY e.employee_id
            ORDER BY e.employee_id
        """)
        return cur.fetchall()

    # ------------------------------------------------------------------ #
    # Cameras
    # ------------------------------------------------------------------ #
    def upsert_camera(self, camera_id: str, camera_name: str = "", camera_type: str = "",
                       location: str = "", stream_url: str = ""):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO cameras (camera_id, camera_name, camera_type, location, stream_url, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?)
            ON CONFLICT(camera_id) DO UPDATE SET
                camera_name = excluded.camera_name,
                camera_type = excluded.camera_type,
                location = excluded.location,
                stream_url = excluded.stream_url
            """,
            (camera_id, camera_name, camera_type, location, stream_url, now_str()),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def log_attendance_event(self, employee_id, camera_id, event_type, recognition_score, image_path):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO attendance_events
                (employee_id, camera_id, event_type, event_time, recognition_score, image_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (employee_id, camera_id, event_type, now_str(), recognition_score, image_path, now_str()),
        )
        self.conn.commit()
        return cur.lastrowid

    def log_presence_event(self, employee_id, camera_id, status, recognition_score, image_path):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO presence_events
                (employee_id, camera_id, sample_time, status, recognition_score, image_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (employee_id, camera_id, now_str(), status, recognition_score, image_path, now_str()),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_last_event_type_today(self, employee_id: str):
        """
        Returns the event_type ('CHECK_IN' / 'CHECK_OUT') of this employee's
        most recent attendance event *today*, or None if they have no event
        yet today. Used to auto-toggle check-in/check-out at the gate.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT event_type FROM attendance_events
            WHERE employee_id = ? AND date(event_time) = date('now', 'localtime')
            ORDER BY event_id DESC LIMIT 1
            """,
            (employee_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def recent_attendance_events(self, limit=20):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT event_id, employee_id, camera_id, event_type, event_time, recognition_score, image_path "
            "FROM attendance_events ORDER BY event_id DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()

    def recent_presence_events(self, limit=20):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT presence_id, employee_id, camera_id, sample_time, status, recognition_score, image_path "
            "FROM presence_events ORDER BY presence_id DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()

    def close(self):
        self.conn.close()