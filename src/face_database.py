"""
face_database.py  (Technical Document, sections 6, 17, 21)

Storage layer, matching the database structure in the design doc:
    - employees
    - face_embeddings
    - cameras
    - attendance_events
    - presence_events

Backend-agnostic via SQLAlchemy Core: pass any SQLAlchemy connection URL
and it works the same way, whether that's SQLite (the default, zero-setup
option for local testing) or a real Postgres/MySQL server for production:

    sqlite:///database/lams.db
    postgresql+psycopg2://lams_user:PASSWORD@localhost:5432/lams
    mysql+pymysql://lams_user:PASSWORD@localhost:3306/lams

See config.yaml's `database:` section and TESTING.md for how to set the
URL. All the calling code (attendance.py, presence.py, main.py, the two
daemons) is unchanged - it only ever calls FaceDatabase's methods, never
raw SQL, so switching backends is a one-line config change.

Responsibilities:
    - Employee enrollment (store embeddings against an employee ID)
    - Retrieve employee embeddings for the recognizer
    - Record check-in/check-out / presence events
"""

import numpy as np
import sqlalchemy as sa

from .utils import now_str


class FaceDatabase:
    def __init__(self, database_url: str):
        """
        database_url: any SQLAlchemy connection URL, e.g.
            'sqlite:///database/lams.db'
            'postgresql+psycopg2://user:pass@host:5432/dbname'
            'mysql+pymysql://user:pass@host:3306/dbname'
        """
        self.database_url = database_url
        connect_args = {}
        if database_url.startswith("sqlite"):
            # Needed because the SQLite connection is used from more than
            # one thread in the daemons (main loop + any background work).
            connect_args = {"check_same_thread": False}

        self.engine = sa.create_engine(database_url, connect_args=connect_args, future=True)
        self.metadata = sa.MetaData()
        self._define_tables()
        self.metadata.create_all(self.engine)

    # ------------------------------------------------------------------ #
    # Schema (SQLAlchemy Core - generic types map to the right column
    # type on whichever backend you point this at: BLOB/BYTEA, etc.)
    # ------------------------------------------------------------------ #
    def _define_tables(self):
        m = self.metadata

        self.employees = sa.Table(
            "employees", m,
            sa.Column("employee_id", sa.String(64), primary_key=True),
            sa.Column("employee_name", sa.String(255)),
            sa.Column("department", sa.String(255)),
            sa.Column("status", sa.String(32), server_default="ACTIVE"),
            sa.Column("created_at", sa.String(32)),
        )

        self.face_embeddings = sa.Table(
            "face_embeddings", m,
            sa.Column("embedding_id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("employee_id", sa.String(64), sa.ForeignKey("employees.employee_id")),
            sa.Column("embedding", sa.LargeBinary),
            sa.Column("created_at", sa.String(32)),
        )

        self.cameras = sa.Table(
            "cameras", m,
            sa.Column("camera_id", sa.String(64), primary_key=True),
            sa.Column("camera_name", sa.String(255)),
            sa.Column("camera_type", sa.String(32)),
            sa.Column("location", sa.String(255)),
            sa.Column("stream_url", sa.Text),
            sa.Column("status", sa.String(32), server_default="ACTIVE"),
            sa.Column("created_at", sa.String(32)),
        )

        self.attendance_events = sa.Table(
            "attendance_events", m,
            sa.Column("event_id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("employee_id", sa.String(64), sa.ForeignKey("employees.employee_id")),
            sa.Column("camera_id", sa.String(64), sa.ForeignKey("cameras.camera_id")),
            sa.Column("event_type", sa.String(16)),   # 'CHECK_IN' / 'CHECK_OUT'
            sa.Column("event_time", sa.String(32)),   # 'YYYY-MM-DD HH:MM:SS', tz-correct string
            sa.Column("recognition_score", sa.Float),
            sa.Column("image_path", sa.Text),
            sa.Column("created_at", sa.String(32)),
        )

        self.presence_events = sa.Table(
            "presence_events", m,
            sa.Column("presence_id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("employee_id", sa.String(64)),  # may be 'UNKNOWN', so no FK
            sa.Column("camera_id", sa.String(64), sa.ForeignKey("cameras.camera_id")),
            sa.Column("sample_time", sa.String(32)),
            sa.Column("status", sa.String(16)),        # PRESENT / ABSENT / UNKNOWN / EXCEPTION
            sa.Column("recognition_score", sa.Float),
            sa.Column("image_path", sa.Text),
            sa.Column("created_at", sa.String(32)),
        )

    # ------------------------------------------------------------------ #
    # Employees / enrollment
    # ------------------------------------------------------------------ #
    def upsert_employee(self, employee_id: str, employee_name: str = "", department: str = ""):
        with self.engine.begin() as conn:
            dialect = self.engine.dialect.name
            if dialect == "sqlite":
                stmt = sa.text("""
                    INSERT INTO employees (employee_id, employee_name, department, status, created_at)
                    VALUES (:eid, :name, :dept, 'ACTIVE', :now)
                    ON CONFLICT(employee_id) DO UPDATE SET
                        employee_name = excluded.employee_name,
                        department = excluded.department
                """)
            elif dialect == "postgresql":
                stmt = sa.text("""
                    INSERT INTO employees (employee_id, employee_name, department, status, created_at)
                    VALUES (:eid, :name, :dept, 'ACTIVE', :now)
                    ON CONFLICT (employee_id) DO UPDATE SET
                        employee_name = EXCLUDED.employee_name,
                        department = EXCLUDED.department
                """)
            else:
                # Generic fallback (e.g. MySQL): update-then-insert-if-missing.
                exists = conn.execute(
                    sa.select(self.employees.c.employee_id)
                    .where(self.employees.c.employee_id == employee_id)
                ).fetchone()
                if exists:
                    conn.execute(
                        self.employees.update()
                        .where(self.employees.c.employee_id == employee_id)
                        .values(employee_name=employee_name, department=department)
                    )
                else:
                    conn.execute(
                        self.employees.insert().values(
                            employee_id=employee_id, employee_name=employee_name,
                            department=department, status="ACTIVE", created_at=now_str(),
                        )
                    )
                return
            conn.execute(stmt, {"eid": employee_id, "name": employee_name,
                                 "dept": department, "now": now_str()})

    def add_embedding(self, employee_id: str, embedding: np.ndarray):
        blob = embedding.astype(np.float32).tobytes()
        with self.engine.begin() as conn:
            conn.execute(
                self.face_embeddings.insert().values(
                    employee_id=employee_id, embedding=blob, created_at=now_str()
                )
            )

    def get_all_embeddings(self):
        """Returns list of (employee_id, np.ndarray) for every stored embedding."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                sa.select(self.face_embeddings.c.employee_id, self.face_embeddings.c.embedding)
            ).fetchall()
        result = []
        for employee_id, blob in rows:
            vec = np.frombuffer(bytes(blob), dtype=np.float32)
            result.append((employee_id, vec))
        return result

    def list_employees(self):
        with self.engine.connect() as conn:
            rows = conn.execute(sa.text("""
                SELECT e.employee_id, e.employee_name, e.department, e.status,
                       COUNT(f.embedding_id) as n_embeddings
                FROM employees e
                LEFT JOIN face_embeddings f ON e.employee_id = f.employee_id
                GROUP BY e.employee_id, e.employee_name, e.department, e.status
                ORDER BY e.employee_id
            """)).fetchall()
        return [tuple(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Cameras
    # ------------------------------------------------------------------ #
    def upsert_camera(self, camera_id: str, camera_name: str = "", camera_type: str = "",
                       location: str = "", stream_url: str = ""):
        with self.engine.begin() as conn:
            dialect = self.engine.dialect.name
            if dialect in ("sqlite", "postgresql"):
                conflict_kw = "ON CONFLICT(camera_id)" if dialect == "sqlite" else "ON CONFLICT (camera_id)"
                excluded = "excluded" if dialect == "sqlite" else "EXCLUDED"
                stmt = sa.text(f"""
                    INSERT INTO cameras (camera_id, camera_name, camera_type, location, stream_url, status, created_at)
                    VALUES (:cid, :name, :ctype, :loc, :url, 'ACTIVE', :now)
                    {conflict_kw} DO UPDATE SET
                        camera_name = {excluded}.camera_name,
                        camera_type = {excluded}.camera_type,
                        location = {excluded}.location,
                        stream_url = {excluded}.stream_url
                """)
                conn.execute(stmt, {"cid": camera_id, "name": camera_name, "ctype": camera_type,
                                     "loc": location, "url": stream_url, "now": now_str()})
            else:
                exists = conn.execute(
                    sa.select(self.cameras.c.camera_id)
                    .where(self.cameras.c.camera_id == camera_id)
                ).fetchone()
                if exists:
                    conn.execute(
                        self.cameras.update().where(self.cameras.c.camera_id == camera_id)
                        .values(camera_name=camera_name, camera_type=camera_type,
                                location=location, stream_url=stream_url)
                    )
                else:
                    conn.execute(
                        self.cameras.insert().values(
                            camera_id=camera_id, camera_name=camera_name, camera_type=camera_type,
                            location=location, stream_url=stream_url, status="ACTIVE",
                            created_at=now_str(),
                        )
                    )

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def log_attendance_event(self, employee_id, camera_id, event_type, recognition_score,
                              image_path, event_time=None):
        """
        event_time: pass an explicit 'YYYY-MM-DD HH:MM:SS' string (e.g. from
        utils.now_str_tz('Asia/Karachi')) so the stored time is correct
        regardless of the server OS's own timezone. Falls back to naive
        server-local time if omitted (kept for backward compatibility with
        the interactive CLI in main.py).
        """
        event_time = event_time or now_str()
        with self.engine.begin() as conn:
            result = conn.execute(
                self.attendance_events.insert().values(
                    employee_id=employee_id, camera_id=camera_id, event_type=event_type,
                    event_time=event_time, recognition_score=recognition_score,
                    image_path=image_path, created_at=now_str(),
                )
            )
            return result.inserted_primary_key[0]

    def log_presence_event(self, employee_id, camera_id, status, recognition_score,
                            image_path, sample_time=None):
        sample_time = sample_time or now_str()
        with self.engine.begin() as conn:
            result = conn.execute(
                self.presence_events.insert().values(
                    employee_id=employee_id, camera_id=camera_id, sample_time=sample_time,
                    status=status, recognition_score=recognition_score, image_path=image_path,
                    created_at=now_str(),
                )
            )
            return result.inserted_primary_key[0]

    def get_last_event_type_today(self, employee_id: str, today_date: str = None):
        """
        Returns the event_type ('CHECK_IN' / 'CHECK_OUT') of this employee's
        most recent attendance event *today*, or None if they have no event
        yet today. Used to auto-toggle check-in/check-out at the gate.

        today_date: pass 'YYYY-MM-DD' in the correct timezone (e.g.
        utils.today_str_tz('Asia/Karachi')). Comparison is a plain string
        range on event_time ('YYYY-MM-DD HH:MM:SS' sorts correctly as text),
        which behaves identically on SQLite, Postgres, and MySQL - no
        backend-specific date() function needed.
        """
        day_start, day_end = self._day_bounds(today_date)
        with self.engine.connect() as conn:
            row = conn.execute(
                sa.select(self.attendance_events.c.event_type)
                .where(
                    self.attendance_events.c.employee_id == employee_id,
                    self.attendance_events.c.event_time >= day_start,
                    self.attendance_events.c.event_time <= day_end,
                )
                .order_by(self.attendance_events.c.event_id.desc())
                .limit(1)
            ).fetchone()
        return row[0] if row else None

    def has_event_type_today(self, employee_id: str, event_type: str, today_date: str) -> bool:
        """
        True if employee_id already has an event_type ('CHECK_IN' or
        'CHECK_OUT') logged on today_date ('YYYY-MM-DD', in your chosen
        timezone). Used by the gate daemon to make sure a lingering
        employee doesn't get logged twice for the same event type.
        """
        day_start, day_end = self._day_bounds(today_date)
        with self.engine.connect() as conn:
            row = conn.execute(
                sa.select(sa.literal(1))
                .where(
                    self.attendance_events.c.employee_id == employee_id,
                    self.attendance_events.c.event_type == event_type,
                    self.attendance_events.c.event_time >= day_start,
                    self.attendance_events.c.event_time <= day_end,
                )
                .limit(1)
            ).fetchone()
        return row is not None

    @staticmethod
    def _day_bounds(today_date: str = None):
        """('YYYY-MM-DD 00:00:00', 'YYYY-MM-DD 23:59:59') for the given
        date, or for the server's own local today if today_date is None
        (only safe if the server itself runs in your target timezone)."""
        if today_date is None:
            today_date = now_str()[:10]
        return f"{today_date} 00:00:00", f"{today_date} 23:59:59"

    def recent_attendance_events(self, limit=20):
        with self.engine.connect() as conn:
            rows = conn.execute(
                sa.select(
                    self.attendance_events.c.event_id, self.attendance_events.c.employee_id,
                    self.attendance_events.c.camera_id, self.attendance_events.c.event_type,
                    self.attendance_events.c.event_time, self.attendance_events.c.recognition_score,
                    self.attendance_events.c.image_path,
                )
                .order_by(self.attendance_events.c.event_id.desc())
                .limit(limit)
            ).fetchall()
        return [tuple(r) for r in rows]

    def recent_presence_events(self, limit=20):
        with self.engine.connect() as conn:
            rows = conn.execute(
                sa.select(
                    self.presence_events.c.presence_id, self.presence_events.c.employee_id,
                    self.presence_events.c.camera_id, self.presence_events.c.sample_time,
                    self.presence_events.c.status, self.presence_events.c.recognition_score,
                    self.presence_events.c.image_path,
                )
                .order_by(self.presence_events.c.presence_id.desc())
                .limit(limit)
            ).fetchall()
        return [tuple(r) for r in rows]

    def close(self):
        self.engine.dispose()