import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id       TEXT NOT NULL,
    series_id        TEXT NOT NULL,
    topic            TEXT NOT NULL,
    format           TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    checkpoint       TEXT,
    created_at       TEXT NOT NULL,
    scheduled_for    TEXT NOT NULL,
    started_at       TEXT,
    completed_at     TEXT,
    retry_count      INTEGER DEFAULT 0,
    error_message    TEXT,
    youtube_video_id TEXT,
    workspace_path   TEXT
);

CREATE TABLE IF NOT EXISTS used_topics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id  TEXT NOT NULL,
    series_id   TEXT NOT NULL,
    topic       TEXT NOT NULL,
    used_at     TEXT NOT NULL,
    job_id      INTEGER REFERENCES jobs(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_used_topics ON used_topics(channel_id, series_id, topic);

CREATE TABLE IF NOT EXISTS uploads (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           INTEGER REFERENCES jobs(id),
    channel_id       TEXT NOT NULL,
    youtube_video_id TEXT NOT NULL,
    title            TEXT,
    published_at     TEXT,
    url              TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_channel ON jobs(channel_id);
CREATE INDEX IF NOT EXISTS idx_jobs_scheduled ON jobs(scheduled_for);
"""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateDB:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # ── Job lifecycle ──────────────────────────────────────────────────────────

    def create_job(self, channel_id: str, series_id: str, topic: str,
                   fmt: str, scheduled_for: str | None = None) -> int:
        now = _now_utc()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO jobs (channel_id, series_id, topic, format, status,
                                    created_at, scheduled_for)
                   VALUES (?, ?, ?, ?, 'queued', ?, ?)""",
                (channel_id, series_id, topic, fmt, now, scheduled_for or now),
            )
            return cur.lastrowid

    def get_next_queued_job(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM jobs WHERE status = 'queued'
                   ORDER BY scheduled_for ASC LIMIT 1"""
            ).fetchone()
            return dict(row) if row else None

    def mark_running(self, job_id: int, workspace_path: str = ""):
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status='running', started_at=?, workspace_path=? WHERE id=?",
                (_now_utc(), workspace_path, job_id),
            )

    def mark_completed(self, job_id: int, youtube_video_id: str = ""):
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status='completed', completed_at=?, youtube_video_id=? WHERE id=?",
                (_now_utc(), youtube_video_id, job_id),
            )

    def mark_failed(self, job_id: int, error: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status='failed', completed_at=?, error_message=? WHERE id=?",
                (_now_utc(), error[:2000], job_id),
            )
        log.error(f"Job {job_id} failed: {error[:200]}")

    def mark_retry(self, job_id: int, error: str):
        with self._conn() as conn:
            conn.execute(
                """UPDATE jobs SET status='queued', retry_count=retry_count+1,
                   error_message=?, checkpoint=NULL WHERE id=?""",
                (error[:2000], job_id),
            )
        log.warning(f"Job {job_id} queued for retry: {error[:100]}")

    # ── Checkpoint (pipeline resume) ───────────────────────────────────────────

    def get_checkpoint(self, job_id: int) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT checkpoint FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            return row["checkpoint"] if row else None

    def set_checkpoint(self, job_id: int, stage_name: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET checkpoint=? WHERE id=?", (stage_name, job_id)
            )

    # ── Topic deduplication ────────────────────────────────────────────────────

    def record_used_topic(self, channel_id: str, series_id: str,
                          topic: str, job_id: int | None = None):
        with self._conn() as conn:
            try:
                conn.execute(
                    """INSERT INTO used_topics (channel_id, series_id, topic, used_at, job_id)
                       VALUES (?, ?, ?, ?, ?)""",
                    (channel_id, series_id, topic, _now_utc(), job_id),
                )
            except sqlite3.IntegrityError:
                pass  # already recorded

    def topic_was_used_recently(self, channel_id: str, series_id: str,
                                topic: str, days: int = 90) -> bool:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            row = conn.execute(
                """SELECT 1 FROM used_topics
                   WHERE channel_id=? AND series_id=? AND topic=? AND used_at > ?
                   LIMIT 1""",
                (channel_id, series_id, topic, cutoff),
            ).fetchone()
            return row is not None

    def get_recent_used_topics(self, channel_id: str, series_id: str,
                               days: int = 90) -> list[str]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT topic FROM used_topics
                   WHERE channel_id=? AND series_id=? AND used_at > ?
                   ORDER BY used_at DESC""",
                (channel_id, series_id, cutoff),
            ).fetchall()
            return [r["topic"] for r in rows]

    # ── Upload tracking ────────────────────────────────────────────────────────

    def record_upload(self, job_id: int, channel_id: str,
                      youtube_video_id: str, title: str, url: str):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO uploads (job_id, channel_id, youtube_video_id, title, published_at, url)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (job_id, channel_id, youtube_video_id, title, _now_utc(), url),
            )

    # ── Dashboard queries ──────────────────────────────────────────────────────

    def get_jobs(self, channel_id: str | None = None,
                 status: str | None = None, limit: int = 50) -> list[dict]:
        where_clauses = []
        params: list[Any] = []
        if channel_id:
            where_clauses.append("channel_id = ?")
            params.append(channel_id)
        if status:
            where_clauses.append("status = ?")
            params.append(status)
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs {where} ORDER BY scheduled_for DESC LIMIT ?", params
            ).fetchall()
            return [dict(r) for r in rows]

    def get_job(self, job_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def get_channel_stats(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT channel_id,
                          COUNT(*) AS total,
                          SUM(status='completed') AS completed,
                          SUM(status='failed') AS failed,
                          SUM(status='queued') AS queued,
                          SUM(status='running') AS running,
                          MAX(completed_at) AS last_completed
                   FROM jobs
                   GROUP BY channel_id"""
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_uploads(self, channel_id: str | None = None, limit: int = 20) -> list[dict]:
        where = "WHERE channel_id = ?" if channel_id else ""
        params = [channel_id, limit] if channel_id else [limit]
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM uploads {where} ORDER BY published_at DESC LIMIT ?", params
            ).fetchall()
            return [dict(r) for r in rows]

    def reset_orphaned_running_jobs(self):
        """Reset any jobs stuck in 'running' from a previous crash back to queued."""
        with self._conn() as conn:
            result = conn.execute(
                """UPDATE jobs SET status='queued', retry_count=retry_count+1,
                   error_message='Reset from orphaned running state (process crash)'
                   WHERE status='running'"""
            )
            count = result.rowcount
        if count:
            log.warning(f"Reset {count} orphaned running job(s) to queued")
