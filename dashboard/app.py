import os
from pathlib import Path

from flask import Flask, jsonify, render_template

app = Flask(__name__)

_db = None
_base_dir: Path = Path(__file__).parent.parent


def init_app(db, base_dir: Path | None = None):
    global _db, _base_dir
    _db = db
    if base_dir:
        _base_dir = base_dir


@app.route("/")
def index():
    stats = _db.get_channel_stats() if _db else []
    recent = _db.get_recent_uploads(limit=10) if _db else []
    queued = len(_db.get_jobs(status="queued", limit=100)) if _db else 0
    running = len(_db.get_jobs(status="running", limit=10)) if _db else 0
    return render_template("index.html", stats=stats, recent=recent,
                           queued=queued, running=running)


@app.route("/channel/<channel_id>")
def channel_detail(channel_id: str):
    jobs = _db.get_jobs(channel_id=channel_id, limit=30) if _db else []
    uploads = _db.get_recent_uploads(channel_id=channel_id, limit=10) if _db else []
    return render_template("channel.html", channel_id=channel_id,
                           jobs=jobs, uploads=uploads)


@app.route("/job/<int:job_id>")
def job_detail(job_id: int):
    job = _db.get_job(job_id) if _db else None
    log_tail = _get_log_tail(job.get("channel_id", "") if job else "", 50)
    return render_template("job.html", job=job, log_tail=log_tail)


@app.route("/api/status")
def api_status():
    if not _db:
        return jsonify({"error": "db not initialized"}), 500
    queued_jobs = _db.get_jobs(status="queued", limit=100)
    running_jobs = _db.get_jobs(status="running", limit=10)
    completed = _db.get_jobs(status="completed", limit=1)
    return jsonify({
        "running_jobs": len(running_jobs),
        "queued_jobs": len(queued_jobs),
        "last_completed": completed[0]["completed_at"] if completed else None,
        "running": [{"id": j["id"], "channel": j["channel_id"],
                     "topic": j["topic"]} for j in running_jobs],
    })


def _get_log_tail(channel_id: str, lines: int) -> list[str]:
    log_dir = _base_dir / "logs"
    log_file = log_dir / f"{channel_id}.log"
    if not log_file.exists():
        log_file = log_dir / "system.log"
    if not log_file.exists():
        return []
    try:
        with open(log_file, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return [ln.rstrip() for ln in all_lines[-lines:]]
    except Exception:
        return []
