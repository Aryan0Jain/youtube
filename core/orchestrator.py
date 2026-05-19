import logging
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from core.state_db import StateDB
from core.config_loader import load_all
from formats import get_format_spec
from pipeline.base import JobContext, PipelineStageError
from pipeline.script_writer import ScriptWriter
from pipeline.tts_generator import TTSGenerator
from pipeline.subtitle_generator import SubtitleGenerator
from pipeline.clip_fetcher import ClipFetcher
from pipeline.video_assembler import VideoAssembler
from pipeline.thumbnail_maker import ThumbnailMaker
from pipeline.youtube_uploader import YouTubeUploader

log = logging.getLogger(__name__)

STAGES = [
    ScriptWriter(),
    TTSGenerator(),
    SubtitleGenerator(),   # word-level captions from the TTS audio
    ClipFetcher(),
    VideoAssembler(),
    ThumbnailMaker(),
    YouTubeUploader(),
]

_pipeline_lock = threading.Lock()
_stop_event = threading.Event()


def _build_context(job: dict, db: StateDB, workspace_root: Path,
                   base_dir: Path | None) -> JobContext:
    channels = load_all(base_dir)
    channel = channels.get(job["channel_id"])
    if not channel:
        raise ValueError(f"Channel '{job['channel_id']}' not found in config")

    series = next((s for s in channel.series if s.series_id == job["series_id"]), None)
    if not series:
        raise ValueError(f"Series '{job['series_id']}' not found in channel '{job['channel_id']}'")

    spec = get_format_spec(job["format"])
    workspace = workspace_root / f"job_{job['id']}_{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=True)

    return JobContext(
        job_id=job["id"],
        channel_id=job["channel_id"],
        series_id=job["series_id"],
        topic=job["topic"],
        format=job["format"],
        niche=series.niche,
        style_notes=series.style_notes,
        resolved=series.resolved,
        format_spec=spec,
        workspace=workspace,
        db=db,
    )


def run_pipeline(job: dict, db: StateDB, workspace_root: Path,
                 base_dir: Path | None = None) -> str:
    """
    Run all pipeline stages for a job. Returns the YouTube video ID on success.
    Raises PipelineStageError on stage failure.
    """
    ctx = _build_context(job, db, workspace_root, base_dir)
    db.mark_running(job["id"], str(ctx.workspace))

    for stage in STAGES:
        ctx = stage.run(ctx)

    return ctx.youtube_video_id or ""


def _run_loop(db: StateDB, workspace_root: Path, base_dir: Path | None,
              poll_interval: int = 30, max_retries: int = 3,
              retry_backoff: int = 300):
    """Background thread: poll SQLite for queued jobs and run them one at a time."""
    log.info("Orchestrator loop started")

    while not _stop_event.is_set():
        acquired = _pipeline_lock.acquire(blocking=False)
        if not acquired:
            time.sleep(poll_interval)
            continue

        job = None
        try:
            job = db.get_next_queued_job()
            if job is None:
                _pipeline_lock.release()
                time.sleep(poll_interval)
                continue

            log.info(
                f"Starting job#{job['id']}: "
                f"[{job['channel_id']}/{job['series_id']}] {job['topic']!r}"
            )

            video_id = run_pipeline(job, db, workspace_root, base_dir)
            db.mark_completed(job["id"], video_id)
            log.info(f"Completed job#{job['id']} → {video_id}")

            # Clean up workspace
            workspace = Path(job.get("workspace_path") or workspace_root / f"job_{job['id']}_*")
            _cleanup_workspace(db, job)

        except PipelineStageError as exc:
            if job:
                retry_count = job.get("retry_count", 0)
                if retry_count < max_retries:
                    log.warning(f"Job#{job['id']} stage error, will retry: {exc}")
                    db.mark_retry(job["id"], str(exc))
                    time.sleep(retry_backoff)
                else:
                    log.error(f"Job#{job['id']} exhausted retries: {exc}")
                    db.mark_failed(job["id"], str(exc))
        except Exception:
            tb = traceback.format_exc()
            log.error(f"Unexpected error in orchestrator:\n{tb}")
            if job:
                db.mark_failed(job["id"], tb[-2000:])
        finally:
            if _pipeline_lock.locked():
                try:
                    _pipeline_lock.release()
                except RuntimeError:
                    pass

        time.sleep(1)

    log.info("Orchestrator loop stopped")


def _cleanup_workspace(db: StateDB, job: dict):
    ws_path = job.get("workspace_path")
    if ws_path and Path(ws_path).exists():
        try:
            shutil.rmtree(ws_path)
            log.info(f"Cleaned workspace: {ws_path}")
        except Exception as e:
            log.warning(f"Could not clean workspace {ws_path}: {e}")


def start_background(db: StateDB, workspace_root: Path, base_dir: Path | None = None,
                     poll_interval: int = 30, max_retries: int = 3,
                     retry_backoff: int = 300) -> threading.Thread:
    """Start the orchestrator in a daemon background thread."""
    thread = threading.Thread(
        target=_run_loop,
        args=(db, workspace_root, base_dir, poll_interval, max_retries, retry_backoff),
        daemon=True,
        name="orchestrator",
    )
    thread.start()
    return thread


def stop():
    _stop_event.set()
