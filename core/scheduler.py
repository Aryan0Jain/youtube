import logging
import os
import signal
import threading
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.cron import CronTrigger

from core.config_loader import load_all, SeriesConfig, ChannelConfig
from core.state_db import StateDB
from integrations import sheets_client
from integrations import claude_client

log = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None
_db: StateDB | None = None
_base_dir = None


def _enqueue_series_job(channel_id: str, series_id: str):
    """Called by APScheduler when a series cron fires. Claims a topic and enqueues a job."""
    global _db, _base_dir

    try:
        channels = load_all(_base_dir)
        channel = channels.get(channel_id)
        if not channel:
            log.error(f"Scheduler fired for unknown channel '{channel_id}'")
            return

        series = next((s for s in channel.series if s.series_id == series_id), None)
        if not series:
            log.error(f"Scheduler fired for unknown series '{channel_id}/{series_id}'")
            return

        # Guard 1: don't produce a new video while the previous one is still
        # sitting in the Telegram review queue (unapproved/unpublished).
        if _db.has_pending_review(channel_id):
            log.info(
                f"[{channel_id}/{series_id}] Skipping schedule — previous video still "
                "awaiting Telegram review. Approve or reject it first."
            )
            return

        # Guard 2: a job for this series is already queued or running — no need
        # to add another one (handles cases where the cron fires twice or the
        # previous run stalled).
        if _db.has_active_job(channel_id, series_id):
            log.info(
                f"[{channel_id}/{series_id}] Skipping schedule — job already "
                "queued or running."
            )
            return

        topic = _claim_topic(channel, series)
        if not topic:
            return  # warning already logged in _claim_topic

        job_id = _db.create_job(
            channel_id=channel_id,
            series_id=series_id,
            topic=topic,
            fmt=series.format,
            scheduled_for=datetime.now(timezone.utc).isoformat(),
        )
        _db.record_used_topic(channel_id, series_id, topic, job_id)
        log.info(f"Enqueued job#{job_id}: [{channel_id}/{series_id}] {topic!r}")

    except Exception:
        log.exception(f"Error enqueuing job for {channel_id}/{series_id}")


def _claim_topic(channel: ChannelConfig, series: SeriesConfig) -> str | None:
    ts = series.topic_source
    claude_cfg = series.resolved.get("claude", {})
    haiku_model = claude_cfg.get("haiku_model", "claude-haiku-4-5-20251001")

    if ts.type == "google_sheet":
        use_mock = os.environ.get("USE_MOCK_SHEETS") == "1"
        if use_mock:
            return f"Mock topic for {series.series_id} at {datetime.now(timezone.utc).strftime('%H:%M')}"

        result = sheets_client.get_next_unused_topic(
            sheet_id=ts.sheet_id,
            tab_name=ts.tab_name,
            topic_column=ts.topic_column or "A",
            used_column=ts.used_column or "B",
        )
        if result is None:
            log.warning(f"No unused topics left for {channel.channel_id}/{series.series_id}")
            return None

        row_index, topic = result
        sheets_client.mark_topic_used(
            sheet_id=ts.sheet_id,
            tab_name=ts.tab_name,
            row_index=row_index,
            used_column=ts.used_column or "B",
        )
        return topic

    elif ts.type == "claude_autogen":
        use_mock = os.environ.get("USE_MOCK_CLAUDE") == "1"
        if use_mock:
            return f"Auto-generated: {series.niche} topic #{datetime.now(timezone.utc).microsecond}"

        avoid_list: list[str] = []
        if ts.avoid_recent_topics:
            avoid_list = _db.get_recent_used_topics(
                channel.channel_id, series.series_id, days=90
            )

        refreshable_topics: list[str] = []
        if ts.allow_topic_refresh:
            refreshable_topics = _db.get_old_used_topics(
                channel.channel_id, series.series_id,
                older_than_days=ts.refresh_after_days,
            )

        topic = claude_client.generate_topic(
            autogen_prompt=ts.autogen_prompt or "",
            avoid_list=avoid_list,
            refreshable_topics=refreshable_topics,
            refresh_after_days=ts.refresh_after_days,
            haiku_model=haiku_model,
        )
        return topic

    elif ts.type == "trend_autogen":
        use_mock = os.environ.get("USE_MOCK_CLAUDE") == "1"
        if use_mock:
            return f"Trend topic: {series.niche} #{datetime.now(timezone.utc).microsecond}"

        avoid_list: list[str] = []
        if ts.avoid_recent_topics:
            avoid_list = _db.get_recent_used_topics(
                channel.channel_id, series.series_id, days=90
            )

        try:
            from integrations.trend_discovery import find_best_topics
            candidates = find_best_topics(
                niche=series.niche,
                haiku_model=haiku_model,
                count=10,
                opportunity_threshold=ts.opportunity_threshold,
                timeframe=ts.timeframe,
            )
            avoid_lower = {t.lower() for t in avoid_list}
            for candidate in candidates:
                title = candidate.get("topic", "")
                if title and title.lower() not in avoid_lower:
                    log.info(
                        f"Trend topic selected: {title!r} "
                        f"(opportunity={candidate.get('opportunity_score', '?')}, "
                        f"niche_fit={candidate.get('niche_fit', '?')})"
                    )
                    return title
            log.warning(
                f"All {len(candidates)} trend candidates were recently used "
                f"-- falling back to autogen"
            )
        except Exception as exc:
            log.warning(f"Trend discovery failed ({exc}) -- falling back to autogen")

        # Fallback: use Claude autogen with the fallback prompt
        fallback_prompt = (
            ts.fallback_autogen_prompt
            or f"Generate a compelling YouTube video topic for the {series.niche} niche. "
               f"Return only the topic title, nothing else."
        )
        return claude_client.generate_topic(
            autogen_prompt=fallback_prompt,
            avoid_list=avoid_list,
            haiku_model=haiku_model,
        )

    log.error(f"Unknown topic_source type: {ts.type}")
    return None


def _register_channel_series(channel: ChannelConfig, series: SeriesConfig):
    job_id = f"{channel.channel_id}__{series.series_id}"
    trigger = CronTrigger(
        day_of_week=series.schedule.apscheduler_days,
        hour=series.schedule.hour,
        minute=series.schedule.minute,
        timezone="UTC",
    )
    _scheduler.add_job(
        _enqueue_series_job,
        trigger=trigger,
        id=job_id,
        args=[channel.channel_id, series.series_id],
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )
    log.info(f"Registered cron: {job_id} → {series.schedule.days_of_week} {series.schedule.time_utc} UTC")


def _reload_config():
    """Re-read all channel YAMLs and reconcile APScheduler job registry."""
    global _base_dir
    log.info("Reloading channel configs…")
    try:
        channels = load_all(_base_dir)
        wanted_ids: set[str] = set()

        for channel in channels.values():
            for series in channel.series:
                job_id = f"{channel.channel_id}__{series.series_id}"
                wanted_ids.add(job_id)
                _register_channel_series(channel, series)

        # Remove jobs for deleted series
        existing_ids = {job.id for job in _scheduler.get_jobs()}
        for stale_id in existing_ids - wanted_ids:
            _scheduler.remove_job(stale_id)
            log.info(f"Removed stale cron job: {stale_id}")

        log.info(f"Config reload complete. {len(wanted_ids)} series registered.")
    except Exception:
        log.exception("Config reload failed — keeping existing schedule")


def start(db: StateDB, db_path: str, base_dir=None):
    global _scheduler, _db, _base_dir
    _db = db
    _base_dir = base_dir

    jobstores = {"default": SQLAlchemyJobStore(url=f"sqlite:///{db_path}")}
    _scheduler = BackgroundScheduler(jobstores=jobstores, timezone="UTC")
    _scheduler.start()
    log.info("APScheduler started")

    _reload_config()

    # SIGHUP → hot-reload without restart
    def _sighup_handler(signum, frame):
        threading.Thread(target=_reload_config, daemon=True).start()

    try:
        signal.signal(signal.SIGHUP, _sighup_handler)
    except (OSError, ValueError):
        pass  # SIGHUP not available on Windows


def stop():
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("APScheduler stopped")
