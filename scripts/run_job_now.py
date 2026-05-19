"""
CLI: manually trigger a pipeline job, bypassing the scheduler.

Usage:
  python scripts/run_job_now.py \\
    --channel horror_stories \\
    --series real_horror_shorts \\
    --topic "The Winchester Mystery House"

All mock flags (USE_MOCK_TTS=1, etc.) are honoured from the environment.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

BASE_DIR = Path(__file__).parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s UTC [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Manually run a pipeline job")
    parser.add_argument("--channel", required=True, help="channel_id from config")
    parser.add_argument("--series", required=True, help="series_id from config")
    parser.add_argument("--topic", required=True, help="Topic title for the video")
    parser.add_argument("--format", default=None,
                        help="Override format: full_length | shorts")
    args = parser.parse_args()

    from core.config_loader import load_all, load_master_infra
    from core.state_db import StateDB
    from core.orchestrator import run_pipeline

    infra = load_master_infra(BASE_DIR)
    db_path = BASE_DIR / infra.get("db_path", "data/jobs.db")
    workspace_root = BASE_DIR / infra.get("workspace_dir", "workspace")
    workspace_root.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "logs").mkdir(exist_ok=True)

    db = StateDB(db_path)

    channels = load_all(BASE_DIR)
    channel = channels.get(args.channel)
    if not channel:
        log.error(f"Channel '{args.channel}' not found. Available: {list(channels)}")
        sys.exit(1)

    series = next((s for s in channel.series if s.series_id == args.series), None)
    if not series:
        available = [s.series_id for s in channel.series]
        log.error(f"Series '{args.series}' not found in '{args.channel}'. Available: {available}")
        sys.exit(1)

    fmt = args.format or series.format
    job_id = db.create_job(
        channel_id=args.channel,
        series_id=args.series,
        topic=args.topic,
        fmt=fmt,
    )
    db.record_used_topic(args.channel, args.series, args.topic, job_id)

    job = db.get_job(job_id)
    log.info(f"Created job#{job_id}: [{args.channel}/{args.series}] {args.topic!r} ({fmt})")

    try:
        video_id = run_pipeline(job, db, workspace_root, BASE_DIR)
        db.mark_completed(job_id, video_id)
        log.info(f"Job#{job_id} complete. Video: https://youtu.be/{video_id}")
    except Exception as e:
        db.mark_failed(job_id, str(e))
        log.error(f"Job#{job_id} failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
