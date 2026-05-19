"""
Entry point: starts APScheduler + orchestrator + Flask dashboard.
Run with: python scripts/run_scheduler.py
On VM:    gunicorn -w 1 -t 3600 -b 127.0.0.1:8080 'scripts.run_scheduler:create_app()'
"""
import logging
import os
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from core.config_loader import load_master_infra, load_dashboard_config
from core.state_db import StateDB
from core import scheduler as sched
from core import orchestrator
from dashboard.app import app as flask_app, init_app

BASE_DIR = Path(__file__).parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s UTC [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "logs" / "system.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def create_app():
    """Factory for gunicorn: gunicorn 'scripts.run_scheduler:create_app()'"""
    _bootstrap()
    return flask_app


def _bootstrap():
    infra = load_master_infra(BASE_DIR)
    db_path = BASE_DIR / infra.get("db_path", "data/jobs.db")
    workspace_root = BASE_DIR / infra.get("workspace_dir", "workspace")
    workspace_root.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "logs").mkdir(exist_ok=True)

    db = StateDB(db_path)
    db.reset_orphaned_running_jobs()

    init_app(db, BASE_DIR)

    sched.start(db=db, db_path=str(db_path), base_dir=BASE_DIR)
    orchestrator.start_background(
        db=db,
        workspace_root=workspace_root,
        base_dir=BASE_DIR,
        poll_interval=int(os.environ.get("POLL_INTERVAL", "30")),
        max_retries=int(infra.get("max_retries", 3)),
        retry_backoff=int(infra.get("retry_backoff_seconds", 300)),
    )
    log.info("Bootstrap complete. Scheduler + orchestrator running.")


if __name__ == "__main__":
    _bootstrap()
    dash_cfg = load_dashboard_config(BASE_DIR)
    host = dash_cfg.get("host", "127.0.0.1")
    port = int(dash_cfg.get("port", 8080))
    log.info(f"Dashboard: http://{host}:{port}")
    flask_app.run(host=host, port=port, use_reloader=False)
