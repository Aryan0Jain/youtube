"""
DEV LAYER 6 — Thumbnail Maker
Reads dev/workspace/final_video.mp4 + meta.json, produces thumbnail.jpg.

Run:
  python scripts/dev_stage6_thumbnail.py

Inputs:
  dev/workspace/final_video.mp4
  dev/workspace/meta.json         (title used as overlay text)

Output:
  dev/workspace/thumbnail.jpg

Iterate on this layer without touching the video or earlier stages.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("dev_stage6")

# ── Config ────────────────────────────────────────────────────────────────────
NICHE   = "horror"
FORMAT  = "full_length"
TITLE   = ""   # leave blank to read from meta.json
CHANNEL = "horror_stories"
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DEV_WS   = BASE_DIR / "dev" / "workspace"

video_path = DEV_WS / "final_video.mp4"
if not video_path.exists():
    log.error("dev/workspace/final_video.mp4 not found -- run dev_stage5_video.py first")
    sys.exit(1)

meta_path = DEV_WS / "meta.json"
title = TITLE
if not title and meta_path.exists():
    meta  = json.loads(meta_path.read_text(encoding="utf-8"))
    title = meta.get("title", "")
if not title:
    title = "The Dyatlov Pass Incident"

from formats import get_format_spec
from pipeline.thumbnail_maker import make_thumbnail

spec   = get_format_spec(FORMAT)
output = DEV_WS / "thumbnail.jpg"

log.info(f"Generating thumbnail  title={title!r}  niche={NICHE}")
make_thumbnail(
    video_path=video_path,
    title=title,
    niche=NICHE,
    channel_id=CHANNEL,
    spec=spec,
    output_path=output,
    base_dir=BASE_DIR,
)
size_kb = output.stat().st_size // 1024
log.info(f"thumbnail.jpg -> {size_kb} KB")
log.info("Layer 6 done. Open dev/workspace/thumbnail.jpg to review.")
log.info("Happy with all layers? Run the full pipeline with:")
log.info("  USE_MOCK_YOUTUBE=1 python scripts/run_job_now.py \\")
log.info("    --channel horror_stories --series horror_explained \\")
log.info("    --topic \"The Dyatlov Pass Incident\"")
