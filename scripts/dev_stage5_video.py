"""
DEV LAYER 5 — Video Assembler
Reads from dev/workspace/ and produces dev/workspace/final_video.mp4.

Run:
  python scripts/dev_stage5_video.py

Inputs (must exist):
  dev/workspace/audio.mp3
  dev/workspace/clips/*.mp4
  dev/workspace/subtitles.srt   (optional — skipped if missing)

Output:
  dev/workspace/final_video.mp4

Iterate on this layer freely — it only touches local files, no API calls.
Edit pipeline/video_assembler.py and re-run to see changes immediately.
"""
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("dev_stage5")

# ── Config ────────────────────────────────────────────────────────────────────
NICHE        = "horror"
FORMAT       = "full_length"
MUSIC_VOLUME = 0.08
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DEV_WS   = BASE_DIR / "dev" / "workspace"

# ── Validate inputs ───────────────────────────────────────────────────────────
audio_path = DEV_WS / "audio.mp3"
if not audio_path.exists():
    log.error("dev/workspace/audio.mp3 not found -- run dev_stage2_tts.py first")
    sys.exit(1)

clip_paths = sorted((DEV_WS / "clips").glob("*.mp4"))
if not clip_paths:
    log.error("No clips in dev/workspace/clips/ -- run dev_stage4_clips.py first")
    sys.exit(1)

subtitle_path = DEV_WS / "subtitles.srt"
if not subtitle_path.exists():
    log.warning("No subtitles.srt found -- video will be produced without captions")
    subtitle_path = None

log.info(f"Audio   : {audio_path.name}  ({audio_path.stat().st_size // 1024} KB)")
log.info(f"Clips   : {len(clip_paths)} files")
log.info(f"Subtitles: {'yes' if subtitle_path else 'no'}")

# ── Fresh tmp dir so each run starts clean ────────────────────────────────────
tmp_dir = DEV_WS / "tmp_clips"
if tmp_dir.exists():
    shutil.rmtree(tmp_dir, ignore_errors=True)

# ── Run assembler ─────────────────────────────────────────────────────────────
from formats import get_format_spec
from pipeline.video_assembler import assemble_video

spec = get_format_spec(FORMAT)

# ── Load niche metadata (populated by dev_stage1_script.py if it ran) ─────────
import json as _json
niche_metadata_path = DEV_WS / "niche_metadata.json"
niche_metadata: dict = {}
if niche_metadata_path.exists():
    try:
        niche_metadata = _json.loads(niche_metadata_path.read_text(encoding="utf-8"))
        overlay_type = niche_metadata.get("overlay_type", "none")
        items = niche_metadata.get("items", [])
        count = len(items) if isinstance(items, list) else (1 if items else 0)
        log.info(f"Loaded niche_metadata: overlay={overlay_type}, items={count}")
    except Exception as e:
        log.warning(f"Could not load niche_metadata.json: {e}")
else:
    log.info("No niche_metadata.json found -- overlays will be skipped")

script_text = ""
script_path = DEV_WS / "script.txt"
if script_path.exists():
    script_text = script_path.read_text(encoding="utf-8")

result = assemble_video(
    clip_paths=clip_paths,
    audio_path=audio_path,
    subtitle_path=subtitle_path,
    niche=NICHE,
    spec=spec,
    music_volume=MUSIC_VOLUME,
    workspace=DEV_WS,
    niche_metadata=niche_metadata,
    script_text=script_text,
)

size_mb = result.stat().st_size / 1024 / 1024
log.info(f"Output  : {result}  ({size_mb:.1f} MB)")
log.info("Layer 5 done. Open dev/workspace/final_video.mp4 to review.")
log.info("Happy with it? Next: python scripts/dev_stage6_thumbnail.py")
