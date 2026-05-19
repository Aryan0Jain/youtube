"""
DEV LAYER 4 — Clip Fetcher
Reads dev/workspace/script.txt, extracts keywords with Claude Haiku,
downloads matching clips from Pexels to dev/workspace/clips/.

Run:
  python scripts/dev_stage4_clips.py

Outputs:
  dev/workspace/clips/clip_NNN_keyword.mp4   (up to 10 clips)

NOTE: Pexels rate limit is 200 requests/hour on the free tier.
      Already have good clips? Skip this and go straight to stage 5.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("dev_stage4")

# ── Config ────────────────────────────────────────────────────────────────────
NICHE       = "horror"
FORMAT      = "full_length"
MAX_CLIPS   = 10
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DEV_WS   = BASE_DIR / "dev" / "workspace"

script_path = DEV_WS / "script.txt"
if not script_path.exists():
    log.error("dev/workspace/script.txt not found -- run dev_stage1_script.py first")
    sys.exit(1)

clips_dir = DEV_WS / "clips"
clips_dir.mkdir(exist_ok=True)

from formats import get_format_spec
from integrations import claude_client, pexels_client
from pipeline.niche_config import get_niche_profile
spec = get_format_spec(FORMAT)
profile = get_niche_profile(NICHE)

script = script_path.read_text(encoding="utf-8")
from core.config_loader import load_master_infra
infra = load_master_infra(BASE_DIR)
claude_cfg = infra.get("claude", {})
haiku_model = claude_cfg.get("haiku_model", "claude-haiku-4-5-20251001")

# ── Probe audio duration for accurate segment count ───────────────────────────
import json, subprocess
audio_path = DEV_WS / "audio.mp3"
if audio_path.exists():
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(audio_path)],
            capture_output=True, text=True)
        audio_duration = float(json.loads(r.stdout)["format"]["duration"])
        log.info(f"Audio duration: {audio_duration:.1f}s")
    except Exception as e:
        log.warning(f"Could not probe audio: {e}. Estimating from word count.")
        audio_duration = len(script.split()) / 130.0 * 60.0
else:
    log.warning("No audio.mp3 found -- estimating duration from word count")
    audio_duration = len(script.split()) / 130.0 * 60.0

# ── Per-segment keyword extraction (one keyword per 30s) ─────────────────────
SEGMENT_SEC = 30.0
log.info(f"Extracting per-segment keywords for ~{audio_duration:.0f}s audio ...")
segment_keywords = claude_client.extract_segment_keywords(
    script_text=script,
    segment_duration_sec=SEGMENT_SEC,
    audio_duration_sec=audio_duration,
    haiku_model=haiku_model,
    niche=NICHE,
)
log.info(f"Segment keywords ({len(segment_keywords)}): {segment_keywords}")

# ── Download clips_per_segment clips per keyword ──────────────────────────────
seen_ids: set = set()
all_clips = []
clip_index = 0

for kw in segment_keywords:
    new_clips = pexels_client.download_clips_for_keyword(
        keyword=kw,
        count=profile.clips_per_segment,
        dest_dir=clips_dir,
        clip_index_start=clip_index,
        orientation=spec.clip_orientation,
        min_duration=int(getattr(spec, "clip_min_seconds", 4)),
        seen_ids=seen_ids,
    )
    all_clips.extend(new_clips)
    clip_index += len(new_clips)
    log.info(f"  '{kw}': {len(new_clips)} clips  (total: {len(all_clips)})")

log.info(f"Downloaded {len(all_clips)} clips ({len(seen_ids)} unique Pexels IDs)")
log.info(f"Clips stored in dev/workspace/clips/ ({len(segment_keywords)} keywords x {profile.clips_per_segment} clips/keyword)")
log.info("Layer 4 done. Next: python scripts/dev_stage5_video.py")
