"""
DEV LAYER 1 — Script Writer
Generates a YouTube script and saves to dev/workspace/script.txt.
Uses niche-specific system prompt from config/niches.yaml.

Run:
  python scripts/dev_stage1_script.py

Outputs:
  dev/workspace/script.txt           <- spoken script
  dev/workspace/meta.json            <- title, description, tags
  dev/workspace/niche_metadata.json  <- overlay data for dev_stage5_video.py
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("dev_stage1")

# ── Config (edit these to change what gets generated) ─────────────────────────
TOPIC       = "The Dyatlov Pass Incident"
NICHE       = "horror"
STYLE_NOTES = (
    "Deep-dive educational explanation of a real horror event. "
    "Serious tone, second-person narration, build dread gradually. "
    "End with an unsettling open question."
)
FORMAT      = "full_length"
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DEV_WS   = BASE_DIR / "dev" / "workspace"
DEV_WS.mkdir(parents=True, exist_ok=True)

from formats import get_format_spec
from integrations import claude_client
from pipeline.niche_config import get_niche_profile
from core.config_loader import load_master_infra

spec    = get_format_spec(FORMAT)
profile = get_niche_profile(NICHE)
infra   = load_master_infra(BASE_DIR)
claude_cfg  = infra.get("claude", {})
model       = claude_cfg.get("model", "claude-opus-4-5")
haiku_model = claude_cfg.get("haiku_model", "claude-haiku-4-5-20251001")
max_tokens  = claude_cfg.get("max_tokens", 4096)

log.info(f"Niche: {NICHE} | Target: {profile.target_word_count} words | Overlay: {profile.overlay_type}")
log.info(f"Generating script for: {TOPIC!r}")

script = claude_client.write_script(
    topic=TOPIC,
    style_notes=STYLE_NOTES,
    niche=NICHE,
    script_prompt_suffix=spec.script_prompt_suffix,
    target_word_count=profile.target_word_count,
    model=model,
    max_tokens=max_tokens,
)
(DEV_WS / "script.txt").write_text(script, encoding="utf-8")
word_count = len(script.split())
log.info(f"script.txt  -> {word_count} words  (target: {profile.target_word_count})")
if word_count > profile.target_word_count * 1.15:
    log.warning(f"Script is {word_count - profile.target_word_count} words over target -- "
                "consider re-running or editing script.txt")

# ── SEO metadata ──────────────────────────────────────────────────────────────
log.info("Generating SEO metadata (title/description/tags) ...")
meta_raw = claude_client.generate_video_metadata(
    topic=TOPIC,
    niche=NICHE,
    script_text=script,
    haiku_model=haiku_model,
)
title = meta_raw["title"] + spec.title_suffix
meta = {
    "title": title,
    "description": meta_raw["description"],
    "tags": meta_raw["tags"],
}
(DEV_WS / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
log.info(f"meta.json   -> {title!r}")

# ── Niche metadata (overlay data) ─────────────────────────────────────────────
log.info(f"Extracting niche overlay metadata (overlay_type={profile.overlay_type}) ...")
niche_metadata = claude_client.extract_niche_metadata(
    script_text=script,
    niche=NICHE,
    haiku_model=haiku_model,
)
if niche_metadata:
    (DEV_WS / "niche_metadata.json").write_text(
        json.dumps(niche_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    items = niche_metadata.get("items", [])
    count = len(items) if isinstance(items, list) else (1 if items else 0)
    log.info(f"niche_metadata.json -> overlay={niche_metadata.get('overlay_type')}, items={count}")
else:
    log.info("No niche metadata extracted (overlay type may be 'none' or extraction failed)")

log.info("Layer 1 done. Next: python scripts/dev_stage2_tts.py")
