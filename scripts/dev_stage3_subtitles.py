"""
DEV LAYER 3 — Subtitle Generator
Reads dev/workspace/audio.mp3, transcribes with Whisper, writes SRT.

Run:
  python scripts/dev_stage3_subtitles.py

Outputs:
  dev/workspace/subtitles.srt

Options (env vars):
  WHISPER_MODEL=tiny    use "tiny" for faster transcription (~75 MB model)
  WHISPER_MODEL=base    (default) good accuracy, ~145 MB model
  WHISPER_MODEL=small   best accuracy, ~470 MB model
"""
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("dev_stage3")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
WORDS_PER_LINE = 5

BASE_DIR = Path(__file__).parent.parent
DEV_WS   = BASE_DIR / "dev" / "workspace"

audio_path = DEV_WS / "audio.mp3"
if not audio_path.exists():
    log.error("dev/workspace/audio.mp3 not found — run dev_stage2_tts.py first")
    sys.exit(1)

srt_path = DEV_WS / "subtitles.srt"

def srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms = int(round((seconds % 1) * 1000))
    s  = int(seconds) % 60
    m  = int(seconds) // 60 % 60
    h  = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# ── Whisper transcription ─────────────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
    log.info(f"Loading Whisper '{WHISPER_MODEL}' model (first run downloads ~145 MB)...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    log.info(f"Transcribing {audio_path.name}  ({audio_path.stat().st_size // 1024} KB)...")
    segments, info = model.transcribe(str(audio_path), word_timestamps=True, language="en")

    entries, idx = [], 1
    for seg in segments:
        words = list(seg.words) if seg.words else []
        if not words:
            text = seg.text.strip()
            if text:
                entries.append(f"{idx}\n{srt_time(seg.start)} --> {srt_time(seg.end)}\n{text}\n")
                idx += 1
            continue
        for i in range(0, len(words), WORDS_PER_LINE):
            group = words[i:i+WORDS_PER_LINE]
            text  = " ".join(w.word.strip() for w in group).strip()
            if text:
                entries.append(f"{idx}\n{srt_time(group[0].start)} --> {srt_time(group[-1].end)}\n{text}\n")
                idx += 1

    srt_path.write_text("\n".join(entries), encoding="utf-8")
    log.info(f"subtitles.srt → {idx-1} entries  ({srt_path.stat().st_size} bytes)")

except Exception as exc:
    log.warning(f"Whisper failed ({exc}) — using estimated timing from script text")
    script_path = DEV_WS / "script.txt"
    if not script_path.exists():
        log.error("No script.txt either — cannot generate fallback SRT")
        sys.exit(1)
    script = script_path.read_text(encoding="utf-8")
    words = re.sub(r'\s+', ' ', script.strip()).split()
    t, idx, entries = 0.0, 1, []
    for i in range(0, len(words), WORDS_PER_LINE):
        group = words[i:i+WORDS_PER_LINE]
        dur   = len(group) / 2.5
        entries.append(f"{idx}\n{srt_time(t)} --> {srt_time(t+dur)}\n{' '.join(group)}\n")
        t += dur
        idx += 1
    srt_path.write_text("\n".join(entries), encoding="utf-8")
    log.info(f"subtitles.srt → {idx-1} estimated entries")

log.info("Layer 3 done. Next: python scripts/dev_stage4_clips.py")
