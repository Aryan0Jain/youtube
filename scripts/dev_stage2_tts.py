"""
DEV LAYER 2 — TTS Generator
Reads dev/workspace/script.txt and generates audio.

Run:
  python scripts/dev_stage2_tts.py

Outputs:
  dev/workspace/audio.mp3

Provider priority:
  1. ElevenLabs  (if ELEVENLABS_API_KEY is set and not a placeholder)
  2. Google TTS  (fallback)
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("dev_stage2")

# ── Config ────────────────────────────────────────────────────────────────────
NICHE  = "horror"
FORMAT = "full_length"
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DEV_WS   = BASE_DIR / "dev" / "workspace"

script_path = DEV_WS / "script.txt"
if not script_path.exists():
    log.error("dev/workspace/script.txt not found -- run dev_stage1_script.py first")
    sys.exit(1)

script = script_path.read_text(encoding="utf-8")
audio_path = DEV_WS / "audio.mp3"

from formats import get_format_spec
spec = get_format_spec(FORMAT)

# ── Provider selection ────────────────────────────────────────────────────────
elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "")
is_placeholder = not elevenlabs_key or elevenlabs_key.startswith("your_") or len(elevenlabs_key) < 20

if not is_placeholder:
    try:
        from integrations.elevenlabs_client import generate_audio
        log.info(f"ElevenLabs TTS -> niche={NICHE}")
        generate_audio(script_text=script, niche=NICHE, output_path=audio_path)
        log.info(f"audio.mp3  -> {audio_path.stat().st_size // 1024} KB  (ElevenLabs)")
        log.info("Layer 2 done. Next: python scripts/dev_stage3_subtitles.py")
        sys.exit(0)
    except Exception as exc:
        log.warning(f"ElevenLabs failed ({exc}), falling back to Google TTS")

# ── Google TTS ────────────────────────────────────────────────────────────────
import io
from google.cloud import texttospeech
from pydub import AudioSegment

VOICE_MAP = {
    "horror":       "en-US-Neural2-J",
    "shock_facts":  "en-US-Neural2-D",
    "quiz":         "en-US-Neural2-F",
    "ranking":      "en-US-Neural2-D",
    "what_if":      "en-US-Neural2-J",
    "myth_busting": "en-US-Neural2-J",
}
voice_name = VOICE_MAP.get(NICHE, "en-US-Neural2-D")
rate = spec.speaking_rate_multiplier

import re
def split_text(text, max_bytes=4800):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        candidate = (current + " " + s).strip()
        if len(candidate.encode()) > max_bytes:
            if current: chunks.append(current)
            current = s
        else:
            current = candidate
    if current: chunks.append(current)
    return chunks or [text[:max_bytes]]

client = texttospeech.TextToSpeechClient()
chunks = split_text(script)
log.info(f"Google TTS: {len(chunks)} chunk(s), voice={voice_name}")

segments = []
for i, chunk in enumerate(chunks):
    resp = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=chunk),
        voice=texttospeech.VoiceSelectionParams(
            language_code="-".join(voice_name.split("-")[:2]), name=voice_name),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=rate, pitch=0.0),
    )
    segments.append(resp.audio_content)
    log.info(f"  chunk {i+1}/{len(chunks)}: {len(resp.audio_content)//1024} KB")

if len(segments) == 1:
    audio_path.write_bytes(segments[0])
else:
    combined = AudioSegment.empty()
    for seg in segments:
        combined += AudioSegment.from_mp3(io.BytesIO(seg))
    combined.export(str(audio_path), format="mp3")

log.info(f"audio.mp3  -> {audio_path.stat().st_size // 1024} KB  (Google TTS)")
log.info("Layer 2 done. Next: python scripts/dev_stage3_subtitles.py")
