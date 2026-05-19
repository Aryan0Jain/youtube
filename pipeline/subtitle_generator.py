"""
SubtitleGenerator — Stage 3 of the pipeline.

Transcribes the TTS audio with faster-whisper (Whisper "base" model, CPU int8)
to get word-level timestamps, then groups words into SRT subtitle entries.

Falls back to estimated-timing SRT from the script text when:
  - USE_MOCK_TTS=1 (silent audio → nothing to transcribe)
  - faster-whisper not installed
  - Whisper transcription produces no output
"""
import logging
import os
import re
from pathlib import Path

from pipeline.base import PipelineStage, JobContext

log = logging.getLogger(__name__)

USE_MOCK = os.environ.get("USE_MOCK_TTS") == "1"

# Whisper model size: "tiny" (~75 MB), "base" (~145 MB), "small" (~470 MB)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

# Words per subtitle line (3-5 feels natural, 5 is good for English)
WORDS_PER_LINE = 5

# Estimated words-per-second for mock SRT timing
MOCK_WPS = 2.5


class SubtitleGenerator(PipelineStage):
    name = "subtitle_generator"

    def execute(self, ctx: JobContext) -> JobContext:
        srt_path = ctx.workspace / "subtitles.srt"

        if USE_MOCK or not (ctx.audio_path and ctx.audio_path.exists()):
            log.info("SubtitleGenerator: using estimated SRT from script text")
            _write_estimated_srt(ctx.script_text or "", srt_path)
            ctx.subtitle_path = srt_path
            return ctx

        try:
            _transcribe_to_srt(ctx.audio_path, srt_path)
            log.info(f"Whisper subtitles: {srt_path} ({srt_path.stat().st_size} bytes)")
        except Exception as exc:
            log.warning(
                f"Whisper transcription failed ({exc}). "
                "Falling back to estimated SRT from script text."
            )
            _write_estimated_srt(ctx.script_text or "", srt_path)

        ctx.subtitle_path = srt_path
        return ctx

    def _load_from_checkpoint(self, ctx: JobContext) -> JobContext:
        srt_path = ctx.workspace / "subtitles.srt"
        if srt_path.exists():
            ctx.subtitle_path = srt_path
        return ctx


# ── Whisper transcription ────────────────────────────────────────────────────

def _transcribe_to_srt(audio_path: Path, srt_path: Path) -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper not installed. "
            "Run: pip install faster-whisper"
        )

    log.info(f"Loading Whisper model '{WHISPER_MODEL}' (downloads on first use ~145 MB) ...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    log.info(f"Transcribing {audio_path.name} ...")
    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
        beam_size=5,
    )

    entries: list[str] = []
    idx = 1

    for seg in segments:
        if not seg.words:
            # No word-level data for this segment — treat as one subtitle line
            start = _srt_time(seg.start)
            end = _srt_time(seg.end)
            text = seg.text.strip()
            if text:
                entries.append(f"{idx}\n{start} --> {end}\n{text}\n")
                idx += 1
            continue

        # Punctuation-aware grouping: break at sentence-end and natural pauses
        for group in _group_words_into_lines(list(seg.words)):
            start = _srt_time(group[0].start)
            end = _srt_time(group[-1].end)
            text = " ".join(w.word.strip() for w in group).strip()
            if text:
                entries.append(f"{idx}\n{start} --> {end}\n{text}\n")
                idx += 1

    if not entries:
        raise RuntimeError("Whisper produced no subtitle entries")

    srt_path.write_text("\n".join(entries), encoding="utf-8")
    log.info(f"  → {idx - 1} subtitle entries written")


# ── Fallback: estimated timing from script ───────────────────────────────────

def _write_estimated_srt(script_text: str, srt_path: Path) -> None:
    """
    Generate an SRT file with estimated timing from the script text.
    Assumes MOCK_WPS words per second (good approximation for ElevenLabs / Neural2 TTS).
    """
    # Collapse whitespace and split into words
    words = re.sub(r"\s+", " ", script_text.strip()).split()
    if not words:
        srt_path.write_text("", encoding="utf-8")
        return

    entries: list[str] = []
    t = 0.0
    idx = 1

    for i in range(0, len(words), WORDS_PER_LINE):
        group = words[i : i + WORDS_PER_LINE]
        duration = len(group) / MOCK_WPS
        start = _srt_time(t)
        end = _srt_time(t + duration)
        text = " ".join(group)
        entries.append(f"{idx}\n{start} --> {end}\n{text}\n")
        t += duration
        idx += 1

    srt_path.write_text("\n".join(entries), encoding="utf-8")
    log.info(f"Estimated SRT: {idx - 1} entries, ~{t:.1f}s total")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _group_words_into_lines(words: list, max_words: int = WORDS_PER_LINE) -> list[list]:
    """
    Group Whisper word objects into subtitle lines using punctuation-aware breaks.

    Break rules (in priority order):
    1. After sentence-ending punctuation (. ! ?) — always break here.
    2. After a comma if the group already has 3+ words — natural speech pause.
    3. After max_words words — hard cap to prevent very long lines.

    This produces lines that align with natural speech rhythm instead of cutting
    mid-phrase every N words regardless of sentence structure.
    """
    groups: list[list] = []
    current: list = []

    for w in words:
        current.append(w)
        text = w.word.strip()

        is_sentence_end = text.endswith(('.', '!', '?'))
        is_natural_break = text.endswith(',') and len(current) >= 3
        is_max_words = len(current) >= max_words

        if is_sentence_end or is_natural_break or is_max_words:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    return groups


def _srt_time(seconds: float) -> str:
    """Format a time value as HH:MM:SS,mmm for SRT."""
    seconds = max(0.0, seconds)
    ms = int(round((seconds % 1) * 1000))
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
