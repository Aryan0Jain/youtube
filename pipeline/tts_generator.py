import io
import logging
import os
import shutil
from pathlib import Path

from pipeline.base import PipelineStage, JobContext

log = logging.getLogger(__name__)

USE_MOCK = os.environ.get("USE_MOCK_TTS") == "1"


class TTSGenerator(PipelineStage):
    name = "tts_generator"

    def execute(self, ctx: JobContext) -> JobContext:
        audio_path = ctx.workspace / "audio.mp3"

        if USE_MOCK:
            _write_mock_audio(audio_path)
            ctx.audio_path = audio_path
            log.info(f"Mock audio written: {audio_path}")
            return ctx

        # ── Try ElevenLabs first (much more natural) ───────────────────────
        elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "")
        looks_like_placeholder = (
            not elevenlabs_key
            or elevenlabs_key.startswith("your_")
            or len(elevenlabs_key) < 20
        )
        if elevenlabs_key and not looks_like_placeholder:
            try:
                log.info("TTS provider: ElevenLabs")
                from integrations.elevenlabs_client import generate_audio
                generate_audio(
                    script_text=ctx.script_text or "",
                    niche=ctx.niche,
                    output_path=audio_path,
                )
                ctx.audio_path = audio_path
                log.info(f"ElevenLabs audio: {audio_path} ({audio_path.stat().st_size // 1024} KB)")
                return ctx
            except Exception as exc:
                log.warning(f"ElevenLabs TTS failed ({exc}), falling back to Google TTS")

        # ── Fallback: Google Cloud TTS ─────────────────────────────────────
        log.info("TTS provider: Google Cloud TTS (set ELEVENLABS_API_KEY for better quality)")
        _google_tts(ctx, audio_path)
        ctx.audio_path = audio_path
        log.info(f"Google TTS audio: {audio_path} ({audio_path.stat().st_size // 1024} KB)")
        return ctx

    def _load_from_checkpoint(self, ctx: JobContext) -> JobContext:
        audio_path = ctx.workspace / "audio.mp3"
        if audio_path.exists():
            ctx.audio_path = audio_path
        return ctx


# ── Google TTS ──────────────────────────────────────────────────────────────

def _google_tts(ctx: JobContext, audio_path: Path) -> None:
    from google.cloud import texttospeech

    voice_name = ctx.resolved.get("tts_voice", "en-US-Neural2-D")

    # Apply niche-specific speaking rate (from config/niches.yaml) on top of
    # the format-level multiplier and the channel-level resolved rate.
    niche_rate = 1.0
    try:
        from pipeline.niche_config import get_niche_profile
        niche_rate = get_niche_profile(ctx.niche).speaking_rate
    except Exception:
        pass

    speaking_rate = (
        ctx.resolved.get("tts_speaking_rate", 1.0)
        * ctx.format_spec.speaking_rate_multiplier
        * niche_rate
    )
    script = ctx.script_text or ""

    chunks = _split_text(script, max_bytes=4800)
    log.info(f"Google TTS: {len(chunks)} chunk(s), voice={voice_name}, rate={speaking_rate:.2f}")

    client = texttospeech.TextToSpeechClient()
    audio_segments: list[bytes] = []

    for i, chunk in enumerate(chunks):
        synthesis_input = texttospeech.SynthesisInput(text=chunk)
        voice = texttospeech.VoiceSelectionParams(
            language_code="-".join(voice_name.split("-")[:2]),
            name=voice_name,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=0.0,
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        audio_segments.append(response.audio_content)
        log.info(f"  chunk {i+1}/{len(chunks)}: {len(response.audio_content)} bytes")

    if len(audio_segments) == 1:
        audio_path.write_bytes(audio_segments[0])
    else:
        _concatenate_mp3(audio_segments, audio_path)


def _split_text(text: str, max_bytes: int = 4800) -> list[str]:
    sentences = _sentence_split(text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip()
        if len(candidate.encode("utf-8")) > max_bytes:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text[:max_bytes]]


def _sentence_split(text: str) -> list[str]:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _concatenate_mp3(segments: list[bytes], output_path: Path) -> None:
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    for seg_bytes in segments:
        seg = AudioSegment.from_mp3(io.BytesIO(seg_bytes))
        combined += seg
    combined.export(str(output_path), format="mp3")


# ── Mock ────────────────────────────────────────────────────────────────────

def _write_mock_audio(path: Path) -> None:
    """Write a 10-second silent MP3 for mock/test mode."""
    try:
        from pydub import AudioSegment
        silence = AudioSegment.silent(duration=10_000)  # 10 s
        silence.export(str(path), format="mp3")
    except ImportError:
        # Minimal valid MP3 frame header (silent)
        path.write_bytes(bytes([0xFF, 0xFB, 0x90, 0x00] * 400))
