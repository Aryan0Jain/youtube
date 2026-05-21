"""
TTSGenerator — Stage 2 of the pipeline.

Provider priority:
  1. ElevenLabs (best quality, requires ELEVENLABS_API_KEY with quota)
  2. Gemini TTS  (excellent quality, free via AI Studio, requires GEMINI_API_KEY)
  3. Mock        (silent audio, USE_MOCK_TTS=1)
"""
import io
import logging
import os
from pathlib import Path

from pipeline.base import PipelineStage, JobContext

log = logging.getLogger(__name__)

USE_MOCK = os.environ.get("USE_MOCK_TTS") == "1"

# Gemini TTS model — flash is free-tier, pro is higher quality
GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")

# Gemini voice mapping — each voice has a distinct personality and cadence
# Voices: Aoede, Charon, Fenrir, Kore, Leda, Orus, Puck, Zephyr,
#         Schehedar, Sulafat, Achernar, Algieba, Despina, Erinome
_GEMINI_VOICE_MAP: dict[str, str] = {
    "horror":            "Charon",   # deep male, informative — most cinematic
    "ranking":           "Charon",   # authority + gravitas for countdown
    "what_if":           "Orus",     # neutral steady male, builds scenarios well
    "myth_busting":      "Kore",     # firm, confident — authoritative debunker
    "historical_versus": "Orus",     # balanced, analytical
    "shock_facts":       "Fenrir",   # excitable, energetic — punchy delivery
    "quiz":              "Aoede",    # bright, positive — friendly host
}
_GEMINI_DEFAULT_VOICE = "Charon"


def _get_gemini_voice(niche: str) -> str:
    """Return the Gemini TTS voice name for the given niche."""
    return _GEMINI_VOICE_MAP.get(niche, _GEMINI_DEFAULT_VOICE)


class TTSGenerator(PipelineStage):
    name = "tts_generator"

    def execute(self, ctx: JobContext) -> JobContext:
        audio_path = ctx.workspace / "audio.mp3"

        if USE_MOCK:
            _write_mock_audio(audio_path)
            ctx.audio_path = audio_path
            log.info(f"Mock audio written: {audio_path}")
            return ctx

        # -- Try ElevenLabs first (highest quality) -------------------------
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
                log.warning(f"ElevenLabs TTS failed ({exc}), falling back to Gemini TTS")

        # -- Fallback: Gemini TTS (free, excellent quality) -----------------
        log.info("TTS provider: Gemini TTS (set ELEVENLABS_API_KEY for even better quality)")
        _gemini_tts(ctx, audio_path)
        ctx.audio_path = audio_path
        log.info(f"Gemini TTS audio: {audio_path} ({audio_path.stat().st_size // 1024} KB)")
        return ctx

    def _load_from_checkpoint(self, ctx: JobContext) -> JobContext:
        audio_path = ctx.workspace / "audio.mp3"
        if audio_path.exists():
            ctx.audio_path = audio_path
        return ctx


# ── Gemini TTS ───────────────────────────────────────────────────────────────

def generate_audio_gemini(
    script_text: str,
    niche: str,
    output_path: Path,
    speaking_rate: float = 1.0,
) -> None:
    """
    Public helper: generate TTS audio via Gemini AI Studio and save as MP3.

    Used by both the pipeline stage and the test scripts so the implementation
    lives in one place.

    Args:
        script_text:   The full narration text.
        niche:         Niche name — selects the voice personality.
        output_path:   Where to write the final MP3.
        speaking_rate: Playback speed multiplier (0.85–1.2 typical range).
                       Applied as a post-processing step via frame-rate shift.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to .env — get a free key at "
            "https://aistudio.google.com/app/apikey"
        )

    from google import genai
    from google.genai import types
    from pydub import AudioSegment

    voice_name = _get_gemini_voice(niche)
    client = genai.Client(api_key=api_key)

    log.info(f"Gemini TTS: model={GEMINI_TTS_MODEL}, voice={voice_name}, rate={speaking_rate:.2f}")

    # Split script into chunks the API can handle in one call
    chunks = _split_text(script_text, max_bytes=3000)
    log.info(f"  {len(chunks)} chunk(s) to synthesize")

    combined = AudioSegment.empty()

    for i, chunk in enumerate(chunks):
        response = client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=chunk,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    )
                ),
            ),
        )

        # Response is raw PCM: 24 000 Hz, 16-bit, mono
        pcm_data = response.candidates[0].content.parts[0].inline_data.data
        seg = AudioSegment(data=pcm_data, sample_width=2, frame_rate=24000, channels=1)
        combined += seg
        log.info(f"  chunk {i + 1}/{len(chunks)}: {len(pcm_data) // 1024} KB PCM → {seg.duration_seconds:.1f}s")

    # Apply speaking rate: shift frame rate (also shifts pitch slightly, acceptable for ±20%)
    if abs(speaking_rate - 1.0) > 0.03:
        combined = _adjust_speed(combined, speaking_rate)
        log.info(f"  speed adjusted to {speaking_rate:.2f}x")

    # Export as 192 kbps MP3
    combined.export(str(output_path), format="mp3", bitrate="192k")


def _gemini_tts(ctx: JobContext, audio_path: Path) -> None:
    """Pipeline-internal wrapper: extracts params from JobContext."""
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

    generate_audio_gemini(
        script_text=ctx.script_text or "",
        niche=ctx.niche,
        output_path=audio_path,
        speaking_rate=speaking_rate,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _adjust_speed(audio: "AudioSegment", speed: float) -> "AudioSegment":
    """
    Shift playback speed by resampling frame rate.
    At ±20% the pitch change is barely perceptible on spoken voice.
    """
    new_frame_rate = int(audio.frame_rate * speed)
    # Resample to 44100 Hz for MP3 compatibility after the rate shift
    return audio._spawn(
        audio.raw_data, overrides={"frame_rate": new_frame_rate}
    ).set_frame_rate(44100)


def _split_text(text: str, max_bytes: int = 3000) -> list[str]:
    """Split text into chunks under max_bytes at sentence boundaries."""
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


# ── Mock ─────────────────────────────────────────────────────────────────────

def _write_mock_audio(path: Path) -> None:
    """Write a 10-second silent MP3 for mock/test mode."""
    try:
        from pydub import AudioSegment
        silence = AudioSegment.silent(duration=10_000)
        silence.export(str(path), format="mp3")
    except ImportError:
        path.write_bytes(bytes([0xFF, 0xFB, 0x90, 0x00] * 400))


# ── Legacy stubs (kept so old imports don't break) ───────────────────────────
# These were used by test_all_niches.py stage2_tts. Now it calls
# generate_audio_gemini() directly, but keep stubs to avoid ImportError
# if any other code still references them.

def _get_google_voice(niche: str) -> str:
    """Deprecated — returns Gemini voice name instead."""
    return _get_gemini_voice(niche)


def _apply_ssml_markup(text: str, niche: str) -> str | None:
    """Deprecated — Gemini TTS does not use SSML. Always returns None."""
    return None


def _split_ssml(ssml: str, max_chars: int = 4500) -> list[str]:
    """Deprecated — kept for import compatibility."""
    return [ssml]
