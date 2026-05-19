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

        # -- Try ElevenLabs first (much more natural) -----------------------
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

        # -- Fallback: Google Cloud TTS -------------------------------------
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


# -- Google TTS voice mapping (niche -> Studio/Journey voices) ----------------
#
# Studio voices (en-US-Studio-*) are significantly more natural than Neural2.
# They are free-tier and have no quota limits beyond standard API limits.
# Journey voices are optimized for expressive, energetic delivery.
#
_NICHE_VOICE_MAP: dict[str, str] = {
    "horror":              "en-US-Studio-O",    # deep male, cinematic, slow cadence
    "ranking":             "en-US-Studio-O",    # authority + gravitas for countdown
    "what_if":             "en-US-Studio-Q",    # warm male, builds dread and wonder
    "myth_busting":        "en-US-Studio-Q",    # authoritative but approachable
    "historical_versus":   "en-US-Studio-Q",    # analytical, balanced
    "shock_facts":         "en-US-Journey-D",   # high-energy male, punchy
    "quiz":                "en-US-Neural2-F",   # friendly female, warm and encouraging
}
_DEFAULT_VOICE = "en-US-Studio-Q"

# Niches that benefit from SSML prosody (slower rate, lower pitch, pause markers)
_SSML_NICHES = {"horror", "ranking"}


def _get_google_voice(niche: str) -> str:
    """Return the best Google TTS voice name for the given niche."""
    return _NICHE_VOICE_MAP.get(niche, _DEFAULT_VOICE)


def _apply_ssml_markup(text: str, niche: str) -> str | None:
    """
    Wrap text in SSML for niches that benefit from dramatic pacing.
    Returns an SSML string, or None if this niche does not use SSML.

    For horror/ranking: slows rate slightly, lowers pitch, converts "..."
    markers into 600ms breaks for cinematic pauses.
    """
    if niche not in _SSML_NICHES:
        return None

    import re
    # Replace "..." with an SSML break tag (600ms pause)
    ssml_text = re.sub(r'\.\.\.', '<break time="600ms"/>', text)
    # Escape any bare XML-special characters that aren't already SSML tags
    # (only & needs escaping in text nodes; < > are only in our injected tags)
    ssml_text = ssml_text.replace("&", "&amp;")

    return (
        '<speak>'
        '<prosody rate="90%" pitch="-1st">'
        + ssml_text +
        '</prosody>'
        '</speak>'
    )


# -- Google TTS --------------------------------------------------------------

def _google_tts(ctx: JobContext, audio_path: Path) -> None:
    from google.cloud import texttospeech

    # Use niche-aware Studio voice; fall back to channel-resolved voice if set
    niche_voice = _get_google_voice(ctx.niche)
    voice_name = ctx.resolved.get("tts_voice") or niche_voice

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

    # Attempt SSML markup for dramatic niches; fall back to plain text on error
    use_ssml = ctx.niche in _SSML_NICHES
    log.info(
        f"Google TTS: voice={voice_name}, rate={speaking_rate:.2f}, "
        f"ssml={'yes' if use_ssml else 'no'}"
    )

    # For SSML niches, send the full script as one SSML block (Google TTS
    # SSML does not support chunking by byte count the same way plain text
    # does, so we split by sentence count instead).
    if use_ssml:
        ssml_body = _apply_ssml_markup(script, ctx.niche)
        chunks_ssml = _split_ssml(ssml_body, max_chars=4500) if ssml_body else None
    else:
        chunks_ssml = None

    chunks_text = _split_text(script, max_bytes=4800)

    client = texttospeech.TextToSpeechClient()
    audio_segments: list[bytes] = []

    if chunks_ssml:
        log.info(f"  sending {len(chunks_ssml)} SSML chunk(s)")
        for i, ssml_chunk in enumerate(chunks_ssml):
            synthesis_input = texttospeech.SynthesisInput(ssml=ssml_chunk)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=voice_name,
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate,
                pitch=0.0,
            )
            try:
                response = client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )
                audio_segments.append(response.audio_content)
                log.info(f"  ssml chunk {i+1}/{len(chunks_ssml)}: {len(response.audio_content)} bytes")
            except Exception as exc:
                log.warning(f"  SSML chunk {i+1} failed ({exc}), retrying as plain text")
                # Fall back to plain text for this chunk
                plain = chunks_text[min(i, len(chunks_text) - 1)]
                synthesis_input_plain = texttospeech.SynthesisInput(text=plain)
                response = client.synthesize_speech(
                    input=synthesis_input_plain, voice=voice, audio_config=audio_config
                )
                audio_segments.append(response.audio_content)
    else:
        log.info(f"  sending {len(chunks_text)} plain-text chunk(s)")
        for i, chunk in enumerate(chunks_text):
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
            log.info(f"  chunk {i+1}/{len(chunks_text)}: {len(response.audio_content)} bytes")

    if len(audio_segments) == 1:
        audio_path.write_bytes(audio_segments[0])
    else:
        _concatenate_mp3(audio_segments, audio_path)


def _split_ssml(ssml: str, max_chars: int = 4500) -> list[str]:
    """
    Split a full SSML document into chunks under max_chars.
    Wraps each chunk in <speak><prosody ...>...</prosody></speak>.
    Simple implementation: splits at sentence boundaries within the inner text.
    """
    import re
    # Extract inner content between the prosody tags
    match = re.search(r'<prosody[^>]*>(.*)</prosody>', ssml, re.DOTALL)
    if not match:
        return [ssml]
    inner = match.group(1)
    header_match = re.search(r'(<speak><prosody[^>]*>)', ssml)
    header = header_match.group(1) if header_match else '<speak><prosody rate="90%" pitch="-1st">'
    footer = '</prosody></speak>'

    # Split inner content at sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', inner)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        candidate = (current + " " + sent).strip()
        full = header + candidate + footer
        if len(full.encode("utf-8")) > max_chars and current:
            chunks.append(header + current.strip() + footer)
            current = sent
        else:
            current = candidate
    if current.strip():
        chunks.append(header + current.strip() + footer)
    return chunks or [ssml]


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


# -- Mock --------------------------------------------------------------------

def _write_mock_audio(path: Path) -> None:
    """Write a 10-second silent MP3 for mock/test mode."""
    try:
        from pydub import AudioSegment
        silence = AudioSegment.silent(duration=10_000)  # 10 s
        silence.export(str(path), format="mp3")
    except ImportError:
        # Minimal valid MP3 frame header (silent)
        path.write_bytes(bytes([0xFF, 0xFB, 0x90, 0x00] * 400))
