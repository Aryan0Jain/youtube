"""
integrations/elevenlabs_client.py

ElevenLabs TTS wrapper. All voice settings are read from pipeline.niche_config
(which loads config/niches.yaml) — no hardcoded niche dicts here.

To change a voice, stability, style, or speaking rate for any niche:
  edit config/niches.yaml — no code changes needed.
"""
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# Fallback voice used when niche is not found in config/niches.yaml
_FALLBACK_VOICE_ID = "pNInz6obpgDQGcFmaJgB"   # Adam
_FALLBACK_SETTINGS = {
    "stability": 0.50,
    "similarity_boost": 0.80,
    "style": 0.15,
    "use_speaker_boost": True,
}


def generate_audio(script_text: str, niche: str, output_path: Path,
                   voice_id: str | None = None,
                   model: str = "eleven_multilingual_v2") -> Path:
    """
    Generate audio with ElevenLabs using niche-specific voice settings
    from config/niches.yaml.

    voice_id: override the niche-default voice (optional).
    Returns output_path on success.
    """
    from elevenlabs import ElevenLabs, VoiceSettings

    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        raise EnvironmentError("ELEVENLABS_API_KEY not set")

    # Load voice settings from niche config
    try:
        from pipeline.niche_config import get_niche_profile
        profile = get_niche_profile(niche)
        vid = voice_id or profile.elevenlabs_voice_id
        settings = VoiceSettings(
            stability=profile.elevenlabs_stability,
            similarity_boost=profile.elevenlabs_similarity_boost,
            style=profile.elevenlabs_style,
            use_speaker_boost=True,
        )
        log.info(
            f"ElevenLabs TTS: niche={niche} voice={vid} "
            f"stability={profile.elevenlabs_stability} "
            f"style={profile.elevenlabs_style} "
            f"rate={profile.speaking_rate}"
        )
    except (KeyError, Exception) as exc:
        log.warning(
            f"Could not load niche profile for '{niche}': {exc}. "
            "Using fallback ElevenLabs voice settings."
        )
        vid = voice_id or _FALLBACK_VOICE_ID
        settings = VoiceSettings(**_FALLBACK_SETTINGS)

    client = ElevenLabs(api_key=api_key)

    audio_stream = client.text_to_speech.convert(
        voice_id=vid,
        text=script_text,
        model_id=model,
        voice_settings=settings,
        output_format="mp3_44100_128",
    )

    with open(output_path, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)

    size_kb = output_path.stat().st_size // 1024
    log.info(f"ElevenLabs audio saved: {size_kb} KB → {output_path}")
    return output_path
