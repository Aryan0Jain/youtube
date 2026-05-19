from .base import FormatSpec

FULL_LENGTH_SPEC = FormatSpec(
    name="full_length",

    target_word_count=1400,
    max_word_count=1800,
    script_prompt_suffix=(
        "This is for a full-length YouTube video. Target 8–12 minutes when read aloud "
        "at a natural pace (~150 wpm). Structure: engaging hook (30s), three main segments "
        "with clear transitions, and an outro with a call to action. Do not include stage "
        "directions, headers, or timestamps — return only the spoken script text."
    ),

    speaking_rate_multiplier=1.0,

    width=1920,
    height=1080,
    clip_orientation="landscape",
    crop_filter=None,
    max_duration_seconds=720,
    clip_min_seconds=5,
    clip_max_seconds=10,

    music_enabled=True,
    default_music_volume=0.08,

    thumb_width=1280,
    thumb_height=720,

    title_suffix="",
)
