from .base import FormatSpec

SHORTS_SPEC = FormatSpec(
    name="shorts",

    target_word_count=130,
    max_word_count=150,
    script_prompt_suffix=(
        "This is for a YouTube Short. Maximum 60 seconds when read aloud. "
        "Hook in the very first sentence — no warm-up, no intro. "
        "Use punchy, declarative sentences. End with one memorable line. "
        "Do not include stage directions, headers, or timestamps — return only the spoken script text."
    ),

    speaking_rate_multiplier=1.05,

    width=1080,
    height=1920,
    clip_orientation="portrait",
    crop_filter="crop=1080:1920:(iw-1080)/2:(ih-1920)/2",
    max_duration_seconds=59,
    clip_min_seconds=3,
    clip_max_seconds=6,

    music_enabled=False,
    default_music_volume=0.0,

    thumb_width=1080,
    thumb_height=1920,

    title_suffix=" #Shorts",
)
