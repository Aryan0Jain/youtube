from dataclasses import dataclass


@dataclass(frozen=True)
class FormatSpec:
    name: str

    # Script
    target_word_count: int
    max_word_count: int
    script_prompt_suffix: str

    # TTS
    speaking_rate_multiplier: float

    # Video geometry
    width: int
    height: int
    clip_orientation: str           # "landscape" | "portrait"
    crop_filter: str | None         # FFmpeg -vf crop fragment, or None
    max_duration_seconds: int
    clip_min_seconds: int
    clip_max_seconds: int

    # Audio
    music_enabled: bool
    default_music_volume: float

    # Thumbnail
    thumb_width: int
    thumb_height: int

    # YouTube
    title_suffix: str               # e.g. " #Shorts" or ""

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def scale_filter(self) -> str:
        return f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
