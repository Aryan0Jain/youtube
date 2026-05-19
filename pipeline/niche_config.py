"""
pipeline/niche_config.py — Generic niche configuration loader.

Loads config/niches.yaml at import time into typed NicheProfile dataclasses.

Usage:
    from pipeline.niche_config import get_niche_profile, list_niches

    profile = get_niche_profile("horror")
    print(profile.script_system_prompt)
    print(profile.speaking_rate)

Adding a new niche:   add one block to config/niches.yaml — zero code changes.
Removing a niche:     delete its block from config/niches.yaml — zero code changes.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)

# Path to the niche config file relative to this file's directory
_NICHES_YAML = Path(__file__).parent.parent / "config" / "niches.yaml"


@dataclass(frozen=True)
class NicheProfile:
    """All settings for one niche. Loaded from config/niches.yaml."""

    niche_id: str
    display_name: str

    # ── Script ────────────────────────────────────────────────────────────────
    target_word_count: int
    script_system_prompt: str

    # ── Voice (ElevenLabs) ────────────────────────────────────────────────────
    elevenlabs_voice_id: str
    elevenlabs_stability: float
    elevenlabs_style: float
    elevenlabs_similarity_boost: float
    speaking_rate: float          # multiplied with format-level speaking_rate

    # ── Clips ─────────────────────────────────────────────────────────────────
    clip_duration_min: float      # seconds
    clip_duration_max: float      # seconds
    clips_per_segment: int        # clips downloaded per 30-second script segment
    xfade_transition: str         # FFmpeg xfade transition name
    xfade_duration: float         # seconds

    # ── Visuals ───────────────────────────────────────────────────────────────
    color_filter: Optional[str]   # FFmpeg -vf fragment; None = no colour grade
    subtitle_style: str           # ASS force_style string for subtitle burning

    # ── Overlay ───────────────────────────────────────────────────────────────
    # Supported values: none | entity_cards | fact_counter | quiz |
    #                   ranking_card | myth_stamp | scale_text | side_labels
    overlay_type: str

    # ── Music ─────────────────────────────────────────────────────────────────
    music_file: Optional[str]     # filename under music/ directory; None = no music


# ── Loader ────────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS = {
    "display_name",
    "target_word_count",
    "script_system_prompt",
    "elevenlabs_voice_id",
    "elevenlabs_stability",
    "elevenlabs_style",
    "elevenlabs_similarity_boost",
    "speaking_rate",
    "clip_duration_min",
    "clip_duration_max",
    "clips_per_segment",
    "xfade_transition",
    "xfade_duration",
    "subtitle_style",
    "overlay_type",
}


def _load_profiles(yaml_path: Path) -> dict[str, NicheProfile]:
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Niche config not found: {yaml_path}\n"
            "Create config/niches.yaml with at least one niche entry."
        )

    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "niches" not in raw:
        raise ValueError(f"{yaml_path}: expected top-level key 'niches'")

    profiles: dict[str, NicheProfile] = {}

    for niche_id, data in raw["niches"].items():
        if data is None:
            log.warning(f"Niche '{niche_id}' has no settings — skipping")
            continue

        # Validate required fields
        missing = _REQUIRED_FIELDS - set(data.keys())
        if missing:
            raise ValueError(
                f"Niche '{niche_id}' in {yaml_path} is missing required fields: "
                + ", ".join(sorted(missing))
            )

        profiles[niche_id] = NicheProfile(
            niche_id=niche_id,
            display_name=str(data["display_name"]),
            target_word_count=int(data["target_word_count"]),
            script_system_prompt=str(data["script_system_prompt"]).strip(),
            elevenlabs_voice_id=str(data["elevenlabs_voice_id"]),
            elevenlabs_stability=float(data["elevenlabs_stability"]),
            elevenlabs_style=float(data["elevenlabs_style"]),
            elevenlabs_similarity_boost=float(data["elevenlabs_similarity_boost"]),
            speaking_rate=float(data["speaking_rate"]),
            clip_duration_min=float(data["clip_duration_min"]),
            clip_duration_max=float(data["clip_duration_max"]),
            clips_per_segment=int(data["clips_per_segment"]),
            xfade_transition=str(data["xfade_transition"]),
            xfade_duration=float(data["xfade_duration"]),
            color_filter=data.get("color_filter") or None,
            subtitle_style=str(data["subtitle_style"]),
            overlay_type=str(data.get("overlay_type", "none")),
            music_file=data.get("music_file") or None,
        )

    log.debug(f"Loaded {len(profiles)} niche profiles: {list(profiles.keys())}")
    return profiles


# Module-level cache — loaded once on first import
_PROFILES: dict[str, NicheProfile] | None = None


def _ensure_loaded() -> dict[str, NicheProfile]:
    global _PROFILES
    if _PROFILES is None:
        _PROFILES = _load_profiles(_NICHES_YAML)
    return _PROFILES


# ── Public API ────────────────────────────────────────────────────────────────

def get_niche_profile(niche: str) -> NicheProfile:
    """
    Return the NicheProfile for the given niche ID.

    Raises KeyError with a helpful message if the niche is not in niches.yaml.
    """
    profiles = _ensure_loaded()
    if niche not in profiles:
        valid = ", ".join(sorted(profiles.keys()))
        raise KeyError(
            f"Unknown niche '{niche}'. "
            f"Valid niches (from config/niches.yaml): {valid}"
        )
    return profiles[niche]


def list_niches() -> list[str]:
    """Return sorted list of all configured niche IDs."""
    return sorted(_ensure_loaded().keys())


def reload() -> None:
    """
    Force reload of niches.yaml (useful after editing the file at runtime).
    Normally not needed — profiles are loaded once at startup.
    """
    global _PROFILES
    _PROFILES = None
    _ensure_loaded()
    log.info(f"Niche config reloaded: {list_niches()}")
