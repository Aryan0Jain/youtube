"""
VideoAssembler -- Stage 5 of the pipeline.

Five-phase FFmpeg pipeline:
  1. Pre-process each clip: Ken Burns motion + niche colour grade
  2. Concatenate clips with xfade cross-dissolve transitions
  3. Mix audio: voiceover + sidechain-compressed background music
  3.5 Apply niche overlays (entity cards, fact counter, ranking cards, etc.)
  4. Burn subtitles (if subtitle_path set on ctx)
  LUFS. Normalize audio loudness to -14 LUFS (YouTube standard)

All niche-specific settings (xfade durations, color grades, transitions,
subtitle styles, music files) are read from pipeline.niche_config, which
loads config/niches.yaml. No hardcoded niche dicts here.

To change any niche setting: edit config/niches.yaml only.
"""
import json
import logging
import os
import random
import shutil
import subprocess
from pathlib import Path

from pipeline.base import PipelineStage, JobContext

log = logging.getLogger(__name__)

# -- Paths ---------------------------------------------------------------------
MUSIC_DIR = Path(__file__).parent.parent / "music"

# Default xfade duration (used if niche profile is unavailable)
_DEFAULT_XFADE_DUR = 0.75

# Default clip duration range (used if niche profile is unavailable)
_DEFAULT_CLIP_SEC_MIN = 4.0
_DEFAULT_CLIP_SEC_MAX = 7.0

# Default subtitle style (used if niche profile is unavailable)
_DEFAULT_SUBTITLE_STYLE = (
    "Fontsize=52,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
    "Outline=3,Shadow=1,Alignment=2,MarginV=65,Bold=1"
)


# -- FFmpeg helpers -------------------------------------------------------------

def _run_ffmpeg(args: list[str], label: str = "ffmpeg") -> None:
    cmd = ["ffmpeg", "-y"] + args
    log.debug(f"FFmpeg [{label}]: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg [{label}] failed:\n{result.stderr[-3000:]}")


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


# -- Niche settings helpers -----------------------------------------------------

def _get_niche_profile(niche: str):
    """Return NicheProfile or None -- never raises."""
    try:
        from pipeline.niche_config import get_niche_profile
        return get_niche_profile(niche)
    except Exception as exc:
        log.warning(f"Could not load niche profile for '{niche}': {exc}")
        return None


# -- Ken Burns -----------------------------------------------------------------

def _kenburns_filter(clip_idx: int, w: int, h: int, duration_secs: float) -> str:
    """
    Build a -vf fragment that applies a Ken Burns motion effect to one clip.

    Uses scale-to-1.3x + time-based crop (NOT zoompan -- zoompan is O(W*H*frames)
    and takes 10-30s per clip; this approach is near-realtime).

    5 directions cycle per clip_idx so consecutive shots look varied:
      0 -> diagonal drift (top-left to bottom-right)  — 1.30x scale
      1 -> pan left to right                          — 1.30x scale
      2 -> pan right to left                          — 1.30x scale
      3 -> pan bottom to top                          — 1.30x scale
      4 -> zoom punch (diagonal, 1.55x scale)         — visual contrast "punch"
    """
    sw = (int(w * 1.30) + 1) & ~1  # 30% oversize, forced even
    sh = (int(h * 1.30) + 1) & ~1
    dx = sw - w
    dy = sh - h
    dur = max(0.01, duration_secs)

    direction = clip_idx % 5  # 5 directions: 4 standard Ken Burns + 1 zoom punch

    if direction == 4:
        # Zoom punch: 55% oversize (vs 30%), fast top-left diagonal drift.
        # Every 5th clip appears dramatically more zoomed-in for visual contrast —
        # the jump from 1.30x to 1.55x scale creates the "punch" feel without
        # any complex per-frame easing.
        spw = (int(w * 1.55) + 1) & ~1
        sph = (int(h * 1.55) + 1) & ~1
        dpx = spw - w
        dpy = sph - h
        scale_z = f"scale={spw}:{sph}:force_original_aspect_ratio=increase"
        center_z = f"crop={spw}:{sph}"
        crop_z   = f"crop={w}:{h}:x={dpx}*t/{dur:.4f}:y={dpy}*t/{dur:.4f}"
        return f"{scale_z},{center_z},{crop_z},setsar=1"

    if direction == 0:
        crop = f"crop={w}:{h}:x={dx}*t/{dur:.4f}:y={dy}*t/{dur:.4f}"
    elif direction == 1:
        crop = f"crop={w}:{h}:x={dx}*t/{dur:.4f}:y={dy//2}"
    elif direction == 2:
        crop = f"crop={w}:{h}:x={dx}-{dx}*t/{dur:.4f}:y={dy//2}"
    else:  # direction == 3
        crop = f"crop={w}:{h}:x={dx//2}:y={dy}-{dy}*t/{dur:.4f}"

    # scale-to-cover: scale up, center-crop to exact sw x sh (handles off-AR sources)
    scale = f"scale={sw}:{sh}:force_original_aspect_ratio=increase"
    center_crop = f"crop={sw}:{sh}"
    return f"{scale},{center_crop},{crop},setsar=1"


# -- Phase 1: Pre-process clips ------------------------------------------------

def _preprocess_clip(src: Path, out: Path, clip_idx: int, duration: float,
                     niche: str, spec) -> None:
    """Trim, apply Ken Burns + niche colour grade. Output: 30fps libx264, no audio."""
    kb = _kenburns_filter(clip_idx, spec.width, spec.height, duration)

    profile = _get_niche_profile(niche)
    color = profile.color_filter if profile else None
    vf = f"{kb},{color}" if color else kb

    _run_ffmpeg([
        "-ss", "0",
        "-i", str(src),
        "-vf", vf,
        "-t", str(duration),
        "-r", "30",
        "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        str(out),
    ], label=f"preprocess_{clip_idx:03d}")


# -- Phase 2: Xfade concatenation ----------------------------------------------

def _xfade_concat(clip_files: list[Path], clip_durations: list[float],
                  niche: str, output: Path) -> float:
    """Concatenate pre-processed clips with xfade transitions. Returns total duration."""
    n = len(clip_files)

    profile = _get_niche_profile(niche)
    transition = profile.xfade_transition if profile else "fade"
    xfade_dur = profile.xfade_duration if profile else _DEFAULT_XFADE_DUR

    inputs: list[str] = []
    for f in clip_files:
        inputs += ["-i", str(f)]

    if n == 1:
        _run_ffmpeg(inputs + [
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-pix_fmt", "yuv420p",
            str(output),
        ], label="concat_single")
        return clip_durations[0]

    fc_parts: list[str] = []
    last = "[0:v]"
    offset = 0.0

    for i in range(1, n):
        offset += clip_durations[i - 1] - xfade_dur
        out_label = "[vfinal]" if i == n - 1 else f"[v{i}]"
        fc_parts.append(
            f"{last}[{i}:v]xfade=transition={transition}"
            f":duration={xfade_dur:.3f}:offset={max(0.0, offset):.3f}{out_label}"
        )
        last = out_label

    fc = ";".join(fc_parts)
    total_dur = sum(clip_durations) - (n - 1) * xfade_dur

    _run_ffmpeg(inputs + [
        "-filter_complex", fc,
        "-map", "[vfinal]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        str(output),
    ], label="xfade_concat")

    return total_dur


# -- Phase 3: Audio mixing ------------------------------------------------------

def _ensure_sfx_library() -> dict[str, Path]:
    """
    Return a dict of all SFX sound paths, generating any that are missing.

    Four sounds — all synthesised via FFmpeg lavfi (no downloads):

    impact  — 65 Hz soft boom, 0.3s  (regular clip transitions)
    whoosh  — 80→4000 Hz sweep, 0.25s (rank-entry transitions, ranks 4-10)
    bass    — 45 Hz heavy hit, 0.5s   (rank reveals, ranks 1-3)
    rumble  — 38+76 Hz sustained, 1.2s (added at the #1 reveal moment only)

    Returns only successfully generated paths; missing entries are silently omitted.
    """
    MUSIC_DIR.mkdir(exist_ok=True)
    specs = {
        "impact": (
            "sfx_impact.mp3", 0.3,
            "aevalsrc="
            "sin(2*PI*65*t)*exp(-t*16)*0.7|"
            "sin(2*PI*65*t)*exp(-t*16)*0.7"
            ":s=44100:d=0.3"
        ),
        "whoosh": (
            "sfx_whoosh.mp3", 0.25,
            "aevalsrc="
            "sin(2*PI*(80+3920*t/0.25)*t)*exp(-t*8)*0.5|"
            "sin(2*PI*(80+3920*t/0.25)*t)*exp(-t*8)*0.5"
            ":s=44100:d=0.25"
        ),
        "bass": (
            "sfx_bass_hit.mp3", 0.5,
            "aevalsrc="
            "sin(2*PI*45*t)*exp(-t*10)*0.85|"
            "sin(2*PI*45*t)*exp(-t*10)*0.85"
            ":s=44100:d=0.5"
        ),
        "rumble": (
            "sfx_rumble.mp3", 1.2,
            "aevalsrc="
            "(sin(2*PI*38*t)+0.5*sin(2*PI*76*t))*exp(-t*4)*0.9|"
            "(sin(2*PI*38*t)+0.5*sin(2*PI*76*t))*exp(-t*4)*0.9"
            ":s=44100:d=1.2"
        ),
    }
    lib: dict[str, Path] = {}
    for key, (filename, _dur, aevalsrc) in specs.items():
        path = MUSIC_DIR / filename
        if not path.exists():
            try:
                _run_ffmpeg([
                    "-f", "lavfi", "-i", aevalsrc,
                    "-c:a", "libmp3lame", "-b:a", "128k",
                    str(path),
                ], label=f"gen_sfx_{key}")
                log.info(f"  Generated SFX: {filename}")
            except Exception as exc:
                log.warning(f"  Could not generate {filename}: {exc}")
                continue
        lib[key] = path
    return lib


def _build_sfx_track(
    timestamps: list[float],
    total_dur: float,
    sfx_lib: dict[str, Path],
    out_path: Path,
    rank_timestamps: dict[int, float] | None = None,
) -> None:
    """
    Build a context-aware SFX audio track using different sounds for different moments.

    Sound selection per clip transition:
    - Within 0.5s of a top-3 rank reveal → bass_hit (heavy)
    - Within 0.5s of a rank 4-10 reveal  → whoosh   (sweeping)
    - All other clip transitions          → impact   (soft boom)

    Additionally, a rumble sound is placed at the exact #1 reveal timestamp
    (not tied to a clip cut — it plays under the narrator's voice).
    """
    from pydub import AudioSegment

    def load(key: str, db_reduction: int = 10) -> AudioSegment | None:
        if key in sfx_lib and sfx_lib[key].exists():
            return AudioSegment.from_mp3(str(sfx_lib[key])) - db_reduction
        return None

    impact_snd = load("impact", 10)
    whoosh_snd = load("whoosh", 8)
    bass_snd   = load("bass",   6)
    rumble_snd = load("rumble", 8)

    if impact_snd is None:
        return   # nothing to build

    base = AudioSegment.silent(duration=int(total_dur * 1000), frame_rate=44100)

    rank_ts_flat = list((rank_timestamps or {}).items())  # [(rank, ts), ...]

    def nearest_rank(ts: float) -> tuple[int, float] | None:
        """Return (rank, rank_ts) if any rank reveal is within 0.5s of ts."""
        for rank, rts in rank_ts_flat:
            if abs(ts - rts) <= 0.5:
                return (rank, rts)
        return None

    for ts in timestamps:
        pos_ms = int(ts * 1000)
        if not (0 < pos_ms < len(base)):
            continue
        hit = nearest_rank(ts)
        if hit is not None:
            rank, _ = hit
            snd = bass_snd if rank <= 3 else (whoosh_snd or impact_snd)
        else:
            snd = impact_snd
        if snd is not None:
            base = base.overlay(snd, position=pos_ms)

    # Add rumble at the #1 reveal (independent of clip cuts)
    if rumble_snd is not None and rank_timestamps and 1 in rank_timestamps:
        pos_ms = int(rank_timestamps[1] * 1000)
        if 0 < pos_ms < len(base):
            base = base.overlay(rumble_snd, position=pos_ms)

    base.export(str(out_path), format="mp3", bitrate="128k")


def _mix_audio(raw_video: Path, vo_audio: Path, music_path: Path | None,
               music_volume: float, max_dur: float, output: Path,
               sfx_track: Path | None = None) -> None:
    """
    Mix voiceover with optional sidechain-compressed background music.
    Sidechain: music ducks 4:1 whenever VO is active (threshold 0.02,
    attack 10ms, release 250ms).
    """
    if music_path is None or not music_path.exists():
        _run_ffmpeg([
            "-i", str(raw_video),
            "-i", str(vo_audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-t", str(max_dur), "-shortest",
            str(output),
        ], label="mix_vo_only")
        return

    # Build filter complex — optionally include a 3rd SFX layer
    if sfx_track and sfx_track.exists():
        fc = (
            "[1:a]apad[vo];"
            f"[2:a]volume={music_volume}[music_raw];"
            "[vo]asplit=2[vo_direct][vo_sc];"
            "[music_raw][vo_sc]sidechaincompress="
            "threshold=0.02:ratio=4:attack=10:release=250[music_ducked];"
            "[3:a]volume=0.18[sfx];"
            "[vo_direct][music_ducked][sfx]amix=inputs=3:duration=first:dropout_transition=2[aout]"
        )
        ffmpeg_inputs = [
            "-i", str(raw_video),
            "-i", str(vo_audio),
            "-stream_loop", "-1", "-i", str(music_path),
            "-i", str(sfx_track),
        ]
        log.info("  audio: VO + sidechain music + SFX impact layer")
    else:
        fc = (
            "[1:a]apad[vo];"
            f"[2:a]volume={music_volume}[music_raw];"
            "[vo]asplit=2[vo_direct][vo_sc];"
            "[music_raw][vo_sc]sidechaincompress="
            "threshold=0.02:ratio=4:attack=10:release=250[music_ducked];"
            "[vo_direct][music_ducked]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
        ffmpeg_inputs = [
            "-i", str(raw_video),
            "-i", str(vo_audio),
            # -stream_loop -1 loops the music file so it never cuts out on long videos.
            "-stream_loop", "-1", "-i", str(music_path),
        ]

    _run_ffmpeg(ffmpeg_inputs + [
        "-filter_complex", fc,
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", str(max_dur),
        str(output),
    ], label="mix_sidechain")


# -- Phase 3.5: Niche overlays -------------------------------------------------

def _word_idx_to_ts(word_idx: int, total_words: int, audio_duration: float,
                    word_ts_map: dict[int, float] | None = None) -> float:
    """
    Convert a word index to a video timestamp.

    If word_ts_map is provided (built from Whisper SRT timestamps), use the
    actual timing for precision. Otherwise fall back to linear approximation.
    """
    if word_ts_map:
        # Find the closest word index in the map
        if word_idx in word_ts_map:
            return word_ts_map[word_idx]
        # Nearest-neighbour fallback
        keys = sorted(word_ts_map.keys())
        if keys:
            closest = min(keys, key=lambda k: abs(k - word_idx))
            return word_ts_map[closest]
    if total_words <= 0:
        return 0.0
    return max(0.0, (word_idx / total_words) * audio_duration)


def _build_word_ts_map(srt_path: Path) -> dict[int, float]:
    """
    Parse subtitles.srt to build a cumulative {word_index -> start_time} map.

    Each SRT entry covers a group of 3-5 words. We assign the entry's start
    time to each word in the group, giving us per-word timing anchored to
    Whisper's actual transcription rather than linear approximation.

    Returns an empty dict if the file is missing or unparseable.
    """
    if not srt_path or not srt_path.exists():
        return {}

    import re as _re

    ts_map: dict[int, float] = {}
    word_idx = 0

    try:
        content = srt_path.read_text(encoding="utf-8")
        # Each SRT block: index, timecode line, text line(s), blank line
        blocks = _re.split(r"\n\n+", content.strip())
        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) < 3:
                continue
            # lines[0] = entry index, lines[1] = timecodes, lines[2+] = text
            tc_match = _re.match(
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})",
                lines[1],
            )
            if not tc_match:
                continue
            h, m, s, ms = [int(x) for x in tc_match.groups()[:4]]
            start_sec = h * 3600 + m * 60 + s + ms / 1000.0
            text = " ".join(lines[2:])
            words_in_group = len(text.split())
            for i in range(words_in_group):
                ts_map[word_idx + i] = start_sec
            word_idx += words_in_group
    except Exception:
        return {}

    return ts_map


def _escape_drawtext(text: str) -> str:
    """Escape special characters for FFmpeg drawtext filter."""
    return (text
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace(":", "\\:")
            .replace(",", "\\,"))


def _apply_niche_overlays(
    video_in: Path,
    niche_metadata: dict,
    audio_duration: float,
    script_text: str,
    output: Path,
    srt_path: Path | None = None,
) -> bool:
    """
    Apply time-gated FFmpeg drawtext/drawbox overlays based on niche_metadata.

    srt_path: if provided, uses Whisper-accurate word timestamps instead of
              linear approximation for overlay positioning.

    Returns True if overlays were applied, False if skipped (non-critical).
    On any error: logs warning, returns False -- the pipeline continues unaffected.
    """
    if not niche_metadata:
        return False

    overlay_type = niche_metadata.get("overlay_type", "none")
    items = niche_metadata.get("items", [])

    if overlay_type == "none" or not items:
        return False

    total_words = len(script_text.split())
    word_ts_map = _build_word_ts_map(srt_path) if srt_path else {}
    if word_ts_map:
        log.debug(f"Overlay timing: using Whisper SRT map ({len(word_ts_map)} words)")
    else:
        log.debug("Overlay timing: using linear word-index approximation")

    try:
        if overlay_type == "entity_cards":
            return _overlay_entity_cards(video_in, items, total_words, audio_duration, output, word_ts_map)
        elif overlay_type == "fact_counter":
            return _overlay_fact_counter(video_in, items, total_words, audio_duration, output, word_ts_map)
        elif overlay_type == "ranking_card":
            return _overlay_ranking_cards(video_in, items, total_words, audio_duration, output, word_ts_map, srt_path)
        elif overlay_type == "myth_stamp":
            return _overlay_myth_stamp(video_in, items, total_words, audio_duration, output)
        elif overlay_type == "scale_text":
            return _overlay_scale_text(video_in, items, total_words, audio_duration, output, word_ts_map)
        elif overlay_type == "side_labels":
            return _overlay_side_labels(video_in, items, total_words, audio_duration, output, word_ts_map)
        elif overlay_type == "quiz":
            return _overlay_quiz(video_in, items, total_words, audio_duration, output, word_ts_map)
        else:
            log.info(f"Overlay type '{overlay_type}' has no renderer -- skipping")
            return False
    except Exception as exc:
        log.warning(f"Overlay pass failed ({overlay_type}): {exc} -- continuing without overlays")
        return False


def _overlay_entity_cards(video_in: Path, items: list, total_words: int,
                           audio_dur: float, output: Path,
                           word_ts_map: dict | None = None) -> bool:
    """
    Horror: semi-transparent dark box + name/role text at bottom-left.
    Shows for 4 seconds at the moment each entity is first mentioned.
    """
    if not items:
        return False

    vf_parts: list[str] = []
    for item in items[:8]:  # cap at 8 overlays
        ts = _word_idx_to_ts(item.get("word_idx", 0), total_words, audio_dur, word_ts_map)
        end = ts + 4.0
        enable = f"between(t\\,{ts:.2f}\\,{end:.2f})"

        text = _escape_drawtext(str(item.get("text", "")).upper())
        role = _escape_drawtext(str(item.get("role", "")))

        box = (f"drawbox=x=20:y=ih-120:w=500:h=90:"
               f"color=black@0.6:t=fill:"
               f"enable='{enable}'")
        line1 = (f"drawtext=text='{text}':"
                 f"x=30:y=h-110:"
                 f"fontsize=28:fontcolor=white:"
                 f"enable='{enable}'")
        line2 = (f"drawtext=text='{role}':"
                 f"x=30:y=h-78:"
                 f"fontsize=20:fontcolor=0xFFCCCC:"
                 f"enable='{enable}'")
        vf_parts += [box, line1, line2]

    if not vf_parts:
        return False

    vf = ",".join(vf_parts)
    _run_ffmpeg([
        "-i", str(video_in),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output),
    ], label="overlay_entity_cards")
    return True


def _overlay_fact_counter(video_in: Path, items: list, total_words: int,
                           audio_dur: float, output: Path,
                           word_ts_map: dict | None = None) -> bool:
    """Shock facts: bold yellow 'FACT #N' counter top-right, updates per fact."""
    if not items:
        return False

    vf_parts: list[str] = []
    sorted_items = sorted(items, key=lambda x: x.get("word_idx", 0))

    for i, item in enumerate(sorted_items):
        ts_start = _word_idx_to_ts(item.get("word_idx", 0), total_words, audio_dur, word_ts_map)
        ts_end = (
            _word_idx_to_ts(sorted_items[i + 1].get("word_idx", 0), total_words, audio_dur, word_ts_map)
            if i + 1 < len(sorted_items)
            else ts_start + 30.0
        )
        enable = f"between(t\\,{ts_start:.2f}\\,{ts_end:.2f})"

        number = item.get("number", i + 1)
        headline = _escape_drawtext(str(item.get("headline", f"FACT #{number}")))

        counter = (f"drawtext=text='FACT #{number}':"
                   f"x=w-220:y=20:"
                   f"fontsize=36:fontcolor=yellow:"
                   f"enable='{enable}'")
        sub = (f"drawtext=text='{headline}':"
               f"x=w-220:y=65:"
               f"fontsize=18:fontcolor=white:"
               f"enable='{enable}'")
        vf_parts += [counter, sub]

    if not vf_parts:
        return False

    vf = ",".join(vf_parts)
    _run_ffmpeg([
        "-i", str(video_in),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output),
    ], label="overlay_fact_counter")
    return True


_RANK_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


def _find_rank_ts_in_srt(srt_path: Path | None) -> dict[int, float]:
    """
    Scan the SRT file for rank announcements and return {rank_int: timestamp_seconds}.

    Gemini TTS renders "Number ten," as a spoken digit "10." — Whisper transcribes
    this as a standalone subtitle group containing just "10." or "10,".  We match:

      Pattern A (digit): subtitle text is just the digit, e.g. "10." or "10,"
      Pattern B (word):  "number ten" / "number 10" anywhere in the line (case-insensitive)

    Both patterns look for entries counting down from 10 → 1.
    Returns {} if file missing or no matches found.
    """
    if not srt_path or not srt_path.exists():
        return {}

    import re as _re

    rank_ts: dict[int, float] = {}
    try:
        content = srt_path.read_text(encoding="utf-8")
        blocks = _re.split(r"\n\n+", content.strip())
        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) < 3:
                continue
            tc_match = _re.match(
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->",
                lines[1],
            )
            if not tc_match:
                continue
            h, m, s, ms = [int(x) for x in tc_match.groups()]
            start_sec = h * 3600 + m * 60 + s + ms / 1000.0
            text = " ".join(lines[2:]).strip()
            text_lower = text.lower()

            for rank_int, word in _RANK_WORDS.items():
                if rank_int in rank_ts:
                    continue  # already found

                # Pattern A: subtitle contains only the digit (e.g. "10." or "10,")
                if _re.fullmatch(rf'{rank_int}[.,:]?', text.strip()):
                    rank_ts[rank_int] = start_sec
                    continue

                # Pattern B: "number ten" or "number 10" (spoken form)
                if (_re.search(rf'\bnumber\s+{word}\b', text_lower) or
                        _re.search(rf'\bnumber\s+{rank_int}\b', text_lower)):
                    rank_ts[rank_int] = start_sec
    except Exception:
        pass
    return rank_ts


def _overlay_ranking_cards(video_in: Path, items: list, total_words: int,
                            audio_dur: float, output: Path,
                            word_ts_map: dict | None = None,
                            srt_path: Path | None = None) -> bool:
    """
    Ranking: tiered lower-third with rank number, entry name, year, and scale.

    Card layout:
        [box]  #N  Entry Name               YEAR
                   ~71,000 dead

    Visual hierarchy by rank:
    - Ranks 1-3: larger text, bright gold box, longer display (8/7/6s)
    - Ranks 4-5: medium text, dark box, 6s display
    - Ranks 6-10: standard text, dark box, 5s display

    Timing: searches SRT file for spoken "number N" phrase for exact placement,
    falling back to word_idx approximation if not found.
    """
    if not items:
        return False

    # Build SRT-based rank→timestamp map for exact spoken timing
    rank_srt_ts = _find_rank_ts_in_srt(srt_path)
    if rank_srt_ts:
        log.debug(f"  Ranking card SRT timing: found {len(rank_srt_ts)} spoken rank cues")
    else:
        log.debug("  Ranking card SRT timing: falling back to word_idx approximation")

    vf_parts: list[str] = []
    for item in items[:12]:
        rank = item.get("rank", 99)
        name = _escape_drawtext(str(item.get("name", "")))
        year = _escape_drawtext(str(item.get("year", "")))
        scale = _escape_drawtext(str(item.get("scale", "")))

        try:
            rank_int = int(rank)
        except (ValueError, TypeError):
            rank_int = 99

        # Use Whisper-accurate SRT cue if we found "number N" in the audio;
        # fall back to word_idx linear approximation otherwise.
        if rank_int in rank_srt_ts:
            ts = rank_srt_ts[rank_int]
        else:
            ts = _word_idx_to_ts(item.get("word_idx", 0), total_words, audio_dur, word_ts_map)

        if rank_int <= 3:
            duration = 9.0 - rank_int      # rank1=8s, rank2=7s, rank3=6s
            rank_size, name_size, meta_size = 72, 44, 26
            box_h = 140                     # extra height for scale line
            box_color = "0xFFD700@0.55"
            rank_color = "0xFFD700"
            name_color = "0xFFFFFF"
        elif rank_int <= 5:
            duration = 6.0
            rank_size, name_size, meta_size = 60, 40, 24
            box_h = 130
            box_color = "black@0.65"
            rank_color = "0xFFD700"
            name_color = "0xFFFFFF"
        else:
            duration = 5.0
            rank_size, name_size, meta_size = 52, 36, 22
            box_h = 120
            box_color = "black@0.50"
            rank_color = "0xFFD700"
            name_color = "white"

        end = ts + duration
        enable = f"between(t\\,{ts:.2f}\\,{end:.2f})"

        # Semi-transparent background box
        box = (f"drawbox=x=0:y=ih-{box_h}:w=iw:h={box_h}:"
               f"color={box_color}:t=fill:enable='{enable}'")

        # Rank number (large, gold, left side)
        rank_text = (f"drawtext=text='#{rank}':"
                     f"x=30:y=h-{box_h - 20}:"
                     f"fontsize={rank_size}:fontcolor={rank_color}:"
                     f"enable='{enable}'")

        # Entry name (white, indented from rank number)
        name_text = (f"drawtext=text='{name}':"
                     f"x=130:y=h-{box_h - 30}:"
                     f"fontsize={name_size}:fontcolor={name_color}:"
                     f"enable='{enable}'")

        vf_parts += [box, rank_text, name_text]

        # Year — right-aligned, small grey text at top-right of box
        if year:
            year_text = (f"drawtext=text='{year}':"
                         f"x=w-tw-20:y=h-{box_h - 10}:"
                         f"fontsize={meta_size}:fontcolor=0xAAAAAA:"
                         f"enable='{enable}'")
            vf_parts.append(year_text)

        # Scale / death toll — below entry name, light grey
        if scale:
            scale_text = (f"drawtext=text='{scale}':"
                          f"x=130:y=h-{box_h - 30 - name_size - 6}:"
                          f"fontsize={meta_size}:fontcolor=0xCCCCCC:"
                          f"enable='{enable}'")
            vf_parts.append(scale_text)

    if not vf_parts:
        return False

    vf = ",".join(vf_parts)
    _run_ffmpeg([
        "-i", str(video_in),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output),
    ], label="overlay_ranking")
    return True


def _overlay_myth_stamp(video_in: Path, items, total_words: int,
                         audio_dur: float, output: Path) -> bool:
    """Myth busting: 'MYTH' stamp at intro, 'BUSTED'/'CONFIRMED' at verdict."""
    if isinstance(items, list) and items:
        items = items[0]
    if not isinstance(items, dict):
        return False

    myth_ts = _word_idx_to_ts(items.get("myth_word_idx", 0), total_words, audio_dur)
    verdict_ts = _word_idx_to_ts(items.get("verdict_word_idx", 0), total_words, audio_dur)
    verdict_word = str(items.get("verdict", "BUSTED")).upper()
    verdict_color = "red" if verdict_word == "CONFIRMED" else "0x00FF44"

    myth_enable = f"between(t\\,{myth_ts:.2f}\\,{verdict_ts:.2f})"
    verdict_enable = f"between(t\\,{verdict_ts:.2f}\\,{verdict_ts + 6.0:.2f})"

    vf_parts = [
        (f"drawtext=text='MYTH':"
         f"x=(w-tw)/2:y=(h-th)/2:"
         f"fontsize=96:fontcolor=red@0.85:"
         f"enable='{myth_enable}'"),
        (f"drawtext=text='{_escape_drawtext(verdict_word)}':"
         f"x=(w-tw)/2:y=(h-th)/2:"
         f"fontsize=96:fontcolor={verdict_color}@0.90:"
         f"enable='{verdict_enable}'"),
    ]

    vf = ",".join(vf_parts)
    _run_ffmpeg([
        "-i", str(video_in),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output),
    ], label="overlay_myth_stamp")
    return True


def _overlay_scale_text(video_in: Path, items: list, total_words: int,
                         audio_dur: float, output: Path,
                         word_ts_map: dict | None = None) -> bool:
    """What If: subtle scale labels top-left ('Day 1', 'Year 10')."""
    if not items:
        return False

    vf_parts: list[str] = []
    sorted_items = sorted(items, key=lambda x: x.get("word_idx", 0))

    for i, item in enumerate(sorted_items):
        ts_start = _word_idx_to_ts(item.get("word_idx", 0), total_words, audio_dur, word_ts_map)
        ts_end = (
            _word_idx_to_ts(sorted_items[i + 1].get("word_idx", 0), total_words, audio_dur, word_ts_map)
            if i + 1 < len(sorted_items)
            else ts_start + 20.0
        )
        enable = f"between(t\\,{ts_start:.2f}\\,{ts_end:.2f})"
        label = _escape_drawtext(str(item.get("label", "")))

        vf_parts.append(
            f"drawtext=text='{label}':"
            f"x=30:y=30:"
            f"fontsize=32:fontcolor=white@0.75:"
            f"enable='{enable}'"
        )

    if not vf_parts:
        return False

    vf = ",".join(vf_parts)
    _run_ffmpeg([
        "-i", str(video_in),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output),
    ], label="overlay_scale_text")
    return True


def _overlay_side_labels(video_in: Path, items: list, total_words: int,
                          audio_dur: float, output: Path,
                          word_ts_map: dict | None = None) -> bool:
    """Comparison: small corner labels showing both sides + category."""
    if not items:
        return False

    vf_parts: list[str] = []
    for item in items[:6]:
        ts = _word_idx_to_ts(item.get("word_idx", 0), total_words, audio_dur, word_ts_map)
        end = ts + 8.0
        enable = f"between(t\\,{ts:.2f}\\,{end:.2f})"

        side_a = _escape_drawtext(str(item.get("side_a", "")))
        side_b = _escape_drawtext(str(item.get("side_b", "")))
        category = _escape_drawtext(str(item.get("category", "")).upper())

        vf_parts += [
            (f"drawtext=text='< {side_a}':"
             f"x=20:y=20:fontsize=28:fontcolor=white@0.85:"
             f"enable='{enable}'"),
            (f"drawtext=text='{side_b} >':"
             f"x=w-tw-20:y=20:fontsize=28:fontcolor=white@0.85:"
             f"enable='{enable}'"),
            (f"drawtext=text='{category}':"
             f"x=(w-tw)/2:y=20:fontsize=24:fontcolor=yellow@0.80:"
             f"enable='{enable}'"),
        ]

    if not vf_parts:
        return False

    vf = ",".join(vf_parts)
    _run_ffmpeg([
        "-i", str(video_in),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output),
    ], label="overlay_side_labels")
    return True


def _overlay_quiz(video_in: Path, items: list, total_words: int,
                  audio_dur: float, output: Path,
                  word_ts_map: dict | None = None) -> bool:
    """Quiz: question banner at top, answer reveal banner at bottom."""
    if not items:
        return False

    vf_parts: list[str] = []
    for item in items[:10]:
        ts = _word_idx_to_ts(item.get("word_idx", 0), total_words, audio_dur, word_ts_map)
        ans_ts = ts + 8.0  # reveal answer ~8s after question
        end = ts + 18.0
        q_enable = f"between(t\\,{ts:.2f}\\,{ans_ts:.2f})"
        a_enable = f"between(t\\,{ans_ts:.2f}\\,{end:.2f})"

        question = _escape_drawtext(str(item.get("question", "")))
        answer = _escape_drawtext(str(item.get("answer", "")))

        # Question: cyan banner top-center
        vf_parts += [
            (f"drawtext=text='{question}':"
             f"x=(w-tw)/2:y=25:"
             f"fontsize=30:fontcolor=cyan:"
             f"enable='{q_enable}'"),
            # Answer: gold banner bottom-center
            (f"drawtext=text='Answer\\: {answer}':"
             f"x=(w-tw)/2:y=h-55:"
             f"fontsize=32:fontcolor=yellow:"
             f"enable='{a_enable}'"),
        ]

    if not vf_parts:
        return False

    vf = ",".join(vf_parts)
    _run_ffmpeg([
        "-i", str(video_in),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output),
    ], label="overlay_quiz")
    return True


# -- LUFS normalization ---------------------------------------------------------

def _normalize_lufs(video_in: Path, output: Path,
                    target_lufs: float = -14.0,
                    target_tp: float = -1.0,
                    lra: float = 11.0) -> None:
    """
    Normalize audio loudness to YouTube standard (-14 LUFS).
    Two-pass loudnorm: first pass measures, second pass applies correction.
    """
    # Pass 1: measure current loudness
    fc1 = f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={lra}:print_format=json"
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_in), "-af", fc1, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    stderr = result.stderr
    try:
        start = stderr.rfind("{")
        end = stderr.rfind("}") + 1
        if start >= 0 and end > start:
            meas = json.loads(stderr[start:end])
        else:
            raise ValueError("No JSON found in ffmpeg stderr")
    except Exception as exc:
        log.warning(f"LUFS measurement failed ({exc}), using single-pass loudnorm")
        _run_ffmpeg([
            "-i", str(video_in),
            "-af", f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={lra}",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            str(output),
        ], label="lufs_single_pass")
        return

    # Pass 2: apply measured correction (linear mode for accuracy)
    il = meas.get("input_i", str(target_lufs))
    itp = meas.get("input_tp", "0.0")
    ilra = meas.get("input_lra", str(lra))
    ithresh = meas.get("input_thresh", "-70.0")
    offset = meas.get("target_offset", "0.0")

    fc2 = (
        f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={lra}"
        f":measured_I={il}:measured_TP={itp}"
        f":measured_LRA={ilra}:measured_thresh={ithresh}"
        f":offset={offset}:linear=true:print_format=summary"
    )
    _run_ffmpeg([
        "-i", str(video_in),
        "-af", fc2,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        str(output),
    ], label="lufs_normalize")


# -- Music file lookup (local cache + GCS fallback) ----------------------------

def _get_music_file(niche: str, music_enabled: bool) -> Path | None:
    """
    Resolve the music file for a niche.

    Resolution order:
      1. Local music/ directory (instant)
      2. GCS gs://<bucket>/music/<filename> (downloaded and cached locally)
      3. None (log warning, video proceeds without music)
    """
    if not music_enabled:
        return None
    profile = _get_niche_profile(niche)
    if not profile or not profile.music_file:
        return None

    local_path = MUSIC_DIR / profile.music_file

    # 1. Local cache hit
    if local_path.exists():
        return local_path

    # 2. GCS download -> local cache
    log.info(f"Music '{profile.music_file}' not found locally -- trying GCS...")
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from integrations.gcs_client import download_file
        gcs_path = f"music/{profile.music_file}"
        if download_file(gcs_path, local_path):
            return local_path
    except Exception as exc:
        log.warning(f"GCS music download failed: {exc}")

    # 3. Give up -- proceed without music
    log.warning(
        f"Music file '{profile.music_file}' not found locally or in GCS. "
        "Upload with: python scripts/upload_music.py"
    )
    return None


# -- Phase 2.5: Camera shake ---------------------------------------------------

def _apply_shake(video_in: Path, rank_ts: dict[int, float], output: Path) -> None:
    """
    Apply sinusoidal horizontal camera shake at each rank reveal timestamp.

    Mechanism: crop a 20px-narrower/shorter window and animate its x offset
    as a sine wave during each shake window, then scale back to original
    dimensions.  The 20px headroom (10px each edge) comfortably contains
    the largest ±10px amplitude.

    Amplitude tiers:
      Ranks 1–3:  ±10 px, 0.5 s  (dramatic)
      Ranks 4–6:  ±6 px,  0.4 s  (noticeable)
      Ranks 7–10: ±4 px,  0.3 s  (subtle)

    Shake frequency: 25 Hz — fast impact/explosion feel.
    Y axis is held static (horizontal-only shake reads more natural on video).
    """
    if not rank_ts:
        return

    headroom = 20  # pixels cropped off the total width/height (10 px per edge)
    half = headroom // 2  # crop x/y start in the neutral (no-shake) position

    conditions_x: list[str] = []
    for rank, ts in sorted(rank_ts.items()):
        if rank <= 3:
            amplitude, dur_shake = 10, 0.5
        elif rank <= 6:
            amplitude, dur_shake = 6, 0.4
        else:
            amplitude, dur_shake = 4, 0.3
        end_ts = ts + dur_shake
        conditions_x.append(
            f"if(between(t,{ts:.3f},{end_ts:.3f}),sin(t*25*2*PI)*{amplitude},0)"
        )

    x_expr = "+".join(conditions_x)

    # crop=iw-20:ih-20 gives 10 px headroom on each edge.
    # x position: half + shake offset (0 ≤ x ≤ 20, always in bounds for ≤10 px amp).
    # scale=iw+20:ih+20 restores original dimensions (iw here = post-crop width).
    vf = (
        f"crop=iw-{headroom}:ih-{headroom}:"
        f"'{half}+({x_expr})':"
        f"'{half}',"
        f"scale=iw+{headroom}:ih+{headroom}"
    )

    _run_ffmpeg([
        "-i", str(video_in),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-an",   # raw_video has no audio at this stage
        str(output),
    ], label="camera_shake")


# -- Main assembly --------------------------------------------------------------

def assemble_video(
    clip_paths: list[Path],
    audio_path: Path,
    subtitle_path: Path | None,
    niche: str,
    spec,
    music_volume: float,
    workspace: Path,
    niche_metadata: dict | None = None,
    script_text: str = "",
    emotional_keywords: list[str] | None = None,
) -> Path:
    """
    Run all assembly phases and return the path to final_video.mp4.
    """
    tmp_dir = workspace / "tmp_clips"
    tmp_dir.mkdir(exist_ok=True)

    audio_dur = _probe_duration(audio_path)
    target_dur = min(audio_dur, spec.max_duration_seconds)

    profile = _get_niche_profile(niche)
    clip_sec_min = profile.clip_duration_min if profile else _DEFAULT_CLIP_SEC_MIN
    clip_sec_max = profile.clip_duration_max if profile else _DEFAULT_CLIP_SEC_MAX

    log.info(
        f"VideoAssembler start -- audio={audio_dur:.1f}s  target={target_dur:.1f}s  "
        f"clip_range={clip_sec_min:.1f}-{clip_sec_max:.1f}s  niche={niche}"
    )

    # -- Phase 1: Pre-process clips ------------------------------------------
    proc_clips: list[Path] = []
    proc_durs: list[float] = []
    accumulated = 0.0
    clip_idx = 0
    src_idx = 0

    while accumulated < target_dur - 0.5:
        remaining = target_dur - accumulated
        src = clip_paths[src_idx % len(clip_paths)]

        try:
            src_dur = _probe_duration(src)
        except Exception:
            log.warning(f"  Could not probe {src.name}, skipping")
            src_idx += 1
            continue

        # Vary clip duration within niche range for a natural editing feel
        clip_sec = random.uniform(clip_sec_min, clip_sec_max)
        trim_dur = min(clip_sec, remaining, src_dur)
        if trim_dur < 1.0:
            break

        out = tmp_dir / f"seg_{clip_idx:04d}.mp4"
        log.info(f"  Clip {clip_idx}: {src.name[:40]} -> {trim_dur:.1f}s")
        _preprocess_clip(src, out, clip_idx, trim_dur, niche, spec)

        proc_clips.append(out)
        proc_durs.append(trim_dur)
        accumulated += trim_dur
        clip_idx += 1
        src_idx += 1

    if not proc_clips:
        raise RuntimeError("No clips were successfully pre-processed")

    # Compute clip transition timestamps for SFX layer (cumulative clip ends,
    # excluding the last clip — no transition after the final cut)
    transition_ts: list[float] = []
    t = 0.0
    for dur in proc_durs[:-1]:
        t += dur
        transition_ts.append(t)

    log.info(f"Phase 1 complete -- {len(proc_clips)} clips, {accumulated:.1f}s")

    # -- Phase 2: Xfade concatenation ----------------------------------------
    raw_video = workspace / "raw_video.mp4"
    video_dur = _xfade_concat(proc_clips, proc_durs, niche, raw_video)
    log.info(f"Phase 2 complete -- {raw_video.name}  ({video_dur:.1f}s)")

    # Compute rank_srt_ts once — shared by both Phase 2.5 (camera shake) and
    # Phase 3 (SFX tier selection).  Only relevant for the ranking niche.
    rank_srt_ts: dict[int, float] = {}
    if (niche_metadata and
            niche_metadata.get("overlay_type") == "ranking_card" and
            subtitle_path and subtitle_path.exists()):
        rank_srt_ts = _find_rank_ts_in_srt(subtitle_path)
        if rank_srt_ts:
            log.info(f"  Rank timing: found {len(rank_srt_ts)}/10 spoken cues in SRT")

    # -- Phase 2.5: Camera shake on rank reveals (ranking niche only) ---------
    video_for_mix = raw_video  # default: no shake
    if rank_srt_ts:
        shake_out = workspace / "shaken_video.mp4"
        try:
            _apply_shake(raw_video, rank_srt_ts, shake_out)
            video_for_mix = shake_out
            log.info(f"Phase 2.5 complete -- shake at {len(rank_srt_ts)} rank reveals")
        except Exception as exc:
            log.warning(f"Camera shake failed ({exc}), skipping shake")

    # -- Phase 3: Audio mixing ------------------------------------------------
    audio_mixed = workspace / "video_with_audio.mp4"
    music_file = _get_music_file(niche, spec.music_enabled)

    # Build multi-tier SFX track — context-aware sounds at each clip cut
    sfx_lib = _ensure_sfx_library()
    sfx_track: Path | None = None
    if sfx_lib and transition_ts:
        sfx_track = workspace / "sfx_track.mp3"
        try:
            _build_sfx_track(transition_ts, target_dur, sfx_lib, sfx_track,
                              rank_timestamps=rank_srt_ts or None)
            log.info(f"  SFX track built: {len(transition_ts)} transitions, "
                     f"{len(rank_srt_ts)} rank cues")
        except Exception as exc:
            log.warning(f"  SFX track generation failed ({exc}), skipping SFX layer")
            sfx_track = None

    if music_file:
        log.info(f"Phase 3: mixing with music ({music_file.name}, vol={music_volume})")
    else:
        log.info("Phase 3: mixing VO only (no music file found)")
    _mix_audio(video_for_mix, audio_path, music_file, music_volume, target_dur, audio_mixed,
               sfx_track=sfx_track)
    log.info(f"Phase 3 complete -- {audio_mixed.name}")

    # -- Phase 3.5: Niche overlays --------------------------------------------
    overlaid_video = workspace / "video_overlaid.mp4"
    overlay_applied = False

    if niche_metadata:
        log.info(f"Phase 3.5: applying niche overlays (type={niche_metadata.get('overlay_type')})")
        overlay_applied = _apply_niche_overlays(
            video_in=audio_mixed,
            niche_metadata=niche_metadata,
            audio_duration=audio_dur,
            script_text=script_text,
            output=overlaid_video,
            srt_path=subtitle_path,
        )

    pre_subtitle_video = overlaid_video if overlay_applied else audio_mixed
    if overlay_applied:
        log.info(f"Phase 3.5 complete -- overlays applied")
    else:
        log.info("Phase 3.5: no overlays")

    # -- Phase 4: Subtitle burning --------------------------------------------
    subtitled_video = workspace / "video_subtitled.mp4"

    if subtitle_path and subtitle_path.exists():
        log.info(f"Phase 4: burning subtitles from {subtitle_path.name}")
        _burn_subtitles(pre_subtitle_video, subtitle_path, niche, subtitled_video,
                        emotional_keywords=emotional_keywords or [])
        log.info(f"Phase 4 complete -- {subtitled_video.name}")
        pre_lufs_video = subtitled_video
    else:
        log.info("Phase 4: no subtitles -- skipping burn")
        pre_lufs_video = pre_subtitle_video

    # -- LUFS normalization ---------------------------------------------------
    final_video = workspace / "final_video.mp4"
    log.info("LUFS: normalizing audio to -14 LUFS (YouTube standard)")
    try:
        _normalize_lufs(pre_lufs_video, final_video)
        log.info(f"LUFS normalization complete -- {final_video.name}")
    except Exception as exc:
        log.warning(f"LUFS normalization failed ({exc}), copying pre-LUFS video as final")
        shutil.copy2(str(pre_lufs_video), str(final_video))

    size_mb = final_video.stat().st_size / 1024 / 1024
    log.info(f"VideoAssembler done -- {final_video.name}  ({size_mb:.1f} MB)")
    return final_video


# -- Phase 4: Subtitle burning --------------------------------------------------

def _srt_ffmpeg_path(srt_path: Path) -> str:
    """
    Return the SRT path in a form FFmpeg's subtitles filter can accept on Windows.
    The drive-letter colon must be escaped as \\: to avoid being parsed as an
    option separator inside the filtergraph.
    """
    p = srt_path.as_posix()
    if len(p) >= 2 and p[1] == ":":
        p = p[0] + "\\:" + p[2:]
    return p


def _srt_to_ass(srt_path: Path, style_str: str, ass_path: Path,
                emotional_keywords: list[str] | None = None) -> None:
    """
    Convert an SRT subtitle file to ASS (Advanced SubStation Alpha) format.

    Two layers of word emphasis:
    1. Numbers → cyan color tag  {\\c&H00FFFF&}71,000{\\r}
    2. Emotional keywords → bold {\\b1}died{\\b0}
       Pass emotional_keywords=[] to skip the bold pass.
    """
    import re

    # Parse subtitle_style string → dict
    style_kv: dict[str, str] = {}
    for pair in style_str.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            style_kv[k.strip()] = v.strip()

    fontname    = style_kv.get("Fontname", "Arial")
    fontsize    = style_kv.get("Fontsize", "52")
    primary     = style_kv.get("PrimaryColour", "&H00FFFFFF")
    outline_c   = style_kv.get("OutlineColour", "&H00000000")
    outline_w   = style_kv.get("Outline", "3")
    shadow      = style_kv.get("Shadow", "1")
    alignment   = style_kv.get("Alignment", "2")
    margin_v    = style_kv.get("MarginV", "65")
    bold        = style_kv.get("Bold", "0")

    ass_header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "Collisions: Normal\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{fontname},{fontsize},{primary},&H00FFFFFF,"
        f"{outline_c},&H00000000,{bold},0,0,0,"
        f"100,100,0,0,1,{outline_w},{shadow},{alignment},10,10,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    def _srt_ts_to_ass(ts: str) -> str:
        """Convert SRT timestamp '00:01:23,456' to ASS '0:01:23.46'."""
        ts = ts.replace(",", ".")
        parts = ts.split(":")
        h, m = parts[0], parts[1]
        s_ms = parts[2]
        s, ms = s_ms.split(".")
        cs = ms[:2]  # centiseconds (2 digits)
        return f"{int(h)}:{m}:{s}.{cs}"

    def _highlight_numbers(text: str) -> str:
        """Wrap standalone numbers/stats in cyan color tags."""
        # Match: digits possibly with commas/dots (e.g. 71,000  or  3.5  or  1815)
        return re.sub(
            r'\b(\d[\d,\.]*)\b',
            r'{\\c&H00FFFF&}\1{\\r}',
            text,
        )

    def _highlight_emotional_words(text: str) -> str:
        """Bold-emphasise emotional keywords supplied by Claude Haiku."""
        if not emotional_keywords:
            return text
        pattern = r'\b(' + '|'.join(re.escape(w) for w in emotional_keywords) + r')\b'
        return re.sub(pattern, r'{\\b1}\1{\\b0}', text, flags=re.IGNORECASE)

    srt_text = srt_path.read_text(encoding="utf-8")
    # Split SRT into blocks separated by blank lines
    blocks = re.split(r'\n\s*\n', srt_text.strip())

    dialogue_lines: list[str] = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        # Line 0: index (ignore), Line 1: timestamps, Lines 2+: text
        ts_line = lines[1]
        if " --> " not in ts_line:
            continue
        start_srt, end_srt = ts_line.split(" --> ", 1)
        start_ass = _srt_ts_to_ass(start_srt.strip())
        end_ass   = _srt_ts_to_ass(end_srt.strip())
        text = " ".join(lines[2:]).strip()
        text = _highlight_numbers(text)           # numbers → cyan
        text = _highlight_emotional_words(text)   # key words → bold
        dialogue_lines.append(
            f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}"
        )

    ass_path.write_text(ass_header + "\n".join(dialogue_lines) + "\n", encoding="utf-8")


def _burn_subtitles(video_in: Path, srt_path: Path, niche: str, output: Path,
                    emotional_keywords: list[str] | None = None) -> None:
    profile = _get_niche_profile(niche)
    style = profile.subtitle_style if profile else _DEFAULT_SUBTITLE_STYLE

    # Convert SRT → ASS with number (cyan) + emotional keyword (bold) highlighting
    ass_path = srt_path.with_suffix(".ass")
    try:
        _srt_to_ass(srt_path, style, ass_path, emotional_keywords=emotional_keywords)
        # FFmpeg ass= filter path: escape Windows drive letter colon
        ass_str = _srt_ffmpeg_path(ass_path)
        vf = f"ass='{ass_str}'"
        kw_note = f" + {len(emotional_keywords)} bold keywords" if emotional_keywords else ""
        log.info(f"  subtitle format: ASS (numbers cyan{kw_note})")
    except Exception as exc:
        log.warning(f"  ASS conversion failed ({exc}), falling back to SRT+force_style")
        srt_str = _srt_ffmpeg_path(srt_path)
        vf = f"subtitles='{srt_str}':force_style='{style}'"

    _run_ffmpeg([
        "-i", str(video_in),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output),
    ], label="burn_subtitles")


# -- PipelineStage wrapper ------------------------------------------------------

class VideoAssembler(PipelineStage):
    name = "video_assembler"

    def execute(self, ctx: JobContext) -> JobContext:
        if not ctx.clip_paths:
            raise RuntimeError("No clip paths available for video assembly")
        if not ctx.audio_path or not ctx.audio_path.exists():
            raise RuntimeError("Audio file missing for video assembly")

        music_volume = ctx.resolved.get("music_volume", ctx.format_spec.default_music_volume)

        ctx.video_path = assemble_video(
            clip_paths=ctx.clip_paths,
            audio_path=ctx.audio_path,
            subtitle_path=ctx.subtitle_path,
            niche=ctx.niche,
            spec=ctx.format_spec,
            music_volume=music_volume,
            workspace=ctx.workspace,
            niche_metadata=ctx.niche_metadata,
            script_text=ctx.script_text or "",
            emotional_keywords=ctx.emotional_keywords or [],
        )
        return ctx

    def _load_from_checkpoint(self, ctx: JobContext) -> JobContext:
        path = ctx.workspace / "final_video.mp4"
        if path.exists():
            ctx.video_path = path
        return ctx
