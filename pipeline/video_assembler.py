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

    4 directions cycle per clip_idx so consecutive shots look varied:
      0 -> diagonal drift (top-left to bottom-right)
      1 -> pan left to right
      2 -> pan right to left
      3 -> pan bottom to top
    """
    sw = (int(w * 1.30) + 1) & ~1  # 30% oversize, forced even
    sh = (int(h * 1.30) + 1) & ~1
    dx = sw - w
    dy = sh - h
    dur = max(0.01, duration_secs)

    direction = clip_idx % 4
    if direction == 0:
        crop = f"crop={w}:{h}:x={dx}*t/{dur:.4f}:y={dy}*t/{dur:.4f}"
    elif direction == 1:
        crop = f"crop={w}:{h}:x={dx}*t/{dur:.4f}:y={dy//2}"
    elif direction == 2:
        crop = f"crop={w}:{h}:x={dx}-{dx}*t/{dur:.4f}:y={dy//2}"
    else:
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

def _mix_audio(raw_video: Path, vo_audio: Path, music_path: Path | None,
               music_volume: float, max_dur: float, output: Path) -> None:
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

    fc = (
        "[1:a]apad[vo];"
        f"[2:a]volume={music_volume}[music_raw];"
        "[vo]asplit=2[vo_direct][vo_sc];"
        "[music_raw][vo_sc]sidechaincompress="
        "threshold=0.02:ratio=4:attack=10:release=250[music_ducked];"
        "[vo_direct][music_ducked]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )
    _run_ffmpeg([
        "-i", str(raw_video),
        "-i", str(vo_audio),
        "-i", str(music_path),
        "-filter_complex", fc,
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", str(max_dur),
        str(output),
    ], label="mix_sidechain")


# -- Phase 3.5: Niche overlays -------------------------------------------------

def _word_idx_to_ts(word_idx: int, total_words: int, audio_duration: float) -> float:
    """Convert a word index to an approximate video timestamp."""
    if total_words <= 0:
        return 0.0
    return max(0.0, (word_idx / total_words) * audio_duration)


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
) -> bool:
    """
    Apply time-gated FFmpeg drawtext/drawbox overlays based on niche_metadata.

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

    try:
        if overlay_type == "entity_cards":
            return _overlay_entity_cards(video_in, items, total_words, audio_duration, output)
        elif overlay_type == "fact_counter":
            return _overlay_fact_counter(video_in, items, total_words, audio_duration, output)
        elif overlay_type == "ranking_card":
            return _overlay_ranking_cards(video_in, items, total_words, audio_duration, output)
        elif overlay_type == "myth_stamp":
            return _overlay_myth_stamp(video_in, items, total_words, audio_duration, output)
        elif overlay_type == "scale_text":
            return _overlay_scale_text(video_in, items, total_words, audio_duration, output)
        elif overlay_type == "side_labels":
            return _overlay_side_labels(video_in, items, total_words, audio_duration, output)
        elif overlay_type == "quiz":
            return _overlay_quiz(video_in, items, total_words, audio_duration, output)
        else:
            log.info(f"Overlay type '{overlay_type}' has no renderer -- skipping")
            return False
    except Exception as exc:
        log.warning(f"Overlay pass failed ({overlay_type}): {exc} -- continuing without overlays")
        return False


def _overlay_entity_cards(video_in: Path, items: list, total_words: int,
                           audio_dur: float, output: Path) -> bool:
    """
    Horror: semi-transparent dark box + name/role text at bottom-left.
    Shows for 4 seconds at the moment each entity is first mentioned.
    """
    if not items:
        return False

    vf_parts: list[str] = []
    for item in items[:8]:  # cap at 8 overlays
        ts = _word_idx_to_ts(item.get("word_idx", 0), total_words, audio_dur)
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
                           audio_dur: float, output: Path) -> bool:
    """Shock facts: bold yellow 'FACT #N' counter top-right, updates per fact."""
    if not items:
        return False

    vf_parts: list[str] = []
    sorted_items = sorted(items, key=lambda x: x.get("word_idx", 0))

    for i, item in enumerate(sorted_items):
        ts_start = _word_idx_to_ts(item.get("word_idx", 0), total_words, audio_dur)
        ts_end = (
            _word_idx_to_ts(sorted_items[i + 1].get("word_idx", 0), total_words, audio_dur)
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


def _overlay_ranking_cards(video_in: Path, items: list, total_words: int,
                            audio_dur: float, output: Path) -> bool:
    """Ranking: gold lower-third with rank number and entry name."""
    if not items:
        return False

    vf_parts: list[str] = []
    for item in items[:12]:
        ts = _word_idx_to_ts(item.get("word_idx", 0), total_words, audio_dur)
        end = ts + 5.0
        enable = f"between(t\\,{ts:.2f}\\,{end:.2f})"

        rank = item.get("rank", "?")
        name = _escape_drawtext(str(item.get("name", "")))

        box = (f"drawbox=x=0:y=ih-100:w=iw:h=100:"
               f"color=black@0.5:t=fill:"
               f"enable='{enable}'")
        rank_text = (f"drawtext=text='#{rank}':"
                     f"x=30:y=h-80:"
                     f"fontsize=52:fontcolor=0xFFD700:"
                     f"enable='{enable}'")
        name_text = (f"drawtext=text='{name}':"
                     f"x=120:y=h-65:"
                     f"fontsize=36:fontcolor=white:"
                     f"enable='{enable}'")
        vf_parts += [box, rank_text, name_text]

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
                         audio_dur: float, output: Path) -> bool:
    """What If: subtle scale labels top-left ('Day 1', 'Year 10')."""
    if not items:
        return False

    vf_parts: list[str] = []
    sorted_items = sorted(items, key=lambda x: x.get("word_idx", 0))

    for i, item in enumerate(sorted_items):
        ts_start = _word_idx_to_ts(item.get("word_idx", 0), total_words, audio_dur)
        ts_end = (
            _word_idx_to_ts(sorted_items[i + 1].get("word_idx", 0), total_words, audio_dur)
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
                          audio_dur: float, output: Path) -> bool:
    """Comparison: small corner labels showing both sides + category."""
    if not items:
        return False

    vf_parts: list[str] = []
    for item in items[:6]:
        ts = _word_idx_to_ts(item.get("word_idx", 0), total_words, audio_dur)
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
                  audio_dur: float, output: Path) -> bool:
    """Quiz: question banner at top, answer reveal banner at bottom."""
    if not items:
        return False

    vf_parts: list[str] = []
    for item in items[:10]:
        ts = _word_idx_to_ts(item.get("word_idx", 0), total_words, audio_dur)
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

    log.info(f"Phase 1 complete -- {len(proc_clips)} clips, {accumulated:.1f}s")

    # -- Phase 2: Xfade concatenation ----------------------------------------
    raw_video = workspace / "raw_video.mp4"
    video_dur = _xfade_concat(proc_clips, proc_durs, niche, raw_video)
    log.info(f"Phase 2 complete -- {raw_video.name}  ({video_dur:.1f}s)")

    # -- Phase 3: Audio mixing ------------------------------------------------
    audio_mixed = workspace / "video_with_audio.mp4"
    music_file = _get_music_file(niche, spec.music_enabled)
    if music_file:
        log.info(f"Phase 3: mixing with music ({music_file.name}, vol={music_volume})")
    else:
        log.info("Phase 3: mixing VO only (no music file found)")
    _mix_audio(raw_video, audio_path, music_file, music_volume, target_dur, audio_mixed)
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
        _burn_subtitles(pre_subtitle_video, subtitle_path, niche, subtitled_video)
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


def _burn_subtitles(video_in: Path, srt_path: Path, niche: str, output: Path) -> None:
    profile = _get_niche_profile(niche)
    style = profile.subtitle_style if profile else _DEFAULT_SUBTITLE_STYLE
    srt_str = _srt_ffmpeg_path(srt_path)

    _run_ffmpeg([
        "-i", str(video_in),
        "-vf", f"subtitles='{srt_str}':force_style='{style}'",
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
        )
        return ctx

    def _load_from_checkpoint(self, ctx: JobContext) -> JobContext:
        path = ctx.workspace / "final_video.mp4"
        if path.exists():
            ctx.video_path = path
        return ctx
