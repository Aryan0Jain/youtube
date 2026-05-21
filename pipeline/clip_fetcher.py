"""
ClipFetcher — Stage 3 of the pipeline.

Per-segment clip strategy (produces 80-100 unique clips for an 8-min video):

  1. Probe audio duration to determine how many 30-second segments exist.
  2. Ask Claude Haiku for ONE visual search phrase per segment.
     Example: 16 segments → 16 specific, visually distinct keywords.
  3. Download clips_per_segment clips per keyword (from config/niches.yaml).
     Example: 16 keywords × 6 clips = 96 unique clips.
  4. Deduplicate by Pexels video ID — the same footage never appears twice.

This replaces the old strategy of downloading ~10 clips total and looping them
3-4 times, which viewers notice and close the tab.
"""
import json
import logging
import os
import shutil
from pathlib import Path

from pipeline.base import PipelineStage, JobContext
from integrations import pexels_client, claude_client

log = logging.getLogger(__name__)

USE_MOCK = os.environ.get("USE_MOCK_PEXELS") == "1"
FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "clips"

# Duration of each script segment in seconds
SEGMENT_DURATION_SEC = 30.0

# Minimum clips to aim for — if fewer are available, video assembler will loop
MIN_CLIPS_TARGET = 30


class ClipFetcher(PipelineStage):
    name = "clip_fetcher"

    def execute(self, ctx: JobContext) -> JobContext:
        clips_dir = ctx.workspace / "clips"
        clips_dir.mkdir(exist_ok=True)
        spec = ctx.format_spec

        if USE_MOCK:
            ctx.clip_paths = _copy_fixture_clips(clips_dir, spec.clip_orientation)
            log.info(f"Mock: {len(ctx.clip_paths)} fixture clips copied")
            return ctx

        script = ctx.script_text or ""
        claude_cfg = ctx.resolved.get("claude", {})
        haiku_model = claude_cfg.get("haiku_model", "claude-haiku-4-5-20251001")

        # Load niche profile for clips_per_segment setting
        try:
            from pipeline.niche_config import get_niche_profile
            profile = get_niche_profile(ctx.niche)
            clips_per_segment = profile.clips_per_segment
        except Exception:
            clips_per_segment = 5  # safe fallback

        # Probe audio duration if available (for accurate segment count)
        audio_duration = _probe_audio_duration(ctx)
        if audio_duration is None:
            # Estimate from word count: ~130 words/min
            word_count = len(script.split())
            audio_duration = word_count / 130.0 * 60.0
            log.info(f"Audio not yet available; estimated duration={audio_duration:.0f}s from word count")
        else:
            log.info(f"Audio duration: {audio_duration:.1f}s")

        # Step 1: Extract keywords — one per rank entry for ranking niche,
        #         or one per 30-second segment for all other niches.
        rank_items = (
            ctx.niche_metadata.get("items", [])
            if ctx.niche == "ranking" and ctx.niche_metadata
            else []
        )

        if rank_items:
            log.info(
                f"Ranking niche: extracting per-rank keywords "
                f"({len(rank_items)} rank entries)..."
            )
            segment_keywords = claude_client.extract_ranking_clip_keywords(
                rank_items=rank_items,
                script_text=script,
                haiku_model=haiku_model,
            )
            log.info(f"Per-rank keywords ({len(segment_keywords)}): {segment_keywords}")
        else:
            log.info(f"Extracting segment keywords for ~{audio_duration:.0f}s audio "
                     f"({SEGMENT_DURATION_SEC:.0f}s/segment)...")
            segment_keywords = claude_client.extract_segment_keywords(
                script_text=script,
                segment_duration_sec=SEGMENT_DURATION_SEC,
                audio_duration_sec=audio_duration,
                haiku_model=haiku_model,
                niche=ctx.niche,
            )
            log.info(f"Segment keywords ({len(segment_keywords)}): {segment_keywords}")

        # Step 2: Download clips_per_segment clips per keyword, deduplicating by Pexels video ID
        rate_buffer = ctx.resolved.get("pexels", {}).get("rate_limit_buffer", 5)
        seen_ids: set[int] = set()
        all_clips: list[Path] = []
        clip_index = 0

        for kw in segment_keywords:
            new_clips = pexels_client.download_clips_for_keyword(
                keyword=kw,
                count=clips_per_segment,
                dest_dir=clips_dir,
                clip_index_start=clip_index,
                orientation=spec.clip_orientation,
                min_duration=int(spec.clip_min_seconds),
                rate_buffer=rate_buffer,
                seen_ids=seen_ids,
            )
            all_clips.extend(new_clips)
            clip_index += len(new_clips)
            log.info(f"  '{kw}': {len(new_clips)} clips  (total so far: {len(all_clips)})")

        if not all_clips:
            raise RuntimeError(
                "No video clips could be downloaded from Pexels. "
                "Check PEXELS_API_KEY and rate limits."
            )

        # Save segment-to-keyword mapping for transparency (dev + debug)
        mapping_path = clips_dir / "keywords.json"
        mapping_path.write_text(
            json.dumps({"keywords": segment_keywords, "total_clips": len(all_clips)},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        ctx.clip_paths = sorted(clips_dir.glob("*.mp4"))
        log.info(
            f"Clip fetcher done: {len(ctx.clip_paths)} unique clips "
            f"({len(seen_ids)} unique Pexels IDs) for {len(segment_keywords)} keywords"
        )
        return ctx

    def _load_from_checkpoint(self, ctx: JobContext) -> JobContext:
        clips_dir = ctx.workspace / "clips"
        if clips_dir.exists():
            ctx.clip_paths = sorted(clips_dir.glob("*.mp4"))
        return ctx


# ── Helpers ───────────────────────────────────────────────────────────────────

def _probe_audio_duration(ctx: JobContext) -> float | None:
    """Return audio duration in seconds, or None if audio is not available."""
    import json
    import subprocess

    audio_path = ctx.audio_path or ctx.workspace / "audio.mp3"
    if not audio_path or not audio_path.exists():
        return None

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(audio_path)],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception as exc:
        log.warning(f"Could not probe audio duration: {exc}")
        return None


def _copy_fixture_clips(dest_dir: Path, orientation: str) -> list[Path]:
    """Copy fixture clips from tests/fixtures/clips/ for mock mode."""
    result: list[Path] = []
    if FIXTURE_DIR.exists():
        for i, src in enumerate(sorted(FIXTURE_DIR.glob("*.mp4"))[:8]):
            dest = dest_dir / f"clip_{i:03d}_{src.name}"
            shutil.copy2(src, dest)
            result.append(dest)

    if not result:
        placeholder = dest_dir / "clip_000_placeholder.mp4"
        placeholder.write_bytes(b"")
        result.append(placeholder)
        log.warning(
            f"No fixture clips found in {FIXTURE_DIR}. "
            "Add .mp4 files there for mock testing."
        )

    return result
