"""
test_all_niches.py -- Full pipeline test for all 7 niches.

For each niche:
  Stage 1: Generate script (real Claude Opus, niche-specific prompt)
  Stage 2: Generate audio (Google TTS at niche speaking rate)
  Stage 3: Generate subtitles (Whisper transcription)
  Stage 4: Download clips (per-segment Pexels strategy)
  Stage 5: Assemble 90-second video (Ken Burns + overlays + LUFS)

Output: dev/workspace_<niche>/
        ├-- script.txt
        ├-- audio.mp3
        ├-- subtitles.srt
        ├-- clips/
        ├-- final_video.mp4
        └-- test_summary.txt

Run: python scripts/test_all_niches.py
"""
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(
            io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            if hasattr(sys.stdout, "buffer") else sys.stdout
        ),
    ]
)
log = logging.getLogger("test_all_niches")

# -- Niche -> Topic mapping -----------------------------------------------------
NICHE_TOPICS = {
    "horror":             "The Isdal Woman: Norway's Most Mysterious Unsolved Death",
    "what_if":            "What If All Satellites Disappeared Overnight",
    "shock_facts":        "10 Shocking Facts About the Human Brain",
    "quiz":               "The Ultimate World Capitals Quiz",
    "ranking":            "Top 10 Most Extreme Natural Disasters in History",
    "historical_versus":  "Roman Empire vs Han Dynasty: Which Shaped the Modern World More?",
    "myth_busting":       "The Great Wall of China Can Be Seen from Space",
}

FORMAT = "full_length"
# Cap video at 90s for test speed (shows all pipeline stages without 8-min assembly)
TEST_VIDEO_MAX_SEC = 90
# Clips per segment for test (reduced from production for speed)
TEST_CLIPS_PER_SEGMENT = 3
# Whisper model
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

BASE_DIR = Path(__file__).parent.parent
RESULTS: dict[str, dict] = {}


# -- Helpers -------------------------------------------------------------------

def section(msg: str):
    log.info("")
    log.info("-" * 70)
    log.info(f"  {msg}")
    log.info("-" * 70)


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms = int(round((seconds % 1) * 1000))
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# -- Stage functions -----------------------------------------------------------

def stage1_script(ws: Path, niche: str, topic: str,
                  profile, spec, model: str, haiku_model: str, max_tokens: int) -> str:
    from integrations import claude_client

    log.info(f"[{niche}] Stage 1: Generating script via Claude ({model})")
    script = claude_client.write_script(
        topic=topic,
        style_notes="",
        niche=niche,
        script_prompt_suffix=spec.script_prompt_suffix,
        target_word_count=profile.target_word_count,
        model=model,
        max_tokens=max_tokens,
    )
    (ws / "script.txt").write_text(script, encoding="utf-8")
    wc = len(script.split())
    log.info(f"[{niche}] script.txt -> {wc} words (target: {profile.target_word_count})")

    # Extract niche overlay metadata
    log.info(f"[{niche}] Extracting niche metadata (overlay_type={profile.overlay_type})...")
    niche_metadata = claude_client.extract_niche_metadata(
        script_text=script, niche=niche, haiku_model=haiku_model
    )
    if niche_metadata:
        (ws / "niche_metadata.json").write_text(
            json.dumps(niche_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        items = niche_metadata.get("items", [])
        count = len(items) if isinstance(items, list) else (1 if items else 0)
        log.info(f"[{niche}] niche_metadata.json -> overlay={niche_metadata.get('overlay_type')}, {count} items")

    # Extract emotional keywords for bold subtitle emphasis
    log.info(f"[{niche}] Extracting emotional keywords for subtitle emphasis...")
    emotional_keywords = claude_client.extract_emotional_keywords(
        script_text=script, haiku_model=haiku_model
    )
    if emotional_keywords:
        (ws / "emotional_keywords.json").write_text(
            json.dumps(emotional_keywords, ensure_ascii=False), encoding="utf-8"
        )
        log.info(f"[{niche}] emotional_keywords.json -> {emotional_keywords}")

    return script, emotional_keywords


def stage2_tts(ws: Path, niche: str, script: str, profile, spec) -> Path:
    from pipeline.tts_generator import generate_audio_gemini, _get_gemini_voice

    audio_path = ws / "audio.mp3"
    voice_name = _get_gemini_voice(niche)
    speaking_rate = profile.speaking_rate * spec.speaking_rate_multiplier

    log.info(f"[{niche}] Stage 2: Gemini TTS voice={voice_name} rate={speaking_rate:.2f}")
    generate_audio_gemini(
        script_text=script,
        niche=niche,
        output_path=audio_path,
        speaking_rate=speaking_rate,
    )

    dur = probe_duration(audio_path)
    log.info(f"[{niche}] audio.mp3 -> {audio_path.stat().st_size//1024} KB, {dur:.1f}s")
    return audio_path


def stage3_subtitles(ws: Path, niche: str, audio_path: Path, script: str) -> Path:
    srt_path = ws / "subtitles.srt"

    try:
        from faster_whisper import WhisperModel
        log.info(f"[{niche}] Stage 3: Whisper transcription (model={WHISPER_MODEL})")
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_path), word_timestamps=True, language="en")

        entries, idx = [], 1
        for seg in segments:
            if not seg.words:
                if seg.text.strip():
                    entries.append(f"{idx}\n{srt_time(seg.start)} --> {srt_time(seg.end)}\n{seg.text.strip()}\n")
                    idx += 1
                continue
            current, cur_words = [], []
            for w in seg.words:
                cur_words.append(w)
                text = w.word.strip()
                if text.endswith(('.', '!', '?')) or (text.endswith(',') and len(cur_words) >= 3) or len(cur_words) >= 5:
                    line = " ".join(x.word.strip() for x in cur_words).strip()
                    if line:
                        entries.append(f"{idx}\n{srt_time(cur_words[0].start)} --> {srt_time(cur_words[-1].end)}\n{line}\n")
                        idx += 1
                    cur_words = []
            if cur_words:
                line = " ".join(x.word.strip() for x in cur_words).strip()
                if line:
                    entries.append(f"{idx}\n{srt_time(cur_words[0].start)} --> {srt_time(cur_words[-1].end)}\n{line}\n")
                    idx += 1

        srt_path.write_text("\n".join(entries), encoding="utf-8")
        log.info(f"[{niche}] subtitles.srt -> {idx-1} entries")

    except Exception as exc:
        log.warning(f"[{niche}] Whisper failed ({exc}), using estimated SRT")
        words = re.sub(r'\s+', ' ', script.strip()).split()
        t, idx, entries = 0.0, 1, []
        for i in range(0, len(words), 5):
            group = words[i:i+5]
            dur = len(group) / 2.5
            entries.append(f"{idx}\n{srt_time(t)} --> {srt_time(t+dur)}\n{' '.join(group)}\n")
            t += dur; idx += 1
        srt_path.write_text("\n".join(entries), encoding="utf-8")
        log.info(f"[{niche}] subtitles.srt (estimated) -> {idx-1} entries")

    return srt_path


def stage4_clips(ws: Path, niche: str, script: str, audio_path: Path,
                 profile, spec, haiku_model: str) -> list[Path]:
    from integrations import pexels_client, claude_client

    clips_dir = ws / "clips"
    clips_dir.mkdir(exist_ok=True)

    # For ranking niche, use per-rank keywords from niche_metadata for visual variety
    nm_path = ws / "niche_metadata.json"
    rank_items = []
    if niche == "ranking" and nm_path.exists():
        try:
            niche_metadata = json.loads(nm_path.read_text())
            rank_items = niche_metadata.get("items", [])
        except Exception:
            pass

    if rank_items:
        log.info(f"[{niche}] Stage 4: Extracting per-rank keywords ({len(rank_items)} rank entries) ...")
        keywords = claude_client.extract_ranking_clip_keywords(
            rank_items=rank_items,
            script_text=script,
            haiku_model=haiku_model,
        )
        log.info(f"[{niche}] Per-rank keywords ({len(keywords)}): {keywords}")
    else:
        audio_dur = probe_duration(audio_path)
        # For 90s test video, we only need clips for that duration
        effective_dur = min(audio_dur, TEST_VIDEO_MAX_SEC * 1.5)  # slight overshoot
        seg_dur = 30.0
        num_segments = max(2, round(effective_dur / seg_dur))

        log.info(f"[{niche}] Stage 4: Extracting {num_segments} segment keywords for {effective_dur:.0f}s ...")
        keywords = claude_client.extract_segment_keywords(
            script_text=script,
            segment_duration_sec=seg_dur,
            audio_duration_sec=effective_dur,
            haiku_model=haiku_model,
            niche=niche,
        )
        log.info(f"[{niche}] Keywords: {keywords}")

    seen_ids: set = set()
    all_clips: list[Path] = []
    clip_idx = 0

    for kw in keywords:
        new = pexels_client.download_clips_for_keyword(
            keyword=kw,
            count=TEST_CLIPS_PER_SEGMENT,
            dest_dir=clips_dir,
            clip_index_start=clip_idx,
            orientation=spec.clip_orientation,
            min_duration=3,
            seen_ids=seen_ids,
        )
        all_clips.extend(new)
        clip_idx += len(new)
        log.info(f"[{niche}]   '{kw}': {len(new)} clips  (total: {len(all_clips)})")

    log.info(f"[{niche}] clips/ -> {len(all_clips)} unique clips ({len(seen_ids)} unique Pexels IDs)")
    return sorted(clips_dir.glob("*.mp4"))


def stage5_video(ws: Path, niche: str, clip_paths: list[Path],
                 audio_path: Path, srt_path: Path, spec, profile,
                 script: str, niche_metadata: dict,
                 emotional_keywords: list[str] | None = None) -> Path:
    from pipeline.video_assembler import assemble_video
    from formats.base import FormatSpec
    import dataclasses

    # Override max duration for test speed
    test_spec = dataclasses.replace(spec, max_duration_seconds=TEST_VIDEO_MAX_SEC)

    log.info(f"[{niche}] Stage 5: Assembling {TEST_VIDEO_MAX_SEC}s video ...")
    result = assemble_video(
        clip_paths=clip_paths,
        audio_path=audio_path,
        subtitle_path=srt_path if srt_path.exists() else None,
        niche=niche,
        spec=test_spec,
        music_volume=0.08,
        workspace=ws,
        niche_metadata=niche_metadata,
        script_text=script,
        emotional_keywords=emotional_keywords or [],
    )
    size_mb = result.stat().st_size / 1024 / 1024
    log.info(f"[{niche}] final_video.mp4 -> {size_mb:.1f} MB")
    return result


# -- Main ----------------------------------------------------------------------

def run_niche(niche: str, topic: str):
    from formats import get_format_spec
    from pipeline.niche_config import get_niche_profile
    from core.config_loader import load_master_infra

    ws = BASE_DIR / "dev" / f"workspace_{niche}"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "clips").mkdir(exist_ok=True)

    spec = get_format_spec(FORMAT)
    profile = get_niche_profile(niche)
    infra = load_master_infra(BASE_DIR)
    claude_cfg = infra.get("claude", {})
    model       = claude_cfg.get("model", "claude-opus-4-5")
    haiku_model = claude_cfg.get("haiku_model", "claude-haiku-4-5-20251001")
    max_tokens  = claude_cfg.get("max_tokens", 4096)

    result = {
        "niche": niche,
        "topic": topic,
        "workspace": str(ws),
        "stages": {},
        "errors": [],
    }
    t_start = time.time()

    section(f"NICHE: {niche.upper()} -- {topic}")

    # Stage 1: Script
    emotional_keywords: list[str] = []
    try:
        t = time.time()
        script, emotional_keywords = stage1_script(ws, niche, topic, profile, spec, model, haiku_model, max_tokens)
        result["stages"]["script"] = {
            "words": len(script.split()),
            "time_s": round(time.time() - t, 1),
            "hook": " ".join(script.split()[:50]),  # first 50 words
        }
    except Exception as exc:
        log.error(f"[{niche}] Stage 1 FAILED: {exc}")
        result["errors"].append(f"script: {exc}")
        return result

    # Stage 2: TTS
    try:
        t = time.time()
        audio_path = stage2_tts(ws, niche, script, profile, spec)
        dur = probe_duration(audio_path)
        result["stages"]["tts"] = {
            "duration_s": round(dur, 1),
            "size_kb": audio_path.stat().st_size // 1024,
            "time_s": round(time.time() - t, 1),
        }
    except Exception as exc:
        log.error(f"[{niche}] Stage 2 FAILED: {exc}")
        result["errors"].append(f"tts: {exc}")
        return result

    # Stage 3: Subtitles
    try:
        t = time.time()
        srt_path = stage3_subtitles(ws, niche, audio_path, script)
        result["stages"]["subtitles"] = {
            "entries": srt_path.read_text().count("\n\n"),
            "time_s": round(time.time() - t, 1),
        }
    except Exception as exc:
        log.warning(f"[{niche}] Stage 3 FAILED (non-fatal): {exc}")
        result["errors"].append(f"subtitles: {exc}")
        srt_path = ws / "subtitles.srt"

    # Stage 4: Clips
    try:
        t = time.time()
        clip_paths = stage4_clips(ws, niche, script, audio_path, profile, spec, haiku_model)
        result["stages"]["clips"] = {
            "count": len(clip_paths),
            "time_s": round(time.time() - t, 1),
        }
    except Exception as exc:
        log.error(f"[{niche}] Stage 4 FAILED: {exc}")
        result["errors"].append(f"clips: {exc}")
        return result

    # Stage 5: Video
    niche_metadata = {}
    nm_path = ws / "niche_metadata.json"
    if nm_path.exists():
        try:
            niche_metadata = json.loads(nm_path.read_text())
        except Exception:
            pass

    try:
        t = time.time()
        video_path = stage5_video(
            ws, niche, clip_paths, audio_path, srt_path, spec, profile,
            script, niche_metadata, emotional_keywords
        )
        result["stages"]["video"] = {
            "size_mb": round(video_path.stat().st_size / 1024 / 1024, 1),
            "path": str(video_path),
            "time_s": round(time.time() - t, 1),
        }
    except Exception as exc:
        log.error(f"[{niche}] Stage 5 FAILED: {exc}")
        result["errors"].append(f"video: {exc}")

    result["total_time_s"] = round(time.time() - t_start, 1)

    # Write per-niche summary
    summary_lines = [
        f"NICHE: {niche.upper()}",
        f"TOPIC: {topic}",
        f"TOTAL TIME: {result['total_time_s']}s",
        "",
        "-- SCRIPT -------------------------------------------",
        f"Words: {result['stages'].get('script', {}).get('words', '?')}",
        "",
        "Hook (first 50 words):",
        result["stages"].get("script", {}).get("hook", ""),
        "",
        "-- STAGES -------------------------------------------",
    ]
    for stage, data in result["stages"].items():
        summary_lines.append(f"  {stage}: {data}")
    if result["errors"]:
        summary_lines.append("")
        summary_lines.append("-- ERRORS -------------------------------------------")
        for e in result["errors"]:
            summary_lines.append(f"  {e}")

    (ws / "test_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    return result


def main():
    section("TEST ALL NICHES -- Full Pipeline")
    log.info(f"Niches: {list(NICHE_TOPICS.keys())}")
    log.info(f"Video cap: {TEST_VIDEO_MAX_SEC}s per niche | Clips per segment: {TEST_CLIPS_PER_SEGMENT}")
    log.info("")

    all_results = []
    for niche, topic in NICHE_TOPICS.items():
        try:
            result = run_niche(niche, topic)
            all_results.append(result)
        except Exception as exc:
            log.error(f"Niche '{niche}' crashed: {exc}")
            all_results.append({"niche": niche, "topic": topic, "crash": str(exc)})

    # -- Final summary ---------------------------------------------------------
    section("FINAL RESULTS -- ALL NICHES")
    for r in all_results:
        niche = r.get("niche", "?")
        errors = r.get("errors", [])
        stages = r.get("stages", {})
        crash = r.get("crash")

        if crash:
            log.info(f"  [{niche}] CRASH: {crash}")
            continue

        script_words = stages.get("script", {}).get("words", "?")
        audio_dur = stages.get("tts", {}).get("duration_s", "?")
        clip_count = stages.get("clips", {}).get("count", "?")
        video_mb = stages.get("video", {}).get("size_mb", "?")
        total_t = r.get("total_time_s", "?")
        status = "OK" if not errors else f"PARTIAL ({len(errors)} errors)"

        log.info(f"  [{niche:12s}] {status:20s} | script={script_words}w | audio={audio_dur}s | clips={clip_count} | video={video_mb}MB | total={total_t}s")
        hook = stages.get("script", {}).get("hook", "")
        log.info(f"               hook: {hook[:100]}...")
        log.info("")

    # Write master results JSON
    results_path = BASE_DIR / "dev" / "test_results.json"
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    log.info(f"Full results written to {results_path}")

    section("DONE")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full pipeline test for one or all niches")
    parser.add_argument("--niche", "-n", help="Run only this niche (e.g. ranking, horror)")
    args = parser.parse_args()

    if args.niche:
        if args.niche not in NICHE_TOPICS:
            print(f"Unknown niche '{args.niche}'. Valid: {list(NICHE_TOPICS.keys())}")
            sys.exit(1)
        topic = NICHE_TOPICS[args.niche]
        NICHE_TOPICS.clear()
        NICHE_TOPICS[args.niche] = topic

    main()
