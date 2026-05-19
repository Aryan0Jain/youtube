import logging
import os
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

PEXELS_VIDEO_SEARCH = "https://api.pexels.com/videos/search"
RATE_LIMIT_BUFFER = 5          # stop if remaining quota drops below this


def _api_key() -> str:
    key = os.environ.get("PEXELS_API_KEY", "")
    if not key:
        raise EnvironmentError("PEXELS_API_KEY environment variable not set")
    return key


def search_videos(keyword: str, orientation: str = "landscape",
                  per_page: int = 5) -> tuple[list[dict], dict]:
    """
    Search Pexels for videos matching keyword.
    Returns (video_list, rate_limit_info).
    rate_limit_info: {"remaining": int, "reset_ts": int}
    """
    headers = {"Authorization": _api_key()}
    params = {
        "query": keyword,
        "orientation": orientation,
        "per_page": per_page,
        "size": "large",
    }
    resp = requests.get(PEXELS_VIDEO_SEARCH, headers=headers, params=params, timeout=30)
    resp.raise_for_status()

    rate_info = {
        "remaining": int(resp.headers.get("X-Ratelimit-Remaining", 999)),
        "reset_ts": int(resp.headers.get("X-Ratelimit-Reset", 0)),
    }

    videos = resp.json().get("videos", [])
    return videos, rate_info


def pick_best_clip(videos: list[dict], orientation: str,
                   min_duration: int = 5) -> str | None:
    """
    Select the best video file URL from Pexels results.
    Prefers HD (width ≥ 1920 for landscape, height ≥ 1920 for portrait).
    Falls back to 720p if no HD found.
    """
    if not videos:
        return None

    def _score(v: dict) -> int:
        if v.get("duration", 0) < min_duration:
            return -1
        # Find the best video file
        files = v.get("video_files", [])
        for f in files:
            if orientation == "portrait":
                if f.get("height", 0) >= 1920:
                    return 2
                if f.get("height", 0) >= 1280:
                    return 1
            else:
                if f.get("width", 0) >= 1920:
                    return 2
                if f.get("width", 0) >= 1280:
                    return 1
        return 0

    best_video = max(videos, key=_score, default=None)
    if not best_video or _score(best_video) < 0:
        return None

    files = best_video.get("video_files", [])
    # Select best quality file
    target_key = "height" if orientation == "portrait" else "width"
    files_sorted = sorted(files, key=lambda f: f.get(target_key, 0), reverse=True)
    for f in files_sorted:
        if f.get("link"):
            return f["link"]

    return None


def download_clip(url: str, dest_path: Path) -> Path:
    """Stream-download a video clip to dest_path."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            f.write(chunk)
    return dest_path


def download_clips_for_keyword(
    keyword: str,
    count: int,
    dest_dir: Path,
    clip_index_start: int = 0,
    orientation: str = "landscape",
    min_duration: int = 4,
    rate_buffer: int = RATE_LIMIT_BUFFER,
    seen_ids: set[int] | None = None,
) -> list[Path]:
    """
    Download up to `count` unique clips matching `keyword` from Pexels.

    seen_ids:  a caller-owned set of Pexels video IDs already downloaded this
               session.  Clips whose IDs are in this set are skipped so the same
               footage never repeats across keywords.  Pass the same set object
               for every call to share the dedup state.

    Returns a list of local paths for clips actually downloaded.
    """
    if seen_ids is None:
        seen_ids = set()

    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    # Search with a larger per_page so we have candidates after dedup
    per_page = min(80, max(count * 3, 10))
    try:
        videos, rate_info = search_videos(keyword, orientation, per_page=per_page)
        _check_rate_limit(rate_info, rate_buffer)
    except Exception as exc:
        log.warning(f"Pexels search failed for '{keyword}': {exc}")
        return downloaded

    # Filter: skip already-seen IDs and too-short clips
    candidates = [
        v for v in videos
        if v.get("id") not in seen_ids
        and v.get("duration", 0) >= min_duration
    ]

    for video in candidates:
        if len(downloaded) >= count:
            break

        vid_id = video.get("id")
        url = _best_file_url(video, orientation)
        if not url:
            continue

        safe_kw = _sanitize(keyword)
        idx = clip_index_start + len(downloaded)
        dest = dest_dir / f"clip_{idx:04d}_{safe_kw}.mp4"

        try:
            download_clip(url, dest)
            seen_ids.add(vid_id)
            downloaded.append(dest)
            log.info(f"  [{idx}] '{keyword}' (id={vid_id}) -> {dest.name}")
        except Exception as exc:
            log.warning(f"  Download failed for '{keyword}' id={vid_id}: {exc}")

    return downloaded


def _best_file_url(video: dict, orientation: str) -> str | None:
    """Return the URL of the best-quality video file in a Pexels video dict."""
    files = video.get("video_files", [])
    if not files:
        return None
    target_key = "height" if orientation == "portrait" else "width"
    files_sorted = sorted(files, key=lambda f: f.get(target_key, 0), reverse=True)
    for f in files_sorted:
        if f.get("link"):
            return f["link"]
    return None


def download_clips_for_keywords(keywords: list[str], dest_dir: Path,
                                orientation: str = "landscape",
                                min_duration: int = 5,
                                rate_buffer: int = RATE_LIMIT_BUFFER) -> list[Path]:
    """
    For each keyword, search Pexels and download the best clip.
    Deduplicates keywords. Falls back to 'abstract background' on failure.
    Returns ordered list of local paths (same order as input keywords).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Dedup while preserving order
    seen: set[str] = set()
    unique_keywords: list[str] = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw)

    # Build keyword → path map
    kw_to_path: dict[str, Path] = {}

    for i, kw in enumerate(unique_keywords):
        dest = dest_dir / f"clip_{i:03d}_{_sanitize(kw)}.mp4"
        url = _search_with_fallback(kw, orientation, min_duration, rate_buffer)

        if url:
            try:
                download_clip(url, dest)
                kw_to_path[kw.lower()] = dest
                log.info(f"Downloaded clip for '{kw}': {dest.name}")
            except Exception as e:
                log.warning(f"Download failed for '{kw}': {e}")
                kw_to_path[kw.lower()] = _get_fallback_clip(dest_dir, orientation, rate_buffer)
        else:
            kw_to_path[kw.lower()] = _get_fallback_clip(dest_dir, orientation, rate_buffer)

    # Map original ordered keywords to paths
    result: list[Path] = []
    for kw in keywords:
        path = kw_to_path.get(kw.lower())
        if path and path.exists():
            result.append(path)
        else:
            log.warning(f"No clip for keyword '{kw}' — skipping")

    return result


def _search_with_fallback(keyword: str, orientation: str,
                           min_duration: int, rate_buffer: int) -> str | None:
    try:
        videos, rate_info = search_videos(keyword, orientation)
        _check_rate_limit(rate_info, rate_buffer)
        url = pick_best_clip(videos, orientation, min_duration)
        if url:
            return url
        # Try broader search without orientation filter
        videos, rate_info = search_videos(keyword, "landscape", per_page=10)
        _check_rate_limit(rate_info, rate_buffer)
        return pick_best_clip(videos, orientation, min_duration=3)
    except requests.HTTPError as e:
        log.warning(f"Pexels search failed for '{keyword}': {e}")
        return None


def _get_fallback_clip(dest_dir: Path, orientation: str, rate_buffer: int) -> Path | None:
    fallback_kw = "abstract background" if orientation == "landscape" else "nature vertical"
    dest = dest_dir / "clip_fallback.mp4"
    if dest.exists():
        return dest
    url = _search_with_fallback(fallback_kw, orientation, 3, rate_buffer)
    if url:
        try:
            download_clip(url, dest)
            return dest
        except Exception:
            pass
    return None


def _check_rate_limit(rate_info: dict, buffer: int):
    remaining = rate_info.get("remaining", 999)
    if remaining < buffer:
        reset_ts = rate_info.get("reset_ts", 0)
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc).timestamp()
        sleep_secs = max(0, int(reset_ts - now)) + 2
        log.warning(f"Pexels rate limit low ({remaining} remaining). Sleeping {sleep_secs}s")
        time.sleep(sleep_secs)


def _sanitize(keyword: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in keyword)[:30]
