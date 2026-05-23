import logging
import os
import time
from pathlib import Path

from pipeline.base import PipelineStage, JobContext

log = logging.getLogger(__name__)

USE_MOCK = os.environ.get("USE_MOCK_YOUTUBE") == "1"

NICHE_CATEGORY_MAP = {
    "horror":      "24",   # Entertainment
    "quiz":        "26",   # Howto & Style
    "historical_versus": "28",   # Science & Technology
    "what_if":     "28",
    "shock_facts": "28",
    "ranking":     "28",
    "myth_busting": "28",
}


class YouTubeUploader(PipelineStage):
    name = "youtube_uploader"

    def execute(self, ctx: JobContext) -> JobContext:
        if USE_MOCK:
            fake_id = f"mock_{ctx.job_id}_{ctx.channel_id}"
            ctx.youtube_video_id = fake_id
            url = f"https://youtu.be/{fake_id}"
            log.info(f"Mock upload: {url}")
            ctx.db.record_upload(ctx.job_id, ctx.channel_id, fake_id, ctx.video_title, url)
            return ctx

        from integrations.youtube_client import get_client
        from googleapiclient.http import MediaFileUpload

        if not ctx.video_path or not ctx.video_path.exists():
            raise RuntimeError("Video file missing for upload")

        client = get_client(ctx.resolved["oauth_token_path"])
        category_id = NICHE_CATEGORY_MAP.get(ctx.niche, "28")
        privacy = ctx.resolved.get("upload_privacy", "private")

        video_body = {
            "snippet": {
                "title": ctx.video_title[:100],
                "description": ctx.video_description[:5000],
                "tags": ctx.video_tags[:500],
                "categoryId": category_id,
                "defaultLanguage": "en",
            },
            "status": {
                "privacyStatus": "private",  # always upload private first
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(
            str(ctx.video_path),
            mimetype="video/mp4",
            chunksize=4 * 1024 * 1024,
            resumable=True,
        )

        log.info(f"Uploading to YouTube: {ctx.video_title!r}")
        request = client.videos().insert(part="snippet,status", body=video_body, media_body=media)

        video_id = _execute_resumable_upload(request)
        log.info(f"Uploaded: video_id={video_id}")

        # Persist video_id immediately so retries detect and skip re-uploading
        ctx.youtube_video_id = video_id
        ctx.db.record_upload(ctx.job_id, ctx.channel_id, video_id,
                             ctx.video_title or ctx.topic,
                             f"https://youtu.be/{video_id}")

        # Set thumbnail (non-fatal — requires YouTube channel to be verified)
        if ctx.thumbnail_path and ctx.thumbnail_path.exists():
            try:
                _set_thumbnail(client, video_id, ctx.thumbnail_path)
            except Exception as exc:
                log.warning(
                    f"Custom thumbnail upload skipped ({exc}). "
                    "To enable: YouTube Studio → Settings → Channel → "
                    "Feature eligibility → Custom thumbnail."
                )

        # Assign to playlist if configured
        playlist_id = ctx.resolved.get("playlist_id") or ""
        if playlist_id:
            _add_to_playlist(client, video_id, playlist_id)

        # Now publish (set to desired privacy)
        if privacy != "private":
            _update_privacy(client, video_id, privacy)

        url = f"https://youtu.be/{video_id}"
        log.info(f"Published ({privacy}): {url}")
        return ctx

    def _load_from_checkpoint(self, ctx: JobContext) -> JobContext:
        """If this job already has a completed upload, restore the video_id to skip re-uploading."""
        existing = ctx.db.get_upload_by_job(ctx.job_id)
        if existing:
            ctx.youtube_video_id = existing["youtube_video_id"]
            log.info(
                f"Checkpoint: video already uploaded ({ctx.youtube_video_id}) — skipping re-upload"
            )
        return ctx


def _execute_resumable_upload(request) -> str:
    response = None
    error = None
    retry = 0
    max_retries = 5

    while response is None:
        try:
            _status, response = request.next_chunk()
            if response is not None:
                return response["id"]
        except Exception as e:
            error = e
            if retry < max_retries:
                wait = 2 ** retry
                log.warning(f"Upload chunk error (retry {retry+1}/{max_retries}): {e}. Waiting {wait}s")
                time.sleep(wait)
                retry += 1
            else:
                raise RuntimeError(f"Upload failed after {max_retries} retries: {error}") from e


def _set_thumbnail(client, video_id: str, thumb_path: Path):
    from googleapiclient.http import MediaFileUpload
    media = MediaFileUpload(str(thumb_path), mimetype="image/jpeg")
    client.thumbnails().set(videoId=video_id, media_body=media).execute()
    log.info(f"Thumbnail set for {video_id}")


def _add_to_playlist(client, video_id: str, playlist_id: str):
    client.playlistItems().insert(
        part="snippet",
        body={"snippet": {
            "playlistId": playlist_id,
            "resourceId": {"kind": "youtube#video", "videoId": video_id},
        }},
    ).execute()
    log.info(f"Added {video_id} to playlist {playlist_id}")


def _update_privacy(client, video_id: str, privacy: str):
    client.videos().update(
        part="status",
        body={"id": video_id, "status": {"privacyStatus": privacy}},
    ).execute()
    log.info(f"Privacy updated to '{privacy}' for {video_id}")
