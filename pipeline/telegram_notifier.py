"""
TelegramNotifier — pipeline stage that runs after YouTubeUploader.

Sends a Telegram message with the video thumbnail, title, private URL,
and two inline buttons (✅ Publish / 🚫 Keep Private).

Non-blocking: does NOT wait for the user to tap a button.  The callback
poller (started independently in orchestrator.start_background) handles
the user's response asynchronously.

Graceful degradation:
  - If TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars are absent → skip silently
  - If the Telegram API call fails → log warning, continue pipeline
  - Either way the video is left as-is on YouTube (private)
"""
import logging

from pipeline.base import PipelineStage, JobContext

log = logging.getLogger(__name__)


class TelegramNotifier(PipelineStage):
    name = "telegram_notifier"

    def execute(self, ctx: JobContext) -> JobContext:
        from integrations.telegram_client import (
            load_credentials,
            send_review_notification,
        )

        # ── 1. Credentials ─────────────────────────────────────────────────────
        bot_token, chat_id = load_credentials(ctx.resolved)
        if not bot_token or not chat_id:
            log.info(
                "Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID "
                "env vars missing) — skipping review notification"
            )
            return ctx

        # ── 2. Require a valid video_id ────────────────────────────────────────
        video_id = ctx.youtube_video_id
        if not video_id:
            log.warning("TelegramNotifier: no youtube_video_id on ctx — skipping")
            return ctx

        url = f"https://youtu.be/{video_id}"

        # ── 3. Send notification ───────────────────────────────────────────────
        log.info(f"Sending Telegram review notification for {video_id!r} ({ctx.video_title!r})")
        message_id = send_review_notification(
            bot_token=bot_token,
            chat_id=chat_id,
            video_id=video_id,
            title=ctx.video_title or ctx.topic,
            url=url,
            thumbnail_path=ctx.thumbnail_path,
            niche=ctx.niche,
            job_id=ctx.job_id,
        )

        # ── 4. Record in DB for idempotency / dashboard visibility ─────────────
        try:
            ctx.db.add_pending_review(
                job_id=ctx.job_id,
                channel_id=ctx.channel_id,
                youtube_video_id=video_id,
                telegram_message_id=message_id,
                title=ctx.video_title or ctx.topic,
                url=url,
            )
        except Exception as exc:
            log.warning(f"TelegramNotifier: failed to record pending_review: {exc}")

        if message_id:
            log.info(f"Review notification sent (message_id={message_id}). "
                     "Waiting for user approval in Telegram.")
        else:
            log.warning("Telegram notification failed — video remains private until manually published")

        return ctx

    def _load_from_checkpoint(self, ctx: JobContext) -> JobContext:
        # Notifications are fire-and-forget; don't re-send on pipeline resume
        return ctx
