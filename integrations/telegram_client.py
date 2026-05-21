"""
telegram_client.py — Telegram Bot API integration for video review notifications.

Flow:
  1. After a video is uploaded as private, TelegramNotifier calls
     send_review_notification() which sends a photo message with two buttons:
       ✅ Publish  |  🚫 Keep Private

  2. start_callback_poller() launches a daemon thread that long-polls
     Telegram's getUpdates endpoint.  When the user taps a button:
       - answerCallbackQuery dismisses the loading spinner
       - YouTube privacy is updated via the uploader's _update_privacy helper
       - The message is edited to remove the buttons and show the outcome
       - pending_reviews table in the DB is updated

All Telegram calls use plain requests (no python-telegram-bot dependency).
Graceful on every error: logs warnings, never raises to the caller.
"""
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
_POLL_TIMEOUT = 25   # long-poll wait seconds (< 30 to avoid proxy timeouts)
_RETRY_SLEEP  = 10   # seconds to wait after a network/API error before retrying


# ── Low-level API helper ──────────────────────────────────────────────────────

def _tg(bot_token: str, method: str, **payload) -> dict | None:
    """
    POST to the Telegram Bot API.  Returns the parsed response dict on success,
    or None on any error (network, non-200, Telegram error flag).
    """
    url = _TELEGRAM_API.format(token=bot_token, method=method)
    try:
        resp = requests.post(url, json=payload, timeout=_POLL_TIMEOUT + 5)
        data = resp.json()
        if not data.get("ok"):
            log.warning(f"Telegram [{method}] not ok: {data.get('description')}")
            return None
        return data
    except requests.RequestException as exc:
        log.warning(f"Telegram [{method}] network error: {exc}")
        return None
    except Exception as exc:
        log.warning(f"Telegram [{method}] unexpected error: {exc}")
        return None


# ── Notification sender ───────────────────────────────────────────────────────

def send_review_notification(
    bot_token: str,
    chat_id: str,
    video_id: str,
    title: str,
    url: str,
    thumbnail_path: Path | None,
    niche: str,
    job_id: int,
) -> int | None:
    """
    Send a Telegram review notification for a newly uploaded (private) video.

    Sends a photo message (thumbnail) with a caption and two inline buttons:
      ✅ Publish   → sets YouTube video to public
      🚫 Keep Private → leaves as private

    Returns the Telegram message_id on success, or None on failure.
    """
    caption = (
        f"🎬 *New video ready for review*\n\n"
        f"*{_escape_md(title)}*\n"
        f"Niche: `{niche}` · Job: `#{job_id}`\n\n"
        f"🔗 [Watch on YouTube \\(private\\)]({url})\n\n"
        f"_Tap a button below to publish or keep private\\._"
    )

    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ Publish",       "callback_data": f"publish:{video_id}"},
            {"text": "🚫 Keep Private",  "callback_data": f"reject:{video_id}"},
        ]]
    }

    # Prefer sending with the thumbnail; fall back to text-only message
    if thumbnail_path and thumbnail_path.exists():
        try:
            with open(thumbnail_path, "rb") as img:
                url_endpoint = _TELEGRAM_API.format(token=bot_token, method="sendPhoto")
                resp = requests.post(
                    url_endpoint,
                    data={
                        "chat_id": chat_id,
                        "caption": caption,
                        "parse_mode": "MarkdownV2",
                        "reply_markup": json.dumps(keyboard),
                    },
                    files={"photo": img},
                    timeout=30,
                )
                data = resp.json()
                if data.get("ok"):
                    msg_id = data["result"]["message_id"]
                    log.info(f"Telegram review notification sent: message_id={msg_id}")
                    return msg_id
                log.warning(f"Telegram sendPhoto failed: {data.get('description')}")
        except Exception as exc:
            log.warning(f"Telegram sendPhoto error: {exc}")

    # Fallback: plain text message (no photo)
    data = _tg(bot_token, "sendMessage",
                chat_id=chat_id,
                text=caption,
                parse_mode="MarkdownV2",
                reply_markup=keyboard,
                disable_web_page_preview=False)
    if data:
        msg_id = data["result"]["message_id"]
        log.info(f"Telegram review notification sent (text fallback): message_id={msg_id}")
        return msg_id

    log.error("Failed to send Telegram review notification (both photo and text failed)")
    return None


# ── Callback poller ───────────────────────────────────────────────────────────

def start_callback_poller(bot_token: str, db, base_dir: Path) -> threading.Thread:
    """
    Start a daemon thread that long-polls Telegram for button-tap callbacks
    and handles publish / reject actions.

    db   — StateDB instance (already open)
    base_dir — project root, used to resolve OAuth token paths
    """
    thread = threading.Thread(
        target=_poll_loop,
        args=(bot_token, db, base_dir),
        daemon=True,
        name="telegram-callback-poller",
    )
    thread.start()
    log.info("Telegram callback poller started")
    return thread


def _poll_loop(bot_token: str, db, base_dir: Path):
    """Main long-poll loop — runs forever in its daemon thread."""
    offset = 0  # Telegram update_id offset; tracks which updates we've processed

    while True:
        try:
            data = _tg(bot_token, "getUpdates",
                       offset=offset,
                       timeout=_POLL_TIMEOUT,
                       allowed_updates=["callback_query"])
            if data is None:
                time.sleep(_RETRY_SLEEP)
                continue

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                cq = update.get("callback_query")
                if cq:
                    _handle_callback(bot_token, cq, db, base_dir)

        except Exception as exc:
            log.warning(f"Telegram poll loop error: {exc}. Retrying in {_RETRY_SLEEP}s")
            time.sleep(_RETRY_SLEEP)


def _handle_callback(bot_token: str, cq: dict, db, base_dir: Path):
    """
    Process a single callback_query (button tap).

    callback_data format:  "publish:<video_id>"  or  "reject:<video_id>"
    """
    cq_id     = cq["id"]
    chat_id   = str(cq["message"]["chat"]["id"])
    message_id = cq["message"]["message_id"]
    data      = cq.get("data", "")

    # Always acknowledge the callback immediately to dismiss the spinner
    _tg(bot_token, "answerCallbackQuery", callback_query_id=cq_id)

    parts = data.split(":", 1)
    if len(parts) != 2 or parts[0] not in ("publish", "reject"):
        log.warning(f"Unexpected callback_data: {data!r}")
        return

    action, video_id = parts

    # Idempotency: skip if already reviewed
    review = db.get_pending_review(video_id)
    if review is None:
        log.warning(f"callback for unknown video_id {video_id!r} — ignoring")
        _tg(bot_token, "answerCallbackQuery",
            callback_query_id=cq_id,
            text="Video not found in review queue.")
        return

    if review["status"] != "pending":
        log.info(f"Video {video_id} already {review['status']}, ignoring duplicate tap")
        _tg(bot_token, "editMessageReplyMarkup",
            chat_id=chat_id, message_id=message_id,
            reply_markup={"inline_keyboard": []})
        return

    if action == "publish":
        success = _publish_video(video_id, review["channel_id"], base_dir)
        if success:
            db.update_review_status(video_id, "published")
            new_caption = f"✅ *Published!*\n\n_{_escape_md(review.get('title', video_id))}_"
            log.info(f"Video {video_id} published via Telegram")
        else:
            new_caption = f"❌ *Publish failed* — check logs\\.\n\n_{_escape_md(review.get('title', video_id))}_"
            log.error(f"Failed to publish video {video_id} via YouTube API")
    else:  # reject
        db.update_review_status(video_id, "rejected")
        new_caption = f"🚫 *Kept private*\n\n_{_escape_md(review.get('title', video_id))}_"
        log.info(f"Video {video_id} kept private via Telegram")

    # Edit original message: update caption and remove buttons
    _tg(bot_token, "editMessageCaption",
        chat_id=chat_id, message_id=message_id,
        caption=new_caption, parse_mode="MarkdownV2",
        reply_markup={"inline_keyboard": []})


def _publish_video(video_id: str, channel_id: str, base_dir: Path) -> bool:
    """
    Set the YouTube video's privacy to 'public'.
    Returns True on success, False on any error.
    """
    try:
        from core.config_loader import load_all
        from integrations.youtube_client import get_client
        from pipeline.youtube_uploader import _update_privacy

        channels = load_all(base_dir)
        channel = channels.get(channel_id)
        if not channel:
            log.error(f"Channel '{channel_id}' not found — cannot publish video")
            return False

        client = get_client(channel.oauth_token_path)
        _update_privacy(client, video_id, "public")
        return True
    except Exception as exc:
        log.error(f"_publish_video({video_id}): {exc}")
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _escape_md(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    special = r"\_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special else c for c in str(text))


def load_credentials(resolved: dict) -> tuple[str, str]:
    """
    Read bot_token and chat_id from env vars named in ctx.resolved["telegram"].
    Returns (bot_token, chat_id) — either may be "" if not configured.
    """
    tg_cfg = resolved.get("telegram", {})
    bot_token = os.environ.get(tg_cfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN"), "")
    chat_id   = os.environ.get(tg_cfg.get("chat_id_env",   "TELEGRAM_CHAT_ID"),   "")
    return bot_token, chat_id
