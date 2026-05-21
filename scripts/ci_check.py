"""
CI sanity checks — runs in GitHub Actions (no secrets, no real API calls).

Checks:
  1. All pipeline modules import cleanly
  2. All channel YAML configs are valid
  3. StateDB CRUD (using :memory: — covers the persistent-connection fix)
  4. Topic source config fields parse correctly
  5. No hardcoded secrets in tracked source files

Exit 0 on pass, 1 on any failure.
"""
import sys
import io
import os
import re
import traceback
from pathlib import Path

# Windows console UTF-8 fix (GitHub Actions runs on Linux — no issue there)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Run from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures: list[str] = []


def check(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
    except Exception as exc:
        print(f"  {FAIL}  {name}")
        traceback.print_exc()
        failures.append(name)


# ── 1. Module imports ──────────────────────────────────────────────────────────

print("\n── Module imports ──────────────────────────────────────────────────")

MODULES = [
    ("pipeline.base",               "PipelineStage"),
    ("pipeline.script_writer",      "ScriptWriter"),
    ("pipeline.tts_generator",      "TTSGenerator"),
    ("pipeline.subtitle_generator", "SubtitleGenerator"),
    ("pipeline.clip_fetcher",       "ClipFetcher"),
    ("pipeline.video_assembler",    "assemble_video"),
    ("pipeline.thumbnail_maker",    "ThumbnailMaker"),
    ("pipeline.youtube_uploader",   "YouTubeUploader"),
    ("pipeline.telegram_notifier",  "TelegramNotifier"),
    ("integrations.claude_client",  "generate_video_metadata"),
    ("integrations.telegram_client","send_review_notification"),
    ("core.state_db",               "StateDB"),
    ("core.orchestrator",           "STAGES"),
    ("core.scheduler",              "start"),
    ("core.config_loader",          "load_all"),
    ("formats",                     "get_format_spec"),
]

for mod, attr in MODULES:
    def _import(m=mod, a=attr):
        obj = __import__(m, fromlist=[a])
        assert hasattr(obj, a), f"{m} missing {a}"
    check(f"import {mod}.{attr}", _import)


# ── 2. Channel config validation ──────────────────────────────────────────────

print("\n── Channel configs ─────────────────────────────────────────────────")

def _load_channels():
    from core.config_loader import load_all
    channels = load_all(ROOT)
    assert channels, "No channels loaded"
    for ch_id, ch in channels.items():
        assert ch.youtube_channel_id, f"{ch_id}: missing youtube_channel_id"
        assert ch.series, f"{ch_id}: no series"
        for s in ch.series:
            assert s.schedule.days_of_week, f"{ch_id}/{s.series_id}: no schedule days"
    return channels

check("all channel YAMLs parse", _load_channels)


# ── 3. Stage ordering ─────────────────────────────────────────────────────────

print("\n── Pipeline stage ordering ─────────────────────────────────────────")

def _stage_order():
    from core.orchestrator import STAGES
    names = [s.name for s in STAGES]
    assert "youtube_uploader" in names
    assert "telegram_notifier" in names
    assert names.index("youtube_uploader") < names.index("telegram_notifier"), \
        "telegram_notifier must come after youtube_uploader"
    assert names[-1] == "telegram_notifier", "telegram_notifier must be last stage"

check("stage order (telegram after youtube)", _stage_order)


# ── 4. StateDB :memory: CRUD ──────────────────────────────────────────────────

print("\n── StateDB :memory: ────────────────────────────────────────────────")

def _db_jobs():
    from core.state_db import StateDB
    db = StateDB(":memory:")
    jid = db.create_job("ch", "series", "Test Topic", "full_length")
    assert jid > 0
    job = db.get_job(jid)
    assert job["topic"] == "Test Topic"
    db.mark_running(jid, "/tmp/ws")
    db.mark_completed(jid, "yt_vid_123")
    assert db.get_job(jid)["status"] == "completed"

check("jobs CRUD", _db_jobs)

def _db_reviews():
    from core.state_db import StateDB
    db = StateDB(":memory:")
    db.add_pending_review(1, "ch", "vid1", 42, "Title", "https://youtu.be/vid1")
    rows = db.get_pending_reviews()
    assert len(rows) == 1 and rows[0]["youtube_video_id"] == "vid1"
    db.update_review_status("vid1", "published")
    assert db.get_pending_review("vid1")["status"] == "published"

check("pending_reviews CRUD", _db_reviews)

def _db_topics():
    from core.state_db import StateDB
    db = StateDB(":memory:")
    db.record_used_topic("ch", "s", "Topic A", None)
    recent = db.get_recent_used_topics("ch", "s", days=90)
    assert "Topic A" in recent
    old = db.get_old_used_topics("ch", "s", older_than_days=1)
    assert old == []  # just inserted — not old yet

check("topic dedup + refresh query", _db_topics)


# ── 5. TopicSourceConfig refresh fields ───────────────────────────────────────

print("\n── Config dataclass fields ─────────────────────────────────────────")

def _topic_source_fields():
    from core.config_loader import TopicSourceConfig
    ts = TopicSourceConfig(
        type="claude_autogen",
        autogen_prompt="test",
        allow_topic_refresh=True,
        refresh_after_days=180,
    )
    assert ts.allow_topic_refresh is True
    assert ts.refresh_after_days == 180

check("TopicSourceConfig refresh fields", _topic_source_fields)


# ── 6. Secret scan ────────────────────────────────────────────────────────────

print("\n── Secret scan ─────────────────────────────────────────────────────")

SECRET_RE = re.compile(
    r"(sk-ant-api|sk_e4[0-9a-f]{40}|AIzaSy[A-Za-z0-9_-]{33}"
    r"|[0-9]{10}:AA[A-Za-z0-9_-]{33})",  # Telegram bot token pattern
)
SCAN_DIRS = ["pipeline", "integrations", "core", "dashboard", "formats", "scripts"]
SKIP_FILES = {"ci_check.py"}  # this file mentions patterns for detection

def _secret_scan():
    hits = []
    for d in SCAN_DIRS:
        for py in (ROOT / d).rglob("*.py"):
            if py.name in SKIP_FILES:
                continue
            text = py.read_text(encoding="utf-8", errors="ignore")
            if SECRET_RE.search(text):
                hits.append(str(py.relative_to(ROOT)))
    assert not hits, f"Possible secrets in: {hits}"

check("no hardcoded secrets in source", _secret_scan)


# ── Summary ───────────────────────────────────────────────────────────────────

print()
total = len(MODULES) + 8  # module imports + other checks
if failures:
    print(f"❌  {len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
else:
    print(f"✅  All checks passed.")
    sys.exit(0)
