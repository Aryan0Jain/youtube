import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from mergedeep import merge


class ConfigValidationError(Exception):
    def __init__(self, yaml_file: str, field_path: str, message: str):
        super().__init__(f"[{yaml_file}] {field_path}: {message}")
        self.yaml_file = yaml_file
        self.field_path = field_path


@dataclass
class TopicSourceConfig:
    type: str                        # "google_sheet" | "claude_autogen" | "trend_autogen"
    sheet_id: str | None = None
    tab_name: str | None = None
    topic_column: str | None = None
    used_column: str | None = None
    autogen_prompt: str | None = None
    avoid_recent_topics: bool = False
    allow_topic_refresh: bool = False      # allow revisiting old topics with updated data
    refresh_after_days: int = 180          # how old a topic must be to qualify for refresh
    # trend_autogen fields
    opportunity_threshold: int = 5
    timeframe: str = "now 7-d"
    fallback_autogen_prompt: str | None = None


@dataclass
class ScheduleConfig:
    days_of_week: list[str]
    time_utc: str                    # "HH:MM"

    @property
    def hour(self) -> int:
        return int(self.time_utc.split(":")[0])

    @property
    def minute(self) -> int:
        return int(self.time_utc.split(":")[1])

    @property
    def apscheduler_days(self) -> str:
        abbr = {"monday": "mon", "tuesday": "tue", "wednesday": "wed",
                "thursday": "thu", "friday": "fri", "saturday": "sat", "sunday": "sun"}
        return ",".join(abbr[d.lower()] for d in self.days_of_week)


@dataclass
class SeriesConfig:
    series_id: str
    display_name: str
    format: str                      # "full_length" | "shorts"
    niche: str
    style_notes: str
    schedule: ScheduleConfig
    topic_source: TopicSourceConfig
    playlist_id: str | None = None
    resolved: dict = field(default_factory=dict)  # merged final params


@dataclass
class ChannelConfig:
    channel_id: str
    display_name: str
    youtube_channel_id: str
    oauth_token_path: str
    series: list[SeriesConfig]


def _deep_merge(*dicts: dict) -> dict:
    result: dict = {}
    for d in dicts:
        merge(result, d)
    return result


def _require(data: dict, key: str, source_file: str, path: str) -> Any:
    if key not in data or data[key] is None:
        raise ConfigValidationError(source_file, f"{path}.{key}", "is required but missing")
    return data[key]


def _load_topic_source(raw: dict, source_file: str, ctx: str) -> TopicSourceConfig:
    ts_type = _require(raw, "type", source_file, ctx)
    if ts_type not in ("google_sheet", "claude_autogen", "trend_autogen"):
        raise ConfigValidationError(source_file, f"{ctx}.type",
                                    f"must be 'google_sheet', 'claude_autogen', or 'trend_autogen', got '{ts_type}'")

    if ts_type == "google_sheet":
        return TopicSourceConfig(
            type=ts_type,
            sheet_id=_require(raw, "sheet_id", source_file, ctx),
            tab_name=_require(raw, "tab_name", source_file, ctx),
            topic_column=raw.get("topic_column", "A"),
            used_column=raw.get("used_column", "B"),
        )

    if ts_type == "claude_autogen":
        return TopicSourceConfig(
            type=ts_type,
            autogen_prompt=_require(raw, "autogen_prompt", source_file, ctx),
            avoid_recent_topics=raw.get("avoid_recent_topics", False),
            allow_topic_refresh=raw.get("allow_topic_refresh", False),
            refresh_after_days=int(raw.get("refresh_after_days", 180)),
        )

    # trend_autogen
    return TopicSourceConfig(
        type=ts_type,
        opportunity_threshold=int(raw.get("opportunity_threshold", 5)),
        timeframe=raw.get("timeframe", "now 7-d"),
        fallback_autogen_prompt=raw.get("fallback_autogen_prompt"),
        avoid_recent_topics=raw.get("avoid_recent_topics", True),
    )


def _load_schedule(raw: dict, source_file: str, ctx: str) -> ScheduleConfig:
    days = _require(raw, "days_of_week", source_file, ctx)
    time_utc = _require(raw, "time_utc", source_file, ctx)
    valid_days = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    for d in days:
        if d.lower() not in valid_days:
            raise ConfigValidationError(source_file, f"{ctx}.days_of_week", f"invalid day '{d}'")
    if ":" not in str(time_utc):
        raise ConfigValidationError(source_file, f"{ctx}.time_utc", "must be HH:MM format")
    return ScheduleConfig(days_of_week=[d.lower() for d in days], time_utc=str(time_utc))


def _load_series(raw: dict, source_file: str, index: int, merged_defaults: dict) -> SeriesConfig:
    ctx = f"series[{index}]"
    series_id = _require(raw, "series_id", source_file, ctx)
    fmt = _require(raw, "format", source_file, ctx)
    if fmt not in ("full_length", "shorts"):
        raise ConfigValidationError(source_file, f"{ctx}.format",
                                    f"must be 'full_length' or 'shorts', got '{fmt}'")

    series_overrides = raw.get("overrides", {}) or {}
    resolved = _deep_merge(merged_defaults, series_overrides)
    resolved["oauth_token_path"] = merged_defaults.get("oauth_token_path", "")
    resolved["niche"] = raw.get("niche", "general")

    if raw.get("playlist_id"):
        resolved["playlist_id"] = raw["playlist_id"]

    return SeriesConfig(
        series_id=series_id,
        display_name=raw.get("display_name", series_id),
        format=fmt,
        niche=raw.get("niche", "general"),
        style_notes=raw.get("style_notes", ""),
        schedule=_load_schedule(_require(raw, "schedule", source_file, ctx), source_file, ctx),
        topic_source=_load_topic_source(_require(raw, "topic_source", source_file, ctx), source_file, ctx),
        playlist_id=raw.get("playlist_id"),
        resolved=resolved,
    )


def load_all(base_dir: Path | None = None) -> dict[str, ChannelConfig]:
    if base_dir is None:
        base_dir = Path(__file__).parent.parent

    master_path = base_dir / "config" / "master.yaml"
    with open(master_path, encoding="utf-8") as f:
        master = yaml.safe_load(f)

    global_defaults: dict = master.get("defaults", {})
    infra: dict = master.get("infrastructure", {})
    claude_cfg: dict = master.get("claude", {})
    pexels_cfg: dict = master.get("pexels", {})
    telegram_cfg: dict = master.get("telegram", {})

    channels: dict[str, ChannelConfig] = {}
    channels_dir = base_dir / "config" / "channels"

    for yaml_path in sorted(channels_dir.glob("*.yaml")):
        if yaml_path.name.startswith("_"):
            continue

        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        source_file = yaml_path.name
        channel_id = _require(raw, "channel_id", source_file, "root")

        if yaml_path.stem != channel_id:
            raise ConfigValidationError(
                source_file, "channel_id",
                f"must match filename stem '{yaml_path.stem}', got '{channel_id}'"
            )

        channel_overrides = raw.get("overrides", {}) or {}
        channel_defaults = _deep_merge(global_defaults, channel_overrides)
        channel_defaults["oauth_token_path"] = _require(raw, "oauth_token_path", source_file, "root")
        channel_defaults["claude"] = claude_cfg
        channel_defaults["pexels"] = pexels_cfg
        channel_defaults["infrastructure"] = infra
        channel_defaults["telegram"] = telegram_cfg

        raw_series = raw.get("series", [])
        if not raw_series:
            raise ConfigValidationError(source_file, "series", "must have at least one series")

        series_list = [
            _load_series(s, source_file, i, channel_defaults)
            for i, s in enumerate(raw_series)
        ]

        channels[channel_id] = ChannelConfig(
            channel_id=channel_id,
            display_name=_require(raw, "display_name", source_file, "root"),
            youtube_channel_id=_require(raw, "youtube_channel_id", source_file, "root"),
            oauth_token_path=channel_defaults["oauth_token_path"],
            series=series_list,
        )

    return channels


def load_master_infra(base_dir: Path | None = None) -> dict:
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    master_path = base_dir / "config" / "master.yaml"
    with open(master_path, encoding="utf-8") as f:
        master = yaml.safe_load(f)
    return master.get("infrastructure", {})


def load_dashboard_config(base_dir: Path | None = None) -> dict:
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    master_path = base_dir / "config" / "master.yaml"
    with open(master_path, encoding="utf-8") as f:
        master = yaml.safe_load(f)
    return master.get("dashboard", {"enabled": True, "port": 8080, "host": "127.0.0.1"})
