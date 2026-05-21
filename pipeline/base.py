import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from formats.base import FormatSpec
from core.state_db import StateDB

log = logging.getLogger(__name__)


@dataclass
class JobContext:
    job_id: int
    channel_id: str
    series_id: str
    topic: str
    format: str               # "full_length" | "shorts"
    niche: str
    style_notes: str
    resolved: dict            # deep-merged config: tts_voice, music_volume, etc.
    format_spec: FormatSpec
    workspace: Path
    db: StateDB

    # Populated progressively by each stage:
    script_text: str | None = None
    audio_path: Path | None = None
    clip_paths: list[Path] = field(default_factory=list)
    video_path: Path | None = None
    thumbnail_path: Path | None = None
    youtube_video_id: str | None = None

    # Subtitles (filled by subtitle_generator)
    subtitle_path: Path | None = None

    # Niche overlay metadata (filled by script_writer via extract_niche_metadata)
    # Schema: {"overlay_type": str, "items": list|dict} — empty dict = no overlays
    niche_metadata: dict = field(default_factory=dict)

    # Emotional keywords for bold subtitle emphasis (filled by script_writer)
    # List of lowercase single words, e.g. ["collapsed", "drowned", "deadliest"]
    emotional_keywords: list[str] = field(default_factory=list)

    # Generated metadata (filled by script_writer for youtube_uploader)
    video_title: str = ""
    video_description: str = ""
    video_tags: list[str] = field(default_factory=list)


class PipelineStageError(Exception):
    def __init__(self, stage_name: str, original: Exception):
        super().__init__(f"[{stage_name}] {original}")
        self.stage_name = stage_name
        self.original = original


class PipelineStage(ABC):
    name: str  # must be set by subclass

    def run(self, ctx: JobContext) -> JobContext:
        checkpoint = ctx.db.get_checkpoint(ctx.job_id)
        if checkpoint == self.name:
            log.info(f"[{ctx.channel_id}/{ctx.series_id}] job#{ctx.job_id} SKIP {self.name} (already done)")
            return self._load_from_checkpoint(ctx)

        log.info(f"[{ctx.channel_id}/{ctx.series_id}] job#{ctx.job_id} START {self.name}")
        try:
            ctx = self.execute(ctx)
        except Exception as exc:
            raise PipelineStageError(self.name, exc) from exc

        ctx.db.set_checkpoint(ctx.job_id, self.name)
        log.info(f"[{ctx.channel_id}/{ctx.series_id}] job#{ctx.job_id} DONE  {self.name}")
        return ctx

    def _load_from_checkpoint(self, ctx: JobContext) -> JobContext:
        """Override in stages that persist output to disk and need to reload it on skip."""
        return ctx

    @abstractmethod
    def execute(self, ctx: JobContext) -> JobContext: ...
