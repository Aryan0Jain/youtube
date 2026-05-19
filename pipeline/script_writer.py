import os
import logging
from pathlib import Path

from pipeline.base import PipelineStage, JobContext
from integrations import claude_client

log = logging.getLogger(__name__)

USE_MOCK = os.environ.get("USE_MOCK_CLAUDE") == "1"


class ScriptWriter(PipelineStage):
    name = "script_writer"

    def execute(self, ctx: JobContext) -> JobContext:
        claude_cfg = ctx.resolved.get("claude", {})
        model = claude_cfg.get("model", "claude-opus-4-5")
        haiku_model = claude_cfg.get("haiku_model", "claude-haiku-4-5-20251001")
        max_tokens = claude_cfg.get("max_tokens", 4096)
        spec = ctx.format_spec

        if USE_MOCK:
            script_text = _mock_script(ctx.topic, spec.target_word_count)
        else:
            script_text = claude_client.write_script(
                topic=ctx.topic,
                style_notes=ctx.style_notes,
                niche=ctx.niche,
                script_prompt_suffix=spec.script_prompt_suffix,
                target_word_count=spec.target_word_count,
                model=model,
                max_tokens=max_tokens,
            )

        script_path = ctx.workspace / "script.txt"
        script_path.write_text(script_text, encoding="utf-8")
        ctx.script_text = script_text
        word_count = len(script_text.split())
        log.info(f"Script: {word_count} words → {script_path}")

        # Generate SEO metadata (cheap Haiku call)
        if USE_MOCK:
            ctx.video_title = f"{ctx.topic}{ctx.format_spec.title_suffix}"
            ctx.video_description = f"A video about {ctx.topic}."
            ctx.video_tags = [ctx.niche, "youtube", "viral"]
        else:
            meta = claude_client.generate_video_metadata(
                topic=ctx.topic,
                niche=ctx.niche,
                script_text=script_text,
                haiku_model=haiku_model,
            )
            ctx.video_title = meta["title"] + ctx.format_spec.title_suffix
            ctx.video_description = meta["description"]
            ctx.video_tags = meta["tags"]

        log.info(f"Title: {ctx.video_title!r}")

        # Extract niche overlay metadata (cheap Haiku call, ~$0.002)
        # Non-critical: failures are logged and silently skipped
        if not USE_MOCK:
            ctx.niche_metadata = claude_client.extract_niche_metadata(
                script_text=script_text,
                niche=ctx.niche,
                haiku_model=haiku_model,
            )
            if ctx.niche_metadata:
                overlay_type = ctx.niche_metadata.get("overlay_type", "none")
                items = ctx.niche_metadata.get("items", [])
                count = len(items) if isinstance(items, list) else 1
                log.info(f"Niche metadata: overlay={overlay_type}, items={count}")
            else:
                log.info("Niche metadata: none extracted (overlay skipped)")

        return ctx

    def _load_from_checkpoint(self, ctx: JobContext) -> JobContext:
        script_path = ctx.workspace / "script.txt"
        if script_path.exists():
            ctx.script_text = script_path.read_text(encoding="utf-8")
        return ctx


def _mock_script(topic: str, word_count: int) -> str:
    sentence = f"This is a test script about {topic}. "
    words_per_sentence = len(sentence.split())
    repetitions = (word_count // words_per_sentence) + 1
    text = sentence * repetitions
    return " ".join(text.split()[:word_count])
