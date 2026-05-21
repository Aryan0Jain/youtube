import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pipeline.base import PipelineStage, JobContext

log = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).parent.parent / "assets"

# ---------------------------------------------------------------------------
# Imagen 4 background generation
# ---------------------------------------------------------------------------

# Per-niche cinematic prompt templates.
# {subject} is replaced with a cleaned version of the video topic.
_NICHE_IMAGEN_PROMPTS: dict[str, str] = {
    "ranking": (
        "cinematic ultra-wide shot depicting {subject}, massive scale and dramatic impact, "
        "epic atmosphere, golden hour lighting, dust clouds, debris, crowd of people for scale, "
        "photorealistic, 4K, sharp focus, high contrast, no text, no words, no captions"
    ),
    "horror": (
        "dark atmospheric horror scene of {subject}, deep shadows, eerie cold lighting, "
        "abandoned location, fog, sense of dread and mystery, cinematic thriller aesthetic, "
        "no text, no words, no captions"
    ),
    "shock_facts": (
        "stunning dramatic visualization of {subject}, awe-inspiring scale, "
        "vibrant colors, scientific accuracy, photorealistic, dramatic studio lighting, "
        "no text, no words, no captions"
    ),
    "quiz": (
        "vibrant colorful knowledge concept for {subject}, world maps, icons, geography, "
        "bright engaging colors, clean composition, no text, no words, no captions"
    ),
    "historical_versus": (
        "epic historical scene depicting {subject}, dramatic confrontation between two forces, "
        "cinematic lighting, period-accurate details, oil-painting realism, "
        "no text, no words, no captions"
    ),
    "myth_busting": (
        "dramatic investigative reveal scene about {subject}, split between illusion and reality, "
        "high contrast lighting, cracked surface revealing truth beneath, "
        "documentary cinematic style, no text, no words, no captions"
    ),
    "what_if": (
        "surreal sci-fi concept visualization of {subject}, alternate reality sky, "
        "futuristic atmospheric elements, photorealistic CGI quality, cinematic scope, "
        "no text, no words, no captions"
    ),
}

_DEFAULT_IMAGEN_PROMPT = (
    "cinematic dramatic scene depicting {subject}, high impact visuals, "
    "professional photography, dramatic lighting, no text, no words, no captions"
)

# Prefixes to strip from topic strings before building the visual prompt
_TOPIC_STRIP_RE = re.compile(
    r"^(top\s+\d+|what\s+if|the\s+truth\s+about|debunking|"
    r"history\s+of|facts\s+about)\s*[:\-–]?\s*",
    re.IGNORECASE,
)


def _topic_to_subject(topic: str) -> str:
    """Strip meta-prefixes ('Top 10', 'What If') to get a visual noun phrase."""
    return _TOPIC_STRIP_RE.sub("", topic).strip() or topic


def _generate_imagen_background(topic: str, niche: str,
                                 workspace: Path, W: int, H: int) -> Path | None:
    """
    Generate a cinematic background image with Imagen 4 (Fast).

    Returns the saved JPEG path on success, or None on any error
    (caller falls back to video-frame extraction).

    Uses GEMINI_API_KEY from environment (already present in .env).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        log.debug("GEMINI_API_KEY not set — skipping Imagen background")
        return None

    subject = _topic_to_subject(topic)
    template = _NICHE_IMAGEN_PROMPTS.get(niche, _DEFAULT_IMAGEN_PROMPT)
    prompt = template.format(subject=subject)

    try:
        import google.genai as genai
        from PIL import Image
        import io

        client = genai.Client(api_key=api_key)
        log.info(f"Generating Imagen 4 background for: {subject!r}")
        response = client.models.generate_images(
            model="imagen-4.0-fast-generate-001",
            prompt=prompt,
            config=genai.types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9",
            ),
        )
        raw_bytes = response.generated_images[0].image.image_bytes
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize((W, H))
        out = workspace / "thumb_bg.jpg"
        img.save(str(out), "JPEG", quality=92)
        log.info(f"Imagen 4 background saved: {out} ({len(raw_bytes)//1024}KB)")
        return out

    except Exception as exc:
        log.warning(f"Imagen 4 generation failed ({exc}) — falling back to video frame")


@dataclass(frozen=True)
class ThumbnailStyle:
    bg_color: tuple[int, int, int]
    title_color: tuple[int, int, int]
    accent_color: tuple[int, int, int]
    gradient_bottom: tuple[int, int, int] | None = None


NICHE_STYLES: dict[str, ThumbnailStyle] = {
    "horror":      ThumbnailStyle((8, 0, 0),     (180, 0, 0),   (60, 0, 0),    (0, 0, 0)),
    "what_if":     ThumbnailStyle((5, 10, 40),   (255, 215, 0), (0, 100, 200), (0, 0, 0)),
    "historical_versus": ThumbnailStyle((20, 10, 5),  (210, 170, 100), (160, 120, 60), (0, 0, 0)),
    "quiz":        ThumbnailStyle((80, 0, 180),  (255, 230, 0), (0, 180, 200), (0, 80, 120)),
    "shock_facts": ThumbnailStyle((0, 0, 0),     (255, 230, 0), (255, 80, 0),  None),
    "ranking":     ThumbnailStyle((20, 20, 25),  (255, 215, 0), (200, 160, 40), None),
    "myth_busting":ThumbnailStyle((15, 10, 20),  (255, 255, 255), (220, 0, 0), (0, 0, 0)),
}

DEFAULT_STYLE = ThumbnailStyle((20, 20, 30), (255, 255, 255), (100, 180, 255), (0, 0, 0))


def _extract_frame(video_path: Path, output_path: Path, timestamp: float = 3.0):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
         "-frames:v", "1", "-q:v", "2", str(output_path)],
        capture_output=True, check=False,
    )


def _generate_thumbnail(topic: str, niche: str, spec, video_path: Path | None,
                        workspace: Path) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    style = NICHE_STYLES.get(niche, DEFAULT_STYLE)
    W, H = spec.thumb_width, spec.thumb_height
    output_path = workspace / "thumbnail.jpg"

    # Base image priority:
    #   1. Imagen 4 generated cinematic background (free, best CTR)
    #   2. Frame extracted from the assembled video
    #   3. Solid colour gradient (last resort)
    img = None

    imagen_path = _generate_imagen_background(topic, niche, workspace, W, H)
    if imagen_path and imagen_path.exists():
        img = Image.open(imagen_path).convert("RGB")
        log.info("Thumbnail base: Imagen 4 generated background")

    if img is None and video_path and video_path.exists():
        frame_path = workspace / "thumb_frame.jpg"
        _extract_frame(video_path, frame_path)
        if frame_path.exists() and frame_path.stat().st_size > 0:
            img = Image.open(frame_path).convert("RGB").resize((W, H))
            log.info("Thumbnail base: video frame fallback")

    if img is None:
        img = _gradient_background(W, H, style)
        log.info("Thumbnail base: gradient fallback")

    draw = ImageDraw.Draw(img)

    # Dark overlay (bottom 40%)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    for i in range(H):
        fraction = i / H
        if fraction > 0.5:
            alpha = int(180 * ((fraction - 0.5) / 0.5))
            ov_draw.line([(0, i), (W, i)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Niche badge (top-left colored bar)
    badge_label = niche.upper().replace("_", " ")
    _draw_badge(draw, badge_label, style.accent_color, W, H)

    # Title text
    font = _load_font(72 if spec.name == "full_length" else 88)
    title = topic[:70]
    _draw_wrapped_text(draw, title, font, style.title_color, W, H)

    # Watermark text for specific niches
    if niche == "quiz":
        wm_font = _load_font(400)
        _draw_shadow_text(draw, "?", wm_font, (255, 255, 255, 25), W // 2 - 120, H // 2 - 220)

    if niche == "myth_busting":
        badge_font = _load_font(60)
        _draw_shadow_text(draw, "MYTH BUSTED", badge_font, (255, 50, 50), 30, H - 100)

    img.save(str(output_path), "JPEG", quality=92)
    log.info(f"Thumbnail: {W}×{H} → {output_path}")
    return output_path


def _gradient_background(W: int, H: int, style: ThumbnailStyle) -> "Image":
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    top = style.bg_color
    bot = style.gradient_bottom or style.bg_color
    for y in range(H):
        ratio = y / H
        r = int(top[0] * (1 - ratio) + bot[0] * ratio)
        g = int(top[1] * (1 - ratio) + bot[1] * ratio)
        b = int(top[2] * (1 - ratio) + bot[2] * ratio)
        draw.line([(0, y), (W, y)], fill=(r, g, b))
    return img


def _load_font(size: int):
    from PIL import ImageFont
    font_path = ASSETS_DIR / "font_bold.ttf"
    if font_path.exists():
        try:
            return ImageFont.truetype(str(font_path), size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_badge(draw, label: str, color: tuple, W: int, H: int):
    from PIL import ImageFont
    font = _load_font(28)
    padding = 12
    bbox = draw.textbbox((0, 0), label, font=font)
    bw = bbox[2] - bbox[0] + padding * 2
    bh = bbox[3] - bbox[1] + padding * 2
    draw.rectangle([20, 20, 20 + bw, 20 + bh], fill=color)
    draw.text((20 + padding, 20 + padding), label, font=font, fill=(255, 255, 255))


def _draw_wrapped_text(draw, text: str, font, color: tuple, W: int, H: int):
    max_width = int(W * 0.9)
    lines = _wrap_text(draw, text, font, max_width)
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + 8
    total_height = len(lines) * line_height
    y = H - total_height - 40

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        x = (W - lw) // 2
        # Shadow
        draw.text((x + 3, y + 3), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=color)
        y += line_height


def _wrap_text(draw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _draw_shadow_text(draw, text: str, font, color: tuple, x: int, y: int):
    if len(color) == 4:
        shadow_color = (0, 0, 0, color[3])
        text_color = color
    else:
        shadow_color = (0, 0, 0)
        text_color = color
    draw.text((x + 4, y + 4), text, font=font, fill=shadow_color)
    draw.text((x, y), text, font=font, fill=text_color)


class ThumbnailMaker(PipelineStage):
    name = "thumbnail_maker"

    def execute(self, ctx: JobContext) -> JobContext:
        ctx.thumbnail_path = _generate_thumbnail(
            topic=ctx.topic,
            niche=ctx.niche,
            spec=ctx.format_spec,
            video_path=ctx.video_path,
            workspace=ctx.workspace,
        )
        return ctx

    def _load_from_checkpoint(self, ctx: JobContext) -> JobContext:
        path = ctx.workspace / "thumbnail.jpg"
        if path.exists():
            ctx.thumbnail_path = path
        return ctx
