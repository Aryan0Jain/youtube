import json
import logging
import os

import anthropic

log = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


# ── Script generation ─────────────────────────────────────────────────────────

def write_script(topic: str, style_notes: str, niche: str,
                 script_prompt_suffix: str, target_word_count: int,
                 model: str, max_tokens: int = 4096) -> str:
    """
    Generate a YouTube script using the niche-specific system prompt from
    config/niches.yaml (loaded via pipeline.niche_config).

    Falls back to a generic system prompt if the niche is not found.
    Returns the raw script text only.
    """
    # Use niche-specific system prompt (the main quality driver)
    try:
        from pipeline.niche_config import get_niche_profile
        profile = get_niche_profile(niche)
        system = profile.script_system_prompt
        word_cap = profile.target_word_count
    except (KeyError, Exception) as exc:
        log.warning(f"Could not load niche profile for '{niche}': {exc}. Using generic prompt.")
        system = (
            f"You are a professional YouTube scriptwriter specializing in {niche} content. "
            f"{style_notes.strip()}"
        )
        word_cap = target_word_count

    # Merge in any series-level style notes and format instructions
    user_parts = [
        f'Write a YouTube script about: "{topic}"',
    ]
    if style_notes and style_notes.strip():
        user_parts.append(f"Additional style guidance: {style_notes.strip()}")
    if script_prompt_suffix and script_prompt_suffix.strip():
        user_parts.append(script_prompt_suffix.strip())
    user_parts.append(
        f"Hard limit: {word_cap} words maximum. "
        "Return ONLY the script text. No stage directions, no headers, "
        "no timestamps, no meta-commentary."
    )
    user = "\n\n".join(user_parts)

    log.info(f"Generating {niche} script (<=​{word_cap}w) for: {topic!r}")
    response = _get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text.strip()


# ── Topic generation ──────────────────────────────────────────────────────────

def generate_topic(autogen_prompt: str, avoid_list: list[str],
                   haiku_model: str, max_tokens: int = 256) -> str:
    """Generate a new topic title. avoid_list is injected as context."""
    avoid_section = ""
    if avoid_list:
        recent = "\n".join(f"- {t}" for t in avoid_list[:40])
        avoid_section = f"\n\nDo NOT generate any of these recently used topics:\n{recent}"

    response = _get_client().messages.create(
        model=haiku_model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": autogen_prompt.strip() + avoid_section,
        }],
    )
    return response.content[0].text.strip()


# ── Keyword extraction (legacy — whole-script, for small clip counts) ─────────

def extract_keywords(script_text: str, count: int, haiku_model: str) -> list[str]:
    """Extract visual search keywords from a script for Pexels queries."""
    prompt = (
        f"From the following YouTube script, extract exactly {count} short visual "
        f"search terms suitable for a stock video library (e.g. 'dark forest', "
        f"'astronaut floating', 'city traffic at night'). Return one term per line, "
        f"no numbering, no extra text.\n\nSCRIPT:\n{script_text[:3000]}"
    )
    response = _get_client().messages.create(
        model=haiku_model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    lines = [ln.strip(" -•*") for ln in response.content[0].text.strip().splitlines()]
    return [ln for ln in lines if ln][:count]


# ── Segment keyword extraction (per-30s, for 80-100 unique clips) ─────────────

def extract_segment_keywords(script_text: str, segment_duration_sec: float,
                              audio_duration_sec: float, haiku_model: str) -> list[str]:
    """
    Divide the script into ~30-second segments and extract one specific visual
    search phrase per segment. Returns a list with one entry per segment.

    This produces 16-20 targeted keywords for an 8-min video, allowing 80-100
    unique clips (clips_per_segment x num_segments) with no repetition.
    """
    num_segments = max(1, round(audio_duration_sec / segment_duration_sec))

    prompt = (
        f"The following YouTube script will be narrated over approximately "
        f"{audio_duration_sec:.0f} seconds ({num_segments} segments of "
        f"~{segment_duration_sec:.0f}s each).\n\n"
        f"Divide the script into {num_segments} equal narrative segments. "
        f"For each segment, return ONE short visual search phrase (2-4 words) "
        f"that best matches the imagery being described in that part of the script.\n\n"
        f"Requirements:\n"
        f"- Return exactly {num_segments} lines, one phrase per line\n"
        f"- Phrases must be specific and visually distinct from each other\n"
        f"- Suitable for a stock video library (e.g. 'frozen mountain trail', "
        f"'glowing city skyline', 'scientist in lab')\n"
        f"- No numbering, no extra text -- just the phrases\n\n"
        f"SCRIPT:\n{script_text[:4000]}"
    )

    response = _get_client().messages.create(
        model=haiku_model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    lines = [ln.strip(" -•*0123456789.)") for ln in response.content[0].text.strip().splitlines()]
    keywords = [ln for ln in lines if ln]

    # Pad or trim to exactly num_segments
    if len(keywords) < num_segments:
        while len(keywords) < num_segments:
            keywords.append(keywords[-1] if keywords else "nature scenery")
    return keywords[:num_segments]


# ── Niche metadata extraction (for time-gated overlays) ──────────────────────

# Per-overlay-type extraction prompts
_OVERLAY_PROMPTS: dict[str, str] = {
    "entity_cards": (
        "Extract all named people, specific dates, and notable locations mentioned "
        "in this script. For each, return a JSON array of objects with keys: "
        '"text" (the name/date/place as it would appear on screen, max 35 chars), '
        '"role" (one short descriptor like "Expedition Leader" or "January 1959", max 25 chars), '
        '"word_idx" (approximate word index in the script where first mentioned, starting from 0). '
        "Return ONLY valid JSON array, nothing else. Example: "
        '[{"text": "IGOR DYATLOV", "role": "Expedition Leader", "word_idx": 45}]'
    ),
    "fact_counter": (
        "This script counts down facts. Extract each fact number and its headline. "
        "Return a JSON array with keys: "
        '"number" (integer, 1-10), "headline" (the fact in 4-6 words, all caps), '
        '"word_idx" (approximate word index where this fact begins, starting from 0). '
        "Return ONLY valid JSON array, nothing else. Example: "
        '[{"number": 10, "headline": "SHARKS PREDATE DINOSAURS", "word_idx": 120}]'
    ),
    "quiz": (
        "This script is a quiz. Extract each question. Return a JSON array with keys: "
        '"question" (the question text, max 60 chars), '
        '"answer" (the correct answer, max 30 chars), '
        '"hint" (one-word hint), '
        '"word_idx" (approximate word index where this question is asked, starting from 0). '
        "Return ONLY valid JSON array, nothing else."
    ),
    "ranking_card": (
        "This script is a ranking countdown. Extract each ranked entry. "
        "Return a JSON array with keys: "
        '"rank" (integer), "name" (the entry name, max 30 chars), '
        '"word_idx" (approximate word index where this rank is introduced, starting from 0). '
        "Return ONLY valid JSON array, nothing else. Example: "
        '[{"rank": 10, "name": "Amazon River", "word_idx": 85}]'
    ),
    "myth_stamp": (
        "This script busts a myth. Extract: the myth statement and the verdict. "
        "Return a JSON object with keys: "
        '"myth" (the myth in 4-6 words, all caps), '
        '"verdict" (one word: BUSTED or CONFIRMED), '
        '"myth_word_idx" (word index where myth is first stated), '
        '"verdict_word_idx" (word index of the "The truth is..." pivot line). '
        "Return ONLY valid JSON object, nothing else."
    ),
    "scale_text": (
        "This script describes events at different time scales. Extract each scale jump. "
        "Return a JSON array with keys: "
        '"label" (short label like "Day 1", "Year 10", "1000 Years Later", max 20 chars), '
        '"word_idx" (approximate word index where this time scale begins, starting from 0). '
        "Return ONLY valid JSON array, nothing else."
    ),
    "side_labels": (
        "This comparison script compares two things. Extract each category comparison. "
        "Return a JSON array with keys: "
        '"side_a" (left item label, max 20 chars), '
        '"side_b" (right item label, max 20 chars), '
        '"category" (comparison category like "Cost" or "Performance", max 15 chars), '
        '"word_idx" (approximate word index where this category begins, starting from 0). '
        "Return ONLY valid JSON array, nothing else."
    ),
}


def extract_niche_metadata(script_text: str, niche: str,
                            haiku_model: str) -> dict:
    """
    Extract structured overlay metadata from a script using Claude Haiku.
    Returns a dict with keys: overlay_type (str) and items (list or dict).

    Returns {} on any failure -- overlays are non-critical and must never
    block the pipeline.

    Cost: ~$0.002 per call (Haiku, <1000 input tokens).
    """
    try:
        from pipeline.niche_config import get_niche_profile
        profile = get_niche_profile(niche)
        overlay_type = profile.overlay_type
    except Exception:
        return {}

    if overlay_type == "none" or overlay_type not in _OVERLAY_PROMPTS:
        return {}

    extraction_prompt = _OVERLAY_PROMPTS[overlay_type]
    prompt = (
        f"{extraction_prompt}\n\n"
        f"SCRIPT (word count ~{len(script_text.split())}):\n"
        f"{script_text[:4000]}"
    )

    try:
        response = _get_client().messages.create(
            model=haiku_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                ln for ln in lines
                if not ln.strip().startswith("```")
            ).strip()

        parsed = json.loads(raw)
        count = len(parsed) if isinstance(parsed, list) else 1
        log.info(f"extract_niche_metadata({niche}): overlay={overlay_type}, items={count}")
        return {"overlay_type": overlay_type, "items": parsed}

    except Exception as exc:
        log.warning(f"extract_niche_metadata failed for niche='{niche}': {exc}")
        return {}


# ── Video metadata (SEO title/description/tags) ───────────────────────────────

def generate_video_metadata(topic: str, niche: str, script_text: str,
                             haiku_model: str) -> dict:
    """Generate SEO title, description, and tags from script content."""
    prompt = (
        f"Given this YouTube video about '{topic}' in the {niche} niche, generate:\n"
        "1. SEO-optimized title (max 70 chars, no clickbait, no ALL CAPS)\n"
        "2. Description (150-300 words, natural keyword embedding, ends with CTA)\n"
        "3. 10 tags (comma-separated, mix of broad and specific)\n\n"
        f"Script excerpt (first 1000 chars):\n{script_text[:1000]}\n\n"
        "Return in this exact format:\n"
        "TITLE: <title>\n"
        "DESCRIPTION: <description>\n"
        "TAGS: <tag1>, <tag2>, ..., <tag10>"
    )
    response = _get_client().messages.create(
        model=haiku_model,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    result: dict = {"title": topic, "description": "", "tags": []}

    desc_lines: list[str] = []
    in_desc = False

    for line in raw.splitlines():
        if line.startswith("TITLE:"):
            result["title"] = line[6:].strip()
            in_desc = False
        elif line.startswith("DESCRIPTION:"):
            in_desc = True
            first = line[12:].strip()
            if first:
                desc_lines.append(first)
        elif line.startswith("TAGS:"):
            in_desc = False
            result["tags"] = [t.strip() for t in line[5:].split(",") if t.strip()]
        elif in_desc:
            desc_lines.append(line)

    result["description"] = "\n".join(desc_lines).strip()
    return result
