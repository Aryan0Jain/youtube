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

    If the first draft is under 90% of the target word count, a single
    expansion pass is made automatically — Claude is asked to lengthen
    the shortest sections until the script hits the target range.

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

    word_min = int(word_cap * 0.91)   # 91% floor (~1000 for a 1100-word target)

    # Merge in any series-level style notes and format instructions
    user_parts = [f'Write a YouTube script about: "{topic}"']
    if style_notes and style_notes.strip():
        user_parts.append(f"Additional style guidance: {style_notes.strip()}")
    if script_prompt_suffix and script_prompt_suffix.strip():
        user_parts.append(script_prompt_suffix.strip())
    user_parts.append(
        f"Required word count: {word_min}–{word_cap} words. "
        "Return ONLY the script text. No stage directions, no headers, "
        "no timestamps, no meta-commentary."
    )
    user = "\n\n".join(user_parts)

    log.info(f"Generating {niche} script ({word_min}–{word_cap}w) for: {topic!r}")
    response = _get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    script = response.content[0].text.strip()
    word_count = len(script.split())
    log.info(f"  Draft: {word_count} words (target {word_min}–{word_cap})")

    # -- Expansion loop: retry up to 3 times until word_min is reached ----------
    for pass_num in range(1, 4):
        if word_count >= word_min:
            break
        deficit = word_min - word_count
        log.info(
            f"  Script is {deficit} words short — expansion pass {pass_num}/3 "
            f"(add ~{deficit} more words)"
        )
        expand_user = (
            f"This script is {word_count} words. It needs to reach at least {word_min} words "
            f"(about {deficit} more words). Expand it by:\n"
            "1. Adding one extra detail or consequence sentence to the entries that are shortest.\n"
            "2. Strengthening any tease lines that feel thin.\n"
            "Do NOT add new ranked entries. Do NOT change the structure or tone.\n"
            "Return the complete expanded script only — no commentary.\n\n"
            f"CURRENT SCRIPT:\n{script}"
        )
        expand_response = _get_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": expand_user}],
        )
        expanded = expand_response.content[0].text.strip()
        new_count = len(expanded.split())
        log.info(f"  After expansion pass {pass_num}: {new_count} words")
        # Only accept expansion if it actually made it longer (sanity check)
        if new_count > word_count:
            script = expanded
            word_count = new_count
        else:
            log.warning(f"  Expansion pass {pass_num} did not increase length — stopping")
            break

    return script


# ── Topic generation ──────────────────────────────────────────────────────────

def generate_topic(autogen_prompt: str, avoid_list: list[str],
                   haiku_model: str, max_tokens: int = 256,
                   refreshable_topics: list[str] | None = None,
                   refresh_after_days: int = 180) -> str:
    """
    Generate a new topic title.

    avoid_list        — recently used topics that must NOT be repeated.
    refreshable_topics — old topics (> refresh_after_days) that MAY be revisited
                         with updated data; Claude adds a year marker if it picks one.
    """
    avoid_section = ""
    if avoid_list:
        recent = "\n".join(f"- {t}" for t in avoid_list[:40])
        avoid_section = f"\n\nDo NOT generate any of these recently used topics:\n{recent}"

    refresh_section = ""
    if refreshable_topics:
        old = "\n".join(f"- {t}" for t in refreshable_topics[:20])
        refresh_section = (
            f"\n\nThese topics were covered over {refresh_after_days} days ago. "
            f"Rankings and records change — you MAY revisit one of these if current data "
            f"would make for a significantly better/updated video. "
            f"If you do, add a year marker to distinguish it (e.g. '2026 Edition' or "
            f"'Updated: ...'):\n{old}"
        )

    prompt = autogen_prompt.strip() + avoid_section + refresh_section

    response = _get_client().messages.create(
        model=haiku_model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
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


# ── Niche-specific keyword style instructions ─────────────────────────────────
#
# Topic-literal keywords ("bodies found in ravine") return wrong clips on Pexels.
# Each niche needs a different visual strategy to find clips that feel right.
#
_KEYWORD_STYLE: dict[str, str] = {
    "horror": (
        "IMPORTANT: Return atmosphere and mood keywords, NOT topic-literal keywords. "
        "Do NOT describe what happens in the script -- describe how it should FEEL visually. "
        "Instead of 'bodies found in ravine' return 'dark pine forest fog'. "
        "Instead of 'Soviet expedition camp' return 'abandoned tent snow isolation'. "
        "Keywords should create emotional dread and tension through imagery, "
        "not illustrate the story literally."
    ),
    "what_if": (
        "IMPORTANT: Return visual consequence keywords that show the EFFECT, not the cause. "
        "Instead of 'bees disappearing' return 'empty garden withered flowers'. "
        "Instead of 'economic collapse' return 'empty supermarket shelves'. "
        "Show the world AFTER the scenario -- the visible consequences."
    ),
    "ranking": (
        "IMPORTANT: Return dramatic scale and impact keywords. "
        "Instead of 'earthquake in Turkey' return 'earthquake rubble destruction aftermath'. "
        "Instead of 'pandemic' return 'empty streets abandoned city'. "
        "Keywords should convey scale, devastation, and magnitude -- "
        "not just name the event."
    ),
    "shock_facts": (
        "Return visually striking science and nature keywords. "
        "Prefer close-up, dramatic, or counter-intuitive visuals: "
        "'brain neuron glow', 'deep ocean darkness', 'cell microscope'. "
        "Avoid generic keywords -- be specific about what makes each fact visual."
    ),
    "historical_versus": (
        "Return historically evocative stock footage keywords for each segment. "
        "Think: 'ancient Rome colosseum', 'medieval knight armor', 'samurai sword ceremony', "
        "'roman legion march', 'ancient temple ruins'. "
        "Avoid modern footage -- keep it historical and cinematic."
    ),
}


def extract_segment_keywords(script_text: str, segment_duration_sec: float,
                              audio_duration_sec: float, haiku_model: str,
                              niche: str = "") -> list[str]:
    """
    Divide the script into ~30-second segments and extract one specific visual
    search phrase per segment. Returns a list with one entry per segment.

    This produces 16-20 targeted keywords for an 8-min video, allowing 80-100
    unique clips (clips_per_segment x num_segments) with no repetition.

    Pass niche= for atmosphere-first keyword selection (recommended).
    Without niche, falls back to generic topic-literal keywords.
    """
    num_segments = max(1, round(audio_duration_sec / segment_duration_sec))

    # Niche-specific visual strategy instruction
    niche_instruction = _KEYWORD_STYLE.get(niche, "")
    if niche_instruction:
        niche_instruction = f"\n\n{niche_instruction}"

    prompt = (
        f"The following YouTube script will be narrated over approximately "
        f"{audio_duration_sec:.0f} seconds ({num_segments} segments of "
        f"~{segment_duration_sec:.0f}s each).\n\n"
        f"Divide the script into {num_segments} equal narrative segments. "
        f"For each segment, return ONE short visual search phrase (2-4 words) "
        f"that best matches the imagery being described in that part of the script."
        f"{niche_instruction}\n\n"
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


def extract_ranking_clip_keywords(rank_items: list[dict], script_text: str,
                                   haiku_model: str) -> list[str]:
    """
    For the ranking niche: extract one Pexels search keyword per rank entry.
    Uses the rank name and surrounding script context to produce visually
    specific, emotionally resonant search phrases.

    Args:
        rank_items: list of {"rank": N, "name": "...", "word_idx": N} from niche_metadata
        script_text: full script text (for context)
        haiku_model: Claude Haiku model identifier

    Returns:
        list of keyword strings, one per rank item (same order as rank_items).
        Falls back to generic terms on any failure.
    """
    if not rank_items:
        return []

    rank_lines = "\n".join(
        f"#{item.get('rank', '?')}: {item.get('name', 'unknown')}"
        for item in rank_items
    )
    n = len(rank_items)

    prompt = (
        f"A YouTube ranking video has {n} entries. For each entry below, return ONE "
        f"3-5 word Pexels stock video search phrase.\n\n"
        f"Rules:\n"
        f"- Focus on dramatic, visual, emotionally resonant footage (scale, impact, aftermath)\n"
        f"- NOT the literal name: instead of '1887 Yellow River Flood' use 'river flooding "
        f"swallowed villages'\n"
        f"- NOT generic: instead of 'earthquake' use 'earthquake rubble collapse chaos'\n"
        f"- Each phrase must be visually DISTINCT from the others\n"
        f"- Return exactly {n} lines, one phrase per line, no numbering, no extra text\n\n"
        f"Ranking entries:\n{rank_lines}\n\n"
        f"Script excerpt (for context):\n{script_text[:2000]}"
    )

    try:
        response = _get_client().messages.create(
            model=haiku_model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        lines = [ln.strip(" -•*0123456789.)") for ln in
                 response.content[0].text.strip().splitlines()]
        keywords = [ln for ln in lines if ln]
        # Pad if needed
        while len(keywords) < n:
            keywords.append(keywords[-1] if keywords else "natural disaster aftermath")
        return keywords[:n]
    except Exception as exc:
        log.warning(f"extract_ranking_clip_keywords failed ({exc}), using fallback keywords")
        return [f"{item.get('name', 'disaster')} aftermath" for item in rank_items]


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
        '"rank" (integer), '
        '"name" (entry name, max 28 chars), '
        '"year" (string, when it occurred, e.g. "1815" or "1931-32", max 9 chars — use empty string if unknown), '
        '"scale" (string, death toll or key impact in max 22 chars, e.g. "~71,000 dead" or "500k km² lost" — use empty string if none), '
        '"word_idx" (approximate word index where this rank is introduced, starting from 0). '
        "Return ONLY valid JSON array, nothing else. Example: "
        '[{"rank":10,"name":"Shaanxi Earthquake","year":"1556","scale":"~830,000 dead","word_idx":85}]'
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


# ── Emotional keyword extraction ──────────────────────────────────────────────

def extract_emotional_keywords(script_text: str, haiku_model: str,
                                max_words: int = 12) -> list[str]:
    """
    Ask Claude Haiku to identify the most emotionally impactful words in the
    script for bold-emphasis in subtitles. Returns a list of lowercase single
    words (no numbers, no phrases).

    Cost: ~$0.001/call. Falls back to [] on any error so the pipeline
    never blocks on this non-critical step.
    """
    prompt = (
        f"Read this YouTube script and return the {max_words} single words that carry "
        "the most emotional weight — words that represent death, destruction, scale, "
        "or shock. These words will be bold-emphasized in subtitles.\n"
        "Rules: only lowercase single words (no phrases, no numbers), no duplicates, "
        "return as a JSON array of strings only.\n"
        "Example output: [\"collapsed\",\"drowned\",\"never\",\"deadliest\",\"erased\"]\n\n"
        f"SCRIPT:\n{script_text[:3000]}"
    )
    try:
        response = _get_client().messages.create(
            model=haiku_model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        result = json.loads(raw)
        if isinstance(result, list):
            # Sanitise: lowercase strings only, no numbers
            return [str(w).lower() for w in result if str(w).isalpha()]
        return []
    except Exception as exc:
        log.warning(f"extract_emotional_keywords failed: {exc}")
        return []


# ── Video metadata (SEO title/description/tags) ───────────────────────────────

_NICHE_TITLE_FORMULAS = {
    "ranking":            'Use a "Top 10 [X] That [Strong Verb]" format. End with "#1 Will Shock You" or "You Won\'t Believe #1". Numbers beat words ("10" not "ten"). One or two POWER WORDS in caps allowed.',
    "horror":             'Use "The [Disturbing/Terrifying/Dark] [Secret/Mystery/Truth] of [Subject]" or "What REALLY Happened to [Person/Place]".',
    "shock_facts":        'Use "[N] [Topic] Facts That [Emotional Hook]" or "[N] Things About [Topic] That Will [Reaction]".',
    "quiz":               'Use "[Topic] Quiz — Can YOU Score 100%? 🎯" or "Only [X]% of People Can Pass This [Topic] Quiz".',
    "historical_versus":  'Use "[A] vs [B]: Who [Actually/Really] [Verb]?" or "The REAL Difference Between [A] and [B]".',
    "myth_busting":       'Use "The [Topic] Myth That [Millions/Everyone] Believed" or "Was [Famous Claim] Actually TRUE?".',
    "what_if":            'Use "What If [Scenario]? The [Shocking/Terrifying] Answer" or "What Would Happen If [Scenario]?".',
}

_NICHE_HASHTAGS = {
    "ranking":           "#Top10 #Rankings #History",
    "horror":            "#TrueStory #Mystery #Horror",
    "shock_facts":       "#ShockingFacts #DidYouKnow #MindBlown",
    "quiz":              "#Quiz #Challenge #Trivia",
    "historical_versus": "#History #VsDebate #Historical",
    "myth_busting":      "#MythBusted #Facts #TrueOrFalse",
    "what_if":           "#WhatIf #Science #ThoughtExperiment",
}

_NICHE_CONTENT_TYPE_TAGS = {
    "ranking":           "top 10 list, countdown video, ranking video",
    "horror":            "true crime documentary, mystery explained, unsolved mysteries",
    "shock_facts":       "amazing facts, fun facts, educational video",
    "quiz":              "quiz video, trivia challenge, knowledge test",
    "historical_versus": "history comparison, versus video, historical debate",
    "myth_busting":      "myth busted, fact check, debunked",
    "what_if":           "what if scenario, hypothetical, thought experiment",
}


def generate_video_metadata(topic: str, niche: str, script_text: str,
                             haiku_model: str) -> dict:
    """
    Generate SEO-optimised title, description, and tags from script content.

    Title:       ≤70 chars, niche-specific formula, 1-2 capitalised POWER WORDS
    Description: 400-500 words, structured with hook + bullets + CTA + hashtags
    Tags:        15-20 tags mixing exact-match, broad category, long-tail, and
                 content-type keywords
    """
    title_guidance = _NICHE_TITLE_FORMULAS.get(
        niche,
        "Write a compelling title (≤70 chars) that creates curiosity or urgency."
    )
    hashtags = _NICHE_HASHTAGS.get(niche, "#YouTube #Education #Interesting")
    content_type_tags = _NICHE_CONTENT_TYPE_TAGS.get(niche, "educational video, youtube video")

    prompt = (
        f"You are an expert YouTube SEO specialist. Write metadata for a video:\n"
        f"Topic: {topic}\n"
        f"Niche: {niche}\n\n"
        f"Script excerpt:\n{script_text[:1500]}\n\n"
        "---\n\n"
        "TITLE RULES:\n"
        f"- {title_guidance}\n"
        "- Maximum 70 characters\n"
        "- Sentence case with 1-2 POWER WORDS capitalised for emphasis\n"
        "- Use numbers where possible (digits, not words)\n"
        "- No trailing punctuation\n\n"
        "DESCRIPTION RULES (write 400-500 words total):\n"
        "- Line 1 (≤125 chars): the single most shocking/compelling thing in the video. "
        "This is the hook visible before 'Show more'.\n"
        "- Line 2: blank line\n"
        "- 2-sentence summary with the primary keyword used naturally\n"
        "- Blank line\n"
        "- '📌 IN THIS VIDEO:' header followed by 4-5 bullet points starting with •\n"
        "- Blank line\n"
        "- 1-2 paragraphs of context (natural keyword embedding, no stuffing)\n"
        "- Blank line\n"
        "- '👉 SUBSCRIBE for more [relevant keyword] every week'\n"
        "- '🔔 Ring the bell so you never miss a video'\n"
        "- Blank line\n"
        f"- End with exactly these hashtags on the last line: {hashtags}\n\n"
        "TAGS RULES (return exactly 18 comma-separated tags, no #):\n"
        "- 4 exact-match tags (keywords from the title, verbatim)\n"
        "- 4 broad category tags (e.g. 'history', 'natural disasters', 'science')\n"
        "- 4 long-tail tags (4-6 word phrases people actually search)\n"
        f"- 4 content-type tags: {content_type_tags}, plus 2 variations\n"
        "- 2 channel brand tags: 'facts you didn't know', 'most shocking moments'\n\n"
        "Return in EXACTLY this format (no extra text before or after):\n"
        "TITLE: <title>\n"
        "DESCRIPTION: <full description>\n"
        "TAGS: <tag1>, <tag2>, ..., <tag18>"
    )

    response = _get_client().messages.create(
        model=haiku_model,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    result: dict = {"title": topic, "description": "", "tags": []}

    desc_lines: list[str] = []
    in_desc = False

    for line in raw.splitlines():
        if line.startswith("TITLE:"):
            result["title"] = line[6:].strip()[:100]  # hard cap for YouTube API
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

    # Fallback: ensure we always have some tags even if parsing failed
    if not result["tags"]:
        result["tags"] = [topic, niche, "top 10", "facts", "history",
                          "educational", "documentary"]

    return result
