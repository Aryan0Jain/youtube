"""
integrations/trend_discovery.py -- Trending topic discovery for YouTube niches.

Three-stage pipeline:
  1. get_trending_topics()    -- pytrends (Google Trends, no API key) -> rising queries
  2. search_youtube_coverage() -- YouTube Data API v3 -> coverage quality score
  3. score_topics_for_niche()  -- Claude Haiku -> niche fit + freshness ranking

Typical usage (via scripts/find_topics.py):
  topics = get_trending_topics(["true crime", "unexplained mystery"], timeframe="now 7-d")
  filtered = [t for t in topics if search_youtube_coverage(t)["opportunity_score"] >= 6]
  ranked = score_topics_for_niche(filtered, niche="horror", haiku_model="claude-haiku-...")
  for r in ranked:
      print(r["rank"], r["topic"], "--", r["reason"])

Requirements:
  pip install pytrends>=4.9.2
  YOUTUBE_DATA_API_KEY in .env (free; 10,000 units/day quota)
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


# -- Stage 1: Google Trends rising queries -------------------------------------

def get_trending_topics(
    niche_keywords: list[str],
    timeframe: str = "now 7-d",
    geo: str = "US",
    max_results: int = 15,
) -> list[str]:
    """
    Return rising Google Trends search queries related to niche_keywords.

    Uses pytrends (no API key required). Rising queries are those with a
    significant increase in search interest (breakout = >5000% rise).

    Args:
        niche_keywords: seed keyword list (up to 5 used per pytrends call)
        timeframe:      pytrends timeframe string ("now 7-d", "now 1-m", etc.)
        geo:            country code ("US", "GB", "" for worldwide)
        max_results:    cap on returned topics

    Returns:
        Deduplicated list of rising query strings. Empty list on any failure.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        log.error(
            "pytrends not installed. Run: pip install pytrends>=4.9.2"
        )
        return []

    seeds = niche_keywords[:5]  # pytrends max is 5 keywords per payload
    log.info(f"Google Trends: seeds={seeds}, timeframe={timeframe}, geo={geo}")

    rising: list[str] = []
    seen: set[str] = set()

    try:
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pytrends.build_payload(seeds, timeframe=timeframe, geo=geo)

        related = pytrends.related_queries()

        for seed in seeds:
            seed_data = related.get(seed, {})
            if not seed_data:
                continue
            rising_df = seed_data.get("rising")
            if rising_df is None or rising_df.empty:
                continue
            for query in rising_df["query"].tolist():
                q = str(query).strip()
                if q and q.lower() not in seen:
                    seen.add(q.lower())
                    rising.append(q)
                    if len(rising) >= max_results:
                        break
            if len(rising) >= max_results:
                break

        log.info(f"Google Trends: found {len(rising)} rising queries: {rising[:5]}...")
        return rising[:max_results]

    except Exception as exc:
        log.warning(f"Google Trends query failed: {exc}")
        return []


# -- Stage 2: YouTube coverage quality score -----------------------------------

def search_youtube_coverage(
    topic: str,
    days_back: int = 30,
    min_views_for_saturation: int = 200_000,
) -> dict:
    """
    Check how well a topic is covered on YouTube recently.

    Returns a dict:
        coverage:          "none" | "low" | "medium" | "high"
        video_count:       number of videos in the last `days_back` days
        top_view_count:    view count of the most-viewed recent video
        opportunity_score: int 1-10 (10 = great opportunity, 1 = saturated)
        reason:            short explanation string

    Requires YOUTUBE_DATA_API_KEY in environment.
    Falls back gracefully (opportunity_score=5) if API key missing or quota exceeded.
    """
    api_key = os.environ.get("YOUTUBE_DATA_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        log.warning(
            "YOUTUBE_DATA_API_KEY not set -- skipping YouTube coverage check. "
            "Add it to .env to enable opportunity scoring."
        )
        return {
            "coverage": "unknown",
            "video_count": 0,
            "top_view_count": 0,
            "opportunity_score": 5,
            "reason": "YouTube API key not configured",
        }

    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=api_key)

        published_after = (
            datetime.utcnow() - timedelta(days=days_back)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Search for recent videos on this topic
        search_response = youtube.search().list(
            q=topic,
            part="id",
            type="video",
            publishedAfter=published_after,
            order="viewCount",
            maxResults=5,
        ).execute()

        items = search_response.get("items", [])
        video_count = len(items)

        if video_count == 0:
            return {
                "coverage": "none",
                "video_count": 0,
                "top_view_count": 0,
                "opportunity_score": 10,
                "reason": f"No videos found for '{topic}' in the last {days_back} days",
            }

        # Fetch view counts for the top results
        video_ids = [item["id"]["videoId"] for item in items if "videoId" in item.get("id", {})]
        top_view_count = 0

        if video_ids:
            stats_response = youtube.videos().list(
                part="statistics",
                id=",".join(video_ids),
            ).execute()

            view_counts = []
            for vid in stats_response.get("items", []):
                vc = int(vid.get("statistics", {}).get("viewCount", 0))
                view_counts.append(vc)

            if view_counts:
                top_view_count = max(view_counts)

        # Score: high views on a recent video = saturated; few/low views = opportunity
        if top_view_count >= min_views_for_saturation:
            coverage = "high"
            opportunity_score = max(1, 4 - min(3, video_count))
            reason = (
                f"Top video has {top_view_count:,} views in {days_back}d -- "
                "topic is actively covered by large channels"
            )
        elif top_view_count >= 50_000:
            coverage = "medium"
            opportunity_score = 6
            reason = (
                f"Top video has {top_view_count:,} views -- "
                "some coverage, but not saturated"
            )
        elif top_view_count > 0:
            coverage = "low"
            opportunity_score = 8
            reason = (
                f"Top video has only {top_view_count:,} views -- "
                "weak coverage despite search interest"
            )
        else:
            coverage = "none"
            opportunity_score = 10
            reason = f"No meaningful coverage found in the last {days_back} days"

        log.info(
            f"YouTube coverage: '{topic}' -> "
            f"coverage={coverage}, score={opportunity_score}, top_views={top_view_count:,}"
        )
        return {
            "coverage": coverage,
            "video_count": video_count,
            "top_view_count": top_view_count,
            "opportunity_score": opportunity_score,
            "reason": reason,
        }

    except Exception as exc:
        log.warning(f"YouTube coverage check failed for '{topic}': {exc}")
        return {
            "coverage": "unknown",
            "video_count": 0,
            "top_view_count": 0,
            "opportunity_score": 5,
            "reason": f"API error: {exc}",
        }


# -- Stage 3: Claude Haiku scoring ---------------------------------------------

def score_topics_for_niche(
    topics: list[str],
    niche: str,
    haiku_model: str,
    top_n: int = 5,
) -> list[dict]:
    """
    Use Claude Haiku to rank topic candidates by niche fit and freshness.

    Returns a list of dicts (up to top_n), sorted by score descending:
        rank:        int (1 = best)
        topic:       str
        niche_fit:   int 1-10
        freshness:   int 1-10
        score:       float (combined)
        reason:      str (one sentence explanation)

    Falls back to returning topics in original order with score=5 on failure.
    """
    if not topics:
        return []

    # Import here to avoid circular dependency
    try:
        from integrations.claude_client import _get_client
        client = _get_client()
    except Exception as exc:
        log.warning(f"Claude client unavailable for topic scoring: {exc}")
        return [
            {"rank": i + 1, "topic": t, "niche_fit": 5, "freshness": 5,
             "score": 5.0, "reason": "Scoring unavailable"}
            for i, t in enumerate(topics[:top_n])
        ]

    topic_list = "\n".join(f"- {t}" for t in topics[:20])

    prompt = (
        f"You are a YouTube content strategist specializing in the '{niche}' niche.\n\n"
        f"Evaluate each topic candidate below for a new '{niche}' YouTube channel. "
        f"Score each on:\n"
        f"  - niche_fit (1-10): does this topic work well for '{niche}' format and audience?\n"
        f"  - freshness (1-10): is there a timely angle, recent development, or "
        f"underexplored perspective? (10 = very timely, 1 = exhausted/cliched)\n\n"
        f"Topics to evaluate:\n{topic_list}\n\n"
        f"Return a JSON array of objects with exactly these keys:\n"
        f'  "topic", "niche_fit", "freshness", "reason"\n'
        f"where reason is ONE sentence (max 20 words) explaining the combined score.\n"
        f"Sort by (niche_fit + freshness) descending. Return ONLY valid JSON, no other text."
    )

    try:
        response = client.messages.create(
            model=haiku_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                ln for ln in lines if not ln.strip().startswith("```")
            ).strip()

        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("Expected JSON array")

        results = []
        for i, item in enumerate(parsed[:top_n]):
            nf = int(item.get("niche_fit", 5))
            fr = int(item.get("freshness", 5))
            results.append({
                "rank": i + 1,
                "topic": str(item.get("topic", "")),
                "niche_fit": nf,
                "freshness": fr,
                "score": round((nf + fr) / 2, 1),
                "reason": str(item.get("reason", "")),
            })

        log.info(
            f"Claude scored {len(results)} topics for niche='{niche}'. "
            f"Top: {results[0]['topic']!r} (score={results[0]['score']})"
            if results else f"No results returned for niche='{niche}'"
        )
        return results

    except Exception as exc:
        log.warning(f"Claude topic scoring failed: {exc}")
        return [
            {"rank": i + 1, "topic": t, "niche_fit": 5, "freshness": 5,
             "score": 5.0, "reason": "Scoring failed -- using original order"}
            for i, t in enumerate(topics[:top_n])
        ]


# -- Full pipeline: find best topics for a niche --------------------------------

def find_best_topics(
    niche: str,
    haiku_model: str,
    count: int = 5,
    opportunity_threshold: int = 5,
    timeframe: str = "now 7-d",
) -> list[dict]:
    """
    Full three-stage pipeline: Trends -> YouTube coverage filter -> Claude ranking.

    Args:
        niche:                  niche ID (must exist in config/niches.yaml)
        haiku_model:            Claude Haiku model ID for scoring
        count:                  number of final results to return
        opportunity_threshold:  minimum YouTube opportunity_score to keep a topic (1-10)
        timeframe:              Google Trends timeframe string

    Returns:
        Ranked list of topic dicts from score_topics_for_niche().
    """
    # Load niche trend keywords
    try:
        from pipeline.niche_config import get_niche_profile
        profile = get_niche_profile(niche)
        seed_keywords = list(profile.trend_keywords)
    except Exception as exc:
        log.warning(f"Could not load trend_keywords for niche='{niche}': {exc}")
        seed_keywords = [niche.replace("_", " ")]

    log.info(f"[{niche}] Step 1: Fetching Google Trends rising queries...")
    trending = get_trending_topics(seed_keywords, timeframe=timeframe)

    if not trending:
        log.warning(f"[{niche}] No trending topics found. Try a different timeframe or seeds.")
        return []

    log.info(f"[{niche}] Step 2: Checking YouTube coverage for {len(trending)} topics...")
    filtered: list[str] = []
    coverage_results: list[dict] = []

    for topic in trending:
        coverage = search_youtube_coverage(topic)
        coverage_results.append({"topic": topic, **coverage})
        if coverage["opportunity_score"] >= opportunity_threshold:
            filtered.append(topic)
        # Small delay to respect API rate limits
        time.sleep(0.5)

    log.info(
        f"[{niche}] After coverage filter: {len(filtered)}/{len(trending)} topics pass "
        f"(opportunity >= {opportunity_threshold})"
    )

    if not filtered:
        # Relax threshold and take the best available
        log.warning(
            f"[{niche}] No topics passed threshold={opportunity_threshold}. "
            "Using all trending topics for scoring."
        )
        filtered = trending

    log.info(f"[{niche}] Step 3: Claude Haiku scoring {len(filtered)} topics...")
    ranked = score_topics_for_niche(filtered, niche=niche, haiku_model=haiku_model, top_n=count)

    # Attach coverage data to final results
    coverage_map = {r["topic"]: r for r in coverage_results}
    for result in ranked:
        cov = coverage_map.get(result["topic"], {})
        result["youtube_coverage"] = cov.get("coverage", "unknown")
        result["opportunity_score"] = cov.get("opportunity_score", 5)
        result["top_view_count"] = cov.get("top_view_count", 0)

    return ranked
