"""
scripts/find_topics.py -- Find trending topics for a YouTube niche.

Uses Google Trends (pytrends) + YouTube Data API v3 + Claude Haiku to
discover topics with high current interest and low existing quality coverage.

Usage:
  python scripts/find_topics.py --niche horror
  python scripts/find_topics.py --niche ranking --count 8
  python scripts/find_topics.py --niche what_if --timeframe "now 1-m"
  python scripts/find_topics.py --niche horror --threshold 6

Options:
  --niche      Niche ID from config/niches.yaml (required)
  --count      Number of final ranked topics to display (default: 5)
  --timeframe  Google Trends timeframe: "now 7-d" | "now 1-m" | "today 3-m" (default: "now 7-d")
  --threshold  Minimum YouTube opportunity score 1-10 to include a topic (default: 5)
  --list-niches  Print all configured niches and exit

Requirements:
  pip install pytrends>=4.9.2
  YOUTUBE_DATA_API_KEY=... in .env  (optional but recommended; free Google Cloud key)
  ANTHROPIC_API_KEY=...             (already in .env)

Output example:
  [1] "The Oakville Blobs (1994)"
      Niche fit: 9/10  |  Freshness: 8/10  |  Score: 8.5
      YouTube coverage: none (0 recent videos)
      Why: Obscure paranormal incident with vivid visual details; zero recent coverage.

  [2] "Valentich Disappearance"
      Niche fit: 8/10  |  Freshness: 7/10  |  Score: 7.5
      YouTube coverage: low (top video: 12,400 views)
      Why: Unsolved aviation mystery with rising search interest this week.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("find_topics")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find trending YouTube topics with low competition.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--niche", type=str, help="Niche ID (e.g. horror, ranking)")
    parser.add_argument("--count", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument(
        "--timeframe", type=str, default="now 7-d",
        help='Google Trends timeframe: "now 7-d" | "now 1-m" | "today 3-m" (default: "now 7-d")',
    )
    parser.add_argument(
        "--threshold", type=int, default=5,
        help="Min YouTube opportunity score 1-10 (default: 5)",
    )
    parser.add_argument(
        "--list-niches", action="store_true",
        help="Print all configured niches and exit",
    )
    args = parser.parse_args()

    # -- List niches and exit ---------------------------------------------------
    if args.list_niches:
        try:
            from pipeline.niche_config import list_niches
            niches = list_niches()
            print("\nConfigured niches (config/niches.yaml):")
            for n in niches:
                print(f"  {n}")
            print()
        except Exception as exc:
            print(f"Error loading niches: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    # -- Validate niche ---------------------------------------------------------
    if not args.niche:
        parser.error("--niche is required. Use --list-niches to see options.")

    try:
        from pipeline.niche_config import get_niche_profile, list_niches
        get_niche_profile(args.niche)  # validates existence
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error loading niche config: {exc}", file=sys.stderr)
        sys.exit(1)

    # -- Load haiku model from infra config -------------------------------------
    try:
        from core.config_loader import load_master_infra
        BASE_DIR = Path(__file__).parent.parent
        infra = load_master_infra(BASE_DIR)
        haiku_model = infra.get("claude", {}).get("haiku_model", "claude-haiku-4-5-20251001")
    except Exception:
        haiku_model = "claude-haiku-4-5-20251001"

    # -- Run discovery pipeline -------------------------------------------------
    print(f"\nFinding trending topics for niche: {args.niche!r}")
    print(f"  Timeframe  : {args.timeframe}")
    print(f"  Min opportunity score: {args.threshold}/10")
    print(f"  Results    : {args.count}")
    print()

    from integrations.trend_discovery import find_best_topics
    results = find_best_topics(
        niche=args.niche,
        haiku_model=haiku_model,
        count=args.count,
        opportunity_threshold=args.threshold,
        timeframe=args.timeframe,
    )

    # -- Display results --------------------------------------------------------
    if not results:
        print("No topics found. Try:")
        print("  --timeframe 'now 1-m'   (look back further)")
        print("  --threshold 3           (lower opportunity bar)")
        print("  Check that pytrends is installed: pip install pytrends>=4.9.2")
        print("  Add YOUTUBE_DATA_API_KEY to .env for better filtering")
        sys.exit(0)

    print(f"Top {len(results)} topics for '{args.niche}':\n")
    print("-" * 60)

    for result in results:
        rank = result["rank"]
        topic = result["topic"]
        niche_fit = result.get("niche_fit", "?")
        freshness = result.get("freshness", "?")
        score = result.get("score", "?")
        coverage = result.get("youtube_coverage", "unknown")
        top_views = result.get("top_view_count", 0)
        reason = result.get("reason", "")
        opp_score = result.get("opportunity_score", "?")

        print(f"[{rank}] \"{topic}\"")
        print(f"    Niche fit: {niche_fit}/10  |  Freshness: {freshness}/10  |  Score: {score}")

        if coverage == "none":
            print(f"    YouTube coverage: none (no recent videos found)")
        elif coverage == "unknown":
            print(f"    YouTube coverage: unknown (API key not configured)")
        else:
            views_str = f"{top_views:,}" if top_views else "unknown"
            print(
                f"    YouTube coverage: {coverage} "
                f"(top video: {views_str} views, opportunity: {opp_score}/10)"
            )

        if reason:
            print(f"    Why: {reason}")
        print()

    print("-" * 60)
    print(f"\nTo use a topic, run:")
    best_topic = results[0]["topic"] if results else "your topic here"
    print(f"  python scripts/dev_stage1_script.py")
    print(f"  (set NICHE = \"{args.niche}\" and TOPIC = \"{best_topic}\")")
    print()


if __name__ == "__main__":
    main()
