"""
CLI: push topics from a local CSV into a Google Sheet tab.

Usage:
  python scripts/backfill_topics.py \\
    --sheet-id 1BxiMVs0XRA5nF... \\
    --tab horror_shorts_topics \\
    --csv my_topics.csv

CSV format: one topic per line (no header row needed).
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Backfill topics into a Google Sheet")
    parser.add_argument("--sheet-id", required=True)
    parser.add_argument("--tab", required=True, help="Sheet tab name")
    parser.add_argument("--csv", required=True, help="Path to CSV file with topics")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    topics: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                topics.append(row[0].strip())

    if not topics:
        print("No topics found in CSV")
        sys.exit(1)

    print(f"Appending {len(topics)} topics to '{args.tab}'...")

    from integrations import sheets_client
    for i, topic in enumerate(topics):
        sheets_client.append_topic(args.sheet_id, args.tab, topic)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(topics)}")

    print(f"Done. {len(topics)} topics appended.")


if __name__ == "__main__":
    main()
