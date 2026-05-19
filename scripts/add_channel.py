"""
CLI wizard for adding a new YouTube channel.

Usage:
  # Step 1: Run OAuth consent flow (opens browser, saves token)
  python scripts/add_channel.py --auth --channel-id my_channel

  # Step 2: Validate a channel YAML
  python scripts/add_channel.py --validate --channel-id my_channel

  # List all configured channels
  python scripts/add_channel.py --list
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).parent.parent
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def cmd_auth(channel_id: str):
    yaml_path = BASE_DIR / "config" / "channels" / f"{channel_id}.yaml"
    if not yaml_path.exists():
        log.error(f"Channel config not found: {yaml_path}")
        log.error("Create the YAML first, then run --auth")
        sys.exit(1)

    import yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    oauth_token_path = cfg.get("oauth_token_path", f"config/credentials/{channel_id}_oauth.json")
    token_abs = BASE_DIR / oauth_token_path

    client_secrets = BASE_DIR / "config" / "credentials" / "client_secrets.json"
    if not client_secrets.exists():
        log.error(f"client_secrets.json not found at {client_secrets}")
        log.error("Download it from Google Cloud Console → APIs & Services → Credentials")
        sys.exit(1)

    print(f"\nStarting OAuth flow for channel: {channel_id}")
    print("A browser window will open. Log in with the YouTube channel's Google account.\n")

    from integrations.youtube_client import run_oauth_flow
    run_oauth_flow(
        channel_id=channel_id,
        client_secrets_path=str(client_secrets),
        token_output_path=str(token_abs),
    )
    print(f"\nDone. Token saved to: {token_abs}")
    print("On the VM, place this file at the same relative path.")


def cmd_validate(channel_id: str):
    from core.config_loader import load_all, ConfigValidationError

    print(f"\nValidating config for: {channel_id}")
    try:
        channels = load_all(BASE_DIR)
    except ConfigValidationError as e:
        log.error(f"Config error: {e}")
        sys.exit(1)

    channel = channels.get(channel_id)
    if not channel:
        log.error(f"Channel '{channel_id}' not found. Available: {list(channels)}")
        sys.exit(1)

    print(f"  display_name:       {channel.display_name}")
    print(f"  youtube_channel_id: {channel.youtube_channel_id}")
    print(f"  oauth_token_path:   {channel.oauth_token_path}")
    print(f"  series ({len(channel.series)}):")
    for s in channel.series:
        print(f"    [{s.series_id}] format={s.format} niche={s.niche}")
        print(f"      schedule: {s.schedule.days_of_week} at {s.schedule.time_utc} UTC")
        print(f"      topic_source: {s.topic_source.type}")

    token_path = BASE_DIR / channel.oauth_token_path
    if token_path.exists():
        print(f"\n  OAuth token: FOUND ✓")
    else:
        print(f"\n  OAuth token: MISSING — run --auth first")

    print("\nValidation passed.")


def cmd_list():
    from core.config_loader import load_all, ConfigValidationError
    try:
        channels = load_all(BASE_DIR)
    except ConfigValidationError as e:
        log.error(f"Config error: {e}")
        sys.exit(1)

    if not channels:
        print("No channels configured. Add a YAML to config/channels/")
        return

    for ch_id, ch in channels.items():
        print(f"\n{ch_id} — {ch.display_name}")
        for s in ch.series:
            print(f"  {s.series_id} ({s.format}, {s.niche})"
                  f" → {s.schedule.days_of_week} {s.schedule.time_utc} UTC")


def main():
    parser = argparse.ArgumentParser(description="Add and manage YouTube channels")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--auth", action="store_true", help="Run OAuth flow for a channel")
    group.add_argument("--validate", action="store_true", help="Validate channel YAML")
    group.add_argument("--list", action="store_true", help="List all configured channels")
    parser.add_argument("--channel-id", help="Channel ID (filename stem of YAML)")
    args = parser.parse_args()

    if args.auth:
        if not args.channel_id:
            parser.error("--auth requires --channel-id")
        cmd_auth(args.channel_id)
    elif args.validate:
        if not args.channel_id:
            parser.error("--validate requires --channel-id")
        cmd_validate(args.channel_id)
    elif args.list:
        cmd_list()


if __name__ == "__main__":
    main()
