import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube"]

_clients: dict[str, object] = {}


def get_client(oauth_token_path: str):
    """
    Return a per-channel YouTube API client.
    Auto-refreshes the access token using the stored refresh_token.
    Token file is updated in-place after refresh.
    """
    if oauth_token_path in _clients:
        client = _clients[oauth_token_path]
        # Re-use unless token is expired; rely on google-auth to detect expiry
        return client

    client = _build_client(oauth_token_path)
    _clients[oauth_token_path] = client
    return client


def _build_client(oauth_token_path: str):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    token_path = Path(oauth_token_path)
    if not token_path.exists():
        raise FileNotFoundError(
            f"YouTube OAuth token not found: {oauth_token_path}\n"
            "Run: python scripts/add_channel.py --auth --channel-id <id>"
        )

    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.expired and creds.refresh_token:
        log.info(f"Refreshing YouTube token: {oauth_token_path}")
        creds.refresh(Request())
        token_path.write_text(creds.to_json())

    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def run_oauth_flow(channel_id: str, client_secrets_path: str, token_output_path: str):
    """
    Run the one-time OAuth consent flow (requires a browser).
    Called by scripts/add_channel.py --auth.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, SCOPES)
    creds = flow.run_local_server(port=0)

    Path(token_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(token_output_path).write_text(creds.to_json())
    log.info(f"OAuth token saved: {token_output_path}")
    print(f"\nToken saved to: {token_output_path}")
    print("Copy this file to the VM at the same path, then add it to your channel YAML.")
