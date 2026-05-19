import os
import logging
import json
import time

import requests
from google.oauth2.service_account import Credentials

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_credentials: Credentials | None = None


def _get_token() -> str:
    global _credentials
    if _credentials is None:
        sa_path = os.environ.get("GOOGLE_SA_KEY_PATH", "config/credentials/service_account.json")
        _credentials = Credentials.from_service_account_file(sa_path, scopes=SCOPES)
    if _credentials.expired or not _credentials.token:
        import google.auth.transport.requests
        _credentials.refresh(google.auth.transport.requests.Request())
    return _credentials.token


def _headers() -> dict:
    return {"Authorization": f"Bearer {_get_token()}", "Content-Type": "application/json"}


def _col_letter_to_index(letter: str) -> int:
    """Convert column letter (A, B, C...) to 0-based index."""
    return ord(letter.upper()) - ord("A")


def get_next_unused_topic(sheet_id: str, tab_name: str,
                          topic_column: str = "A",
                          used_column: str = "B") -> tuple[int, str] | None:
    """
    Scan the sheet tab for the first row where topic_column is non-empty and
    used_column is not 'TRUE'. Returns (row_index_1based, topic) or None.
    """
    range_notation = f"'{tab_name}'!A:Z"
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values/{range_notation}"
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    values = resp.json().get("values", [])

    topic_idx = _col_letter_to_index(topic_column)
    used_idx = _col_letter_to_index(used_column)

    for row_num, row in enumerate(values, start=1):
        if row_num == 1:
            continue  # skip header

        topic_val = row[topic_idx].strip() if len(row) > topic_idx else ""
        used_val = row[used_idx].strip().upper() if len(row) > used_idx else ""

        if topic_val and used_val != "TRUE":
            log.info(f"Found unused topic at row {row_num}: {topic_val!r}")
            return (row_num, topic_val)

    log.warning(f"No unused topics in '{tab_name}' of sheet {sheet_id}")
    return None


def mark_topic_used(sheet_id: str, tab_name: str,
                    row_index: int, used_column: str = "B"):
    """Set the used_column cell in the given row to 'TRUE'."""
    cell = f"'{tab_name}'!{used_column}{row_index}"
    url = (f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}"
           f"/values/{cell}?valueInputOption=RAW")
    body = {"values": [["TRUE"]]}
    resp = requests.put(url, headers=_headers(), json=body, timeout=30)
    resp.raise_for_status()
    log.info(f"Marked row {row_index} as used in '{tab_name}'")


def append_topic(sheet_id: str, tab_name: str, topic: str):
    """Append a new topic row (for backfill_topics CLI)."""
    url = (f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}"
           f"/values/'{tab_name}'!A:A:append?valueInputOption=RAW&insertDataOption=INSERT_ROWS")
    body = {"values": [[topic]]}
    resp = requests.post(url, headers=_headers(), json=body, timeout=30)
    resp.raise_for_status()
