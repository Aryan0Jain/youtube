"""
integrations/gcs_client.py -- Thin Google Cloud Storage wrapper.

Used by:
  - pipeline/video_assembler.py  -- download music files on cache miss
  - scripts/upload_music.py      -- upload local music files to GCS

Bucket name is read from config/master.yaml -> infrastructure.gcs_bucket.
Credentials use the existing GOOGLE_APPLICATION_CREDENTIALS / service account.

GCS path convention:
  music/dark_ambient.mp3          -> gs://<bucket>/music/dark_ambient.mp3
  clips/some_clip.mp4             -> gs://<bucket>/clips/some_clip.mp4
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_bucket_cache: object = None   # google.cloud.storage.Bucket, cached after first call


def _get_bucket(bucket_name: str | None = None):
    """
    Return a google.cloud.storage.Bucket object.
    Reads bucket name from master.yaml if not provided.
    Returns None (with a warning) if bucket is not configured or GCS is unreachable.
    """
    global _bucket_cache
    if _bucket_cache is not None:
        return _bucket_cache

    name = bucket_name or _bucket_name_from_config()
    if not name:
        log.warning(
            "GCS bucket not configured. Set infrastructure.gcs_bucket in "
            "config/master.yaml to enable GCS music downloads."
        )
        return None

    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(name)
        _bucket_cache = bucket
        log.debug(f"GCS bucket ready: gs://{name}/")
        return bucket
    except Exception as exc:
        log.warning(f"GCS client init failed: {exc}")
        return None


def _bucket_name_from_config() -> str:
    """Read infrastructure.gcs_bucket from config/master.yaml."""
    try:
        from core.config_loader import load_master_infra
        infra = load_master_infra()
        return infra.get("gcs_bucket", "")
    except Exception:
        return os.environ.get("GCS_BUCKET", "")


# -- Public API ----------------------------------------------------------------

def download_file(gcs_path: str, local_path: Path) -> bool:
    """
    Download a file from GCS to local_path.

    Args:
        gcs_path:   path within the bucket, e.g. "music/dark_ambient.mp3"
        local_path: local destination path (parent dirs created if needed)

    Returns:
        True on success, False on any failure (logs warning).
    """
    bucket = _get_bucket()
    if bucket is None:
        return False

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(gcs_path)
        if not blob.exists():
            log.warning(f"GCS: gs://{bucket.name}/{gcs_path} does not exist")
            return False
        blob.download_to_filename(str(local_path))
        size_kb = local_path.stat().st_size // 1024
        log.info(f"GCS: downloaded gs://{bucket.name}/{gcs_path} -> {local_path} ({size_kb} KB)")
        return True
    except Exception as exc:
        log.warning(f"GCS download failed for '{gcs_path}': {exc}")
        return False


def upload_file(local_path: Path, gcs_path: str,
                content_type: str = "audio/mpeg") -> bool:
    """
    Upload a local file to GCS.

    Args:
        local_path:   source file on disk
        gcs_path:     destination path within the bucket, e.g. "music/dark_ambient.mp3"
        content_type: MIME type (default: audio/mpeg for .mp3)

    Returns:
        True on success, False on any failure (logs warning).
    """
    bucket = _get_bucket()
    if bucket is None:
        return False

    if not local_path.exists():
        log.warning(f"GCS upload: local file not found: {local_path}")
        return False

    try:
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path), content_type=content_type)
        size_kb = local_path.stat().st_size // 1024
        log.info(f"GCS: uploaded {local_path} -> gs://{bucket.name}/{gcs_path} ({size_kb} KB)")
        return True
    except Exception as exc:
        log.warning(f"GCS upload failed for '{local_path}' -> '{gcs_path}': {exc}")
        return False


def list_files(prefix: str = "") -> list[str]:
    """
    List all blob names under the given prefix.

    Args:
        prefix: e.g. "music/" to list all music files

    Returns:
        List of blob name strings. Empty list on failure.
    """
    bucket = _get_bucket()
    if bucket is None:
        return []

    try:
        blobs = bucket.list_blobs(prefix=prefix)
        return [b.name for b in blobs]
    except Exception as exc:
        log.warning(f"GCS list failed for prefix='{prefix}': {exc}")
        return []


def file_exists(gcs_path: str) -> bool:
    """Return True if the blob exists in the bucket."""
    bucket = _get_bucket()
    if bucket is None:
        return False
    try:
        return bucket.blob(gcs_path).exists()
    except Exception:
        return False


def reset_client() -> None:
    """Force re-initialisation of the GCS client (useful for tests)."""
    global _bucket_cache
    _bucket_cache = None
