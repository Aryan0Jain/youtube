"""
scripts/upload_music.py -- Upload local music files to Google Cloud Storage.

Run this once after placing .mp3 files in the music/ directory.
The pipeline will automatically download music from GCS when needed,
so you can delete the local files after upload to save disk space.

Usage:
  python scripts/upload_music.py              # upload all music/*.mp3 to GCS
  python scripts/upload_music.py --list       # list music files already in GCS
  python scripts/upload_music.py --delete-local  # delete local files after upload
  python scripts/upload_music.py --download   # download all GCS music files locally

GCS path: gs://<bucket>/music/<filename>
Bucket:   read from config/master.yaml -> infrastructure.gcs_bucket

Expected files (from config/niches.yaml):
  dark_ambient.mp3           (horror)
  cinematic_orchestral.mp3   (what_if)
  high_energy_electronic.mp3 (shock_facts)
  playful_upbeat.mp3         (quiz)
  epic_buildup.mp3           (ranking)
  neutral_corporate.mp3      (historical_versus)
  investigative_jazz.mp3     (myth_busting)
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
log = logging.getLogger("upload_music")

BASE_DIR = Path(__file__).parent.parent
MUSIC_DIR = BASE_DIR / "music"

# Files expected per niches.yaml
EXPECTED_FILES = [
    "dark_ambient.mp3",
    "cinematic_orchestral.mp3",
    "high_energy_electronic.mp3",
    "playful_upbeat.mp3",
    "epic_buildup.mp3",
    "neutral_corporate.mp3",
    "investigative_jazz.mp3",
]


def _get_gcs_bucket_name() -> str:
    from core.config_loader import load_master_infra
    infra = load_master_infra(BASE_DIR)
    name = infra.get("gcs_bucket", "")
    if not name:
        log.error(
            "GCS bucket not configured. Set infrastructure.gcs_bucket in "
            "config/master.yaml"
        )
        sys.exit(1)
    return name


def cmd_upload(delete_local: bool = False) -> None:
    """Upload all music/*.mp3 files to GCS."""
    from integrations.gcs_client import upload_file, reset_client
    reset_client()

    bucket_name = _get_gcs_bucket_name()
    log.info(f"Target bucket: gs://{bucket_name}/music/")

    mp3_files = sorted(MUSIC_DIR.glob("*.mp3"))
    if not mp3_files:
        log.error(
            f"No .mp3 files found in {MUSIC_DIR}/\n"
            "Download royalty-free tracks and place them there first.\n"
            "Good sources:\n"
            "  https://pixabay.com/music/\n"
            "  https://freemusicarchive.org/\n"
            "  https://incompetech.com/music/royalty-free/"
        )
        sys.exit(1)

    log.info(f"Found {len(mp3_files)} file(s) to upload:")
    for f in mp3_files:
        log.info(f"  {f.name}  ({f.stat().st_size // 1024} KB)")

    print()
    uploaded, failed = 0, 0

    for mp3 in mp3_files:
        gcs_path = f"music/{mp3.name}"
        ok = upload_file(mp3, gcs_path, content_type="audio/mpeg")
        if ok:
            uploaded += 1
            if delete_local:
                mp3.unlink()
                log.info(f"  deleted local: {mp3.name}")
        else:
            failed += 1

    print()
    log.info(f"Upload complete: {uploaded} succeeded, {failed} failed")

    if failed == 0:
        log.info("All music files are now in GCS.")
        log.info("The pipeline will download them automatically when needed.")
        if not delete_local:
            log.info(
                "To free local disk space, re-run with --delete-local, or "
                "manually delete files from music/"
            )

    # Show which expected files are still missing from GCS
    from integrations.gcs_client import file_exists
    missing = [f for f in EXPECTED_FILES if not file_exists(f"music/{f}")]
    if missing:
        log.warning(f"Still missing from GCS ({len(missing)} file(s)):")
        for m in missing:
            log.warning(f"  music/{m}")


def cmd_list() -> None:
    """List music files currently in GCS."""
    from integrations.gcs_client import list_files, reset_client
    reset_client()

    bucket_name = _get_gcs_bucket_name()
    blobs = list_files("music/")

    music_blobs = [b for b in blobs if b.startswith("music/") and b.endswith(".mp3")]

    if not music_blobs:
        log.info(f"No music files found in gs://{bucket_name}/music/")
        log.info("Run without --list to upload files.")
        return

    log.info(f"Music files in gs://{bucket_name}/music/ ({len(music_blobs)} file(s)):")
    for b in sorted(music_blobs):
        local_path = BASE_DIR / b
        cached = " [cached locally]" if local_path.exists() else ""
        log.info(f"  gs://{bucket_name}/{b}{cached}")

    # Show which expected files are missing
    gcs_names = {b.removeprefix("music/") for b in music_blobs}
    missing = [f for f in EXPECTED_FILES if f not in gcs_names]
    if missing:
        log.warning(f"\nExpected but missing from GCS ({len(missing)}):")
        for m in missing:
            log.warning(f"  {m}")
    else:
        log.info("\nAll 7 expected music files are present in GCS.")


def cmd_download() -> None:
    """Download all GCS music files to local music/ directory."""
    from integrations.gcs_client import list_files, download_file, reset_client
    reset_client()

    bucket_name = _get_gcs_bucket_name()
    blobs = list_files("music/")
    music_blobs = [b for b in blobs if b.startswith("music/") and b.endswith(".mp3")]

    if not music_blobs:
        log.error(f"No music files in gs://{bucket_name}/music/ -- upload first.")
        sys.exit(1)

    MUSIC_DIR.mkdir(exist_ok=True)
    log.info(f"Downloading {len(music_blobs)} file(s) from gs://{bucket_name}/music/...")

    downloaded, failed = 0, 0
    for gcs_path in sorted(music_blobs):
        filename = gcs_path.split("/")[-1]
        local_path = MUSIC_DIR / filename
        if local_path.exists():
            log.info(f"  {filename} already cached locally -- skip")
            downloaded += 1
            continue
        ok = download_file(gcs_path, local_path)
        if ok:
            downloaded += 1
        else:
            failed += 1

    log.info(f"Download complete: {downloaded} files ready, {failed} failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage music files in Google Cloud Storage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List music files already in GCS",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download all GCS music files to local music/ directory",
    )
    parser.add_argument(
        "--delete-local", action="store_true",
        help="Delete local files after successful upload (save disk space)",
    )
    args = parser.parse_args()

    if args.list:
        cmd_list()
    elif args.download:
        cmd_download()
    else:
        cmd_upload(delete_local=args.delete_local)


if __name__ == "__main__":
    main()
