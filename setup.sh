#!/usr/bin/env bash
# GCE VM bootstrap — run once as root after cloning the repo to /home/ubuntu/youtube-automation
set -euo pipefail

REPO_DIR="/home/ubuntu/youtube-automation"
VENV="$REPO_DIR/.venv"
SERVICE="youtube-automation"

echo "=== 1. System packages ==="
apt-get update -qq
apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    ffmpeg \
    fonts-liberation \
    git curl wget

echo "=== 2. Python venv ==="
python3.11 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel
"$VENV/bin/pip" install -r "$REPO_DIR/requirements.txt"

echo "=== 3. Directories & permissions ==="
mkdir -p "$REPO_DIR"/{logs,workspace,data,music,assets,config/credentials}
chown -R ubuntu:ubuntu "$REPO_DIR"

echo "=== 4. .env file ==="
if [ ! -f "$REPO_DIR/.env" ]; then
    cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
    echo "→ Edit $REPO_DIR/.env with your API keys before starting the service"
fi

echo "=== 5. systemd service ==="
cp "$REPO_DIR/systemd/$SERVICE.service" "/etc/systemd/system/$SERVICE.service"
systemctl daemon-reload
systemctl enable "$SERVICE"

echo "=== 6. Sync music beds from GCS ==="
# Run this after setting GOOGLE_CLOUD_PROJECT and GCS_BUCKET in .env
# Uncomment when ready:
# source "$REPO_DIR/.env"
# gsutil -m cp "gs://$GCS_BUCKET_NAME/music/*" "$REPO_DIR/music/"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit $REPO_DIR/.env with your API keys"
echo "  2. Place service_account.json in $REPO_DIR/config/credentials/"
echo "  3. For each channel, run: python scripts/add_channel.py --auth --channel-id <id>"
echo "  4. Start: systemctl start $SERVICE"
echo "  5. Status: systemctl status $SERVICE"
echo "  6. Dashboard (via SSH tunnel): ssh -L 8080:localhost:8080 ubuntu@<vm-ip>"
echo "     Then open: http://localhost:8080"
echo ""
echo "Manual test: python scripts/run_job_now.py --channel horror_stories --series real_horror_shorts --topic 'Test'"
