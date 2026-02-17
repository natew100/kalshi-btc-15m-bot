#!/usr/bin/env bash
set -euo pipefail

ROOT="${BOT_ROOT:-/root/projects/polymarket-btc-5m-bot}"
cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [[ -f /etc/polymarket-btc5m.env ]]; then
  echo "Using /etc/polymarket-btc5m.env"
else
  echo "Create /etc/polymarket-btc5m.env from .env.example before starting services"
fi

sudo cp systemd/pm-btc5m-runner.service /etc/systemd/system/
sudo cp systemd/pm-btc5m-sync.service /etc/systemd/system/
sudo systemctl daemon-reload

echo "Bootstrap complete. Enable services after env setup:"
echo "  sudo systemctl enable --now pm-btc5m-runner.service"
echo "  sudo systemctl enable --now pm-btc5m-sync.service"
