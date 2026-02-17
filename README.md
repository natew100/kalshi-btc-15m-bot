# Kalshi BTC 15m Bot (Standalone VPS Service)

This package is a separate bot runtime intended to live in its own repo on VPS:

- Runtime path: `/root/projects/kalshi-btc-15m-bot`
- Env file: `/etc/kalshi-btc15m.env`
- Data path:
  - `/root/projects/kalshi-btc-15m-bot/data/bot.db`
  - `/root/projects/kalshi-btc-15m-bot/data/status.json`
  - `/root/projects/kalshi-btc-15m-bot/data/sync_state.json`

It integrates with KalshiHQ through existing endpoints:

- `POST /api/sync`
- `POST /api/sync/settle`

## What it does

- Captures Kalshi BTC 15m market microstructure (`series_ticker=KXBTC15M`) at 1s cadence.
- Generates features at `open + 180s` by default.
- Trains `LogisticRegression(L2, C=0.1)` + Platt scaling (`CalibratedClassifierCV`) on rolling 21 days.
- Runs strict paper-first policy with go-live gating.
- Settles outcomes from Kalshi official `market.result`.
- Syncs live state + trades to KalshiHQ.

## Setup

```bash
cd /root/projects/kalshi-btc-15m-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Load env before running:

```bash
set -a
source /etc/kalshi-btc15m.env
set +a
```

## Commands

Runner:

```bash
python -m btc15m_bot.runner
```

Manual retrain:

```bash
python -m btc15m_bot.retrain
```

One-shot sync:

```bash
python -m btc15m_bot.hq_sync --once
```

Daily report:

```bash
python -m btc15m_bot.daily_report
```

## KalshiHQ bot registration

Use `sql/register_bot.sql` after replacing placeholders.

## Systemd

Install:

- `systemd/kalshi-btc15m-runner.service`
- `systemd/kalshi-btc15m-sync.service`

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now kalshi-btc15m-runner.service
sudo systemctl enable --now kalshi-btc15m-sync.service
```
