-- Register Kalshi BTC 15m bot in KalshiHQ.
-- Replace plaintext API key below with your real secret.

-- Ensure pgcrypto is available for digest().
create extension if not exists pgcrypto;

insert into kalshi_bots (
  name,
  display_name,
  bot_type,
  status,
  config,
  api_key_hash
) values (
  'kalshi-btc-15m-v1',
  'Kalshi BTC 15m v1',
  'crypto_5m',
  'offline',
  jsonb_build_object(
    'series_ticker', 'KXBTC15M',
    'mode', 'paper',
    'default_cost_cents', 1.5,
    'min_edge_cents', 1.8,
    'max_spread_cents', 2.0,
    'decision_offset_seconds', 180,
    'paper_days_required', 7
  ),
  encode(digest('REPLACE_WITH_REAL_API_KEY', 'sha256'), 'hex')
)
on conflict (name)
do update set
  display_name = excluded.display_name,
  bot_type = excluded.bot_type,
  config = excluded.config,
  api_key_hash = excluded.api_key_hash,
  updated_at = now();
