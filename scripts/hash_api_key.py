#!/usr/bin/env python3
import argparse
import hashlib


def main() -> int:
    parser = argparse.ArgumentParser(description="SHA-256 hash helper for KalshiHQ bot API key")
    parser.add_argument("api_key", help="plaintext API key")
    args = parser.parse_args()
    print(hashlib.sha256(args.api_key.encode("utf-8")).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
