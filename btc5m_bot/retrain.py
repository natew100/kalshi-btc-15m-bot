from __future__ import annotations

import sys

from . import db
from .config import ensure_runtime_paths, load_settings
from .modeling import train_model


def main() -> int:
    settings = load_settings()
    ensure_runtime_paths(settings)
    conn = db.connect_db(settings.db_path)
    db.init_db(conn)
    ok, msg = train_model(conn, settings)
    print(msg, file=sys.stdout if ok else sys.stderr)
    conn.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
