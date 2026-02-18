"""FastAPI entrypoint.

This repo uses a ``src/`` layout. If you run commands without installing the
package (``pip install -e .``), Python may not find ``fraud_dashboard``.

This script makes running the API foolproof:

  python api.py

It injects ``./src`` into ``sys.path`` and starts Uvicorn.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    src_str = str(SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Fraud Dashboard FastAPI server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (dev only). This spawns extra processes on Windows.",
    )
    p.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    uvicorn.run(
        "fraud_dashboard.api.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        reload_excludes=[".venv", "__pycache__"],
    )


if __name__ == "__main__":
    main()
