"""Streamlit entrypoint.

This repository uses a ``src/`` layout.

When you run Streamlit directly (without ``pip install -e .``), Python does
not automatically include ``./src`` on the import path. That can cause:

    ModuleNotFoundError: No module named 'fraud_dashboard'

We defensively add ``./src`` to ``sys.path`` so the UI runs smoothly even in
"quick demo" mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    src_str = str(SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def main() -> None:
    # Local import avoids ruff E402 while still supporting "quick demo" mode.
    from fraud_dashboard.ui.app import main as ui_main

    ui_main()


if __name__ == "__main__":
    main()
