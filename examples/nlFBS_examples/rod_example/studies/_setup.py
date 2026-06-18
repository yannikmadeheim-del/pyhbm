"""Shared bootstrap for the rod example scripts.

Puts ``src/`` on the path and makes stdout UTF-8 / line-buffered so the
continuation's live "Δω" prints do not crash under Windows cp1252 and appear
immediately rather than block-buffered.  Import this FIRST in any entry script::

    import _setup  # noqa: F401
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:                                    # only meaningful for a real console
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass
