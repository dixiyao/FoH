"""Shared path constants for task scripts run directly or as modules."""

from __future__ import annotations

import sys
from pathlib import Path


TASKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASKS_DIR.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
