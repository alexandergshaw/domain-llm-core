"""conftest.py – Add repo root to sys.path so tests can import from src/."""
import sys
from pathlib import Path

# Make `src` importable as `src.model`, `src.dataset`, etc.
sys.path.insert(0, str(Path(__file__).parent.parent))
