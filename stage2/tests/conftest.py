"""Make the pipeline modules importable when running `python -m pytest tests`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
