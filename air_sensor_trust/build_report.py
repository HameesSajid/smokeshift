"""Rebuild outputs/report.html from the saved results without refitting models."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from airtrust import config, report  # noqa: E402

if __name__ == "__main__":
    results = json.loads((config.OUT / "results.json").read_text(encoding="utf-8"))
    print(f"Report: {report.build(results)}")
