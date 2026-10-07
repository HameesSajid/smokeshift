"""Command-line entry point.

    python run.py                 list the stages
    python run.py panel naive     run the named stages in order
    python run.py all             run every stage
"""
import functools
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*(empty slice|All-NaN|Degrees of freedom).*")
sys.path.insert(0, str(Path(__file__).parent / "src"))
print = functools.partial(print, flush=True)


def main():
    """Parse the stage names from the command line and run them in order."""
    from smokeshift.pipeline import STAGES
    wanted = sys.argv[1:]
    if not wanted:
        print(__doc__)
        for name, (title, _) in STAGES.items():
            print(f"  {name:<12} {title}")
        return
    if wanted == ["all"]:
        wanted = list(STAGES)
    unknown = [w for w in wanted if w not in STAGES]
    if unknown:
        sys.exit(f"Unknown stage(s): {', '.join(unknown)}")
    for name in wanted:
        title, fn = STAGES[name]
        t = time.time()
        print(f"[{name}] {title}")
        fn(log=print)
        print(f"[{name}] done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
