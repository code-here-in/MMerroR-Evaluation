"""Legacy entrypoint for the publish-ready MMErroR evaluation bundle.

This wrapper keeps the original filename usable while delegating execution to
the refactored config-driven runner in ``mmerror_eval_release``.
"""

from pathlib import Path
import sys


RELEASE_ROOT = Path(__file__).resolve().parent / "mmerror_eval_release"
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from mmerror_eval.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
