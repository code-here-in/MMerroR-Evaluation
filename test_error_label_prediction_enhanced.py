"""Compatibility alias for older automation that still invokes this filename.

Prefer ``run.py`` or ``python -m mmerror_eval`` for new workflows.
"""

from mmerror_eval.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
