"""
Utility script to print the entire Streamlit application source.

This script simply streams the contents of app.py to stdout so users can
view or redirect the full code without needing an editor that handles
large files. It intentionally avoids any mutations and stops with a clear
error if the file is missing.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
APP_FILE = ROOT / "app.py"


def main() -> int:
    if not APP_FILE.exists():
        sys.stderr.write("app.py not found in repository root.\n")
        return 1

    try:
        with APP_FILE.open("r", encoding="utf-8") as handle:
            for line in handle:
                sys.stdout.write(line)
    except BrokenPipeError:
        # Allow commands like `python scripts/show_app_code.py | head` to
        # terminate early without surfacing an error to the caller.
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
