"""Step 2 entry. New jobs use --config; historical commands remain compatible."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    if (
        not sys.argv[1:]
        or any(arg.startswith("--config") for arg in sys.argv[1:])
        or sys.argv[1:] in (["--help"], ["-h"])
    ):
        from pipeline.augmentation.cli import main as execute

        execute(default_stage="extend")
    else:
        from pipeline.legacy.extend import main as execute

        execute()


if __name__ == "__main__":
    main()
