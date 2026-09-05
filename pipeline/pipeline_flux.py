"""Compatibility facade. Implementation lives in pipeline.augmentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.augmentation.compat import extend, generate, merge, sample
from pipeline.augmentation.references import prepare_reference

if __name__ == "__main__":
    if any(arg.startswith("--config") for arg in sys.argv[1:]):
        from pipeline.augmentation.cli import main
    else:
        from pipeline.augmentation.compat import main
    main()
