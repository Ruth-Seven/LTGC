"""CLI for CLIP-guided Flux replacement."""
import argparse
from pathlib import Path
from .configuration import RefinementConfig
from .storage import RefinementPaths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["plan", "run", "worker"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--round", type=int)
    options = parser.parse_args()
    config = RefinementConfig.load(options.config)
    if options.operation == "plan":
        from .pipeline import plan
        print(plan(config))
    elif options.operation == "run":
        from .pipeline import run
        print(run(Path(options.config).resolve(), config))
    else:
        from .worker import run_worker
        run_worker(config, RefinementPaths(Path(config.pipeline.run_dir)),
                   options.gpu, options.round)


if __name__ == "__main__":
    main()
