"""JSON-configured command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import adapter_names
from .run import expand, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed anomaly perturbations")
    parser.add_argument(
        "--config", help="Path to an evaluation JSON file naming one model or several"
    )
    parser.add_argument("--list-models", action="store_true", help="List registered adapters")
    args = parser.parse_args()
    if args.list_models:
        print("\n".join(adapter_names()))
        return
    if not args.config:
        parser.error("--config is required unless --list-models is used")
    path = Path(args.config).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    for result in run(expand(payload)):
        print(result)


if __name__ == "__main__":
    main()

