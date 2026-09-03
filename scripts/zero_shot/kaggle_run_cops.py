"""Run cops evaluation from a Kaggle JSON configuration."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


CONFIG = "/kaggle/working/cops.json"


def main() -> None:
    config = Path(os.environ.get("FPEVAL_CONFIG", CONFIG))
    if not config.is_file():
        raise FileNotFoundError(f"Create {config} from configs/cops.example.json")
    subprocess.run(
        [sys.executable, "-m", "fpeval", "--config", str(config)], check=True
    )


if __name__ == "__main__":
    main()
