"""Small Kaggle entry point; edit CONFIG or pass FPEVAL_CONFIG."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


CONFIG = "/kaggle/working/anomalyclip.json"


def main() -> None:
    config = Path(os.environ.get("FPEVAL_CONFIG", CONFIG))
    if not config.is_file():
        raise FileNotFoundError(
            f"Create {config} from configs/anomalyclip.example.json first"
        )
    subprocess.run(
        [sys.executable, "-m", "fpeval", "--config", str(config)], check=True
    )


if __name__ == "__main__":
    main()
