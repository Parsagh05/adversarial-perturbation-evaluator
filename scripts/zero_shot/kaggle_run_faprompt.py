"""Run FAPrompt evaluation from a Kaggle JSON configuration."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


CONFIG = "/kaggle/working/faprompt.json"


def main() -> None:
    config = Path(os.environ.get("FPEVAL_CONFIG", CONFIG))
    if not config.is_file():
        raise FileNotFoundError(f"Create {config} from configs/faprompt.example.json")
    subprocess.run(
        [sys.executable, "-m", "fpeval", "--config", str(config)], check=True
    )


if __name__ == "__main__":
    main()
