"""Kaggle mount discovery helpers used by the self-contained notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SETUP_PATTERN = re.compile(
    r"steps\d+_eps\d+(?:_margin_topk)?(?:_learnable_prompt)?", re.IGNORECASE
)


@dataclass(frozen=True)
class KaggleInputs:
    mvtec_root: Path
    visa_root: Path
    attacks_root: Path


def _unique(paths: set[Path], label: str) -> Path:
    resolved = sorted({path.resolve() for path in paths if path.is_dir()})
    if len(resolved) != 1:
        formatted = "\n".join(f"  - {path}" for path in resolved) or "  (none)"
        raise RuntimeError(
            f"Expected exactly one mounted {label} root, found {len(resolved)}:\n"
            f"{formatted}"
        )
    return resolved[0]


def _mvtec_candidates(root: Path) -> set[Path]:
    candidates = {
        path
        for path in root.rglob("mvtec_anomaly_detection")
        if path.is_dir() and any(path.glob("*/test/good"))
    }
    if candidates:
        return candidates
    # Support differently named Kaggle datasets while retaining structural
    # validation: <root>/<category>/test/good.
    return {
        good.parents[2]
        for good in root.rglob("good")
        if good.is_dir()
        and good.parent.name == "test"
        and (good.parents[1] / "ground_truth").is_dir()
    }


def _visa_candidates(root: Path) -> set[Path]:
    candidates: set[Path] = set()
    for manifest in root.rglob("1cls.csv"):
        if manifest.parent.name == "split_csv":
            candidates.add(manifest.parent.parent)
        else:
            candidates.add(manifest.parent)
    for manifest in root.rglob("1cls.csv.csv"):
        if manifest.parent.name == "split_csv":
            candidates.add(manifest.parent.parent)
    return candidates


def _attack_candidates(root: Path) -> set[Path]:
    return {
        setups.parent
        for setups in root.rglob("setups")
        if setups.is_dir()
        and any((setups / mode).is_dir() for mode in ("frozen_prompt", "learnable_prompt"))
    }


def discover_kaggle_inputs(root: str | Path = "/kaggle/input") -> KaggleInputs:
    """Discover the three required mounted datasets by their structure."""

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Kaggle input root not found: {root}")
    return KaggleInputs(
        mvtec_root=_unique(_mvtec_candidates(root), "MVTec AD"),
        visa_root=_unique(_visa_candidates(root), "VisA"),
        attacks_root=_unique(_attack_candidates(root), "perturbation dataset"),
    )


def inventory_attack_setups(attacks_root: str | Path) -> dict[str, dict[str, str]]:
    """Map prompt mode and normalized setup ID to its on-disk directory name."""

    setups = Path(attacks_root).expanduser().resolve() / "setups"
    if not setups.is_dir():
        raise FileNotFoundError(f"Attack setups directory not found: {setups}")
    inventory: dict[str, dict[str, str]] = {}
    for prompt_mode in ("frozen_prompt", "learnable_prompt"):
        mode_root = setups / prompt_mode
        if not mode_root.is_dir():
            continue
        mode_setups: dict[str, str] = {}
        for directory in sorted(path for path in mode_root.iterdir() if path.is_dir()):
            matches = SETUP_PATTERN.findall(directory.name)
            setup_id = matches[-1].lower() if matches else directory.name
            if setup_id.endswith("_learnable_prompt"):
                setup_id = setup_id[: -len("_learnable_prompt")]
            if setup_id in mode_setups:
                raise ValueError(
                    f"Duplicate normalized setup {setup_id!r} below {mode_root}"
                )
            mode_setups[setup_id] = directory.name
        if mode_setups:
            inventory[prompt_mode] = mode_setups
    if not inventory:
        raise ValueError(f"No frozen or learnable attack setups found below {setups}")
    return inventory

