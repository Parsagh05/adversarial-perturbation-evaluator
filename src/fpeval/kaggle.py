"""Kaggle mount discovery helpers used by the self-contained notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


# Kept in step with attacks.SETUP_PATTERN; see the grammar note there.
SETUP_PATTERN = re.compile(
    r"steps\d+(?:_cat\d+_img\d+)?_eps[\dp]+"
    r"(?:_margin_topk)?(?:_train[\dp]+)?(?:_learnable_prompt)?",
    re.IGNORECASE,
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


def _selected_attack_root(value: str | Path) -> Path:
    selected = Path(value).expanduser().resolve()
    if selected.name == "setups":
        selected = selected.parent
    setups = selected / "setups"
    if not setups.is_dir() or not any(
        (setups / mode).is_dir() for mode in ("frozen_prompt", "learnable_prompt")
    ):
        raise ValueError(
            "Selected perturbation input must contain setups/frozen_prompt and/or "
            f"setups/learnable_prompt: {selected}"
        )
    return selected


def _selected_mvtec_root(value: str | Path) -> Path:
    selected = Path(value).expanduser().resolve()
    if not selected.is_dir() or not any(selected.glob("*/test/good")):
        raise ValueError(
            "Selected MVTec input must contain <category>/test/good: "
            f"{selected}"
        )
    return selected


def _selected_visa_root(value: str | Path) -> Path:
    selected = Path(value).expanduser().resolve()
    manifests = (
        selected / "split_csv" / "1cls.csv",
        selected / "split_csv" / "1cls.csv.csv",
        selected / "1cls.csv",
    )
    if not selected.is_dir() or not any(path.is_file() for path in manifests):
        raise ValueError(
            "Selected VisA input must contain split_csv/1cls.csv: "
            f"{selected}"
        )
    return selected


def discover_kaggle_inputs(
    root: str | Path = "/kaggle/input",
    *,
    mvtec_root: str | Path | None = None,
    visa_root: str | Path | None = None,
    attacks_root: str | Path | None = None,
) -> KaggleInputs:
    """Discover target datasets and validate an optional selected attack root."""

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Kaggle input root not found: {root}")
    return KaggleInputs(
        mvtec_root=(
            _selected_mvtec_root(mvtec_root)
            if mvtec_root is not None
            else _unique(_mvtec_candidates(root), "MVTec AD")
        ),
        visa_root=(
            _selected_visa_root(visa_root)
            if visa_root is not None
            else _unique(_visa_candidates(root), "VisA")
        ),
        attacks_root=(
            _selected_attack_root(attacks_root)
            if attacks_root is not None
            else _unique(_attack_candidates(root), "perturbation dataset")
        ),
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


def download_kaggle_dataset(
    reference: str, *, download_root: str | Path | None = None
) -> Path:
    """Return a local directory holding the unpacked public Kaggle dataset.

    Checkpoints that live on Kaggle rather than Drive or HuggingFace are fetched
    through this one entry point so no adapter needs a manual download step. A
    dataset already attached to a Kaggle session is preferred over re-fetching
    it, which is both faster and works when the session has no network.

    Off Kaggle, ``kagglehub`` needs credentials: either ``~/.kaggle/kaggle.json``
    or the ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` environment variables.
    """

    slug = reference.split("/")[-1]
    mounted = Path("/kaggle/input")
    if mounted.is_dir():
        attached = mounted / slug
        if attached.is_dir():
            return attached
    if download_root:
        cached = Path(download_root).expanduser().resolve() / slug
        if cached.is_dir() and any(cached.rglob("*")):
            return cached
    try:
        import kagglehub
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            f"Fetching {reference} needs kagglehub. Install it, attach the "
            f"dataset to the session, or pass the checkpoint paths explicitly."
        ) from error
    return Path(kagglehub.dataset_download(reference))


def find_kaggle_files(root: str | Path, pattern: str) -> list[Path]:
    """Every file below ``root`` matching a glob, in a stable order.

    Kaggle unpacks a dataset under a version directory whose name is not part of
    the reference, so callers search rather than assume a layout.
    """

    return sorted({path.resolve() for path in Path(root).rglob(pattern) if path.is_file()})
