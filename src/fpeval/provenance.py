"""Provenance for an evaluation run, written before any inference happens.

Two records: `<output_root>/evaluator_config.json` for the whole invocation
(see `run.py`), and `run_config_<model>.json` in each model's folder for what
only that model knows - its loaded settings, repositories and checkpoints.

The per-model record used to be written only once every metric had been computed,
so a run that crashed left no record of what it had been asked to do. It is now
written at the start with ``status: "running"``, updated as soon as the resolved
model settings are known, and rewritten at the end as ``"completed"`` or
``"failed"``.

Nothing here changes an existing key. `asdict(EvaluationConfig)` and
`resolved_model_settings_by_target` keep their exact former contents and
spelling, because analysis scripts read them (``run_metadata.replicate``, for
one). Everything this module adds is a new top-level key.

Every lookup here is best-effort: a missing git binary, an unreadable
checkpoint or a machine without a GPU records ``null`` rather than failing a run
that would otherwise have produced results.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .attacks import Attack, file_sha256

SCHEMA_VERSION = 1

# Keys of a resolved model setting that may name a file worth hashing.
CHECKPOINT_KEYS = ("checkpoint", "checkpoints", "weights", "weight_path", "ckpt")
# Keys that may name the directory a relative checkpoint hangs off.
ROOT_KEYS = ("repository", "clip_download_root", "download_root", "root")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(repository: Path, *arguments: str) -> str | None:
    try:
        finished = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if finished.returncode != 0:
        return None
    return finished.stdout.strip()


def git_state(repository: Path | str | None) -> dict[str, Any]:
    """Commit and dirty flag for a checkout, or nulls when it is not one."""

    if repository is None:
        return {"path": None, "commit": None, "dirty": None}
    path = Path(repository).expanduser()
    commit = _git(path, "rev-parse", "HEAD") if path.exists() else None
    if commit is None:
        return {"path": str(path), "commit": None, "dirty": None}
    status = _git(path, "status", "--porcelain")
    return {"path": str(path), "commit": commit, "dirty": bool(status)}


def _hashed(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            return {"path": str(path), "sha256": file_sha256(path)}
    except OSError:
        pass
    return {"path": str(path), "sha256": None}


def _checkpoints(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Every checkpoint a resolved setting names, hashed where it resolves.

    The paths are already recorded; what was missing is the hash, which is what
    actually pins the weights a number was produced with.
    """

    roots = [Path(str(settings[key])).expanduser()
             for key in ROOT_KEYS if settings.get(key)]
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in CHECKPOINT_KEYS:
        value = settings.get(key)
        if not value:
            continue
        for name in (value if isinstance(value, (list, tuple)) else [value]):
            if not isinstance(name, (str, Path)):
                continue
            candidate = Path(str(name)).expanduser()
            options = [candidate] if candidate.is_absolute() else [
                root / candidate for root in roots
            ] + [candidate]
            resolved = next((option for option in options if option.is_file()), options[0])
            if str(resolved) in seen:
                continue
            seen.add(str(resolved))
            found.append({"setting": key, **_hashed(resolved)})
    return found


def environment() -> dict[str, Any]:
    gpu_name = None
    torch_version = None
    cuda_version = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - provenance must never fail a run
        pass
    return {
        "hostname": socket.gethostname(),
        "gpu_name": gpu_name,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "python_version": platform.python_version(),
    }


def code(resolved_model_settings: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The evaluator's own commit, its version, and each adapter's upstream."""

    try:
        from . import __version__ as fpeval_version
    except ImportError:
        fpeval_version = None
    payload: dict[str, Any] = {
        "evaluator": git_state(Path(__file__).resolve().parent.parent.parent),
        "fpeval_version": fpeval_version,
        "models": {},
    }
    for target, settings in (resolved_model_settings or {}).items():
        if not isinstance(settings, dict):
            continue
        payload["models"][target] = {
            "repository": git_state(settings.get("repository")),
            "checkpoints": _checkpoints(settings),
        }
    return payload


def invocation() -> dict[str, Any]:
    """The command line, and the config file it named if it named one."""

    argv = list(sys.argv)
    config_path: str | None = None
    for index, argument in enumerate(argv):
        if argument == "--config" and index + 1 < len(argv):
            config_path = argv[index + 1]
        elif argument.startswith("--config="):
            config_path = argument.split("=", 1)[1]
    payload: dict[str, Any] = {"argv": argv, "config_file": None}
    if config_path:
        payload["config_file"] = _hashed(Path(config_path).expanduser())
    return payload


def _bundle_file(bundle: Path, name: str) -> dict[str, Any] | None:
    """A file beside the bundle, hashed. Bundles nest, so search upward too."""

    direct = bundle / name
    if direct.is_file():
        return _hashed(direct)
    for parent in list(bundle.parents)[:4]:
        candidate = parent / name
        if candidate.is_file():
            return _hashed(candidate)
    return None


def attacks(evaluated: list[Attack]) -> list[dict[str, Any]]:
    """One entry per evaluated bundle, linking results to the attack settings.

    `attack_manifest.csv` pins the perturbations; `generation_config.json`, when
    the generator writes one, pins the settings that produced them.
    """

    by_bundle: dict[str, dict[str, Any]] = {}
    for attack in evaluated:
        bundle = Path(attack.bundle)
        key = str(bundle)
        entry = by_bundle.get(key)
        if entry is None:
            entry = {
                "bundle": key,
                "setup_ids": [],
                "scopes": [],
                "prompt_modes": [],
                "attack_manifest": _bundle_file(bundle, "attack_manifest.csv"),
                "generation_config": _bundle_file(bundle, "generation_config.json"),
                "conditions": 0,
            }
            by_bundle[key] = entry
        entry["conditions"] += 1
        for name, field in (("setup_id", "setup_ids"), ("scope", "scopes"),
                            ("prompt_mode", "prompt_modes")):
            value = attack.record.get(name)
            if value is not None and str(value) not in entry[field]:
                entry[field].append(str(value))
    return list(by_bundle.values())


# The parts of the generator's generation_config.json that say how the attack
# was set up. Its code, data and environment sections describe the generator's
# own run, not the setup, and stay pinned by the file's hash.
GENERATION_SECTIONS = ("setup", "hyperparameters", "execution")
# A setting with more distinct values than this across one bundle is a
# per-condition identifier (a file name, a hash), not a setting to read here;
# manifest_snapshot.json keeps every value.
DISTINCT_LIMIT = 10


def _manifest_settings(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The bundle's manifest columns: one value, one per direction, or several."""

    constant: dict[str, Any] = {}
    varying: dict[str, Any] = {}
    for column in sorted({key for record in records for key in record}):
        values = sorted({str(record.get(column, "")) for record in records})
        if len(values) == 1:
            constant[column] = values[0]
            continue
        # Most settings that vary inside a bundle vary by attack direction (the
        # margin top-k fraction, the source and target labels), and a flat list
        # of values would lose which direction had which.
        by_direction: dict[str, set[str]] = {}
        for record in records:
            by_direction.setdefault(str(record.get("direction", "")), set()).add(
                str(record.get(column, ""))
            )
        if column != "direction" and all(len(found) == 1 for found in by_direction.values()):
            varying[column] = {"by_direction": {
                direction: next(iter(found)) for direction, found in sorted(by_direction.items())
            }}
        elif len(values) <= DISTINCT_LIMIT:
            varying[column] = {"values": values}
        else:
            varying[column] = {"distinct_values": len(values)}
    return {"constant": constant, "varying": varying}


def attack_settings(evaluated: list[Attack]) -> list[dict[str, Any]]:
    """How every evaluated bundle was set up, in the generator's own words.

    Copied, never re-derived: the generation_config.json sections when the
    bundle has one (every bundle generated since it was introduced), and the
    attack manifest's columns for every bundle, older ones included.
    """

    grouped: dict[str, list[Attack]] = {}
    for attack in evaluated:
        grouped.setdefault(str(Path(attack.bundle)), []).append(attack)
    entries = []
    for bundle, members in grouped.items():
        generation = _bundle_file(Path(bundle), "generation_config.json")
        if generation is not None:
            try:
                raw = json.loads(Path(generation["path"]).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raw = None
            if isinstance(raw, dict):
                generation.update({section: raw.get(section) for section in GENERATION_SECTIONS})
            else:
                generation["unreadable"] = True
        entries.append({
            "bundle": bundle,
            "conditions": len(members),
            "generation_config": generation,
            "manifest": _manifest_settings([attack.record for attack in members]),
        })
    return entries


def cohort(image_rows: list[dict[str, Any]],
           manifest_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Split hashes and image counts, per target, partition and label."""

    splits: dict[str, list[str]] = {}
    for record in manifest_records:
        target = str(record.get("target_dataset") or "")
        digest = str(record.get("protocol_split_sha256") or "")
        if not target or not digest:
            continue
        splits.setdefault(target, [])
        if digest not in splits[target]:
            splits[target].append(digest)

    counts: dict[str, dict[str, dict[str, int]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in image_rows:
        target = str(row.get("target_dataset") or "")
        partition = str(row.get("partition") or "evaluation")
        identifier = str(row.get("protocol_id") or "")
        if (target, partition, identifier) in seen:
            continue
        seen.add((target, partition, identifier))
        label = "abnormal" if int(row.get("label") or 0) == 1 else "normal"
        bucket = counts.setdefault(target, {}).setdefault(partition, {})
        bucket[label] = bucket.get(label, 0) + 1
    for target, partitions in counts.items():
        for partition, labels in partitions.items():
            labels["total"] = sum(value for key, value in labels.items() if key != "total")
    return {"protocol_split_sha256": splits, "images": counts}


class RunRecord:
    """The run_config_<model>.json writer: start, update, finish - always atomically."""

    def __init__(self, path: Path, base: dict[str, Any]) -> None:
        self.path = Path(path)
        # `base` is asdict(config): every existing key, untouched.
        self.payload: dict[str, Any] = dict(base)
        self.payload.setdefault("resolved_model_settings_by_target", {})
        self.payload.update({
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "created_at_utc": _utc_now(),
            "finished_at_utc": None,
            **environment(),
            "code": code(),
            "invocation": invocation(),
            "attacks": [],
            "cohort": {},
        })

    def update(self, **values: Any) -> None:
        self.payload.update(values)
        self.write()

    def resolved(self, settings: dict[str, dict[str, Any]]) -> None:
        """Called once the models are loaded; only then is this knowable."""

        self.payload["resolved_model_settings_by_target"] = settings
        self.payload["code"] = code(settings)
        self.write()

    def finish(self, status: str, error: str | None = None) -> None:
        self.payload["status"] = status
        self.payload["finished_at_utc"] = _utc_now()
        if error is not None:
            self.payload["error"] = error
        self.write()

    def write(self) -> None:
        write_json(self.path, self.payload)


def write_json(path: Path, payload: Any, *, default=None) -> None:
    """Atomic write, so a crash mid-write cannot truncate an existing record."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=default or str), encoding="utf-8"
    )
    os.replace(temporary, path)
