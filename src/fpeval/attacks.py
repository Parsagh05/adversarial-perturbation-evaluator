"""Discover, normalize, and validate fixed perturbation bundles."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
import zipfile
from collections.abc import Iterable
from typing import Any

import torch


SCOPE_NAMES = {
    "dataset": "per_dataset", "per_dataset": "per_dataset",
    "category": "per_category", "per_category": "per_category",
    "image": "per_image", "per_image": "per_image",
}
DIRECTION_LABELS = {"normal_to_abnormal": (0, 1), "abnormal_to_normal": (1, 0)}
SETUP_PATTERN = re.compile(r"steps\d+_eps\d+(?:_margin_topk)?(?:_learnable_prompt)?", re.I)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    marker = destination / ".complete"
    if marker.is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            resolved = (destination / member.filename).resolve()
            if root != resolved and root not in resolved.parents:
                raise ValueError(f"Unsafe ZIP member in {archive}: {member.filename}")
        package.extractall(destination)
    marker.touch()


def materialize_input(root: str | Path, cache: str | Path) -> list[Path]:
    """Return every manifest directory from extracted folders and ZIP archives."""
    root = Path(root).expanduser().resolve()
    cache = Path(cache).expanduser().resolve()
    search_roots = [root] if root.is_dir() else []
    archives = [root] if root.is_file() and root.suffix.lower() == ".zip" else []
    if root.is_dir():
        archives.extend(sorted(root.rglob("*.zip")))
    for archive in archives:
        fingerprint = file_sha256(archive)[:16]
        extracted = cache / f"{archive.stem}-{fingerprint}"
        _safe_extract(archive, extracted)
        search_roots.append(extracted)
    found: dict[Path, None] = {}
    for search_root in search_roots:
        for manifest in search_root.rglob("attack_manifest.csv"):
            found[manifest.parent.resolve()] = None
    if not found:
        raise FileNotFoundError(f"No attack_manifest.csv found in {root} or its ZIP files")
    return sorted(found)


def _metadata(bundle: Path) -> tuple[str, str]:
    combined = "/".join(part.lower() for part in bundle.parts)
    prompt_mode = "learnable_prompt" if "learnable_prompt" in combined else "frozen_prompt"
    matches = SETUP_PATTERN.findall(combined)
    setup_id = matches[-1] if matches else "unspecified_setup"
    if setup_id.endswith("_learnable_prompt"):
        setup_id = setup_id[: -len("_learnable_prompt")]
    return prompt_mode, setup_id


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV cannot be empty: {path}")
    return rows


def _field(row: dict[str, str], *names: str, required: bool = True, default: str = "") -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    if required:
        raise ValueError(f"Manifest is missing one of these fields: {names}")
    return default


def _artifact_path(bundle: Path, recorded: str) -> Path:
    parts = PurePosixPath(recorded.replace("\\", "/")).parts
    # Manifests may preserve a generator-side prefix. Prefer a suffix anchored
    # at the portable perturbation/noise directory, then fall back to basename.
    anchors = {"perturbations", "noises", "noise", "deltas"}
    for index, part in enumerate(parts):
        if part.lower() in anchors:
            candidate = bundle.joinpath(*parts[index:])
            if candidate.is_file():
                return candidate
    candidate = bundle.joinpath(*parts)
    if candidate.is_file():
        return candidate
    matches = list(bundle.rglob(Path(recorded).name))
    if len(matches) == 1:
        return matches[0]
    return candidate


@dataclass(frozen=True)
class Attack:
    bundle: Path
    record: dict[str, Any]
    perturbation_path: Path
    evaluation_ids: tuple[str, ...]
    attacked_ids: tuple[str, ...]

    @property
    def condition_id(self) -> str:
        fields = (
            "prompt_mode", "setup_id", "source_dataset", "target_dataset",
            "scope", "category", "direction", "loss_formulation", "loss_mode",
        )
        return "__".join(str(self.record.get(name) or "all") for name in fields)

    def load(self, *, verify_checksum: bool = True) -> tuple[torch.Tensor, dict[str, int]]:
        expected_hash = str(self.record.get("artifact_sha256") or "").lower()
        if verify_checksum and expected_hash:
            actual_hash = file_sha256(self.perturbation_path)
            if actual_hash != expected_hash:
                raise ValueError(f"Checksum mismatch: {self.perturbation_path}")
        try:
            payload = torch.load(self.perturbation_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(self.perturbation_path, map_location="cpu")
        tensor_key = str(self.record["tensor_key"])
        delta = payload.get(tensor_key) if isinstance(payload, dict) else payload
        if not isinstance(delta, torch.Tensor):
            raise TypeError(f"Tensor {tensor_key!r} missing from {self.perturbation_path}")
        if delta.ndim == 3:
            delta = delta[None]
        size = int(self.record["image_size"])
        if delta.ndim != 4 or tuple(delta.shape[1:]) != (3, size, size):
            raise ValueError(f"Unexpected perturbation shape: {tuple(delta.shape)}")
        if not torch.isfinite(delta).all():
            raise ValueError(f"Non-finite perturbation: {self.perturbation_path}")
        epsilon = float(self.record["epsilon"])
        if float(delta.abs().max()) > epsilon + 5e-5:
            raise ValueError(f"Perturbation exceeds epsilon={epsilon}")
        if self.record["scope"] != "per_image":
            if len(delta) != 1:
                raise ValueError("Universal perturbation files must contain one tensor")
            return delta.float().contiguous(), {sample_id: 0 for sample_id in self.attacked_ids}
        ids_key = str(self.record.get("sample_ids_key") or "sample_ids")
        stored = payload.get(ids_key) if isinstance(payload, dict) else None
        if stored is None:
            raise ValueError(f"Per-image payload must contain {ids_key!r}")
        stored_ids = tuple(value.decode() if isinstance(value, bytes) else str(value) for value in stored)
        if len(stored_ids) != len(set(stored_ids)) or set(stored_ids) != set(self.attacked_ids):
            raise ValueError("Per-image sample IDs do not match the fixed attacked cohort")
        index = {sample_id: position for position, sample_id in enumerate(stored_ids)}
        if len(delta) != len(stored_ids):
            raise ValueError("Per-image delta count does not match sample_ids")
        return delta.float().contiguous(), index


def discover_attacks(
    bundles: Iterable[Path], *, scopes: Iterable[str], targets: Iterable[str],
    prompt_modes: Iterable[str] | None = None, setup_ids: Iterable[str] | None = None,
    sources: Iterable[str] | None = None, categories: Iterable[str] | None = None,
    directions: Iterable[str] | None = None, loss_modes: Iterable[str] | None = None,
    loss_formulations: Iterable[str] | None = None,
) -> list[Attack]:
    filters = {
        "scope": set(scopes), "target_dataset": set(targets),
        "prompt_mode": set(prompt_modes) if prompt_modes else None,
        "setup_id": set(setup_ids) if setup_ids else None,
        "source_dataset": set(sources) if sources else None,
        "category": set(categories) if categories else None,
        "direction": set(directions) if directions else None,
        "loss_mode": set(loss_modes) if loss_modes else None,
        "loss_formulation": set(loss_formulations) if loss_formulations else None,
    }
    attacks: list[Attack] = []
    seen: dict[str, Attack] = {}
    for bundle in bundles:
        prompt_mode, setup_id = _metadata(bundle)
        protocol_path = bundle / "evaluation_test_indices.csv"
        if not protocol_path.is_file():
            candidates = list(bundle.parent.rglob("evaluation_test_indices.csv"))
            candidates = [path for path in candidates if "protocol" in path.parts]
            if len(candidates) == 1:
                protocol_path = candidates[0]
        protocol = _read_csv(protocol_path)
        required_protocol = {"protocol_id", "dataset", "category", "label", "partition"}
        if not required_protocol.issubset(protocol[0]):
            raise ValueError(f"Protocol is missing columns: {sorted(required_protocol - set(protocol[0]))}")
        for raw in _read_csv(bundle / "attack_manifest.csv"):
            scope_raw = _field(raw, "scope").lower()
            try:
                scope = SCOPE_NAMES[scope_raw]
            except KeyError as error:
                raise ValueError(f"Unknown scope {scope_raw!r}") from error
            source = _field(raw, "source_dataset")
            target = _field(raw, "target_dataset")
            direction = _field(raw, "direction")
            source_label, target_label = int(_field(raw, "source_label")), int(_field(raw, "target_label"))
            if DIRECTION_LABELS.get(direction) != (source_label, target_label):
                raise ValueError(f"Direction/label mismatch for {direction}")
            category = _field(raw, "category", required=False)
            inferred_formulation = "margin_topk" if "margin_topk" in setup_id else "ce_focal_dice"
            formulation = _field(
                raw, "loss_formulation", required=False, default=inferred_formulation
            )
            loss_mode = _field(raw, "loss_mode", "objective")
            normalized = {
                **raw, "prompt_mode": prompt_mode, "setup_id": setup_id,
                "scope": scope, "source_dataset": source, "target_dataset": target,
                "direction": direction, "source_label": source_label,
                "target_label": target_label, "category": category,
                "loss_formulation": formulation, "loss_mode": loss_mode,
                "image_size": int(_field(raw, "image_size")),
                "epsilon": float(_field(raw, "epsilon")),
                "tensor_key": _field(raw, "noise_tensor_key", "tensor_key", required=False,
                                     default="deltas" if scope == "per_image" else "delta"),
                "sample_ids_key": _field(raw, "sample_ids_key", required=False, default="sample_ids"),
            }
            if any(allowed is not None and str(normalized[name]) not in allowed for name, allowed in filters.items()):
                continue
            cohort = [
                row for row in protocol
                if row["dataset"] == target and row["partition"] == "evaluation"
                and (scope == "per_dataset" or row["category"] == category)
            ]
            if not cohort:
                raise ValueError(f"No evaluation cohort for {target}/{category or 'all'}")
            evaluation_ids = tuple(row["protocol_id"] for row in cohort)
            if len(evaluation_ids) != len(set(evaluation_ids)):
                raise ValueError("Protocol contains duplicate evaluation IDs")
            attacked_ids = tuple(row["protocol_id"] for row in cohort if int(row["label"]) == source_label)
            expected_count = int(_field(raw, "evaluation_attacked_image_count"))
            if len(attacked_ids) != expected_count:
                raise ValueError(f"Attacked cohort count is {len(attacked_ids)}, manifest says {expected_count}")
            path_text = _field(
                raw, "noise_file", "perturbation_file", "perturbation_path",
                "noise_path", "delta_file", "artifact_path",
            )
            attack = Attack(bundle, normalized, _artifact_path(bundle, path_text), evaluation_ids, attacked_ids)
            if not attack.perturbation_path.is_file():
                raise FileNotFoundError(attack.perturbation_path)
            if attack.condition_id in seen:
                previous = seen[attack.condition_id]
                previous_hash = str(previous.record.get("artifact_sha256") or "")
                current_hash = str(attack.record.get("artifact_sha256") or "")
                if previous_hash and previous_hash == current_hash:
                    # A complete Kaggle setup commonly includes both the
                    # extracted scope directory and a portable ZIP of it.
                    continue
                raise ValueError(f"Conflicting duplicate condition: {attack.condition_id}")
            seen[attack.condition_id] = attack
            attacks.append(attack)
    if not attacks:
        raise ValueError("No attack conditions matched the configuration")
    return sorted(attacks, key=lambda item: item.condition_id)
