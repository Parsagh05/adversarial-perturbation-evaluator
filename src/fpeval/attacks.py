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
    "cross_dataset": "cross_dataset",
    "category": "per_category", "per_category": "per_category",
    "image": "per_image", "per_image": "per_image",
}
# Scopes whose perturbation covers a whole dataset rather than one category, so
# the evaluation cohort is every category instead of one. Under the balanced
# protocol the generator delivers the same delta to both, split only by whether
# the evaluation dataset is the source dataset; under the full protocol
# cross-dataset optimises its own delta over the complete source dataset. Either
# way each bundle names its own perturbation file, so only the cohort matters
# here.
DATASET_LEVEL_SCOPES = frozenset({"per_dataset", "cross_dataset"})
DIRECTION_LABELS = {"normal_to_abnormal": (0, 1), "abnormal_to_normal": (1, 0)}
# ep{E}[_cat{C}_img{I}]_eps{E}[_ce_focal_dice][_full]
# [_fullcross|_halfcross][_train{P}][_learnable_prompt],
# matching setup_catalog.compose_setup_id. Every number may be fractional, with
# "p" for the decimal point (ep7p14, eps0p02, train12p5).
#
# The generator budgets each scope in epochs and derives its own step count from
# the training-set size, so the name carries the budget rather than the steps.
# margin_topk is the default objective and adds no component; ce_focal_dice
# names itself. Two spellings are historical and still appear in bundles and
# results produced earlier - steps{N} for the budget, and "_margin_topk" from
# when that was the named case rather than the default - so both are accepted.
# The _cat/_img pair is absent whenever the three scopes agree, and "_full"
# marks the split protocol that keeps every test image; balanced adds nothing.
#
# Every optional component must be matched. Missing one does not fail loudly: a
# partly matched name is truncated to the part that did match, so a "_full" run
# would carry the same setup_id as the balanced run it must stay separate from.
_NUMBER = r"\d+(?:p\d+)?"
SETUP_PATTERN = re.compile(
    # Budget: per-dataset, then the cross, per-category and per-image budgets
    # wherever they differ from it. steps{N} is the pre-budget spelling.
    rf"(?:ep{_NUMBER}|steps\d+)"
    rf"(?:_cross{_NUMBER})?"
    rf"(?:_cat{_NUMBER}_img{_NUMBER})?"
    rf"_eps{_NUMBER}"
    # Objective: ce_focal_dice names itself; margin_topk is the default and
    # names only its displacement hinge. _margin_topk is the pre-flip spelling.
    rf"(?:_ce_focal_dice|_margin_topk|_hinge{_NUMBER})?"
    rf"(?:_mom{_NUMBER})?"
    rf"(?:_(?:linear|cosine)_step)?"
    rf"(?:_best)?"
    # "_full" is the split protocol and "_fullcross" the cross data mode; the
    # lookahead stops the protocol eating the front of the data mode, which
    # would truncate everything after it off the name.
    rf"(?:_full(?!cross))?(?:_fullcross|_halfcross)?"
    rf"(?:_train{_NUMBER})?(?:_learnable_prompt)?",
    re.I,
)


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


def _formulation_from_setup_id(setup_id: str) -> str:
    """Which objective a setup ID implies, for manifests that omit the column.

    Only one of the two formulations is ever named, and which one flipped: the
    generator now treats margin_topk as the default and names ce_focal_dice,
    having previously done the reverse. The budget spelling dates the ID -
    "steps{N}" predates both changes, "ep{E}" follows them - so the default a
    bare name implies is unambiguous and old bundles keep their old reading.
    """
    lowered = setup_id.lower()
    if "ce_focal_dice" in lowered:
        return "ce_focal_dice"
    if "margin_topk" in lowered:
        return "margin_topk"
    return "ce_focal_dice" if lowered.startswith("steps") else "margin_topk"


# The generator groups its tree as
# setups/<settings>/<scope>/ep<budget>/<frozen|learnable>/, where <settings> is
# the setup ID minus its leading epochs component and the prompt-family suffix,
# and each scope directory carries only the budget that scope spends.
_BUDGET_TAG = re.compile(
    rf"^(?:ep{_NUMBER}|steps\d+)(?:_cross{_NUMBER})?"
    rf"(?:_cat{_NUMBER}_img{_NUMBER})?",
    re.I,
)
# Which number of the budget each scope actually spends.
_SCOPE_BUDGET = {
    "per_dataset": "ep", "cross_dataset": "cross",
    "per_category": "cat", "per_image": "img",
}


def split_setup_id(setup_id: str) -> tuple[str, str]:
    """``(budget tag, settings)``, mirroring the generator's own split.

    An ID the pattern cannot read yields an empty budget and itself as the
    settings, which keeps an unrecognised name in one piece rather than
    silently reshaping the tree around a half-parse.
    """
    match = _BUDGET_TAG.match(setup_id)
    if not match or not match.group():
        return "", setup_id
    return match.group(), setup_id[len(match.group()):].lstrip("_")


def scope_budget_tag(setup_id: str, scope: str) -> str:
    """The ``ep{N}`` a scope spends, read off the ID's budget component.

    The budget names a scope only where it differs from the per-dataset one, so
    an absent component means that scope shares it.
    """
    budget, _ = split_setup_id(setup_id)
    if not budget:
        return ""
    parts = budget.split("_")
    wanted = _SCOPE_BUDGET.get(scope, "ep")
    for part in parts[1:]:
        if part.lower().startswith(wanted):
            return "ep" + part[len(wanted):]
    return parts[0] if parts[0].lower().startswith("ep") else ""


PROMPT_FAMILY_DIRECTORIES = frozenset({"frozen_prompt", "learnable_prompt"})
_SCOPE_DIRECTORIES = frozenset(_SCOPE_BUDGET)
_BUDGET_DIRECTORY = re.compile(rf"^ep{_NUMBER}$", re.I)


def layout_from_bundle(bundle: Path) -> tuple[str, str] | None:
    """``(settings, scope budget)`` when the bundle sits in the grouped tree.

    Read from the path rather than derived from the setup ID because the two
    can disagree: under halfcross the cross scope reuses the per-dataset delta,
    so its budget is dropped from the ID while the directory still records it.
    The path is how the generator actually filed the bundle.
    """
    parts = bundle.parts
    if len(parts) < 4:
        return None
    settings, scope, budget, family = parts[-4:]
    if family not in PROMPT_FAMILY_DIRECTORIES or scope not in _SCOPE_DIRECTORIES:
        return None
    if not _BUDGET_DIRECTORY.match(budget):
        return None
    return settings, budget


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


def _protocol_path(bundle: Path, filename: str) -> Path:
    """Resolve one protocol CSV from either the scope or setup directory."""

    direct = bundle / filename
    if direct.is_file():
        return direct
    candidates = [
        path for path in bundle.parent.rglob(filename) if "protocol" in path.parts
    ]
    if len(candidates) == 1:
        return candidates[0]
    return direct


def _validate_protocol(rows: list[dict[str, str]]) -> None:
    required = {"protocol_id", "dataset", "category", "label", "partition"}
    if not required.issubset(rows[0]):
        raise ValueError(
            f"Protocol is missing columns: {sorted(required - set(rows[0]))}"
        )


def _field(row: dict[str, str], *names: str, required: bool = True, default: str = "") -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    if required:
        raise ValueError(f"Manifest is missing one of these fields: {names}")
    return default


def _full_data_cross(
    row: dict[str, str], setup_id: str, split_protocol: str
) -> bool:
    """Resolve the cross-data cohort switch, preserving legacy bundles.

    Current generators record the boolean directly and also name the mode in
    the setup ID. Older full-protocol bundles predate that independent switch,
    so only those untagged bundles retain the former ``split=full`` inference.
    """

    value = str(row.get("full_data_cross", "")).strip().lower()
    if value:
        if value not in {"true", "false"}:
            raise ValueError(f"Invalid full_data_cross value: {value!r}")
        return value == "true"
    lowered = setup_id.lower()
    if "_fullcross" in lowered:
        return True
    if "_halfcross" in lowered:
        return False
    return split_protocol == "full"


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
    # The images this perturbation was fitted on. Scoring them alongside the
    # held-out ones is what makes the generalisation gap visible: a delta that
    # only works where it was fitted memorised its cohort. Empty for per_image,
    # which fits the single image it attacks and so has nothing held out, and
    # for older bundles that ship no attack_train_indices.csv.
    train_ids: tuple[str, ...] = ()
    train_attacked_ids: tuple[str, ...] = ()

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
            # One tensor for every image, so the training half indexes it too.
            covered = (*self.attacked_ids, *self.train_attacked_ids)
            return delta.float().contiguous(), {sample_id: 0 for sample_id in covered}
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
        evaluation_protocol = _read_csv(
            _protocol_path(bundle, "evaluation_test_indices.csv")
        )
        _validate_protocol(evaluation_protocol)
        attack_train_protocol: list[dict[str, str]] | None = None
        train_protocol: list[dict[str, str]] | None = None
        complete_protocol: list[dict[str, str]] | None = None
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
            inferred_formulation = _formulation_from_setup_id(setup_id)
            formulation = _field(
                raw, "loss_formulation", required=False, default=inferred_formulation
            )
            split_protocol = _field(
                raw, "split_protocol", required=False,
                default=(
                    "full"
                    if re.search(r"_full(?:_|$)", setup_id.lower())
                    else "balanced"
                ),
            ).lower()
            full_data_cross = (
                _full_data_cross(raw, setup_id, split_protocol)
                if scope == "cross_dataset" else None
            )
            loss_mode = _field(raw, "loss_mode", "objective")
            # The grouped tree splits the flat ID across levels, so the path
            # can no longer supply it; the manifest records it directly.
            recorded_setup_id = _field(raw, "setup_id", required=False)
            if recorded_setup_id.endswith("_learnable_prompt"):
                recorded_setup_id = recorded_setup_id[: -len("_learnable_prompt")]
            placed = layout_from_bundle(bundle)
            if placed is not None and not recorded_setup_id:
                # The grouped tree splits the ID across levels and drops the
                # budgets a scope does not spend, so the path provably cannot
                # rebuild it. Labelling the condition "unspecified_setup" would
                # pool every setup under one name and only surface later, as an
                # empty selection after the whole run.
                raise ValueError(
                    f"{bundle} is filed under <settings>/<scope>/<budget>/"
                    "<prompt family> but its attack_manifest.csv has no "
                    "setup_id column, and that layout cannot supply one"
                )
            effective_setup_id = recorded_setup_id or setup_id
            if placed is not None:
                settings, scope_budget = placed
            else:
                settings = split_setup_id(effective_setup_id)[1]
                scope_budget = scope_budget_tag(effective_setup_id, scope)
            normalized = {
                **raw, "prompt_mode": prompt_mode,
                "setup_id": recorded_setup_id or setup_id,
                "settings": settings, "scope_budget": scope_budget,
                "scope": scope, "source_dataset": source, "target_dataset": target,
                "direction": direction, "source_label": source_label,
                "target_label": target_label, "category": category,
                "loss_formulation": formulation, "loss_mode": loss_mode,
                "split_protocol": split_protocol,
                "full_data_cross": (
                    full_data_cross if full_data_cross is not None else ""
                ),
                "image_size": int(_field(raw, "image_size")),
                "epsilon": float(_field(raw, "epsilon")),
                "tensor_key": _field(raw, "noise_tensor_key", "tensor_key", required=False,
                                     default="deltas" if scope == "per_image" else "delta"),
                "sample_ids_key": _field(raw, "sample_ids_key", required=False, default="sample_ids"),
            }
            if any(allowed is not None and str(normalized[name]) not in allowed for name, allowed in filters.items()):
                continue
            full_cross_dataset = (
                scope == "cross_dataset"
                and source != target
                and full_data_cross is True
            )
            if full_cross_dataset:
                complete_path = _protocol_path(
                    bundle, "complete_retained_indices.csv"
                )
                if complete_path.is_file():
                    if complete_protocol is None:
                        complete_protocol = _read_csv(complete_path)
                        _validate_protocol(complete_protocol)
                    protocol_rows = complete_protocol
                else:
                    recorded_source = str(
                        raw.get("evaluation_ids_source", "")
                    ).strip()
                    if Path(recorded_source).name == "complete_retained_indices.csv":
                        raise FileNotFoundError(complete_path)
                    # Compatibility with older two-source bundles. This can
                    # reconstruct a complete target only when that dataset was
                    # also a source; current generators always package the
                    # dedicated complete-retained protocol instead.
                    if attack_train_protocol is None:
                        attack_train_protocol = _read_csv(
                            _protocol_path(bundle, "attack_train_indices.csv")
                        )
                        _validate_protocol(attack_train_protocol)
                    protocol_rows = [*attack_train_protocol, *evaluation_protocol]
                cohort = [
                    row for row in protocol_rows
                    if row["dataset"] == target
                    and (scope in DATASET_LEVEL_SCOPES or row["category"] == category)
                ]
            else:
                cohort = [
                    row for row in evaluation_protocol
                    if row["dataset"] == target and row["partition"] == "evaluation"
                    and (scope in DATASET_LEVEL_SCOPES or row["category"] == category)
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
            # The training half, for runs that also score what the delta was
            # fitted on. Two conditions have no such half. per_image fits the
            # single image it attacks, so it has no held-out set and no gap to
            # measure. A delta delivered to a dataset it was not fitted on has
            # the mirror problem: its fitted images live in the source, so the
            # target cohort contains none of them and every target image is
            # already held out. The comparison that would give is reported
            # anyway, because under halfcross the cross delta IS the
            # per_dataset delta and its fitted rows appear on that condition.
            # Comparing the datasets rather than naming the scope keeps this
            # right for any future condition that crosses. Older bundles may
            # ship no attack_train_indices.csv, and an absent one is not an
            # error here because the held-out evaluation does not depend on it.
            train_ids: tuple[str, ...] = ()
            train_attacked_ids: tuple[str, ...] = ()
            if scope != "per_image" and source == target:
                if train_protocol is None:
                    train_path = _protocol_path(bundle, "attack_train_indices.csv")
                    if train_path.is_file():
                        train_protocol = _read_csv(train_path)
                        _validate_protocol(train_protocol)
                    else:
                        train_protocol = []
                train_cohort = [
                    row for row in train_protocol
                    if row["dataset"] == source
                    and row["partition"] == "attack_train"
                    and (scope in DATASET_LEVEL_SCOPES or row["category"] == category)
                ]
                train_ids = tuple(row["protocol_id"] for row in train_cohort)
                train_attacked_ids = tuple(
                    row["protocol_id"] for row in train_cohort
                    if int(row["label"]) == source_label
                )
            attack = Attack(
                bundle, normalized, _artifact_path(bundle, path_text),
                evaluation_ids, attacked_ids, train_ids, train_attacked_ids,
            )
            if not attack.perturbation_path.is_file():
                raise FileNotFoundError(attack.perturbation_path)
            if attack.condition_id in seen:
                previous = seen[attack.condition_id]
                previous_hash = str(previous.record.get("artifact_sha256") or "").lower()
                current_hash = str(attack.record.get("artifact_sha256") or "").lower()
                if not previous_hash or not current_hash:
                    # Manifests without recorded checksums still describe the same
                    # artifact when the extracted files are byte-identical.
                    previous_hash = file_sha256(previous.perturbation_path)
                    current_hash = file_sha256(attack.perturbation_path)
                if previous_hash == current_hash:
                    # A complete Kaggle setup commonly includes both the
                    # extracted scope directory and a portable ZIP of it.
                    continue
                raise ValueError(f"Conflicting duplicate condition: {attack.condition_id}")
            seen[attack.condition_id] = attack
            attacks.append(attack)
    if not attacks:
        raise ValueError("No attack conditions matched the configuration")
    return sorted(attacks, key=lambda item: item.condition_id)
