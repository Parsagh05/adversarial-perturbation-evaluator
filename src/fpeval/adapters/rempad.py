"""ReMP-AD adapter matching the official few-shot evaluation (test.py)."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from ..kaggle import download_kaggle_dataset, find_kaggle_files
from .aprilgan import CLIP_MEAN, CLIP_STD, AprilGANFewShotAdapter
from .base import ModelAdapter, register_adapter
from .reference import NormalReference


# The authors release no weights. These were trained for this project with the
# official train.py and configs (MVTec model on VisA for 15 epochs, VisA model on
# MVTec for 3, seed 111), and the official test.py at k = 4 reproduced the paper
# (results/alirezas_server/train_k4/comparison/). Mirrored as a Kaggle dataset.
KAGGLE_DATASET = "parsagholami1383/rempad-few-shot-checkpoints-mvtec-ad-and-visa"
# test.py records the reference images it drew in k_shot.txt; only the k = 4
# selection was recorded, so that is the only count reproducible exactly.
SHOT_VALUES = (4,)
# config/<dataset>.yaml, the values test.py ran with.
CONFIG = {
    "mvtec": {"lambd": 0.64, "epochs": 15},
    "visa": {"lambd": 0.72, "epochs": 3},
}
MASK_TOPK = 100
MASK_TH = 0.5
MASK_WEIGHT = (1.2, 0.8)
MASK_LAYERS = (12, 18, 24)
REGION_ALPHA = 0.2


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (
        root / "model.py",
        root / "prompt_ensemble.py",
        root / "open_clip",
        root / "modules" / "ICTR.py",
        root / "modules" / "scoring.py",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official ReMP-AD repository is incomplete at {root}; missing {missing}"
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names (APRIL-GAN above all).
    for name in list(sys.modules):
        if not (
            name in {"model", "prompt_ensemble", "open_clip", "dataset", "utils", "modules", "few_shot"}
            or name.startswith(("open_clip.", "modules."))
        ):
            continue
        module = sys.modules.get(name)
        locations = list(getattr(module, "__path__", []) or []) if module else []
        origin = str(getattr(module, "__file__", "") or (locations[0] if locations else ""))
        if module and not origin.startswith(root_text):
            sys.modules.pop(name, None)
    return (
        importlib.import_module("open_clip"),
        importlib.import_module("model"),
        importlib.import_module("prompt_ensemble"),
        importlib.import_module("modules.ICTR"),
        importlib.import_module("modules.scoring"),
    )


def resolve_release(
    target_dataset: str,
    *,
    checkpoint_root: str | Path | None = None,
    download_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """The trained linear layers and the recorded k-shot list of one target."""

    target = target_dataset.strip().lower()
    root = (
        Path(checkpoint_root).expanduser().resolve()
        if checkpoint_root
        else download_kaggle_dataset(KAGGLE_DATASET, download_root=download_root)
    )
    epochs = CONFIG[target]["epochs"]
    weights = [p for p in find_kaggle_files(root, f"epoch_{epochs}.pth") if target in p.parent.name]
    shots = [p for p in find_kaggle_files(root, "k_shot.txt") if p.parent.name == f"{target}_4shot"]
    for found, label in ((weights, f"{target}/epoch_{epochs}.pth"), (shots, f"{target}_4shot/k_shot.txt")):
        if len(found) != 1:
            raise FileNotFoundError(f"Expected exactly one {label} below {root}, found {found}")
    return weights[0], shots[0]


def read_shot_list(path: str | Path) -> dict[str, list[str]]:
    """k_shot.txt as ``category -> relative image paths``, in the drawn order.

    test.py appends one line per drawn image (torch.randint, so a repeat is
    legitimate and kept).
    """

    selection: dict[str, list[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip().replace("\\", "/")
        if line:
            selection.setdefault(line.split("/")[0], []).append(line)
    return selection


def match_samples(candidates: Sequence, relative_paths: Sequence[str]) -> list:
    """The candidate normal images named by k_shot.txt, by path suffix."""

    picked = []
    for relative in relative_paths:
        found = [s for s in candidates if Path(s.image_path).as_posix().endswith("/" + relative)]
        if len(found) != 1:
            raise FileNotFoundError(f"k_shot.txt names {relative!r}, found {len(found)} matching images")
        picked.append(found[0])
    return picked


@register_adapter("remp-ad")
@register_adapter("rempad")
class ReMPADAdapter(ModelAdapter):
    """Official ReMP-AD few-shot inference for MVTec AD and VisA.

    Follows test.py with config/<dataset>.yaml: CLIP ViT-L-14-336 at 518 px,
    features 6/12/18/24, fp16 autocast. The zero-shot half is APRIL-GAN's
    text-vs-patch map refined by VLPF (VPK and LPK masks intersected into region
    features, blended at region_alpha 0.2); the few-shot half is ICTR (intra-class
    token retrieval with class-token reweighting and hard shrinkage at lambd),
    plus the plain nearest-patch distance. The two maps are summed, and the image
    score is ``0.5 * (text probability + per-category min-max normalized map
    maximum)``, exactly as APRIL-GAN few-shot, with the normalization fitted on
    the clean cohort and frozen for the adversarial pass.

    The reference images are the ones test.py drew and recorded in k_shot.txt,
    so the clean pass reproduces the measured result rather than a new draw.
    """

    name = "rempad"
    postprocess_image_scores = AprilGANFewShotAdapter.postprocess_image_scores
    postprocess_image_scores_with_reference = (
        AprilGANFewShotAdapter.postprocess_image_scores_with_reference
    )

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        k_shot: int = 4,
        mvtec_root: str | None = None,
        visa_root: str | None = None,
        checkpoint_root: str | None = None,
        download_root: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        backbone: str = "ViT-L-14-336",
        pretrained: str = "openai",
        features: Sequence[int] = (6, 12, 18, 24),
        seed: int = 10,
    ) -> None:
        target = target_dataset.strip().lower()
        if target not in CONFIG:
            raise ValueError("ReMP-AD target_dataset must be 'mvtec' or 'visa'")
        if int(k_shot) not in SHOT_VALUES:
            raise ValueError(f"ReMP-AD reference sets are recorded for k_shot in {SHOT_VALUES}")
        if image_size != 518 or backbone != "ViT-L-14-336":
            raise ValueError("The trained ReMP-AD weights use ViT-L-14-336 at image_size=518")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.k_shot = int(k_shot)
        self.features = [int(level) for level in features]

        open_clip, model_module, prompt_module, ictr_module, scoring = _import_official_repository(repository)
        self._prompt_module, self._ictr, self._scoring = prompt_module, ictr_module, scoring
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        weights_path, shots_path = resolve_release(
            target, checkpoint_root=checkpoint_root, download_root=download_root
        )
        model, _, _ = open_clip.create_model_and_transforms(backbone, self.image_size, pretrained=pretrained)
        model.to(self.device).eval().requires_grad_(False)
        self._model = model
        self._tokenizer = open_clip.get_tokenizer(backbone)
        configs = json.loads(
            (Path(repository).expanduser().resolve() / "open_clip" / "model_configs" / f"{backbone}.json")
            .read_text(encoding="utf-8")
        )
        linear = model_module.LinearLayer(
            configs["vision_cfg"]["width"], configs["embed_dim"], len(self.features), backbone
        ).to(self.device)
        linear.load_state_dict(torch.load(str(weights_path), map_location=self.device)["trainable_linearlayer"])
        linear.eval().requires_grad_(False)
        self._linear = linear

        self._mask_info = {
            "mask_th": MASK_TH, "mask_weight": list(MASK_WEIGHT), "mask_topk": MASK_TOPK,
            "mask_layers": list(MASK_LAYERS), "mask_attention": True,
        }
        # ICTR reads its settings from test.py's argparse namespace.
        self._ictr_args = SimpleNamespace(
            lambd=CONFIG[target]["lambd"], image_size=self.image_size,
            device=str(self.device), model=backbone, k_shot=self.k_shot,
        )
        self._reference = NormalReference(
            dataset=target, mvtec_root=mvtec_root, visa_root=visa_root, image_size=self.image_size
        )
        self._shot_list = read_shot_list(shots_path)
        self._memory: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        self._text_cache: dict[str, torch.Tensor] = {}
        self._selection: dict[str, list[str]] = {}
        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._runtime_metadata = {
            "adapter": self.name,
            "mode": "few_shot",
            "official_entrypoint_defaults": f"ReMP-AD/test.py --config_path config/{target}.yaml",
            "target_dataset": target,
            "backbone": backbone,
            "pretrained": pretrained,
            "image_size": self.image_size,
            "features": self.features,
            "k_shot": self.k_shot,
            "lambd": CONFIG[target]["lambd"],
            "mask_info": self._mask_info,
            "region_alpha": REGION_ALPHA,
            "seed": int(seed),
            "checkpoint": f"{KAGGLE_DATASET}:{weights_path.parent.name}/{weights_path.name}",
            "reference_selection": "k_shot.txt recorded by the official test.py",
            "image_score": "0.5 * (text probability + normalized map maximum)",
            "official_gaussian_sigma": 0.0,
        }

    def runtime_metadata(self) -> dict[str, object]:
        data = dict(self._runtime_metadata)
        data["reference_images"] = dict(sorted(self._selection.items()))
        return data

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda")

    def _text_features(self, category: str) -> torch.Tensor:
        if category not in self._text_cache:
            with self._autocast(), torch.no_grad():
                prompts = self._prompt_module.encode_text_with_prompt_ensemble(
                    self._model, [category], self._tokenizer, self.device
                )
            self._text_cache[category] = prompts[category]
        return self._text_cache[category]

    def _ensure_memory(self, category: str):
        """few_shot.memory: patch tokens and class tokens of the k references (fp32)."""

        if category in self._memory:
            return self._memory[category]
        samples = match_samples(self._reference.candidates(category), self._shot_list[category])
        self._selection[category] = NormalReference.describe(samples)
        references = (self._reference.load(samples).to(self.device) - self._mean) / self._std
        patches, classes = [], []
        with torch.no_grad():
            for index in range(len(references)):
                _, tokens = self._model.encode_image(
                    references[index : index + 1], self.features, weighted=False, mask_info=None
                )
                patches.append([token[0, 1:, :] for token in tokens])
                classes.append([token[0, 0:1, :] for token in tokens])
        layers = range(len(patches[0]))
        memory = (
            [torch.cat([shot[layer] for shot in patches], dim=0) for layer in layers],
            [torch.cat([shot[layer] for shot in classes], dim=0) for layer in layers],
        )
        self._memory[category] = memory
        return memory

    def predict(self, images: torch.Tensor, categories: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("ReMP-AD received mismatched images and categories")
        scoring = self._scoring
        batch = (images.to(self.device, dtype=torch.float32) - self._mean) / self._std
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty((len(batch), self.image_size, self.image_size), dtype=np.float32)
        for index, raw in enumerate(categories):
            category = str(raw)
            mem_patches, mem_classes = self._ensure_memory(category)
            image = batch[index : index + 1]
            with torch.no_grad(), self._autocast():
                image_features, patch_tokens = self._model.encode_image(
                    image, self.features, weighted=True, mask_info=self._mask_info
                )
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = torch.stack([self._text_features(category)], dim=0)
                scores[index] = float((100.0 * image_features @ text_features[0]).softmax(dim=-1)[0][1])

                vpk = scoring.VPKscoring(
                    torch.stack(patch_tokens, dim=1)[:, :, 1:, :], topk=MASK_TOPK, threshold=MASK_TH
                )
                patch_tokens = self._linear(patch_tokens)
                patch_tokens = [torch.nn.functional.normalize(tokens, dim=-1) for tokens in patch_tokens]
                lpk = scoring.LPKscoring(patch_tokens, text_features, threshold=MASK_TH)
                region = scoring.genregion_feature(patch_tokens, vpk, lpk, self.device)
                layer_maps = scoring.VL_seg(
                    patch_tokens, text_features, region, image_size=self.image_size, region_alpha=REGION_ALPHA
                )
                anomaly_map = np.sum([m[:, 1:, :, :].squeeze(1).float().cpu().numpy() for m in layer_maps], axis=0)

                _, few_tokens = self._model.encode_image(image, self.features, weighted=False, mask_info=None)
                few_maps = self._ictr.ICTR(
                    few_tokens, {category: mem_patches}, {category: mem_classes}, [category], self._ictr_args
                )
                anomaly_map = anomaly_map + np.sum(few_maps, axis=0)
            maps[index] = np.asarray(anomaly_map, dtype=np.float32).reshape(self.image_size, self.image_size)
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._linear = None
        self._memory.clear()
        self._text_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
