"""FiLo adapter matching the official zero-shot evaluation defaults."""

from __future__ import annotations

from collections.abc import Sequence
import copy
import hashlib
import importlib
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch

from .base import ModelAdapter, register_adapter


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# Grounding DINO uses its own ImageNet normalization, not CLIP's.
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)

HF_REPOSITORY = "FantasticGNU/FiLo"
# filename -> sha256, taken from the LFS metadata of the released repository.
CHECKPOINTS = {
    "filo_train_on_mvtec": (
        "filo_train_on_mvtec.pth",
        "e14e74abcc188afeb2d4a0212ab75d5f66b9a4e9d6b2aa64854a05001dfe5773",
    ),
    "filo_train_on_visa": (
        "filo_train_on_visa.pth",
        "8a240ca0d6538c7daf36ffbf0860caa8012837a8df5b1642a352f56ed890ba27",
    ),
    "grounding_train_on_mvtec": (
        "grounding_train_on_mvtec.pth",
        "90b45c23a8154274182625ef341b88252220321ddd3bd02a14936daad85486da",
    ),
    "grounding_train_on_visa": (
        "grounding_train_on_visa.pth",
        "b9cd932c570cc26e288515510b32e47a5e50536da02807231166380721c98ec2",
    ),
}
# test.sh evaluates a dataset with the weights trained on the other one, for the
# FiLo head and its fine-tuned Grounding DINO alike.
ZERO_SHOT_CHECKPOINT = {"mvtec": "train_on_visa", "visa": "train_on_mvtec"}

# The GPT-written anomaly descriptions of the official test.py. They differ from
# the copies in models/FiLo.py only in capitalization, and this Grounding DINO
# side is the one that must match: the caption is lowercased before the phrases
# are matched back against these strings. tests/test_filo.py checks them against
# the repository.
MVTEC_ANOMALY_DETAIL = {
    "carpet": "discoloration in a specific area,irregular patch or section with a different texture,frayed edges or unraveling fibers,burn mark or scorching",
    "grid": "crooked,cracks,excessive gaps,discoloration,deformation,missing,inconsistent spacing between grid elements,corrosion,visible signs,chipping",
    "leather": "scratches,discoloration,creases,uneven texture,tears,brittleness,damage,seams,heat damage,mold",
    "tile": "chipped,irregularities,discoloration,efflorescence,warping,missing,depressions,lippage,fungus,damage",
    "wood": "knots,warping,cracks along the grain,mold growth on the surface,staining from water damage,wood rot,woodworm holes,rough patches,protruding knots",
    "bottle": "cracked large,cracked small,dented large,dented small,leaking,discolored,deformed,missing cap,excessive condensation,unusual odor",
    "cable": "twisted,knotted cable strands,detached connectors,excessive stretching,dents,corrosion,scorching along the cable,exposed conductive material",
    "capsule": "irregular shape,discoloration coloring,crinkled,uneven seam,condensation inside the capsule,foreign particles,unusually soft or hard",
    "hazelnut": "fungal growth,unusual discoloration,rotten or foul odor emanating,insect infestation,wetness,misshapen shell,unusually thin,contaminants,unusual texture",
    "metal nut": "cracks,irregular threading,corrosion,missing,distortion,signs of discoloration,excessive wear on contact surfaces,inconsistent texture",
    "pill": "irregular shape,crumbling texture,excessive powder,Uneven coating,presence of air bubbles,disintegration,abnormal specks",
    "screw": "rust on the surface,bent,damaged threads,stripped threads,deformed top,coating damage,uneven grooves,inconsistent size",
    "toothbrush": "loose bristles,uneven bristle distribution,excessive shedding of bristles,staining on the bristles,abrasive texture,irregularities in the shape",
    "transistor": "burn marks,detached leads,signs of corrosion,irregularities in the shape,presence of cracks or fractures,signs of physical trauma,irregularities in the surface texture",
    "zipper": "bent,frayed,misaligned,excessive stiffness,corroded,detaches,loose,warped",
}
VISA_ANOMALY_DETAIL = {
    "candle": "cracks or fissures in the wax,Wax pooling unevenly around the wick,tunneling,incomplete wax melt pool,irregular or flickering flame,other,extra wax in candle,wax melded out of the candle",
    "capsules": "uneven capsule size,capsule shell appears brittle,excessively soft,dents,condensation,irregular seams or joints,specks",
    "cashew": "uneven coloring,fungal growth,presence of foreign objects,unusual texture,empty shells,signs of moisture,stuck together",
    "chewinggum": "consistency,presence of foreign objects,uneven coloring,excessive hardness,similar colour spot",
    "fryum": "irregular shape,unusual odor,uneven coloring,unusual texture,small scratches,different colour spot,fryum stuck together,other",
    "macaroni1": "uneven shape ,small scratches,small cracks,uneven coloring,signs of insect infestation,uneven texture,Unusual consistency",
    "macaroni2": "irregular shape,small scratches,presence of foreign particles,excessive moisture,Signs of infestation,small cracks,unusual texture",
    "pcb1": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb2": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb3": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb4": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pipe fryum": "uneven shape,presence of foreign objects,different colour spot,unusual odor,empty interior,unusual texture,similar colour spot,stuck together",
}
ANOMALY_DETAIL = {
    "mvtec": {name: text.split(",") for name, text in MVTEC_ANOMALY_DETAIL.items()},
    "visa": {name: text.split(",") for name, text in VISA_ANOMALY_DETAIL.items()},
}
ANOMALY_STATUS_GENERAL = ["anomaly", "damage", "broken", "defect", "contamination"]

# The nine position regions, in the order PromptLearner_abnormal interleaves its
# prompts, on the official 518-pixel grid.
POSITIONS = {
    "top left": ((0, 0), (172, 172)),
    "top": ((173, 0), (344, 172)),
    "top right": ((345, 0), (517, 172)),
    "left": ((0, 173), (172, 344)),
    "center": ((173, 173), (344, 344)),
    "right": ((345, 173), (517, 344)),
    "bottom left": ((0, 345), (172, 517)),
    "bottom": ((173, 345), (344, 517)),
    "bottom right": ((345, 345), (517, 517)),
}

GROUNDING_CONFIG = (
    "models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
# state_dict prefixes the checkpoint must cover; the official load is
# strict=False, which would silently leave a trained module randomly
# initialized.
TRAINED_MODULES = (
    "decoder_cov.",
    "decoder_linear.",
    "normal_prompt_learner.",
    "abnormal_prompt_learner.",
    "adapter.",
)


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    grounding_root = root / "models" / "GroundingDINO"
    required = (root / "models" / "FiLo.py", grounding_root / "groundingdino", root / "test.py")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Official FiLo repository is incomplete at {root}; missing {missing}"
        )
    # groundingdino imports itself absolutely, so its own directory has to be
    # importable as well as the FiLo root.
    roots = (str(root), str(grounding_root))
    for entry in roots:
        if entry in sys.path:
            sys.path.remove(entry)
    sys.path[:0] = roots
    importlib.invalidate_caches()
    # Other target repositories ship modules with these names.
    for name in list(sys.modules):
        if not (
            name in {"models", "datasets", "data", "utils", "groundingdino"}
            or name.startswith(("models.", "datasets.", "data.", "utils.", "groundingdino."))
        ):
            continue
        module = sys.modules.get(name)
        if module is None:
            continue
        locations = list(getattr(module, "__path__", []) or [])
        origin = str(
            getattr(module, "__file__", "") or (locations[0] if locations else "")
        )
        if not any(origin.startswith(entry) for entry in roots):
            sys.modules.pop(name, None)
    return (
        importlib.import_module("models.FiLo"),
        importlib.import_module("groundingdino.models"),
        importlib.import_module("groundingdino.util.slconfig"),
        importlib.import_module("groundingdino.util.utils"),
    )


def _enable_pytorch_deformable_attention() -> bool:
    """Route deformable attention through the reference implementation.

    Grounding DINO calls a compiled CUDA extension whenever the tensors are on a
    GPU, and building it needs the CUDA toolkit. When the extension is absent the
    call raises instead of falling back, so this swaps in the pure-PyTorch path
    the module already uses on CPU. Returns True when the swap was needed.
    """

    module = importlib.import_module(
        "groundingdino.models.GroundingDINO.ms_deform_attn"
    )
    try:
        importlib.import_module("groundingdino._C")
        return False
    except ImportError:
        pass

    reference = module.multi_scale_deformable_attn_pytorch

    class _PyTorchDeformableAttention:
        @staticmethod
        def apply(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        ):
            del level_start_index, im2col_step
            return reference(
                value, spatial_shapes, sampling_locations, attention_weights
            )

    module.MultiScaleDeformableAttnFunction = _PyTorchDeformableAttention
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(
    checkpoint: str | Path, *, download_root: str | Path | None = None
) -> Path:
    """Return a local checkpoint, downloading a released name from HuggingFace."""

    text = str(checkpoint).strip()
    key = text.lower()
    if key not in CHECKPOINTS:
        path = Path(text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"FiLo checkpoint not found: {path}. Pass a path or one of "
                f"{sorted(CHECKPOINTS)}."
            )
        return path
    filename, expected = CHECKPOINTS[key]
    root = (
        Path(download_root).expanduser().resolve()
        if download_root
        else Path.home() / ".cache" / "filo"
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / filename
    if destination.is_file() and _sha256(destination) == expected:
        return destination
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Downloading FiLo weights needs huggingface-hub; install the filo "
            "extra or pass an existing checkpoint path."
        ) from error
    downloaded = Path(
        hf_hub_download(
            repo_id=HF_REPOSITORY, filename=filename, local_dir=str(root)
        )
    )
    actual = _sha256(downloaded)
    if actual != expected:
        raise ValueError(
            f"FiLo checkpoint checksum mismatch for {filename}: expected "
            f"{expected}, got {actual}"
        )
    return downloaded


def _phrase_matches(descriptions: Sequence[str], phrase: str) -> bool:
    """check_elements_in_array: any description occurring in the phrase."""

    return any(description in phrase for description in descriptions)


@register_adapter("filo")
class FiLoAdapter(ModelAdapter):
    """Official FiLo zero-shot inference for MVTec AD and VisA.

    Reproduces test.py: a fine-tuned Grounding DINO proposes anomaly boxes from
    the GPT-written descriptions of the category, the box with the highest score
    picks one of nine position words, and FiLo scores the image with position-
    enhanced prompts. The per-layer maps are blurred inside the model, so the
    shared evaluator adds no blur; the map is finally damped to 0.7 outside the
    proposed boxes.

    Both branches are driven from the same evaluated tensor. The official loop
    re-reads the file from disk for Grounding DINO, which would bypass the
    perturbation entirely, and its aspect-preserving resize is a no-op on the
    square 518-pixel cohort images.
    """

    name = "filo"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        checkpoint: str | None = None,
        grounding_checkpoint: str | None = None,
        download_root: str | None = None,
        device: str = "cuda",
        image_size: int = 518,
        clip_model: str = "ViT-L-14-336",
        clip_pretrained: str = "openai",
        features: Sequence[int] = (6, 12, 18, 24),
        n_ctx: int = 12,
        box_threshold: float = 0.25,
        area_threshold: float = 0.7,
        outside_box_weight: float = 0.7,
        seed: int = 111,
    ) -> None:
        target_key = target_dataset.strip().lower()
        if target_key not in ZERO_SHOT_CHECKPOINT:
            raise ValueError("FiLo target_dataset must be 'mvtec' or 'visa'")
        if image_size != 518:
            raise ValueError("Official FiLo evaluation uses image_size=518")
        if clip_model != "ViT-L-14-336":
            raise ValueError("Official FiLo evaluation uses ViT-L-14-336")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.target = target_key
        self.box_threshold = float(box_threshold)
        self.area_threshold = float(area_threshold)
        self.outside_box_weight = float(outside_box_weight)
        self.descriptions = ANOMALY_DETAIL[target_key]

        filo_module, grounding_models, slconfig, grounding_utils = (
            _import_official_repository(repository)
        )
        import torchvision

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        self._get_phrases = grounding_utils.get_phrases_from_posmap
        self._pytorch_deformable_attention = _enable_pytorch_deformable_attention()

        suffix = ZERO_SHOT_CHECKPOINT[target_key]
        filo_name = checkpoint or f"filo_{suffix}"
        grounding_name = grounding_checkpoint or f"grounding_{suffix}"
        filo_path = resolve_checkpoint(filo_name, download_root=download_root)
        grounding_path = resolve_checkpoint(
            grounding_name, download_root=download_root
        )

        config_path = Path(repository).expanduser().resolve() / GROUNDING_CONFIG
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Grounding DINO config not found: {config_path}"
            )
        grounding_args = slconfig.SLConfig.fromfile(str(config_path))
        grounding_args.device = str(self.device)
        grounding = grounding_models.build_model(grounding_args)
        payload = torch.load(str(grounding_path), map_location="cpu")
        grounding.load_state_dict(
            grounding_utils.clean_state_dict(payload), strict=False
        )
        grounding.to(self.device).eval().requires_grad_(False)
        self._grounding = grounding

        # FiLo reads these off the argparse namespace.
        self._args = SimpleNamespace(
            clip_model=clip_model,
            clip_pretrained=clip_pretrained,
            image_size=self.image_size,
            features_list=[int(level) for level in features],
            n_ctx=int(n_ctx),
            device=str(self.device),
        )
        # The datasets iterate their CLSNAMES, which are alphabetical, and the
        # abnormal prompt learner keys pcb1-4 and macaroni1-2 onto shared names,
        # so the order decides which description table survives per shared key.
        categories = sorted(self.descriptions)
        model = filo_module.FiLo(categories, self._args, self.device).to(self.device)
        weights = torch.load(str(filo_path), map_location="cpu")["filo"]
        missing = [
            name
            for name in model.state_dict()
            if name.startswith(TRAINED_MODULES) and name not in weights
        ]
        if missing:
            raise ValueError(
                f"FiLo checkpoint {filo_path.name} is missing {len(missing)} trained "
                f"tensors, starting with {missing[:3]}; the official strict=False "
                "load would leave them randomly initialized"
            )
        model.load_state_dict(weights, strict=False)
        model.to(self.device).eval().requires_grad_(False)
        self._model = model
        self._blur = torchvision.transforms.GaussianBlur(3, 4.0)

        self._runtime_metadata = {
            "adapter": self.name,
            "official_entrypoint_defaults": "FiLo/test.sh",
            "target_dataset": target_key,
            "clip_model": clip_model,
            "clip_pretrained": clip_pretrained,
            "image_size": self.image_size,
            "features": self._args.features_list,
            "n_ctx": int(n_ctx),
            "box_threshold": self.box_threshold,
            # test.py overwrites text_threshold with box_threshold, so its own
            # --text_threshold flag never takes effect.
            "text_threshold": self.box_threshold,
            "area_threshold": self.area_threshold,
            "outside_box_weight": self.outside_box_weight,
            "grounding_config": GROUNDING_CONFIG,
            "grounding_checkpoint": grounding_path.name,
            "checkpoint": filo_path.name,
            "checkpoint_selection": suffix,
            "categories": categories,
            # The official test.py seeds nothing; this only fixes the
            # initialization the checkpoint then overwrites.
            "seed": int(seed),
            # GaussianBlur(3, 4.0) runs per layer inside the model.
            "official_gaussian_sigma": 4.0,
            "official_gaussian_kernel_size": 3,
            "gaussian_applied_inside_adapter": True,
            "grounding_dino_input": "the evaluated 518-pixel tensor",
            "pytorch_deformable_attention": self._pytorch_deformable_attention,
        }
        self._clip_mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._clip_std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self._dino_mean = torch.tensor(DINO_MEAN, device=self.device).view(1, 3, 1, 1)
        self._dino_std = torch.tensor(DINO_STD, device=self.device).view(1, 3, 1, 1)

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self._runtime_metadata)

    def _grounding_boxes(self, image: torch.Tensor, category: str):
        """get_grounding_output, then test.py box filtering and scaling."""

        descriptions = self.descriptions[category]
        caption = " . ".join(ANOMALY_STATUS_GENERAL + descriptions).lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        outputs = self._grounding(image, captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        areas = boxes[:, 2] * boxes[:, 3]
        keep = torch.bitwise_and(
            logits.max(dim=1)[0] > self.box_threshold, areas < self.area_threshold
        )
        if torch.sum(keep) == 0:
            keep = torch.argmax(logits.max(dim=1)[0])
            logits, boxes = logits[keep].unsqueeze(0), boxes[keep].unsqueeze(0)
        else:
            logits, boxes = logits[keep], boxes[keep]

        tokenized = self._grounding.tokenizer(caption)
        phrases = [
            self._get_phrases(logit > self.box_threshold, tokenized, self._grounding.tokenizer)
            + f"({str(logit.max().item())[:4]})"
            for logit in logits
        ]

        # Boxes whose phrase names no anomaly keep their normalized cxcywh
        # coordinates; test.py marks them and leaves them unscaled, which makes
        # their integer pixel rectangle empty later on.
        scaled = copy.deepcopy(boxes)
        for index in range(boxes.size(0)):
            if not _phrase_matches(
                descriptions + ANOMALY_STATUS_GENERAL, phrases[index]
            ):
                phrases[index] += "#$%"
                continue
            scaled[index] = scaled[index] * self.image_size
            scaled[index][:2] -= scaled[index][2:] / 2
            scaled[index][2:] += scaled[index][:2]
        return scaled, phrases

    def _position(self, boxes: torch.Tensor, phrases: Sequence[str]) -> list[str]:
        best_box, best_score = None, 0.0
        for index in range(boxes.size(0)):
            if "#$%" in phrases[index]:
                continue
            score = float(re.search(r"\((.*?)\)", phrases[index]).group(1))
            if score >= best_score:
                best_box, best_score = boxes[index], score
        if best_box is not None:
            center = (
                (best_box[0] + best_box[2]) / 2,
                (best_box[1] + best_box[3]) / 2,
            )
        else:
            center = (259, 259)
        for region, ((x1, y1), (x2, y2)) in POSITIONS.items():
            if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                return [region]
        return []

    def predict(
        self, images: torch.Tensor, categories: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("FiLo received mismatched images and categories")
        batch = images.to(self.device, dtype=torch.float32)
        clip_batch = (batch - self._clip_mean) / self._clip_std
        dino_batch = (batch - self._dino_mean) / self._dino_std
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty(
            (len(batch), self.image_size, self.image_size), dtype=np.float32
        )
        with torch.no_grad():
            for index, raw_category in enumerate(categories):
                # The datasets hand FiLo the spaced form of the category.
                category = str(raw_category).replace("_", " ")
                if category not in self.descriptions:
                    raise KeyError(
                        f"FiLo has no anomaly descriptions for {category!r} in "
                        f"{self.target}; known categories are "
                        f"{sorted(self.descriptions)}"
                    )
                boxes, phrases = self._grounding_boxes(
                    dino_batch[index : index + 1], category
                )
                position = self._position(boxes, phrases)

                items = {
                    "img": clip_batch[index : index + 1],
                    "cls_name": [category],
                }
                text_probs, anomaly_maps = self._model(
                    items, with_adapter=True, positions=position
                )
                blurred = [
                    self._blur((layer[:, 1, :, :] - layer[:, 0, :, :] + 1) / 2)
                    for layer in anomaly_maps
                ]
                anomaly_map = torch.mean(torch.stack(blurred, dim=0), dim=0)
                scores[index] = float(
                    (text_probs.flatten()[1].item() + anomaly_map.max().item()) / 2
                )

                # test.py marks the boxes by writing 1 into a copy and then
                # damps everything that is not exactly 1, so a pixel that
                # already equals 1 outside a box is spared too.
                marked = anomaly_map.clone()
                for box in boxes:
                    left, top, right, bottom = (int(value.item()) for value in box)
                    marked[:, top:bottom, left:right] = 1
                anomaly_map = torch.where(
                    marked == 1, anomaly_map, anomaly_map * self.outside_box_weight
                )
                maps[index] = anomaly_map[0].float().cpu().numpy()
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._grounding = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
