"""IIPAD (One-for-All few-shot AD) adapter matching the official test pass of main.py."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

from ..kaggle import download_kaggle_dataset, find_kaggle_files
from .base import ModelAdapter, register_adapter
from .reference import NormalReference


# The authors release no weights. Trained for this project with the official
# main.py at k = 4 (test.sh settings, 20 epochs, seed 4), patched only for paths,
# the shot count and weight saving (full_pipeline_run/train_pipeline/patch_iipad.py).
KAGGLE_DATASET = "parsagholami1383/iipad-4-shot-checkpoints-mvtec-ad-and-visa"
# The soft prompt is trained against a k-shot memory, so only that k is valid.
SHOT_VALUES = (4,)
# soft_prompt_best.pt is the epoch main.py reports (best test pixel AUROC);
# soft_prompt_last.pt is the final epoch, without that selection.
CHECKPOINTS = ("best", "last")
MODEL_IMAGE_SIZE = 224
FEATURES = (6, 12, 18, 24)
DPAM_LAYER = 20
SIGMA = 4
# Only build VVCLIP; main.py says these are unused by the model.
VVCLIP_PARAMETERS = {
    "Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4,
}
OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


def _import_official_repository(repository: str | Path):
    root = Path(repository).expanduser().resolve()
    required = (root / "main.py", root / "VVCLIP_lib", root / "open_clip", root / "prompt_generator.py",
                root / "prompt_ensemble.py")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Official IIPAD repository is incomplete at {root}; missing {missing}")
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    for name in list(sys.modules):
        if not (
            name in {"open_clip", "prompt_ensemble", "prompt_generator", "VVCLIP_lib", "clip", "utils", "dataset"}
            or name.startswith(("open_clip.", "VVCLIP_lib.", "clip."))
        ):
            continue
        module = sys.modules.get(name)
        locations = list(getattr(module, "__path__", []) or []) if module else []
        origin = str(getattr(module, "__file__", "") or (locations[0] if locations else ""))
        if module and not origin.startswith(root_text):
            sys.modules.pop(name, None)
    return (
        importlib.import_module("open_clip"),
        importlib.import_module("VVCLIP_lib"),
        importlib.import_module("prompt_ensemble"),
        importlib.import_module("prompt_generator"),
    )


def resolve_checkpoint(
    target_dataset: str,
    selection: str = "best",
    *,
    checkpoint_root: str | Path | None = None,
    download_root: str | Path | None = None,
) -> Path:
    if selection not in CHECKPOINTS:
        raise ValueError(f"IIPAD checkpoint must be one of {CHECKPOINTS}")
    target = target_dataset.strip().lower()
    root = (
        Path(checkpoint_root).expanduser().resolve()
        if checkpoint_root
        else download_kaggle_dataset(KAGGLE_DATASET, download_root=download_root)
    )
    found = [p for p in find_kaggle_files(root, f"soft_prompt_{selection}.pt") if p.parent.name == target]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one {target}/soft_prompt_{selection}.pt below {root}, found {found}")
    return found[0]


def shot_positions(k_shot: int) -> list[int]:
    """main.py loads '00{3i}.png' (MVTec) / '000{3i}.JPG' (VisA) for i < k.

    In each class's name-sorted normal training images that is position 3i: the
    exact files on MVTec, and on VisA the same positions of the 1cls train split
    the author's renumbered folder indexed (several VisA classes have no such
    file names in the official split).
    """

    return [3 * i for i in range(int(k_shot))]


@register_adapter("iipad")
class IIPADAdapter(ModelAdapter):
    """Official IIPAD few-shot inference for MVTec AD and VisA (one-for-all).

    Reproduces the test pass inside main.py: a VV-attention CLIP ViT-L-14 (DPAM
    from layer 20) at 224 px, BLIP-Diffusion's Q-Former turning each image into a
    query, and the trained soft-prompt generator turning query + image token into
    a positive and a negative text prompt. The memory holds the k shots of
    **every** class of the target (one model for all classes); for each test
    image the k most query-similar shots are retrieved, their prompts averaged
    with the image's own, and the map is the mean text map of the four layers
    plus the nearest-patch distance on two layers, at 16x16, upsampled to 224 and
    blurred with sigma 4. The image score is
    ``-p(normal) - mean query similarity + max(text map) + max(patch map)``.

    Two departures, both needed for a fair attack measurement:
    the soft prompt runs in eval mode (main.py never calls ``.eval()``, so its
    dropout stays on at test time and the official scores are slightly random),
    and only the Q-Former of BLIP-Diffusion is moved to the GPU - the UNet and
    VAE main.py loads are never used, and on a 24 GB card they leave no room.
    """

    name = "iipad"

    def __init__(
        self,
        *,
        repository: str,
        target_dataset: str,
        k_shot: int = 4,
        checkpoint: str = "best",
        mvtec_root: str | None = None,
        visa_root: str | None = None,
        checkpoint_root: str | None = None,
        download_root: str | None = None,
        blip_model: str = "Salesforce/blipdiffusion",
        device: str = "cuda",
        image_size: int = 518,
        seed: int = 4,
    ) -> None:
        target = target_dataset.strip().lower()
        if target not in {"mvtec", "visa"}:
            raise ValueError("IIPAD target_dataset must be 'mvtec' or 'visa'")
        if int(k_shot) not in SHOT_VALUES:
            raise ValueError(f"The trained IIPAD soft prompts cover k_shot in {SHOT_VALUES}")
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.k_shot = int(k_shot)
        self.target = target
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        open_clip, vvclip, prompt_ensemble, prompt_generator = _import_official_repository(repository)
        from diffusers.pipelines import BlipDiffusionPipeline
        from diffusers.utils import load_image

        self._vvclip = vvclip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", MODEL_IMAGE_SIZE, pretrained="openai")
        self._clip = clip_model.to(self.device).eval().requires_grad_(False)
        model, _ = vvclip.load("ViT-L-14", device=self.device, design_details=VVCLIP_PARAMETERS)
        model.eval().requires_grad_(False)
        model.visual.DAPM_replace(DPAM_layer=DPAM_LAYER)
        self._model = model.to(self.device)
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self._padding = tokenizer("").to(self.device)
        self._blip = BlipDiffusionPipeline.from_pretrained(blip_model, torch_dtype=torch.float32)
        self._blip.qformer.to(self.device)

        checkpoint_path = resolve_checkpoint(
            target, checkpoint, checkpoint_root=checkpoint_root, download_root=download_root
        )
        payload = torch.load(str(checkpoint_path), map_location=self.device)
        soft_prompt = prompt_generator.SoftPrompt().to(self.device)
        soft_prompt.load_state_dict(payload["soft_prompt"])
        self._soft_prompt = soft_prompt.eval().requires_grad_(False)

        mean = torch.tensor(OPENAI_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(OPENAI_STD, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

        reference = NormalReference(dataset=target, mvtec_root=mvtec_root, visa_root=visa_root)
        self._selection: dict[str, list[str]] = {}
        with torch.no_grad():
            text_prompts, _ = prompt_ensemble.encode_text_with_prompt_ensemble(
                self._clip, ["object"], tokenizer, self.device, dataset=target
            )
            self._build_memory(reference, load_image)
        self._runtime_metadata = {
            "adapter": self.name,
            "mode": "few_shot",
            "official_entrypoint_defaults": "IIPAD main.py (test.sh: 224 px, depth 9, n_ctx 12, t_n_ctx 4)",
            "target_dataset": target,
            "backbone": "ViT-L-14 (openai), VV-attention / DPAM from layer 20",
            "model_image_size": MODEL_IMAGE_SIZE,
            "features": list(FEATURES),
            "k_shot": self.k_shot,
            "blip_model": blip_model,
            "checkpoint": f"{KAGGLE_DATASET}:{target}/{checkpoint_path.name}",
            "checkpoint_selection": checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "reference_selection": "positions 0, 3, 6, 9 of each class's sorted normal training images",
            "memory": "one-for-all: k shots of every class, k most query-similar retrieved per image",
            "soft_prompt_mode": "eval (upstream leaves dropout on at test time)",
            "official_gaussian_sigma": SIGMA,
            "seed": int(seed),
        }

    def _build_memory(self, reference: NormalReference, load_image) -> None:
        """The memory-bank block of main.py, over every class of the target."""

        bank_1, bank_2, queries, prompts = [], [], [], []
        for category in reference.categories():
            candidates = reference.candidates(category)
            samples = [candidates[i] for i in shot_positions(self.k_shot)]
            self._selection[category] = NormalReference.describe(samples)
            for sample in samples:
                image = self._blip.image_processor.preprocess(
                    load_image(str(sample.image_path)), image_mean=self._blip.config.mean,
                    image_std=self._blip.config.std, return_tensors="pt",
                )["pixel_values"].to(self.device)
                query = self._blip.get_query_embeddings(image, ["object"] * 10).mean(dim=0).unsqueeze(dim=0)
                embedding, _, memory_tokens = self._model.encode_image(
                    image, list(FEATURES), DPAM_layer=DPAM_LAYER, ffn=False
                )
                embedding = embedding.mean(dim=0).unsqueeze(dim=0)
                positive, negative = self._soft_prompt(query, embedding)
                positive, _ = self._clip.encode_text_prompt(positive, self._padding, self.device)
                negative, _ = self._clip.encode_text_prompt(negative, self._padding, self.device)
                bank_1.append(memory_tokens[0][0][1:])
                bank_2.append(memory_tokens[2][0][1:])
                queries.append(query / query.norm(dim=-1, keepdim=True))
                prompts += [positive, negative]
        self._bank_1 = F.normalize(torch.stack(bank_1, dim=0), dim=-1)
        self._bank_2 = F.normalize(torch.stack(bank_2, dim=0), dim=-1)
        shots = torch.stack(prompts, dim=0).view(-1, 2, 768)
        self._shot_prompts = shots / shots.norm(dim=-1, keepdim=True)
        self._query_bank = torch.vstack(queries)

    def runtime_metadata(self) -> dict[str, object]:
        data = dict(self._runtime_metadata)
        data["reference_images"] = dict(sorted(self._selection.items()))
        return data

    def _score(self, image: torch.Tensor) -> tuple[float, np.ndarray]:
        """The per-image test block of main.py."""

        shot = self.k_shot
        features, patches, memory_tokens = self._model.encode_image(
            image, list(FEATURES), DPAM_layer=DPAM_LAYER, ffn=False
        )
        features = features / features.norm(dim=-1, keepdim=True)
        query = self._blip.get_query_embeddings(image, ["object"] * 10).mean(dim=0).unsqueeze(dim=0)
        positive, negative = self._soft_prompt(query, features)
        positive, _ = self._clip.encode_text_prompt(positive, self._padding, self.device)
        negative, _ = self._clip.encode_text_prompt(negative, self._padding, self.device)
        positive = positive / positive.norm()
        negative = negative / negative.norm()

        query = query / query.norm(dim=-1, keepdim=True)
        query = query.expand(len(self._query_bank), -1, -1)
        query_sim = torch.mean(torch.sum(torch.mul(self._query_bank, query), dim=-1), dim=-1)
        picked = torch.topk(query_sim, k=shot)[1].cpu().numpy().tolist()
        bank_1 = self._bank_1[picked].view(-1, 1024)
        bank_2 = self._bank_2[picked].view(-1, 1024)

        text = self._shot_prompts.clone()
        current = text[picked].clone().mean(dim=0)
        current[0] = text[picked][:, 0, :].clone().mean(dim=0)
        current[1] = text[picked][:, 1, :].clone().mean(dim=0)
        probs = torch.matmul(text, features.T).permute(0, 2, 1)
        probs = (probs / 0.07).softmax(-1).view(len(text), -1)[picked].mean(dim=0)
        current[0] = (current[0].unsqueeze(dim=0) + positive) / 2
        current[1] = (current[1].unsqueeze(dim=0) + negative) / 2
        current[0] = current[0] / current[0].norm()
        current[1] = current[1] / current[1].norm()
        current = current.T

        text_maps = []
        for patch in patches:
            patch = patch / patch.norm(dim=-1, keepdim=True)
            similarity, _ = self._vvclip.compute_similarity(patch, current.T)
            similarity = similarity[:, 1:, :].reshape(similarity.shape[0], 16, 16, -1)
            text_maps.append((similarity[..., 1] + 1 - similarity[..., 0]) / 2.0)
        token_1 = memory_tokens[0][0][1:]
        token_2 = memory_tokens[2][0][1:]
        token_1 = token_1 / token_1.norm(dim=-1, keepdim=True)
        token_2 = token_2 / token_2.norm(dim=-1, keepdim=True)
        score_1 = (1.0 - token_1 @ bank_1.t()).min(dim=-1)[0] / 2.0
        score_2 = (1.0 - token_2 @ bank_2.t()).min(dim=-1)[0] / 2.0
        visual = (score_1 + score_2).reshape(1, 1, 16, 16)
        textual = (torch.stack(text_maps).sum(dim=0) / 4.0).reshape(1, 1, 16, 16)

        score = -probs[0] - query_sim[picked].mean(dim=0) + torch.max(textual) + torch.max(visual)
        anomaly_map = F.interpolate(
            textual + visual, size=(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), mode="bilinear", align_corners=False
        )[0, 0]
        return float(score), gaussian_filter(anomaly_map.float().cpu().numpy(), sigma=SIGMA)

    def predict(self, images: torch.Tensor, categories: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        if len(images) != len(categories):
            raise ValueError("IIPAD received mismatched images and categories")
        # get_transform: bicubic resize to 224 and the OpenAI CLIP normalization.
        batch = F.interpolate(
            images.to(self.device, dtype=torch.float32), size=(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
            mode="bicubic", align_corners=False, antialias=True,
        ).clamp(0, 1)
        batch = (batch - self._mean) / self._std
        scores = np.empty(len(batch), dtype=np.float32)
        maps = np.empty((len(batch), MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), dtype=np.float32)
        with torch.no_grad():
            for index in range(len(batch)):
                scores[index], maps[index] = self._score(batch[index : index + 1])
        return scores, maps

    def close(self) -> None:
        self._model = None
        self._clip = None
        self._blip = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
