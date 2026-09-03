# Fixed Perturbation Evaluator

This repository is a fresh, model-extensible black-box evaluation pipeline for
the perturbation datasets produced by the object-agnostic prompt experiments.
It does not generate or optimize attacks. The perturbation manifest and the
fixed `evaluation_test_indices.csv` are always authoritative.

The target adapters are AnomalyCLIP, AA-CLIP, AdaCLIP, FAPrompt, Crane,
APRIL-GAN, FB-CLIP, Tipsomaly, VCP-CLIP, FiLo, Bayes-PFL, AF-CLIP, CoPS, MRAD
and WinCLIP, plus few-shot variants of WinCLIP, AF-CLIP and APRIL-GAN. The
shared evaluator owns attack
discovery, fixed-cohort validation, RGB construction, metrics, thresholds, and
result files; model-specific loading, preprocessing, prompting, and inference
stay behind a small adapter interface.

## What it accepts

`attacks_root` may point to the complete uploaded Kaggle dataset, one setup
directory, or one ZIP file. ZIP files anywhere below the root are discovered
and safely extracted into the writable evaluation cache. Both layouts below are
supported:

```text
setups/{frozen_prompt,learnable_prompt}/<setup_id>/
  canonical_clip_per_dataset/
  canonical_clip_cross_dataset/
  canonical_clip_per_category/
  canonical_clip_per_image/
```

`per_dataset` and `cross_dataset` carry the same universal delta and differ only
in whether the evaluation dataset is the attack's source dataset, so both are
dataset-level scopes evaluated over the full target cohort. Setup IDs follow
`steps{N}_eps{E}[_margin_topk][_train{P}][_learnable_prompt]`, mirroring the
generator's `compose_setup_id`. The step and epsilon grids are swept, so neither
is a fixed set, and a decimal point becomes `p` (`eps0p02`, `train12p5`). The
`_train{P}` component appears only when the attack-train fraction is below 1.0,
and it is preserved: a 20% run and a full run are different setups. The
normalized setup ID drops only `_learnable_prompt`, which is already carried by
the prompt-mode axis. `attack_train_fraction` is also kept as a result column.

and separate scope ZIPs such as
`canonical_clip_per_dataset_<datasets>_<setup_id>.zip`. A bundle must contain:

- `attack_manifest.csv`;
- `evaluation_test_indices.csv` (directly, or in the setup's `protocol/` folder);
- the `.pt` perturbations referenced by the manifest.

The reader accepts `noise_file`, `perturbation_file`, `delta_file`, or
`artifact_path`, and accepts scope names with or without the `per_` prefix.
`loss_formulation`, prompt provenance, setup ID, source/target dataset, scope,
direction, and loss mode are preserved as result axes.

Validation is deliberately strict:

- fixed protocol IDs must exist in the mounted target dataset;
- attacked IDs must equal the source-label subset of the fixed cohort;
- direction and source/target labels must agree;
- checksums, tensor shapes, finite values, and the L-infinity budget are checked;
- per-image tensors must carry a unique `sample_ids` list with exact alignment.

## Metrics

All performance metrics use continuous scores and a `0-100` scale:

- image AUROC, average precision, and F1-max;
- pixel AUROC and F1-max;
- AUPRO integrated through FPR 0.30.

For every metric, `delta = clean - adversarial`, so positive values mean
degradation. The evaluator also reports clean/adversarial accuracy, FPR, FNR,
image flip rate, eligible targeted image ASR, directional score/map shifts, and
realized L-infinity distance.

Pixel decisions are reported under three frozen modes from one inference pass:

1. `fixed_0_5`;
2. `image_f1` (the clean category-level image F1 threshold, applied to pixels
   only as an ablation);
3. `clean_pixel_f1` (a separately calibrated clean category-level pixel F1
   threshold).

Target-region success uses the recorded fixed region for normal-to-abnormal and
the ground-truth defect mask for abnormal-to-normal. Normal-to-abnormal also
gets the newer location-free Top-K metric. Only pixels cleanly predicted as the
source class are eligible, and an image succeeds when at least 50% of eligible
pixels flip. Both category-macro and count-weighted micro rates are exported.

Thresholds are calibrated only from the clean fixed evaluation cohort and then
frozen. Because this uses labeled evaluation data, these are benchmark oracle
operating points, not deployment calibration.

## Official model defaults

AnomalyCLIP uses the official `ViT-L/14@336px` path, 518-pixel inputs, seed
111, prompt depth/context values 9/12/4, DPAM layer 20, feature levels
6/12/18/24, temperature 0.07, and Gaussian sigma 4. MVTec and VisA use their
corresponding released AnomalyCLIP checkpoints.

AA-CLIP uses the official `ViT-L-14-336` OpenAI checkpoint, 518-pixel inputs,
seed 111, residual weights 0.1/0.1, text/image adaptation depths 3/6, levels
6/12/18/24, and CLI-default `relu=False`. Its industrial 7x7, sigma-1 blur is
performed inside the official similarity-map function, so the shared evaluator
sets `gaussian_sigma=0` for AA-CLIP. Official category-level min/max score
aggregation is fitted on clean predictions and frozen before adversarial
evaluation. For zero-shot evaluation, the MVTec target uses `TrainOnVisA`
weights and the VisA target uses `TrainOnMVTec` weights.

AdaCLIP uses the official `ViT-L-14-336` OpenAI backbone, 518-pixel inputs,
seed 111, prompting depth/length 4/5, prompting type `SD`, prompting branch
`VL`, HSF aggregation with 20 clusters, and the feature hierarchy derived from
the model config as `layers//4 * {1,2,3,4}`. Its released weights are published
on the authors' HuggingFace Space (the README links Google Drive, which cannot
be fetched unattended), so the adapter downloads and checksum-verifies them on
first use and caches them afterwards. Evaluation is zero-shot exactly as in the
official `test.sh`: the MVTec target uses the `visa_clinicdb` weights and the
VisA target uses the `mvtec_colondb` weights. AdaCLIP returns full-resolution
maps and its official evaluation applies `gaussian_filter(sigma=4)` to them, so
the shared evaluator keeps `gaussian_sigma=4.0`.

The official HSF module constructs scikit-learn `KMeans` without a
`random_state`. The adapter pins that existing instance to `hsf_seed` (default:
the model `seed`, therefore 111) so image scores do not depend on evaluation
order or prior NumPy RNG use. This changes only KMeans initialization
reproducibility; HSF, its 20 clusters, and the official 0.2 score-fusion weight
remain unchanged.

AdaCLIP requires CUDA: its prompt layers cast the learned context with `.half()`
and the official evaluation runs under `torch.cuda.amp.autocast`. Its official
`test.py` refuses any batch size above one because the text-prompt layer is not
batch-safe, so the adapter forwards a single image at a time whatever
`batch_size` is set to.

FAPrompt uses the official `ViT-L/14@336px` path, 518-pixel inputs, seed 111,
prompt depth/context values 9/12/4, DPAM layer 20, temperature 0.07, and the
hardcoded 5/2 positive/negative context lengths with ten negative prompts. Its
score is `0.5 * (text_probs + map_max_score)` over two similarity passes, the
second conditioned on each image's own top-10 patch tokens, so images are
evaluated one at a time as in the official `test.py`. Its official
`gaussian_filter` uses **sigma 10**, so the shared evaluator sets
`gaussian_sigma=10.0` for FAPrompt. The released weights are published only as a
Google Drive folder, so the adapter fetches them by pinned file ID through
`gdown` and verifies a pinned sha256; the MVTec target uses `train_on_visa` and
the VisA target uses `train_on_mvtecad`.

Crane uses the official `ViT-L/14@336px` path, 518-pixel inputs, seed 111,
prompt depth/context values 9/12/4, `replace_with_EAttn` to layer 20 with
`qq+kk+vv` attention, and score-base pooling at alpha 0.5. This is the base
`Crane` row of the paper's Table 1, which `test.sh` runs with
`--dino_model none --soft_mean True --features_list 6 12 18 24 --epoch 5`; the
`Crane+` variant additionally loads DINOv2 and is not covered. Crane ships its
released checkpoints inside the repository, so cloning it is the download. The
MVTec target uses `trained_on_visa_crane` and the VisA target uses
`trained_on_mvtec_crane`, exactly as `test.sh` does.

APRIL-GAN uses the settings of the official `test_zero_shot.sh`, not the
argparse defaults, which describe its few-shot `ViT-B-16` run: `ViT-L-14-336`
with OpenAI weights, 518-pixel inputs, seed 10, feature levels 6/12/18/24, and
logit scale 100. Its `test.py` sums the per-layer maps and applies no Gaussian
filter at all, so the shared evaluator sets `gaussian_sigma=0` for APRIL-GAN.
Both released checkpoints ship in-repo under `exps/pretrained`, so cloning it is
the download; the MVTec target uses `visa_pretrained` and the VisA target uses
`mvtec_pretrained`.

FB-CLIP uses the official `ViT-L/14@336px` path, 518-pixel inputs, seed 111,
prompt depth/context values 9/12/4, and the `test.sh` feature layers
1/6/12/18/24, which override the argparse default. It returns patch-resolution
maps that the shared evaluator upsamples and blurs at `gaussian_sigma=4.0`,
matching the official loop. Its released weights are Drive-only, so the adapter
fetches them by pinned file ID and verifies a pinned sha256; the MVTec target
uses `train_on_visa` and the VisA target uses `train_on_mvtec`. The adapter
rejects a checkpoint without `model_trainable_params`, where the official script
only warns and silently keeps randomly initialized tensors.

Tipsomaly follows the official `reproduce.sh`: the TIPS `l14h` backbone,
518-pixel inputs, seed 111, industrial fixed prompts, 8 learnable prompt tokens
with `concat` learning, decoupled prompting, and local-to-global aggregation.
TIPS is trained **without image normalization** (`create_transforms_tips` uses
mean 0 and std 1), so the adapter leaves inputs in `[0, 1]`. Its
`regrid_upsample_smooth` bilinear-upsamples and then applies
`gaussian_filter(sigma=4)`, which the shared evaluator reproduces at
`gaussian_sigma=4.0`. The official loop reports two image scores, one per TIPS
class token; `cls_token_index` (default 0) selects which one is evaluated.
Learned prompts ship in-repo under `workspaces/`, so cloning it is the download;
the MVTec target uses `trained_on_visa_default` and the VisA target uses
`trained_on_mvtec_default`.

VCP-CLIP uses the settings of the official `test.sh`, which match its argparse
defaults: `ViT-L-14-336` on the exact OpenAI `ViT-L-14-336px.pt` backbone,
518-pixel inputs, seed 333, feature levels 6/12/18/24, `prompt_len` 2,
`deep_prompt_len` 1, `total_d_layer_len` 11, and `use_global` enabled. Its
`calcuate_metric` blends the two layer-averaged maps at `alpha=0.2`, blurs the
blend at **sigma 8**, and only then min-max normalizes each category's maps and
image scores; because the blur precedes that normalization, the adapter applies
it internally and the shared evaluator sets `gaussian_sigma=0` for VCP-CLIP.
Image scores are the summed means of each unblurred map's top-2000 pixels. The
category-level min/max aggregation is fitted on clean predictions and frozen
before adversarial evaluation, as for AA-CLIP. The backbone is downloaded from
OpenAI and the released weights from Google Drive, both checksum-verified; the
MVTec target uses `train_visa` and the VisA target uses `train_mvtec`.

FiLo uses the official `test.sh` and its argparse defaults: CLIP `ViT-L-14-336`
with OpenAI weights at 518 pixels, feature levels 6/12/18/24, `n_ctx` 12, and a
fine-tuned Grounding DINO (`GroundingDINO_SwinT_OGC`) at box/area thresholds
0.25/0.7. Grounding DINO is captioned with the GPT-written descriptions of the
category, the highest-scoring anomaly box selects one of nine position words for
the position-enhanced prompts, and the final map is damped to 0.7 outside the
proposed boxes. Its `GaussianBlur(3, 4.0)` runs per layer inside the model, so
the shared evaluator sets `gaussian_sigma=0` for FiLo, and the image score is
`(abnormal text logit + map max) / 2` taken before the box damping. All four
weights are pulled from the authors' HuggingFace repository and verified against
its LFS digests; the MVTec target uses `filo_train_on_visa` with
`grounding_train_on_visa` and the VisA target the MVTec-trained pair. The FiLo
head stores the frozen CLIP alongside its trained modules, so each target needs
about 3.3 GB of weights.

Two FiLo details differ from a naive reading of its `test.py`. Its argparse
exposes `--text_threshold`, but `test.py` immediately overwrites it with
`box_threshold`, so only `box_threshold` has any effect and the adapter exposes
only that. And the official loop re-reads each image from disk for Grounding
DINO while feeding CLIP the transformed tensor; the adapter drives both branches
from the evaluated tensor instead, since re-reading the file would bypass the
perturbation entirely. That branch's aspect-preserving resize is a no-op on the
square 518-pixel cohort images, so the only remaining difference is its ImageNet
normalization, which the adapter applies. Grounding DINO also calls a compiled
CUDA extension for deformable attention whenever its tensors are on a GPU and
raises when it was never built, so the adapter falls back to the reference
PyTorch implementation the same module already uses on CPU; `runtime_metadata`
records which path ran. FiLo scores one image at a time whatever `batch_size` is
set to, because Grounding DINO picks a position word per image.

Bayes-PFL uses the settings of the official `test.sh`: `ViT-L-14-336` on the
pinned OpenAI `ViT-L-14-336px.pt` backbone at 518 pixels, features 6/12/18/24, a
prompt bank of 3 prompts with 5 context and 5 state tokens, 10 planar flows and
10 Monte Carlo samples, and seed 333. That gives 30 prompt draws per image.
`calcuate_metric_pixel` blurs the maps at sigma 8 and does **not** normalize
them, so the shared evaluator keeps `gaussian_sigma=8.0`. The image score fuses
two per-category min-max normalized parts at 0.5/0.5: the Monte Carlo averaged
text probability, and the mean of the unblurred map's top-k pixels, where k is 20
for the fine-grained VisA categories (`capsules`, `cashew`, `chewinggum`,
`macaroni1`, `macaroni2`, `pipe_fryum`, `screw`) and 2000 otherwise. That
normalization is fitted on the clean cohort and frozen for the adversarial pass.
The MVTec target uses `train_visa` and the VisA target uses `train_mvtec`; both
are Drive-hosted and checksum-verified.

Two properties of Bayes-PFL need care under a matched clean-versus-adversarial
protocol. Its prompt sampling is **stochastic**, so the adapter resets the RNG to
the seed before each image; a given image then draws the same 30 prompts on both
passes, where the official loop lets the RNG advance across the test set and
would make the two passes incomparable. And `forward_ensemble` caches the
image-agnostic state latents from the first test image it ever sees and reuses
them for the rest of the run; that cache is deliberately left alone, so the
latents are fitted on clean data and frozen for the adversarial pass. Its
image-score softmax is also taken over the flattened batch, which is only correct
for a batch of one, so the adapter scores one image at a time whatever
`batch_size` is set to.

AF-CLIP uses the zero-shot rows of the official `test.sh` and its argparse
defaults: CLIP `ViT-L/14@336px` at 518 pixels, feature layers 6/12/18/24 each
aggregated over 1x1, 3x3 and 5x5 Gaussian neighbourhoods, a 12-token learned
state prompt in front of the fixed `without defect.` / `with defect.` pair, the
trained adaptor on the patch tokens, and seed 122. The zero-shot branch leaves
`memorybank` unset, so `detect_forward` reduces to `detect_forward_seg` and the
`alpha` flag never applies; the adapter refuses to run if the memory bank is
somehow populated. Scores and maps are used raw — the official loop performs no
normalization anywhere — and its bilinear upsample plus `gaussian_filter(sigma=4)`
is exactly what the shared evaluator does, so `gaussian_sigma=4.0`. The weights
are committed to the repository as `<source>_prompt.pt` and `<source>_adaptor.pt`
named after the dataset they were trained on, so cloning it is the download; the
MVTec target uses the VisA-trained pair and the VisA target the MVTec-trained
one. They are pickled tensors and modules rather than state dicts, so the adapter
loads them with `weights_only=False`.

CoPS uses the settings of the official `shell/test.sh`: CLIP `ViT-L/14@336px` at
518 pixels with DPAM at layer 24, prompt depth/context 8/12/4, a 6-vector
prototype bank, feature layer 24 only, and seed 0. Its weights are committed to
the repository, so cloning it is the download; the MVTec target uses the
VisA-trained `epoch_10` and the VisA target the MVTec-trained `epoch_5`. Its
`gaussian_filter(sigma=4)` runs on the upsampled map, which the shared evaluator
reproduces at `gaussian_sigma=4.0`.

Three CoPS details are easy to get wrong. `get_fullsize_map` is called without
its `mode` argument, so it falls through to the training branch and the map is
upsampled with **nearest** interpolation rather than bilinear. The `alpha`/`beta`
fusion weights (0.35/1.0 versus 0.26/0.9) and the neighbourhood kernel (3 versus
5) branch on whether "visa" appears in the **test** dataset name, not the
trained one. And the prototype distances are min-max normalized over the whole
batch tensor, which only matches the official `batch_size 1` when images are
scored individually, so the adapter scores one at a time. Its ICTS branch also
samples 10 latents per image, so the RNG is reset to the seed before each image,
exactly as for Bayes-PFL.

MRAD uses the README quick start and the `test.py` defaults: CLIP
`ViT-L/14@336px` at 518 pixels with DPAM at layer 24, prompt depth/context
9/12/4, the `mrad-clip` variant, sigma 4, and fusion weight `k=0.7`. The image
score combines the CLIP-side probability with the mean of the top 1 percent of
map pixels, and because that top-k is taken on the blurred map the blur runs
inside the adapter and the evaluator keeps `gaussian_sigma=0`. `feature_map_layer`
defaults to `[0, 1, 2, 3]` and the loop keeps only indices at or above its last
element, so exactly one projection layer contributes. Six Drive files are needed
per run - one checkpoint per target plus the image and patch memory banks for
each source dataset - and all are checksum-verified.

MRAD's memory banks deserve a note, because a retrieval model is exactly the
shape of method that breaks a matched protocol. They do not: `test.py` loads
them with `load_cache=True` before the test loop and nothing writes to them while
images are scored, so a prediction never depends on which images preceded it.
The banks are built from the dataset that is not being evaluated, which is what
makes the protocol zero-shot.

WinCLIP uses `eval_WinCLIP.py` at its defaults: the `ViT-B-16-plus-240` backbone
with `laion400m_e32` weights, 240-pixel inputs, window scales (2, 3), a
400-pixel output grid, and seed 111. It is **training-free**, so there is no
checkpoint at all beyond the backbone, which open_clip downloads on first use.
The image score is the maximum of the map, exactly as `metric_cal` computes it,
and the official Gaussian filter is commented out, so `gaussian_sigma=0`.

Two WinCLIP properties are reproduced deliberately rather than simplified. With
`k_shot` at zero the visual gallery stays empty, so the `textual_visual` fusion
evaluates `1 / (1/t + 1/t)`, i.e. **half** the textual map; that halving is
monotonic and cannot move AUROC, but the frozen thresholds are calibrated on
absolute values, so it matters here. And WinCLIP is a 240-pixel model while the
cohort arrives at 518 pixels, the grid the perturbations are defined on: the
adapter performs the official bicubic resize down to 240 itself, which means the
L-infinity budget applies at 518 and is attenuated by that downsample. That is a
property of evaluating a 240-pixel model against a 518-pixel perturbation, and
`runtime_metadata` records both sizes.

## Few-shot adapters

Three adapters have few-shot counterparts: `winclip_fewshot`, `afclip_fewshot`
and `aprilgan_fewshot`. Each scores the cohort against a small set of normal
**training** images. That reference set is built once per category and never
touched while the cohort is scored, so unlike a transductive method it never
makes one prediction depend on which cohort images came before it. The
references are always the clean originals, loaded through the same `load_image`
the evaluator uses for the cohort: perturbing them would change what "normal"
means between the two passes. Both roots therefore have to be repeated inside
`model_kwargs_by_target` so the adapter can find them.

The three official selection protocols differ, and each is reproduced rather
than unified:

- **WinCLIP** commits its selection to the repository under
  `datasets/seeds_mvtec/<category>/selected_samples_per_run.txt`, which pins the
  exact file stems per run and shot count, so the MVTec runs reproduce the
  paper's exactly. `k_shot` must be 1, 5 or 10 and `experiment_indx` 0, 1 or 2.
  VisA ships no such file — the official loader draws `random.sample` under the
  run seed (111/333/999) — so the adapter draws deterministically from the
  sorted normal training images with that seed.
- **AF-CLIP** uses `np.random.choice` without replacement under the process
  seed, and its few-shot script passes `--seed -1`, a fresh random seed per
  repeat with five repeats averaged. There is no canonical selection, so
  `shot_seed` pins it.
- **APRIL-GAN** uses `torch.randint` under `--seed 42`, which samples **with
  replacement**, so a shot may legitimately repeat. That is reproduced, not
  corrected.

Whichever protocol applies, the drawn file names land in `runtime_metadata`
under `reference_images`, so a result always records the exact reference set.

Two of the three change more than the map. AF-CLIP's `detect_forward` stops
being the zero-shot branch and returns `memory + alpha * segmentation` for both
the map and the image score, which is where its otherwise-dead `alpha` of 0.1
applies. APRIL-GAN overrides `few_shot_features` to 6/12/18/24 where the
argparse default is the few-shot ViT-B 3/6/9, adds a retrieval map to the
zero-shot map, and changes the image score to
`0.5 * (text probability + per-category min-max normalized map maximum)`; that
normalization is fitted on the clean cohort and frozen for the adversarial pass.
WinCLIP's gallery, meanwhile, turns the `textual_visual` fusion into the real
harmonic mean instead of the halved textual map the zero-shot path reduces to.

`build_image_feature_gallery` and `store_memory` both reset their state on every
call, so the official loops only work because their training batch holds every
shot at once; both adapters build each category's reference in a single call for
the same reason.

Batch size is an execution-only setting and may be reduced for Kaggle without
changing model outputs.

## Install and run

```bash
python -m pip install -e ".[anomalyclip]"
cp configs/anomalyclip.example.json anomalyclip.json
# edit the paths
fpeval --config anomalyclip.json
```

The AnomalyCLIP extra includes `thop`, which the official repository imports
while loading its model implementation.

For AA-CLIP, install with `python -m pip install -e ".[aaclip]"`, start from
`configs/aaclip.example.json`, and run the same command. For AdaCLIP, install with
`python -m pip install -e ".[adaclip]"` and start from
`configs/adaclip.example.json`; its `checkpoint` accepts either a local path or
one of the released names `visa_clinicdb`, `mvtec_colondb`, or `all`, and a
name is downloaded into `download_root` and verified against a pinned sha256.
Every other adapter follows the same shape: install `".[<model>]"` and start
from the matching `configs/<model>.example.json`. The FAPrompt, FB-CLIP,
VCP-CLIP, Bayes-PFL, and MRAD extras pull `gdown` because those weights are
Drive-only; if Drive rate-limits a download, fetch the files by hand and pass
their paths. FiLo takes its weights from HuggingFace instead, and its extra
carries Grounding DINO's own dependencies. AF-CLIP and CoPS need no download
beyond the OpenAI backbone, since their weights are committed to their
repositories, and WinCLIP needs no checkpoint at all. These extras install model runtime libraries without
replacing the environment's PyTorch with an old repository pin. The AA-CLIP extra also includes `ipdb` and `regex`, which
the official repository imports from `model/`, `forward_utils.py`, and its
tokenizer but omits from its own `requirements.txt`.

For local/server runs, each adapter expects an existing checkout of its official
repository and released checkpoints. Target-specific paths are configured under
`model_kwargs_by_target`.

For Kaggle, use the `notebooks/zero_shot/kaggle_<model>.ipynb` matching the
adapter
(`anomalyclip`, `aaclip`, `adaclip`, `faprompt`, `crane`, `aprilgan`, `fbclip`,
`tipsomaly`, `vcpclip`, `filo`, `bayespfl`, `afclip`, `cops`, `mrad`, or
`winclip`), enable GPU and Internet, and attach only:

Notebooks and run scripts are split by regime: `notebooks/zero_shot/` and
`scripts/zero_shot/` hold everything above, and `notebooks/few_shot/` and
`scripts/few_shot/` hold `winclip_fewshot`, `afclip_fewshot` and
`aprilgan_fewshot`.

- MVTec AD;
- VisA;
- the generated perturbation dataset containing `setups/`.

The notebooks clone this evaluator and the selected official model repository
into `/kaggle/working` automatically. They discover dataset mounts by structure,
so Kaggle slug/capitalization differences such as
`MVTec-AD/mvtec_anomaly_detection`, `VisA-AD/VisA_20220922`, and
`perturbation-generated/setups` do not require path edits. The AA-CLIP notebook
also downloads its public adapter-checkpoint dataset when it is not attached and
downloads/checksum-verifies the exact OpenAI ViT-L/14@336px base model. The
AdaCLIP notebook downloads and checksum-verifies the released AdaCLIP weights.

The first notebook cell sets `MVTEC_INPUT`, `VISA_INPUT`, and
`PERTURBATION_INPUT`, then selects prompt modes, normalized setup IDs, scopes,
categories, directions, and losses. Source datasets are not a notebook option;
all source datasets present in the selected manifests are included. With
`SETUP_IDS=None`, the evaluator
recursively processes every available setup under both `frozen_prompt/` and
`learnable_prompt/`; filters are applied only after the mounted inventory is
printed.

Both notebooks calculate thresholds automatically during their evaluation
run. For every target/category, clean image F1 and clean pixel F1 thresholds
are calibrated first, frozen, written to `thresholds.json`, and then reused for
all adversarial conditions. No separate threshold notebook is required.

Use `targets`, `scopes`, `prompt_modes`, `setup_ids`, `source_datasets`,
`categories`, `directions`, `loss_modes`, or `loss_formulations` to run a
subset. `max_conditions: 1` is useful for plumbing checks and must not be
reported as a complete benchmark.

## Outputs

```text
<output_root>/<model>/
  summary.csv
  category_metrics.csv
  per_image.csv
  thresholds.json
  run_config.json
  manifest_snapshot.json
  predictions/                 # only when save_predictions=true
  extracted_attacks/           # ZIP cache; ignored by numerical outputs

<output_root>/<model>_separated/
  setups/
    <frozen_prompt|learnable_prompt>/
      <setup_id>/
        datasets/
          <source>_to_<target>/
            <scope>/
              numerical/       # CSV/JSON results for only this slice

<output_root>/<model>_samples_separated/
  setups/<prompt_mode>/<setup_id>/datasets/<source>_to_<target>/<scope>/
    <threshold_mode>/<compact_condition>/<selection>__<protocol_id>/...

<output_root>/<model>.zip
<output_root>/<model>_separated.zip
<output_root>/<model>_samples_separated.zip
```

`summary.csv` has one row per condition and pixel-threshold mode.
`category_metrics.csv` contains directly computed category rows, and
`per_image.csv` retains exact IDs, predictions, attack eligibility/success,
pixel diagnostics, shifts, and realized norms. Low-resolution prediction maps
can optionally be saved as compressed NPZ files; full 518-pixel maps are used
for metrics regardless.

The consolidated model directory remains the authoritative full-run result.
Samples are kept in their own consolidated directory, and both numerical
results and samples also receive independent input-shaped separated trees.
The numerical directories and separated sample directory are archived
automatically after a successful run. The redundant consolidated-sample ZIP
is intentionally not created because it contains the same sample files. The
disposable `extracted_attacks` input cache is excluded from the result ZIP.

Structured per-dataset sample conditions use the compact name
`<direction>__<loss_formulation>__<loss_mode>` because prompt, setup, dataset
pair, scope, and threshold are already present in parent directories.
Per-category and per-image conditions prefix the category to prevent naming
collisions. Flat consolidated sample exports retain the full condition ID.

### Qualitative samples

Each condition selects representative attacked images with two independent
families of criteria.

Threshold-based, ranked by the targeted pixel flip rate at the exported
threshold:

| selection | chosen because |
| --- | --- |
| `strongest_success` | highest targeted pixel flip rate among successful attacks |
| `median_success` | median successful attack, i.e. a typical rather than a best case |
| `worst_failure` | least-effective attack that still had pixels eligible to flip |

Threshold-free, needing no pixel threshold at all:

| selection | chosen because | drives |
| --- | --- | --- |
| `largest_score_shift` | largest directional image-score shift | image AUROC, image AP, image F1-max |
| `largest_map_shift` | largest directional anomaly-map shift | pixel AUROC, pixel F1-max, AUPRO |

Directional shifts are signed so that positive always means *toward the attack
target*. An image chosen by several criteria is exported once and carries every
reason. Because the threshold-free picks do not depend on any threshold, they
repeat identically across threshold modes; exporting a single
`qualitative_threshold_modes` entry avoids duplicating them.

Normal-to-anomalous examples use the same location-free top-k region as the
reported success metric; anomalous-to-normal examples use the ground-truth
anomaly region.

Every sample folder holds the clean and adversarial images, a x10 difference
image, both heatmaps and overlays, the ground-truth mask, the target-region
mask, both pixel predictions, the successful-flip map, a signed heatmap
difference, `metrics.json`, and a `description.md`. `metrics.json` repeats the
full per-image row plus a machine-readable `selection_reasons` list;
`description.md` is the human-readable version:

```markdown
# largest_score_shift - `test/bottle/good/001`

- **Condition**: `frozen_prompt__steps500_eps2__mvtec__mvtec__per_dataset__all__normal_to_abnormal__ce_focal_dice__global`
- **Direction**: normal_to_abnormal (source label 0 -> target label 1)
- **Image**: mvtec/bottle/good (ground-truth label 0)
- **Pixel threshold mode**: `fixed_0_5` (threshold 0.5)
- **Target region**: location-free top-k region

## Why this image was selected

- **largest_score_shift** (threshold-free): Largest threshold-free shift of the
  image score toward the target class (+0.000057) among 6 attacked images. This
  quantity needs no pixel threshold and is what drives image AUROC, image AP,
  and image F1-max.

## Before vs after

| quantity | clean | adversarial | change |
| --- | --- | --- | --- |
| image score | 0.437253 | 0.437311 | +5.73993e-05 |
| image prediction | 0 (normal) | 0 (normal) | unchanged |
| anomaly map mean | 0.437254 | 0.437311 | +5.74291e-05 |
| anomaly map max | 0.441194 | 0.444205 | +0.00301051 |

- targeted image success: **no**
- target-direction score shift: +5.73993e-05
- target-direction map shift: +5.73590e-05

## Targeted pixels in the location-free top-k region

- pixels in region: 53665
- eligible (clean-predicted as the source class): 53665
- flipped to the target class: 0 (0.00%)
- counted as a successful attack: **no**

## Perturbation

- realized L-inf: 0.08
- mean absolute change: 0.0238158
- PSNR: 30.54 dB
```

Sample volume grows with the number of conditions, not with the amount of data
behind them, so the `per_category` and `per_image` scopes would otherwise export
thousands of near-duplicate folders over the same images. `max_sample_conditions`
caps how many conditions keep samples, applied **per scope** and ranked by
targeted attack success rate, which no pixel threshold affects. Weaker
conditions are deleted as soon as a stronger one arrives, so at most that many
survive per scope and peak disk use stays bounded. Set
`save_qualitative_samples: false`, `write_separated_results: false`, or
`create_output_archives: false` to turn off an export. Custom destinations are
available through `separated_output_root` and
`separated_samples_output_root`.

## Adding another target model

Implement `ModelAdapter` in `src/fpeval/adapters/<name>.py`, decorate it with
`@register_adapter("<name>")`, and import it from `adapters/__init__.py`. The
adapter receives `[B,3,H,W]` RGB tensors in `[0,1]` plus category names and
returns one image score and one 2-D anomaly map per image. No attack, dataset,
threshold, or metric code needs to change.

Models whose official evaluation normalizes or fuses over a whole cohort
override the `postprocess_*` hooks instead of doing it inside `predict`. The
`*_with_reference` variants receive the frozen clean predictions, which is how a
cohort statistic stays fitted on clean data through the adversarial pass. The
score hooks are handed `map_mins`, `map_maxs`, and the postprocessed `maps`
themselves, so a model needing another map statistic — Bayes-PFL takes the mean
of the top-k pixels — can compute it without a second inference pass.

## Development checks

```bash
python -m pip install -e ".[dev]"
pytest
```
