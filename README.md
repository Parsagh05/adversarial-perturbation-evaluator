# Fixed Perturbation Evaluator

This repository is a fresh, model-extensible black-box evaluation pipeline for
the perturbation datasets produced by the object-agnostic prompt experiments.
It does not generate or optimize attacks. The perturbation manifest and the
fixed `evaluation_test_indices.csv` are always authoritative.

The first target adapters are AnomalyCLIP and AA-CLIP. The shared evaluator owns attack
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
  canonical_clip_per_category/
  canonical_clip_per_image/
```

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
`configs/aaclip.example.json`, and run the same command. These extras install
model runtime libraries without replacing the environment's PyTorch with an
old repository pin.

For local/server runs, each adapter expects an existing checkout of its official
repository and released checkpoints. Target-specific paths are configured under
`model_kwargs_by_target`.

For Kaggle, use `notebooks/kaggle_anomalyclip.ipynb` or
`notebooks/kaggle_aaclip.ipynb`, enable GPU and Internet, and attach only:

- MVTec AD;
- VisA;
- the generated perturbation dataset containing `setups/`.

The notebooks clone this evaluator and the selected official model repository
into `/kaggle/working` automatically. They discover dataset mounts by structure,
so Kaggle slug/capitalization differences such as
`MVTec-AD/mvtec_anomaly_detection`, `VisA-AD/VisA_20220922`, and
`perturbation-generated/setups` do not require path edits. The AA-CLIP notebook
also downloads its public adapter-checkpoint dataset when it is not attached and
downloads/checksum-verifies the exact OpenAI ViT-L/14@336px base model.

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
              samples/
                <threshold_mode>/
                  <condition_id>/
                    selection_manifest.json
                    <selection>__<protocol_id>/
                      clean.png
                      adversarial.png
                      difference_x10.png
                      clean_heatmap.png
                      adversarial_heatmap.png
                      clean_overlay.png
                      adversarial_overlay.png
                      ground_truth_mask.png
                      target_region_mask.png
                      clean_pixel_prediction.png
                      adversarial_pixel_prediction.png
                      successful_target_pixel_flips.png
                      heatmap_difference.png
                      metrics.json
```

`summary.csv` has one row per condition and pixel-threshold mode.
`category_metrics.csv` contains directly computed category rows, and
`per_image.csv` retains exact IDs, predictions, attack eligibility/success,
pixel diagnostics, shifts, and realized norms. Low-resolution prediction maps
can optionally be saved as compressed NPZ files; full 518-pixel maps are used
for metrics regardless.

The consolidated model directory remains the authoritative full-run result.
By default, the evaluator also writes the input-shaped separated tree and
qualitative examples for every selected threshold mode. Each condition chooses
the strongest successful attack, a median successful attack when distinct,
and the least-effective failure. Normal-to-anomalous examples use the same
location-free top-k region as the reported success metric; anomalous-to-normal
examples use the ground-truth anomaly region. Set
`save_qualitative_samples: false` or `write_separated_results: false` to turn
off either export, and use `separated_output_root` to choose another location.

## Adding another target model

Implement `ModelAdapter` in `src/fpeval/adapters/<name>.py`, decorate it with
`@register_adapter("<name>")`, and import it from `adapters/__init__.py`. The
adapter receives `[B,3,H,W]` RGB tensors in `[0,1]` plus category names and
returns one image score and one 2-D anomaly map per image. No attack, dataset,
threshold, or metric code needs to change.

## Development checks

```bash
python -m pip install -e ".[dev]"
pytest
```
