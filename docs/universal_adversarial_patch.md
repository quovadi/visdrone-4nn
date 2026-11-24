# Stage 5 — universal untargeted adversarial patch

Base: checked Stage 4 commit `e3a3eaadf772acdf754e65b1bf1f3be3c2c0f155`.
New code only. FGSM, PGD, Square, trained checkpoints, data-v1, old results and
Kaggle datasets are not modified. No real experiment or dataset upload runs
implicitly. The primary experiment consists of 12 independent patches (4 baseline
architectures × 3 areas) and 48 fixed-patch evaluations (each on 4 targets).

## Protocol and implementation

`attacks.adversarial_patch` implements RGB **replacement**, never addition. A
single learned tensor is shared by every crop in a run. Side length is
`round(sqrt(area_fraction * H * W))`; allowed fractions .05/.10/.15 give sides
50/71/87 at 224×224. Realized fractions are saved because rounding changes area
slightly. The object crop, not the full frame, is the attacked input.

Render after the existing deterministic timm geometry, before the differentiable
PixelModel normalization. All four current models use 224×224. Different native
patch resolution is resampled bilinearly to the requested area if the helper is
used on another resolution; this is resolution adaptation, not random scaling.
No rotations, random scales, affine transforms or extra EOT are introduced.
Existing model-specific geometry/mean/std differences remain a comparison factor.

Optimization defaults: Adam, lr=.05, 500 attempted iterations, batch=8, uniform
[0,1] initialization, seed=42, train-size=4096. All are CLI-configurable (SGD and
gray=.5 initialization also supported). Only patch pixels are optimizer parameters.
Maximize ordinary unweighted CE on **clean-correct source-model train objects**
by minimizing negative CE. Model is frozen/eval, FP32, AMP/TF32 disabled. Images
are detached; autograd targets only the shared patch. Clamp patch to [0,1] after
each update. Classifier parameters, requires_grad flags and eval/training flags
are preserved after the job, including failures; parameter gradients are not
accumulated. A no-clean-correct batch is logged with loss=null and no update.
A run with no usable updates or entirely zero patch gradients fails rather than
publishing a supposedly optimized patch.

Iterations are not epochs. Each iteration uniformly samples a batch without
replacement within the batch, from the fixed train selection, using a seed/step
counter. Selection is not class-balanced and tiny crops are not removed. Log the
positive CE to maximize, optimizer-update flags, clean-correct count and gradient
L1. Loss need not be monotone across changing batches/positions.

Positions use a local generator keyed by seed, sample_id, area, geometry and
optimization step. Validation uses step=None, so placement is fixed per object,
independent of batch/order, source patch and target model. All legal positions,
including the last lower/right anchor, are possible. New optimization iterations
make new reproducible draws, which may coincide by chance. Use the same validation
placement seed for the complete experiment (default 42).

## Split safety and frozen validation

Optimization accepts only **train.json with VisDrone2019-DET-train image paths**.
Validation paths are rejected even if the file is renamed train.json. CLI also
checks the train index hash against the source classifier's frozen metadata.
No validation predictions, labels or loss are used for optimization, early stopping,
patch selection, LR tuning or choosing the final patch. The final configured train
iteration's patch is saved; no best-by-validation selection exists.

`prepare-subset` writes a NEW Stage 5 subset artifact; it never rewrites the
original validation.json. Research size defaults to 1024. Supply the previous
Square pilot's subset.json to retain its exact 256 IDs and add 768 objects from
the remainder, using Hamilton/largest-remainder proportional ground-truth strata.
If the supplied pilot has a different size, all its valid unique IDs are retained
and the actual counts are recorded, not falsely called 256. The subset reader
checks identities/boxes/labels/positions, class distribution and source-index hash.
The full validation index is used only to select/validate the fixed cohort; no
holdout is opened. Selection is independent of model predictions/attack results.

**The real Square pilot file/data are not present in the Codex workspace.** The
CLI accepts --square-subset explicitly. If omitted, a fresh stratified selection
is annotated inheritance=null; no guess of the historical 256 is made. Actual
1024-object freezing must be run with the real mounted inputs, then reused across
all models/patches. Small --size values exist for synthetic/debug pilots and must
be labelled partial. Full research reports require 1024 objects and all 48 cells.

## Model loading and evaluation

Existing load_classifier verifies MODEL_RUN/commit.json ↔ checkpoint SHA-256,
strictly loads all trained weights with pretrained=False, and validates recorded
preprocessing. Compact tensors use weights_only=True. Full classifier checkpoint
requires explicit --checkpoint-format full --trusted-full-checkpoint. Stage 5 CLI
uses baseline-v1 only, so the separate 50k/lr=3e-5 Base v2 cannot silently replace
the baseline matrix. Synthetic API smokes are explicitly flagged in provenance.

Each fixed source patch can be evaluated on each of the four targets without an
optimizer, gradients or adaptation. Target clean-correct objects define that
cell's fixed ASR denominator. Clean errors are not successful attacks. Global
accuracy and adversarial accuracy divide by all objects; macro-F1/per-class F1
reuse all ten labels via PairedMetrics/ClassificationMetrics. Missing F1 denominator
is 0; undefined per-class ASR or accuracy is null/empty CSV, not a fabricated zero.
Per-class CSV records true-class support, target clean-correct count, successes,
ASR, adversarial accuracy and clean/adversarial F1.

## CLI: Nano pilot, then manual 12/48 experiment

From the checkout, install requirements and choose your actual mounted paths:

```bash
python -m attacks.adversarial_patch --help
ROOT=/kaggle/input/visdrone2019-det/VisDrone
INDICES=/kaggle/input/visdrone-data-v1/data-v1
NANO=/kaggle/input/visdrone-baselines/convnext_nano/RUN_NAME
SQUARE_SUBSET=/kaggle/input/square-pilot/RUN_UUID/subset.json
OUT=/kaggle/working/stage5

python -m attacks.adversarial_patch prepare-subset --validation-index "$INDICES/validation.json" --square-subset "$SQUARE_SUBSET" --size 1024 --seed 42 --output "$OUT"
# Use the emitted path for ALL Stage 5 validation/transfer jobs:
SUBSET="$OUT/subsets/SUBSET_UUID/subset.json"

# Preliminary Nano pilot, one 5% patch, train data only:
python -m attacks.adversarial_patch optimize --model-run "$NANO" --checkpoint "$NANO/best_weights.pt" --root "$ROOT" --train-index "$INDICES/train.json" --train-size 256 --area 0.05 --steps 50 --lr 0.05 --batch-size 4 --optimizer adam --initialization uniform --seed 42 --device cuda --output "$OUT"
PATCH="$OUT/patches/PATCH_UUID"

python -m attacks.adversarial_patch evaluate --model-run "$NANO" --checkpoint "$NANO/best_weights.pt" --patch "$PATCH" --root "$ROOT" --validation-index "$INDICES/validation.json" --subset "$SUBSET" --placement-seed 42 --batch-size 4 --device cuda --output "$OUT"
```

Inspect RGB patch/overlay numerically, loss and nonzero gradients, clean-correct
train counts, frozen weights and exact checkpoint/subset identities. Check GPU
memory and geometry before longer optimization. Position-only training is the
intended current protocol. Do not choose patch hyperparameters by validation ASR;
fix them before the research matrix (use train-only diagnostics for debugging).

For the research matrix, explicitly invoke optimize once for each model and each
area (.05/.10/.15), keeping the intended training subset/steps/lr/batch/init/seed
common. Each invocation initializes a NEW independent tensor and optimizer. There
is no automatic 12-job launcher. Use the four baseline MODEL_RUN paths and retain
the 12 emitted PATCH_UUID paths.

Transfer registry example (paths must be replaced with real values):

```json
{
  "convnext_nano": {"model_run": "/kaggle/input/baselines/nano/RUN", "checkpoint": "/kaggle/input/baselines/nano/RUN/best_weights.pt"},
  "tinyvit": {"model_run": "/kaggle/input/baselines/tiny/RUN", "checkpoint": "/kaggle/input/baselines/tiny/RUN/best_weights.pt"},
  "convnext_base": {"model_run": "/kaggle/input/baselines/base/RUN", "checkpoint": "/kaggle/input/baselines/base/RUN/best_weights.pt"},
  "vit_base": {"model_run": "/kaggle/input/baselines/vit/RUN", "checkpoint": "/kaggle/input/baselines/vit/RUN/best_weights.pt"}
}
```

```bash
# One Nano pilot patch on the explicitly listed targets:
python -m attacks.adversarial_patch transfer --patches "$PATCH" --targets targets.json --root "$ROOT" --validation-index "$INDICES/validation.json" --subset "$SUBSET" --placement-seed 42 --batch-size 4 --device cuda --output "$OUT"

# For the final 48 cells, explicitly pass all 12 PATCH paths to --patches.
# One target classifier is loaded at a time; no additional patch optimization.
# Partial pilot report: explicitly list its evaluation directories.
python -m attacks.adversarial_patch report --evaluations "$OUT/evaluations/EVAL_UUID" --output "$OUT" --allow-partial
# Final report: pass 48 evaluation directories and omit --allow-partial.
```

The report rejects duplicate cells, mixed checkpoint identities, different frozen
cohorts/placement seeds, changed target clean predictions and mismatched source/
target checkpoints on the diagonal. It exports grouped diagonal ASR, adversarial
accuracy and adversarial macro-F1, a 4x4 ASR matrix per area, and a 3x4 grid of patch
patterns. PNG=300 DPI and PDF=vector. Missing/undefined cells in partial plots are
shown as missing, never zero. Source-row/target-column ordering is stored. Source
patch manifests/tensors are verified; no images or classifiers are loaded by report.

## Artifact layout, integrity and backup

Every job creates an exclusive Stage 5 kind/uuid directory:

- patches/UUID: patch.pt (exact CPU FP32 tensor, weights_only=True loading), patch.png
  (8-bit visualization only), metadata.json, config.json, train_subset.json,
  training_loss.csv, training_loss.png/.pdf and last_patch.pt.
- evaluations/UUID: predictions.csv (sample identity, box, true/clean/adversarial
  labels, conditional success, source/target/patch SHA, placement/side/area/seed),
  metrics.json including runtime, per_class_metrics.csv, subset.json, metadata.json.
- subsets/UUID: frozen subset.json with all IDs/labels/classes/quotas/seed/SHA and
  optional Square inheritance hashes/retained IDs.
- reports/UUID: comparison.csv, transfer_matrices.json, patch_grid and metric/ASR
  plots, input manifest digests and plot_status.json.

All include config/provenance (Git/package/code hashes), status and a final
run_manifest.json with file SHA-256. Source/target checkpoints and original image
indices are only read. CSV readers verify artifact hashes, object identities,
placements, global and per-class metrics. Patch loading validates shape/range,
source identity and the exact tensor checksum; a PNG is never used for inference.
Runtime is wall time of the job's computation/writes (optimization also includes
its requested local plots), excluding external model loading; it is not a latency
benchmark. Raw image content hashes are not computed and must be preserved by the
input dataset's own provenance.

Use `--backup-root /content/drive/MyDrive/visdrone-stage5` on optimize/evaluate/
transfer to copy each COMPLETED bundle into a separate durable root, using existing
atomic_save/write_json/file_digest primitives, verified staged copies and manifest
publication last. Existing destination runs are never adopted or overwritten.
Copies include only patch artifacts and patch-optimizer state, not model weights.
No Kaggle dataset is created or updated. A failed copy leaves the source intact
and a partial, unverified destination; retry with a fresh backup root. No old
experiments are deleted. Atomicity is per file, not fsync/multi-file transactions.

The classifier training backup schema assumes full model/optimizer/TrainConfig
and cannot safely be reused verbatim for a patch. The isolated patch schema is
therefore necessary. last_patch.pt is a small safe tensor/container snapshot of
patch, its optimizer, iteration, loss history and identities; it is written each
iteration for diagnosis/manual recovery. **Stage 5 has no optimization-resume CLI**
and never appends to completed jobs. Only completed bundles are automatically
mirrored. Losing temporary Colab/Kaggle storage mid-optimization loses unbacked
work; use a durable --output or retain snapshots manually if needed. Repeating a
job creates a new UUID; byte-identical GPU behavior across versions is not promised.

Plot failures are recorded in metadata.json or plot_status.json without corrupting
patch/evaluation data. Inspect these fields before considering visual exports
complete. A partial or failed job has no valid completed manifest and cannot be
used as a trained patch/report input. No-clean-correct or disconnected/nonfinite
training gradients cause skips or explicit failure, not validation fallback.

## Testing and scope

```bash
MPLCONFIGDIR=/tmp/visdrone-mpl python -m pytest -q
MPLCONFIGDIR=/tmp/visdrone-mpl python -m attacks.patch_smoke --output work/stage5-smoke
```

Smoke: one synthetic 3-iteration patch optimization, six toy validation crops,
two target evaluations and a partial report; artifacts are marked SYNTHETIC. It
is not a real baseline result and does not execute the 12/48 research plan.
CPU integration checks instantiate all four actual architectures with pretrained=
False and verify differentiability from patch to logits with frozen weights.
These graphs are untrained; compatibility with actual trained checkpoint bundles,
T4/Kaggle memory/throughput and dataset paths still require the manual Nano pilot.
The real 1024-object subset, real patches and real 48-cell matrix are not generated
in Codex. No sealed holdout, full VisDrone training or pretrained downloads are used.

Implementation verification in Codex: **168 passed, 1 skipped** (existing CUDA
AMP test; CPU environment), one upstream torch.jit.interface deprecation warning.
The four architecture graph checks passed without pretrained weights. Standalone
CPU smoke completed one toy patch optimization (3 nonzero-gradient steps), two
fixed-patch target evaluations, retained all supplied synthetic Square IDs, and
exported a partial report with PNG/PDF; all plot error fields were null. Patch-grid
PNG was visually inspected and explicitly labels synthetic data/missing cells.
CLI help, source compilation and git diff --check passed. All pre-existing tests
and source files remain unchanged. These are synthetic software checks, not
training or robustness measurements of the four trained VisDrone baselines.
