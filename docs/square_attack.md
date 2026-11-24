# Square Attack: validation-only Stage 4

Based on `d8f3ec2d41a762366f0e75064c1e9785a44f4ac5`. Existing FGSM/PGD code,
experimental settings, tests, results and training files are unchanged. No sealed
holdout or train index is read. No full-validation experiment runs automatically.

## Algorithm and sources

Independent PyTorch implementation of the untargeted **L-infinity Square Attack**:

- [Andriushchenko et al., ECCV 2020, paper](https://arxiv.org/abs/1912.00049).
- [Authors' reference algorithm, pinned attack.py](https://github.com/max-andr/square-attack/blob/ea95eebb5aca62ec790a927b5aa985ba4e87245c/attack.py),
  functions `square_attack_linf` and `p_selection`.

This is score-based random search, not a sequence of independently applied random
patches. Initialize with independent ±epsilon **vertical stripes** per channel
and column. Minimize true-class logit minus the maximum other-class logit. At each
iteration, propose a random square whose per-channel constant sign sets the patch
perturbation to ±epsilon relative to the original RGB, clip, query, and retain
strict margin improvements. Cached accepted scores/predictions determine success.

The area fraction schedule is the reference piecewise schedule on normalized
iteration `int(iteration / attack_budget * 10000)`: p_init through 10, then divisors
2, 4, 8, 16, 32, 64, 128, 256, 512 at upper thresholds 50, 200, 500, 1000, 2000,
4000, 6000, 8000, 10000. Square side is round(sqrt(p*H*W)), clipped to [1,H-1]
(and W-1 for rectangular input). The reference exclusive upper location bound
is retained. p_init defaults to .05 (area fraction, not side length).

### Deliberate differences and edge handling

- NumPy PCG64 has an independent generator per stable sample seed, replacing the
  reference global NumPy stream. Results are reproducible but not bit-identical
  to the authors' trajectories. CLI uses existing frozen-validation sample IDs
  and seeds; the algorithm requires explicit per-sample seeds, never batch indices.
- The score oracle receives only RGB tensors and returns logits. Candidate
  generation never receives a model, parameters, modules, or gradient information.
  It runs entirely under torch.inference_mode. The evaluator supplies eval-mode
  PixelModel, disables AMP/TF32 and uses FP32, preserving original module flags.
- Success means `argmax(logits) != true_label`, not simply nonpositive margin.
  This avoids falsely counting an argmax-preserving tie. A truly misclassified
  tied candidate is retained even when the margin is equal.
- Unchanged proposals are resampled without queries, as in the reference. To avoid
  an infinite loop at tiny/unrepresentable epsilon, retry at most 128 sign draws,
  then try the opposite patch endpoints. If still unchanged, stop with
  `no_effective_proposal`. For rectangular toy inputs both side bounds apply;
  research evaluation remains exactly 3x224x224.
- No targeted/L2 variant, restarts, transfer attacks, defenses or gradient attack
  fallback. Clean-incorrect samples are returned unchanged, not counted as ASR
  successes. The epsilon=0 control returns clean predictions without perturbation.

## Query-budget convention

`max_queries` (default 5000, minimum 1) is the **total per-sample model evaluation
budget**, including:

1. One clean evaluation for every object.
2. One stripe-initialization evaluation for clean-correct objects when epsilon>0
   and at least one query remains.
3. One evaluation for each changed square proposal, including rejected proposals.

A batched call of N images costs one query for each of those N images. Computing
margin, predictions and success from the same logits costs no extra query.
Final predictions are cached; there is **no final hidden reevaluation**. Clean
mistakes and epsilon=0 use one query. A correct object with max_queries=1 also uses
only clean inference. Success or exhausted budget stops that image immediately.
A failed/nonfinite oracle call is logged before the run fails; no valid prediction
row is fabricated for it.

The reference counts initialization/candidate queries but excludes external clean
inference. Here `attack_budget = max_queries - 1` for the schedule: to compare an
N-query reference run, use max_queries=N+1. Report this difference when comparing
published query efficiency. Aggregate query counts include clean errors, with
separate conditional means for clean-correct and successful objects to avoid
confusing accuracy differences with query efficiency.

## Existing models, preprocessing, provenance

`attacks.models.load_classifier`, PixelModel and rgb_transform are reused without
changes. Compact weights must match SHA-256 from **MODEL_RUN/commit.json**; full
trusted weights must match the committed best tensor digest. No pretrained weights
are downloaded, no optimizer is constructed. All four baseline architectures and
the separately labelled Base v2 retain their existing preprocessing and identity.

Attacks operate on FP32 [0,1] RGB **after** deterministic model-specific geometry.
Normalization remains inside PixelModel. The original crop_pct differences
(Base .875, Nano/TinyViT .95, ViT .9) and ViT mean/std differences are unchanged.
Per-query and final independent checks enforce range and Linf (1e-6 tolerance).
Randomness is batch/order independent; near-tie floating-point scores across
hardware/batch kernels can still affect accept decisions. Standard eval-mode
classifiers must give per-image scores independent of other batch members.

## CLI and proposed Kaggle pilot

Select real mounted input paths; commands are examples for **future manual use**:

```bash
MODEL_RUN=/kaggle/input/visdrone-baselines/convnext_nano/RUN_NAME
VISDRONE_ROOT=/kaggle/input/visdrone2019-det/VisDrone
VALIDATION_INDEX=/kaggle/input/visdrone-data-v1/data-v1/validation.json
python -m attacks.square_evaluation --model-run "$MODEL_RUN" --checkpoint "$MODEL_RUN/best_weights.pt" --variant baseline-v1 --root "$VISDRONE_ROOT" --validation-index "$VALIDATION_INDEX" --output /kaggle/working/square-results --epsilon 0 1/255 2/255 4/255 8/255 --max-queries 500 --p-init 0.05 --subset-size 256 --seed 42 --batch-size 8 --device cuda --save-examples 4
```

The Square CLI defaults to a 256-object pilot, not full validation. Full population
requires an explicit --subset-size (e.g. 30570 for the reported frozen index).
First run epsilon=0 on 16 objects and verify clean predictions; then 32 objects,
max_queries=100, epsilon=4/255 to inspect counters, finite scores and budget stops.
Only after review run the 256-object, 500-query pilot above for each baseline,
using the same seed/subset. Consider 1000/5000-query sensitivity on a small fixed
validation subset before committing a mass protocol. Changing the budget also
changes the normalized square-size schedule: these are separate configurations,
not resumptions of one trajectory. Do not run all 30,570 objects automatically.

Use --variant convnext-base-v2 only for that separate training configuration.
The optional full-checkpoint flags match the existing evaluator. Square epsilon
accepts fractions or decimals in [0,1]; unlike the fixed FGSM/PGD CLI it is not
limited to the baseline epsilon grid. Existing FGSM/PGD CLI is unchanged.

## Artifacts and interruption

Unique exclusive OUTPUT/uuid directories reuse existing config/subset/status,
paired metrics, immutable `predictions/batch-*.csv` chunks, bounded examples,
checkpoint/index/subset SHA-256, Git/package/code provenance. Existing CSV schema
and its strict identity/completeness/F1/ASR reader remain unchanged.

Square adds:

- `queries/batch-*.json`: per-object sample_id/run_id/checkpoint hash, actual queries,
  success, final_prediction, actual Linf, accepted update count, final margin and
  stop_reason (`success`, `budget`, `clean_incorrect`, `epsilon_zero`, or
  `no_effective_proposal`).
- `query_metrics.json`: totals and means (all/clean-correct/successful denominators),
  maximum actual count and successful sample count. Empty conditional means = null.
- `diagnostics.jsonl`: flushed per-query accounting and proposal decisions with
  object identities; no images/logit histories are stored by default.

Query chunks commit before their paired CSV chunk, so a completed CSV prefix has
matching query data. Interruption may leave an orphan query chunk or an incomplete
final diagnostic line: readers ignore unpaired query chunks only for partial runs.
Completed runs require exact coverage and matching query aggregates. Use:

```python
from attacks.square_results import read_square_predictions, query_summary
rows = read_square_predictions('/kaggle/working/square-results/RUN_UUID')
queries = query_summary(rows)
```

Existing stale-partial opt-in is supported. Partial diagnostic/query prefixes are
usable for analysis; **attack state is not resumable**. Do not claim in-flight
queries/objects as completed rows. Retry in a new immutable run; do not overwrite
or append to an old run. Temporary Kaggle disk, fsync and abrupt power loss have
no durability guarantee. Manually retain new experiment directories if needed.

## CPU validation

```bash
python -m pytest -q
python -m attacks.square_smoke --output work/square-smoke
```

Smoke generates four toy crops and fixed toy weights, with no pretrained download,
training or GPU. It checks the full Square evaluator and strict result reader.
Synthetic unit tests cover schedule boundaries, stripe initialization, patch
endpoints, greedy rejection, success/early stopping, clean/initial/candidate query
accounting, budgets 1/2/3/20, all RGB/Linf checks, per-sample order/batch reproducibility,
no autograd and global RNG preservation, artifact identity and corrupt query counts.
Existing FGSM/PGD and training tests must still pass. Real checkpoint gradients are
irrelevant here; real score inference/preprocessing/memory/I/O still need a small
manual pilot. These synthetic results are not robustness measurements on VisDrone.

Implementation validation in Codex: **128 passed, 1 skipped** (existing CUDA test),
one upstream torch.jit.interface deprecation warning; git diff --check passed.
Separate CPU smoke completed on four generated crops: 25 total queries, three
clean-correct objects at the 8-query cap and one clean error at one query. No
trained VisDrone scores or real validation results were measured. The Square CLI
help and all existing FGSM/PGD integrity/regression tests passed.
