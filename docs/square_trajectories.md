# Exact query-efficiency trajectories (Square only)

Base: PR #5 commit `3b2f1d6860db73794dd1dff5f87c81463b3aa3ef`.
FGSM/PGD code, their artifacts, existing immutable runs and the Square proposal,
margin acceptance, RNG, budget convention and precision are unchanged.

## First-hit observation

`generate_square` observes successful misclassification immediately after each
oracle response, using the actual per-object counter. Clean is query 1;
initial stripes are query 2 when evaluated; every square proposal (including
rejected candidates) consumes a query. No additional scoring call is introduced.
The linked `queries/batch-*.json` now includes `first_success_query`:

- Clean-correct and successfully attacked: an integer >1 and <=queries<=max_queries.
- Clean-incorrect: null (excluded from the ASR denominator).
- Clean-correct with no successful query: null, including epsilon=0 or budget=1.

The current attack stops immediately on first success, so a successful row also
requires first_success_query==queries and stop_reason==success. The strict reader
rejects null/noninteger/bool/out-of-range indices on successful rows, non-null
indices on failed/clean-incorrect rows and inconsistent stopping metadata.
A `square_first_success` diagnostic records the observed index; query JSON and
existing paired CSV remain the authoritative completed-object artifacts.

## One run, one fixed schedule, constant denominator

For a completed run with configured cap M, define D as the number of initially
correctly classified objects in its full selected subset. For every integer
q=1,...,M, compute:

`ASR(q) = count(clean_correct AND first_success_query <= q) / D`.

D never shrinks when samples stop, succeed or exhaust budget. Cumulative counts
are nondecreasing. ASR(1)=0 when D>0. The endpoint is checked for exact equality
with the validated metrics.json ASR, and D with clean_correct_count. For D=0,
all values are undefined (JSON null, empty CSV fields); the plot explicitly says
ASR undefined and does not draw a fictitious zero curve.

**This is a prefix trajectory of one run's schedule configured with cap M.**
It is not the result of re-running Square with max_queries=q. Different caps
change p_selection; do not concatenate results from independent runs/caps, even
if sample IDs/seed/epsilon agree. The API/CLI accepts one completed run, no run
lists, no cap override, and no legacy/partial trajectory interpolation.
Checkpoints beyond M are explicitly unavailable, never extrapolated or clamped.

## Analysis CLI (no model inference)

Run the attack as before, then invoke optional standalone postprocessing:

```bash
python -m attacks.square_trajectory \
  --run /kaggle/working/square-results/ATTACK_RUN_UUID \
  --output /kaggle/working/square-analysis \
  --display-checkpoints 100 250 500 750 1000 1500
```

Optional `--display-checkpoints` annotates only requested checkpoints within M
and writes an availability/ASR table. Omit it to export the unannotated full curve.
`MPLCONFIGDIR=/kaggle/working/mpl-cache` can select a writable matplotlib cache.
No pretrained weights, VisDrone images/index reads, new queries, GPU or training
are needed for this step: it reads completed attack artifacts only.

A new exclusive OUTPUT/uuid contains:

- `trajectory.csv`: every integer budget, successful_samples, fixed
  clean_correct_samples and ASR as a fraction [0,1].
- `asr_vs_queries.png`: 300-DPI, 6.4x4.2-inch headless figure; y-axis in percent.
- `asr_vs_queries.pdf`: vector PDF with embedded TrueType fonts. Plot title
  identifies model/variant, epsilon, configured M and clean-correct denominator.
- `checkpoints.json`: selected Q100/Q250/Q500/Q750/Q1000/Q1500 values; unavailable
  checkpoints have available=false. A value can also be null with available=true
  when D=0, which is scientifically different from an out-of-range checkpoint.
- `analysis.json`: single source run identity, config/seed/cap, sample selection,
  checkpoint digest, hashes of all analyzed input files, code/package provenance,
  matplotlib version, denominator and endpoint. Source files are never modified.
- `postprocessing_status.json`: running/completed/plot_failed, timestamps and error.

The output root must be separate from the immutable input run. Repeated exports
create new directories; they never overwrite historical plots or attack files.

### Plot failures and attack completion

Postprocessing is not called by the attack. `status.json` of the source attack is
never written by the analysis module. CSV and metadata are exported before any
plotting imports/rendering. A backend/rendering failure keeps those files and
writes plot_failed (with csv_complete=true); the CLI prints the destination and
exits 2. The attack remains completed. If just one plot succeeded before a later
failure, that file may exist but the postprocessing status still reports failure.
Fix the plotting environment and rerun into a new analysis directory. IO failure
on the separate analysis disk can prevent writing a status file; it still cannot
alter the original attack. Atomicity is per file, not a multi-file/fsync guarantee.

## Migration note for legacy Square runs

Historical query JSON without first_success_query is **not rewritten**. Default
`read_square_predictions` fails with an explicit legacy error. To read old final
metrics/predictions deliberately:

```python
from attacks.square_results import read_square_predictions
rows = read_square_predictions(legacy_run, allow_legacy=True)
```

The reader warns and returns first_success_query=None with
first_success_query_recorded=False for missing fields. Modern rows, including
legitimate nulls for failures, have first_success_query_recorded=True. A present
but invalid/null successful index is an error even in legacy mode; the flag does
not relax validation of malformed modern data. Other identity/budget/metric and
immutable-prefix checks remain intact.

Do not replace unknown indices with queries or infer them from final success/
stop_reason, reruns, assumptions about older code, or logs lacking conclusive
per-sample identities and exact oracle results. This implementation does not
import historical logs or synthesize first-hit times. Even legacy runs with only
failures are not automatically presented as exact trajectories. The analysis CLI
requires fully recorded modern data; rerun a small pilot in a new immutable run
if an exact trajectory is needed. No retrospective mutation of old artifacts.

## CPU verification and smoke

```bash
MPLCONFIGDIR=/tmp/visdrone-mpl python -m pytest -q
python -m attacks.square_smoke --output work/trajectory-smoke
# Select the emitted new RUN_UUID:
MPLCONFIGDIR=/tmp/visdrone-mpl python -m attacks.square_trajectory \
  --run work/trajectory-smoke/RUN_UUID --output work/trajectory-analysis \
  --display-checkpoints 100 250 500 750 1000 1500
```

Tests exercise initialization hits, rejected candidates before a late hit,
clean errors, budget failure/epsilon controls, order/batch independence, strict
reader validation, null and explicit legacy handling, full integer trajectories,
constant denominator and exact endpoint, Q100 display, unavailable Q250, empty
denominator, PNG/PDF headers, immutable re-export and isolated plot failure.
No full VisDrone experiment is required or performed.

Validation in Codex: **151 passed, 1 skipped** (existing CUDA test), one upstream
torch.jit.interface deprecation warning. git diff --check passed. Standalone CPU
attack smoke completed on four generated crops (25 actual queries), followed by
the real analysis CLI exporting eight integer-budget rows, PNG and PDF. The PNG
was visually inspected. No pretrained downloads, GPU/VisDrone experiments or
historical output changes were performed.
