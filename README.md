# TiViT-Piano

TiViT-Piano is a video-based piano transcription system built on a ViViT-style
spatio-temporal transformer backbone with task-specific heads for pitch,
onset, offset, hand, and clef predictions. The model ingests tiled RGB video,
learns shared representations across time, and emits multi-label frame-level
logits that support both audio-visual training and downstream theory-aware
post-processing.

## Installation

TiViT-Piano targets Python 3.10+ environments with CUDA-enabled PyTorch
wheels pre-selected in `requirements.txt`. To install dependencies into a
virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The requirements file pins Torch 2.8, TorchVision 0.23, Triton 3.4, and the
associated CUDA 12.x libraries so GPU execution works out of the box. If you
prefer CPU-only or alternative CUDA versions, adjust the Torch packages before
running `pip install`.

## Datasets

The configuration file and dataset loaders support both the OMAPS, PianoYT and PianoVAM 
collections. Each dataset resolver allows explicit paths in the YAML config and
falls back to environment variables when unset.

### Environment variables

Set one of the following to point at a directory that contains dataset
subfolders:

```bash
export TIVIT_DATA_DIR=/path/to/datasets
# or
export DATASETS_HOME=/path/to/datasets
```

When these variables are defined, the loaders expect `OMAPS/`, `PianoYT/` and `PianoVAM_v1.0/`
children within the directory. If neither variable is set, the code searches the
paths declared in `configs/config.yaml` and finally falls back to
`~/datasets/<name>` or repository-local `data/PianoYT` when available.

### OMAPS

To train on OMAPS, update `configs/config.yaml` with `dataset.name: "OMAPS"`
and (optionally) provide `root_dir`/`annotations_root`. The loader resolves the
root using the priority order shown below:

1. Explicit `root_dir` from the config, if it exists.
2. `$TIVIT_DATA_DIR/OMAPS` or `$DATASETS_HOME/OMAPS`.
3. `~/datasets/OMAPS` as a final fallback.

Store media under split directories (for example `train/video_*.mp4`) and
provide manifest files when you want to subset clips; manifests are simple text
files with one clip stem per line and `#` comments allowed.

### PianoYT

The PianoYT loader mirrors the same resolution logic and also recognises
repository checkouts stored at `data/PianoYT/`. Expected structure:

```
PianoYT/
  metadata.csv
  splits/
    train.txt
    val.txt
    test.txt
  train/
    video_<id>.0.mp4
    audio_<id>.0.midi
  val/
  test/
```

If split text files are missing, the loader infers clip IDs by scanning the
split directories. Configure `dataset.manifest.*` or `split_*` keys in
`configs/config.yaml` to select subsets during training, validation, or testing.

### PianoVAM

To train on PianoVAM, set `dataset.name: "PianoVAM"` in `configs/config.yaml`. The loader resolves the root directory using the same priority order:

1. Explicit `root_dir` from the config, if provided.
2. `$TIVIT_DATA_DIR/PianoVAM_v1.0` or `$DATASETS_HOME/PianoVAM_v1.0`.
3. `~/datasets/PianoVAM_v1.0` as a final fallback.

Expected structure under the resolved root:

```text
PianoVAM_v1.0/
  Audio/
  MIDI/
  Video/
  TSV/
  Handskeleton/
  metadata_v1.json
  metadata_v2.json


### Onset-balanced sampling

Both dataset factories now accept an optional sampler block that biases the
training loader toward clips with annotated onsets (with a configurable mix of
near-miss and background windows). Enable it in the dataset section of your
configuration:

```yaml
dataset:
  sampler:
    mode: onset_balanced
    onset_frac: 0.50
    nearmiss_frac: 0.25
    bg_frac: 0.25
    nearmiss_radius: 6  # frames
```

When the dataset exposes onset/near-miss indices the helper wires up a
`WeightedRandomSampler`; otherwise the loader falls back to the default uniform
shuffle, so leaving the block in place is safe even on older manifests.

## Quick start

1. Update `configs/config.yaml` with your dataset paths and experiment metadata.
   The default experiment writes logs to `logs/` and checkpoints to
   `checkpoints/` under the project root.
2. Launch training:

   ```bash
   python scripts/train.py --config configs/config.yaml
   ```

   Common overrides include `--train-split`, `--val-split`, `--max-clips`, and
   `--frames` for rapid experiments. Use `--verbose info` (or `--verbose debug`)
   to surface additional logging and `--smoke` to run a synthetic dry run
   without touching the datasets.

Checkpoints and TensorBoard summaries are emitted according to the `logging`
section of the configuration (for example `logging.checkpoint_dir` and
`logging.log_dir`).

### Registration QA

Use the refinement smoke test to verify cached homographies before extended
training or calibration runs:

```bash
python scripts/check/test_refinement.py \
  --split val \
  --clips 2 \
  --inspect 5 \
  --log-level debug
```

The helper streams the usual `reg_refined:` log lines while it samples clips and
then summarises the most recent cache entries, reporting the pre/post edge
alignment error and the deviation from the baseline scale transform. Adjust
`--clips`, `--frames`, or `--batch-size` to keep the probe lightweight. Results
are cached in `reg_refined.json` at the repository root so repeated runs can
reuse prior estimates.

## Evaluation

The calibration utilities sweep decision thresholds and can optionally dump raw
logits for later analysis:

```bash
python scripts/calib/eval_thresholds.py \
  --ckpt checkpoints/tivit_best.pt \
  --config configs/config.yaml \
  --split val \
  --thresholds 0.2 0.3 0.4 \
  --dump_logits --out_npz out/val_logits.npz
```

Provide `--prob_thresholds` or `--thresholds` to customise the sweep, and use
`--fixed_*` flags when calibrating only one head. When `--dump_logits` is set the
script saves NPZ files containing flattened logits for each requested head; the
NPZ artefacts can be rescored or re-evaluated without rerunning the forward
pass.

### Debugging mode

Pass `--verbose debug` to calibration or evaluation entry points (for example
`eval_thresholds.py`) to surface detailed per-clip diagnostics. The calibration
helper also retains a `--debug` switch that favours deterministic,
single-process data loading when investigating individual clips: it forces
`num_workers=0`, disables `persistent_workers`, turns off pinned memory, and
prints an acknowledgement before dataset construction so you can confirm the
runtime surface has been adjusted.

To recompute the default operating points from scratch, run the streaming
calibration helper which emits running summaries to `calibration.json` and logs
reliability diagrams:

```bash
python scripts/calib/calibrate_thresholds.py \
  --ckpt checkpoints/tivit_best.pt \
  --split val \
  --max-clips 400 \
  --timeout-mins 20
```

The script honours dataset overrides from `configs/config.yaml` and stops early
when the optional timeout elapses while still persisting partial statistics.

## Autopilot training loop

For unattended experiments use `scripts/train_autopilot.py`. The helper applies
safe defaults to `configs/config.yaml`, alternates between short training
bursts, calibration sweeps, and evaluations, and writes the metrics for each
round to `logs/auto/results.txt` alongside per-round stdout logs. You can start
directly from calibration, skip the first training burst when resuming, or ask
the driver to fall back to a fast grid search if the thorough sweep fails.

```bash
python scripts/train_autopilot.py \
  --mode fresh \
  --first_step train \
  --first_calib thorough \
  --fast_first_calib \
  --burst_epochs 3 \
  --target_ev_f1 0.20 \
  --results runs/auto/results.txt \
  --ckpt_dir checkpoints \
  --split_eval val
```

Pass `--dry_run` to exercise the control flow without launching training or
calibration. Subsequent invocations with `--mode resume` keep the current
experiment name and reuse the latest checkpoints.

## Key-aware decoding

The theory decoder estimates a 24-key posterior using the Krumhansl–Schmuckler
profiles from `theory/key_prior.py`, mixes the posterior with a uniform prior,
and rescales selected heads. You can export posterior traces and adjust the
prior strength, temporal window, and smoothing parameters directly from the CLI:

```bash
python scripts/theory_decode.py \
  --in_npz out/val_logits.npz \
  --out_npz out/val_logits_keyaware.npz \
  --apply_to onset,pitch,offset \
  --lambda_key 0.5 \
  --window_sec 3.0 \
  --save_key_posterior out/val_key_posterior.npz
```

The defaults match the configuration in `KeyPriorConfig`, but you can sweep
`--lambda_key`, `--beta`, `--rho_uniform`, and `--window_sec` to explore
alternative priors. Use the resulting NPZ files with
`scripts/calib/eval_thresholds.py --in_npz ...` to measure the impact of theory
aware decoding.

## Debugging & diagnostics

The repository includes several helpers for sanity-checking data and model
plumbing:

- `python scripts/check/test_loader.py` prints dataset statistics, verifies
  manifests, and checks tensor shapes for the configured split.
- `python scripts/check/test_forward.py` instantiates TiViT-Piano and runs a
  single forward pass to confirm checkpoint compatibility.

Refer to `docs/REPO_TREE.md` for an index of additional utilities, including
profilers, visualization scripts, and document generators. Combine these tools
with `--verbose debug` and `--smoke` when iterating on configuration changes.

## Documentation & references

Auto-generated documentation summarises the repository layout and per-file CLI
entry points:

- [`docs/REPO_TREE.md`](docs/REPO_TREE.md)
- [`docs/FILE_INDEX.md`](docs/FILE_INDEX.md)

Regenerate the docs after structural changes using:

```bash
python scripts/dev/gen_repo_tree.py
python scripts/dev/gen_file_index.py
```

Please cite TiViT-Piano if it supports your research, and consult the project
metadata or accompanying publication for licensing details.
