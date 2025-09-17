# TiViT-Piano

This repository contains experiments around a piano transcription model.

## Logging & Debugging
- Add `--debug` to increase verbosity (DEBUG level).
- Add `--smoke` to run a fast synthetic pass without datasets.

## Repository Docs
- **Repo tree:** see [`docs/REPO_TREE.md`](docs/REPO_TREE.md)
- **Per-file index & CLI:** see [`docs/FILE_INDEX.md`](docs/FILE_INDEX.md)

To regenerate:
```bash
python scripts/dev/gen_repo_tree.py
python scripts/dev/gen_file_index.py
```

## Key-aware decoding
The theory-aware decoder estimates a 24-key posterior from a reference logit head,
converts that posterior into a smooth pitch-class prior, and rescales the selected
heads' logits without zeroing out any classes. Adjusting `--lambda_key` controls
how strongly the prior nudges in-key pitch classes.

**Workflow**

1. Dump logits during evaluation:
   ```bash
   python scripts/eval_thresholds.py --dump_logits --split val --out_npz out/val_logits.npz
   ```
2. Rescore the logits with the theory decoder:
   ```bash
    python scripts/theory_decode.py \
     --in_npz out/val_logits.npz \
     --out_npz out/val_logits_keyaware.npz \
     --apply_to onset,pitch,offset
   ```
3. Evaluate the rescored NPZ (for example):
   ```bash
   python scripts/eval_thresholds.py --in_npz out/val_logits_keyaware.npz
   ```

**Hyperparameters**

- Defaults: `--window_sec 3.0`, `--beta 4.0`, `--rho_uniform 0.8`, `--lambda_key 0.5`.
- Suggested first sweep: `lambda_key ∈ {0.3, 0.5, 0.7}`, `window_sec ∈ {2.0, 3.0, 4.0}`.

