# Dataset Notes

- Data pipeline lives under `tivit/data/`, with `datasets/` using `BasePianoDataset` for decode → registration → augmentation → normalization → tiling → sync → targets.
- Shared helpers: `decode/video_reader.py`, `roi/keyboard_roi.py` (registration), `roi/tiling.py` (token-aligned splits), `targets/frame_targets.py`, `sync/sync.py`, `transforms/augment.py`/`normalize.py`, `sampler.py`.
- Per-dataset entrypoints: `datasets/pianoyt.py`, `datasets/omaps.py`, `datasets/pianovam.py` plus their `*_impl.py` subclasses; configs live in `tivit/configs/dataset/`.
- Tests for the new layout are under `tivit/tests/` (dataset smoke tests, targets alignment, tiling, sampler).
