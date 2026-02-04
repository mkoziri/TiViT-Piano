# Configuration Reference

Summary of the composable YAML fragments under `tivit/configs/`. Each section lists key purpose, expected type, and typical defaults.

## Composition entrypoint
- `tivit/configs/default.yaml` (bases list)
  - Purpose: select experiment metadata, dataset, model, training, decoder, priors, and autopilot fragments.
  - Type: YAML list of relative config paths merged in order.
  - Helper: `tivit.core.config.load_experiment_config` will stack standalone fragments on top of `default.yaml` unless `default_base` is set to `null` (useful for legacy configs).

## Experiment & logging (`experiment/base.yaml`)
- `experiment.name` (str): run label; used in logs/checkpoints. Default: `TiViT-Piano GCR basic`.
- `experiment.seed` (int): RNG seed. Default: `1332`.
- `experiment.deterministic` (bool): enforce deterministic behavior. Default: `true`.
- `experiment.output_dir` (path): base output directory.
- `logging.log_dir` (path): logs location.
- `logging.checkpoint_dir` (path): checkpoints location.
- `logging.tensorboard` (bool): enable TB logs.
- `logging.postproc_debug` (bool): dump postproc debug artifacts.
- `inference.output_dir` (path): inference outputs.
- `inference.predict_on_train` (bool): whether to run inference on train split.

## Datasets (`dataset/*.yaml`)
Common keys across PianoVAM / PianoYT / OMAPS:
- `dataset.name` (str): dataset identifier.
- `dataset.root_dir` / `annotations_root` (path|null): resolved via env when null.
- `dataset.label_format` (str): label encoding (`midi` or `txt`).
- `dataset.label_targets` (list[str]): heads to supervise (pitch/onset/offset/hand/clef).
- `dataset.max_clips` (int|null): limit clips for quick runs.
- `dataset.split*` (str): train/val/test split names.
- `dataset.frames` (int): frames per clip. Typical: 96.
- `dataset.decode_fps` (float): decode frame rate. Typical: 30.0.
- `dataset.hop_seconds` (float): stride in seconds.
- `dataset.skip_seconds` (float): trim leading seconds before sampling clips.
- `dataset.resize` (list[int,int]): decode resize (H,W).
- `dataset.tiles` (int): number of vertical tiles. Typical: 3.
- `dataset.channels` (int): input channels. Typical: 3.
- `dataset.grayscale` (bool): request grayscale decode.
- `dataset.normalize` (bool): apply mean/std normalization.
- `dataset.apply_crop` (bool): apply registration crop.
- `dataset.canonical_hw` (list[int,int]): canonical keyboard crop size.
- `dataset.registration.*` (mapping): registration enable/source/interp and optional global aug knobs.
- `dataset.avlag.enable` (bool): enable AV lag correction metadata.
- `dataset.crop_rescale` / `include_low_res` (optional): crop scaling and low-res inclusion switches.
- `dataset.sampler.*` (mapping): onset-balanced sampler fractions/radius.
- `dataset.frame_targets.*` (mapping): target generation tolerances/caches.
- `dataset.batch_size` (int): loader batch size.
- `dataset.num_workers` (int): dataloader workers.
- `dataset.prefetch_factor` (int|null): torch prefetch_factor.
- `dataset.shuffle` (bool): shuffle flag for train.
- `dataset.testing.*` (mapping): canary/audit toggles, report paths, and debug sampling knobs used by test helpers.

Dataset-specific additions:
- PianoVAM: `hand_supervision.*` reach tuning.
- PianoYT: minimal registration aug.
- OMAPS: dataset is grayscale-disabled by default and uses explicit registration/avlag enables.

## Models (`model/*.yaml`)
- `model.backend` (str): `vivit` or `vits_tile`.
- `model.head_mode` (str): `frame` (current).
- `model.transformer.*`: patch/tube/d_model/heads/depth/dropout.
- `model.vits_tile.*` (ViT-S only): backbone name, pretrained/freeze, input_hw, temporal layers/heads, global mixing, embed_dim, dropout.
- Uses `dataset.tiles` and `dataset.channels` for shape wiring.
- `tiling.*`: patch_w/tokens_split/overlap_tokens for tile packing.

## Training (`train/single_run.yaml`)
- `training.epochs` (int): total epochs.
- `training.learning_rate` (float): base LR.
- `training.weight_decay` (float): weight decay.
- `training.eval_freq` (int): eval interval (epochs).
- `training.debug_dummy_labels` (bool): synthetic labels for smoke tests.
- `training.log_interval` (int): steps between log prints.
- `training.save_every` (int): epochs between checkpoints.
- `training.amp` (bool): mixed precision toggle.
- `training.resume` (bool): resume flag.
- `training.reset_head_bias` (bool): reinit onset/offset heads.
- `training.soft_targets.*` (mapping): enable/which heads/kernel shapes.
- `training.loss.head_weights` (mapping): per-head weights for pitch/onset/offset/hand/clef.
- `training.loss.pos_weight_path` (str): optional dataset-level pos_weight JSON (from `tivit.preproc.threshold_priors`).
- `training.loss.ema_alpha` / `neg_smooth_onoff` (float): EMA alpha for pos_weight_mode=ema; negative smoothing strength.
- `training.loss.per_tile.*` (mapping): enable/heads/mask cushion/debug interval for per-tile supervision.
- `training.loss.heads.*` (mapping): explicit per-head configs. Pitch requires loss/pos_weight_mode/pos_weight/pos_weight_band; onset/offset also require focal_gamma/focal_alpha/prior_mean/prior_weight; hand/clef expose `loss: ce`.
- `training.metrics.*` (mapping): thresholds, temps, biases, decoder params for metrics.
- `training.best_selection.*` (mapping): selection mode/trigger/n; uses dataset frames/max_clips/splits.
- `optim.*`: grad clip, head/offset lr multipliers.
- `train.*`: accumulation and freeze schedule.

## Decoder (`decoder/decoder.yaml`)
- `decoder.global_fusion.*`: fusion mode, cushion keys, apply_to, consistency check.
- `decoder.post.snap.*`: snapping parameters.
- `decoder.post.dp.*`: dynamic programming parameters.
- `decoder.post.key_prior.enabled` (bool): decoding-time key prior master switch; pulls parameters from `priors.key_signature`.
- `decoder.post.hand_gate.enabled` (bool): decoding-time hand gate master switch; pulls parameters from `priors.hand_gating.*`.

## Priors (`priors/priors.yaml`)
- `priors.enabled` (bool): master switch for training-time priors (hand gating, chord smoothness).
- `priors.hand_gating.*`: training-time mode/strength plus decode-time apply_to/decode_mode/decode_strength/clef_thresholds/note_min.
- `priors.chord_smoothness.strength` (float): smoothing weight.
- `priors.key_signature.*`: ref_head/apply_to/window/beta/rho/prior_strength/epsilon/fps/midi range for decoding key prior (controlled by `decoder.post.key_prior.enabled`).

## Autopilot (`train/autopilot.yaml`)
- `autopilot.best_selection.owner` (str): owner of selection metadata.
- `autopilot.fast_strategy` (str): strategy label.
- `autopilot.onset_optimizer.*` / `offset_optimizer.*`: threshold grid and search parameters.

## Calibration (`calib/*.yaml`)
- `calibration.method` (str): `threshold_sweep` (basic). `temperature`/`platt` are reserved for future fits.
- `calibration.output_path` (str): where to write calibration JSON.
- `calibration.threshold_priors_path` (str): dataset threshold priors YAML path for sweep centers.
- `calibration.sweep.delta` (float): step size around the center threshold.
- `calibration.sweep.steps` (int): number of sweep points (rounded up to an odd count).
- `calibration.sweep.min_prob` / `max_prob` (float): clamp sweep values.

## Tips
- Composition order in `default.yaml` defines override precedence.
- Keep dataset-owned values (frames/tiles/channels/splits) the single source; models and training derive from them.
- Use `tivit/core/config.py` helpers to load/merge and materialize resolved configs per run.
