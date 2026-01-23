# Priors and Constraints

- Training-time priors use `tivit/configs/priors/`, are gated by `priors.enabled`, and are applied inside the loss pipeline.
- Decoding-time key priors are configured under `priors.key_signature` and enabled via `decoder.post.key_prior.enabled`.
- Decoding-time hand gating is configured under `priors.hand_gating.*` and enabled via `decoder.post.hand_gate.enabled`.
- The decode-time key prior is applied inside `tivit/postproc/event_decode.build_decoder` when decoding logits.
