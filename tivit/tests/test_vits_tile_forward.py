"""Ad-hoc forward sanity check for the ViT-S tile backend (no pytest needed)."""

import torch

from tivit.models.backbones.vit_small import ViTSTilePiano, build_model


def _make_config() -> dict:
    # Use the backbone's native img_size (224) to satisfy timm's assertions.
    return {
        "dataset": {
            "tiles": 3,
            "channels": 3,
        },
        "model": {
            "backend": "vits_tile",
            "head_mode": "frame",
            "transformer": {
                "input_patch_size": 16,
                "dropout": 0.05,
            },
            "vits_tile": {
                "backbone_name": "vit_small_patch16_224",
                "pretrained": False,
                "freeze_backbone": False,
                "input_hw": [224, 224],
                "temporal_layers": 1,
                "temporal_heads": 2,
                "global_mixing_layers": 1,
                "embed_dim": 384,
                "dropout": 0.05,
            },
        },
        "tiling": {"patch_w": 16, "tokens_split": "auto", "overlap_tokens": 0},
    }


def main() -> None:
    cfg = _make_config()
    try:
        model = build_model(cfg)
    except RuntimeError as exc:
        if "timm is required" in str(exc):
            print("skip: timm not installed, ViT-S tile backend not exercised")
            return
        raise

    assert isinstance(model, ViTSTilePiano)
    model.eval()

    x = torch.randn(1, 2, cfg["dataset"]["channels"], 224, 224)
    with torch.no_grad():
        out = model(x)

    expected_shapes = {
        "pitch_logits": (1, 2, 88),
        "onset_logits": (1, 2, 88),
        "offset_logits": (1, 2, 88),
        "hand_logits": (1, 2, 2),
        "clef_logits": (1, 2, 3),
        "pitch_global": (1, 2, 88),
        "onset_global": (1, 2, 88),
        "offset_global": (1, 2, 88),
        "hand_global": (1, 2, 2),
        "clef_global": (1, 2, 3),
    }
    for key, shape in expected_shapes.items():
        assert key in out, f"missing output '{key}'"
        assert tuple(out[key].shape) == shape, f"{key} has shape {tuple(out[key].shape)}, expected {shape}"

    print("ok: ViT-S tile forward", {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
