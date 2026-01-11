"""Ad-hoc forward sanity check for the ViViT backbone (no pytest needed)."""

import torch

from tivit.models.backbones.vivit import TiViTPiano, build_model


def _make_config() -> dict:
    # Minimal ViViT config that mirrors the new layout defaults.
    return {
        "dataset": {
            "tiles": 3,
            "channels": 3,
        },
        "model": {
            "backend": "vivit",
            "head_mode": "frame",
            "transformer": {
                "input_patch_size": 16,
                "tube_size": 2,
                "d_model": 192,
                "num_heads": 3,
                "depth_temporal": 1,
                "depth_spatial": 1,
                "depth_global": 1,
                "global_tokens": 2,
                "mlp_ratio": 3.0,
                "dropout": 0.2,
            },
        },
        "tiling": {"patch_w": 16, "tokens_split": "auto", "overlap_tokens": 0},
    }


def main() -> None:
    cfg = _make_config()
    model = build_model(cfg)
    assert isinstance(model, TiViTPiano)
    model.eval()

    # Shape is chosen to respect the tubelet and patch sizes (tube=2, patch=16).
    x = torch.randn(1, 4, cfg["dataset"]["channels"], 32, 48)
    with torch.no_grad():
        out = model(x)

    # Tube/patch arithmetic: T=4 -> T'=2, H=32/W=48 -> S'=6 tokens per tile.
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

    print("ok: ViViT forward", {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
