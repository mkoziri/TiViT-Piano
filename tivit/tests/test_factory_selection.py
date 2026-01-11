"""Quick checks that the factory selects the new backbones (no pytest needed)."""

from tivit.models.backbones.vit_small import ViTSTilePiano
from tivit.models.backbones.vivit import TiViTPiano
from tivit.models.factory import build_model


def _vivit_cfg() -> dict:
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
                "dropout": 0.1,
            },
        },
        "tiling": {"patch_w": 16, "tokens_split": "auto", "overlap_tokens": 0},
    }


def _vits_cfg() -> dict:
    return {
        "dataset": {
            "tiles": 3,
            "channels": 3,
        },
        "model": {
            "backend": "vits_tile",
            "head_mode": "frame",
            "transformer": {"input_patch_size": 16, "dropout": 0.05},
            "vits_tile": {
                "backbone_name": "vit_small_patch16_224",
                "pretrained": False,
                "freeze_backbone": False,
                "input_hw": [32, 48],
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
    vivit = build_model(_vivit_cfg())
    assert isinstance(vivit, TiViTPiano)
    print("ok: factory built ViViT backbone ->", type(vivit).__name__)

    try:
        vits = build_model(_vits_cfg())
    except RuntimeError as exc:
        if "timm is required" in str(exc):
            print("skip: timm not installed, ViT-S tile factory path not exercised")
            return
        raise
    assert isinstance(vits, ViTSTilePiano)
    print("ok: factory built ViT-S tile backbone ->", type(vits).__name__)


if __name__ == "__main__":
    main()
