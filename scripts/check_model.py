from utils import load_config
from models import build_model
import torch

def main():
    cfg = load_config("configs/config.yaml")
    model = build_model(cfg).eval()

    # Dummy input to verify forward pass
    B, T, tiles, C, H, W = 2, 32, 3, 3, 224, 224
    x = torch.randn(B, T, tiles, C, H, W)
    with torch.no_grad():
        out = model(x)

    print(f"B (batch size) = {B}")
    for k, v in out.items():
        print(f"{k}: shape = {tuple(v.shape)}")

if __name__ == "__main__":
    main()

