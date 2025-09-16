"""Purpose:
    Run a synthetic forward pass on TiViT-Piano using random inputs to verify
    that the model produces outputs with expected shapes outside the training
    pipeline.

Key Functions/Classes:
    - main(): Instantiates a lightweight TiViT-Piano configuration, feeds random
      tensors, and prints the resulting head dimensions.

CLI:
    Execute ``python scripts/test_synthetic_forward.py``.  No arguments are
    required; the script uses CPU tensors and random inputs.
"""

import os, sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models import TiViTPiano

def main():
    model = TiViTPiano(depth_temporal=1, depth_spatial=1, depth_global=1, global_tokens=2).eval()
    x = torch.randn(2, 8, 3, 3, 64, 64)  # (B,T,tiles,C,H,W)
    with torch.no_grad():
        out = model(x)
    for k, v in out.items():
        print(f"{k}: {tuple(v.shape)}")

if __name__ == '__main__':
    main()
