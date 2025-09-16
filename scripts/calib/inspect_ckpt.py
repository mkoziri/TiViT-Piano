"""Purpose:
    Inspect checkpoint files to print stored metrics, epoch numbers, and
    top-level keys for quick debugging.

Key Functions/Classes:
    - Module-level script logic: Loads a checkpoint path from the first CLI
      argument (default ``checkpoints/tivit_best.pt``) and prints diagnostic
      information.

CLI:
    Execute ``python scripts/inspect_ckpt.py <checkpoint.pt>`` to display the
    contents.  No additional options are supported.
"""

import sys, torch
from pathlib import Path
ck = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/tivit_best.pt")
obj = torch.load(ck, map_location="cpu")
print("Keys:", list(obj.keys()))
epoch = obj.get("epoch", None)
print("epoch in ckpt:", epoch)
for k in ["total","onset_f1","onset_pred_rate","onset_pos_rate"]:
    if k in obj:
        print(k, ":", obj[k])

