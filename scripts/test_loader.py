from utils import load_config
from data import make_dataloader

def main():
    cfg = load_config("configs/config.yaml")
    split = cfg["dataset"].get("split", "test")
    loader = make_dataloader(cfg, split=split)

    print(f"Dataset split: {split}")
    for i, batch in enumerate(loader):
        x = batch["video"]           # B, T, tiles, C, H, W
        paths = batch["path"]
        B, T, M, C, H, W = x.shape
        print(f"[{i}] batch: {B} clips | shape={x.shape}")
        print(" sample:", paths[0])
        print(f"  B={B}, T={T}, tiles={M}, C={C}, H={H}, W={W}")
        if i == 1:
            break

if __name__ == "__main__":
    main()

