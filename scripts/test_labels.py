from utils import load_config
from data import make_dataloader

def main():
    cfg = load_config("configs/config.yaml")
    loader = make_dataloader(cfg, split=cfg["dataset"].get("split","test"))

    batch = next(iter(loader))
    print("Keys:", list(batch.keys()))
    print("Video:", tuple(batch["video"].shape))
    print("Path[0]:", batch["path"][0])

    if "labels" in batch:
        lbls = batch["labels"]            # list of length B; each item is (Ni, 3)
        print("Num samples with labels:", len(lbls))
        if len(lbls) and lbls[0] is not None:
            print("labels[0] shape:", tuple(lbls[0].shape))
            print("First 5 rows of labels[0] (onset, offset, pitch):")
            print(lbls[0][:5])
    else:
        print("No 'labels' key in batch (no .txt found).")

    # Clip-level targets (if present)
    for k in ["pitch","onset","offset","hand","clef"]:
        if k in batch:
            print(f"{k}[0]:", batch[k][0].item() if batch[k].ndim == 1 else batch[k][0])

if __name__ == "__main__":
    main()

