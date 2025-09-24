#!/usr/bin/env python3
import argparse, csv, os, re, sys, shutil, subprocess
from pathlib import Path

# Split codes per dataset README: 1=train, 3=test
SPLIT_CODE = {"1": "train", "3": "test"}
VIDEO_EXTS = [".mp4", ".mkv", ".webm", ".m4v", ".mov", ".avi"]

def die(msg, code=1):
    print(f"[ERR] {msg}", file=sys.stderr)
    sys.exit(code)

def which_or_die(name, hint):
    p = shutil.which(name)
    if not p:
        die(f"Required tool '{name}' not found. {hint}")
    return p

def sniff_rows(csv_path: Path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        data = f.read()
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(data, delimiters=",\t;")
            rows = list(csv.reader(f, dialect))
        except Exception:
            f.seek(0)
            rows = [ (ln.rstrip("\n").split("\t") if "\t" in ln else ln.rstrip("\n").split(",")) for ln in f ]
    # drop header if first cell isn't an integer
    if rows and (not rows[0] or not rows[0][0].strip().isdigit()):
        rows = rows[1:]
    return rows

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_symlink(src: Path, dst: Path, force=False):
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            try:
                if Path(os.readlink(dst)) == src:
                    return "ok"  # already correct
            except OSError:
                pass
        if force:
            try: dst.unlink()
            except FileNotFoundError: pass
        else:
            return "skip"
    try:
        dst.symlink_to(src)
        return "linked"
    except FileExistsError:
        return "skip"

def find_video_file(videos_src: Path, idx: str):
    # prefer exact name video_<idx>.0.<ext>
    for ext in VIDEO_EXTS:
        p = videos_src / f"video_{idx}.0{ext}"
        if p.exists(): return p
    # fallback: any file starting with video_<idx>.0.
    cands = list(videos_src.glob(f"video_{idx}.0.*"))
    return cands[0] if cands else None

def find_midi_file(midis_src: Path, idx: str):
    for ext in (".midi", ".mid"):
        p = midis_src / f"audio_{idx}.0{ext}"
        if p.exists(): return p
    return None

def probe_wh(ffprobe: str, path: Path):
    out = subprocess.check_output([
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", str(path)
    ], text=True).strip()
    w, h = out.split("x")
    return int(w), int(h)

def main():
    ap = argparse.ArgumentParser(
        description="Create data/PianoYT/{train,test} with videos+midis together; exclude LOW-res videos by default."
    )
    ap.add_argument("--dataset-root", default="~/datasets/PianoYT", help="PianoYT root (raw_videos, pseudo_midis, metadata)")
    ap.add_argument("--project-root", default="~/dev/tivit", help="Your tivit repo root")
    ap.add_argument("--min-width", type=int, default=1920)
    ap.add_argument("--min-height", type=int, default=1080)
    ap.add_argument("--include-low", action="store_true", help="Include low-res videos instead of excluding them")
    ap.add_argument("--force", action="store_true", help="Overwrite existing symlinks")
    ap.add_argument("--link-metadata", action="store_true", help="Also symlink CSV/README/license into project data/metadata")
    ap.add_argument("--dry-run", action="store_true", help="Show actions without creating links")
    args = ap.parse_args()

    ds = Path(os.path.expanduser(args.dataset_root))
    pr = Path(os.path.expanduser(args.project_root))

    csv_path   = ds / "metadata" / "pianoyt.csv"
    videos_src = ds / "raw_videos"
    midis_src  = ds / "pseudo_midis"

    if not csv_path.is_file():  die(f"CSV not found: {csv_path}")
    if not videos_src.is_dir(): die(f"raw_videos not found: {videos_src}")
    if not midis_src.is_dir():  die(f"pseudo_midis not found: {midis_src}")

    ffprobe = None
    if not args.include_low:
        ffprobe = which_or_die("ffprobe", "Install ffmpeg (e.g., sudo apt install ffmpeg), or use --include-low to skip resolution checks.")

    # Dest layout
    base     = pr / "data" / "PianoYT"
    train_d  = base / "train"
    test_d   = base / "test"
    splits_d = base / "splits"
    meta_d   = base / "metadata"
    for d in (train_d, test_d, splits_d):
        ensure_dir(d)
    if args.link_metadata:
        ensure_dir(meta_d)

    # Build index -> split mapping
    rows = sniff_rows(csv_path)
    idx2split = {}
    for r in rows:
        if not r: continue
        idx = r[0].strip()
        if not idx.isdigit(): continue
        split_val = (r[2].strip() if len(r) > 2 else "")
        split = SPLIT_CODE.get(split_val)
        if split: idx2split[idx] = split

    kept = {"train": [], "test": []}
    excluded_low = []
    missing_video = []
    missing_midi  = []

    def link_pair(idx: str, split: str):
        # locate files in source dataset
        vsrc = find_video_file(videos_src, idx)
        if vsrc is None:
            missing_video.append(idx)
            return  # skip entirely if no video

        # resolution gate (unless include-low)
        if ffprobe:
            try:
                w,h = probe_wh(ffprobe, vsrc)
                if (w < args.min_width or h < args.min_height):
                    excluded_low.append(idx)
                    return
            except Exception:
                excluded_low.append(idx)
                return

        msrc = find_midi_file(midis_src, idx)
        if msrc is None:
            missing_midi.append(idx)  # we will still link the video

        # destination filenames keep their original names
        dest_dir = train_d if split == "train" else test_d
        vdst = dest_dir / vsrc.name
        mdst = dest_dir / msrc.name if msrc else None

        if args.dry_run:
            print(f"[DRY] link VIDEO {vsrc} -> {vdst}")
            if msrc: print(f"[DRY] link MIDI  {msrc} -> {mdst}")
        else:
            _ = safe_symlink(vsrc, vdst, force=args.force)
            if msrc:
                _ = safe_symlink(msrc, mdst, force=args.force)

        kept[split].append(idx)

    for idx, split in idx2split.items():
        link_pair(idx, split)

    # Write lists of indices actually linked
    (splits_d / "train.txt").write_text("\n".join(sorted(set(kept["train"]), key=lambda x:int(x))) + "\n")
    (splits_d / "test.txt").write_text("\n".join(sorted(set(kept["test"]), key=lambda x:int(x))) + "\n")
    (splits_d / "excluded_low.txt").write_text("\n".join(sorted(set(excluded_low), key=lambda x:int(x))) + "\n")

    if args.link_metadata:
        for name in ("pianoyt.csv", "README.md", "license.txt"):
            src = ds / "metadata" / name
            if src.exists():
                dst = meta_d / name
                if not args.dry_run:
                    _ = safe_symlink(src, dst, force=args.force)

    print(f"[DONE] Kept: train={len(kept['train'])}, test={len(kept['test'])}, "
          f"excluded_low={len(set(excluded_low))}, missing_video={len(set(missing_video))}, "
          f"missing_midi={len(set(missing_midi))}")
    print(f"[INFO] Layout root: {base}")
    print(f"[INFO] Splits: {splits_d}/train.txt, {splits_d}/test.txt, excluded_low.txt")

if __name__ == "__main__":
    main()
