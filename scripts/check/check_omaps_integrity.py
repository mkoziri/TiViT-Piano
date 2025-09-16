#!/usr/bin/env python3
"""Purpose:
    Validate the on-disk OMAPS dataset by checking video/label pairs, reporting
    anomalies, and collecting metadata summaries.

Key Functions/Classes:
    - scan_split(): Scans a split directory to match MP4 and TXT files, reading
      note annotations and computing statistics.
    - read_labels(): Parses sidecar label files and ensures timings and MIDI
      values are valid.
    - main(): Handles CLI arguments, writes JSON/TSV reports, and prints
      high-level dataset diagnostics.

CLI:
    Execute ``python scripts/check_omaps_integrity.py --root ~/datasets/OMAPS``
    with optional ``--probe-media`` to query ``ffprobe`` details and
    ``--meta-dir``/``--report-dir`` to control output locations.
"""

import argparse, json, os, re, shutil
from pathlib import Path
from statistics import median

VIDEO_EXTS = {".mp4", ".MP4"}
LABEL_EXT  = ".txt"

def read_labels(txt_path: Path):
    rows = []
    with txt_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", s)
            if len(parts) < 3:
                raise ValueError(f"{txt_path}:{ln} needs 3 columns 'onset offset pitch'")
            onset, offset = float(parts[0]), float(parts[1])
            pitch = int(round(float(parts[2])))
            if not (onset >= 0 and onset < offset):
                raise ValueError(f"{txt_path}:{ln} onset<{offset} violated: {onset} !< {offset}")
            if not (0 <= pitch <= 127):
                raise ValueError(f"{txt_path}:{ln} pitch out of MIDI range: {pitch}")
            rows.append((onset, offset, pitch))
    if not rows:
        raise ValueError(f"No labels found in {txt_path}")
    return rows

def ffprobe_info(mp4: Path):
    """Return dict with r_frame_rate,width,height,audio_sr or {} if ffprobe missing."""
    if shutil.which("ffprobe") is None:
        return {}
    import subprocess
    try:
        v = subprocess.check_output(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","stream=r_frame_rate,width,height",
             "-of","default=noprint_wrappers=1:nokey=1", str(mp4)],
            stderr=subprocess.DEVNULL, text=True).splitlines()
        a_sr = subprocess.check_output(
            ["ffprobe","-v","error","-select_streams","a:0",
             "-show_entries","stream=sample_rate",
             "-of","default=noprint_wrappers=1:nokey=1", str(mp4)],
            stderr=subprocess.DEVNULL, text=True).strip()
        out = {}
        if len(v) >= 3:
            out.update({"r_frame_rate": v[0], "width": int(v[1]), "height": int(v[2])})
        if a_sr:
            out.update({"audio_sr": a_sr})
        return out
    except Exception:
        return {}

def scan_split(split_dir: Path, probe_media: bool, anomalies: list):
    """Return list of piece dicts for a split directory that contains *.mp4+*.txt pairs."""
    if not split_dir.exists():
        raise SystemExit(f"Expected split directory missing: {split_dir}")

    # index by stem
    vmap, lmap = {}, {}
    for p in split_dir.rglob("*"):
        if p.is_file():
            ext = p.suffix
            stem = p.with_suffix("").name
            if ext in VIDEO_EXTS:
                vmap[stem] = p
            elif ext == LABEL_EXT:
                lmap[stem] = p

    stems = sorted(set(vmap.keys()) | set(lmap.keys()))
    if not stems:
        anomalies.append(f"[EMPTY SPLIT] {split_dir}")
        return []

    pieces = []
    for stem in stems:
        txt = lmap.get(stem)
        if txt is None:
            anomalies.append(f"[MISSING TXT] split={split_dir.name} stem={stem}")
            continue
        try:
            rows = read_labels(txt)
            duration = max(off for _, off, _ in rows)
            n_notes = len(rows)
            speed = n_notes / duration if duration > 0 else 0.0
            rec = {
                "stem": stem,
                "txt": str(txt.resolve()),
                "duration_sec": duration,
                "n_notes": n_notes,
                "notes_per_sec": speed,
                "pitch_min": min(p for *_, p in rows),
                "pitch_max": max(p for *_, p in rows),
                "has_video": stem in vmap,
            }
            if stem in vmap:
                rec["mp4"] = str(vmap[stem].resolve())
                if probe_media:
                    rec.update(ffprobe_info(vmap[stem]))
            pieces.append(rec)
        except Exception as e:
            anomalies.append(f"[BAD LABELS] split={split_dir.name} txt={txt} :: {e}")

    return pieces

def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def write_tsv(path: Path, pieces):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["stem","duration_sec","n_notes","notes_per_sec","pitch_min","pitch_max","has_video","mp4","txt"]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in sorted(pieces, key=lambda x: x["stem"]):
            row = [str(r.get(c,"")) for c in cols]
            f.write("\t".join(row) + "\n")

def summarize(name: str, pieces):
    if not pieces:
        print(f"[{name}] 0 pieces")
        return
    speeds = [p["notes_per_sec"] for p in pieces]
    durs   = [p["duration_sec"]  for p in pieces]
    print(f"[{name}] pieces={len(pieces)} "
          f"| notes/s median={median(speeds):.3f} "
          f"| dur median={median(durs):.2f}s "
          f"| with_video={sum(1 for p in pieces if p.get('has_video'))}")

def main():
    ap = argparse.ArgumentParser(description="OMAPS Step 0: integrity + metadata for ~/datasets/OMAPS/{train,test}.")
    ap.add_argument("--root", type=Path, required=True, help="Dataset root that contains 'train' and 'test' subfolders")
    ap.add_argument("--probe-media", action="store_true", help="Use ffprobe to log FPS/resolution/audio rate if available")
    ap.add_argument("--meta-dir", type=Path, default=Path("metadata"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    root = Path(os.path.expanduser(str(args.root))).resolve()
    train_dir = root / "train"
    test_dir  = root / "test"

    anomalies = []
    train_p = scan_split(train_dir, args.probe_media, anomalies)
    test_p  = scan_split(test_dir,  args.probe_media, anomalies)

    # write JSONs
    write_json(args.meta_dir / "omaps_train.json", {"root": str(root), "split": "train", "n": len(train_p), "pieces": train_p})
    write_json(args.meta_dir / "omaps_test.json",  {"root": str(root), "split": "test",  "n": len(test_p),  "pieces": test_p})
    write_json(args.meta_dir / "omaps_all.json",   {"root": str(root), "split": "all",   "n": len(train_p)+len(test_p), "pieces": train_p+test_p})

    # write TSVs (easy to eyeball in a text editor)
    write_tsv(args.report_dir / "omaps_train.tsv", train_p)
    write_tsv(args.report_dir / "omaps_test.tsv",  test_p)

    # anomalies (if any)
    if anomalies:
        (args.report_dir / "omaps_anomalies.txt").write_text("\n".join(anomalies), encoding="utf-8")
        print(f"[WARN] {len(anomalies)} anomalies → {args.report_dir/'omaps_anomalies.txt'}")

    # console summaries
    summarize("TRAIN", train_p)
    summarize("TEST",  test_p)
    print(f"[OK] JSON → {args.meta_dir}/omaps_*.json")
    print(f"[OK] TSV  → {args.report_dir}/omaps_*.tsv")

if __name__ == "__main__":
    main()

