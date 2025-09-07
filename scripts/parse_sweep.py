#!/usr/bin/env python3
# scripts/parse_sweep.py
import argparse, csv, math, re, sys
from pathlib import Path

KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")

def to_num_or_none(s: str):
    if s is None: return None
    t = s.strip()
    if t == "" or t.lower() in {"none", "nan", "null"}:
        return None
    try:
        # integers without dot/exponent
        if "." not in t and "e" not in t.lower():
            return int(t)
        return float(t)
    except ValueError:
        return None

def parse_val_line(s: str) -> dict:
    """Parse 'key=value' tokens separated by spaces into numbers when possible."""
    out = {}
    if not s: return out
    for m in KV_RE.finditer(s.strip()):
        k, v = m.group(1), m.group(2)
        nv = to_num_or_none(v)
        out[k] = nv if nv is not None else v
    return out

def main():
    ap = argparse.ArgumentParser(description="Parse sweep/calibration results and summarize.")
    ap.add_argument("--results", default="sweep_results.txt", help="Path to results file.")
    ap.add_argument("--out_csv", default="sweep_summary.csv", help="Output CSV path.")
    ap.add_argument("--out_md", default="sweep_summary.md", help="Output Markdown table path.")
    ap.add_argument("--sort_by", default="onset_f1",
                    help="Metric to sort by (desc). Use '-metric' for ascending, e.g., '-total'.")
    ap.add_argument("--filter_retcode", type=int, default=0,
                    help="Keep only rows with this retcode. Set to -1 to keep all.")
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found", file=sys.stderr); sys.exit(1)

    rows = []
    with results_path.open("r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            # Expect: 12 columns: iso, gamma, alpha, thr, pm, pw, tol, dil, mclips, exp, val_line, retcode
            if len(parts) < 12:
                continue
            iso8601, gamma, alpha, thr, pm, pw, tol, dil, mclips, exp, val_line, retcode = parts[:12]
            row = {
                "iso8601": iso8601,
                "gamma": to_num_or_none(gamma),
                "alpha": to_num_or_none(alpha),
                "thr": to_num_or_none(thr),
                "prior_mean": to_num_or_none(pm),
                "prior_wt": to_num_or_none(pw),
                "tol": to_num_or_none(tol),
                "dilate": to_num_or_none(dil),
                "max_clips": to_num_or_none(mclips),
                "exp": exp,
                "retcode": to_num_or_none(retcode),
            }
            row.update(parse_val_line(val_line))
            rows.append(row)

    if args.filter_retcode != -1:
        rows = [r for r in rows if r.get("retcode") == args.filter_retcode]

    if not rows:
        print("No data rows after filtering.", file=sys.stderr); sys.exit(1)

    # columns
    fixed_cols = ["iso8601","gamma","alpha","thr","prior_mean","prior_wt","tol","dilate","max_clips","exp","retcode"]
    metric_cols = sorted({k for r in rows for k in r.keys()} - set(fixed_cols))
    cols_all = fixed_cols + metric_cols

    # sorting
    sort_key = args.sort_by.strip()
    ascending = False
    if sort_key.startswith("-"):
        ascending = True
        sort_key = sort_key[1:]
    if sort_key not in cols_all:
        print(f"WARNING: sort_by '{args.sort_by}' not found; falling back to 'onset_f1'.", file=sys.stderr)
        sort_key = "onset_f1"
        if sort_key not in cols_all:
            sort_key = "total"; ascending = True

    def sort_val(r):
        v = r.get(sort_key)
        # Normalize None to sort last for desc, first for asc
        return (v is None, v)

    rows_sorted = sorted(rows, key=sort_val, reverse=not ascending)

    # CSV
    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=cols_all)
        w.writeheader()
        for r in rows_sorted:
            w.writerow({k: r.get(k, "") for k in cols_all})

    # Markdown
    out_md = Path(args.out_md)
    with out_md.open("w") as mf:
        mf.write(f"# Summary (sorted by {'ascending ' if ascending else ''}`{sort_key}`)\n\n")
        mf.write("| " + " | ".join(cols_all) + " |\n")
        mf.write("|" + "|".join(["---"] * len(cols_all)) + "|\n")
        for r in rows_sorted:
            mf.write("| " + " | ".join(str(r.get(k, "")) for k in cols_all) + " |\n")

    # Console top-5
    print(f"\nTop 5 by {'ascending ' if ascending else ''}{sort_key}:")
    for i, r in enumerate(rows_sorted[:5], 1):
        print(f"{i}. exp={r.get('exp')}  {sort_key}={r.get(sort_key)}  "
              f"onset_f1={r.get('onset_f1')}  onset_pred_rate={r.get('onset_pred_rate')}  "
              f"onset_pos_rate={r.get('onset_pos_rate')}  total={r.get('total')}")

    print(f"\nWrote CSV: {out_csv}")
    print(f"Wrote Markdown: {out_md}")

if __name__ == "__main__":
    main()

