#!/usr/bin/env python3
"""Purpose:
    Summarize sweep or calibration experiment logs by parsing TSV-like rows,
    sorting metrics, and writing CSV/Markdown reports.

Key Functions/Classes:
    - parse_val_line(): Converts ``key=value`` fragments into numeric entries
      when possible.
    - to_num_or_none(): Utility that casts tokens to numbers while tolerating
      sentinel strings.
    - main(): Provides the CLI flow for reading the results file, filtering by
      return code, sorting, and writing outputs.

CLI:
    Invoke ``python scripts/parse_sweep.py --results sweep_results.txt`` with
    optional ``--sort_by`` or ``--filter_retcode`` flags to control the summary
    outputs ``--out_csv`` and ``--out_md``.
"""

import argparse, csv, math, re, sys
from pathlib import Path

KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
DEFAULT_HEADER = [
    "iso8601",
    "gamma",
    "alpha",
    "thr",
    "prior_mean",
    "prior_wt",
    "tol",
    "dilate",
    "max_clips",
    "exp",
    "val_line",
    "retcode",
]

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

    header = None
    rows = []
    with results_path.open("r") as f:
        for line in f:
            if line.startswith("#"):
                if line.lower().startswith("# columns:"):
                    header = [tok for tok in line.split(":", 1)[1].strip().split("\t") if tok]
                continue
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            cols = header or DEFAULT_HEADER
            if len(parts) < len(cols):
                continue
            if len(parts) > len(cols) and "val_line" in cols:
                val_idx = cols.index("val_line")
                prefix = parts[:val_idx]
                suffix_len = len(cols) - val_idx - 1
                val_tokens = parts[val_idx : len(parts) - suffix_len]
                suffix = parts[len(parts) - suffix_len :]
                parts = prefix + ["\t".join(val_tokens)] + suffix
            mapped = dict(zip(cols, parts[: len(cols)]))
            val_line = mapped.get("val_line", "")
            row = {}
            for key, value in mapped.items():
                if key == "val_line":
                    continue
                num = to_num_or_none(value)
                row[key] = num if num is not None else value
            row.update(parse_val_line(val_line))
            rows.append(row)

    if args.filter_retcode != -1:
        rows = [r for r in rows if r.get("retcode") == args.filter_retcode]

    if not rows:
        print("No data rows after filtering.", file=sys.stderr); sys.exit(1)

    # columns
    cols_from_header = header or DEFAULT_HEADER
    fixed_cols = [c for c in cols_from_header if c not in {"val_line"}]
    if "retcode" not in fixed_cols:
        fixed_cols.append("retcode")
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

