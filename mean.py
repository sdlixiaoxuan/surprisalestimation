#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
--------
python mean.py \
    --csv     corpus/ZHANG22/outputs/merged_probabilities.csv \
    --out_dir corpus/ZHANG22/outputs/

python mean.py \
    --csv     corpus/HKC/outputs/merge/merged_probabilities.csv \
    --out_dir corpus/HKC/outputs/merge

python mean.py \
    --csv     corpus/GECOCN/outputs/merge/merged_probabilities.csv \
    --out_dir corpus/GECOCN/outputs/merge
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


def calculate_average_of_probabilities(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    
    # -------- 1. select the columns contains estimation --------
    prob_cols = [c for c in df.columns if c not in ("line_id", "sentence", "target")]
    
    if not prob_cols:
        print("ERROR: no columns found")
        return df

    print(f"calculating averages on {len(prob_cols)}  columns: {prob_cols}")

    # -------- 2. calculating --------
    print(" calculating...")
    df_out = df.copy()
    df_out['avg_probability'] = df[prob_cols].mean(axis=1)
    
    print(f"mean of 'avg_probability' ≈ {df_out['avg_probability'].mean():.4g}")

    # -------- 3. save --------
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "merged_probabilities_with_avg.csv"
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"file to : {output_path}")

    return df_out

def main():
    parser = argparse.ArgumentParser("calculate mean of merged_probabilities.csv")
    parser.add_argument("--csv", required=True, help=" path to merged_probabilities.csv")
    parser.add_argument("--out_dir", required=True, help="outputs path")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    if not csv_path.exists():
        sys.exit(f"cannot find：{csv_path}")

    print(f"loading: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8")
    calculate_average_of_probabilities(df, out_dir)

if __name__ == "__main__":
    main()
