"""
11.5yr 시점 기준 Beauty Coefficient 계산 (age 0~11만 사용)
Output: features/b_11yr.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from config import *

MAX_AGE_11 = 11


def beauty_coefficient(c: pd.Series):
    if len(c) == 0 or c.sum() == 0:
        return 0.0, 0, 0
    t_m = int(c.idxmax())
    if t_m == 0:
        return 0.0, 0, 0
    c0   = float(c.get(0, 0))
    c_tm = float(c[t_m])
    slope = (c_tm - c0) / t_m
    B = sum(
        (slope * t + c0 - float(c.get(t, 0))) / max(1.0, float(c.get(t, 0)))
        for t in range(0, t_m + 1)
    )
    denom_ta = np.sqrt((c_tm - c0)**2 + t_m**2)
    if denom_ta == 0:
        t_a = 0
    else:
        d_vals = {
            t: abs((c_tm - c0)*t - t_m*float(c.get(t, 0)) + t_m*c0) / denom_ta
            for t in range(0, t_m + 1)
        }
        t_a = max(d_vals, key=d_vals.get)
    return B, t_m, int(t_a)


def main():
    t0 = time.time()
    print("Loading citation data...")
    pat    = pd.read_parquet(PATENTS_80S_FILE, columns=["patent_id", "grant_date"])
    annual = pd.read_parquet(CITATION_ANNUAL_FILE)

    pat["grant_year"] = pat["grant_date"].dt.year
    gy = dict(zip(pat["patent_id"], pat["grant_year"]))

    annual = annual[annual["patent_id"].isin(gy)].copy()
    annual["age"] = annual["cite_year"] - annual["patent_id"].map(gy)
    annual = annual[(annual["age"] >= 0) & (annual["age"] <= MAX_AGE_11)]

    pivot = annual.groupby(["patent_id", "age"])["fwd_count"].sum().unstack(fill_value=0)
    for age in range(MAX_AGE_11 + 1):
        if age not in pivot.columns:
            pivot[age] = 0
    pivot = pivot[sorted(pivot.columns)]
    print(f"  Pivot: {pivot.shape}")

    print("Computing B_11yr...")
    results = []
    n = len(pivot)
    for i, (pid, row) in enumerate(pivot.iterrows()):
        B, t_m, t_a = beauty_coefficient(row.copy())
        results.append({"patent_id": pid, "B_11yr": B,
                        "t_m_11yr": t_m, "t_a_11yr": t_a,
                        "cum_cite_11yr": int(row.sum()),
                        "peak_cite_11yr": int(row.max())})
        if (i + 1) % 50_000 == 0:
            print(f"  {i+1:,}/{n:,}", end="\r")

    df = pd.DataFrame(results)
    out = FEATURES_DIR / "b_11yr.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved: {out.name}  ({len(df):,} rows)  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
