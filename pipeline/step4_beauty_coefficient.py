"""
Step 4: Beauty Coefficient 계산 및 PSB 라벨 생성 (다중 임계값)

- 각 특허의 연도별 forward citation (age=0: 등록 연도)
- Beauty Coefficient B 계산
- Awakening time t_a 계산
- PSB_THRESHOLDS 각각에 대해 IPC subclass 내 상위 q% → PSB=1 라벨 생성

Output:
  features/labels_thr001.parquet  (top 0.1%)
  features/labels_thr005.parquet  (top 0.5%)
  features/labels_thr010.parquet  (top 1.0%)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from config import (
    PATENTS_80S_FILE, CITATION_ANNUAL_FILE, IPC_FILE_OUT,
    PSB_THRESHOLDS, labels_file, thr_tag,
)


def beauty_coefficient(c: pd.Series) -> tuple:
    """
    c: index=age(0,1,2,...), value=citation count
    Returns: (B, t_m, t_a)

    B = Σ_{t=0}^{t_m} (l(t) - c(t)) / max(1, c(t))
    where l(t) = baseline line from c(0) to c(t_m)
    """
    if len(c) == 0:
        return 0.0, 0, 0

    t_m = int(c.idxmax())
    if t_m == 0:
        return 0.0, 0, 0

    c0   = float(c.get(0, 0))
    c_tm = float(c[t_m])
    slope = (c_tm - c0) / t_m

    B = 0.0
    for t in range(0, t_m + 1):
        c_t = float(c.get(t, 0))
        l_t = slope * t + c0
        B  += (l_t - c_t) / max(1.0, c_t)

    # Awakening time: baseline까지 수직 거리 최대 시점
    denom_ta = np.sqrt((c_tm - c0) ** 2 + t_m ** 2)
    if denom_ta == 0:
        t_a = 0
    else:
        d_vals = {
            t: abs((c_tm - c0) * t - t_m * float(c.get(t, 0)) + t_m * c0) / denom_ta
            for t in range(0, t_m + 1)
        }
        t_a = max(d_vals, key=d_vals.get)

    return B, t_m, int(t_a)


def main():
    t_start = time.time()

    print("Loading data...")
    pat    = pd.read_parquet(PATENTS_80S_FILE, columns=["patent_id", "grant_date"])
    annual = pd.read_parquet(CITATION_ANNUAL_FILE)
    ipc    = pd.read_parquet(IPC_FILE_OUT, columns=["patent_id", "ipc_subclass"])

    pat["grant_year"] = pat["grant_date"].dt.year
    gy = dict(zip(pat["patent_id"], pat["grant_year"]))
    print(f"  Patents: {len(pat):,}")

    # citation age 계산
    annual = annual[annual["patent_id"].isin(gy)].copy()
    annual["age"] = annual["cite_year"] - annual["patent_id"].map(gy)
    annual = annual[(annual["age"] >= 0) & (annual["age"] <= 25)]

    # pivot: patent_id × age
    pivot = (
        annual.groupby(["patent_id", "age"])["fwd_count"]
        .sum()
        .unstack(fill_value=0)
    )
    print(f"  Pivot shape: {pivot.shape}")

    # Beauty Coefficient 계산
    print("Computing Beauty Coefficients...")
    records = []
    n = len(pivot)
    for i, (pid, row) in enumerate(pivot.iterrows()):
        if row.sum() == 0:
            records.append({"patent_id": pid, "B": 0.0, "t_m": 0, "t_a": 0,
                            "total_citations": 0, "peak_citations": 0})
        else:
            B, t_m, t_a = beauty_coefficient(row)
            records.append({
                "patent_id":       pid,
                "B":               B,
                "t_m":             t_m,
                "t_a":             t_a,
                "total_citations": int(row.sum()),
                "peak_citations":  int(row.max()),
            })
        if (i + 1) % 50_000 == 0:
            print(f"  {i+1:,}/{n:,}...", end="\r")

    print(f"\n  Computed {len(records):,} patents")

    base = pd.DataFrame(records)
    base = base.merge(ipc, on="patent_id", how="left")
    base["ipc_subclass"] = base["ipc_subclass"].fillna("UNKNOWN")

    # ── 각 임계값별 라벨 생성 ───────────────────────────
    for thr in PSB_THRESHOLDS:
        print(f"\nAssigning PSB labels (threshold={thr})...")
        df = base.copy()
        df["psb"] = 0

        q = 1.0 - thr   # quantile (e.g. 0.001 → 99.9th percentile)
        for subclass, grp in df.groupby("ipc_subclass"):
            if len(grp) < 10:
                continue
            threshold_val = grp["B"].quantile(q)
            if threshold_val <= 0:
                continue
            df.loc[grp[grp["B"] >= threshold_val].index, "psb"] = 1

        psb_n = df["psb"].sum()
        total = len(df)
        print(f"  PSB=1: {psb_n:,} ({psb_n/total*100:.3f}%)  PSB=0: {total-psb_n:,}")

        out = labels_file(thr)
        df.to_parquet(out, index=False)
        print(f"  Saved: {out.name}")

    print(f"\nStep 4 done. Total: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
