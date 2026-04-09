"""
Step 5: 3.5 / 7.5 / 11.5년 시점별 feature 테이블 구축
Output:
  features/features_3_5yr.parquet
  features/features_7_5yr.parquet
  features/features_11_5yr.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from config import *

MAINT_COST = {3.5: 1.0, 7.5: 1.88, 11.5: 3.85}


def build_citation_features(pivot: pd.DataFrame, cutoff_age: float) -> pd.DataFrame:
    max_age = int(cutoff_age)
    cols    = [c for c in pivot.columns if c <= max_age]
    sub     = pivot[cols].copy()

    feats = pd.DataFrame(index=pivot.index)
    feats["cum_citations"] = sub.sum(axis=1)

    for age in range(0, max_age + 1):
        feats[f"cite_yr{age}"] = sub[age] if age in sub.columns else 0

    feats["cite_last1yr"] = sub[max_age] if max_age in sub.columns else 0

    last3 = [a for a in range(max(0, max_age - 2), max_age + 1) if a in sub.columns]
    feats["cite_last3yr"] = sub[last3].sum(axis=1) if last3 else 0

    c0  = sub[0].clip(lower=0) if 0 in sub.columns else pd.Series(0, index=sub.index)
    c_t = sub[max_age] if max_age in sub.columns else pd.Series(0, index=sub.index)
    feats["cite_growth_rate"] = (c_t - c0) / (c0 + 1)

    if max_age >= 3:
        c_prev = sub[max_age - 3] if (max_age - 3) in sub.columns else pd.Series(0, index=sub.index)
        feats["cite_growth_last3"] = (c_t - c_prev) / (c_prev + 1)
    else:
        feats["cite_growth_last3"] = 0.0

    if max_age >= 2:
        c_m1 = sub[max_age - 1] if (max_age - 1) in sub.columns else pd.Series(0, index=sub.index)
        c_m2 = sub[max_age - 2] if (max_age - 2) in sub.columns else pd.Series(0, index=sub.index)
        feats["cite_acceleration"] = (c_t - c_m1) - (c_m1 - c_m2)
    else:
        feats["cite_acceleration"] = 0.0

    total = feats["cum_citations"].clip(lower=1)
    feats["cite_peak_ratio"]   = sub.max(axis=1) / total
    feats["cite_active_years"] = (sub > 0).sum(axis=1)
    feats["zero_citation"]     = (feats["cum_citations"] == 0).astype(int)

    return feats.reset_index()


def main():
    t_start = time.time()

    print("Loading processed data...")
    pat      = pd.read_parquet(PATENTS_80S_FILE)
    annual   = pd.read_parquet(CITATION_ANNUAL_FILE)
    bwd      = pd.read_parquet(BACKWARD_CIT_FILE)
    foreign  = pd.read_parquet(FOREIGN_CIT_FILE_OUT)
    ipc      = pd.read_parquet(IPC_FILE_OUT)
    assignee = pd.read_parquet(ASSIGNEE_FILE_OUT)
    inventor = pd.read_parquet(INVENTOR_FILE_OUT)

    pat["grant_year"] = pat["grant_date"].dt.year
    gy = dict(zip(pat["patent_id"], pat["grant_year"]))

    annual = annual[annual["patent_id"].isin(gy)].copy()
    annual["age"] = annual["cite_year"] - annual["patent_id"].map(gy)
    annual = annual[(annual["age"] >= 0) & (annual["age"] <= 20)]

    print("Building citation pivot...")
    pivot = (
        annual.groupby(["patent_id", "age"])["fwd_count"]
        .sum()
        .unstack(fill_value=0)
    )
    for age in range(21):
        if age not in pivot.columns:
            pivot[age] = 0
    pivot = pivot[sorted(pivot.columns)]
    print(f"  Pivot shape: {pivot.shape}")

    static = (
        pat[["patent_id", "grant_year", "num_claims", "filing_to_grant_days",
             "small_entity", "num_figures", "num_sheets"]]
        .merge(ipc,      on="patent_id", how="left")
        .merge(assignee, on="patent_id", how="left")
        .merge(inventor, on="patent_id", how="left")
        .merge(bwd,      on="patent_id", how="left")
        .merge(foreign,  on="patent_id", how="left")
    )
    static["num_figures"]    = static["num_figures"].fillna(0)
    static["num_sheets"]     = static["num_sheets"].fillna(0)
    static["inventor_count"] = static["inventor_count"].fillna(1)
    static["bwd_us_total"]   = static["bwd_us_total"].fillna(0)
    static["bwd_examiner"]   = static["bwd_examiner"].fillna(0)
    static["bwd_applicant"]  = static["bwd_applicant"].fillna(0)
    static["bwd_foreign"]    = static["bwd_foreign"].fillna(0)
    static["is_organization"]= static["is_organization"].fillna(0)
    static["is_us_company"]  = static["is_us_company"].fillna(0)
    static["is_foreign"]     = static["is_foreign"].fillna(0)
    static["is_individual"]  = static["is_individual"].fillna(0)
    static["small_entity"]   = (static["small_entity"] == "Y").astype(int)
    static["bwd_total"]      = static["bwd_us_total"] + static["bwd_foreign"]
    static["bwd_examiner_ratio"] = static["bwd_examiner"] / (static["bwd_us_total"] + 1)

    output_map = {3.5: FEATURES_35_FILE, 7.5: FEATURES_75_FILE, 11.5: FEATURES_115_FILE}

    for cutoff, outfile in output_map.items():
        print(f"\nBuilding features at t={cutoff}yr...")
        t0 = time.time()
        cit_feats = build_citation_features(pivot, cutoff)
        cit_feats.columns = [
            "patent_id" if c == "patent_id"
            else f"t{str(cutoff).replace('.', '')}__{c}"
            for c in cit_feats.columns
        ]
        df = static.merge(cit_feats, on="patent_id", how="left")
        df["decision_point"] = cutoff
        df["maint_cost"]     = MAINT_COST[cutoff]
        df.to_parquet(outfile, index=False)
        print(f"  Saved {outfile.name}: {len(df):,} rows, {len(df.columns)} cols  ({time.time()-t0:.1f}s)")

    print(f"\nStep 5 done. Total: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
