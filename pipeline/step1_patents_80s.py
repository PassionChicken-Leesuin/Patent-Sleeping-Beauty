"""
Step 1: 1980-1989 등록 특허 목록 추출
Output: processed/patents_80s.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from config import *


def load_maint_patents_80s():
    print("Loading MaintFeeEvents...")
    t0 = time.time()
    cols = {0: "record_id", 1: "app_id", 2: "small_entity",
            3: "filing_date_maint", 4: "grant_date_maint",
            5: "event_date", 6: "event_code"}
    chunks = []
    with open(MAINT_FILE, "r", encoding="utf-8", errors="replace") as f:
        buffer = []
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) >= 6:
                buffer.append(parts[:7] if len(parts) >= 7 else parts + [""])
            if len(buffer) >= 500_000:
                df = pd.DataFrame(buffer, columns=list(cols.values()))
                mask = (df["grant_date_maint"] >= "19800101") & (df["grant_date_maint"] <= "19891231")
                chunks.append(df[mask].copy())
                buffer = []
                print(f"  processed {i+1:,} lines...", end="\r")
        if buffer:
            df = pd.DataFrame(buffer, columns=list(cols.values()))
            mask = (df["grant_date_maint"] >= "19800101") & (df["grant_date_maint"] <= "19891231")
            chunks.append(df[mask].copy())

    maint = pd.concat(chunks, ignore_index=True)
    print(f"\n  MaintFeeEvents 80s rows: {len(maint):,}  ({time.time()-t0:.1f}s)")

    maint["patent_id"] = maint["record_id"].str.lstrip("0")
    maint["grant_date_maint"] = pd.to_datetime(maint["grant_date_maint"], format="%Y%m%d", errors="coerce")

    pat_info = (
        maint[["patent_id", "grant_date_maint", "small_entity"]]
        .drop_duplicates("patent_id")
        .rename(columns={"grant_date_maint": "grant_date"})
    )

    maint_out = maint[["patent_id", "event_date", "event_code"]].copy()
    maint_out["event_date"] = pd.to_datetime(maint_out["event_date"], format="%Y%m%d", errors="coerce")
    maint_out.to_parquet(MAINT_EVENTS_FILE, index=False)
    print(f"  Saved: {MAINT_EVENTS_FILE.name}")
    print(f"  Unique 80s patents (maint): {len(pat_info):,}")
    return pat_info


def load_patent_meta(patent_ids):
    print("Loading g_patent...")
    t0 = time.time()
    chunks = []
    for chunk in pd.read_csv(PATENT_FILE, sep="\t", chunksize=200_000,
                              usecols=["patent_id", "patent_type", "patent_date",
                                       "num_claims", "withdrawn"],
                              dtype=str, low_memory=False):
        mask = chunk["patent_id"].isin(patent_ids)
        if mask.any():
            chunks.append(chunk[mask])
    df = pd.concat(chunks, ignore_index=True)
    df["patent_date"] = pd.to_datetime(df["patent_date"], errors="coerce")
    df["num_claims"]  = pd.to_numeric(df["num_claims"], errors="coerce")
    df["withdrawn"]   = pd.to_numeric(df["withdrawn"], errors="coerce").fillna(0).astype(int)
    print(f"  g_patent matched: {len(df):,}  ({time.time()-t0:.1f}s)")
    return df


def load_application(patent_ids):
    print("Loading g_application...")
    t0 = time.time()
    chunks = []
    for chunk in pd.read_csv(APPLICATION_FILE, sep="\t", chunksize=200_000,
                              usecols=["patent_id", "filing_date"],
                              dtype=str, low_memory=False):
        mask = chunk["patent_id"].isin(patent_ids)
        if mask.any():
            chunks.append(chunk[mask])
    df = pd.concat(chunks, ignore_index=True).drop_duplicates("patent_id")
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    print(f"  g_application matched: {len(df):,}  ({time.time()-t0:.1f}s)")
    return df


def load_figures(patent_ids):
    print("Loading g_figures...")
    t0 = time.time()
    chunks = []
    for chunk in pd.read_csv(FIGURES_FILE, sep="\t", chunksize=200_000,
                              usecols=["patent_id", "num_figures", "num_sheets"],
                              dtype=str, low_memory=False):
        mask = chunk["patent_id"].isin(patent_ids)
        if mask.any():
            chunks.append(chunk[mask])
    df = pd.concat(chunks, ignore_index=True).drop_duplicates("patent_id")
    df["num_figures"] = pd.to_numeric(df["num_figures"], errors="coerce").fillna(0)
    df["num_sheets"]  = pd.to_numeric(df["num_sheets"],  errors="coerce").fillna(0)
    print(f"  g_figures matched: {len(df):,}  ({time.time()-t0:.1f}s)")
    return df


def main():
    t_start = time.time()
    pat_maint  = load_maint_patents_80s()
    patent_ids = set(pat_maint["patent_id"])
    pat_meta   = load_patent_meta(patent_ids)
    pat_app    = load_application(patent_ids)
    pat_fig    = load_figures(patent_ids)

    df = pat_maint.merge(pat_meta, on="patent_id", how="left")
    df = df.merge(pat_app,         on="patent_id", how="left")
    df = df.merge(pat_fig,         on="patent_id", how="left")

    df["grant_year"]          = df["grant_date"].dt.year
    df["filing_to_grant_days"]= (df["grant_date"] - df["filing_date"]).dt.days
    df = df[df["patent_type"] == "utility"].copy()
    df = df[df["withdrawn"] == 0].copy()

    print(f"\nFinal 80s utility patents: {len(df):,}")
    print(df["grant_year"].value_counts().sort_index())

    df.to_parquet(PATENTS_80S_FILE, index=False)
    print(f"\nSaved: {PATENTS_80S_FILE}")
    print(f"Total time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
