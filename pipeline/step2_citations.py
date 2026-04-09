"""
Step 2: Citation 데이터 처리
Output:
  processed/citation_annual.parquet   (patent_id, cite_year, fwd_count)
  processed/backward_citation.parquet (patent_id, bwd_us, bwd_examiner, bwd_applicant)
  processed/foreign_citation.parquet  (patent_id, bwd_foreign)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import time
from config import *


def load_all_patent_dates() -> dict:
    print("Loading all patent grant dates...")
    t0 = time.time()
    date_map = {}
    for chunk in pd.read_csv(PATENT_FILE, sep="\t", chunksize=300_000,
                              usecols=["patent_id", "patent_date"],
                              dtype=str, low_memory=False):
        chunk["patent_date"] = pd.to_datetime(chunk["patent_date"], errors="coerce")
        for pid, dt in zip(chunk["patent_id"], chunk["patent_date"]):
            if pd.notna(dt):
                date_map[pid] = dt.year
    print(f"  Loaded {len(date_map):,} patent dates  ({time.time()-t0:.1f}s)")
    return date_map


def build_forward_citation_annual(patent_ids: set, all_patent_dates: dict):
    print("Building forward citations...")
    t0 = time.time()
    records = []
    total_rows = 0
    for chunk in pd.read_csv(CITATION_FILE, sep="\t", chunksize=500_000,
                              usecols=["patent_id", "citation_patent_id", "citation_category"],
                              dtype=str, low_memory=False):
        total_rows += len(chunk)
        mask = chunk["citation_patent_id"].isin(patent_ids)
        if mask.any():
            sub = chunk[mask].copy()
            sub["cite_year"] = sub["patent_id"].map(all_patent_dates)
            sub = sub.dropna(subset=["cite_year"])
            sub["cite_year"] = sub["cite_year"].astype(int)
            records.append(sub[["citation_patent_id", "cite_year", "citation_category"]])
        print(f"  fwd: {total_rows:,} rows scanned...", end="\r")

    print(f"\n  Scan done ({time.time()-t0:.1f}s)")
    fwd = pd.concat(records, ignore_index=True).rename(
        columns={"citation_patent_id": "patent_id"})
    annual = (
        fwd.groupby(["patent_id", "cite_year"])
        .size()
        .reset_index(name="fwd_count")
    )
    annual.to_parquet(CITATION_ANNUAL_FILE, index=False)
    print(f"  Saved citation_annual: {len(annual):,} rows")
    return annual


def build_backward_citation(patent_ids: set):
    print("Building backward citations (US)...")
    t0 = time.time()
    records = []
    for chunk in pd.read_csv(CITATION_FILE, sep="\t", chunksize=500_000,
                              usecols=["patent_id", "citation_patent_id", "citation_category"],
                              dtype=str, low_memory=False):
        mask = chunk["patent_id"].isin(patent_ids)
        if mask.any():
            records.append(chunk[mask])
    bwd = pd.concat(records, ignore_index=True)
    print(f"  backward rows: {len(bwd):,}  ({time.time()-t0:.1f}s)")
    agg = bwd.groupby("patent_id").agg(
        bwd_us_total  = ("citation_patent_id", "count"),
        bwd_examiner  = ("citation_category",  lambda x: (x == "cited by examiner").sum()),
        bwd_applicant = ("citation_category",  lambda x: (x == "cited by applicant").sum()),
    ).reset_index()
    agg.to_parquet(BACKWARD_CIT_FILE, index=False)
    print(f"  Saved backward_citation: {len(agg):,} rows")
    return agg


def build_foreign_citation(patent_ids: set):
    print("Building foreign backward citations...")
    t0 = time.time()
    records = []
    for chunk in pd.read_csv(FOREIGN_FILE, sep="\t", chunksize=300_000,
                              usecols=["patent_id", "citation_sequence"],
                              dtype=str, low_memory=False):
        mask = chunk["patent_id"].isin(patent_ids)
        if mask.any():
            records.append(chunk[mask])
    foreign = pd.concat(records, ignore_index=True)
    agg = foreign.groupby("patent_id").size().reset_index(name="bwd_foreign")
    agg.to_parquet(FOREIGN_CIT_FILE_OUT, index=False)
    print(f"  Saved foreign_citation: {len(agg):,} rows  ({time.time()-t0:.1f}s)")
    return agg


def main():
    t_start = time.time()
    pat = pd.read_parquet(PATENTS_80S_FILE, columns=["patent_id", "grant_date"])
    patent_ids = set(pat["patent_id"])
    print(f"80s patents: {len(patent_ids):,}")
    all_patent_dates = load_all_patent_dates()
    build_forward_citation_annual(patent_ids, all_patent_dates)
    build_backward_citation(patent_ids)
    build_foreign_citation(patent_ids)
    print(f"\nStep 2 done. Total: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
