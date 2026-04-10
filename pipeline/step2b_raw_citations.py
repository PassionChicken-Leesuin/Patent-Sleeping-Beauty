"""
Step 2b: Forward citation raw pair 저장

step2 에서는 (patent_id, cite_year, fwd_count) 로 집계만 저장해서
citing patent 의 식별자가 사라진다. 동적 citer-quality feature
(F01~F10) 계산을 위해 raw triple 을 보존한다.

Output:
    processed/forward_citation_raw.parquet
        columns:
            cited_patent_id  : 1980s cohort patent (we are predicting)
            citing_patent_id : newer patent that cites it
            cite_year        : citing patent's grant year
            age              : cite_year - cited_grant_year (int)
            citation_category: examiner/applicant/etc.

주의: 80s cohort (516,225건) 의 raw forward citation pair 만 저장.
전체 dataset 이 아니다.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import pandas as pd

from config import (
    PATENT_FILE, CITATION_FILE,
    PATENTS_80S_FILE, PROCESSED_DIR,
)

RAW_FWD_FILE = PROCESSED_DIR / "forward_citation_raw.parquet"


def load_all_patent_years() -> dict:
    """전체 g_patent.tsv 에서 patent_id -> grant_year 맵 구축."""
    print("Loading all patent grant years...")
    t0 = time.time()
    year_map = {}
    for chunk in pd.read_csv(PATENT_FILE, sep="\t", chunksize=300_000,
                              usecols=["patent_id", "patent_date"],
                              dtype=str, low_memory=False):
        chunk["yr"] = pd.to_datetime(chunk["patent_date"], errors="coerce").dt.year
        for pid, yr in zip(chunk["patent_id"], chunk["yr"]):
            if pd.notna(yr):
                year_map[pid] = int(yr)
    print(f"  Loaded {len(year_map):,} patent years  ({time.time()-t0:.1f}s)")
    return year_map


def build_raw_forward_citations(cited_ids: set, cited_year_map: dict,
                                 all_year_map: dict) -> pd.DataFrame:
    """
    cited_ids 에 포함된 특허를 인용한 raw citation pair 추출.
    """
    print("Scanning g_us_patent_citation.tsv ...")
    t0 = time.time()
    parts = []
    total = 0
    for chunk in pd.read_csv(CITATION_FILE, sep="\t", chunksize=500_000,
                              usecols=["patent_id", "citation_patent_id",
                                       "citation_category"],
                              dtype=str, low_memory=False):
        total += len(chunk)
        mask = chunk["citation_patent_id"].isin(cited_ids)
        if mask.any():
            sub = chunk[mask].copy()
            sub["cite_year"] = sub["patent_id"].map(all_year_map)
            sub = sub.dropna(subset=["cite_year"])
            sub["cite_year"] = sub["cite_year"].astype(int)
            sub["cited_year"] = sub["citation_patent_id"].map(cited_year_map)
            sub = sub.dropna(subset=["cited_year"])
            sub["age"] = (sub["cite_year"] - sub["cited_year"]).astype(int)
            sub = sub[(sub["age"] >= 0) & (sub["age"] <= 25)]
            parts.append(sub[[
                "citation_patent_id", "patent_id",
                "cite_year", "age", "citation_category",
            ]])
        print(f"  scanned {total:,}", end="\r")

    print(f"\n  done ({time.time()-t0:.1f}s)")
    fwd = pd.concat(parts, ignore_index=True)
    fwd = fwd.rename(columns={
        "citation_patent_id": "cited_patent_id",
        "patent_id":          "citing_patent_id",
    })
    print(f"  raw rows: {len(fwd):,}")
    return fwd


def main():
    t_start = time.time()

    pat = pd.read_parquet(PATENTS_80S_FILE, columns=["patent_id", "grant_date"])
    pat["grant_year"] = pat["grant_date"].dt.year
    cited_year_map = dict(zip(pat["patent_id"].astype(str),
                               pat["grant_year"].astype(int)))
    cited_ids = set(cited_year_map.keys())
    print(f"80s cohort: {len(cited_ids):,} patents")

    all_year_map = load_all_patent_years()
    fwd = build_raw_forward_citations(cited_ids, cited_year_map, all_year_map)

    fwd.to_parquet(RAW_FWD_FILE, index=False)
    print(f"\nSaved: {RAW_FWD_FILE.name}  ({len(fwd):,} rows)")
    print(f"Step 2b done. ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
