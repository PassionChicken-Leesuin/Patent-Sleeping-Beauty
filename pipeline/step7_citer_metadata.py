"""
Step 7: Citer metadata 병합

forward_citation_raw.parquet (step2b) 의 citing_patent_id 에
각종 metadata 를 붙인다:
    - citing_grant_year        (이미 step2b 에서 cite_year 로 저장됨)
    - citing_ipc_subclass      from g_ipc_at_issue
    - citing_assignee_id       from g_assignee_disambiguated (disambiguated)
    - citing_assignee_country  from g_assignee_disambiguated (location_id 가 의미 있음)
    - citing_inventor_count    from g_inventor_disambiguated

Output:
    processed/citer_metadata.parquet
        per (cited_patent_id, citing_patent_id, age) row 에 위 컬럼 추가
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import pandas as pd

from config import (
    IPC_FILE, ASSIGNEE_FILE, INVENTOR_FILE,
    PROCESSED_DIR,
)

RAW_FWD_FILE       = PROCESSED_DIR / "forward_citation_raw.parquet"
CITER_METADATA_FILE = PROCESSED_DIR / "citer_metadata.parquet"


def load_ipc_subclass(citing_ids: set) -> pd.DataFrame:
    """g_ipc_at_issue 에서 citing patent 의 primary IPC subclass 추출."""
    print("Loading IPC (subset to citing patents)...")
    t0 = time.time()
    parts = []
    for chunk in pd.read_csv(IPC_FILE, sep="\t", chunksize=300_000,
                              usecols=["patent_id", "ipc_sequence",
                                       "section", "ipc_class", "subclass"],
                              dtype=str, low_memory=False):
        mask = chunk["patent_id"].isin(citing_ids)
        if mask.any():
            parts.append(chunk[mask])
    df = pd.concat(parts, ignore_index=True)
    df["ipc_sequence"] = pd.to_numeric(df["ipc_sequence"], errors="coerce")
    df = df.sort_values(["patent_id", "ipc_sequence"]).drop_duplicates("patent_id")
    df["citing_ipc_subclass"] = (
        df["section"].fillna("") + df["ipc_class"].fillna("") + df["subclass"].fillna("")
    )
    df = df[["patent_id", "citing_ipc_subclass"]]
    print(f"  matched: {len(df):,}  ({time.time()-t0:.1f}s)")
    return df


def load_assignee(citing_ids: set) -> pd.DataFrame:
    """g_assignee 에서 primary assignee_id 추출."""
    print("Loading assignee (subset)...")
    t0 = time.time()
    parts = []
    for chunk in pd.read_csv(ASSIGNEE_FILE, sep="\t", chunksize=300_000,
                              usecols=["patent_id", "assignee_sequence",
                                       "assignee_id", "location_id"],
                              dtype=str, low_memory=False):
        mask = chunk["patent_id"].isin(citing_ids)
        if mask.any():
            parts.append(chunk[mask])
    df = pd.concat(parts, ignore_index=True)
    df["assignee_sequence"] = pd.to_numeric(df["assignee_sequence"], errors="coerce")
    df = df.sort_values(["patent_id", "assignee_sequence"]).drop_duplicates("patent_id")
    df = df.rename(columns={
        "assignee_id":  "citing_assignee_id",
        "location_id":  "citing_location_id",
    })[["patent_id", "citing_assignee_id", "citing_location_id"]]
    print(f"  matched: {len(df):,}  ({time.time()-t0:.1f}s)")
    return df


def load_inventor_count(citing_ids: set) -> pd.DataFrame:
    """citing 특허별 발명자 수."""
    print("Loading inventor count (subset)...")
    t0 = time.time()
    parts = []
    for chunk in pd.read_csv(INVENTOR_FILE, sep="\t", chunksize=300_000,
                              usecols=["patent_id", "inventor_id"],
                              dtype=str, low_memory=False):
        mask = chunk["patent_id"].isin(citing_ids)
        if mask.any():
            parts.append(chunk[mask])
    df = pd.concat(parts, ignore_index=True)
    agg = df.groupby("patent_id").size().reset_index(name="citing_inventor_count")
    print(f"  matched: {len(agg):,}  ({time.time()-t0:.1f}s)")
    return agg


def main():
    t_start = time.time()

    print("Loading raw forward citations...")
    fwd = pd.read_parquet(RAW_FWD_FILE)
    print(f"  rows: {len(fwd):,}")

    citing_ids = set(fwd["citing_patent_id"].unique())
    print(f"  unique citing patents: {len(citing_ids):,}")

    ipc = load_ipc_subclass(citing_ids)
    asg = load_assignee(citing_ids)
    inv = load_inventor_count(citing_ids)

    print("\nJoining...")
    t0 = time.time()
    out = fwd.merge(ipc.rename(columns={"patent_id": "citing_patent_id"}),
                    on="citing_patent_id", how="left")
    out = out.merge(asg.rename(columns={"patent_id": "citing_patent_id"}),
                    on="citing_patent_id", how="left")
    out = out.merge(inv.rename(columns={"patent_id": "citing_patent_id"}),
                    on="citing_patent_id", how="left")
    print(f"  joined ({time.time()-t0:.1f}s)")

    # fillna for robust downstream
    out["citing_ipc_subclass"]   = out["citing_ipc_subclass"].fillna("UNK")
    out["citing_assignee_id"]    = out["citing_assignee_id"].fillna("UNK")
    out["citing_location_id"]    = out["citing_location_id"].fillna("UNK")
    out["citing_inventor_count"] = out["citing_inventor_count"].fillna(1).astype(int)

    out.to_parquet(CITER_METADATA_FILE, index=False)
    print(f"\nSaved: {CITER_METADATA_FILE.name}  ({len(out):,} rows)")
    print(f"Step 7 done. ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
