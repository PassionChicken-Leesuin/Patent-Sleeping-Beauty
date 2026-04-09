"""
Step 3: 정적 feature 추출 - IPC, Assignee, Inventor
Output:
  processed/ipc_main.parquet
  processed/assignee.parquet
  processed/inventor_count.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import time
from config import *


def build_ipc(patent_ids: set):
    print("Building IPC...")
    t0 = time.time()
    records = []
    for chunk in pd.read_csv(IPC_FILE, sep="\t", chunksize=300_000,
                              usecols=["patent_id", "ipc_sequence", "section",
                                       "ipc_class", "subclass"],
                              dtype=str, low_memory=False):
        mask = chunk["patent_id"].isin(patent_ids)
        if mask.any():
            records.append(chunk[mask])
    ipc = pd.concat(records, ignore_index=True)
    ipc["ipc_sequence"] = pd.to_numeric(ipc["ipc_sequence"], errors="coerce").fillna(999)
    ipc_main = ipc[ipc["ipc_sequence"] == 0].drop_duplicates("patent_id").copy()
    ipc_main["ipc_subclass"]   = (ipc_main["section"].fillna("") +
                                   ipc_main["ipc_class"].fillna("") +
                                   ipc_main["subclass"].fillna(""))
    ipc_main["ipc_section"]    = ipc_main["section"].fillna("")
    ipc_main["ipc_class_full"] = ipc_main["section"].fillna("") + ipc_main["ipc_class"].fillna("")
    out = ipc_main[["patent_id", "ipc_section", "ipc_class_full", "ipc_subclass"]].copy()
    out.to_parquet(IPC_FILE_OUT, index=False)
    print(f"  Saved ipc_main: {len(out):,} rows  ({time.time()-t0:.1f}s)")
    return out


def build_assignee(patent_ids: set):
    print("Building assignee...")
    t0 = time.time()
    records = []
    for chunk in pd.read_csv(ASSIGNEE_FILE, sep="\t", chunksize=300_000,
                              usecols=["patent_id", "assignee_sequence", "assignee_type",
                                       "disambig_assignee_organization",
                                       "disambig_assignee_individual_name_last"],
                              dtype=str, low_memory=False):
        mask = chunk["patent_id"].isin(patent_ids)
        if mask.any():
            records.append(chunk[mask])
    asgn = pd.concat(records, ignore_index=True)
    asgn["assignee_sequence"] = pd.to_numeric(asgn["assignee_sequence"], errors="coerce").fillna(999)
    asgn_main = asgn[asgn["assignee_sequence"] == 0].drop_duplicates("patent_id").copy()
    asgn_main["assignee_type"]  = pd.to_numeric(asgn_main["assignee_type"], errors="coerce")
    asgn_main["is_organization"]= asgn_main["disambig_assignee_organization"].notna().astype(int)
    asgn_main["is_us_company"]  = asgn_main["assignee_type"].isin([2]).astype(int)
    asgn_main["is_foreign"]     = asgn_main["assignee_type"].isin([3, 9]).astype(int)
    asgn_main["is_individual"]  = asgn_main["assignee_type"].isin([1]).astype(int)
    out = asgn_main[["patent_id", "assignee_type", "is_organization",
                      "is_us_company", "is_foreign", "is_individual"]].copy()
    out.to_parquet(ASSIGNEE_FILE_OUT, index=False)
    print(f"  Saved assignee: {len(out):,} rows  ({time.time()-t0:.1f}s)")
    return out


def build_inventor_count(patent_ids: set):
    print("Building inventor count...")
    t0 = time.time()
    records = []
    for chunk in pd.read_csv(INVENTOR_FILE, sep="\t", chunksize=300_000,
                              usecols=["patent_id", "inventor_id"],
                              dtype=str, low_memory=False):
        mask = chunk["patent_id"].isin(patent_ids)
        if mask.any():
            records.append(chunk[mask])
    inv = pd.concat(records, ignore_index=True)
    cnt = inv.groupby("patent_id")["inventor_id"].nunique().reset_index(name="inventor_count")
    cnt.to_parquet(INVENTOR_FILE_OUT, index=False)
    print(f"  Saved inventor_count: {len(cnt):,} rows  ({time.time()-t0:.1f}s)")
    return cnt


def main():
    t_start = time.time()
    pat = pd.read_parquet(PATENTS_80S_FILE, columns=["patent_id"])
    patent_ids = set(pat["patent_id"])
    print(f"80s patents: {len(patent_ids):,}")
    build_ipc(patent_ids)
    build_assignee(patent_ids)
    build_inventor_count(patent_ids)
    print(f"\nStep 3 done. Total: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
