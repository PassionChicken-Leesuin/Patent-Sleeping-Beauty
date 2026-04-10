"""
Step 8: Dynamic feature 계산

각 cutoff t in {3.5, 7.5, 11.5}yr 에서 관측 가능한 citation 만
사용해 동적 피처를 유도한다. 미래 정보 사용 금지.

피처 카테고리 (전략 문서 번호 기준):

== Citer-quality 동적 피처 (F01~F10) ==
    F01 citer_recency_mean          cited_grant_year 대비 citing_year gap 평균
    F02 citer_ipc_cross_rate        cited 와 IPC subclass 다른 citer 비율
    F03 citer_ipc_diversity         citer IPC subclass Shannon entropy
    F04 citer_assignee_hhi          citer assignee Herfindahl
    F05 citer_assignee_unique_rate  유일 assignee 수 / n_citer
    F06 citer_self_rate             cited 와 같은 assignee citer 비율
    F07 citer_loc_diversity         citer location 다양성 (Shannon)
    F08 citer_examiner_rate         examiner citation 비율
    F09 n_citers                    로그 스케일
    F10 cross_ipc_slope             cross-ipc rate 의 age 구간별 기울기
                                    (young half vs old half)

== Citation shape (F17~F22) ==
    F17 cite_burstiness             연도별 인용 분포의 coefficient of variation
    F18 cite_gini                   Gini (0 ~ 1)
    F19 cite_first_nonzero_age      첫 인용 연령
    F20 cite_longest_zero_run       t 까지 가장 긴 0-연속 구간
    F21 cite_resurgence_count       0 -> 양수 전환 횟수
    F22 cite_ipc_percentile         같은 IPC subclass + grant_year cohort
                                     내 누적 인용 percentile rank

Output:
    features/dynamic_t35.parquet
    features/dynamic_t75.parquet
    features/dynamic_t115.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import pandas as pd

from config import (
    PATENTS_80S_FILE, CITATION_ANNUAL_FILE, IPC_FILE_OUT,
    ASSIGNEE_FILE,
    PROCESSED_DIR, FEATURES_DIR,
    DECISION_POINTS,
)

CITER_METADATA_FILE = PROCESSED_DIR / "citer_metadata.parquet"


# ── 공통 통계 ────────────────────────────────────────
def shannon_entropy(counts: np.ndarray) -> float:
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts[counts > 0] / s
    return float(-(p * np.log(p)).sum())


def hhi(counts: np.ndarray) -> float:
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    return float((p ** 2).sum())


def gini(values: np.ndarray) -> float:
    v = np.sort(values.astype(float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * cum.sum() / v.sum()) / n)


# ── F01~F10: citer-quality ───────────────────────────
def compute_citer_features(citer_meta: pd.DataFrame,
                            cited_info: pd.DataFrame,
                            cutoff_age: float) -> pd.DataFrame:
    """
    citer_meta : per (cited_pid, citing_pid, age, ...) row
    cited_info : cited patent 자체의 ipc_subclass / assignee_id (self 비교용)
    cutoff_age : 3, 7, 11  (int 로 floor 처리)
    """
    max_age = int(np.floor(cutoff_age))
    cm = citer_meta[citer_meta["age"] <= max_age].copy()

    # cited 자체 metadata 와 병합
    cm = cm.merge(cited_info.rename(columns={
        "patent_id": "cited_patent_id",
        "ipc_subclass": "cited_ipc_subclass",
        "assignee_id":  "cited_assignee_id",
    }), on="cited_patent_id", how="left")

    # boolean flags
    cm["is_cross_ipc"] = (
        (cm["citing_ipc_subclass"] != cm["cited_ipc_subclass"])
        & (cm["citing_ipc_subclass"] != "UNK")
        & (cm["cited_ipc_subclass"].notna())
    ).astype(int)
    cm["is_self_assignee"] = (
        (cm["citing_assignee_id"] == cm["cited_assignee_id"])
        & (cm["citing_assignee_id"] != "UNK")
        & (cm["cited_assignee_id"].notna())
    ).astype(int)
    cm["is_examiner"] = (cm["citation_category"] == "cited by examiner").astype(int)

    # per-cited aggregate (vectorized)
    print(f"  aggregating per cited (cutoff age<={max_age}, rows={len(cm):,}) ...")
    t0 = time.time()

    # half split for slope
    half = max_age // 2
    cm["is_old"] = (cm["age"] > half).astype(int)
    cm["is_young"] = (cm["age"] <= half).astype(int)
    cm["cross_old"]   = cm["is_cross_ipc"] * cm["is_old"]
    cm["cross_young"] = cm["is_cross_ipc"] * cm["is_young"]

    # scalar aggregates via groupby.agg (very fast)
    g = cm.groupby("cited_patent_id")
    agg = g.agg(
        n_citers           = ("age", "size"),
        citer_recency_mean = ("age", "mean"),
        citer_ipc_cross_rate = ("is_cross_ipc", "mean"),
        citer_self_rate      = ("is_self_assignee", "mean"),
        citer_examiner_rate  = ("is_examiner", "mean"),
        n_old              = ("is_old",   "sum"),
        n_young            = ("is_young", "sum"),
        cross_old_sum      = ("cross_old",   "sum"),
        cross_young_sum    = ("cross_young", "sum"),
    )
    agg["cross_ipc_slope"] = (
        (agg["cross_old_sum"]   / agg["n_old"].replace(0, np.nan))
        - (agg["cross_young_sum"] / agg["n_young"].replace(0, np.nan))
    ).fillna(0.0)
    agg = agg.drop(columns=["n_old", "n_young", "cross_old_sum", "cross_young_sum"])

    # entropy/HHI features need value_counts → done via secondary groupby
    print(f"  computing entropy/HHI ({time.time()-t0:.1f}s) ...")
    t1 = time.time()

    # IPC entropy (Shannon)
    ipc_grp = (cm.groupby(["cited_patent_id", "citing_ipc_subclass"])
                  .size().rename("c").reset_index())
    ipc_grp["n"] = ipc_grp.groupby("cited_patent_id")["c"].transform("sum")
    ipc_grp["p"] = ipc_grp["c"] / ipc_grp["n"]
    ipc_grp["plogp"] = -ipc_grp["p"] * np.log(ipc_grp["p"].clip(lower=1e-12))
    ipc_ent = ipc_grp.groupby("cited_patent_id")["plogp"].sum().rename("citer_ipc_diversity")

    # Assignee HHI + unique count
    asg_grp = (cm.groupby(["cited_patent_id", "citing_assignee_id"])
                  .size().rename("c").reset_index())
    asg_grp["n"] = asg_grp.groupby("cited_patent_id")["c"].transform("sum")
    asg_grp["p"] = asg_grp["c"] / asg_grp["n"]
    asg_grp["p2"] = asg_grp["p"] ** 2
    asg_hhi = asg_grp.groupby("cited_patent_id")["p2"].sum().rename("citer_assignee_hhi")
    asg_unique = (asg_grp.groupby("cited_patent_id").size()
                          / asg_grp.groupby("cited_patent_id")["c"].sum()
                  ).rename("citer_assignee_unique_rate")

    # Location entropy
    loc_grp = (cm.groupby(["cited_patent_id", "citing_location_id"])
                  .size().rename("c").reset_index())
    loc_grp["n"] = loc_grp.groupby("cited_patent_id")["c"].transform("sum")
    loc_grp["p"] = loc_grp["c"] / loc_grp["n"]
    loc_grp["plogp"] = -loc_grp["p"] * np.log(loc_grp["p"].clip(lower=1e-12))
    loc_ent = loc_grp.groupby("cited_patent_id")["plogp"].sum().rename("citer_loc_diversity")

    print(f"  entropy/HHI done ({time.time()-t1:.1f}s)")

    out = (agg
           .join(ipc_ent, how="left")
           .join(asg_hhi, how="left")
           .join(asg_unique, how="left")
           .join(loc_ent, how="left")
           .reset_index())
    out["n_citers_log"] = np.log1p(out["n_citers"])
    print(f"  citer features: {len(out):,}  ({time.time()-t0:.1f}s)")
    return out


# ── F17~F22: citation shape ──────────────────────────
def compute_shape_features(pivot: pd.DataFrame,
                            cutoff_age: float) -> pd.DataFrame:
    """
    pivot : per patent row, columns = age (0..25), value = fwd count
    cutoff_age : int cap
    """
    max_age = int(np.floor(cutoff_age))
    cols = [c for c in pivot.columns if isinstance(c, (int, np.integer)) and c <= max_age]
    sub  = pivot[cols].fillna(0).values.astype(float)   # (N, T)
    N, T = sub.shape

    feats = {}

    # F17 burstiness = std / mean
    mu = sub.mean(axis=1)
    sd = sub.std(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        feats["cite_burstiness"] = np.where(mu > 0, sd / mu, 0.0)

    # F18 Gini per row (vectorized)
    sorted_sub = np.sort(sub, axis=1)
    cum = np.cumsum(sorted_sub, axis=1)
    row_sum = sorted_sub.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gini_vec = np.where(
            row_sum > 0,
            (T + 1 - 2 * cum.sum(axis=1) / row_sum) / T,
            0.0,
        )
    feats["cite_gini"] = gini_vec

    # F19 first nonzero age
    nz_mask = sub > 0
    first_nz = np.where(nz_mask.any(axis=1),
                         nz_mask.argmax(axis=1).astype(float),
                         -1.0)
    feats["cite_first_nonzero_age"] = first_nz

    # F20 longest zero run (vectorized)
    is_zero = (sub == 0).astype(int)
    # cumulative run via reset trick
    longest = np.zeros(N)
    cur = np.zeros(N)
    for j in range(T):
        cur = (cur + is_zero[:, j]) * is_zero[:, j]
        longest = np.maximum(longest, cur)
    feats["cite_longest_zero_run"] = longest

    # F21 resurgence count: count of 0->nonzero transitions
    if T >= 2:
        prev_zero = is_zero[:, :-1]
        cur_nonzero = (sub[:, 1:] > 0).astype(int)
        resurg = (prev_zero & cur_nonzero).sum(axis=1).astype(float)
    else:
        resurg = np.zeros(N)
    feats["cite_resurgence_count"] = resurg

    df = pd.DataFrame(feats)
    df.index = pivot.index
    return df.reset_index().rename(columns={"index": "patent_id"})


# ── F22: IPC percentile (동일 IPC subclass + grant_year cohort 내) ─
def compute_ipc_percentile(cited_info: pd.DataFrame,
                            cumcite: pd.DataFrame) -> pd.DataFrame:
    """
    cited_info : patent_id, ipc_subclass, grant_year
    cumcite    : patent_id, cum_to_cutoff
    """
    df = cited_info.merge(cumcite, on="patent_id", how="left")
    df["cum_to_cutoff"] = df["cum_to_cutoff"].fillna(0)
    df["cite_ipc_percentile"] = (
        df.groupby(["ipc_subclass", "grant_year"])["cum_to_cutoff"]
          .rank(pct=True, method="average")
          .fillna(0.5)
    )
    return df[["patent_id", "cite_ipc_percentile"]]


# ── Main ─────────────────────────────────────────────
def load_cited_assignee_id_from_raw() -> pd.DataFrame:
    """g_assignee_disambiguated.tsv 에서 80s cohort 의 primary assignee_id 추출.
    기존 processed/assignee.parquet 는 disambiguated id 를 갖고 있지 않으므로
    원본 TSV 를 한 번 스캔한다.
    """
    print("  loading 80s cohort ids ...")
    pat = pd.read_parquet(PATENTS_80S_FILE, columns=["patent_id"])
    cohort = set(pat["patent_id"].astype(str))

    print("  scanning g_assignee_disambiguated.tsv (subset) ...")
    parts = []
    for chunk in pd.read_csv(ASSIGNEE_FILE, sep="\t", chunksize=300_000,
                              usecols=["patent_id", "assignee_sequence", "assignee_id"],
                              dtype=str, low_memory=False):
        m = chunk["patent_id"].isin(cohort)
        if m.any():
            parts.append(chunk[m])
    df = pd.concat(parts, ignore_index=True)
    df["assignee_sequence"] = pd.to_numeric(df["assignee_sequence"], errors="coerce")
    df = df.sort_values(["patent_id", "assignee_sequence"]).drop_duplicates("patent_id")
    return df[["patent_id", "assignee_id"]]


def load_cited_info() -> pd.DataFrame:
    pat = pd.read_parquet(PATENTS_80S_FILE, columns=["patent_id", "grant_date"])
    pat["grant_year"] = pat["grant_date"].dt.year

    ipc = pd.read_parquet(IPC_FILE_OUT, columns=["patent_id", "ipc_subclass"])
    asg = load_cited_assignee_id_from_raw()

    info = pat.merge(ipc, on="patent_id", how="left").merge(asg, on="patent_id", how="left")
    info["ipc_subclass"] = info["ipc_subclass"].fillna("UNK")
    info["assignee_id"]  = info["assignee_id"].fillna("UNK")
    return info[["patent_id", "grant_year", "ipc_subclass", "assignee_id"]]


def load_citation_pivot() -> pd.DataFrame:
    pat = pd.read_parquet(PATENTS_80S_FILE, columns=["patent_id", "grant_date"])
    pat["grant_year"] = pat["grant_date"].dt.year
    gy = dict(zip(pat["patent_id"], pat["grant_year"]))

    annual = pd.read_parquet(CITATION_ANNUAL_FILE)
    annual = annual[annual["patent_id"].isin(gy)].copy()
    annual["age"] = annual["cite_year"] - annual["patent_id"].map(gy)
    annual = annual[(annual["age"] >= 0) & (annual["age"] <= 20)]
    pivot = (annual.groupby(["patent_id", "age"])["fwd_count"].sum()
                    .unstack(fill_value=0))
    for a in range(21):
        if a not in pivot.columns:
            pivot[a] = 0
    pivot = pivot[sorted(pivot.columns)]
    return pivot


def main():
    t_start = time.time()

    print("Loading cited_info ...")
    cited_info = load_cited_info()
    print(f"  {len(cited_info):,} patents")

    print("Loading citation pivot ...")
    pivot = load_citation_pivot()
    print(f"  pivot shape: {pivot.shape}")

    print("Loading citer metadata (step 7 output) ...")
    if CITER_METADATA_FILE.exists():
        citer_meta = pd.read_parquet(CITER_METADATA_FILE)
        print(f"  {len(citer_meta):,} rows")
    else:
        print(f"  [WARN] {CITER_METADATA_FILE.name} missing — F01-F10 skipped.")
        citer_meta = None

    for cutoff in DECISION_POINTS:
        print(f"\n{'='*50}")
        print(f"  cutoff = {cutoff}yr")
        print(f"{'='*50}")

        # F17-F21
        shape = compute_shape_features(pivot, cutoff)

        # F22: 누적 인용까지 cutoff age
        max_age = int(np.floor(cutoff))
        cum = pd.DataFrame({
            "patent_id":     pivot.index,
            "cum_to_cutoff": pivot[[c for c in pivot.columns if c <= max_age]].sum(axis=1).values,
        })
        ipc_pct = compute_ipc_percentile(cited_info, cum)

        # F01-F10
        if citer_meta is not None:
            citer_feats = compute_citer_features(citer_meta, cited_info, cutoff)
        else:
            citer_feats = pd.DataFrame({"cited_patent_id": []})

        # merge
        out = cited_info[["patent_id"]].copy()
        out = out.merge(shape, on="patent_id", how="left")
        out = out.merge(ipc_pct, on="patent_id", how="left")
        if len(citer_feats) > 0:
            out = out.merge(
                citer_feats.rename(columns={"cited_patent_id": "patent_id"}),
                on="patent_id", how="left",
            )
        # fillna for robustness
        for c in out.columns:
            if c != "patent_id":
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

        # prefix cutoff tag
        tag  = f"t{str(cutoff).replace('.', '')}"
        feat_cols = [c for c in out.columns if c != "patent_id"]
        out = out.rename(columns={c: f"{tag}__dyn_{c}" for c in feat_cols})

        out_file = FEATURES_DIR / f"dynamic_{tag}.parquet"
        out.to_parquet(out_file, index=False)
        print(f"  Saved: {out_file.name}  "
              f"({len(out):,} rows, {len(out.columns)-1} features)")

    print(f"\nStep 8 done. ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
