"""
Step 10: Embedding-derived dynamic features (F24-F30)

step9 의 abstract_embed.parquet (PCA-50d) 를 사용해 다음 피처 생성.
모든 통계는 TRAIN fold 만으로 fit 한 뒤 전 cohort 에 apply.

  F24 abs_ipc_centroid_dist     자신 IPC subclass train centroid 과의 cosine 거리
  F25 abs_cross_ipc_nn_dist     다른 IPC subclass centroid 중 최근접 거리
  F26 abs_centroid_margin       (cross_ipc_nn) - (own_ipc_centroid) — 음수면 자기 분야와 가깝
  F27 abs_density_in_ipc        같은 IPC 내 k=20 nearest neighbor 평균 거리 (낮을수록 dense)
  F28 abs_outlier_score         전체 cohort 기준 train mean centroid 과의 거리
  F29 abs_kmeans_topic_id       train fold 에서 fit 한 KMeans(k=50) 의 cluster id
  F30 abs_kmeans_topic_dist     해당 cluster centroid 와의 거리

이 피처들은 모두 patent 자체의 정적 텍스트로부터 나오므로 cutoff
시점에 무관하다. 하나의 features/abstract_dynamic.parquet 으로
저장하고, utils 에서 모든 cutoff 에 동일하게 join.

Output:
    features/abstract_dynamic.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from config import (
    PATENTS_80S_FILE, IPC_FILE_OUT, FEATURES_DIR,
    TRAIN_YEARS,
)

ABSTRACT_EMB_FILE = FEATURES_DIR / "abstract_embed.parquet"
OUT_FILE          = FEATURES_DIR / "abstract_dynamic.parquet"

KMEANS_K = 50
KNN_K    = 20


def cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """row-wise cosine distance between matched rows."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - (a_n * b_n).sum(axis=1)


def main():
    t_start = time.time()

    print("Loading embeddings ...")
    emb_df = pd.read_parquet(ABSTRACT_EMB_FILE)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    print(f"  {len(emb_df):,} rows, {len(emb_cols)} dims")

    print("Loading patent metadata ...")
    pat = pd.read_parquet(PATENTS_80S_FILE, columns=["patent_id", "grant_date"])
    pat["grant_year"] = pat["grant_date"].dt.year
    ipc = pd.read_parquet(IPC_FILE_OUT, columns=["patent_id", "ipc_subclass"])

    df = emb_df.merge(pat, on="patent_id", how="left")
    df = df.merge(ipc, on="patent_id", how="left")
    df["ipc_subclass"] = df["ipc_subclass"].fillna("UNK")
    df = df.dropna(subset=["grant_year"]).reset_index(drop=True)
    df["grant_year"] = df["grant_year"].astype(int)

    X = df[emb_cols].values.astype(np.float32)
    train_mask = df["grant_year"].isin(TRAIN_YEARS).values
    print(f"  full {len(df):,}  train {train_mask.sum():,}")

    # ── F24, F25, F26: IPC centroid distances ─────────
    print("\nComputing IPC centroids on train fold ...")
    t0 = time.time()
    ipc_train = df.loc[train_mask, ["ipc_subclass"]].copy()
    ipc_train_X = X[train_mask]
    centroids = {}
    for ipc_name in ipc_train["ipc_subclass"].unique():
        m = (ipc_train["ipc_subclass"] == ipc_name).values
        if m.sum() >= 5:
            centroids[ipc_name] = ipc_train_X[m].mean(axis=0)
    # global mean for unknown IPCs
    global_centroid = ipc_train_X.mean(axis=0)
    print(f"  {len(centroids):,} IPC centroids ({time.time()-t0:.1f}s)")

    # vectorize centroid lookup
    ipc_list   = list(centroids.keys())
    ipc_idx    = {n: i for i, n in enumerate(ipc_list)}
    cent_mat   = np.stack([centroids[n] for n in ipc_list], axis=0)         # (Cipc, D)
    cent_mat_n = cent_mat / (np.linalg.norm(cent_mat, axis=1, keepdims=True) + 1e-12)

    # own_ipc centroid index (-1 if missing)
    own_idx = df["ipc_subclass"].map(ipc_idx).fillna(-1).astype(int).values
    own_centroid = np.where(
        own_idx[:, None] >= 0,
        cent_mat[np.clip(own_idx, 0, len(ipc_list)-1)],
        global_centroid[None, :],
    )
    f24 = cosine_dist(X, own_centroid)

    # cross_ipc_nn: cosine distance to all OTHER centroids, take min
    print("Computing cross-IPC nearest centroid (vectorized) ...")
    t0 = time.time()
    X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sims = X_n @ cent_mat_n.T              # (N, Cipc)
    # mask own centroid
    valid = own_idx >= 0
    sims_masked = sims.copy()
    sims_masked[np.arange(len(sims))[valid], own_idx[valid]] = -np.inf
    nn_max_sim = sims_masked.max(axis=1)
    f25 = 1.0 - nn_max_sim
    print(f"  done ({time.time()-t0:.1f}s)")

    f26 = f25 - f24

    # F28 global outlier score
    f28 = cosine_dist(X, np.tile(global_centroid, (len(X), 1)))

    # ── F27: in-IPC density (mean distance to k=20 NN within same IPC) ─
    print("Computing in-IPC density ...")
    t0 = time.time()
    f27 = np.full(len(X), 0.5, dtype=float)
    for ipc_name, group_idx in df.groupby("ipc_subclass").groups.items():
        idx = np.array(list(group_idx))
        if len(idx) < KNN_K + 1:
            continue
        # NN fit on TRAIN-only (avoid test→test leakage)
        train_idx_in = idx[df.iloc[idx]["grant_year"].isin(TRAIN_YEARS).values]
        if len(train_idx_in) < KNN_K + 1:
            continue
        nbrs = NearestNeighbors(n_neighbors=KNN_K, metric="cosine")
        nbrs.fit(X[train_idx_in])
        dist, _ = nbrs.kneighbors(X[idx])
        f27[idx] = dist.mean(axis=1)
    print(f"  done ({time.time()-t0:.1f}s)")

    # ── F29, F30: KMeans cluster ──────────────────────
    print(f"Fitting KMeans (k={KMEANS_K}) on train fold ...")
    t0 = time.time()
    km = KMeans(n_clusters=KMEANS_K, random_state=42, n_init=10)
    km.fit(X[train_mask])
    f29 = km.predict(X)
    cent_k_mat = km.cluster_centers_
    f30 = cosine_dist(X, cent_k_mat[f29])
    print(f"  done ({time.time()-t0:.1f}s)")

    # ── 결합 ────────────────────────────────────────
    out = pd.DataFrame({
        "patent_id":              df["patent_id"].values,
        "abs_ipc_centroid_dist":  f24,
        "abs_cross_ipc_nn_dist":  f25,
        "abs_centroid_margin":    f26,
        "abs_density_in_ipc":     f27,
        "abs_outlier_score":      f28,
        "abs_kmeans_topic_id":    f29,
        "abs_kmeans_topic_dist":  f30,
    })
    for c in out.columns:
        if c == "patent_id":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out.to_parquet(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE.name}  ({len(out):,} rows)")
    print(f"Step 10 done. ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
