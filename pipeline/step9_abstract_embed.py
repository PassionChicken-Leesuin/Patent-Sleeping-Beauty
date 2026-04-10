"""
Step 9: Abstract Embedding (sentence-transformers)

g_patent_abstract.tsv 에서 80s cohort 의 abstract 를 추출하고
sentence-transformers/all-MiniLM-L6-v2 (384d) 로 임베딩한다.
이후 PCA 50d 로 축소해 features/abstract_embed.parquet 에 저장.

CPU 환경 기준으로 80s cohort (~516K) 처리에 1~2 시간 예상.

Output:
    features/abstract_embed.parquet
        columns: patent_id, emb_0 .. emb_49 (PCA-50)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import (
    PATENT_FILE, FEATURES_DIR, PROCESSED_DIR, BULK_DIR,
    TRAIN_YEARS, PATENTS_80S_FILE,
)

ABSTRACT_FILE = BULK_DIR / "g_patent_abstract.tsv"
ABSTRACT_RAW_FILE  = PROCESSED_DIR / "abstracts_80s.parquet"
ABSTRACT_EMB_FILE  = FEATURES_DIR / "abstract_embed.parquet"

MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
PCA_DIM       = 50
BATCH_SIZE    = 256


def extract_abstracts():
    """80s cohort 의 abstract 를 추출해 parquet 으로 캐시."""
    if ABSTRACT_RAW_FILE.exists():
        print(f"  cache hit: {ABSTRACT_RAW_FILE.name}")
        return pd.read_parquet(ABSTRACT_RAW_FILE)

    print(f"  scanning {ABSTRACT_FILE.name} ...")
    pat = pd.read_parquet(PATENTS_80S_FILE, columns=["patent_id"])
    cohort = set(pat["patent_id"].astype(str))

    parts = []
    t0 = time.time()
    total = 0
    for chunk in pd.read_csv(ABSTRACT_FILE, sep="\t", chunksize=200_000,
                              dtype=str, low_memory=False):
        m = chunk["patent_id"].isin(cohort)
        if m.any():
            parts.append(chunk[m][["patent_id", "patent_abstract"]])
        total += len(chunk)
        print(f"    scanned {total:,}", end="\r")
    print()
    df = pd.concat(parts, ignore_index=True).drop_duplicates("patent_id")
    df["patent_abstract"] = df["patent_abstract"].fillna("").astype(str)
    df.to_parquet(ABSTRACT_RAW_FILE, index=False)
    print(f"  cached {len(df):,} abstracts  ({time.time()-t0:.1f}s)")
    return df


def encode_abstracts(df: pd.DataFrame) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"  loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    texts = df["patent_abstract"].tolist()
    print(f"  encoding {len(texts):,} abstracts (batch_size={BATCH_SIZE}) ...")
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  encoded shape {emb.shape}  ({time.time()-t0:.1f}s)")
    return emb


def fit_pca(emb: np.ndarray, train_mask: np.ndarray) -> tuple:
    """train fold 만으로 PCA fit."""
    print(f"  fitting PCA-{PCA_DIM} on train fold ({train_mask.sum():,} rows) ...")
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(emb[train_mask])
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA explained var: {var_explained:.4f}")
    reduced = pca.transform(emb)
    return reduced, pca


def main():
    t_start = time.time()

    print("Loading abstracts ...")
    abs_df = extract_abstracts()

    # 80s cohort 와 join 해서 grant_year 매칭 (train fold 표시용)
    pat = pd.read_parquet(PATENTS_80S_FILE, columns=["patent_id", "grant_date"])
    pat["grant_year"] = pat["grant_date"].dt.year
    abs_df = abs_df.merge(pat, on="patent_id", how="inner")
    abs_df = abs_df.sort_values("patent_id").reset_index(drop=True)
    print(f"  joined: {len(abs_df):,} patents")

    print("\nEncoding abstracts ...")
    emb = encode_abstracts(abs_df)

    train_mask = abs_df["grant_year"].isin(TRAIN_YEARS).values
    reduced, pca = fit_pca(emb, train_mask)

    out = pd.DataFrame(reduced, columns=[f"emb_{i}" for i in range(PCA_DIM)])
    out.insert(0, "patent_id", abs_df["patent_id"].values)

    out.to_parquet(ABSTRACT_EMB_FILE, index=False)
    print(f"\nSaved: {ABSTRACT_EMB_FILE.name}  ({len(out):,} rows, {PCA_DIM} dim)")
    print(f"Step 9 done. ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
