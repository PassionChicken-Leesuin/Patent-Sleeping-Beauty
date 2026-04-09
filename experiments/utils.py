"""
experiments/utils.py
모든 실험 스크립트가 공유하는 유틸리티 함수 모음.
중복 정의 없이 이 파일 하나에서 import.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, mean_squared_error

from config import (
    FEATURES_35_FILE, FEATURES_75_FILE, FEATURES_115_FILE,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    STATIC_COLS, XGB_PARAMS, PSB_WEIGHT,
)


# ── Feature 컬럼 헬퍼 ────────────────────────────────
def get_cite_cols(cutoff: float) -> list:
    """시점별 citation feature 컬럼명 리스트"""
    prefix = f"t{str(cutoff).replace('.', '')}"
    return [
        f"{prefix}__cum_citations",
        f"{prefix}__cite_last1yr",
        f"{prefix}__cite_last3yr",
        f"{prefix}__cite_growth_rate",
        f"{prefix}__cite_growth_last3",
        f"{prefix}__cite_acceleration",
        f"{prefix}__cite_peak_ratio",
        f"{prefix}__cite_active_years",
        f"{prefix}__zero_citation",
    ]


def get_feature_cols(cutoff: float) -> list:
    return STATIC_COLS + get_cite_cols(cutoff)


# ── IPC Frequency Encoding ───────────────────────────
def label_encode_ipc(df: pd.DataFrame) -> pd.DataFrame:
    """IPC 컬럼을 빈도 기반 정수 인코딩 (train 분포 기준)"""
    df = df.copy()
    for col, enc_col in [
        ("ipc_section",    "ipc_section_enc"),
        ("ipc_class_full", "ipc_class_enc"),
        ("ipc_subclass",   "ipc_subclass_enc"),
    ]:
        if col in df.columns:
            freq = df[col].value_counts()
            df[enc_col] = df[col].map(freq).fillna(0).astype(int)
        else:
            df[enc_col] = 0
    return df


# ── 데이터 로드 & 병합 ───────────────────────────────
def load_and_merge(cutoff: float, labels: pd.DataFrame) -> pd.DataFrame:
    """피처 파일 + 라벨 병합 후 IPC 인코딩 및 결측치 처리"""
    feat_map = {3.5: FEATURES_35_FILE, 7.5: FEATURES_75_FILE, 11.5: FEATURES_115_FILE}
    feat = pd.read_parquet(feat_map[cutoff])
    df = feat.merge(
        labels[["patent_id", "B", "t_m", "t_a", "psb", "ipc_subclass"]],
        on="patent_id", how="inner", suffixes=("", "_lbl"),
    )
    df = label_encode_ipc(df)
    for c in get_feature_cols(cutoff):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df[c] = df[c].replace([np.inf, -np.inf], 0)
    return df


# ── Train/Val/Test Split ─────────────────────────────
def split(df: pd.DataFrame):
    """grant_year 기준 시계열 분할 (전 실험 고정)"""
    train = df[df["grant_year"].isin(TRAIN_YEARS)].copy()
    val   = df[df["grant_year"].isin(VAL_YEARS)].copy()
    test  = df[df["grant_year"].isin(TEST_YEARS)].copy()
    return train, val, test


def print_split_stats(name: str, train, val, test):
    print(f"  [{name}]  train={len(train):,}(PSB={train['psb'].sum()})  "
          f"val={len(val):,}(PSB={val['psb'].sum()})  "
          f"test={len(test):,}(PSB={test['psb'].sum()})")


# ── 샘플 가중치 ──────────────────────────────────────
def make_sample_weights(psb: np.ndarray, psb_weight: float = PSB_WEIGHT) -> np.ndarray:
    """PSB=1 샘플에 높은 가중치 부여"""
    w = np.ones(len(psb), dtype=float)
    w[psb == 1] = psb_weight
    return w


# ── XGBoost Q 회귀 학습 ──────────────────────────────
def train_qmodel(X_tr, y_tr, X_val, y_val, tag: str,
                 psb_tr=None, psb_weight: float = PSB_WEIGHT):
    """Fitted Q-Iteration용 XGBoost 회귀 모델 학습"""
    model = xgb.XGBRegressor(**XGB_PARAMS)
    sw = make_sample_weights(psb_tr, psb_weight) if psb_tr is not None else None
    model.fit(
        X_tr, y_tr,
        sample_weight=sw,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    rmse = mean_squared_error(y_val, model.predict(X_val)) ** 0.5
    print(f"  [{tag}] val RMSE: {rmse:.4f}")
    return model


# ── XGBoost 분류기 학습 ──────────────────────────────
def train_xgb_classifier(X_tr, y_tr, X_val, y_val,
                          psb_tr=None, psb_weight: float = PSB_WEIGHT):
    model = xgb.XGBClassifier(**XGB_PARAMS)
    sw = make_sample_weights(psb_tr, psb_weight) if psb_tr is not None else None
    model.fit(X_tr, y_tr, sample_weight=sw,
              eval_set=[(X_val, y_val)], verbose=False)
    ap = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"  [XGB Classifier] val AP: {ap:.4f}")
    return model


# ── Random Forest 분류기 학습 ────────────────────────
def train_rf_classifier(X_tr, y_tr, X_val, y_val,
                         psb_tr=None, psb_weight: float = PSB_WEIGHT):
    model = RandomForestClassifier(
        n_estimators=500, max_depth=10, random_state=42, n_jobs=-1
    )
    sw = make_sample_weights(psb_tr, psb_weight) if psb_tr is not None else None
    model.fit(X_tr, y_tr, sample_weight=sw)
    ap = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"  [RF Classifier]  val AP: {ap:.4f}")
    return model


# ── 평가: Precision@k + AP ───────────────────────────
def evaluate_ranking(y_true: pd.Series, scores: pd.Series,
                     label: str = "", k_list: list = None) -> dict:
    """랭킹 기반 평가 (Precision@k, Average Precision)"""
    if k_list is None:
        k_list = [100, 200, 500, 1000]

    combined = pd.DataFrame({"y": y_true, "score": scores}).dropna()
    combined = combined.sort_values("score", ascending=False)

    # 스코어가 단일값(모두 0 등)이면 AP 계산 불가 → 0 처리
    if combined["score"].nunique() < 2:
        ap = 0.0
    else:
        ap = average_precision_score(combined["y"], combined["score"])
    results = {"average_precision": ap}

    lines = [f"  [{label}]  AP={ap:.4f}"]
    for k in k_list:
        k_ = min(k, len(combined))
        p  = combined.head(k_)["y"].sum() / k_
        results[f"precision@{k}"] = p
        lines.append(f"    P@{k}={p:.4f} ({int(combined.head(k_)['y'].sum())} PSB)")

    print("\n".join(lines))
    return results


# ── 모델 저장/로드 헬퍼 ──────────────────────────────
def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(path, "wb"))
    print(f"  Saved: {path.name}")


def load_model(path: Path):
    return pickle.load(open(path, "rb"))
