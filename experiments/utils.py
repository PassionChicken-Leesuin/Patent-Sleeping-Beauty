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
    MAINT_FEATURES_FILE,
    DYN_FEATURES_35_FILE, DYN_FEATURES_75_FILE, DYN_FEATURES_115_FILE,
    ABSTRACT_DYN_FILE,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    STATIC_COLS, MAINT_COLS_BY_CUTOFF, XGB_PARAMS, PSB_WEIGHT,
    get_dynamic_cols, dynamic_features_file, get_abstract_cols,
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


def get_maint_cols(cutoff: float) -> list:
    """의사결정 시점 t 에서 관측 가능한 납부 이력 컬럼."""
    return MAINT_COLS_BY_CUTOFF.get(cutoff, [])


def get_feature_cols(cutoff: float) -> list:
    return (STATIC_COLS
            + get_cite_cols(cutoff)
            + get_maint_cols(cutoff)
            + get_dynamic_cols(cutoff)
            + get_abstract_cols())


# ── IPC Frequency Encoding ───────────────────────────
_IPC_COLS = [
    ("ipc_section",    "ipc_section_enc"),
    ("ipc_class_full", "ipc_class_enc"),
    ("ipc_subclass",   "ipc_subclass_enc"),
]


def fit_ipc_freq(df: pd.DataFrame) -> dict:
    """train fold 에서만 호출. 각 IPC 컬럼의 value_counts 를 반환."""
    freqs = {}
    for col, _ in _IPC_COLS:
        if col in df.columns:
            freqs[col] = df[col].value_counts()
        else:
            freqs[col] = pd.Series(dtype=int)
    return freqs


def apply_ipc_freq(df: pd.DataFrame, freqs: dict) -> pd.DataFrame:
    """fit_ipc_freq 로 얻은 train 분포를 val/test 에 동일하게 적용.
    train 에 존재하지 않던 카테고리는 0 으로 처리된다 (unseen=0)."""
    df = df.copy()
    for col, enc_col in _IPC_COLS:
        if col in df.columns and col in freqs:
            df[enc_col] = df[col].map(freqs[col]).fillna(0).astype(int)
        else:
            df[enc_col] = 0
    return df


def label_encode_ipc(df: pd.DataFrame) -> pd.DataFrame:
    """[DEPRECATED] 전체 df 분포로 인코딩 — leakage.
    fit_ipc_freq / apply_ipc_freq 를 쓰고 그 결과를 split 이후 각
    fold 에 apply 해야 한다. 하위 호환 목적으로만 남겨둠."""
    return apply_ipc_freq(df, fit_ipc_freq(df))


# ── 데이터 로드 & 병합 ───────────────────────────────
def load_and_merge(
    cutoff: float,
    labels: pd.DataFrame,
    survival_filter: bool = False,
) -> pd.DataFrame:
    """피처 파일 + 라벨 + 납부 이력 병합 후 결측치 처리.

    ⚠ IPC frequency encoding 은 여기서 수행하지 않는다.
       split() 으로 train 을 분리한 뒤 encode_ipc_splits() 로
       train-fit / val·test-transform 해야 leakage 가 없다.

    survival_filter=True 이면 해당 cutoff 에 살아있었던 특허만 남긴다
    (실제 owner 의 납부 이력 기반):
        7.5yr  → paid_3_5 == 1
        11.5yr → paid_3_5 == 1 AND paid_7_5 == 1
    3.5yr 은 필터 없음.
    """
    feat_map = {3.5: FEATURES_35_FILE, 7.5: FEATURES_75_FILE, 11.5: FEATURES_115_FILE}
    feat = pd.read_parquet(feat_map[cutoff])

    df = feat.merge(
        labels[["patent_id", "B", "t_m", "t_a", "psb", "ipc_subclass"]],
        on="patent_id", how="inner", suffixes=("", "_lbl"),
    )

    # 납부 이력 병합 (maint_features.parquet 이 있을 때만)
    if MAINT_FEATURES_FILE.exists():
        mf = pd.read_parquet(MAINT_FEATURES_FILE)
        df = df.merge(mf, on="patent_id", how="left")
        for c in ["paid_3_5", "paid_7_5", "paid_11_5"]:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype(int)

    # 동적 피처 병합 (step8 산출물)
    dyn_path = dynamic_features_file(cutoff)
    if dyn_path.exists():
        dyn = pd.read_parquet(dyn_path)
        df = df.merge(dyn, on="patent_id", how="left")

    # abstract embedding 기반 정적 피처 (step10 산출물)
    if ABSTRACT_DYN_FILE.exists():
        abs_dyn = pd.read_parquet(ABSTRACT_DYN_FILE)
        df = df.merge(abs_dyn, on="patent_id", how="left")

    # IPC encoded 컬럼을 미리 0 으로 채워둠 (split 이전 단계)
    for _, enc_col in _IPC_COLS:
        df[enc_col] = 0

    for c in get_feature_cols(cutoff):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df[c] = df[c].replace([np.inf, -np.inf], 0)

    if survival_filter:
        if cutoff == 7.5 and "paid_3_5" in df.columns:
            df = df[df["paid_3_5"] == 1].copy()
        elif cutoff == 11.5 and "paid_3_5" in df.columns and "paid_7_5" in df.columns:
            df = df[(df["paid_3_5"] == 1) & (df["paid_7_5"] == 1)].copy()
    return df


def encode_ipc_splits(train: pd.DataFrame,
                      val:   pd.DataFrame,
                      test:  pd.DataFrame):
    """train 분포로 IPC freq encoding 을 fit 하고 val/test 에 apply.

    기존 load_and_merge 가 전체 df 분포로 인코딩하던 leakage 를
    차단한다. 호출자는 split 직후 이 함수를 호출하면 된다.

    Returns: (train, val, test) — 모두 인코딩 적용 후의 사본.
    """
    freqs = fit_ipc_freq(train)
    return (
        apply_ipc_freq(train, freqs),
        apply_ipc_freq(val,   freqs),
        apply_ipc_freq(test,  freqs),
    )


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
