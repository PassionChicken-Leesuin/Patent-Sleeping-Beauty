"""
A_integrated: PSB x PSB_Refinement 통합 Fitted-Q
================================================
Phase 3 #1 — 두 트랙의 레버를 동시에 당긴다.

PSB_Refinement A1 winner 설정 (yearly panel + BI cascade) 를
base 로 하고, 여기에 PSB Phase 1/2 의 feature 를 추가로 투입:

  (a) PSB Phase 1 dynamic citer-quality + shape features (17개 x cutoff)
      features/dynamic_t35.parquet, dynamic_t75.parquet, dynamic_t115.parquet

  (b) PSB Phase 2 abstract embedding derived features (7개, cutoff 무관)
      features/abstract_dynamic.parquet

최종 피처 구성 (각 decision age 별):

    STATIC    : PSB STATIC_COLS + maint features (paid_3_5, paid_7_5)
    DYNAMIC   : yearly panel [last pool] of DYNAMIC_COLS (11개 x 1 시점)
    CITER     : dynamic_tXX.parquet (17개, PSB cutoff 기준)
    EMBED     : abstract_dynamic.parquet (7개, cutoff 무관)

A1 winner 와 동일한 learning schedule:
    pool       = "last"
    psb_weight = 50
    no_clip    = True

사용법:
    python train_bi_integrated.py --thr 0.001
    python train_bi_integrated.py --thr all
    python train_bi_integrated.py --thr all --variant integrated_v1
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, mean_squared_error

from config import (
    panel_loader as pl,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    DECISION_AGES, MAINT_COST, GAMMA, PSB_REWARD, PSB_WEIGHT,
    STATIC_COLS, DYNAMIC_COLS, XGB_PARAMS, PSB_THRESHOLDS,
    MAINT_COLS_BY_AGE, a_models_dir, a_results_dir, DECISION_LABELS,
    FEATURES_DIR,
)


# ── PSB Phase 1/2 feature 경로 ────────────────────
DYN_FILES = {
    3:  FEATURES_DIR / "dynamic_t35.parquet",    # PSB cutoff 3.5yr
    7:  FEATURES_DIR / "dynamic_t75.parquet",    # PSB cutoff 7.5yr
    11: FEATURES_DIR / "dynamic_t115.parquet",   # PSB cutoff 11.5yr
}
ABSTRACT_DYN_FILE = FEATURES_DIR / "abstract_dynamic.parquet"


# ════════════════════════════════════════════════════
#  유틸
# ════════════════════════════════════════════════════
def make_sample_weights(psb: np.ndarray, w: float) -> np.ndarray:
    out = np.ones(len(psb), dtype=float)
    out[psb == 1] = w
    return out


def train_qmodel(X_tr, y_tr, X_val, y_val, tag: str, psb_tr, psb_weight: float):
    model = xgb.XGBRegressor(**XGB_PARAMS)
    sw = make_sample_weights(psb_tr, psb_weight)
    model.fit(
        X_tr, y_tr,
        sample_weight=sw,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    rmse = mean_squared_error(y_val, model.predict(X_val)) ** 0.5
    print(f"  [{tag}] val RMSE: {rmse:.4f}  n_features={X_tr.shape[1]}")
    return model


def pooled_matrix_last(dyn, order, up_to_age):
    """A1 winner 의 pool='last' — age=up_to_age 시점의 11 cols 만."""
    X3 = pl.dynamic_tensor(dyn, order, up_to_age=up_to_age)    # (N,T,F)
    return X3[:, -1, :]


def build_integrated_feature_matrix(
    df_split: pd.DataFrame,
    dyn: pd.DataFrame,
    age: int,
    psb_dyn_df: pd.DataFrame | None,
    abs_dyn_df: pd.DataFrame | None,
):
    """A1 (yearly panel last pool + static + maint) + PSB Phase 1/2.

    - df_split : static_lbl subset (train/val/test)
    - dyn      : yearly_panel_dynamic long-form
    - age      : decision age (3/7/11)
    - psb_dyn_df  : PSB features/dynamic_t{35,75,115}.parquet (age 별로 다름)
    - abs_dyn_df  : PSB features/abstract_dynamic.parquet (cutoff 무관)
    """
    order = df_split["patent_id"].values

    # 1) STATIC (PSB STATIC_COLS 기준, 누락은 0)
    Xs, static_names = pl.static_feature_matrix(df_split)

    # 2) YEARLY PANEL last pool (11 cols)
    Xd = pooled_matrix_last(dyn, order, up_to_age=age)
    dyn_names = [f"{c}__last" for c in DYNAMIC_COLS]

    parts = [Xs, Xd]
    names = list(static_names) + list(dyn_names)

    # 3) MAINT (paid_3_5, paid_7_5 at age>=7)
    for col in MAINT_COLS_BY_AGE.get(age, []):
        if col in df_split.columns:
            v = df_split[col].fillna(0).astype(np.float32).to_numpy().reshape(-1, 1)
        else:
            v = np.zeros((len(df_split), 1), dtype=np.float32)
        parts.append(v)
        names.append(col)

    # 4) PSB Phase 1: citer-quality + shape features (17개)
    if psb_dyn_df is not None:
        # psb_dyn_df 는 prefix tXX__dyn_ 로 된 컬럼을 가짐. patent_id 로 lookup.
        feat_cols = [c for c in psb_dyn_df.columns if c != "patent_id"]
        lookup = psb_dyn_df.set_index("patent_id")[feat_cols]
        sub = lookup.reindex(order).fillna(0.0).to_numpy(np.float32, copy=True)
        sub[~np.isfinite(sub)] = 0.0
        parts.append(sub)
        names.extend(feat_cols)

    # 5) PSB Phase 2: abstract embedding derived (7개, cutoff 무관)
    if abs_dyn_df is not None:
        feat_cols = [c for c in abs_dyn_df.columns if c != "patent_id"]
        lookup = abs_dyn_df.set_index("patent_id")[feat_cols]
        sub = lookup.reindex(order).fillna(0.0).to_numpy(np.float32, copy=True)
        sub[~np.isfinite(sub)] = 0.0
        parts.append(sub)
        names.extend(feat_cols)

    X = np.concatenate(parts, axis=1).astype(np.float32)
    return X, names


def evaluate_ranking(y_true, scores, label: str):
    combined = pd.DataFrame({"y": y_true, "score": scores}).dropna()
    combined = combined.sort_values("score", ascending=False)
    if combined["score"].nunique() < 2:
        ap = 0.0
    else:
        ap = average_precision_score(combined["y"], combined["score"])
    print(f"  [{label}] AP={ap:.4f}")
    for k in [100, 200, 500, 1000]:
        k_ = min(k, len(combined))
        hit = int(combined.head(k_)["y"].sum())
        p = hit / k_
        print(f"    P@{k}={p:.4f}  ({hit} PSB)")
    return ap


# ════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════
def run(thr: float,
        psb_weight: float = 50.0,
        no_clip: bool = True,
        variant: str = "integrated_v1"):
    tag = f"thr={thr}  w={psb_weight}  no_clip={no_clip}  v={variant}"
    print(f"\n{'='*72}\n  A_integrated [{tag}]\n{'='*72}")
    t0 = time.time()

    # 1) yearly panel + labels
    static_lbl, dyn = pl.load_panel_with_labels(thr)
    print(f"Loaded panel: static_lbl={static_lbl.shape}  dyn={dyn.shape}  "
          f"PSB={int(static_lbl['psb'].sum())}")

    tr, va, te = pl.split_static(static_lbl)
    tr, va, te = pl.encode_ipc_static(tr, va, te)
    print(f"Split: train={len(tr):,}  val={len(va):,}  test={len(te):,}  "
          f"(PSB test={int(te['psb'].sum())})")

    # 2) PSB Phase 1/2 features 로드
    psb_dyn = {}
    for age, fp in DYN_FILES.items():
        if fp.exists():
            psb_dyn[age] = pd.read_parquet(fp)
            print(f"  PSB dyn (age={age}): {fp.name} -> {psb_dyn[age].shape}")
        else:
            psb_dyn[age] = None
            print(f"  [WARN] missing {fp}")

    if ABSTRACT_DYN_FILE.exists():
        abs_dyn = pd.read_parquet(ABSTRACT_DYN_FILE)
        print(f"  PSB embedding derived: {ABSTRACT_DYN_FILE.name} -> {abs_dyn.shape}")
    else:
        abs_dyn = None
        print(f"  [WARN] missing {ABSTRACT_DYN_FILE}")

    mdir = a_models_dir(thr)
    rdir = a_results_dir(thr)

    def backup(q_next):
        return (GAMMA * q_next) if no_clip else (GAMMA * np.maximum(q_next, 0.0))

    # ── Step A: Q_11 (terminal) ───────────────────
    print("\n-- Step A: Q_11 (age=11, terminal) --")
    age = 11
    cost = MAINT_COST[age]
    Xtr, names = build_integrated_feature_matrix(tr, dyn, age, psb_dyn.get(age), abs_dyn)
    Xva, _     = build_integrated_feature_matrix(va, dyn, age, psb_dyn.get(age), abs_dyn)
    Xte, _     = build_integrated_feature_matrix(te, dyn, age, psb_dyn.get(age), abs_dyn)
    ytr = -cost + tr["psb"].values * PSB_REWARD
    yva = -cost + va["psb"].values * PSB_REWARD
    print(f"  features total: {len(names)}")
    m11 = train_qmodel(Xtr, ytr, Xva, yva, tag="Q_11",
                        psb_tr=tr["psb"].values, psb_weight=psb_weight)
    pickle.dump(m11, open(mdir / f"q_model_11_{variant}.pkl", "wb"))

    q11_te = m11.predict(Xte)
    q11_map = {pid: float(q) for pid, q in
               zip(tr["patent_id"].values, m11.predict(Xtr))}
    q11_map.update({pid: float(q) for pid, q in
                    zip(va["patent_id"].values, m11.predict(Xva))})
    q11_map.update({pid: float(q) for pid, q in
                    zip(te["patent_id"].values, q11_te)})

    # ── Step B: Q_7 ───────────────────────────────
    print("\n-- Step B: Q_7 (age=7) --")
    age = 7
    cost = MAINT_COST[age]
    Xtr, _ = build_integrated_feature_matrix(tr, dyn, age, psb_dyn.get(age), abs_dyn)
    Xva, _ = build_integrated_feature_matrix(va, dyn, age, psb_dyn.get(age), abs_dyn)
    Xte, _ = build_integrated_feature_matrix(te, dyn, age, psb_dyn.get(age), abs_dyn)
    q_next_tr = tr["patent_id"].map(q11_map).fillna(0).values
    q_next_va = va["patent_id"].map(q11_map).fillna(0).values
    ytr = -cost + backup(q_next_tr)
    yva = -cost + backup(q_next_va)
    m7 = train_qmodel(Xtr, ytr, Xva, yva, tag="Q_7",
                       psb_tr=tr["psb"].values, psb_weight=psb_weight)
    pickle.dump(m7, open(mdir / f"q_model_7_{variant}.pkl", "wb"))

    q7_map = {pid: float(q) for pid, q in
              zip(tr["patent_id"].values, m7.predict(Xtr))}
    q7_map.update({pid: float(q) for pid, q in
                   zip(va["patent_id"].values, m7.predict(Xva))})
    q7_map.update({pid: float(q) for pid, q in
                   zip(te["patent_id"].values, m7.predict(Xte))})

    # ── Step C: Q_3 ───────────────────────────────
    print("\n-- Step C: Q_3 (age=3) --")
    age = 3
    cost = MAINT_COST[age]
    Xtr, _ = build_integrated_feature_matrix(tr, dyn, age, psb_dyn.get(age), abs_dyn)
    Xva, _ = build_integrated_feature_matrix(va, dyn, age, psb_dyn.get(age), abs_dyn)
    Xte, _ = build_integrated_feature_matrix(te, dyn, age, psb_dyn.get(age), abs_dyn)
    q_next_tr = tr["patent_id"].map(q7_map).fillna(0).values
    q_next_va = va["patent_id"].map(q7_map).fillna(0).values
    ytr = -cost + backup(q_next_tr)
    yva = -cost + backup(q_next_va)
    m3 = train_qmodel(Xtr, ytr, Xva, yva, tag="Q_3",
                       psb_tr=tr["psb"].values, psb_weight=psb_weight)
    pickle.dump(m3, open(mdir / f"q_model_3_{variant}.pkl", "wb"))

    q3_te = m3.predict(Xte)

    # ── 결과 ──────────────────────────────────────
    results = te[["patent_id", "grant_year", "psb", "B", "t_a", "ipc_subclass"]].copy()
    results["Q_3_pred"]  = q3_te
    results["Q_7_pred"]  = te["patent_id"].map(q7_map).fillna(0).values
    results["Q_11_pred"] = te["patent_id"].map(q11_map).fillna(0).values

    for age_k, col in [(3, "Q_3_pred"), (7, "Q_7_pred"), (11, "Q_11_pred")]:
        results[f"policy_{age_k}_pos"] = (results[col] > 0).astype(int)
    top1000_mask = results["Q_3_pred"].rank(ascending=False) <= 1000
    results["policy_top1000"] = top1000_mask.astype(int)

    out_path = rdir / f"test_qvalues_A1_{variant}.parquet"
    results.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path.name}  ({len(results):,} rows)")

    psb_total = int(results["psb"].sum())
    print(f"\nTest total={len(results):,}  PSB={psb_total}")
    for col, lbl in [("policy_3_pos", "3.5yr Q>0"),
                     ("policy_7_pos", "7.5yr Q>0"),
                     ("policy_11_pos","11.5yr Q>0"),
                     ("policy_top1000","top1000 Q_3")]:
        sel = int(results[col].sum())
        hit = int(results[results[col] == 1]["psb"].sum())
        prec = hit / sel if sel > 0 else 0.0
        rec  = hit / psb_total if psb_total > 0 else 0.0
        print(f"  [{lbl}] maintain={sel:,}  PSB={hit}  prec={prec:.4f}  recall={rec:.4f}")

    print("\n-- Q_3 ranking --")
    ap = evaluate_ranking(results["psb"], results["Q_3_pred"], label="Q_3")

    print(f"\n[{tag}] Done. ({time.time()-t0:.1f}s)")
    return results, ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", default="all",
                        help="PSB threshold (0.001 / 0.005 / 0.010 / all)")
    parser.add_argument("--psb_weight", type=float, default=50.0)
    parser.add_argument("--no_clip", action="store_true", default=True)
    parser.add_argument("--variant", default="integrated_v1",
                        help="저장 파일명 식별자")
    args = parser.parse_args()

    if args.thr == "all":
        thrs = PSB_THRESHOLDS
    else:
        thrs = [float(args.thr)]

    for thr in thrs:
        run(thr,
            psb_weight=args.psb_weight,
            no_clip=args.no_clip,
            variant=args.variant)


if __name__ == "__main__":
    main()
