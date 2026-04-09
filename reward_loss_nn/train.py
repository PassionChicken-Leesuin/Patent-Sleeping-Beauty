"""
Reward-Loss MLP Experiment  (with optional SMOTE)
==================================================
11.5yr 단일 시점 데이터만 사용하여 PSB 여부를 예측.

핵심 아이디어:
  s_i = MLP(x_i)  →  p_i = sigmoid(s_i)
  r_i = -cost + psb_i * R
  Loss = -mean(p_i * r_i)     ← 기대 보상 직접 최대화 (Adam minimize)

클래스 불균형 보정 전략:
  USE_SMOTE=False: balanced mini-batch (PSB=1/0 50:50 oversampling)
  USE_SMOTE=True : SMOTE로 합성 PSB=1 샘플 생성 → 정규 배치 학습
                   SMOTE_RATIO: PSB=1 비율 목표 (e.g. 0.1 = 10%)

사용법:
  python train.py              # balanced batch (기본)
  python train.py --smote      # SMOTE 사용
  python train.py --smote --ratio 0.2  # PSB=1 비율 20%로 증가

Output:
  models/mlp_reward[_smote].pkl
  results/reward_nn[_smote].parquet
  results/reward_nn_summary.csv  (두 결과 모두 포함)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

# ── 경로 설정 ──────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
FEAT_FILE  = ROOT / "features" / "features_11_5yr.parquet"
LABEL_FILE = ROOT / "features" / "labels.parquet"
MODELS_DIR = Path(__file__).parent / "models";  MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR= Path(__file__).parent / "results"; RESULTS_DIR.mkdir(exist_ok=True)

# ── 파라미터 ──────────────────────────────────────────
MAINT_COST  = 3.85
PSB_REWARD  = 8.0

TRAIN_YEARS = list(range(1982, 1987))
VAL_YEARS   = [1987]
TEST_YEARS  = list(range(1988, 1990))

HIDDEN_SIZES = [128, 64]
LR           = 1e-3
BATCH_SIZE   = 512
N_EPOCHS     = 300
PATIENCE     = 25
SEED         = 42

# ── CLI 인자 ──────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--smote",  action="store_true", help="Use SMOTE oversampling")
parser.add_argument("--ratio",  type=float, default=0.1,
                    help="SMOTE target ratio for PSB=1 (default 0.1 = 10%%)")
_args, _ = parser.parse_known_args()
USE_SMOTE   = _args.smote
SMOTE_RATIO = _args.ratio

# ── Feature 컬럼 ──────────────────────────────────────
STATIC_COLS = [
    "num_claims", "filing_to_grant_days", "small_entity",
    "num_figures", "num_sheets", "inventor_count",
    "bwd_us_total", "bwd_examiner", "bwd_applicant",
    "bwd_foreign", "bwd_total", "bwd_examiner_ratio",
    "is_organization", "is_us_company", "is_foreign", "is_individual",
    "ipc_section_enc", "ipc_class_enc", "ipc_subclass_enc",
]

def get_cite_cols(cutoff: float = 11.5) -> list:
    prefix = f"t{str(cutoff).replace('.','')}"
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

FEAT_COLS = STATIC_COLS + get_cite_cols(11.5)


# ══════════════════════════════════════════════════════
#  수치 안정 sigmoid
# ══════════════════════════════════════════════════════
def sigmoid(x: np.ndarray) -> np.ndarray:
    """수치 안정 sigmoid: overflow 방지"""
    pos = x >= 0
    out = np.empty_like(x)
    out[pos]  = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_neg   = np.exp(x[~pos])
    out[~pos] = exp_neg / (1.0 + exp_neg)
    return out


# ══════════════════════════════════════════════════════
#  MLP (numpy, forward + backprop 수동 구현)
# ══════════════════════════════════════════════════════
class MLP:
    """
    구조: input → [Linear → BatchNorm → ReLU] × L → Linear → scalar logit

    레이어:
      _a[0] = X (입력)
      _z[l] = _a[l] @ W[l] + b[l]    (pre-activation)
      _a[l+1] = ReLU(_z[l])           (hidden) or _z[l] (output)
    """

    def __init__(self, n_in: int, hidden: list, seed: int = 42):
        rng  = np.random.default_rng(seed)
        sizes = [n_in] + hidden + [1]
        self.W = []
        self.b = []
        for i in range(len(sizes) - 1):
            # He initialization
            W = rng.standard_normal((sizes[i], sizes[i+1])) * np.sqrt(2.0 / sizes[i])
            b = np.zeros(sizes[i+1])
            self.W.append(W)
            self.b.append(b)
        self.n_layers = len(self.W)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (N, F) → logit: (N,)  |  저장: _a, _z"""
        self._a = [X]
        self._z = []
        h = X
        for l in range(self.n_layers):
            z = h @ self.W[l] + self.b[l]   # (N, out)
            self._z.append(z)
            if l < self.n_layers - 1:
                h = np.maximum(0.0, z)       # ReLU
            else:
                h = z                        # 출력층: linear
            self._a.append(h)
        return h.squeeze(-1)                 # (N,)

    def loss_and_grad(self,
                      X: np.ndarray,
                      reward: np.ndarray,
                      weights: np.ndarray | None = None):
        """
        Loss = -mean(w_i * p_i * r_i)   where p_i = sigmoid(logit_i)
        Returns (loss, grad_W_list, grad_b_list)
        """
        logit = self.forward(X)              # (N,)
        p     = sigmoid(logit)               # (N,)
        N     = len(p)

        if weights is not None:
            w = weights / weights.mean()     # 정규화: 평균=1
        else:
            w = np.ones(N)

        loss = -float(np.mean(w * p * reward))

        # dL / d_logit_i = -w_i * r_i * p_i * (1 - p_i) / N
        d_logit = (-w * reward * p * (1.0 - p) / N)[:, None]  # (N, 1)

        grad_W = []
        grad_b = []
        delta  = d_logit                     # (N, out_of_last_layer=1)

        for l in range(self.n_layers - 1, -1, -1):
            a_prev = self._a[l]              # (N, in)
            gW = a_prev.T @ delta            # (in, out)
            gb = delta.sum(axis=0)           # (out,)
            grad_W.insert(0, gW)
            grad_b.insert(0, gb)
            if l > 0:
                da     = delta @ self.W[l].T          # (N, in)
                delta  = da * (self._z[l-1] > 0.0)   # ReLU backward

        return loss, grad_W, grad_b

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.forward(X))

    def n_params(self) -> int:
        return sum(W.size for W in self.W) + sum(b.size for b in self.b)


# ══════════════════════════════════════════════════════
#  Adam Optimizer
# ══════════════════════════════════════════════════════
class Adam:
    def __init__(self, model: MLP, lr=1e-3,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        n = model.n_layers
        self.mW = [np.zeros_like(W) for W in model.W]
        self.vW = [np.zeros_like(W) for W in model.W]
        self.mb = [np.zeros_like(b) for b in model.b]
        self.vb = [np.zeros_like(b) for b in model.b]

    def step(self, grad_W, grad_b):
        self.t += 1
        b1, b2  = self.beta1, self.beta2
        lr_t    = self.lr * np.sqrt(1.0 - b2**self.t) / (1.0 - b1**self.t)

        for l in range(self.model.n_layers):
            # W
            self.mW[l] = b1 * self.mW[l] + (1-b1) * grad_W[l]
            self.vW[l] = b2 * self.vW[l] + (1-b2) * grad_W[l]**2
            self.model.W[l] -= lr_t * self.mW[l] / (np.sqrt(self.vW[l]) + self.eps)
            # b
            self.mb[l] = b1 * self.mb[l] + (1-b1) * grad_b[l]
            self.vb[l] = b2 * self.vb[l] + (1-b2) * grad_b[l]**2
            self.model.b[l] -= lr_t * self.mb[l] / (np.sqrt(self.vb[l]) + self.eps)


# ══════════════════════════════════════════════════════
#  Data utilities
# ══════════════════════════════════════════════════════
_IPC_PAIRS = [
    ("ipc_section",    "ipc_section_enc"),
    ("ipc_class_full", "ipc_class_enc"),
    ("ipc_subclass",   "ipc_subclass_enc"),
]


def fit_ipc_freq(df: pd.DataFrame) -> dict:
    """train fold 에서만 fit."""
    return {col: df[col].value_counts() if col in df.columns else pd.Series(dtype=int)
            for col, _ in _IPC_PAIRS}


def apply_ipc_freq(df: pd.DataFrame, freqs: dict) -> pd.DataFrame:
    df = df.copy()
    for col, enc_col in _IPC_PAIRS:
        if col in df.columns and col in freqs:
            df[enc_col] = df[col].map(freqs[col]).fillna(0).astype(int)
        else:
            df[enc_col] = 0
    return df


def load_data():
    """IPC encoding 은 여기서 하지 않는다 — split 후 train-only fit."""
    labels = pd.read_parquet(LABEL_FILE)
    feat   = pd.read_parquet(FEAT_FILE)
    df = feat.merge(
        labels[["patent_id", "B", "psb", "ipc_subclass"]],
        on="patent_id", how="inner",
    )
    # encoded 컬럼을 미리 확보 (값은 split 후에 채워짐)
    for _, enc_col in _IPC_PAIRS:
        df[enc_col] = 0

    feat_cols = [c for c in FEAT_COLS if c in df.columns]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df[c] = df[c].replace([np.inf, -np.inf], 0)

    return df, feat_cols


def split(df: pd.DataFrame):
    tr  = df[df["grant_year"].isin(TRAIN_YEARS)]
    val = df[df["grant_year"].isin(VAL_YEARS)]
    te  = df[df["grant_year"].isin(TEST_YEARS)]
    return tr, val, te


# ══════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════
def evaluate(y_true, scores, label, k_list=(100, 200, 500, 1000)):
    print(f"\n[{label}]")
    df_eval = pd.DataFrame({"y": y_true.values if hasattr(y_true, 'values') else y_true,
                             "s": scores.values if hasattr(scores, 'values') else scores})
    df_eval = df_eval.dropna().sort_values("s", ascending=False)
    results = {}

    if len(df_eval) == 0:
        print("  (no valid samples)")
        return results

    for k in k_list:
        k_ = min(k, len(df_eval))
        if k_ == 0:
            continue
        topk = df_eval.head(k_)
        p = int(topk["y"].sum()) / k_
        results[f"precision@{k}"] = p
        print(f"  Precision@{k:>4}: {p:.4f}  (PSB: {int(topk['y'].sum())})")

    ap = average_precision_score(df_eval["y"], df_eval["s"])
    results["average_precision"] = ap
    print(f"  Avg Precision : {ap:.4f}")
    return results


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    t0  = time.time()
    rng = np.random.default_rng(SEED)

    print("=" * 60)
    print("  Reward-Loss MLP  (t=11.5yr, direct reward maximization)")
    print("=" * 60)

    # ── 데이터 로드 & 분할 ────────────────────────────
    print("\nLoading data...")
    labels = pd.read_parquet(LABEL_FILE)
    df, feat_cols = load_data()
    print(f"  Features: {len(feat_cols)}  |  Total: {len(df):,}  PSB=1: {df['psb'].sum():,}")

    tr, val, te = split(df)
    # IPC freq encoding: train 분포로 fit, val/test 에 apply
    _freqs = fit_ipc_freq(tr)
    tr  = apply_ipc_freq(tr,  _freqs)
    val = apply_ipc_freq(val, _freqs)
    te  = apply_ipc_freq(te,  _freqs)
    print(f"  Train {len(tr):,}(PSB={tr['psb'].sum()})  "
          f"Val {len(val):,}(PSB={val['psb'].sum()})  "
          f"Test {len(te):,}(PSB={te['psb'].sum()})")

    # ── Feature 정규화 (train 기준 fit) ──────────────
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(tr[feat_cols].values.astype(np.float64))
    X_val  = scaler.transform(val[feat_cols].values.astype(np.float64))
    X_te   = scaler.transform(te[feat_cols].values.astype(np.float64))

    psb_tr  = tr["psb"].values.astype(np.float64)
    psb_val = val["psb"].values.astype(np.float64)
    psb_te  = te["psb"].values.astype(np.float64)

    # per-sample reward: r_i = -cost + psb_i * R
    reward_tr  = -MAINT_COST + psb_tr  * PSB_REWARD   # PSB=1: +4.15, PSB=0: -3.85
    reward_val = -MAINT_COST + psb_val * PSB_REWARD

    n_pos_orig = int(psb_tr.sum())
    print(f"\n  Reward per sample:  PSB=1 → {reward_tr[psb_tr==1].mean():.2f}  "
          f"PSB=0 → {reward_tr[psb_tr==0].mean():.2f}")
    print(f"  Original class ratio:  PSB=1 {n_pos_orig}/{len(psb_tr)} ({psb_tr.mean()*100:.3f}%)")

    # ── 클래스 불균형 보정 ────────────────────────────
    if USE_SMOTE:
        from imblearn.over_sampling import SMOTE
        print(f"\n  SMOTE: target PSB=1 ratio = {SMOTE_RATIO:.0%}  "
              f"(k_neighbors=5, seed={SEED})")
        sm = SMOTE(sampling_strategy=SMOTE_RATIO, k_neighbors=5,
                   random_state=SEED)
        X_tr_aug, psb_tr_aug = sm.fit_resample(X_tr, psb_tr.astype(int))
        psb_tr_aug  = psb_tr_aug.astype(np.float64)
        reward_tr_aug = -MAINT_COST + psb_tr_aug * PSB_REWARD
        n_pos_aug = int(psb_tr_aug.sum())
        print(f"  After SMOTE: {len(X_tr_aug):,} samples  "
              f"PSB=1: {n_pos_aug:,} ({psb_tr_aug.mean()*100:.1f}%)")
        # 정규 배치 학습 (이미 balanced)
        X_tr_use      = X_tr_aug
        reward_tr_use = reward_tr_aug
        psb_tr_use    = psb_tr_aug
        tag = "smote"
    else:
        print(f"  Imbalance strategy: balanced mini-batch (50:50 oversampling)")
        X_tr_use      = X_tr
        reward_tr_use = reward_tr
        psb_tr_use    = psb_tr
        tag = "balanced_batch"

    # ── 모델 초기화 ──────────────────────────────────
    n_in  = X_tr.shape[1]
    model = MLP(n_in, HIDDEN_SIZES, seed=SEED)
    opt   = Adam(model, lr=LR)
    print(f"\n  MLP: {n_in} → {' → '.join(map(str, HIDDEN_SIZES))} → 1  "
          f"({model.n_params():,} params)")
    print(f"  Loss: -mean(sigmoid(s_i) * r_i)")

    # ── 학습 루프 설정 ────────────────────────────────
    if USE_SMOTE:
        # SMOTE 후 일반 배치: 전체 데이터 셔플
        n_tr_use        = len(X_tr_use)
        idx_all_use     = np.arange(n_tr_use)
        steps_per_epoch = max(1, n_tr_use // BATCH_SIZE)
        print(f"\nTraining  epochs={N_EPOCHS}  batch={BATCH_SIZE}  "
              f"lr={LR}  patience={PATIENCE}")
        print(f"  steps/epoch: {steps_per_epoch}  (SMOTE-augmented, regular batches)")
    else:
        # balanced batch
        idx_pos = np.where(psb_tr_use == 1)[0]
        idx_neg = np.where(psb_tr_use == 0)[0]
        half_batch      = BATCH_SIZE // 2
        steps_per_epoch = max(1, len(idx_neg) // half_batch)
        print(f"\nTraining  epochs={N_EPOCHS}  batch={BATCH_SIZE}(balanced 50:50)  "
              f"lr={LR}  patience={PATIENCE}")
        print(f"  steps/epoch: {steps_per_epoch}")

    best_val   = np.inf
    best_W     = None
    best_b     = None
    no_improve = 0
    history    = []

    for epoch in range(1, N_EPOCHS + 1):
        epoch_losses = []

        if USE_SMOTE:
            rng.shuffle(idx_all_use)
            for start in range(0, n_tr_use, BATCH_SIZE):
                idx  = idx_all_use[start:start + BATCH_SIZE]
                loss, gW, gb = model.loss_and_grad(
                    X_tr_use[idx], reward_tr_use[idx], weights=None)
                opt.step(gW, gb)
                epoch_losses.append(loss)
        else:
            rng.shuffle(idx_neg)
            for step in range(steps_per_epoch):
                idx1   = rng.choice(idx_pos, half_batch, replace=True)
                start0 = step * half_batch % len(idx_neg)
                end0   = start0 + half_batch
                if end0 <= len(idx_neg):
                    idx0 = idx_neg[start0:end0]
                else:
                    idx0 = np.concatenate([idx_neg[start0:],
                                           idx_neg[:end0 - len(idx_neg)]])
                idx  = np.concatenate([idx1, idx0])
                loss, gW, gb = model.loss_and_grad(
                    X_tr_use[idx], reward_tr_use[idx], weights=None)
                opt.step(gW, gb)
                epoch_losses.append(loss)

        # val metric: Average Precision (higher=better → negate for minimize)
        logit_val = model.predict_logit(X_val)
        val_ap    = average_precision_score(psb_val, logit_val)
        val_loss  = -val_ap   # minimize negative AP

        train_loss = float(np.mean(epoch_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch % 25 == 0 or epoch == 1:
            val_sorted  = np.argsort(-logit_val)
            psb_top100  = int(psb_val[val_sorted[:100]].sum())
            print(f"  Ep {epoch:3d}/{N_EPOCHS}  "
                  f"train={train_loss:+.5f}  val_AP={val_ap:.4f}  "
                  f"val P@100={psb_top100}/100")

        if val_loss < best_val - 1e-7:
            best_val   = val_loss
            best_W     = [W.copy() for W in model.W]
            best_b     = [b.copy() for b in model.b]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {epoch}  (best val={best_val:+.5f})")
                break

    model.W = best_W
    model.b = best_b
    print(f"  Final best val_loss: {best_val:+.5f}")

    # ── Test set 평가 ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)

    logit_te = model.predict_logit(X_te)
    proba_te = sigmoid(logit_te)

    scores_te = pd.Series(logit_te, index=te.index)
    y_te      = te["psb"]

    our_results = evaluate(y_te, scores_te, "Reward-Loss MLP  (11.5yr features)")

    # ── 기존 baseline과 비교 (모델 직접 로드) ──────────
    all_evals = {"reward_loss_mlp_115yr": our_results}
    print("\n── Comparison with existing baselines (direct model load) ──")

    # 3.5yr features 로드 (기존 모델들은 3.5yr 기준으로 학습됨)
    feat35_path = ROOT / "features" / "features_3_5yr.parquet"
    if feat35_path.exists():
        feat35 = pd.read_parquet(feat35_path)
        df35_full = feat35.merge(
            labels[["patent_id", "psb", "ipc_subclass"]],
            on="patent_id", how="inner",
        )
        # IPC 인코딩: train fold (1982~1986) 에서만 fit 후 전체에 apply
        _tr_mask = df35_full["grant_year"].isin(TRAIN_YEARS)
        _freqs35 = fit_ipc_freq(df35_full[_tr_mask])
        df35_full = apply_ipc_freq(df35_full, _freqs35)

        # 3.5yr feature 컬럼
        def get_cite35():
            p = "t35"
            return [f"{p}__cum_citations",f"{p}__cite_last1yr",f"{p}__cite_last3yr",
                    f"{p}__cite_growth_rate",f"{p}__cite_growth_last3",f"{p}__cite_acceleration",
                    f"{p}__cite_peak_ratio",f"{p}__cite_active_years",f"{p}__zero_citation"]
        feat_cols_35 = [c for c in STATIC_COLS + get_cite35() if c in df35_full.columns]
        for c in feat_cols_35:
            df35_full[c] = pd.to_numeric(df35_full[c], errors="coerce").fillna(0).replace([np.inf,-np.inf],0)

        te35 = df35_full[df35_full["grant_year_x"].isin(TEST_YEARS) if "grant_year_x" in df35_full.columns else df35_full["grant_year"].isin(TEST_YEARS)]
        X_te35 = te35[feat_cols_35].values

        # XGBoost Classifier
        xgb_pkl = ROOT / "models" / "baseline_xgb.pkl"
        if xgb_pkl.exists():
            xgb_clf = pickle.load(open(xgb_pkl, "rb"))
            xgb_pred = pd.Series(xgb_clf.predict_proba(X_te35)[:, 1], index=te35.index)
            all_evals["XGBoost Classifier (3.5yr)"] = evaluate(te35["psb"], xgb_pred, "XGBoost Classifier (3.5yr features)")

        # Random Forest
        rf_pkl = ROOT / "models" / "baseline_rf.pkl"
        if rf_pkl.exists():
            rf_clf = pickle.load(open(rf_pkl, "rb"))
            rf_pred = pd.Series(rf_clf.predict_proba(X_te35)[:, 1], index=te35.index)
            all_evals["Random Forest (3.5yr)"] = evaluate(te35["psb"], rf_pred, "Random Forest (3.5yr features)")

        # Beauty Coefficient B 는 label 정의에 사용되므로 baseline 제외.
        cite_col = "t35__cum_citations"
        if cite_col in te35.columns:
            all_evals["Cum Citations"] = evaluate(te35["psb"], te35[cite_col], "Cumulative Citations (3.5yr)")

    # Backward Induction Q35 (기존 test_qvalues.parquet)
    qval_path = ROOT / "results" / "test_qvalues.parquet"
    if qval_path.exists():
        qv = pd.read_parquet(qval_path).dropna(subset=["psb"])
        if "Q_35_pred" in qv.columns:
            all_evals["Backward Induction Q35"] = evaluate(qv["psb"], qv["Q_35_pred"], "Backward Induction Q_35 (3.5yr)")

    # ── 결과 저장 ─────────────────────────────────────
    print("\n── Saving results ──")

    suffix = f"_{tag}"
    out_df = te[["patent_id", "grant_year", "psb"]].copy()
    out_df["logit"]  = logit_te
    out_df["proba"]  = proba_te
    out_df["policy"] = (logit_te > 0).astype(int)
    out_df.to_parquet(RESULTS_DIR / f"reward_nn{suffix}.parquet", index=False)

    # summary CSV: 기존 파일이 있으면 merge
    summary_rows = [{"model": name, **metrics}
                    for name, metrics in all_evals.items()]
    summary_df = pd.DataFrame(summary_rows)
    csv_path   = RESULTS_DIR / "reward_nn_summary.csv"
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        old = old[~old["model"].isin(summary_df["model"])]
        summary_df = pd.concat([old, summary_df], ignore_index=True)
    summary_df.to_csv(csv_path, index=False)
    print(f"  reward_nn{suffix}.parquet  ({len(out_df):,} rows)")
    print(f"  reward_nn_summary.csv  (updated)")

    model_dict = {
        "W": model.W, "b": model.b,
        "scaler": scaler,
        "feat_cols": feat_cols,
        "hidden_sizes": HIDDEN_SIZES,
        "maint_cost": MAINT_COST,
        "psb_reward": PSB_REWARD,
        "use_smote": USE_SMOTE,
        "smote_ratio": SMOTE_RATIO if USE_SMOTE else None,
        "history": history,
    }
    pkl_name = f"mlp_reward{suffix}.pkl"
    pickle.dump(model_dict, open(MODELS_DIR / pkl_name, "wb"))
    print(f"  {pkl_name}")

    print(f"\nDone.  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
