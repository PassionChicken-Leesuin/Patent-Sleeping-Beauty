"""
Neural Fitted Q-Iteration (ev_nn)
===================================
Backward Induction의 3단계 구조를 그대로 유지하되,
XGBoost 대신 MLP(numpy 수동 구현)를 함수 근사기로 사용.

기존 experiments/exp1_backward_induction.py 와의 차이:
  exp1  : XGBRegressor  + sample_weight(PSB_WEIGHT)
  ev_nn : MLP(linear)   + weighted MSE  (또는 Huber / balanced-batch)

Stage A (t=11.5yr) — terminal:
  Y = -cost_115 + psb × R

Stage B (t=7.5yr):
  Y = -cost_75  + γ × max(Q̂_115, 0)

Stage C (t=3.5yr) — 최종 결정 시점:
  Y = -cost_35  + γ × max(Q̂_75,  0)

Loss    : weighted MSE  (기본) / Huber / balanced-batch MSE
Output  : linear (sigmoid 없음) → EV 직접 예측
Decision: Q > 0 → maintain

사용법:
  python train.py                              # weighted MSE, thr=0.001
  python train.py --variant huber              # Huber loss (δ=1.0)
  python train.py --variant balanced           # balanced mini-batch
  python train.py --thr 0.005 --psb_weight 100
"""

import sys
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from config import (
    GAMMA, MAINT_COST, PSB_REWARD, PSB_WEIGHT,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    labels_file,
)
from utils import (
    load_and_merge, split, encode_ipc_splits, print_split_stats,
    get_feature_cols, evaluate_ranking,
)

MODELS_DIR  = Path(__file__).parent / "models";  MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(__file__).parent / "results"; RESULTS_DIR.mkdir(exist_ok=True)

# ── 하이퍼파라미터 기본값 ──────────────────────────────
HIDDEN_SIZES = [128, 64]
LR           = 1e-3
BATCH_SIZE   = 512
N_EPOCHS     = 300
PATIENCE     = 25
SEED         = 42


# ══════════════════════════════════════════════════════
#  MLP — linear output (no sigmoid), EV 직접 예측
# ══════════════════════════════════════════════════════
class MLP:
    """
    구조: input → [Linear → ReLU] × L → Linear (scalar)

    reward_loss_nn 의 MLP와 동일한 아키텍처이나
    출력층이 sigmoid 없는 순수 linear → EV 회귀용.
    """

    def __init__(self, n_in: int, hidden: list, seed: int = 42):
        rng   = np.random.default_rng(seed)
        sizes = [n_in] + hidden + [1]
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            W = rng.standard_normal((sizes[i], sizes[i+1])) * np.sqrt(2.0 / sizes[i])
            self.W.append(W)
            self.b.append(np.zeros(sizes[i+1]))
        self.n_layers = len(self.W)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (N, F) → pred: (N,)  |  저장: _a, _z"""
        self._a = [X]
        self._z = []
        h = X
        for l in range(self.n_layers):
            z = h @ self.W[l] + self.b[l]
            self._z.append(z)
            h = np.maximum(0.0, z) if l < self.n_layers - 1 else z
            self._a.append(h)
        return h.squeeze(-1)                      # (N,)

    def loss_and_grad(self,
                      X: np.ndarray,
                      Y: np.ndarray,
                      weights: np.ndarray | None = None,
                      huber_delta: float | None = None):
        """
        Weighted MSE (기본) 또는 Huber loss.

        Loss = mean(w_i × loss_i)
          MSE  : loss_i = (pred_i - Y_i)²
          Huber: loss_i = 0.5×r² if |r|≤δ  else  δ×(|r|-0.5δ)

        Returns (loss, grad_W_list, grad_b_list)
        """
        pred = self.forward(X)                    # (N,)
        N    = len(pred)
        r    = pred - Y                           # residual (N,)

        w = (weights / weights.mean()) if weights is not None else np.ones(N)

        if huber_delta is not None:
            abs_r  = np.abs(r)
            loss_i = np.where(abs_r <= huber_delta,
                              0.5 * r**2,
                              huber_delta * (abs_r - 0.5 * huber_delta))
            d_pred = np.where(abs_r <= huber_delta,
                              r,
                              huber_delta * np.sign(r))
        else:
            loss_i = r**2
            d_pred = 2.0 * r

        loss   = float(np.mean(w * loss_i))
        d_pred = (w * d_pred / N)[:, None]        # (N, 1)

        # ── backprop ──
        grad_W, grad_b = [], []
        delta = d_pred
        for l in range(self.n_layers - 1, -1, -1):
            gW = self._a[l].T @ delta
            gb = delta.sum(axis=0)
            grad_W.insert(0, gW)
            grad_b.insert(0, gb)
            if l > 0:
                da    = delta @ self.W[l].T
                delta = da * (self._z[l-1] > 0.0)   # ReLU backward

        return loss, grad_W, grad_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def n_params(self) -> int:
        return sum(W.size for W in self.W) + sum(b.size for b in self.b)


# ══════════════════════════════════════════════════════
#  Adam Optimizer
# ══════════════════════════════════════════════════════
class Adam:
    def __init__(self, model: MLP, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t  = 0
        self.mW = [np.zeros_like(W) for W in model.W]
        self.vW = [np.zeros_like(W) for W in model.W]
        self.mb = [np.zeros_like(b) for b in model.b]
        self.vb = [np.zeros_like(b) for b in model.b]

    def step(self, grad_W, grad_b):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        lr_t   = self.lr * np.sqrt(1.0 - b2**self.t) / (1.0 - b1**self.t)
        for l in range(self.model.n_layers):
            self.mW[l] = b1*self.mW[l] + (1-b1)*grad_W[l]
            self.vW[l] = b2*self.vW[l] + (1-b2)*grad_W[l]**2
            self.model.W[l] -= lr_t * self.mW[l] / (np.sqrt(self.vW[l]) + self.eps)
            self.mb[l] = b1*self.mb[l] + (1-b1)*grad_b[l]
            self.vb[l] = b2*self.vb[l] + (1-b2)*grad_b[l]**2
            self.model.b[l] -= lr_t * self.mb[l] / (np.sqrt(self.vb[l]) + self.eps)


# ══════════════════════════════════════════════════════
#  단계별 학습 함수
# ══════════════════════════════════════════════════════
def train_stage(
    X_tr:  np.ndarray, Y_tr:  np.ndarray, psb_tr:  np.ndarray,
    X_val: np.ndarray, Y_val: np.ndarray, psb_val: np.ndarray,
    tag: str,
    hidden_sizes: list  = HIDDEN_SIZES,
    lr: float           = LR,
    batch_size: int     = BATCH_SIZE,
    n_epochs: int       = N_EPOCHS,
    patience: int       = PATIENCE,
    seed: int           = SEED,
    variant: str        = "weighted",
    psb_weight: float   = PSB_WEIGHT,
    huber_delta: float | None = None,
) -> MLP:
    """
    단일 시점 Q-함수를 MLP로 학습.

    variant:
      'weighted'  : 전체 배치 + PSB=1 샘플 가중치 (PSB_WEIGHT)
      'balanced'  : balanced mini-batch (PSB=1/0 50:50 오버샘플링)
      'huber'     : Huber loss + PSB=1 가중치
    """
    rng  = np.random.default_rng(seed)
    n_in = X_tr.shape[1]

    model = MLP(n_in, hidden_sizes, seed=seed)
    opt   = Adam(model, lr=lr)

    print(f"\n  [{tag}]  MLP {n_in}->{'->'.join(map(str,hidden_sizes))}->1  "
          f"({model.n_params():,} params)  variant={variant}")
    print(f"  Y range: [{Y_tr.min():.3f}, {Y_tr.max():.3f}]  "
          f"PSB=1: {int(psb_tr.sum())}/{len(psb_tr)}")

    # ── 배치 전략 설정 ────────────────────────────────
    if variant == "balanced":
        idx_pos         = np.where(psb_tr == 1)[0]
        idx_neg         = np.where(psb_tr == 0)[0]
        half            = batch_size // 2
        steps_per_epoch = max(1, len(idx_neg) // half)
        w_batch         = None
        huber_delta_use = None
    else:
        idx_all         = np.arange(len(X_tr))
        steps_per_epoch = max(1, len(X_tr) // batch_size)
        w_batch         = np.where(psb_tr == 1, psb_weight, 1.0)
        huber_delta_use = huber_delta if variant == "huber" else None

    print(f"  steps/epoch={steps_per_epoch}  batch={batch_size}  "
          f"lr={lr}  patience={patience}")

    best_val   = np.inf
    best_W     = None
    best_b     = None
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        losses = []

        if variant == "balanced":
            rng.shuffle(idx_neg)
            for step in range(steps_per_epoch):
                idx1 = rng.choice(idx_pos, half, replace=True)
                s    = step * half % len(idx_neg)
                e    = s + half
                if e <= len(idx_neg):
                    idx0 = idx_neg[s:e]
                else:
                    idx0 = np.concatenate([idx_neg[s:], idx_neg[:e - len(idx_neg)]])
                idx = np.concatenate([idx1, idx0])
                loss, gW, gb = model.loss_and_grad(X_tr[idx], Y_tr[idx])
                opt.step(gW, gb)
                losses.append(loss)
        else:
            rng.shuffle(idx_all)
            for s in range(0, len(X_tr), batch_size):
                idx  = idx_all[s:s + batch_size]
                loss, gW, gb = model.loss_and_grad(
                    X_tr[idx], Y_tr[idx], w_batch[idx], huber_delta_use)
                opt.step(gW, gb)
                losses.append(loss)

        # ── 검증: RMSE (early stopping 기준) ──────────
        pred_val = model.predict(X_val)
        val_rmse = float(np.sqrt(np.mean((pred_val - Y_val)**2)))

        if epoch % 25 == 0 or epoch == 1:
            ap = (average_precision_score(psb_val, pred_val)
                  if psb_val.sum() > 0 else 0.0)
            print(f"    Ep {epoch:3d}/{n_epochs}  "
                  f"train={np.mean(losses):.5f}  "
                  f"val_RMSE={val_rmse:.4f}  val_AP={ap:.4f}")

        if val_rmse < best_val - 1e-7:
            best_val   = val_rmse
            best_W     = [W.copy() for W in model.W]
            best_b     = [b.copy() for b in model.b]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop @ epoch {epoch}  (best RMSE={best_val:.4f})")
                break

    model.W = best_W
    model.b = best_b
    print(f"  [{tag}] Final best val RMSE: {best_val:.4f}")
    return model


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--thr",         type=float, default=0.001,
                        help="PSB threshold (default 0.001)")
    parser.add_argument("--variant",     default="weighted",
                        choices=["weighted", "balanced", "huber"],
                        help="Training variant (default: weighted)")
    parser.add_argument("--psb_weight",  type=float, default=PSB_WEIGHT,
                        help="Weight for PSB=1 samples (weighted/huber only)")
    parser.add_argument("--huber_delta", type=float, default=1.0,
                        help="Huber loss delta (huber variant only)")
    args, _ = parser.parse_known_args()

    thr         = args.thr
    variant     = args.variant
    psb_weight  = args.psb_weight
    huber_delta = args.huber_delta if variant == "huber" else None
    run_tag     = f"thr{int(thr*1000):03d}_{variant}"

    print("=" * 60)
    print("  ev_nn : Neural Fitted Q-Iteration")
    print(f"  thr={thr}  variant={variant}  psb_weight={psb_weight}")
    if huber_delta:
        print(f"  huber_delta={huber_delta}")
    print("=" * 60)
    t0 = time.time()

    # ── 데이터 로드 ───────────────────────────────────
    print("\nLoading data...")
    labels = pd.read_parquet(labels_file(thr))
    print(f"  Labels: {len(labels):,}  PSB=1: {labels['psb'].sum():,}")

    df35  = load_and_merge(3.5,  labels)
    df75  = load_and_merge(7.5,  labels)
    df115 = load_and_merge(11.5, labels)

    tr35,  val35,  te35  = split(df35)
    tr75,  val75,  te75  = split(df75)
    tr115, val115, te115 = split(df115)
    tr35,  val35,  te35  = encode_ipc_splits(tr35,  val35,  te35)
    tr75,  val75,  te75  = encode_ipc_splits(tr75,  val75,  te75)
    tr115, val115, te115 = encode_ipc_splits(tr115, val115, te115)

    print("\nData split:")
    print_split_stats("3.5yr",  tr35,  val35,  te35)
    print_split_stats("7.5yr",  tr75,  val75,  te75)
    print_split_stats("11.5yr", tr115, val115, te115)

    # ── Feature 컬럼 & Scaler ─────────────────────────
    fc115 = [c for c in get_feature_cols(11.5) if c in tr115.columns]
    fc75  = [c for c in get_feature_cols(7.5)  if c in tr75.columns]
    fc35  = [c for c in get_feature_cols(3.5)  if c in tr35.columns]

    sc115 = StandardScaler()
    sc75  = StandardScaler()
    sc35  = StandardScaler()

    X_tr115  = sc115.fit_transform(tr115[fc115].values.astype(np.float64))
    X_val115 = sc115.transform(val115[fc115].values.astype(np.float64))
    X_te115  = sc115.transform(te115[fc115].values.astype(np.float64))

    X_tr75   = sc75.fit_transform(tr75[fc75].values.astype(np.float64))
    X_val75  = sc75.transform(val75[fc75].values.astype(np.float64))
    X_te75   = sc75.transform(te75[fc75].values.astype(np.float64))

    X_tr35   = sc35.fit_transform(tr35[fc35].values.astype(np.float64))
    X_val35  = sc35.transform(val35[fc35].values.astype(np.float64))
    X_te35   = sc35.transform(te35[fc35].values.astype(np.float64))

    # ════════════════════════════════════════════════════
    #  Stage A: t=11.5yr (terminal)
    #  Y = -cost_115 + psb × R
    # ════════════════════════════════════════════════════
    print("\n" + "─" * 55)
    print("  Stage A: t=11.5yr  (terminal -- no bootstrapping)")
    print("─" * 55)
    cost_115 = MAINT_COST[11.5]

    Y_tr115  = -cost_115 + tr115["psb"].values  * PSB_REWARD
    Y_val115 = -cost_115 + val115["psb"].values * PSB_REWARD
    Y_te115  = -cost_115 + te115["psb"].values  * PSB_REWARD

    model_115 = train_stage(
        X_tr115, Y_tr115, tr115["psb"].values,
        X_val115, Y_val115, val115["psb"].values,
        tag="Q_11.5",
        variant=variant, psb_weight=psb_weight, huber_delta=huber_delta,
    )

    # 전체 split에 대해 Q_115 예측 -> Stage B target 구성에 사용
    Q115_tr  = model_115.predict(X_tr115)
    Q115_val = model_115.predict(X_val115)
    Q115_te  = model_115.predict(X_te115)

    q115_map = {}
    for df_, Q_ in [(tr115, Q115_tr), (val115, Q115_val), (te115, Q115_te)]:
        for pid, q in zip(df_["patent_id"].values, Q_):
            q115_map[pid] = q

    # ════════════════════════════════════════════════════
    #  Stage B: t=7.5yr
    #  Y = -cost_75 + γ × max(Q̂_115, 0)
    # ════════════════════════════════════════════════════
    print("\n" + "─" * 55)
    print("  Stage B: t=7.5yr  (bootstrap from Stage A)")
    print("─" * 55)
    cost_75 = MAINT_COST[7.5]

    def make_Y75(df_):
        q_next = df_["patent_id"].map(q115_map).fillna(0.0).values
        return -cost_75 + GAMMA * np.maximum(q_next, 0.0)

    Y_tr75  = make_Y75(tr75)
    Y_val75 = make_Y75(val75)
    Y_te75  = make_Y75(te75)

    model_75 = train_stage(
        X_tr75, Y_tr75, tr75["psb"].values,
        X_val75, Y_val75, val75["psb"].values,
        tag="Q_7.5",
        variant=variant, psb_weight=psb_weight, huber_delta=huber_delta,
    )

    Q75_tr  = model_75.predict(X_tr75)
    Q75_val = model_75.predict(X_val75)
    Q75_te  = model_75.predict(X_te75)

    q75_map = {}
    for df_, Q_ in [(tr75, Q75_tr), (val75, Q75_val), (te75, Q75_te)]:
        for pid, q in zip(df_["patent_id"].values, Q_):
            q75_map[pid] = q

    # ════════════════════════════════════════════════════
    #  Stage C: t=3.5yr  (최종 결정 시점)
    #  Y = -cost_35 + γ × max(Q̂_75, 0)
    # ════════════════════════════════════════════════════
    print("\n" + "─" * 55)
    print("  Stage C: t=3.5yr  (bootstrap from Stage B -- decision point)")
    print("─" * 55)
    cost_35 = MAINT_COST[3.5]

    def make_Y35(df_):
        q_next = df_["patent_id"].map(q75_map).fillna(0.0).values
        return -cost_35 + GAMMA * np.maximum(q_next, 0.0)

    Y_tr35  = make_Y35(tr35)
    Y_val35 = make_Y35(val35)
    Y_te35  = make_Y35(te35)

    model_35 = train_stage(
        X_tr35, Y_tr35, tr35["psb"].values,
        X_val35, Y_val35, val35["psb"].values,
        tag="Q_3.5",
        variant=variant, psb_weight=psb_weight, huber_delta=huber_delta,
    )

    Q35_te = model_35.predict(X_te35)

    # ════════════════════════════════════════════════════
    #  Test Set 평가
    # ════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)

    te_q75_map  = {pid: q for pid, q in zip(te75["patent_id"].values,  Q75_te)}
    te_q115_map = {pid: q for pid, q in zip(te115["patent_id"].values, Q115_te)}

    results = te35[["patent_id", "grant_year", "psb"]].copy()
    results["Q_35"]  = Q35_te
    results["Q_75"]  = results["patent_id"].map(te_q75_map)
    results["Q_115"] = results["patent_id"].map(te_q115_map)

    results["policy_35"]  = (results["Q_35"]  > 0).astype(int)
    results["policy_75"]  = (results["Q_75"]  > 0).astype(int)
    results["policy_115"] = (results["Q_115"] > 0).astype(int)
    results["policy_full"] = (
        (results["policy_35"] == 1) &
        (results["policy_75"] == 1) &
        (results["policy_115"] == 1)
    ).astype(int)

    print("\n-- Q_35 Ranking Metrics --")
    ranking_res = evaluate_ranking(
        results["psb"], results["Q_35"],
        label=f"Neural FQI Q_35  [{run_tag}]"
    )

    print("\n-- Policy Stats --")
    total = len(results)
    psb_total = int(results["psb"].sum())
    print(f"  Test total={total:,}  PSB={psb_total}")
    for col, lbl in [
        ("policy_35",  "3.5yr"),
        ("policy_75",  "7.5yr"),
        ("policy_115", "11.5yr"),
        ("policy_full","full (all 3)"),
    ]:
        sel    = int(results[col].sum())
        hit    = int(results[results[col] == 1]["psb"].sum())
        prec   = hit / sel       if sel       > 0 else 0.0
        recall = hit / psb_total if psb_total > 0 else 0.0
        print(f"  [{lbl:12s}]  maintain={sel:,}  PSB={hit}  "
              f"prec={prec:.4f}  recall={recall:.4f}")

    # ── 결과 저장 ─────────────────────────────────────
    print("\n-- Saving --")

    out_parq = RESULTS_DIR / f"test_qvalues_{run_tag}.parquet"
    results.to_parquet(out_parq, index=False)
    print(f"  {out_parq.name}  ({len(results):,} rows)")

    # summary CSV
    summary_row = {"model": f"ev_nn_{run_tag}", **ranking_res}
    csv_path    = RESULTS_DIR / "ev_nn_summary.csv"
    summary_new = pd.DataFrame([summary_row])
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        old = old[old["model"] != summary_row["model"]]
        summary_new = pd.concat([old, summary_new], ignore_index=True)
    summary_new.to_csv(csv_path, index=False)
    print(f"  ev_nn_summary.csv  (updated)")

    # 모델 pkl 저장
    for model_, name_, scaler_, fc_ in [
        (model_115, "q_115", sc115, fc115),
        (model_75,  "q_75",  sc75,  fc75),
        (model_35,  "q_35",  sc35,  fc35),
    ]:
        pkl_data = {
            "W": model_.W, "b": model_.b,
            "scaler": scaler_,
            "feat_cols": fc_,
            "hidden_sizes": HIDDEN_SIZES,
            "variant": variant,
            "psb_weight": psb_weight,
            "huber_delta": huber_delta,
            "thr": thr,
        }
        pkl_path = MODELS_DIR / f"{name_}_{run_tag}.pkl"
        pickle.dump(pkl_data, open(pkl_path, "wb"))
        print(f"  {pkl_path.name}")

    print(f"\nDone.  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
