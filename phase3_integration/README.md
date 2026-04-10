# Phase 3 #1 — PSB × PSB_Refinement Integration

두 연구 트랙(PSB feature augmentation + PSB_Refinement sparse-action
semi-Markov MDP)을 결합한 첫 번째 통합 실험.

## 구조

`train_bi_integrated.py` 는 원래 `PSB_Refinement/A_sparse_mdp/` 에
존재하는 실험 스크립트의 **사본**이다. PSB 저장소에도 포함해 재현 가능성을
확보.

원본 위치:
```
../PSB_Refinement/A_sparse_mdp/train_bi_integrated.py
```

## Feature 구성 (각 decision age t ∈ {3, 7, 11})

| 블록 | 출처 | 컬럼 수 | 비고 |
|---|---|---|---|
| STATIC | PSB `STATIC_COLS` | 19 | train-fold IPC freq encoding |
| DYNAMIC (yearly panel) | PSB_Refinement `yearly_panel_dynamic.parquet` | 11 | `last` pool @ age=t |
| MAINT | PSB `features/maint_features.parquet` | 0/1/2 | age=7에서 `paid_3_5`, age=11에서 `paid_3_5, paid_7_5` |
| CITER+SHAPE (Phase 1) | PSB `features/dynamic_t{35,75,115}.parquet` | 17 | cutoff-aware |
| SEMANTIC (Phase 2) | PSB `features/abstract_dynamic.parquet` | 7 | cutoff-무관, MiniLM PCA-50 |

총 약 54~56 features per age.

## Learning Schedule

```
pool       = "last"          # A1 winner
psb_weight = 50               # A1 winner
no_clip    = True             # A1 winner
backbone   = XGBRegressor (PSB/config.XGB_PARAMS)
cascade    = Q_11 -> Q_7 -> Q_3  (backward induction)
```

## 재현

```bash
cd ../PSB_Refinement/A_sparse_mdp
python train_bi_integrated.py --thr all --variant integrated_v1
```

결과는 `PSB_Refinement/A_sparse_mdp/results/thr{001,005,010}/test_qvalues_A1_integrated_v1.parquet`
에 저장되고, 집계 CSV 두 개가 PSB 쪽으로 옮겨짐:

```
PSB/results/phase3_integrated_summary.csv
PSB/results/phase3_integrated_paired_bootstrap.csv
```

## 결과 요약

### 3-way 비교 (test 1988-1989)

| thr | 모델 | AP | hits@100 | hits@500 | hits@1000 | lift@100 |
|---|---|---|---|---|---|---|
| 0.001 | **Integrated** | **0.0095** | **7** | **19** | **30** | **27.45x** |
| 0.001 | PSB Phase 2 BI Q_35 | 0.0064 | 6 | 15 | 22 | 23.53x |
| 0.001 | A1 winner (PSB_Ref) | 0.0067 | 4 | 15 | 26 | 15.69x |
| 0.005 | **Integrated** | **0.0113** | **8** | **21** | **32** | **10.48x** |
| 0.005 | PSB Phase 2 BI Q_35 | 0.0105 | 4 | 15 | 30 | 5.24x |
| 0.005 | A1 winner | 0.0094 | 3 | 15 | 23 | 3.93x |
| 0.010 | Integrated | 0.0176 | 4 | 21 | 34 | 2.86x |
| 0.010 | PSB Phase 2 BI Q_35 | 0.0176 | 4 | 20 | **41** | 2.86x |
| 0.010 | A1 winner | 0.0153 | 3 | 16 | 23 | 2.14x |

### Paired Bootstrap (95% CI excludes 0)

**Integrated vs A1 winner (PSB_Refinement baseline)**:
- 모든 thr에서 **AP 유의하게 승** (thr001 delta=+0.0031, thr005 +0.0021, thr010 +0.0023).
- Hits@k는 델타가 크지만 분산도 커서 유의성 없음.

**Integrated vs PSB Phase 2 BI Q_35**:
- **thr001 hits@1000 유의하게 승** (+8.5, 95% CI [+1, +16]).
- thr001 AP delta=+0.0031, 95% CI [-0.0003, +0.0068] — 경계.
- thr010 hits@1000 에서 오히려 Phase 2가 더 큼 (41 vs 34).

## 해석

1. **통합이 PSB_Refinement A1 winner 를 AP 기준으로 uniformly dominate.** A1 winner의 방법론(yearly panel + BI cascade)은 유효하지만, PSB Phase 1/2의 citer-quality + semantic novelty features 없이는 제한적.
2. **통합이 PSB Phase 2를 Lift@top-100 에서 ~4점 상승** (23.53x → 27.45x, thr001), hits@1000 에서 유의하게 승. 그러나 AP에서는 경계 수준 유의성.
3. **thr010 lenient에서는 PSB Phase 2가 hits@1000 기준 여전히 best** (41). 이 구간에서 yearly panel 효과가 희석되거나 간섭.
4. **가장 큰 개선은 thr001에서 일관되게 나타남** — strict threshold일수록 yearly dynamics가 tail detection에 기여.
5. 통합은 **universal improvement 가 아니라** strict threshold 의 top-k tail detection 을 추가로 밀어올리는 도구로 봐야 함.

## 다음 후보

- `integrated_v2`: `pool="full"` (last + mean + max + last3_mean) 로 시도 — yearly panel summary 확장
- `integrated_v3`: `psb_weight` sweep (30, 50, 100)
- Phase 3 #2: LR + BI 앙상블 (stacking or rank-averaging) — AP와 top-k 양쪽 최적화
