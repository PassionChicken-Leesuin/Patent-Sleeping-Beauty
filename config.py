"""
PSB 프로젝트 전역 설정 — 단일 진실 소스 (Single Source of Truth)
모든 pipeline/ 및 experiments/ 스크립트가 이 파일을 import.
"""
from pathlib import Path

# ── 루트 경로 ─────────────────────────────────────────
ROOT = Path(__file__).parent

# ── 원본 데이터 경로 ──────────────────────────────────
BULK_DIR  = Path(r"C:\Users\User\OneDrive\문서\이수인\서울대학교\Biblo+Text\Patent_Bulk")
MAINT_DIR = Path(r"C:\Users\User\OneDrive\문서\이수인\서울대학교\Sbj_SB\MaintFeeEvents_20260331")

MAINT_FILE       = MAINT_DIR / "MaintFeeEvents_20260330.txt"
PATENT_FILE      = BULK_DIR  / "g_patent.tsv"
APPLICATION_FILE = BULK_DIR  / "g_application.tsv"
CITATION_FILE    = BULK_DIR  / "g_us_patent_citation.tsv"
IPC_FILE         = BULK_DIR  / "g_ipc_at_issue.tsv"
ASSIGNEE_FILE    = BULK_DIR  / "g_assignee_disambiguated.tsv"
INVENTOR_FILE    = BULK_DIR  / "g_inventor_disambiguated.tsv"
FIGURES_FILE     = BULK_DIR  / "g_figures.tsv"
FOREIGN_FILE     = BULK_DIR  / "g_foreign_citation.tsv"

# ── 중간 산출물 (processed/) ──────────────────────────
PROCESSED_DIR = ROOT / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

PATENTS_80S_FILE     = PROCESSED_DIR / "patents_80s.parquet"
MAINT_EVENTS_FILE    = PROCESSED_DIR / "maint_events_80s.parquet"
CITATION_ANNUAL_FILE = PROCESSED_DIR / "citation_annual.parquet"
BACKWARD_CIT_FILE    = PROCESSED_DIR / "backward_citation.parquet"
IPC_FILE_OUT         = PROCESSED_DIR / "ipc_main.parquet"
ASSIGNEE_FILE_OUT    = PROCESSED_DIR / "assignee.parquet"
INVENTOR_FILE_OUT    = PROCESSED_DIR / "inventor_count.parquet"
FIGURES_FILE_OUT     = PROCESSED_DIR / "figures.parquet"
FOREIGN_CIT_FILE_OUT = PROCESSED_DIR / "foreign_citation.parquet"

# ── 피처 산출물 (features/) ───────────────────────────
FEATURES_DIR = ROOT / "features"
FEATURES_DIR.mkdir(exist_ok=True)

FEATURES_35_FILE  = FEATURES_DIR / "features_3_5yr.parquet"
FEATURES_75_FILE  = FEATURES_DIR / "features_7_5yr.parquet"
FEATURES_115_FILE = FEATURES_DIR / "features_11_5yr.parquet"
MAINT_FEATURES_FILE = FEATURES_DIR / "maint_features.parquet"

# step 8 dynamic features (citer-quality + citation shape)
DYN_FEATURES_35_FILE  = FEATURES_DIR / "dynamic_t35.parquet"
DYN_FEATURES_75_FILE  = FEATURES_DIR / "dynamic_t75.parquet"
DYN_FEATURES_115_FILE = FEATURES_DIR / "dynamic_t115.parquet"

# step 10 abstract-embedding-derived features (cutoff-independent)
ABSTRACT_DYN_FILE = FEATURES_DIR / "abstract_dynamic.parquet"

# ── 연구 파라미터 ─────────────────────────────────────
GRANT_YEAR_START = 1980
GRANT_YEAR_END   = 1989
DECISION_POINTS  = [3.5, 7.5, 11.5]
PATENT_TERM_YRS  = 17

# ── Train / Val / Test Split (전 실험 고정) ───────────
TRAIN_YEARS = list(range(1982, 1987))   # 1982~1986 (5개년)
VAL_YEARS   = [1987]                     # 1987       (1개년)
TEST_YEARS  = list(range(1988, 1990))   # 1988~1989  (2개년)

# ── PSB 임계값 — 3가지 실험 조건 ─────────────────────
#   0.001 = subclass 내 상위 0.1%
#   0.005 = subclass 내 상위 0.5%
#   0.010 = subclass 내 상위 1.0%
PSB_THRESHOLDS = [0.001, 0.005, 0.010]

# ── BI 고정 파라미터 ──────────────────────────────────
GAMMA      = 1.0
MAINT_COST = {3.5: 1.0, 7.5: 1.88, 11.5: 3.85}
PSB_REWARD = 8.0
PSB_WEIGHT = 300.0

# ── XGBoost 파라미터 (전 실험 공통) ──────────────────
XGB_PARAMS = dict(
    n_estimators     = 500,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
    tree_method      = "hist",
)

# ── 정적 Feature 컬럼 (전 실험 공통) ─────────────────
STATIC_COLS = [
    "num_claims", "filing_to_grant_days", "small_entity",
    "num_figures", "num_sheets", "inventor_count",
    "bwd_us_total", "bwd_examiner", "bwd_applicant",
    "bwd_foreign", "bwd_total", "bwd_examiner_ratio",
    "is_organization", "is_us_company", "is_foreign", "is_individual",
    "ipc_section_enc", "ipc_class_enc", "ipc_subclass_enc",
]

# ── Maintenance Fee Feature 컬럼 (시점별로 사용 가능 여부 다름) ──
# 3.5yr 모델  : 사용 불가 (미래 정보)
# 7.5yr 모델  : paid_3_5
# 11.5yr 모델 : paid_3_5, paid_7_5
MAINT_COLS_BY_CUTOFF = {
    3.5:  [],
    7.5:  ["paid_3_5"],
    11.5: ["paid_3_5", "paid_7_5"],
}


# ── Dynamic feature 컬럼 헬퍼 (step8 산출물) ─────────
# step8 은 각 cutoff t 별로 prefix "tXX__dyn_" 를 붙여 저장한다.
# get_dynamic_cols(cutoff) 가 시점에 맞는 컬럼명 리스트를 반환.
DYNAMIC_FEATURE_BASE = [
    # F17~F22 citation shape
    "cite_burstiness",
    "cite_gini",
    "cite_first_nonzero_age",
    "cite_longest_zero_run",
    "cite_resurgence_count",
    "cite_ipc_percentile",
    # F01~F10 citer-quality
    "n_citers",
    "n_citers_log",
    "citer_recency_mean",
    "citer_ipc_cross_rate",
    "citer_ipc_diversity",
    "citer_assignee_hhi",
    "citer_assignee_unique_rate",
    "citer_self_rate",
    "citer_loc_diversity",
    "citer_examiner_rate",
    "cross_ipc_slope",
]


ABSTRACT_FEATURE_COLS = [
    "abs_ipc_centroid_dist",
    "abs_cross_ipc_nn_dist",
    "abs_centroid_margin",
    "abs_density_in_ipc",
    "abs_outlier_score",
    "abs_kmeans_topic_id",
    "abs_kmeans_topic_dist",
]


def get_dynamic_cols(cutoff: float) -> list:
    """주어진 cutoff 시점의 dynamic feature 컬럼명 리스트."""
    tag = f"t{str(cutoff).replace('.', '')}"
    return [f"{tag}__dyn_{c}" for c in DYNAMIC_FEATURE_BASE]


def get_abstract_cols() -> list:
    """abstract embedding 기반 정적 피처 (cutoff 무관)."""
    return list(ABSTRACT_FEATURE_COLS)


def dynamic_features_file(cutoff: float):
    return {
        3.5:  DYN_FEATURES_35_FILE,
        7.5:  DYN_FEATURES_75_FILE,
        11.5: DYN_FEATURES_115_FILE,
    }[cutoff]

# ── Threshold별 경로 헬퍼 ────────────────────────────
def thr_tag(thr: float) -> str:
    """0.001 → 'thr001', 0.005 → 'thr005', 0.010 → 'thr010'"""
    return f"thr{int(round(thr * 1000)):03d}"

def labels_file(thr: float) -> Path:
    """threshold별 라벨 파일 경로"""
    return FEATURES_DIR / f"labels_{thr_tag(thr)}.parquet"

def models_dir(thr: float) -> Path:
    """threshold별 모델 저장 디렉토리"""
    d = ROOT / "models" / thr_tag(thr)
    d.mkdir(parents=True, exist_ok=True)
    return d

def results_dir(thr: float) -> Path:
    """threshold별 결과 저장 디렉토리"""
    d = ROOT / "results" / thr_tag(thr)
    d.mkdir(parents=True, exist_ok=True)
    return d
