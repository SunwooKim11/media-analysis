"""ML 분석 설정 모듈."""

from __future__ import annotations

from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 데이터 경로
DATA_DIR = PROJECT_ROOT / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_ANALYSIS_DIR = DATA_DIR / "analysis"
DATA_CACHE_DIR = DATA_DIR / "cache"

# 키워드 파일
KEYWORD_FILE = PROJECT_ROOT / "keyword.xlsx"

# ============================================================
# S-BERT 설정 (키워드 유사도)
# ============================================================
MODEL_SBERT = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
MODEL_SBERT_FALLBACK = "jhgan/ko-sbert-sts"  # 경량 대안

# 유사도 임계값 (0.0 ~ 1.0)
# 높을수록 엄격한 매칭 (권장: 0.5 ~ 0.7)
SIMILARITY_THRESHOLD = 0.6

# ============================================================
# KcBERT 설정 (감성 분석)
# ============================================================
MODEL_SENTIMENT = "beomi/kcbert-base"
MODEL_SENTIMENT_FALLBACK = "klue/roberta-base"  # 대안

# 감성 라벨 매핑
SENTIMENT_LABELS = {
    "긍정": 1,
    "중립": 0,
    "부정": -1,
}

# ============================================================
# 배치 처리 설정
# ============================================================
# GPU 환경 (Colab T4/V100)
GPU_BATCH_SIZE = 128

# CPU 환경
CPU_BATCH_SIZE = 32

# 기본 배치 크기 (자동 감지 전)
DEFAULT_BATCH_SIZE = 64

# 텍스트 최대 토큰 수
MAX_LENGTH = 512

# 파일 단위 처리 청크
CHUNK_SIZE = 5000

# ============================================================
# 캐시 파일명
# ============================================================
KEYWORD_EMBEDDINGS_CACHE = "keyword_embeddings.pt"
ARTICLE_EMBEDDINGS_DIR = "article_embeddings"
SENTIMENT_RESULTS_DIR = "sentiment_results"

# ============================================================
# 출력 파일명
# ============================================================
OUTPUT_SENTIMENT_ML = "sentiment_ml.csv"
OUTPUT_MATCHED_ML = "matched_categories_ml.csv"
OUTPUT_SCORES_ML = "materiality_scores_ml.csv"


def get_device() -> str:
    """사용 가능한 디바이스 반환 (cuda/cpu)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_batch_size(device: str | None = None) -> int:
    """디바이스에 따른 배치 크기 반환."""
    if device is None:
        device = get_device()
    return GPU_BATCH_SIZE if device == "cuda" else CPU_BATCH_SIZE
