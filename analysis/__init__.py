"""ESG 중대성 평가 분석 모듈."""

from __future__ import annotations

from .materiality import run_materiality_assessment
from .media_weights import classify_media, get_media_weight
from .scoring import calculate_impact_materiality, calculate_issue_exposure
from .sentiment import analyze_sentiment
from .visualize import plot_materiality_matrix

# ML 모듈 (선택적 import)
try:
    from .embeddings import KeywordEmbedder, load_ml_matched_results
    from .ml_sentiment import SentimentAnalyzer, load_ml_sentiment_results
    from .visualize_extended import (
        plot_materiality_matrix_interactive,
        plot_sentiment_distribution,
        plot_time_series,
        plot_matching_comparison,
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

__all__ = [
    "run_materiality_assessment",
    "classify_media",
    "get_media_weight",
    "calculate_issue_exposure",
    "calculate_impact_materiality",
    "analyze_sentiment",
    "plot_materiality_matrix",
    # ML 모듈
    "KeywordEmbedder",
    "load_ml_matched_results",
    "SentimentAnalyzer",
    "load_ml_sentiment_results",
    "plot_materiality_matrix_interactive",
    "plot_sentiment_distribution",
    "plot_time_series",
    "plot_matching_comparison",
    "ML_AVAILABLE",
]
