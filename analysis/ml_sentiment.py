"""KcBERT 기반 감성 분석 모듈.

GPU 환경에서는 실시간 분석, CPU 환경에서는 Colab 결과 CSV 로드.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .config import (
    DATA_ANALYSIS_DIR,
    MODEL_SENTIMENT,
    SENTIMENT_LABELS,
    get_batch_size,
    get_device,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """감성 분석 결과."""

    label: str  # "긍정", "부정", "중립"
    score: int  # 1, -1, 0 (기존 호환)
    confidence: float  # 0.0 ~ 1.0
    probabilities: dict[str, float] | None = None

    def to_dict(self) -> dict:
        """딕셔너리 변환."""
        return {
            "label": self.label,
            "score": self.score,
            "confidence": self.confidence,
        }


def load_ml_sentiment_results(
    source: str = "all",
    analysis_dir: Path | None = None,
) -> pd.DataFrame:
    """Colab에서 생성한 ML 감성 분석 결과 CSV 로드.

    Parameters
    ----------
    source : str
        데이터 소스 ("industry_risk", "kookmin_media", "kookmin_press", "all").
    analysis_dir : Path, optional
        분석 결과 디렉토리.

    Returns
    -------
    pd.DataFrame
        ML 감성 분석 결과.
    """
    if analysis_dir is None:
        analysis_dir = DATA_ANALYSIS_DIR

    if source == "all":
        # 통합 파일 로드
        sentiment_file = analysis_dir / "sentiment_ml.csv"
        if sentiment_file.exists():
            df = pd.read_csv(sentiment_file, encoding="utf-8-sig", dtype=str)
            df.fillna("", inplace=True)
            # score 컬럼 정수 변환
            if "sentiment_score_ml" in df.columns:
                df["sentiment_score_ml"] = pd.to_numeric(
                    df["sentiment_score_ml"], errors="coerce"
                ).fillna(0).astype(int)
            logger.info("ML 감성 분석 결과 로드: %d건", len(df))
            return df
        else:
            logger.warning("ML 감성 분석 결과 파일 없음: %s", sentiment_file)
            return pd.DataFrame()
    else:
        # 개별 파일 로드
        file_path = analysis_dir / f"{source}_ml.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, encoding="utf-8-sig", dtype=str)
            df.fillna("", inplace=True)
            if "sentiment_score_ml" in df.columns:
                df["sentiment_score_ml"] = pd.to_numeric(
                    df["sentiment_score_ml"], errors="coerce"
                ).fillna(0).astype(int)
            logger.info("%s ML 감성 결과 로드: %d건", source, len(df))
            return df
        else:
            logger.warning("파일 없음: %s", file_path)
            return pd.DataFrame()


def get_sentiment_from_ml_df(
    df: pd.DataFrame,
    title: str,
) -> SentimentResult | None:
    """ML DataFrame에서 특정 기사의 감성 결과 조회.

    Parameters
    ----------
    df : pd.DataFrame
        ML 감성 분석 결과.
    title : str
        기사 제목.

    Returns
    -------
    SentimentResult | None
        감성 분석 결과.
    """
    if df.empty or "sentiment_label_ml" not in df.columns:
        return None

    row = df[df["title"] == title]
    if row.empty:
        return None

    row = row.iloc[0]
    return SentimentResult(
        label=str(row.get("sentiment_label_ml", "중립")),
        score=int(row.get("sentiment_score_ml", 0)),
        confidence=float(row.get("sentiment_confidence_ml", 0.0)),
    )


class SentimentAnalyzer:
    """KcBERT 기반 감성 분석기.

    GPU 환경에서만 실시간 분석 지원.
    CPU 환경에서는 Colab 결과 CSV 사용 권장.
    """

    # 감성 라벨 매핑
    LABEL_MAP = {
        "긍정적인 뉴스": ("긍정", 1),
        "중립적인 뉴스": ("중립", 0),
        "부정적인 뉴스": ("부정", -1),
    }

    CANDIDATE_LABELS = ["긍정적인 뉴스", "중립적인 뉴스", "부정적인 뉴스"]

    def __init__(
        self,
        model_name: str = MODEL_SENTIMENT,
        device: str | None = None,
    ):
        """초기화.

        Parameters
        ----------
        model_name : str
            감성 분석 모델명.
        device : str, optional
            디바이스 ("cuda" or "cpu").
        """
        self.model_name = model_name
        self.device = device or get_device()
        self._classifier = None

    def _load_model(self):
        """모델 로드 (지연 로딩)."""
        if self._classifier is not None:
            return

        try:
            from transformers import pipeline

            # Zero-shot classification 사용 (다국어 NLI 모델)
            logger.info("감성 분석 모델 로드 (Zero-shot)")
            self._classifier = pipeline(
                "zero-shot-classification",
                model="MoritzLawornia/mDeBERTa-v3-base-mnli-xnli",
                device=0 if self.device == "cuda" else -1,
            )
            logger.info("감성 분석 모델 로드 완료")
        except ImportError:
            logger.error("transformers 패키지가 설치되지 않았습니다.")
            logger.error("GPU 환경(Colab)에서 실행하거나 pip install transformers")
            raise

    def analyze(self, text: str) -> SentimentResult:
        """단일 텍스트 감성 분석.

        Parameters
        ----------
        text : str
            분석할 텍스트.

        Returns
        -------
        SentimentResult
            감성 분석 결과.
        """
        if not text or len(text.strip()) < 10:
            return SentimentResult(label="중립", score=0, confidence=0.0)

        self._load_model()

        try:
            # 텍스트 길이 제한
            text = text[:512]

            result = self._classifier(
                text, self.CANDIDATE_LABELS, multi_label=False
            )

            top_label = result["labels"][0]
            confidence = result["scores"][0]

            label_kr, score = self.LABEL_MAP.get(top_label, ("중립", 0))

            # 확률 분포
            probabilities = {
                self.LABEL_MAP.get(label, (label, 0))[0]: round(prob, 3)
                for label, prob in zip(result["labels"], result["scores"])
            }

            return SentimentResult(
                label=label_kr,
                score=score,
                confidence=round(confidence, 3),
                probabilities=probabilities,
            )
        except Exception as e:
            logger.error("감성 분석 오류: %s", e)
            return SentimentResult(label="중립", score=0, confidence=0.0)

    def analyze_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[SentimentResult]:
        """배치 감성 분석.

        Parameters
        ----------
        texts : list[str]
            분석할 텍스트 리스트.
        batch_size : int, optional
            배치 크기 (현재 미사용, 순차 처리).

        Returns
        -------
        list[SentimentResult]
            감성 분석 결과 리스트.
        """
        from tqdm import tqdm

        results = []
        for text in tqdm(texts, desc="Sentiment analysis"):
            result = self.analyze(text)
            results.append(result)

        return results


def analyze_sentiment_ml(text: str, analyzer: SentimentAnalyzer | None = None) -> int:
    """ML 기반 감성 분석 (기존 인터페이스 호환).

    Parameters
    ----------
    text : str
        분석할 텍스트.
    analyzer : SentimentAnalyzer, optional
        분석기 인스턴스.

    Returns
    -------
    int
        감성 점수 (+1: 긍정, 0: 중립, -1: 부정).
    """
    if analyzer is None:
        analyzer = SentimentAnalyzer()

    result = analyzer.analyze(text)
    return result.score
