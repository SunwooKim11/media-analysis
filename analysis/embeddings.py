"""S-BERT 기반 키워드 임베딩 및 유사도 계산 모듈.

GPU 환경에서는 실시간 계산, CPU 환경에서는 Colab 결과 CSV 로드.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .config import (
    DATA_ANALYSIS_DIR,
    KEYWORD_FILE,
    MODEL_SBERT,
    SIMILARITY_THRESHOLD,
    get_batch_size,
    get_device,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def load_ml_matched_results(
    source: str = "all",
    analysis_dir: Path | None = None,
) -> pd.DataFrame:
    """Colab에서 생성한 ML 매칭 결과 CSV 로드.

    Parameters
    ----------
    source : str
        데이터 소스 ("industry_risk", "kookmin_media", "kookmin_press", "all").
    analysis_dir : Path, optional
        분석 결과 디렉토리.

    Returns
    -------
    pd.DataFrame
        ML 매칭 결과.
    """
    if analysis_dir is None:
        analysis_dir = DATA_ANALYSIS_DIR

    if source == "all":
        # 통합 파일 로드
        matched_file = analysis_dir / "matched_categories_ml.csv"
        if matched_file.exists():
            df = pd.read_csv(matched_file, encoding="utf-8-sig", dtype=str)
            df.fillna("", inplace=True)
            logger.info("ML 매칭 결과 로드: %d건", len(df))
            return df
        else:
            logger.warning("ML 매칭 결과 파일 없음: %s", matched_file)
            return pd.DataFrame()
    else:
        # 개별 파일 로드
        file_path = analysis_dir / f"{source}_ml.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, encoding="utf-8-sig", dtype=str)
            df.fillna("", inplace=True)
            logger.info("%s ML 결과 로드: %d건", source, len(df))
            return df
        else:
            logger.warning("파일 없음: %s", file_path)
            return pd.DataFrame()


def merge_ml_results_to_df(
    df: pd.DataFrame,
    ml_df: pd.DataFrame,
    key_column: str = "title",
) -> pd.DataFrame:
    """원본 DataFrame에 ML 결과 병합.

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터.
    ml_df : pd.DataFrame
        ML 분석 결과.
    key_column : str
        조인 키 컬럼.

    Returns
    -------
    pd.DataFrame
        ML 결과가 병합된 DataFrame.
    """
    if ml_df.empty:
        logger.warning("ML 결과가 비어있음")
        return df

    # ML 결과 컬럼
    ml_columns = [
        "matched_categories_ml",
        "matched_scores_ml",
        "sentiment_label_ml",
        "sentiment_score_ml",
        "sentiment_confidence_ml",
    ]

    # 존재하는 컬럼만 선택
    available_cols = [c for c in ml_columns if c in ml_df.columns]
    if not available_cols:
        logger.warning("ML 결과 컬럼 없음")
        return df

    # 키 컬럼 + ML 컬럼만 선택하여 병합
    merge_cols = [key_column] + available_cols
    ml_subset = ml_df[[c for c in merge_cols if c in ml_df.columns]].drop_duplicates(
        subset=[key_column], keep="first"
    )

    # 병합
    result = df.merge(ml_subset, on=key_column, how="left", suffixes=("", "_ml_new"))

    # 병합된 컬럼 정리
    for col in available_cols:
        if col not in result.columns and f"{col}_ml_new" in result.columns:
            result[col] = result[f"{col}_ml_new"]
            result.drop(columns=[f"{col}_ml_new"], inplace=True)

    result.fillna("", inplace=True)
    logger.info("ML 결과 병합 완료: %d건", len(result))

    return result


class KeywordEmbedder:
    """S-BERT 기반 키워드 임베딩 및 유사도 계산.

    GPU 환경에서만 실시간 계산 지원.
    CPU 환경에서는 Colab 결과 CSV 사용 권장.
    """

    def __init__(
        self,
        keyword_file: Path | str | None = None,
        model_name: str = MODEL_SBERT,
        device: str | None = None,
        threshold: float = SIMILARITY_THRESHOLD,
    ):
        """초기화.

        Parameters
        ----------
        keyword_file : Path, optional
            키워드 파일 경로.
        model_name : str
            S-BERT 모델명.
        device : str, optional
            디바이스 ("cuda" or "cpu").
        threshold : float
            유사도 임계값.
        """
        self.keyword_file = Path(keyword_file) if keyword_file else KEYWORD_FILE
        self.model_name = model_name
        self.device = device or get_device()
        self.threshold = threshold

        self._model = None
        self._keyword_embeddings = None
        self._keyword_mapping = None

    def _load_model(self):
        """모델 로드 (지연 로딩)."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("S-BERT 모델 로드: %s (device=%s)", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("S-BERT 모델 로드 완료")
        except ImportError:
            logger.error("sentence-transformers 패키지가 설치되지 않았습니다.")
            logger.error("GPU 환경(Colab)에서 실행하거나 pip install sentence-transformers")
            raise

    def _load_keywords(self) -> dict[str, list[str]]:
        """키워드 매핑 로드."""
        if self._keyword_mapping is not None:
            return self._keyword_mapping

        if not self.keyword_file.exists():
            logger.error("키워드 파일 없음: %s", self.keyword_file)
            return {}

        df = pd.read_excel(self.keyword_file, engine="openpyxl")

        mapping = {}
        for _, row in df.iterrows():
            item = row.get("진단항목", "")
            if not item or pd.isna(item):
                continue

            keywords = []
            for col in ["키워드 1", "키워드 2", "키워드 3", "키워드 4", "키워드 5"]:
                kw = row.get(col, "")
                if kw and not pd.isna(kw):
                    keywords.append(str(kw).strip())

            if keywords:
                mapping[str(item)] = keywords

        self._keyword_mapping = mapping
        logger.info("키워드 매핑 로드: %d개 진단항목", len(mapping))
        return mapping

    def embed_keywords(self) -> dict:
        """키워드 임베딩 계산.

        Returns
        -------
        dict
            {진단항목: embedding_tensor}
        """
        self._load_model()
        keyword_mapping = self._load_keywords()

        if self._keyword_embeddings is not None:
            return self._keyword_embeddings

        from tqdm import tqdm

        embeddings = {}
        for indicator, keywords in tqdm(keyword_mapping.items(), desc="Embedding keywords"):
            kw_embeddings = self._model.encode(
                keywords, convert_to_tensor=True, device=self.device
            )
            embeddings[indicator] = kw_embeddings

        self._keyword_embeddings = embeddings
        logger.info("키워드 임베딩 계산 완료: %d개", len(embeddings))
        return embeddings

    def match_article(self, text: str) -> list[tuple[str, float]]:
        """단일 기사 매칭.

        Parameters
        ----------
        text : str
            기사 텍스트.

        Returns
        -------
        list[tuple[str, float]]
            [(진단항목, 유사도), ...]
        """
        if not text or len(text.strip()) < 10:
            return []

        self._load_model()
        embeddings = self.embed_keywords()

        from sentence_transformers import util

        # 텍스트 임베딩
        text_embedding = self._model.encode(
            text[:2000], convert_to_tensor=True, device=self.device
        )

        matches = []
        for indicator, kw_embeddings in embeddings.items():
            similarities = util.pytorch_cos_sim(text_embedding, kw_embeddings)
            max_sim = similarities.max().item()

            if max_sim >= self.threshold:
                matches.append((indicator, round(max_sim, 3)))

        matches.sort(key=lambda x: -x[1])
        return matches

    def match_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[tuple[str, float]]]:
        """배치 기사 매칭.

        Parameters
        ----------
        texts : list[str]
            기사 텍스트 리스트.
        batch_size : int, optional
            배치 크기.

        Returns
        -------
        list[list[tuple[str, float]]]
            각 기사의 매칭 결과.
        """
        if batch_size is None:
            batch_size = get_batch_size(self.device)

        from tqdm import tqdm

        results = []
        for text in tqdm(texts, desc="S-BERT matching"):
            matches = self.match_article(text)
            results.append(matches)

        return results
