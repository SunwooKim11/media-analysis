"""중대성 점수 계산 모듈."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .media_weights import calculate_exposure_score, get_media_weight
from .sentiment import analyze_sentiment

if TYPE_CHECKING:
    pass


def _get_sentiment_value(row: dict, use_ml: bool = False) -> int:
    """감성 분석 값 반환 (ML 또는 규칙 기반).

    Parameters
    ----------
    row : dict
        데이터 행.
    use_ml : bool
        ML 결과 사용 여부.

    Returns
    -------
    int
        감성 점수 (+1, 0, -1).
    """
    if use_ml:
        # ML 결과 컬럼 우선 사용
        if "sentiment_score_ml" in row:
            try:
                return int(row["sentiment_score_ml"])
            except (ValueError, TypeError):
                pass
    return 0


def _safe_int(value, default: int = 1) -> int:
    """안전한 정수 변환 (날짜/문자열 등 무효한 값 처리)."""
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 산업 리스크 키워드 → ESG 카테고리 (H/E/S/G)
KEYWORD_ESG_MAP: dict[str, str] = {
    "대학 지역사회 협력": "H",  # 대학경영
    "대학 ESG 교육": "H",       # 대학경영
    "그린캠퍼스": "E",           # 환경
    "대학 폐기물": "E",          # 환경
    "대학 인권": "S",            # 사회
    "대학 안전사고": "S",        # 사회
    "대학 청렴도": "G",          # 지배구조
    "대학 ESG 경영": "G",        # 지배구조
}


def load_keyword_categories(keyword_file: Path | str | None = None) -> dict[str, str]:
    """keyword.xlsx에서 진단항목별 ESG 카테고리 매핑.

    Returns
    -------
    dict[str, str]
        {진단항목: ESG영역} 형태 (E/S/G/HEM).
    """
    if keyword_file is None:
        keyword_file = PROJECT_ROOT / "keyword.xlsx"

    keyword_file = Path(keyword_file)
    if not keyword_file.exists():
        return {}

    df = pd.read_excel(keyword_file, engine="openpyxl")

    category_map = {}
    current_area = None  # 영역은 상위 행에서 상속

    for _, row in df.iterrows():
        # 영역 업데이트 (비어있지 않으면)
        area = row.get("영역", "")
        if area and not pd.isna(area):
            current_area = str(area).strip()

        item = row.get("진단항목", "")
        if item and not pd.isna(item):
            # 영역 → H/E/S/G 매핑
            if current_area:
                area_lower = current_area.lower()
                if "환경" in area_lower:
                    category_map[str(item)] = "E"
                elif "사회" in area_lower:
                    category_map[str(item)] = "S"
                elif "지배구조" in area_lower:
                    category_map[str(item)] = "G"
                else:
                    category_map[str(item)] = "H"  # 대학경영
            else:
                category_map[str(item)] = "H"

    return category_map


def calculate_issue_exposure(
    df: pd.DataFrame,
    group_column: str = "keyword",
    title_column: str = "title",
    content_column: str = "content",
    press_count_column: str = "press_count",
    press_list_column: str = "press_list",
    negative_weight: float = 1.5,
    use_ml: bool = False,
) -> pd.DataFrame:
    """이슈 노출도 (Issue Exposure) 점수 계산.

    대학 산업 전반의 ESG 이슈 트렌드 = 산업 리스크 데이터 기반.
    부정 기사는 리스크 가중치 1.5배 적용.

    Parameters
    ----------
    df : pd.DataFrame
        산업 리스크 데이터.
    group_column : str
        그룹핑 컬럼 (키워드).
    title_column : str
        제목 컬럼.
    content_column : str
        내용 컬럼.
    press_count_column : str
        언론사 수 컬럼.
    press_list_column : str
        언론사 목록 컬럼.
    negative_weight : float
        부정 기사 가중치 (기본: 1.5).
    use_ml : bool
        ML 감성 분석 결과 사용 여부 (기본: False).

    Returns
    -------
    pd.DataFrame
        키워드별 이슈 노출도 점수 (0~5).
    """
    df = df.copy()

    # ML 결과 사용 시 sentiment_score_ml 컬럼 확인
    has_ml_sentiment = use_ml and "sentiment_score_ml" in df.columns

    # 카테고리가 비어있는 행 제외
    if group_column in df.columns:
        df = df[df[group_column].astype(str).str.strip() != ""]

    if df.empty:
        return pd.DataFrame(columns=["indicator", "issue_exposure"])

    # 다중 카테고리 분리 (콤마로 구분된 경우)
    rows = []
    for _, row in df.iterrows():
        categories = str(row.get(group_column, "")).split(", ")
        for cat in categories:
            cat = cat.strip()
            if cat:
                new_row = row.to_dict()
                new_row["_single_category"] = cat
                rows.append(new_row)

    if not rows:
        return pd.DataFrame(columns=["indicator", "issue_exposure"])

    df = pd.DataFrame(rows)

    # 감성 분석
    if has_ml_sentiment:
        # ML 결과 사용
        df["sentiment"] = df["sentiment_score_ml"].apply(
            lambda x: int(x) if pd.notna(x) and x != "" else 0
        )
    else:
        # 규칙 기반 감성 분석
        text_cols = [c for c in [title_column, content_column] if c in df.columns]
        df["_combined_text"] = df.apply(
            lambda row: " ".join(str(row.get(col, "")) for col in text_cols),
            axis=1,
        )
        df["sentiment"] = df["_combined_text"].apply(analyze_sentiment)

    # 미디어 노출도 계산
    if press_count_column in df.columns and press_list_column in df.columns:
        df["exposure"] = df.apply(
            lambda row: calculate_exposure_score(
                _safe_int(row.get(press_count_column, 1), 1),
                str(row.get(press_list_column, "")),
            ),
            axis=1,
        )
    else:
        df["exposure"] = 1.0

    # 부정 기사 가중치 적용
    df["risk_weight"] = df["sentiment"].apply(
        lambda s: negative_weight if s < 0 else 1.0
    )
    df["weighted_exposure"] = df["exposure"] * df["risk_weight"]

    # 그룹별 집계
    grouped = df.groupby("_single_category").agg(
        article_count=("sentiment", "count"),
        positive_count=("sentiment", lambda x: (x > 0).sum()),
        neutral_count=("sentiment", lambda x: (x == 0).sum()),
        negative_count=("sentiment", lambda x: (x < 0).sum()),
        total_exposure=("exposure", "sum"),
        weighted_exposure=("weighted_exposure", "sum"),
    ).reset_index()

    # 0~5 정규화 (Min-Max)
    max_val = grouped["weighted_exposure"].max()
    if max_val > 0:
        grouped["issue_exposure"] = (grouped["weighted_exposure"] / max_val) * 5
    else:
        grouped["issue_exposure"] = 0

    grouped = grouped.rename(columns={"_single_category": "indicator"})

    return grouped


def calculate_impact_materiality(
    df: pd.DataFrame,
    group_column: str = "matched_categories",
    title_column: str = "title",
    content_column: str = "content",
    press_count_column: str = "press_count",
    press_list_column: str = "press_list",
    positive_weight: float = 1.3,
    use_ml: bool = False,
) -> pd.DataFrame:
    """영향 중대성 (Inside-Out) 점수 계산.

    대학이 외부에 미치는 영향 = 국민대 미디어 보도 기반.
    긍정 기사는 영향력 가중치 1.3배 적용.

    Parameters
    ----------
    df : pd.DataFrame
        국민대 미디어 데이터.
    group_column : str
        그룹핑 컬럼.
    title_column : str
        제목 컬럼.
    content_column : str
        내용 컬럼.
    press_count_column : str
        언론사 수 컬럼.
    press_list_column : str
        언론사 목록 컬럼.
    positive_weight : float
        긍정 기사 가중치 (기본: 1.3).
    use_ml : bool
        ML 감성 분석 결과 사용 여부 (기본: False).

    Returns
    -------
    pd.DataFrame
        진단항목별 영향 중대성 점수.
    """
    df = df.copy()

    # ML 결과 사용 시 sentiment_score_ml 컬럼 확인
    has_ml_sentiment = use_ml and "sentiment_score_ml" in df.columns

    # 카테고리가 비어있는 행 제외
    if group_column in df.columns:
        df = df[df[group_column].astype(str).str.strip() != ""]

    if df.empty:
        return pd.DataFrame(columns=["indicator", "impact_materiality"])

    # 다중 카테고리 분리 (콤마로 구분된 경우)
    rows = []
    for _, row in df.iterrows():
        categories = str(row.get(group_column, "")).split(", ")
        for cat in categories:
            cat = cat.strip()
            if cat:
                new_row = row.to_dict()
                new_row["_single_category"] = cat
                rows.append(new_row)

    if not rows:
        return pd.DataFrame(columns=["indicator", "impact_materiality"])

    expanded_df = pd.DataFrame(rows)

    # 감성 분석
    if has_ml_sentiment:
        # ML 결과 사용
        expanded_df["sentiment"] = expanded_df["sentiment_score_ml"].apply(
            lambda x: int(x) if pd.notna(x) and x != "" else 0
        )
    else:
        # 규칙 기반 감성 분석
        text_cols = [c for c in [title_column, content_column] if c in expanded_df.columns]
        expanded_df["_combined_text"] = expanded_df.apply(
            lambda row: " ".join(str(row.get(col, "")) for col in text_cols),
            axis=1,
        )
        expanded_df["sentiment"] = expanded_df["_combined_text"].apply(analyze_sentiment)

    # 미디어 노출도 계산
    if press_count_column in expanded_df.columns and press_list_column in expanded_df.columns:
        expanded_df["exposure"] = expanded_df.apply(
            lambda row: calculate_exposure_score(
                _safe_int(row.get(press_count_column, 1), 1),
                str(row.get(press_list_column, "")),
            ),
            axis=1,
        )
    else:
        # press 컬럼이 있으면 단일 매체 가중치 사용
        if "press" in expanded_df.columns:
            expanded_df["exposure"] = expanded_df["press"].apply(
                lambda p: get_media_weight(str(p))
            )
        else:
            expanded_df["exposure"] = 1.0

    # 긍정 기사 가중치 적용
    expanded_df["impact_weight"] = expanded_df["sentiment"].apply(
        lambda s: positive_weight if s > 0 else 1.0
    )
    expanded_df["weighted_exposure"] = expanded_df["exposure"] * expanded_df["impact_weight"]

    # 그룹별 집계
    grouped = expanded_df.groupby("_single_category").agg(
        article_count=("sentiment", "count"),
        positive_count=("sentiment", lambda x: (x > 0).sum()),
        neutral_count=("sentiment", lambda x: (x == 0).sum()),
        negative_count=("sentiment", lambda x: (x < 0).sum()),
        total_exposure=("exposure", "sum"),
        weighted_exposure=("weighted_exposure", "sum"),
    ).reset_index()

    # 0~5 정규화 (Min-Max)
    max_val = grouped["weighted_exposure"].max()
    if max_val > 0:
        grouped["impact_materiality"] = (grouped["weighted_exposure"] / max_val) * 5
    else:
        grouped["impact_materiality"] = 0

    grouped = grouped.rename(columns={"_single_category": "indicator"})

    return grouped


def merge_materiality_scores(
    issue_df: pd.DataFrame,
    impact_df: pd.DataFrame,
) -> pd.DataFrame:
    """이슈 노출도와 영향 중대성 점수 병합.

    Parameters
    ----------
    issue_df : pd.DataFrame
        이슈 노출도 점수.
    impact_df : pd.DataFrame
        영향 중대성 점수.

    Returns
    -------
    pd.DataFrame
        병합된 중대성 점수.
    """
    # 전체 지표 목록
    all_indicators = set(issue_df["indicator"].tolist()) | set(impact_df["indicator"].tolist())

    result = []
    for indicator in all_indicators:
        iss_row = issue_df[issue_df["indicator"] == indicator]
        imp_row = impact_df[impact_df["indicator"] == indicator]

        iss_score = iss_row["issue_exposure"].values[0] if len(iss_row) > 0 else 0
        imp_score = imp_row["impact_materiality"].values[0] if len(imp_row) > 0 else 0

        iss_exposure = iss_row["total_exposure"].values[0] if len(iss_row) > 0 else 0
        imp_exposure = imp_row["total_exposure"].values[0] if len(imp_row) > 0 else 0

        iss_articles = int(iss_row["article_count"].values[0]) if len(iss_row) > 0 else 0
        imp_articles = int(imp_row["article_count"].values[0]) if len(imp_row) > 0 else 0

        result.append({
            "indicator": indicator,
            "issue_exposure": round(iss_score, 2),
            "impact_materiality": round(imp_score, 2),
            "total_exposure": round(iss_exposure + imp_exposure, 2),
            "article_count": iss_articles + imp_articles,
        })

    result_df = pd.DataFrame(result)

    # 정렬: 복합 점수 기준 (이슈 노출도 + 영향 중대성)
    result_df["composite_score"] = result_df["issue_exposure"] + result_df["impact_materiality"]
    result_df = result_df.sort_values("composite_score", ascending=False)
    result_df = result_df.drop(columns=["composite_score"])

    return result_df.reset_index(drop=True)
