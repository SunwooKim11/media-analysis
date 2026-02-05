"""감성 분석 모듈."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# 긍정 키워드 패턴
POSITIVE_PATTERNS: list[str] = [
    r"선정",
    r"달성",
    r"우수",
    r"혁신",
    r"성과",
    r"수상",
    r"인증",
    r"획득",
    r"1위",
    r"최초",
    r"선도",
    r"모범",
    r"우량",
    r"협약",
    r"협력",
    r"지원",
    r"발전",
    r"성장",
    r"증가",
    r"개선",
    r"강화",
    r"확대",
    r"도입",
    r"추진",
    r"출범",
    r"설립",
    r"개최",
    r"참여",
    r"기여",
    r"공헌",
]

# 부정 키워드 패턴
NEGATIVE_PATTERNS: list[str] = [
    r"미흡",
    r"지적",
    r"논란",
    r"사고",
    r"위반",
    r"제재",
    r"갈등",
    r"비판",
    r"의혹",
    r"조사",
    r"수사",
    r"고발",
    r"소송",
    r"파업",
    r"반발",
    r"항의",
    r"시위",
    r"불만",
    r"문제",
    r"우려",
    r"리스크",
    r"위기",
    r"감소",
    r"하락",
    r"축소",
    r"중단",
    r"폐지",
    r"철회",
    r"실패",
    r"부실",
]


def analyze_sentiment(text: str) -> int:
    """텍스트의 감성 분석.

    Parameters
    ----------
    text : str
        분석할 텍스트 (제목 + 내용).

    Returns
    -------
    int
        감성 점수 (+1: 긍정, 0: 중립, -1: 부정).
    """
    if not text:
        return 0

    text_lower = text.lower()

    positive_count = 0
    negative_count = 0

    for pattern in POSITIVE_PATTERNS:
        if re.search(pattern, text_lower):
            positive_count += 1

    for pattern in NEGATIVE_PATTERNS:
        if re.search(pattern, text_lower):
            negative_count += 1

    # 긍정/부정 키워드 수 비교
    if positive_count > negative_count:
        return 1
    elif negative_count > positive_count:
        return -1
    else:
        return 0


def analyze_sentiment_df(
    df: "pd.DataFrame",
    text_columns: list[str] | None = None,
) -> "pd.DataFrame":
    """DataFrame에 감성 분석 결과 추가.

    Parameters
    ----------
    df : pd.DataFrame
        분석할 DataFrame.
    text_columns : list[str], optional
        분석할 텍스트 컬럼 목록. 기본값은 ['title', 'content'].

    Returns
    -------
    pd.DataFrame
        sentiment 컬럼이 추가된 DataFrame.
    """
    if text_columns is None:
        text_columns = ["title", "content"]

    text_columns = [c for c in text_columns if c in df.columns]

    df = df.copy()

    def get_combined_text(row):
        return " ".join(str(row.get(col, "")) for col in text_columns)

    df["sentiment"] = df.apply(lambda row: analyze_sentiment(get_combined_text(row)), axis=1)

    return df


def get_sentiment_label(sentiment: int) -> str:
    """감성 점수를 레이블로 변환.

    Parameters
    ----------
    sentiment : int
        감성 점수.

    Returns
    -------
    str
        감성 레이블 (긍정/중립/부정).
    """
    if sentiment > 0:
        return "긍정"
    elif sentiment < 0:
        return "부정"
    else:
        return "중립"
