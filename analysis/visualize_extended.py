"""확장 시각화 모듈 (Plotly 기반)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ESG 색상 팔레트
ESG_COLORS = {
    "H": "#e67e22",  # 대학경영: 주황
    "E": "#2ecc71",  # 환경: 녹색
    "S": "#3498db",  # 사회: 파랑
    "G": "#9b59b6",  # 지배구조: 보라
    "unknown": "#95a5a6",  # 기타: 회색
}

ESG_NAMES = {
    "H": "대학경영",
    "E": "환경",
    "S": "사회",
    "G": "지배구조",
}

# 감성 색상
SENTIMENT_COLORS = {
    "긍정": "#27ae60",  # 녹색
    "중립": "#95a5a6",  # 회색
    "부정": "#e74c3c",  # 빨강
}


def plot_materiality_matrix_interactive(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "ESG Materiality Matrix",
) -> "go.Figure":
    """인터랙티브 중대성 매트릭스 생성.

    Parameters
    ----------
    df : pd.DataFrame
        중대성 점수 데이터.
    output_path : str | Path, optional
        HTML 저장 경로.
    title : str
        차트 제목.

    Returns
    -------
    go.Figure
        Plotly Figure 객체.
    """
    if not PLOTLY_AVAILABLE:
        logger.error("plotly가 설치되지 않았습니다: pip install plotly")
        return None

    df = df.copy()

    # ESG 카테고리 색상
    df["color"] = df["esg_category"].map(lambda x: ESG_COLORS.get(x, ESG_COLORS["unknown"]))
    df["esg_name"] = df["esg_category"].map(lambda x: ESG_NAMES.get(x, "기타"))

    # 점 크기 (기사 수 기반)
    min_size = 15
    max_size = 60
    if df["article_count"].max() > 0:
        df["marker_size"] = min_size + (df["article_count"] / df["article_count"].max()) * (max_size - min_size)
    else:
        df["marker_size"] = min_size

    # 호버 텍스트
    df["hover_text"] = df.apply(
        lambda row: (
            f"<b>{row['indicator']}</b><br>"
            f"ESG: {row['esg_name']} ({row['esg_category']})<br>"
            f"이슈 노출도: {row['issue_exposure']:.2f}<br>"
            f"영향 중대성: {row['impact_materiality']:.2f}<br>"
            f"관련 기사: {int(row['article_count'])}건"
        ),
        axis=1,
    )

    # Figure 생성
    fig = go.Figure()

    # ESG 카테고리별 트레이스 추가
    for esg, esg_name in ESG_NAMES.items():
        subset = df[df["esg_category"] == esg]
        if subset.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=subset["issue_exposure"],
                y=subset["impact_materiality"],
                mode="markers+text",
                marker=dict(
                    size=subset["marker_size"],
                    color=ESG_COLORS[esg],
                    opacity=0.8,
                    line=dict(width=1, color="white"),
                ),
                text=subset["indicator"],
                textposition="top center",
                textfont=dict(size=10),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=subset["hover_text"],
                name=f"{esg_name} ({esg})",
            )
        )

    # 사분면 구분선
    fig.add_hline(y=2.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=2.5, line_dash="dash", line_color="gray", opacity=0.5)

    # 사분면 레이블
    annotations = [
        dict(x=4.0, y=4.5, text="Q1: 핵심 중대", showarrow=False, font=dict(size=12, color="gray")),
        dict(x=1.0, y=4.5, text="Q2: 영향 중심", showarrow=False, font=dict(size=12, color="gray")),
        dict(x=1.0, y=0.5, text="Q3: 관찰 필요", showarrow=False, font=dict(size=12, color="gray")),
        dict(x=4.0, y=0.5, text="Q4: 이슈 중심", showarrow=False, font=dict(size=12, color="gray")),
    ]

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(
            title="Issue Exposure (이슈 노출도)",
            range=[-0.2, 5.2],
            dtick=1,
            gridcolor="lightgray",
        ),
        yaxis=dict(
            title="Impact Materiality (영향 중대성)",
            range=[-0.2, 5.2],
            dtick=1,
            gridcolor="lightgray",
        ),
        legend=dict(
            title="ESG 영역",
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        annotations=annotations,
        template="plotly_white",
        width=900,
        height=700,
    )

    # HTML 저장
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info("인터랙티브 매트릭스 저장: %s", output_path)

    return fig


def plot_sentiment_distribution(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "진단항목별 감성 분포",
) -> "go.Figure":
    """진단항목별 감성 분포 Stacked Bar 차트.

    Parameters
    ----------
    df : pd.DataFrame
        감성 분석 결과 (matched_categories, sentiment_label 컬럼 필요).
    output_path : str | Path, optional
        HTML 저장 경로.
    title : str
        차트 제목.

    Returns
    -------
    go.Figure
        Plotly Figure 객체.
    """
    if not PLOTLY_AVAILABLE:
        logger.error("plotly가 설치되지 않았습니다")
        return None

    df = df.copy()

    # 감성 라벨 컬럼 확인
    sentiment_col = "sentiment_label_ml" if "sentiment_label_ml" in df.columns else "sentiment_label"
    category_col = "matched_categories_ml" if "matched_categories_ml" in df.columns else "matched_categories"

    if sentiment_col not in df.columns or category_col not in df.columns:
        logger.warning("감성 또는 카테고리 컬럼이 없습니다")
        return None

    # 카테고리별 감성 집계
    df = df[df[category_col].str.strip() != ""]

    # 다중 카테고리 분리
    rows = []
    for _, row in df.iterrows():
        categories = str(row[category_col]).split(", ")
        for cat in categories:
            cat = cat.strip()
            if cat:
                rows.append({"category": cat, "sentiment": row[sentiment_col]})

    if not rows:
        return None

    agg_df = pd.DataFrame(rows)
    pivot = agg_df.groupby(["category", "sentiment"]).size().unstack(fill_value=0)

    # 정렬 (총 건수 기준)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=True).drop(columns=["total"])

    # Stacked Bar 차트
    fig = go.Figure()

    for sentiment in ["긍정", "중립", "부정"]:
        if sentiment in pivot.columns:
            fig.add_trace(
                go.Bar(
                    name=sentiment,
                    y=pivot.index,
                    x=pivot[sentiment],
                    orientation="h",
                    marker_color=SENTIMENT_COLORS.get(sentiment, "gray"),
                )
            )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        barmode="stack",
        xaxis_title="기사 수",
        yaxis_title="진단항목",
        legend_title="감성",
        template="plotly_white",
        height=max(400, len(pivot) * 25),
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info("감성 분포 차트 저장: %s", output_path)

    return fig


def plot_time_series(
    df: pd.DataFrame,
    date_column: str = "date",
    output_path: str | Path | None = None,
    title: str = "월별 기사 수 및 감성 트렌드",
) -> "go.Figure":
    """시계열 트렌드 차트 (월별 기사 수 + 평균 감성).

    Parameters
    ----------
    df : pd.DataFrame
        기사 데이터 (date, sentiment 컬럼 필요).
    date_column : str
        날짜 컬럼명.
    output_path : str | Path, optional
        HTML 저장 경로.
    title : str
        차트 제목.

    Returns
    -------
    go.Figure
        Plotly Figure 객체.
    """
    if not PLOTLY_AVAILABLE:
        logger.error("plotly가 설치되지 않았습니다")
        return None

    df = df.copy()

    if date_column not in df.columns:
        logger.warning("날짜 컬럼이 없습니다: %s", date_column)
        return None

    # 날짜 파싱
    df["_date"] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=["_date"])
    df["_month"] = df["_date"].dt.to_period("M").astype(str)

    # 감성 점수 컬럼
    sentiment_col = "sentiment_score_ml" if "sentiment_score_ml" in df.columns else "sentiment"
    if sentiment_col in df.columns:
        df["_sentiment"] = pd.to_numeric(df[sentiment_col], errors="coerce").fillna(0)
    else:
        df["_sentiment"] = 0

    # 월별 집계
    monthly = df.groupby("_month").agg(
        article_count=("_date", "count"),
        avg_sentiment=("_sentiment", "mean"),
    ).reset_index()

    # Dual-axis 차트
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 기사 수 (막대)
    fig.add_trace(
        go.Bar(
            x=monthly["_month"],
            y=monthly["article_count"],
            name="기사 수",
            marker_color="#3498db",
            opacity=0.7,
        ),
        secondary_y=False,
    )

    # 평균 감성 (선)
    fig.add_trace(
        go.Scatter(
            x=monthly["_month"],
            y=monthly["avg_sentiment"],
            name="평균 감성",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="월",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=500,
    )

    fig.update_yaxes(title_text="기사 수", secondary_y=False)
    fig.update_yaxes(title_text="평균 감성 점수", secondary_y=True)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info("시계열 차트 저장: %s", output_path)

    return fig


def plot_matching_comparison(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "규칙 vs ML 매칭률 비교",
) -> "go.Figure":
    """규칙 기반 vs ML 매칭률 비교 차트.

    Parameters
    ----------
    df : pd.DataFrame
        매칭 결과 (matched_categories, matched_categories_ml 컬럼 필요).
    output_path : str | Path, optional
        HTML 저장 경로.
    title : str
        차트 제목.

    Returns
    -------
    go.Figure
        Plotly Figure 객체.
    """
    if not PLOTLY_AVAILABLE:
        logger.error("plotly가 설치되지 않았습니다")
        return None

    df = df.copy()

    rule_col = "matched_categories"
    ml_col = "matched_categories_ml"

    if rule_col not in df.columns or ml_col not in df.columns:
        logger.warning("매칭 컬럼이 없습니다")
        return None

    # 매칭 여부
    rule_matched = df[rule_col].str.strip() != ""
    ml_matched = df[ml_col].str.strip() != ""

    # 통계 계산
    total = len(df)
    rule_count = rule_matched.sum()
    ml_count = ml_matched.sum()
    both_count = (rule_matched & ml_matched).sum()
    rule_only = (rule_matched & ~ml_matched).sum()
    ml_only = (~rule_matched & ml_matched).sum()
    neither = (~rule_matched & ~ml_matched).sum()

    # Bar 차트
    categories = ["규칙 기반", "ML 기반", "둘 다 매칭", "규칙만", "ML만", "미매칭"]
    counts = [rule_count, ml_count, both_count, rule_only, ml_only, neither]
    percentages = [c / total * 100 for c in counts]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=categories,
            y=counts,
            text=[f"{c:,} ({p:.1f}%)" for c, p in zip(counts, percentages)],
            textposition="auto",
            marker_color=["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#e67e22", "#95a5a6"],
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="매칭 유형",
        yaxis_title="기사 수",
        template="plotly_white",
        height=500,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info("매칭률 비교 차트 저장: %s", output_path)

    return fig
