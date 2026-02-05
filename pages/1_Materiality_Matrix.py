"""ESG 중대성 매트릭스 페이지."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "analysis"

# ESG 색상 및 이름
ESG_COLORS = {"H": "#e67e22", "E": "#2ecc71", "S": "#3498db", "G": "#9b59b6"}
ESG_NAMES = {"H": "대학경영", "E": "환경", "S": "사회", "G": "지배구조"}

# 사전 정의된 23개 ESG 지표
PREDEFINED_INDICATORS = [
    "ESG 비교과 프로그램 개발 및 운영", "ESG 교육과정 및 교수학습 방법",
    "ESG 관련 캠페인", "ESG 관련 학생 동아리", "지역사회 ESG 파트너십",
    "지역사회 ESG 교육 프로그램", "탄소배출량", "에너지 사용 절감 및 체계적 관리",
    "캠퍼스 차량", "폐기물 배출", "유해 폐기물관리", "취약계층 지원강화",
    "다양성과 형평성 추진체계", "교육 안전관리", "인권경영", "연구윤리",
    "부패예방 및 청렴 강화", "회계의 투명성 관리", "ESG 위원회 조직",
    "ESG 운영위원회의 다양성", "대학평가", "대학 재무안정성", "ESG 경영평가",
]

st.set_page_config(page_title="Materiality Matrix", page_icon="📊", layout="wide")


@st.cache_data
def load_scores():
    """중대성 점수 로드 (23개 사전 정의 지표만)."""
    file_path = DATA_DIR / "materiality_scores_ml.csv"
    if file_path.exists():
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        # 사전 정의된 23개 지표만 필터링
        df = df[df["indicator"].isin(PREDEFINED_INDICATORS)].copy()
        return df
    return pd.DataFrame()


def plot_matrix(df: pd.DataFrame, highlight_q1: bool = True) -> go.Figure:
    """인터랙티브 중대성 매트릭스."""
    if df.empty:
        return go.Figure()

    # ESG 카테고리별 색상
    df["color"] = df["esg_category"].map(ESG_COLORS)
    df["esg_name"] = df["esg_category"].map(ESG_NAMES)

    fig = go.Figure()

    # 사분면 배경
    fig.add_shape(type="rect", x0=0, y0=2.5, x1=2.5, y1=5, fillcolor="rgba(255,193,7,0.1)", line_width=0)
    fig.add_shape(type="rect", x0=2.5, y0=2.5, x1=5, y1=5, fillcolor="rgba(220,53,69,0.1)", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=0, x1=2.5, y1=2.5, fillcolor="rgba(40,167,69,0.1)", line_width=0)
    fig.add_shape(type="rect", x0=2.5, y0=0, x1=5, y1=2.5, fillcolor="rgba(23,162,184,0.1)", line_width=0)

    # 사분면 라벨
    fig.add_annotation(x=1.25, y=4.7, text="Q2: 잠재 리스크", showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(x=3.75, y=4.7, text="Q1: 핵심 중대", showarrow=False, font=dict(size=12, color="red"))
    fig.add_annotation(x=1.25, y=0.3, text="Q3: 저위험", showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(x=3.75, y=0.3, text="Q4: 이슈 우선", showarrow=False, font=dict(size=12, color="gray"))

    # ESG 카테고리별 scatter
    for cat in ["E", "S", "G", "H"]:
        subset = df[df["esg_category"] == cat]
        if subset.empty:
            continue

        fig.add_trace(go.Scatter(
            x=subset["issue_exposure"],
            y=subset["impact_materiality"],
            mode="markers+text",
            marker=dict(size=12, color=ESG_COLORS.get(cat, "#999"), opacity=0.7),
            text=subset["indicator"],
            textposition="top center",
            textfont=dict(size=9),
            name=f"{ESG_NAMES.get(cat, cat)} ({cat})",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "이슈 노출도: %{x:.2f}<br>"
                "영향 중대성: %{y:.2f}<br>"
                "<extra></extra>"
            ),
        ))

    # Q1 강조
    if highlight_q1:
        q1 = df[(df["issue_exposure"] >= 2.5) & (df["impact_materiality"] >= 2.5)]
        if not q1.empty:
            fig.add_trace(go.Scatter(
                x=q1["issue_exposure"],
                y=q1["impact_materiality"],
                mode="markers",
                marker=dict(size=20, color="red", symbol="star", opacity=0.8),
                name="Q1 핵심 지표",
                hoverinfo="skip",
            ))

    fig.update_layout(
        title="ESG Materiality Matrix (Double Materiality)",
        xaxis_title="이슈 노출도 (Issue Exposure)",
        yaxis_title="영향 중대성 (Impact Materiality)",
        xaxis=dict(range=[0, 5.2], dtick=1),
        yaxis=dict(range=[0, 5.2], dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
    )

    return fig


def main():
    st.title("📊 ESG 중대성 매트릭스")
    st.markdown("**Double Materiality Framework**: 이슈 노출도 (X축) × 영향 중대성 (Y축)")

    # 데이터 로드
    df = load_scores()

    if df.empty:
        st.error("중대성 점수 데이터가 없습니다.")
        return

    # 필터
    st.sidebar.header("필터")
    selected_esg = st.sidebar.multiselect(
        "ESG 카테고리",
        options=["E", "S", "G", "H"],
        default=["E", "S", "G", "H"],
        format_func=lambda x: f"{ESG_NAMES.get(x, x)} ({x})",
    )

    highlight_q1 = st.sidebar.checkbox("Q1 강조 표시", value=True)
    show_labels = st.sidebar.checkbox("지표명 표시", value=True)

    # 필터 적용
    filtered_df = df[df["esg_category"].isin(selected_esg)]

    # 매트릭스 시각화
    fig = plot_matrix(filtered_df, highlight_q1=highlight_q1)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 사분면별 분석
    st.subheader("📋 사분면별 지표")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔴 Q1: 핵심 중대 (High-High)**")
        q1 = filtered_df[(filtered_df["issue_exposure"] >= 2.5) & (filtered_df["impact_materiality"] >= 2.5)]
        if not q1.empty:
            for _, row in q1.iterrows():
                st.markdown(f"- {row['indicator']} ({row['esg_category']})")
        else:
            st.info("해당 지표 없음")

        st.markdown("**🟢 Q3: 저위험 (Low-Low)**")
        q3 = filtered_df[(filtered_df["issue_exposure"] < 2.5) & (filtered_df["impact_materiality"] < 2.5)]
        st.caption(f"{len(q3)}개 지표")

    with col2:
        st.markdown("**🟡 Q2: 잠재 리스크 (Low Exposure, High Impact)**")
        q2 = filtered_df[(filtered_df["issue_exposure"] < 2.5) & (filtered_df["impact_materiality"] >= 2.5)]
        if not q2.empty:
            for _, row in q2.iterrows():
                st.markdown(f"- {row['indicator']} ({row['esg_category']})")
        else:
            st.info("해당 지표 없음")

        st.markdown("**🔵 Q4: 이슈 우선 (High Exposure, Low Impact)**")
        q4 = filtered_df[(filtered_df["issue_exposure"] >= 2.5) & (filtered_df["impact_materiality"] < 2.5)]
        if not q4.empty:
            for _, row in q4.iterrows():
                st.markdown(f"- {row['indicator']} ({row['esg_category']})")
        else:
            st.info("해당 지표 없음")

    st.divider()

    # Top 10 지표
    st.subheader("🏆 Top 10 중대 이슈")
    filtered_df["total_score"] = filtered_df["issue_exposure"] + filtered_df["impact_materiality"]
    top10 = filtered_df.nlargest(10, "total_score")[["indicator", "esg_category", "issue_exposure", "impact_materiality", "total_score", "article_count"]]
    top10.columns = ["지표", "ESG", "노출도", "중대성", "종합점수", "기사수"]
    st.dataframe(top10, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
