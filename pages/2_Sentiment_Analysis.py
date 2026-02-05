"""감성 분석 결과 페이지."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "analysis"

SENTIMENT_COLORS = {"긍정": "#27ae60", "중립": "#95a5a6", "부정": "#e74c3c"}
SOURCE_NAMES = {
    "industry_risk": "산업 리스크",
    "kookmin_media": "국민대 미디어",
    "kookmin_press": "국민대신문",
}

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")


@st.cache_data
def load_sentiment_data():
    """감성 분석 데이터 로드."""
    for fname in ["sentiment_ml_v2.csv", "sentiment_ml.csv"]:
        file_path = DATA_DIR / fname
        if file_path.exists():
            df = pd.read_csv(file_path, encoding="utf-8-sig", dtype=str)
            df.fillna("", inplace=True)
            return df
    return pd.DataFrame()


def main():
    st.title("💬 감성 분석 결과")
    st.markdown("**KoELECTRA** 모델을 활용한 뉴스 기사 감성 분석")

    df = load_sentiment_data()

    if df.empty:
        st.error("감성 분석 데이터가 없습니다.")
        return

    # 사이드바 필터
    st.sidebar.header("필터")
    sources = df["source"].unique().tolist() if "source" in df.columns else []
    selected_sources = st.sidebar.multiselect(
        "데이터 소스",
        options=sources,
        default=sources,
        format_func=lambda x: SOURCE_NAMES.get(x, x),
    )

    # 필터 적용
    if selected_sources:
        filtered_df = df[df["source"].isin(selected_sources)]
    else:
        filtered_df = df

    # 메트릭 카드
    col1, col2, col3, col4 = st.columns(4)

    total = len(filtered_df)
    if "sentiment_label_ml" in filtered_df.columns:
        dist = filtered_df["sentiment_label_ml"].value_counts()
        positive = dist.get("긍정", 0)
        negative = dist.get("부정", 0)
        neutral = dist.get("중립", 0)
    else:
        positive, negative, neutral = 0, 0, 0

    with col1:
        st.metric("총 기사", f"{total:,}건")
    with col2:
        st.metric("긍정", f"{positive:,}건", f"{positive/total*100:.1f}%" if total > 0 else "")
    with col3:
        st.metric("부정", f"{negative:,}건", f"{negative/total*100:.1f}%" if total > 0 else "")
    with col4:
        if "sentiment_confidence_ml" in filtered_df.columns:
            conf = pd.to_numeric(filtered_df["sentiment_confidence_ml"], errors="coerce")
            avg_conf = conf.mean()
            st.metric("평균 신뢰도", f"{avg_conf:.1%}" if pd.notna(avg_conf) else "N/A")
        else:
            st.metric("평균 신뢰도", "N/A")

    st.divider()

    # 시각화
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 전체 감성 분포")
        if "sentiment_label_ml" in filtered_df.columns:
            sentiment_counts = filtered_df["sentiment_label_ml"].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map=SENTIMENT_COLORS,
                hole=0.4,
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 소스별 감성 분포")
        if "source" in filtered_df.columns and "sentiment_label_ml" in filtered_df.columns:
            source_sentiment = filtered_df.groupby(["source", "sentiment_label_ml"]).size().reset_index(name="count")
            source_sentiment["source_name"] = source_sentiment["source"].map(SOURCE_NAMES)

            fig = px.bar(
                source_sentiment,
                x="source_name",
                y="count",
                color="sentiment_label_ml",
                color_discrete_map=SENTIMENT_COLORS,
                barmode="group",
                labels={"count": "기사 수", "source_name": "소스", "sentiment_label_ml": "감성"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 신뢰도 분포
    st.subheader("📊 감성 분석 신뢰도 분포")
    if "sentiment_confidence_ml" in filtered_df.columns:
        conf = pd.to_numeric(filtered_df["sentiment_confidence_ml"], errors="coerce").dropna()

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.histogram(
                conf,
                nbins=30,
                labels={"value": "신뢰도", "count": "빈도"},
                color_discrete_sequence=["#3498db"],
            )
            fig.add_vline(x=conf.mean(), line_dash="dash", line_color="red", annotation_text=f"평균: {conf.mean():.3f}")
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**신뢰도 통계**")
            st.write(f"- 평균: {conf.mean():.3f}")
            st.write(f"- 중앙값: {conf.median():.3f}")
            st.write(f"- 최소: {conf.min():.3f}")
            st.write(f"- 최대: {conf.max():.3f}")
            st.write(f"- 표준편차: {conf.std():.3f}")

    st.divider()

    # 샘플 기사
    st.subheader("📰 샘플 기사")

    sentiment_filter = st.selectbox("감성 필터", ["전체", "긍정", "부정", "중립"])

    if sentiment_filter != "전체" and "sentiment_label_ml" in filtered_df.columns:
        sample_df = filtered_df[filtered_df["sentiment_label_ml"] == sentiment_filter]
    else:
        sample_df = filtered_df

    # 랜덤 샘플 10개
    if len(sample_df) > 10:
        display_df = sample_df.sample(10)
    else:
        display_df = sample_df

    if "title" in display_df.columns:
        cols = ["title", "sentiment_label_ml", "sentiment_confidence_ml", "source"]
        cols = [c for c in cols if c in display_df.columns]
        display_df = display_df[cols].copy()
        display_df.columns = [{"title": "제목", "sentiment_label_ml": "감성", "sentiment_confidence_ml": "신뢰도", "source": "소스"}.get(c, c) for c in cols]
        st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
