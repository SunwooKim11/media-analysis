"""기사 탐색 페이지."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "analysis"

SOURCE_NAMES = {
    "industry_risk": "산업 리스크",
    "kookmin_media": "국민대 미디어",
    "kookmin_press": "국민대신문",
}

st.set_page_config(page_title="Article Explorer", page_icon="🔍", layout="wide")


@st.cache_data
def load_all_articles():
    """모든 기사 데이터 로드."""
    dfs = []

    file_patterns = [
        ("industry_risk_ml_v2_light.csv", "industry_risk_ml_v2.csv", "industry_risk_ml.csv"),
        ("kookmin_media_ml_v2_light.csv", "kookmin_media_ml_v2.csv", "kookmin_media_ml.csv"),
        ("kookmin_press_ml_v2_light.csv", "kookmin_press_ml_v2.csv", "kookmin_press_ml.csv"),
    ]

    for filenames in file_patterns:
        for fname in filenames:
            file_path = DATA_DIR / fname
            if file_path.exists():
                df = pd.read_csv(file_path, encoding="utf-8-sig", dtype=str)
                df.fillna("", inplace=True)
                dfs.append(df)
                break

    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        return result

    return pd.DataFrame()


@st.cache_data
def get_all_categories(df: pd.DataFrame) -> list:
    """모든 매칭된 카테고리 추출."""
    categories = set()
    if "matched_categories_ml" in df.columns:
        for cats in df["matched_categories_ml"]:
            if cats:
                for c in str(cats).split(","):
                    c = c.strip()
                    if c:
                        categories.add(c)
    return sorted(categories)


def main():
    st.title("🔍 기사 탐색기")
    st.markdown("ESG 관련 기사를 검색하고 필터링합니다.")

    df = load_all_articles()

    if df.empty:
        st.error("기사 데이터가 없습니다.")
        return

    # 사이드바 필터
    st.sidebar.header("필터")

    # 검색어
    search_query = st.sidebar.text_input("🔎 제목 검색", placeholder="키워드 입력...")

    # 소스 필터
    sources = df["source"].unique().tolist() if "source" in df.columns else []
    selected_sources = st.sidebar.multiselect(
        "📰 데이터 소스",
        options=sources,
        default=sources,
        format_func=lambda x: SOURCE_NAMES.get(x, x),
    )

    # 감성 필터
    sentiments = ["전체"]
    if "sentiment_label_ml" in df.columns:
        sentiments += df["sentiment_label_ml"].unique().tolist()
    selected_sentiment = st.sidebar.selectbox("💬 감성", options=sentiments)

    # ESG 카테고리 필터
    all_categories = get_all_categories(df)
    selected_category = st.sidebar.selectbox(
        "📂 ESG 카테고리",
        options=["전체"] + all_categories,
    )

    # 매칭 여부 필터
    match_filter = st.sidebar.radio(
        "🔗 매칭 상태",
        options=["전체", "매칭됨", "미매칭"],
    )

    # 필터 적용
    filtered_df = df.copy()

    # 검색어 필터
    if search_query:
        if "title" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["title"].str.contains(search_query, case=False, na=False)]

    # 소스 필터
    if selected_sources and "source" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["source"].isin(selected_sources)]

    # 감성 필터
    if selected_sentiment != "전체" and "sentiment_label_ml" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["sentiment_label_ml"] == selected_sentiment]

    # 카테고리 필터
    if selected_category != "전체" and "matched_categories_ml" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["matched_categories_ml"].str.contains(selected_category, case=False, na=False)]

    # 매칭 필터
    if match_filter != "전체" and "matched_categories_ml" in filtered_df.columns:
        if match_filter == "매칭됨":
            filtered_df = filtered_df[filtered_df["matched_categories_ml"].str.strip() != ""]
        else:
            filtered_df = filtered_df[filtered_df["matched_categories_ml"].str.strip() == ""]

    # 결과 표시
    st.subheader(f"📊 검색 결과: {len(filtered_df):,}건")

    # 페이지네이션
    page_size = st.sidebar.slider("페이지당 기사 수", 10, 100, 25)
    total_pages = max(1, (len(filtered_df) - 1) // page_size + 1)
    page = st.sidebar.number_input("페이지", min_value=1, max_value=total_pages, value=1)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_df = filtered_df.iloc[start_idx:end_idx]

    st.caption(f"페이지 {page} / {total_pages} (총 {len(filtered_df):,}건)")

    # 기사 목록
    for i, (_, row) in enumerate(page_df.iterrows()):
        with st.expander(f"📰 {row.get('title', '제목 없음')[:80]}...", expanded=False):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**제목**: {row.get('title', 'N/A')}")

                if "url" in row and row["url"]:
                    st.markdown(f"🔗 [기사 링크]({row['url']})")

                if "date" in row and row["date"]:
                    st.markdown(f"📅 **날짜**: {row['date']}")

                if "matched_categories_ml" in row and row["matched_categories_ml"]:
                    st.markdown(f"🏷️ **매칭 카테고리**: {row['matched_categories_ml']}")

            with col2:
                if "sentiment_label_ml" in row:
                    sentiment = row["sentiment_label_ml"]
                    color = {"긍정": "🟢", "부정": "🔴", "중립": "⚪"}.get(sentiment, "⚪")
                    st.markdown(f"{color} **{sentiment}**")

                if "sentiment_confidence_ml" in row and row["sentiment_confidence_ml"]:
                    try:
                        conf = float(row["sentiment_confidence_ml"])
                        st.markdown(f"신뢰도: {conf:.1%}")
                    except (ValueError, TypeError):
                        pass

                if "source" in row:
                    st.markdown(f"📰 {SOURCE_NAMES.get(row['source'], row['source'])}")

    st.divider()

    # 데이터 다운로드
    st.subheader("📥 데이터 다운로드")

    csv = filtered_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="CSV 다운로드",
        data=csv,
        file_name="filtered_articles.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
