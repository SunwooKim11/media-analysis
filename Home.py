"""ESG 중대성 평가 대시보드 - 홈페이지."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# 경로 설정
DATA_DIR = PROJECT_ROOT / "data" / "analysis"

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

# 페이지 설정
st.set_page_config(
    page_title="국민대 ESG 중대성 평가",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_summary_data():
    """요약 데이터 로드."""
    summary = {}

    # 중대성 점수 (23개 사전 정의 지표만)
    scores_file = DATA_DIR / "materiality_scores_ml.csv"
    if scores_file.exists():
        df = pd.read_csv(scores_file, encoding="utf-8-sig")
        df = df[df["indicator"].isin(PREDEFINED_INDICATORS)].copy()
        summary["total_indicators"] = len(df)
        # Q1 지표 수
        q1 = df[(df["issue_exposure"] >= 2.5) & (df["impact_materiality"] >= 2.5)]
        summary["q1_count"] = len(q1)
        summary["q1_indicators"] = q1["indicator"].tolist()[:5]

    # 감성 분석 (v2 우선)
    for fname in ["sentiment_ml_v2.csv", "sentiment_ml.csv"]:
        sentiment_file = DATA_DIR / fname
        if sentiment_file.exists():
            df = pd.read_csv(sentiment_file, encoding="utf-8-sig", dtype=str)
            summary["total_articles"] = len(df)
            if "sentiment_label_ml" in df.columns:
                dist = df["sentiment_label_ml"].value_counts()
                summary["sentiment_dist"] = dist.to_dict()
            break

    # 매칭률 (v2 우선)
    for fname in ["matched_categories_ml_v2.csv", "matched_categories_ml.csv"]:
        matched_file = DATA_DIR / fname
        if matched_file.exists():
            df = pd.read_csv(matched_file, encoding="utf-8-sig", dtype=str)
            df.fillna("", inplace=True)
            if "matched_categories_ml" in df.columns:
                matched = (df["matched_categories_ml"].str.strip() != "").sum()
                summary["matched_count"] = matched
                summary["match_rate"] = matched / len(df) * 100 if len(df) > 0 else 0
            break

    return summary


def main():
    """메인 페이지."""
    # 헤더
    st.title("🌱 국민대학교 ESG 중대성 평가")
    st.markdown("""
    **Double Materiality Framework**를 활용한 ESG 중대성 분석 대시보드입니다.

    - **S-BERT**: 기사-ESG 지표 키워드 매칭
    - **KoELECTRA**: 감성 분석 (긍정/부정)
    - **분석 기간**: 2023.01 ~ 2025.12
    """)

    st.divider()

    # 요약 데이터 로드
    summary = load_summary_data()

    # 주요 지표 카드
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 총 분석 기사",
            value=f"{summary.get('total_articles', 0):,}건",
        )

    with col2:
        st.metric(
            label="🔍 ESG 키워드 매칭률",
            value=f"{summary.get('match_rate', 0):.1f}%",
            delta=f"{summary.get('matched_count', 0):,}건 매칭",
        )

    with col3:
        sentiment = summary.get("sentiment_dist", {})
        positive = sentiment.get("긍정", 0)
        total = sum(sentiment.values()) if sentiment else 1
        st.metric(
            label="💚 긍정 감성 비율",
            value=f"{positive / total * 100:.1f}%" if total > 0 else "N/A",
        )

    with col4:
        st.metric(
            label="⚠️ Q1 핵심 지표",
            value=f"{summary.get('q1_count', 0)}개",
            help="이슈 노출도 + 영향 중대성 모두 높은 지표",
        )

    st.divider()

    # Q1 핵심 지표 하이라이트
    st.subheader("🎯 Q1 핵심 중대 지표")
    st.markdown("*이슈 노출도와 영향 중대성이 모두 높아 우선 관리가 필요한 지표*")

    q1_indicators = summary.get("q1_indicators", [])
    if q1_indicators:
        for i, indicator in enumerate(q1_indicators, 1):
            st.markdown(f"**{i}.** {indicator}")
    else:
        st.info("Q1 지표가 없습니다.")

    st.divider()

    # 페이지 안내
    st.subheader("📑 페이지 안내")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **1️⃣ Materiality Matrix**
        - ESG 중대성 매트릭스 (산점도)
        - 사분면별 지표 분석
        - Top 10 중대 이슈

        **2️⃣ Sentiment Analysis**
        - 소스별 감성 분포
        - 신뢰도 분석
        - 시계열 트렌드
        """)

    with col2:
        st.markdown("""
        **3️⃣ Article Explorer**
        - 기사 검색 및 필터링
        - ESG 카테고리별 탐색
        - 상세 정보 확인

        **4️⃣ Report**
        - 전체 분석 보고서
        - Markdown 형식
        """)

    # 푸터
    st.divider()
    st.caption("© 2026 국민대학교 ESG 중대성 평가 | Powered by Streamlit")


if __name__ == "__main__":
    main()
