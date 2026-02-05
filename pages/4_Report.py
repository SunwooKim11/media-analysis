"""분석 보고서 페이지."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "analysis"

st.set_page_config(page_title="Analysis Report", page_icon="📝", layout="wide")


@st.cache_data
def load_report():
    """분석 보고서 로드."""
    report_file = DATA_DIR / "analysis_report_ml.md"
    if report_file.exists():
        return report_file.read_text(encoding="utf-8")
    return None


def main():
    st.title("📝 분석 보고서")
    st.markdown("ESG 중대성 평가 종합 분석 보고서")

    report = load_report()

    if report is None:
        st.error("분석 보고서 파일이 없습니다.")
        st.info("먼저 `python -c \"from analysis.report_generator import generate_analysis_report; generate_analysis_report()\"` 를 실행하세요.")
        return

    # 보고서 표시
    st.markdown(report)

    st.divider()

    # 다운로드 버튼
    st.download_button(
        label="📥 Markdown 다운로드",
        data=report,
        file_name="analysis_report_ml.md",
        mime="text/markdown",
    )

    # 시각화 이미지 갤러리
    st.divider()
    st.subheader("📊 시각화 갤러리")

    image_files = [
        ("materiality_matrix_ml.png", "중대성 매트릭스"),
        ("materiality_top10.png", "Top 10 중대 이슈"),
        ("sentiment_by_source.png", "소스별 감성 분포"),
        ("confidence_histogram.png", "신뢰도 분포"),
        ("esg_category_breakdown.png", "ESG 카테고리별 분석"),
        ("matching_rate.png", "키워드 매칭률"),
    ]

    cols = st.columns(3)
    for i, (filename, title) in enumerate(image_files):
        file_path = DATA_DIR / filename
        if file_path.exists():
            with cols[i % 3]:
                st.image(str(file_path), caption=title, use_container_width=True)


if __name__ == "__main__":
    main()
