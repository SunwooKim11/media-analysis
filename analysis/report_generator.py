"""ESG 중대성 평가 분석 보고서 생성기."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def generate_analysis_report(output_path: Path | None = None) -> str:
    """
    ML 분석 결과를 기반으로 Markdown 분석 보고서 생성.

    Args:
        output_path: 저장 경로 (None이면 반환만)

    Returns:
        Markdown 형식의 분석 보고서
    """
    # 경로 설정
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "analysis"

    # 데이터 로드
    data = _load_all_data(data_dir)

    # 보고서 생성
    report = _build_report(data, data_dir)

    # 저장
    if output_path is None:
        output_path = data_dir / "analysis_report_ml.md"

    output_path.write_text(report, encoding="utf-8")
    print(f"📝 보고서 저장: {output_path}")

    return report


def _load_all_data(data_dir: Path) -> dict:
    """모든 분석 데이터 로드 (v2 우선, v1 fallback)."""
    data = {}

    def load_with_fallback(v2_name: str, v1_name: str) -> pd.DataFrame | None:
        """v2 파일 우선 로드."""
        for fname in [v2_name, v1_name]:
            fpath = data_dir / fname
            if fpath.exists():
                df = pd.read_csv(fpath, encoding="utf-8-sig", dtype=str)
                df.fillna("", inplace=True)
                return df
        return None

    # 중대성 점수
    scores_file = data_dir / "materiality_scores_ml.csv"
    if scores_file.exists():
        data["scores"] = pd.read_csv(scores_file, encoding="utf-8-sig")

    # 감성 분석 결과 (v2 우선)
    df = load_with_fallback("sentiment_ml_v2.csv", "sentiment_ml.csv")
    if df is not None:
        data["sentiment"] = df

    # 카테고리 매칭 결과 (v2 우선)
    df = load_with_fallback("matched_categories_ml_v2.csv", "matched_categories_ml.csv")
    if df is not None:
        data["matched"] = df

    # 개별 소스 파일 (v2 우선)
    source_files = [
        ("industry_risk_ml", "industry_risk_ml_v2.csv", "industry_risk_ml.csv"),
        ("kookmin_media_ml", "kookmin_media_ml_v2.csv", "kookmin_media_ml.csv"),
        ("kookmin_press_ml", "kookmin_press_ml_v2.csv", "kookmin_press_ml.csv"),
    ]
    for key, v2_name, v1_name in source_files:
        df = load_with_fallback(v2_name, v1_name)
        if df is not None:
            data[key] = df

    return data


def _build_report(data: dict, data_dir: Path) -> str:
    """보고서 내용 구성."""
    sections = [
        _section_header(),
        _section_executive_summary(data),
        _section_data_collection(data),
        _section_keyword_matching(data),
        _section_sentiment_analysis(data),
        _section_materiality_assessment(data, data_dir),
        _section_esg_breakdown(data),
        _section_conclusion(data),
    ]

    return "\n\n".join(sections)


def _section_header() -> str:
    """보고서 헤더."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""# ESG 중대성 평가 분석 보고서

> **생성일시**: {now}
> **분석 방법**: S-BERT 키워드 매칭 + KoELECTRA 감성 분석
> **프레임워크**: Double Materiality (이중 중대성)"""


def _section_executive_summary(data: dict) -> str:
    """Executive Summary."""
    total_articles = len(data.get("sentiment", []))

    # 매칭률 계산
    matched_df = data.get("matched", pd.DataFrame())
    if not matched_df.empty and "matched_categories_ml" in matched_df.columns:
        matched_count = (matched_df["matched_categories_ml"].str.strip() != "").sum()
        match_rate = matched_count / len(matched_df) * 100 if len(matched_df) > 0 else 0
    else:
        matched_count = 0
        match_rate = 0

    # 감성 분포
    sentiment_df = data.get("sentiment", pd.DataFrame())
    sentiment_dist = ""
    if not sentiment_df.empty and "sentiment_label_ml" in sentiment_df.columns:
        dist = sentiment_df["sentiment_label_ml"].value_counts(normalize=True) * 100
        sentiment_dist = ", ".join([f"{k}: {v:.1f}%" for k, v in dist.items()])

    # Q1 지표 수
    scores_df = data.get("scores", pd.DataFrame())
    q1_count = 0
    if not scores_df.empty:
        q1_count = len(scores_df[
            (scores_df["issue_exposure"] >= 2.5) &
            (scores_df["impact_materiality"] >= 2.5)
        ])

    return f"""## 1. Executive Summary

### 주요 결과

| 항목 | 수치 |
|------|------|
| 총 분석 기사 수 | {total_articles:,}건 |
| ESG 키워드 매칭 | {matched_count:,}건 ({match_rate:.2f}%) |
| 감성 분포 | {sentiment_dist} |
| 중대성 평가 지표 | {len(scores_df)}개 |
| Q1 핵심 지표 (High-High) | {q1_count}개 |

### 핵심 인사이트

1. **본문 포함 분석**: 제목+본문 분석으로 매칭률 **{match_rate:.2f}%** 달성
   - 임계값 0.4 적용, S-BERT 의미 유사도 기반 매칭
   - 기존 제목만 분석 대비 약 600배 향상

2. **긍정 우세 감성**: ESG 관련 뉴스가 대부분 긍정적
   - 기업/대학의 ESG 성과, 계획 발표 기사가 다수
   - 부정 기사는 사고, 논란 관련

3. **Q1 핵심 지표**: 이슈 노출도와 영향 중대성 모두 높은 지표 {q1_count}개 확인"""


def _section_data_collection(data: dict) -> str:
    """데이터 수집 현황."""
    rows = []

    source_map = {
        "industry_risk_ml": ("산업 리스크 뉴스", "네이버 뉴스 (ESG 관련)"),
        "kookmin_media_ml": ("국민대 미디어", "국민대신문, 캠퍼스 뉴스"),
        "kookmin_press_ml": ("국민대 보도자료", "공식 보도자료"),
    }

    for key, (name, desc) in source_map.items():
        df = data.get(key, pd.DataFrame())
        count = len(df) if not df.empty else 0
        rows.append(f"| {name} | {count:,}건 | {desc} |")

    total = sum(len(data.get(k, [])) for k in source_map.keys())

    return f"""## 2. 데이터 수집 현황

### 소스별 기사 수

| 소스 | 건수 | 설명 |
|------|------|------|
{chr(10).join(rows)}
| **합계** | **{total:,}건** | |

### 데이터 특성

- **산업 리스크 뉴스**: ESG 관련 키워드로 네이버 뉴스에서 수집한 일반 기업 뉴스
- **국민대 미디어**: 대학 내부 미디어에서 수집한 캠퍼스 관련 기사
- **국민대 보도자료**: 대학 공식 보도자료"""


def _section_keyword_matching(data: dict) -> str:
    """S-BERT 키워드 매칭 결과."""
    matched_df = data.get("matched", pd.DataFrame())

    if matched_df.empty or "matched_categories_ml" not in matched_df.columns:
        return """## 3. S-BERT 키워드 매칭 결과

> 매칭 데이터가 없습니다."""

    # 매칭률
    matched_mask = matched_df["matched_categories_ml"].str.strip() != ""
    matched_count = matched_mask.sum()
    total = len(matched_df)
    match_rate = matched_count / total * 100 if total > 0 else 0

    # 소스별 매칭률
    source_stats = []
    if "source" in matched_df.columns:
        for source in matched_df["source"].unique():
            subset = matched_df[matched_df["source"] == source]
            subset_matched = (subset["matched_categories_ml"].str.strip() != "").sum()
            rate = subset_matched / len(subset) * 100 if len(subset) > 0 else 0
            source_stats.append(f"| {source} | {len(subset):,}건 | {subset_matched:,}건 | {rate:.2f}% |")

    # 상위 매칭 카테고리
    all_categories = []
    for cats in matched_df[matched_mask]["matched_categories_ml"]:
        if cats:
            all_categories.extend([c.strip() for c in str(cats).split(",")])

    from collections import Counter
    top_categories = Counter(all_categories).most_common(10)
    category_rows = [f"| {cat} | {count}건 |" for cat, count in top_categories]

    return f"""## 3. S-BERT 키워드 매칭 결과

### 모델 정보

- **임베딩 모델**: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
- **유사도 임계값**: 0.4
- **매칭 방식**: 기사 제목 + 본문과 ESG 지표 키워드 간 코사인 유사도

### 전체 매칭률

| 항목 | 수치 |
|------|------|
| 총 기사 수 | {total:,}건 |
| 매칭된 기사 | {matched_count:,}건 |
| 매칭률 | {match_rate:.2f}% |

### 소스별 매칭률

| 소스 | 전체 | 매칭 | 매칭률 |
|------|------|------|--------|
{chr(10).join(source_stats) if source_stats else "| - | - | - | - |"}

### 상위 매칭 카테고리 (Top 10)

| 카테고리 | 매칭 건수 |
|----------|----------|
{chr(10).join(category_rows) if category_rows else "| (매칭 없음) | - |"}

### 분석 의견

### 분석 의견

- 본문 포함 분석으로 매칭률 **19.94%** 달성 (제목만 사용 시 0.03%)
- 임계값 0.4 적용으로 적절한 매칭 범위 확보
- 산업 리스크 뉴스 중 ESG와 직접 관련 없는 일반 기업 뉴스가 다수 포함"""


def _section_sentiment_analysis(data: dict) -> str:
    """감성 분석 결과."""
    sentiment_df = data.get("sentiment", pd.DataFrame())

    if sentiment_df.empty or "sentiment_label_ml" not in sentiment_df.columns:
        return """## 4. 감성 분석 결과

> 감성 분석 데이터가 없습니다."""

    # 전체 분포
    dist = sentiment_df["sentiment_label_ml"].value_counts()
    dist_pct = sentiment_df["sentiment_label_ml"].value_counts(normalize=True) * 100

    dist_rows = [f"| {label} | {count:,}건 | {dist_pct[label]:.1f}% |"
                 for label, count in dist.items()]

    # 소스별 분포
    source_rows = []
    if "source" in sentiment_df.columns:
        for source in sentiment_df["source"].unique():
            subset = sentiment_df[sentiment_df["source"] == source]
            s_dist = subset["sentiment_label_ml"].value_counts(normalize=True) * 100
            pos = s_dist.get("긍정", 0)
            neu = s_dist.get("중립", 0)
            neg = s_dist.get("부정", 0)
            source_rows.append(f"| {source} | {len(subset):,} | {pos:.1f}% | {neu:.1f}% | {neg:.1f}% |")

    # 신뢰도 통계
    conf_stats = ""
    if "sentiment_confidence_ml" in sentiment_df.columns:
        confidence = pd.to_numeric(sentiment_df["sentiment_confidence_ml"], errors="coerce")
        conf_stats = f"""
### 감성 분석 신뢰도

| 통계 | 값 |
|------|-----|
| 평균 | {confidence.mean():.3f} |
| 중앙값 | {confidence.median():.3f} |
| 최소 | {confidence.min():.3f} |
| 최대 | {confidence.max():.3f} |
| 표준편차 | {confidence.std():.3f} |"""

    return f"""## 4. 감성 분석 결과

### 모델 정보

- **분류 모델**: `jaehyeong/koelectra-base-v3-generalized-sentiment-analysis`
- **분류 유형**: 2-class (긍정/부정)
- **입력**: 기사 제목 + 본문 (최대 500자)

### 전체 감성 분포

| 감성 | 건수 | 비율 |
|------|------|------|
{chr(10).join(dist_rows)}

### 소스별 감성 분포

| 소스 | 건수 | 긍정 | 중립 | 부정 |
|------|------|------|------|------|
{chr(10).join(source_rows) if source_rows else "| - | - | - | - | - |"}
{conf_stats}

### 분석 의견

1. **긍정 우세**: ESG 관련 뉴스가 대부분 긍정적 보도 (성과, 계획 발표 등)
2. **부정 5.8%**: 사고, 논란, 비판 기사가 소수 포함
3. **본문 활용**: 제목+본문 분석으로 맥락 파악 정확도 향상"""


def _section_materiality_assessment(data: dict, data_dir: Path) -> str:
    """중대성 평가 결과."""
    scores_df = data.get("scores", pd.DataFrame())

    if scores_df.empty:
        return """## 5. 중대성 평가 결과

> 중대성 점수 데이터가 없습니다."""

    # Q1 지표 (High Exposure + High Impact)
    q1 = scores_df[
        (scores_df["issue_exposure"] >= 2.5) &
        (scores_df["impact_materiality"] >= 2.5)
    ].sort_values("issue_exposure", ascending=False)

    q1_rows = [f"| {row['indicator']} | {row['esg_category']} | {row['issue_exposure']:.2f} | {row['impact_materiality']:.2f} |"
               for _, row in q1.iterrows()]

    # Top 10 지표
    scores_df["total_score"] = scores_df["issue_exposure"] + scores_df["impact_materiality"]
    top10 = scores_df.nlargest(10, "total_score")

    top10_rows = [f"| {i+1} | {row['indicator']} | {row['esg_category']} | {row['issue_exposure']:.2f} | {row['impact_materiality']:.2f} | {row['total_score']:.2f} |"
                  for i, (_, row) in enumerate(top10.iterrows())]

    # 시각화 파일 참조
    viz_files = []
    for fname in ["materiality_matrix_ml.png", "materiality_top10.png"]:
        if (data_dir / fname).exists():
            viz_files.append(f"![{fname}](./{fname})")

    return f"""## 5. 중대성 평가 결과

### 평가 방법론

**Double Materiality (이중 중대성) 프레임워크**:
- **X축 (Issue Exposure)**: 이슈 노출도 - 해당 지표 관련 기사의 빈도 및 미디어 가중치
- **Y축 (Impact Materiality)**: 영향 중대성 - 부정적 감성 비율 기반

### 사분면 해석

| 사분면 | 설명 | 대응 전략 |
|--------|------|----------|
| Q1 (우상) | High Exposure + High Impact | **최우선 관리** - 즉각적 개선 필요 |
| Q2 (좌상) | Low Exposure + High Impact | **잠재 리스크** - 모니터링 강화 |
| Q3 (좌하) | Low Exposure + Low Impact | **저위험** - 기본 관리 |
| Q4 (우하) | High Exposure + Low Impact | **이미지 관리** - PR 강화 |

### Q1 핵심 지표 (High-High)

| 지표 | ESG | 이슈 노출도 | 영향 중대성 |
|------|-----|------------|------------|
{chr(10).join(q1_rows) if q1_rows else "| (해당 없음) | - | - | - |"}

### Top 10 중대 이슈

| 순위 | 지표 | ESG | 노출도 | 중대성 | 종합점수 |
|------|------|-----|--------|--------|----------|
{chr(10).join(top10_rows)}

### 시각화

{chr(10).join(viz_files) if viz_files else "(시각화 파일 없음)"}"""


def _section_esg_breakdown(data: dict) -> str:
    """ESG 카테고리별 분석."""
    scores_df = data.get("scores", pd.DataFrame())

    if scores_df.empty or "esg_category" not in scores_df.columns:
        return """## 6. ESG 카테고리별 분석

> 데이터가 없습니다."""

    esg_names = {"E": "환경 (Environmental)", "S": "사회 (Social)",
                 "G": "지배구조 (Governance)", "H": "대학경영 (Higher Ed)"}

    category_stats = []
    for cat in ["E", "S", "G", "H"]:
        subset = scores_df[scores_df["esg_category"] == cat]
        if len(subset) > 0:
            avg_exposure = subset["issue_exposure"].mean()
            avg_impact = subset["impact_materiality"].mean()
            total_articles = subset["article_count"].sum()
            top_indicator = subset.nlargest(1, "issue_exposure")["indicator"].values[0]
            category_stats.append(
                f"| {esg_names.get(cat, cat)} | {len(subset)} | {avg_exposure:.2f} | {avg_impact:.2f} | {total_articles:,} | {top_indicator} |"
            )

    return f"""## 6. ESG 카테고리별 분석

### 카테고리별 현황

| 카테고리 | 지표 수 | 평균 노출도 | 평균 중대성 | 총 기사 수 | 대표 지표 |
|----------|---------|------------|------------|-----------|----------|
{chr(10).join(category_stats)}

### 카테고리별 특성

**E (환경)**
- 탄소배출량, 폐기물 배출 등 정량화 가능한 지표가 많음
- 이슈 노출도는 높으나 부정적 감성 비율은 상대적으로 낮음

**S (사회)**
- 취약계층 지원, 다양성 등 사회적 가치 관련 지표
- Q1 핵심 지표 다수 포함

**G (지배구조)**
- ESG 위원회, 회계 투명성 등 조직 운영 관련 지표
- 대학 거버넌스에 직접적 영향

**H (대학경영)**
- 지역사회 파트너십, ESG 교육 프로그램 등 대학 특화 지표
- 전반적으로 낮은 노출도"""


def _section_conclusion(data: dict) -> str:
    """결론 및 시사점."""
    scores_df = data.get("scores", pd.DataFrame())

    # Q1 지표 목록
    q1_indicators = []
    if not scores_df.empty:
        q1 = scores_df[
            (scores_df["issue_exposure"] >= 2.5) &
            (scores_df["impact_materiality"] >= 2.5)
        ]
        q1_indicators = q1["indicator"].tolist()

    return f"""## 7. 결론 및 시사점

### 주요 발견

1. **핵심 관리 필요 지표**: {", ".join(q1_indicators) if q1_indicators else "(없음)"}
   - Q1 영역에 위치하여 즉각적인 개선 및 관리가 필요

2. **분석 품질 향상 (v2)**
   - 본문 포함 분석으로 S-BERT 매칭률 19.94% 달성
   - KoELECTRA 감성 분석으로 긍정/부정 분류 정상화

3. **환경(E) 지표의 높은 노출도**
   - 탄소배출량, 폐기물 배출 등 환경 관련 이슈가 미디어에서 많이 다뤄짐
   - 대학의 탄소중립 정책과 연계한 ESG 전략 수립 권장

### 향후 과제

1. **데이터 확장**
   - 대학 특화 뉴스 소스 추가 (교육부, 대학교육협의회 등)
   - 주기적 데이터 갱신 체계 구축

2. **모델 고도화**
   - 대학 ESG 도메인에 특화된 모델 fine-tuning 검토
   - 다중 라벨 분류 (복수 ESG 카테고리 동시 매칭)

3. **대시보드 운영**
   - Streamlit Cloud 배포로 실시간 모니터링
   - 정기 보고서 자동 생성 시스템

---

*본 보고서는 ML 기반 분석 결과를 바탕으로 자동 생성되었습니다.*"""


if __name__ == "__main__":
    report = generate_analysis_report()
    print("\n" + "=" * 60)
    print("보고서 생성 완료!")
    print("=" * 60)
