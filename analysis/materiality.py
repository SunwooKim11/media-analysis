"""ESG 중대성 평가 메인 파이프라인."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import pandas as pd

from .scoring import (
    KEYWORD_ESG_MAP,
    calculate_impact_materiality,
    calculate_issue_exposure,
    load_keyword_categories,
    merge_materiality_scores,
)
from .visualize import plot_materiality_matrix, plot_top_indicators

logger = logging.getLogger(__name__)

# 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"


def load_processed_data() -> dict[str, pd.DataFrame]:
    """전처리된 데이터 로드.

    Returns
    -------
    dict
        {"industry_risk": DataFrame, "kookmin_media": DataFrame, "kookmin_press": DataFrame}
    """
    data = {}

    # 1. 산업 리스크 데이터 (8개 JSONL 병합)
    risk_files = sorted(DATA_PROCESSED_DIR.glob("industry_risk_*_body.jsonl"))
    if risk_files:
        risk_articles = []
        for f in risk_files:
            with open(f, "r", encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        risk_articles.append(json.loads(line))
        data["industry_risk"] = pd.DataFrame(risk_articles)
        data["industry_risk"].fillna("", inplace=True)
        logger.info("산업 리스크 데이터 로드: %d건 (%d개 파일)", len(data["industry_risk"]), len(risk_files))
    else:
        logger.warning("산업 리스크 JSONL 파일 없음: %s/industry_risk_*_body.jsonl", DATA_PROCESSED_DIR)
        data["industry_risk"] = pd.DataFrame()

    # 2. 국민대 미디어 데이터
    media_file = DATA_PROCESSED_DIR / "kookmin_media_cleaned.csv"
    if media_file.exists():
        data["kookmin_media"] = pd.read_csv(media_file, encoding="utf-8-sig", dtype=str)
        data["kookmin_media"].fillna("", inplace=True)
        logger.info("국민대 미디어 데이터 로드: %d건", len(data["kookmin_media"]))
    else:
        logger.warning("국민대 미디어 파일 없음: %s", media_file)
        data["kookmin_media"] = pd.DataFrame()

    # 3. 국민대신문 데이터
    press_file = DATA_PROCESSED_DIR / "kookmin_press_cleaned.jsonl"
    if press_file.exists():
        articles = []
        with open(press_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    articles.append(json.loads(line))
        data["kookmin_press"] = pd.DataFrame(articles)
        data["kookmin_press"].fillna("", inplace=True)
        logger.info("국민대신문 데이터 로드: %d건", len(data["kookmin_press"]))
    else:
        logger.warning("국민대신문 파일 없음: %s", press_file)
        data["kookmin_press"] = pd.DataFrame()

    return data


def load_ml_results() -> dict[str, pd.DataFrame]:
    """Colab에서 생성한 ML 분석 결과 로드.

    v2 파일 우선, 없으면 v1 파일 로드.

    Returns
    -------
    dict
        {"industry_risk": DataFrame, "kookmin_media": DataFrame, ...}
    """
    ml_data = {}

    # 파일명 우선순위: v2 -> v1
    file_patterns = {
        "industry_risk": ["industry_risk_ml_v2.csv", "industry_risk_ml.csv"],
        "kookmin_media": ["kookmin_media_ml_v2.csv", "kookmin_media_ml.csv"],
        "kookmin_press": ["kookmin_press_ml_v2.csv", "kookmin_press_ml.csv"],
    }

    for source, filenames in file_patterns.items():
        ml_data[source] = pd.DataFrame()

        for filename in filenames:
            file_path = DATA_ANALYSIS_DIR / filename
            if file_path.exists():
                ml_data[source] = pd.read_csv(file_path, encoding="utf-8-sig", dtype=str)
                ml_data[source].fillna("", inplace=True)
                logger.info("%s ML 결과 로드: %d건 (%s)", source, len(ml_data[source]), filename)
                break

    return ml_data


def merge_ml_results(data: dict, ml_data: dict, key_column: str = "title") -> dict:
    """원본 데이터에 ML 결과 병합.

    Parameters
    ----------
    data : dict
        원본 데이터.
    ml_data : dict
        ML 분석 결과.
    key_column : str
        조인 키.

    Returns
    -------
    dict
        ML 결과가 병합된 데이터.
    """
    merged = {}

    for source in ["industry_risk", "kookmin_media", "kookmin_press"]:
        if source not in data or data[source].empty:
            merged[source] = data.get(source, pd.DataFrame())
            continue

        df = data[source].copy()
        ml_df = ml_data.get(source, pd.DataFrame())

        if ml_df.empty:
            merged[source] = df
            continue

        # ML 결과 컬럼
        ml_cols = [
            "matched_categories_ml", "matched_scores_ml",
            "sentiment_label_ml", "sentiment_score_ml", "sentiment_confidence_ml",
        ]
        available_cols = [c for c in ml_cols if c in ml_df.columns]

        if available_cols and key_column in ml_df.columns:
            # 중복 제거 후 병합
            ml_subset = ml_df[[key_column] + available_cols].drop_duplicates(
                subset=[key_column], keep="first"
            )
            df = df.merge(ml_subset, on=key_column, how="left", suffixes=("", "_ml_dup"))

            # 중복 컬럼 정리
            for col in available_cols:
                dup_col = f"{col}_ml_dup"
                if dup_col in df.columns:
                    df.drop(columns=[dup_col], inplace=True)

            df.fillna("", inplace=True)
            logger.info("%s에 ML 결과 병합 완료", source)

        merged[source] = df

    return merged


def run_materiality_assessment(
    show_progress: bool = True,
    output_csv: str | Path | None = None,
    output_chart: str | Path | None = None,
    use_ml: bool = False,
) -> pd.DataFrame:
    """ESG 중대성 평가 실행.

    Parameters
    ----------
    show_progress : bool
        진행 상황 출력 여부.
    output_csv : str | Path, optional
        결과 CSV 저장 경로.
    output_chart : str | Path, optional
        차트 저장 경로.
    use_ml : bool
        ML 분석 결과 사용 여부 (기본: False).

    Returns
    -------
    pd.DataFrame
        중대성 점수 결과.
    """
    # 출력 파일 경로 설정
    suffix = "_ml" if use_ml else ""
    if output_csv is None:
        output_csv = DATA_ANALYSIS_DIR / f"materiality_scores{suffix}.csv"
    if output_chart is None:
        output_chart = DATA_ANALYSIS_DIR / f"materiality_matrix{suffix}.png"

    output_csv = Path(output_csv)
    output_chart = Path(output_chart)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_chart.parent.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print("=" * 60)
        print("ESG 중대성 평가 (Materiality Assessment)")
        if use_ml:
            print("[ML 모드: S-BERT 키워드 + KcBERT 감성]")
        print("=" * 60)

    # 1. 데이터 로드
    if show_progress:
        print("\n1. 데이터 로드...")
    data = load_processed_data()

    # ML 결과 병합 (use_ml=True인 경우)
    if use_ml:
        if show_progress:
            print("   ML 분석 결과 로드 중...")
        ml_data = load_ml_results()
        data = merge_ml_results(data, ml_data)

        # ML 결과가 있으면 matched_categories_ml을 사용
        for source in ["industry_risk", "kookmin_media", "kookmin_press"]:
            if source in data and "matched_categories_ml" in data[source].columns:
                # matched_categories_ml이 비어있지 않으면 사용
                has_ml = data[source]["matched_categories_ml"].str.strip() != ""
                data[source].loc[has_ml, "matched_categories"] = data[source].loc[has_ml, "matched_categories_ml"]
                if show_progress:
                    ml_count = has_ml.sum()
                    print(f"   {source}: {ml_count}건 ML 매칭 적용")

    # 2. ESG 카테고리 매핑 로드
    if show_progress:
        print("\n2. ESG 카테고리 매핑...")
    category_map = load_keyword_categories()
    if show_progress:
        print(f"   매핑된 진단항목: {len(category_map)}개")

    # 3. 이슈 노출도 계산 (X축)
    if show_progress:
        print("\n3. 이슈 노출도 계산 (X축: Issue Exposure)...")
        print("   데이터: industry_risk_*_body.jsonl (matched_categories 기준)")
        print("   부정 기사 가중치: 1.5x")

    issue_df = calculate_issue_exposure(
        data["industry_risk"],
        group_column="matched_categories",  # 진단항목 기준으로 변경
        negative_weight=1.5,
        use_ml=use_ml,
    )
    if show_progress:
        print(f"   계산된 진단항목: {len(issue_df)}개")

    # 4. 영향 중대성 계산 (Inside-Out)
    if show_progress:
        print("\n4. 영향 중대성 계산 (Y축: Inside-Out)...")
        print("   데이터: kookmin_media_cleaned.csv + kookmin_press")
        print("   긍정 기사 가중치: 1.3x")

    # 국민대 미디어 + 국민대신문 병합
    media_df = data["kookmin_media"].copy()
    press_df = data["kookmin_press"].copy()

    # 컬럼 통일
    if not press_df.empty:
        if "body" in press_df.columns:
            press_df = press_df.rename(columns={"body": "content"})
        # 국민대신문 press 컬럼 추가
        press_df["press"] = "국민대신문"

        # 필요한 컬럼만 선택
        common_cols = ["title", "content", "press", "matched_categories"]
        media_cols = [c for c in common_cols if c in media_df.columns]
        press_cols = [c for c in common_cols if c in press_df.columns]

        if media_cols and press_cols:
            combined_df = pd.concat([media_df[media_cols], press_df[press_cols]], ignore_index=True)
        else:
            combined_df = media_df
    else:
        combined_df = media_df

    impact_df = calculate_impact_materiality(
        combined_df,
        group_column="matched_categories",
        positive_weight=1.3,
        use_ml=use_ml,
    )
    if show_progress:
        print(f"   계산된 지표: {len(impact_df)}개")

    # 5. 점수 병합
    if show_progress:
        print("\n5. 중대성 점수 병합...")

    result_df = merge_materiality_scores(issue_df, impact_df)

    # ESG 카테고리 추가 (키워드 매핑 우선, 없으면 카테고리 매핑 사용)
    result_df["esg_category"] = result_df["indicator"].apply(
        lambda x: KEYWORD_ESG_MAP.get(x, category_map.get(x, "H"))
    )

    if show_progress:
        print(f"   최종 지표: {len(result_df)}개")

    # 6. 결과 저장
    if show_progress:
        print(f"\n6. 결과 저장: {output_csv}")

    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    # 7. 시각화
    if show_progress:
        print(f"\n7. 시각화: {output_chart}")

    plot_materiality_matrix(
        result_df,
        output_path=output_chart,
        category_map=category_map,
        title="국민대학교 ESG Materiality Matrix",
    )

    # 상위 지표 차트
    top_chart = output_chart.parent / "materiality_top10.png"
    plot_top_indicators(
        result_df,
        top_n=10,
        output_path=top_chart,
        category_map=category_map,
    )

    # 8. 상위 5개 핵심 중대 주제 출력
    if show_progress:
        print("\n" + "=" * 60)
        print("핵심 중대 주제 (Q1: 이슈+영향 모두 높음)")
        print("=" * 60)

        # Q1: 이슈 노출도 > 2.5 AND 영향 중대성 > 2.5 (0~5 스케일)
        q1_df = result_df[
            (result_df["issue_exposure"] > 2.5) &
            (result_df["impact_materiality"] > 2.5)
        ]

        if len(q1_df) > 0:
            q1_df = q1_df.sort_values(
                by=["issue_exposure", "impact_materiality"],
                ascending=False,
            ).head(5)

            for i, (_, row) in enumerate(q1_df.iterrows(), 1):
                print(f"\n{i}. {row['indicator']} [{row['esg_category']}]")
                print(f"   이슈 노출도: {row['issue_exposure']:.2f}")
                print(f"   영향 중대성: {row['impact_materiality']:.2f}")
                print(f"   미디어 노출도: {row['total_exposure']:.1f}")
                print(f"   관련 기사: {row['article_count']}건")
        else:
            # Q1이 없으면 전체 상위 5개
            print("\n(Q1 지표 없음 - 전체 상위 5개 출력)")
            top5 = result_df.head(5)
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                print(f"\n{i}. {row['indicator']} [{row['esg_category']}]")
                print(f"   이슈 노출도: {row['issue_exposure']:.2f}")
                print(f"   영향 중대성: {row['impact_materiality']:.2f}")
                print(f"   미디어 노출도: {row['total_exposure']:.1f}")
                print(f"   관련 기사: {row['article_count']}건")

        print("\n" + "=" * 60)
        print(f"분석 완료")
        print(f"  - 점수 데이터: {output_csv}")
        print(f"  - 매트릭스 차트: {output_chart}")
        print(f"  - Top 10 차트: {top_chart}")
        print("=" * 60)

    return result_df
