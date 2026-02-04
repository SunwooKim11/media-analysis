"""대학 산업 리스크 뉴스 크롤러.

ESG 이중 중대성 평가를 위한 '외부 환경이 대학에 미치는 영향' 데이터 수집.
14개 키워드를 4개씩 병렬로 크롤링하고, Jaccard Similarity로 중복 제거.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from crawlers.naver_news_web import NaverNewsWebCrawler

# -- 8개 키워드 (ESG 카테고리별, 진단항목 연계) ------------------------------
# 각 키워드가 포괄하는 진단항목:
# - 대학 지역사회 협력: 지역사회 ESG 파트너십, 지역사회 ESG 교육 프로그램
# - 대학 ESG 교육: ESG 비교과, 교육과정, 캠페인, 동아리
# - 그린캠퍼스: 에너지 절감, 재생에너지, 탄소배출량
# - 대학 폐기물: 폐기물, 유해폐기물, 음식물 쓰레기
# - 대학 인권: 인권침해 예방, 다양성/형평성, 취약계층
# - 대학 안전사고: 작업장 건강/안전, 교직원 복지
# - 대학 청렴도: 부패예방, 회계 투명성
# - 대학 ESG 경영: ESG 위원회, 운영위원회 다양성

INDUSTRY_RISK_KEYWORDS = [
    # HEM (대학경영) - 2개
    "대학 지역사회 협력",
    "대학 ESG 교육",
    # E (환경경영) - 2개
    "그린캠퍼스",
    "대학 폐기물",
    # S (사회적 책임) - 2개
    "대학 인권",
    "대학 안전사고",
    # G (지배구조) - 2개
    "대학 청렴도",
    "대학 ESG 경영",
]

DATA_RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
OUTPUT_NAME = "industry_risk_raw"

logger = logging.getLogger(__name__)


# -- Jaccard Similarity 중복 제거 -----------------------------------------

def jaccard_similarity(s1: str, s2: str) -> float:
    """두 문자열의 Jaccard Similarity 계산 (공백 기준 토큰화)."""
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def deduplicate_articles(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Jaccard Similarity 기반 중복 제거.

    동일/유사 기사(threshold 이상)는 하나로 합치고,
    press_count 컬럼에 중복된 횟수(언론사 수) 기록.
    """
    if df.empty:
        df["press_count"] = 0
        return df

    df = df.copy()
    df["title_lower"] = df["title"].str.lower().str.strip()

    # 그룹화: (제목 유사도 기준)
    groups: list[list[int]] = []
    used = set()

    indices = df.index.tolist()
    titles = df["title_lower"].tolist()

    for i, idx_i in enumerate(tqdm(indices, desc="중복 검사", unit="건")):
        if idx_i in used:
            continue

        group = [idx_i]
        used.add(idx_i)

        for j in range(i + 1, len(indices)):
            idx_j = indices[j]
            if idx_j in used:
                continue

            sim = jaccard_similarity(titles[i], titles[j])
            if sim >= threshold:
                group.append(idx_j)
                used.add(idx_j)

        groups.append(group)

    # 그룹별로 대표 기사 선택 (첫 번째) + press_count 계산
    result_rows = []
    for group in groups:
        representative = df.loc[group[0]].to_dict()
        # 언론사 목록 수집 (중복 제거)
        press_list = df.loc[group, "press"].dropna().unique().tolist()
        representative["press_count"] = len(press_list)
        # 언론사들을 쉼표로 합치기 (옵션)
        representative["press_list"] = ", ".join(press_list)
        result_rows.append(representative)

    result = pd.DataFrame(result_rows)
    result.drop(columns=["title_lower"], inplace=True, errors="ignore")

    # 컬럼 순서 정리
    cols = ["keyword", "title", "press", "press_count", "press_list", "date",
            "description", "url", "naver_url"]
    if "content" in result.columns:
        cols.append("content")
    result = result[[c for c in cols if c in result.columns]]

    return result.reset_index(drop=True)


# -- 병렬 크롤링 ----------------------------------------------------------

async def crawl_keyword(
    keyword: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    delay: tuple[float, float],
) -> int:
    """단일 키워드 크롤링."""
    crawler = NaverNewsWebCrawler(
        query=keyword,
        delay=delay,
        skip_filter=True,
    )
    count = await crawler.crawl_list(
        output=OUTPUT_NAME,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        append=True,  # 기존 파일에 추가
    )
    return count


async def crawl_batch(
    keywords: list[str],
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    delay: tuple[float, float],
) -> dict[str, int]:
    """키워드 배치 병렬 크롤링."""
    tasks = [
        crawl_keyword(kw, start_year, start_month, end_year, end_month, delay)
        for kw in keywords
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    counts = {}
    for kw, result in zip(keywords, results):
        if isinstance(result, Exception):
            logger.error("키워드 '%s' 크롤링 실패: %s", kw, result)
            counts[kw] = 0
        else:
            counts[kw] = result
    return counts


async def run_phase1(
    args: argparse.Namespace,
) -> int:
    """Phase 1: 키워드별 병렬 크롤링."""
    csv_path = DATA_RAW_DIR / f"{OUTPUT_NAME}.csv"

    # 기존 파일 삭제 (새로 시작)
    if csv_path.exists():
        csv_path.unlink()

    sy, sm = map(int, args.start_date.split("."))
    ey, em = map(int, args.end_date.split("."))
    delay = (args.delay_min, args.delay_max)

    total_count = 0
    batch_size = args.parallel

    # 키워드를 batch_size씩 나눠서 병렬 실행
    for i in range(0, len(INDUSTRY_RISK_KEYWORDS), batch_size):
        batch = INDUSTRY_RISK_KEYWORDS[i : i + batch_size]
        print(f"\n[배치 {i // batch_size + 1}] 키워드: {batch}")

        counts = await crawl_batch(batch, sy, sm, ey, em, delay)
        for kw, cnt in counts.items():
            print(f"  - {kw}: {cnt}건")
            total_count += cnt

    print(f"\nPhase 1 완료: 총 {total_count}건 (중복 포함)")
    return total_count


def run_deduplication() -> int:
    """중복 제거 단계."""
    csv_path = DATA_RAW_DIR / f"{OUTPUT_NAME}.csv"

    if not csv_path.exists():
        logger.error("CSV 파일 없음: %s", csv_path)
        return 0

    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
    df.fillna("", inplace=True)

    original_count = len(df)
    print(f"\n중복 제거 시작: {original_count}건")

    df_dedup = deduplicate_articles(df, threshold=0.8)
    dedup_count = len(df_dedup)

    # 저장
    df_dedup.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
    )

    print(f"중복 제거 완료: {original_count}건 → {dedup_count}건 ({original_count - dedup_count}건 제거)")
    return dedup_count


async def run_phase2(args: argparse.Namespace) -> int:
    """Phase 2: 본문 추출."""
    crawler = NaverNewsWebCrawler(
        body_concurrency=args.body_concurrency,
        body_delay=(args.body_delay_min, args.body_delay_max),
        skip_filter=True,
    )
    success = await crawler.extract_bodies(output=OUTPUT_NAME)
    return success


async def run(args: argparse.Namespace) -> None:
    """메인 실행."""
    if args.phase in ("list", "all"):
        await run_phase1(args)

    if args.phase in ("dedup", "all"):
        run_deduplication()

    if args.phase in ("body", "all"):
        success = await run_phase2(args)
        print(f"\nPhase 2 완료: {success}건 본문 추출")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="대학 산업 리스크 뉴스 크롤러 (ESG 이중 중대성 평가용)"
    )
    parser.add_argument(
        "--phase",
        choices=["list", "dedup", "body", "all"],
        default="all",
        help="실행 단계: list(리스트만), dedup(중복제거만), body(본문만), all(전체) (기본: all)",
    )
    parser.add_argument(
        "--start-date",
        default="2023.01",
        help="시작 연월 (기본: 2023.01)",
    )
    parser.add_argument(
        "--end-date",
        default="2026.02",
        help="종료 연월 (기본: 2026.02)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="병렬 크롤링 키워드 수 (기본: 4)",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=1.0,
        help="리스트 요청 간 최소 대기(초) (기본: 1.0)",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=2.0,
        help="리스트 요청 간 최대 대기(초) (기본: 2.0)",
    )
    parser.add_argument(
        "--body-concurrency",
        type=int,
        default=10,
        help="본문 추출 동시 요청 수 (기본: 10)",
    )
    parser.add_argument(
        "--body-delay-min",
        type=float,
        default=0.3,
        help="본문 요청 간 최소 대기(초) (기본: 0.3)",
    )
    parser.add_argument(
        "--body-delay-max",
        type=float,
        default=0.8,
        help="본문 요청 간 최대 대기(초) (기본: 0.8)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
