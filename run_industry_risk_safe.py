"""산업 리스크 키워드 크롤링 (안전 모드 + 중복 제거).

순차 처리 + 긴 딜레이로 403 방지.
제목 중복 제거로 본문 추출 대상 최소화.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from pathlib import Path

from crawlers.naver_news_web import NaverNewsWebCrawler

logger = logging.getLogger(__name__)

# 키워드 목록
KEYWORDS = [
    "대학 지역사회 협력",
    "대학 ESG 교육",
    "그린캠퍼스",
    "대학 폐기물",
    "대학 인권",
    "대학 안전사고",
    "대학 청렴도",
    "대학 ESG 경영",
]

# 날짜 범위
START_YEAR = 2023
START_MONTH = 1
END_YEAR = 2025
END_MONTH = 12

# 출력 경로
OUTPUT_DIR = Path("data/raw")


def normalize_title(title: str) -> str:
    """제목 정규화 (중복 비교용).

    - 유니코드 정규화 (NFC)
    - 소문자 변환
    - 공백/특수문자 제거
    - 숫자 유지 (날짜 구분용)
    """
    # 유니코드 정규화
    title = unicodedata.normalize("NFC", title)
    # 소문자
    title = title.lower()
    # 한글, 영문, 숫자만 유지
    title = re.sub(r"[^\uac00-\ud7af\u1100-\u11ff\u3130-\u318fa-z0-9]", "", title)
    return title


class DeduplicatedCrawler:
    """중복 제거 기능이 있는 크롤러."""

    def __init__(self):
        self.seen_titles: dict[str, dict] = {}  # normalized_title -> {article, count}
        self.total_raw = 0
        self.total_unique = 0

    def add_article(self, article: dict) -> bool:
        """기사 추가. 중복이면 count 증가, 새 기사면 추가.

        Returns: True if new article, False if duplicate
        """
        self.total_raw += 1
        normalized = normalize_title(article.get("title", ""))

        if not normalized:
            return False

        if normalized in self.seen_titles:
            self.seen_titles[normalized]["count"] += 1
            return False

        self.seen_titles[normalized] = {
            "article": article,
            "count": 1,
        }
        self.total_unique += 1
        return True

    def get_unique_articles(self) -> list[dict]:
        """고유 기사 목록 반환 (count 포함)."""
        result = []
        for data in self.seen_titles.values():
            article = data["article"].copy()
            article["duplicate_count"] = data["count"]
            result.append(article)
        return result

    def get_stats(self) -> dict:
        """통계 반환."""
        return {
            "total_raw": self.total_raw,
            "total_unique": self.total_unique,
            "duplicates_removed": self.total_raw - self.total_unique,
            "dedup_rate": (1 - self.total_unique / max(self.total_raw, 1)) * 100,
        }


async def crawl_keyword_safe(
    keyword: str,
    deduplicator: DeduplicatedCrawler,
) -> tuple[int, int]:
    """단일 키워드 안전 크롤링 (순차 처리 + 긴 딜레이).

    Returns: (raw_count, unique_count)
    """
    crawler = NaverNewsWebCrawler(
        query=keyword,
        delay=(2.0, 3.0),  # 페이지 간 2~3초 딜레이 (요청 수 감소로 완화)
        skip_filter=True,
        sort="0",  # 관련도순 (월 전체에 분산)
        max_pages_per_month=10,  # 월당 최대 100건 (10페이지)
    )

    # 순차 처리 (crawl_list 사용, 병렬 X)
    temp_csv = OUTPUT_DIR / f"_temp_{keyword.replace(' ', '_')}.csv"

    count = await crawler.crawl_list(
        output=temp_csv.stem,
        start_year=START_YEAR,
        start_month=START_MONTH,
        end_year=END_YEAR,
        end_month=END_MONTH,
        append=False,
    )

    if count == 0:
        logger.warning("키워드 '%s' 수집 건수 0", keyword)
        if temp_csv.exists():
            temp_csv.unlink()
        return 0, 0

    # CSV 읽어서 중복 제거 + 키워드별 JSONL 저장
    import pandas as pd
    df = pd.read_csv(temp_csv, encoding="utf-8-sig", dtype=str)
    df.fillna("", inplace=True)

    # 키워드별 출력 파일
    safe_keyword = keyword.replace(" ", "_")
    output_file = OUTPUT_DIR / f"industry_risk_{safe_keyword}.jsonl"

    new_count = 0
    unique_articles = []

    for _, row in df.iterrows():
        article = {
            "keyword": keyword,
            "title": row.get("title", ""),
            "press": row.get("press", ""),
            "date": row.get("date", ""),
            "description": row.get("description", ""),
            "url": row.get("url", ""),
            "naver_url": row.get("naver_url", ""),
        }
        if deduplicator.add_article(article):
            unique_articles.append(article)
            new_count += 1

    # 키워드별 JSONL 저장 (기존 파일 덮어쓰기)
    with open(output_file, "w", encoding="utf-8") as f:
        for article in unique_articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    # 임시 CSV 삭제
    temp_csv.unlink()

    logger.info(
        "키워드 '%s': %d건 수집, %d건 신규 → %s",
        keyword, count, new_count, output_file.name
    )
    return count, new_count


async def main():
    """메인 함수."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("산업 리스크 키워드 크롤링 (최적화 모드)")
    print(f"기간: {START_YEAR}.{START_MONTH:02d} ~ {END_YEAR}.{END_MONTH:02d}")
    print(f"키워드: {len(KEYWORDS)}개")
    print("=" * 60)
    print("설정:")
    print("  - 정렬: 관련도순 (sort=0)")
    print("  - 월당 최대 100건 (10페이지)")
    print("  - 페이지 간 2~3초 딜레이")
    print("  - 제목 기준 실시간 중복 제거")
    print("  - 키워드별 개별 JSONL 파일 저장")
    print(f"  - 예상: ~{36 * len(KEYWORDS) * 10 * 2.5 / 60:.0f}분")
    print("=" * 60)

    deduplicator = DeduplicatedCrawler()
    results = {}

    for i, keyword in enumerate(KEYWORDS, 1):
        print(f"\n[{i}/{len(KEYWORDS)}] {keyword}")
        print("-" * 40)

        raw_count, unique_count = await crawl_keyword_safe(keyword, deduplicator)
        results[keyword] = {"raw": raw_count, "unique": unique_count}

        stats = deduplicator.get_stats()
        safe_kw = keyword.replace(" ", "_")
        print(f"수집: {raw_count}건, 신규: {unique_count}건 → industry_risk_{safe_kw}.jsonl")
        print(f"전체 누적: {stats['total_unique']}건 (중복제거율: {stats['dedup_rate']:.1f}%)")

    # 최종 통계
    stats = deduplicator.get_stats()

    print("\n" + "=" * 60)
    print("크롤링 완료")
    print("=" * 60)
    print("\n키워드별 결과:")
    for kw, counts in results.items():
        safe_kw = kw.replace(" ", "_")
        print(f"  - {kw}: {counts['unique']}건 → industry_risk_{safe_kw}.jsonl")

    print(f"\n총계:")
    print(f"  - 수집된 원본 기사: {stats['total_raw']}건")
    print(f"  - 중복 제거 후: {stats['total_unique']}건")
    print(f"  - 제거된 중복: {stats['duplicates_removed']}건 ({stats['dedup_rate']:.1f}%)")
    print(f"\n저장 위치: {OUTPUT_DIR}/industry_risk_*.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
