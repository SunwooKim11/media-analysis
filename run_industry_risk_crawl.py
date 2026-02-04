"""산업 리스크 키워드 크롤링 (JSONL 출력).

네이버 뉴스 웹 무한스크롤 API 기반.
"""

from __future__ import annotations

import asyncio
import json
import logging
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


async def crawl_keyword_list_only(keyword: str, output_file: Path) -> int:
    """단일 키워드 리스트 수집 (본문 추출 생략)."""
    crawler = NaverNewsWebCrawler(
        query=keyword,
        delay=(1.5, 2.5),  # 요청 간격 증가
        skip_filter=True,  # 국민대 필터 비활성화
    )

    # 리스트 수집 (CSV) - 월별 병렬 처리 (동시성 낮춤)
    temp_csv = OUTPUT_DIR / f"_temp_{keyword.replace(' ', '_')}.csv"
    count = await crawler.crawl_list_parallel(
        output=temp_csv.stem,
        start_year=START_YEAR,
        start_month=START_MONTH,
        end_year=END_YEAR,
        end_month=END_MONTH,
        append=False,
        month_concurrency=3,  # 3개월 동시 처리 (403 방지)
    )

    if count == 0:
        logger.warning("키워드 '%s' 수집 건수 0", keyword)
        if temp_csv.exists():
            temp_csv.unlink()
        return 0

    # CSV → JSONL 변환 및 저장 (본문은 빈 문자열)
    import pandas as pd
    df = pd.read_csv(temp_csv, encoding="utf-8-sig", dtype=str)
    df.fillna("", inplace=True)

    with open(output_file, "a", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "keyword": keyword,
                "title": row.get("title", ""),
                "press": row.get("press", ""),
                "date": row.get("date", ""),
                "description": row.get("description", ""),
                "url": row.get("url", ""),
                "naver_url": row.get("naver_url", ""),
                "content": "",  # 본문은 나중에 별도 추출
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 임시 CSV 삭제
    temp_csv.unlink()
    return count


async def main():
    """메인 함수."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "industry_risk.jsonl"

    # 기존 파일 삭제
    if output_file.exists():
        output_file.unlink()

    print("=" * 60)
    print("산업 리스크 키워드 크롤링 (리스트 수집만)")
    print(f"기간: {START_YEAR}.{START_MONTH:02d} ~ {END_YEAR}.{END_MONTH:02d}")
    print(f"키워드: {len(KEYWORDS)}개")
    print("※ 본문 추출은 별도 실행")
    print("=" * 60)

    total = 0
    results = {}

    for i, keyword in enumerate(KEYWORDS, 1):
        print(f"\n[{i}/{len(KEYWORDS)}] {keyword}")
        print("-" * 40)

        count = await crawl_keyword_list_only(keyword, output_file)
        results[keyword] = count
        total += count

        print(f"완료: {count}건 (누적: {total}건)")

    print("\n" + "=" * 60)
    print("크롤링 완료")
    print("=" * 60)
    for kw, cnt in results.items():
        print(f"  - {kw}: {cnt}건")
    print(f"\n총 {total}건 → {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
