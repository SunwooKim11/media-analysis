"""네이버 뉴스 웹 크롤러 실행 스크립트.

Phase 1: 뉴스 리스트 수집 (search.naver.com 무한스크롤 API)
Phase 2: 본문 추출 (aiohttp + trafilatura)
"""

import argparse
import asyncio
import logging

from crawlers.naver_news_web import NaverNewsWebCrawler


async def run(args: argparse.Namespace) -> None:
    crawler = NaverNewsWebCrawler(
        query=args.query,
        delay=(args.delay_min, args.delay_max),
        body_concurrency=args.body_concurrency,
        body_delay=(args.body_delay_min, args.body_delay_max),
    )

    if args.phase in ("list", "all"):
        sy, sm = map(int, args.start_date.split("."))
        ey, em = map(int, args.end_date.split("."))
        count = await crawler.crawl_list(
            output=args.output,
            start_year=sy,
            start_month=sm,
            end_year=ey,
            end_month=em,
        )
        print(f"\nPhase 1 완료: {count}건 → data/raw/{args.output}.csv")

    if args.phase in ("body", "all"):
        success = await crawler.extract_bodies(output=args.output)
        print(f"\nPhase 2 완료: {success}건 본문 추출")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="네이버 뉴스 웹 크롤러 (무한스크롤 API)"
    )
    parser.add_argument(
        "--phase",
        choices=["list", "body", "all"],
        default="all",
        help="실행 단계: list(리스트만), body(본문만), all(전체) (기본: all)",
    )
    parser.add_argument(
        "-q", "--query", default="국민대",
        help="검색 키워드 (기본: 국민대)",
    )
    parser.add_argument(
        "-o", "--output", default="kookmin_esg_raw",
        help="출력 파일명 (확장자 제외, 기본: kookmin_esg_raw)",
    )
    parser.add_argument(
        "--start-date", default="2023.01",
        help="시작 연월 (기본: 2023.01)",
    )
    parser.add_argument(
        "--end-date", default="2026.02",
        help="종료 연월 (기본: 2026.02)",
    )
    parser.add_argument(
        "--delay-min", type=float, default=1.0,
        help="리스트 요청 간 최소 대기(초) (기본: 1.0)",
    )
    parser.add_argument(
        "--delay-max", type=float, default=2.0,
        help="리스트 요청 간 최대 대기(초) (기본: 2.0)",
    )
    parser.add_argument(
        "--body-concurrency", type=int, default=10,
        help="본문 추출 동시 요청 수 (기본: 10)",
    )
    parser.add_argument(
        "--body-delay-min", type=float, default=0.3,
        help="본문 요청 간 최소 대기(초) (기본: 0.3)",
    )
    parser.add_argument(
        "--body-delay-max", type=float, default=0.8,
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
