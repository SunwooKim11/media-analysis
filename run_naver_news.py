"""네이버 뉴스 크롤러 실행 스크립트."""

import argparse
import asyncio
import logging

from crawlers.naver_news import NaverNewsCrawlerAsync


async def run(args: argparse.Namespace) -> None:
    async with NaverNewsCrawlerAsync(
        query=args.query,
        concurrency=args.concurrency,
        delay=(args.delay_min, args.delay_max),
    ) as crawler:
        count = await crawler.crawl(
            max_results=args.max_results,
            output=args.output,
        )
    print(f"\n수집 완료: {count}건 → data/raw/{args.output}.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="네이버 뉴스 API 크롤러")
    parser.add_argument(
        "-q", "--query", default="국민대학교",
        help="검색 키워드 (기본: 국민대학교)",
    )
    parser.add_argument(
        "-n", "--max-results", type=int, default=1000,
        help="최대 수집 건수 (기본: 1000, API 한도 1000)",
    )
    parser.add_argument(
        "-o", "--output", default="kookmin_esg_news",
        help="출력 파일명 (확장자 제외, 기본: kookmin_esg_news)",
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=10,
        help="본문 추출 동시 요청 수 (기본: 10)",
    )
    parser.add_argument(
        "--delay-min", type=float, default=0.3,
        help="본문 요청 간 최소 대기(초)",
    )
    parser.add_argument(
        "--delay-max", type=float, default=0.8,
        help="본문 요청 간 최대 대기(초)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
