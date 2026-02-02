"""비동기 국민대신문 크롤러 실행 스크립트."""

import argparse
import asyncio
import logging

from crawlers.kookmin_press_async import KookminPressCrawlerAsync


async def run(args: argparse.Namespace) -> None:
    async with KookminPressCrawlerAsync(
        concurrency=args.concurrency,
        delay=(args.delay_min, args.delay_max),
        timeout=args.timeout,
    ) as crawler:
        count = await crawler.crawl(
            max_results=args.max_results,
            output=args.output,
        )
    print(f"\n수집 완료: {count}건 → data/raw/{args.output}.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="국민대신문 비동기 크롤러")
    parser.add_argument(
        "-n", "--max-results", type=int, default=500,
        help="최대 수집 건수 (기본: 500)",
    )
    parser.add_argument(
        "-o", "--output", default="kookmin_press",
        help="출력 파일명 (확장자 제외, 기본: kookmin_press)",
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=5,
        help="동시 요청 수 (기본: 5)",
    )
    parser.add_argument(
        "--delay-min", type=float, default=0.5,
        help="요청 간 최소 대기(초)",
    )
    parser.add_argument(
        "--delay-max", type=float, default=1.5,
        help="요청 간 최대 대기(초)",
    )
    parser.add_argument(
        "--timeout", type=int, default=30000,
        help="페이지 타임아웃 밀리초 (기본: 30000)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
