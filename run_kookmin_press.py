"""국민대신문 크롤러 실행 스크립트."""

import argparse
import logging

from crawlers.kookmin_press import KookminPressCrawler


def main() -> None:
    parser = argparse.ArgumentParser(description="국민대신문 크롤러")
    parser.add_argument(
        "-n",
        "--max-results",
        type=int,
        default=500,
        help="최대 수집 건수 (기본: 500)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="kookmin_press",
        help="출력 파일명 (확장자 제외, 기본: kookmin_press)",
    )
    parser.add_argument(
        "--delay-min", type=float, default=1.0, help="요청 간 최소 대기(초)"
    )
    parser.add_argument(
        "--delay-max", type=float, default=3.0, help="요청 간 최대 대기(초)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    crawler = KookminPressCrawler(delay=(args.delay_min, args.delay_max))
    count = crawler.crawl_and_save(
        filename=args.output,
        max_results=args.max_results,
    )
    print(f"\n수집 완료: {count}건 → data/raw/{args.output}.jsonl")


if __name__ == "__main__":
    main()
