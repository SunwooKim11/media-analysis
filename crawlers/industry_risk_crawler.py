"""산업 리스크 키워드 병렬 크롤러 (JSONL 출력)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as async_tqdm
import trafilatura

logger = logging.getLogger(__name__)
load_dotenv()

NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"

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
START_DATE = date(2023, 1, 1)
END_DATE = date(2025, 12, 31)


def _strip_html(text: str) -> str:
    """HTML 태그 제거."""
    clean = re.sub(r"<[^>]+>", "", text)
    return unescape(clean).strip()


def _parse_rfc822_date(date_str: str) -> date | None:
    """RFC 822 날짜 파싱."""
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.date()
    except Exception:
        return None


class IndustryRiskCrawler:
    """산업 리스크 키워드 크롤러."""

    def __init__(
        self,
        concurrency: int = 10,
        delay: tuple[float, float] = (0.3, 0.8),
    ):
        self.concurrency = concurrency
        self.delay = delay
        self._semaphore = asyncio.Semaphore(concurrency)
        self._session: aiohttp.ClientSession | None = None

        self._client_id = os.getenv("NAVER_CLIENT_ID", "")
        self._client_secret = os.getenv("NAVER_CLIENT_SECRET", "")
        if not self._client_id or not self._client_secret:
            raise ValueError(
                "NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수를 .env에 설정하세요."
            )

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            headers={
                "X-Naver-Client-Id": self._client_id,
                "X-Naver-Client-Secret": self._client_secret,
            }
        )
        return self

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()

    async def _search_page(self, query: str, start: int, display: int = 100) -> dict:
        """Naver Search API 호출."""
        params = {
            "query": query,
            "display": display,
            "start": start,
            "sort": "date",
        }
        async with self._session.get(NAVER_API_URL, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _fetch_body(self, url: str) -> str:
        """본문 추출."""
        async with self._semaphore:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
            try:
                async with self._session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                    ssl=False,
                ) as resp:
                    if resp.status != 200:
                        return ""
                    html = await resp.text()
            except Exception as exc:
                logger.debug("본문 요청 실패 %s: %s", url, exc)
                return ""

            await asyncio.sleep(
                __import__("random").uniform(*self.delay)
            )

            # 네이버 뉴스 본문
            if "news.naver.com" in url:
                soup = BeautifulSoup(html, "html.parser")
                article = soup.select_one("#newsct_article, #articeBody, #articleBodyContents")
                if article:
                    for tag in article.select("script, style, span.end_photo_org"):
                        tag.decompose()
                    return article.get_text(separator="\n", strip=True)

            # 외부 링크 → trafilatura
            result = trafilatura.extract(html)
            return result or ""

    async def crawl_keyword(
        self,
        keyword: str,
        output_file: Path,
        pbar: async_tqdm,
    ) -> int:
        """단일 키워드 크롤링."""
        collected = 0
        start = 1
        display = 100
        max_start = 1000  # Naver API 제한

        while start <= max_start:
            try:
                data = await self._search_page(keyword, start, display)
            except Exception as exc:
                logger.warning("API 오류 [%s] start=%d: %s", keyword, start, exc)
                break

            items = data.get("items", [])
            if not items:
                break

            stop = False
            records = []

            for raw in items:
                pub_date = _parse_rfc822_date(raw.get("pubDate", ""))

                # 날짜 범위 필터
                if pub_date:
                    if pub_date < START_DATE:
                        stop = True
                        break
                    if pub_date > END_DATE:
                        continue

                title = _strip_html(raw.get("title", ""))
                description = _strip_html(raw.get("description", ""))

                # 본문 추출
                naver_url = raw.get("link", "")
                original_url = raw.get("originallink", "")
                body_url = naver_url or original_url

                body = ""
                if body_url:
                    body = await self._fetch_body(body_url)
                    # naver_url 실패 시 original로 재시도
                    if not body and original_url and original_url != naver_url:
                        body = await self._fetch_body(original_url)

                record = {
                    "keyword": keyword,
                    "title": title,
                    "press": "",  # API에서 제공 안함
                    "date": pub_date.isoformat() if pub_date else "",
                    "description": description,
                    "url": original_url,
                    "naver_url": naver_url,
                    "content": body,
                }
                records.append(record)
                collected += 1
                pbar.update(1)

            # 파일에 추가
            if records:
                with open(output_file, "a", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if stop:
                break

            start += display
            await asyncio.sleep(0.1)  # Rate limiting

        return collected

    async def crawl_all(
        self,
        keywords: list[str] | None = None,
        output_dir: str | Path = "data/raw",
    ) -> dict[str, int]:
        """모든 키워드 병렬 크롤링."""
        if keywords is None:
            keywords = KEYWORDS

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 출력 파일 초기화
        output_file = output_dir / "industry_risk.jsonl"
        if output_file.exists():
            output_file.unlink()

        # 전체 예상 건수 (키워드당 ~1000건)
        total_estimate = len(keywords) * 1000
        pbar = async_tqdm(total=total_estimate, desc="산업리스크 크롤링", unit="건")

        results = {}
        for keyword in keywords:
            pbar.set_description(f"[{keyword}]")
            count = await self.crawl_keyword(keyword, output_file, pbar)
            results[keyword] = count
            logger.info("키워드 '%s' 완료: %d건", keyword, count)

        pbar.close()

        # 결과 요약
        total = sum(results.values())
        print(f"\n총 {total}건 크롤링 완료")
        for kw, cnt in results.items():
            print(f"  - {kw}: {cnt}건")

        return results


async def main():
    """메인 함수."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    async with IndustryRiskCrawler(concurrency=10) as crawler:
        await crawler.crawl_all()


if __name__ == "__main__":
    asyncio.run(main())
