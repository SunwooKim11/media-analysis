"""Async crawler for Naver News API + body extraction."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from html import unescape

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm.asyncio import tqdm as async_tqdm
import trafilatura

from utils.async_io import AsyncJsonlWriter

logger = logging.getLogger(__name__)

load_dotenv()

NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"
CUTOFF_DATE = date(2023, 1, 1)
DEFAULT_QUERY = "국민대"

# "국민대"를 포함하지만 국민대학교와 무관한 복합어 필터
_SKIP_PATTERN = re.compile(
    r"국민대중|국민대출|국민대상|국민대통합|국민대책|국민대다수|국민대화|국민대축제"
)


def _strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities."""
    clean = re.sub(r"<[^>]+>", "", text)
    return unescape(clean).strip()


def _parse_rfc822_date(date_str: str) -> date | None:
    """Parse RFC 822 date string (e.g. 'Mon, 01 Jan 2024 09:00:00 +0900')."""
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.date()
    except Exception:
        return None


class NaverNewsCrawlerAsync:
    """비동기 네이버 뉴스 API 크롤러.

    Parameters
    ----------
    query : str
        검색 키워드. Default "국민대".
    concurrency : int
        본문 추출 동시 요청 수. Default 10.
    delay : tuple[float, float]
        본문 요청 간 대기 시간(초). Default (0.3, 0.8).
    """

    def __init__(
        self,
        query: str = DEFAULT_QUERY,
        concurrency: int = 10,
        delay: tuple[float, float] = (0.3, 0.8),
    ):
        self.query = query
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

    # -- Naver API pagination --------------------------------------------------

    async def _search_page(self, start: int, display: int = 100) -> dict:
        """Naver Search API 호출 (1회)."""
        params = {
            "query": self.query,
            "display": display,
            "start": start,
            "sort": "date",
        }
        async with self._session.get(NAVER_API_URL, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    # -- body extraction -------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _fetch_body(self, url: str) -> str:
        """URL에서 기사 본문을 추출.

        - news.naver.com → BS4 파싱 (#newsct_article)
        - 그 외 → trafilatura fallback
        """
        async with self._semaphore:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
            async with self._session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                resp.raise_for_status()
                html = await resp.text()

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

    # -- consumer: process one article -----------------------------------------

    async def _process_article(
        self,
        item: dict,
        writer: AsyncJsonlWriter,
        pbar: async_tqdm,
    ) -> None:
        """본문 추출 후 JSONL에 저장."""
        body_url = item.get("naver_url") or item["url"]
        try:
            body = await self._fetch_body(body_url)
        except Exception as exc:
            logger.warning("본문 추출 실패 %s: %s", body_url, exc)
            body = ""

        # naver_url에서 실패 시 originallink로 재시도
        if not body and item.get("url") != body_url:
            try:
                body = await self._fetch_body(item["url"])
            except Exception as exc:
                logger.warning("원본 링크 본문 추출도 실패 %s: %s", item["url"], exc)
                body = ""

        record = {
            "date": item["date"],
            "title": item["title"],
            "press": item["press"],
            "content": body,
            "url": item["url"],
            "naver_url": item.get("naver_url", ""),
            "section": "",
        }
        await writer.write_record(record)
        pbar.update(1)

    # -- main crawl ------------------------------------------------------------

    async def crawl(
        self,
        max_results: int = 1000,
        output: str = "kookmin_esg_news",
    ) -> int:
        """네이버 뉴스 API 크롤링 + 본문 추출.

        Returns
        -------
        int
            수집된 기사 수.
        """
        writer = AsyncJsonlWriter(output, batch_size=10)
        pending: set[asyncio.Task] = set()
        collected = 0
        start = 1
        display = 100

        pbar = async_tqdm(total=max_results, desc="네이버뉴스 크롤링", unit="건")

        try:
            while collected < max_results and start <= 1000:
                logger.info("API 요청: start=%d, display=%d", start, display)

                try:
                    data = await self._search_page(start, display)
                except Exception as exc:
                    logger.error("API 요청 실패 (start=%d): %s", start, exc)
                    break

                items = data.get("items", [])
                if not items:
                    logger.info("더 이상 검색 결과가 없습니다 (start=%d).", start)
                    break

                stop = False
                for raw in items:
                    # 날짜 파싱 및 필터
                    pub_date = _parse_rfc822_date(raw.get("pubDate", ""))
                    if pub_date and pub_date < CUTOFF_DATE:
                        logger.info(
                            "2023-01-01 이전 기사 발견 (%s) → 크롤링 중단.", pub_date
                        )
                        stop = True
                        break

                    if collected >= max_results:
                        break

                    title = _strip_html(raw.get("title", ""))

                    # skip list 필터: 국민대학교와 무관한 복합어 제외
                    if _SKIP_PATTERN.search(title):
                        logger.debug("skip list 필터: %s", title)
                        continue

                    date_str = (
                        pub_date.isoformat() if pub_date else raw.get("pubDate", "")
                    )

                    item = {
                        "title": title,
                        "date": date_str,
                        "press": "",  # API 응답에 언론사 정보 없음
                        "url": raw.get("originallink", ""),
                        "naver_url": raw.get("link", ""),
                    }

                    task = asyncio.create_task(
                        self._process_article(item, writer, pbar)
                    )
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                    collected += 1

                    # Backpressure
                    if len(pending) >= self.concurrency * 2:
                        _done, pending = await asyncio.wait(
                            pending, return_when=asyncio.FIRST_COMPLETED
                        )

                if stop:
                    break

                start += display

            # 잔여 태스크 완료 대기
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        finally:
            await writer.close()
            pbar.close()

        logger.info("총 %d건 수집 완료.", writer.count)
        return writer.count
