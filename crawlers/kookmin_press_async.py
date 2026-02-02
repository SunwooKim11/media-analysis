"""Async crawler for 국민대신문 (press.kookmin.ac.kr)."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import date

from bs4 import BeautifulSoup, Tag
from tqdm.asyncio import tqdm as async_tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from crawlers.async_base import AsyncCrawler
from utils.async_io import AsyncJsonlWriter

logger = logging.getLogger(__name__)

BASE_URL = "https://press.kookmin.ac.kr"
LIST_URL = f"{BASE_URL}/news/articleList.html"
CUTOFF_DATE = date(2023, 1, 1)
EXCLUDE_KEYWORDS = ["사령", "인사이동", "신간안내", "바로잡습니다", "커버", "알립니다"]


class KookminPressCrawlerAsync(AsyncCrawler):
    """비동기 국민대신문 기사 크롤러 (Producer-Consumer 패턴).

    Parameters
    ----------
    concurrency : int
        동시 상세 페이지 요청 수. Default 5.
    delay : tuple[float, float]
        요청 간 대기 시간(초). Default (0.5, 1.5).
    timeout : int
        페이지 타임아웃(밀리초). Default 30_000.
    """

    def __init__(self, concurrency: int = 5, **kwargs):
        super().__init__(concurrency=concurrency, **kwargs)
        self._stop_event = asyncio.Event()

    # -- static/pure helpers (reused from sync version) ----------------------

    @staticmethod
    def _build_list_url(page: int) -> str:
        return f"{LIST_URL}?page={page}&sc_sub_section_code=S2N1&view_type=sm"

    @staticmethod
    def _parse_date(text: str) -> date | None:
        text = text.strip()
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", text)
        if m:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return None

    def _parse_list_page(self, soup: BeautifulSoup) -> list[dict]:
        """한 페이지의 <li> 목록을 파싱해 기사 메타 리스트를 반환."""
        items: list[dict] = []
        section = soup.select_one("section#section-list")
        if section is None:
            return items

        for li in section.select("ul.type1 > li, ul.type2 > li"):
            a_tag: Tag | None = li.select_one("h2.titles a, H2.titles a")
            if a_tag is None:
                continue

            href = a_tag.get("href", "")
            if not href or "articleView" not in href:
                continue

            title = a_tag.get_text(strip=True)
            url = href if href.startswith("http") else BASE_URL + href

            date_div = li.select_one("div.info.dated")
            if date_div is None:
                byline = li.select_one("div.byline")
                if byline:
                    divs = byline.select("div")
                    date_div = divs[-1] if divs else None
            date_text = date_div.get_text(strip=True) if date_div else ""
            pub_date = self._parse_date(date_text)

            cat_div = li.select_one("div.info.category")
            category = cat_div.get_text(strip=True) if cat_div else ""

            items.append(
                {
                    "url": url,
                    "title": title,
                    "date": date_text,
                    "date_obj": pub_date,
                    "section": category,
                }
            )
        return items

    # -- detail page with retry ----------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _fetch_article_detail(self, url: str) -> dict:
        """상세 페이지에서 제목·본문·작성일·섹션을 추출 (최대 3회 재시도)."""
        soup = await self.fetch_soup(url)

        title = ""
        tag = soup.select_one("header.article-view-header h1.heading")
        if tag:
            title = tag.get_text(strip=True)

        body = ""
        body_div = soup.select_one("div.article-body")
        if body_div:
            for unwanted in body_div.select("script, style, figure figcaption"):
                unwanted.decompose()
            body = body_div.get_text(separator="\n", strip=True)

        date_str = ""
        for li in soup.select("div.info-group ul.infomation li"):
            text = li.get_text(strip=True)
            if "입력" in text:
                date_str = text
                break

        section = ""
        nav = soup.select_one("header.article-view-header > nav")
        if nav:
            links = nav.select("a")
            if len(links) >= 2:
                section = links[-1].get_text(strip=True)

        return {"title": title, "body": body, "date": date_str, "section": section}

    # -- consumer: process one article ---------------------------------------

    async def _process_article(
        self,
        item: dict,
        writer: AsyncJsonlWriter,
        pbar: async_tqdm,
    ) -> None:
        """Consumer: fetch detail → merge → write JSONL."""
        try:
            detail = await self._fetch_article_detail(item["url"])
        except Exception as exc:
            logger.warning("상세 페이지 최종 실패 %s: %s", item["url"], exc)
            detail = None

        if detail:
            item["title"] = detail["title"] or item["title"]
            item["body"] = detail["body"]
            item["date"] = detail["date"] or item["date"]
            item["section"] = detail["section"] or item["section"]
        else:
            item["body"] = ""

        await writer.write_record(item)
        pbar.update(1)

    # -- producer-consumer crawl ---------------------------------------------

    async def crawl(
        self,
        max_results: int = 500,
        output: str = "kookmin_press",
    ) -> int:
        """Producer-Consumer 비동기 크롤링.

        Producer: 리스트 페이지를 순차 요청, 날짜·키워드 필터 후 태스크 생성.
        Consumer: 상세 페이지를 병렬 요청, 파싱 후 JSONL에 스트리밍 저장.

        Returns
        -------
        int
            수집된 기사 수.
        """
        writer = AsyncJsonlWriter(output, batch_size=5)
        pending: set[asyncio.Task] = set()
        collected = 0
        page = 1

        pbar = async_tqdm(total=max_results, desc="국민대신문 비동기 크롤링", unit="건")

        try:
            while not self._stop_event.is_set() and collected < max_results:
                # --- Producer: fetch one list page ---
                url = self._build_list_url(page)
                logger.info("리스트 %d페이지 요청: %s", page, url)

                try:
                    soup = await self.fetch_soup(url)
                except Exception as exc:
                    logger.error("리스트 페이지 실패 (page=%d): %s", page, exc)
                    break

                items = self._parse_list_page(soup)
                if not items:
                    logger.info("더 이상 기사가 없습니다 (page=%d).", page)
                    break

                # --- Filter and dispatch consumers ---
                for item in items:
                    pub_date: date | None = item.pop("date_obj")

                    # 날짜 기반 조기 종료
                    if pub_date and pub_date < CUTOFF_DATE:
                        logger.info(
                            "2023-01-01 이전 기사 발견 (%s) → 크롤링 중단.", pub_date
                        )
                        self._stop_event.set()
                        break

                    # 제외 키워드 필터
                    if any(kw in item["title"] for kw in EXCLUDE_KEYWORDS):
                        logger.debug("제외 키워드 스킵: %s", item["title"])
                        continue

                    if collected >= max_results:
                        break

                    # Consumer 태스크 생성
                    task = asyncio.create_task(
                        self._process_article(item, writer, pbar)
                    )
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                    collected += 1

                    # Backpressure: 대기 태스크가 너무 많으면 일부 완료까지 대기
                    if len(pending) >= self.concurrency * 2:
                        _done, pending = await asyncio.wait(
                            pending, return_when=asyncio.FIRST_COMPLETED
                        )

                page += 1

            # --- 잔여 태스크 완료 대기 ---
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        finally:
            await writer.close()
            pbar.close()

        logger.info("총 %d건 수집 완료.", writer.count)
        return writer.count
