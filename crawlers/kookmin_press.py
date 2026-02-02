"""Crawler for 국민대신문 (press.kookmin.ac.kr)."""

from __future__ import annotations

import re
import logging
from datetime import date, datetime

from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from crawlers.base import Crawler
from utils.io import save_jsonl

logger = logging.getLogger(__name__)

BASE_URL = "https://press.kookmin.ac.kr"
LIST_URL = f"{BASE_URL}/news/articleList.html"
CUTOFF_DATE = date(2023, 1, 1)
EXCLUDE_KEYWORDS = ["사령", "인사이동", "신간안내", "바로잡습니다", "커버", "알립니다"]


class KookminPressCrawler(Crawler):
    """국민대신문 기사 크롤러.

    Parameters
    ----------
    delay : tuple[float, float]
        요청 간 대기 시간(초). 기본 (1.0, 3.0).
    timeout : int
        요청 타임아웃(초). 기본 30.
    """

    # -- list page helpers ---------------------------------------------------

    def _build_list_url(self, page: int) -> str:
        return f"{LIST_URL}?page={page}&sc_sub_section_code=S2N1&view_type=sm"

    @staticmethod
    def _parse_date(text: str) -> date | None:
        """'2025-12-22' 같은 문자열을 date 객체로 변환."""
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
                # view_type=sm: 날짜가 div.byline 내 마지막 div에 위치
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

    # -- detail page helpers -------------------------------------------------

    def _parse_article(self, url: str) -> dict | None:
        """상세 페이지에서 제목·본문·작성일·섹션을 추출."""
        try:
            soup = self.fetch_soup(url)
        except Exception as exc:
            logger.warning("상세 페이지 요청 실패 %s: %s", url, exc)
            return None

        # 제목: header.article-view-header > h1.heading
        title = ""
        tag = soup.select_one("header.article-view-header h1.heading")
        if tag:
            title = tag.get_text(strip=True)

        # 본문: div.article-body (snsAnchor 내부)
        body = ""
        body_div = soup.select_one("div.article-body")
        if body_div:
            for unwanted in body_div.select("script, style, figure figcaption"):
                unwanted.decompose()
            body = body_div.get_text(separator="\n", strip=True)

        # 작성일: div.info-group ul.infomation li 중 "입력" 포함 항목
        date_str = ""
        for li in soup.select("div.info-group ul.infomation li"):
            text = li.get_text(strip=True)
            if "입력" in text:
                date_str = text
                break

        # 섹션: header.article-view-header > nav 내 마지막 <a> (홈 제외)
        section = ""
        nav = soup.select_one("header.article-view-header > nav")
        if nav:
            links = nav.select("a")
            if len(links) >= 2:
                section = links[-1].get_text(strip=True)

        return {
            "title": title,
            "body": body,
            "date": date_str,
            "section": section,
        }

    # -- public API ----------------------------------------------------------

    def crawl(
        self,
        query: str = "",
        max_results: int = 500,
    ) -> list[dict]:
        """기사 리스트를 페이지별로 순회하며 크롤링.

        ``query`` 는 이 크롤러에서는 사용하지 않음 (전체 기사 대상).

        2023-01-01 이전 기사가 나오면 즉시 중단한다.
        """
        articles: list[dict] = []
        page = 1
        stop = False

        pbar = tqdm(desc="국민대신문 크롤링", unit="건")

        while not stop and len(articles) < max_results:
            url = self._build_list_url(page)
            logger.info("리스트 %d페이지 요청: %s", page, url)

            try:
                soup = self.fetch_soup(url)
            except Exception as exc:
                logger.error("리스트 페이지 요청 실패 (page=%d): %s", page, exc)
                break

            items = self._parse_list_page(soup)
            if not items:
                logger.info("더 이상 기사가 없습니다 (page=%d).", page)
                break

            for item in items:
                pub_date: date | None = item.pop("date_obj")

                # 날짜 기반 조기 종료
                if pub_date and pub_date < CUTOFF_DATE:
                    logger.info(
                        "2023-01-01 이전 기사 발견 (%s) → 크롤링 중단.", pub_date
                    )
                    stop = True
                    break

                # 제외 키워드 필터
                if any(kw in item["title"] for kw in EXCLUDE_KEYWORDS):
                    logger.debug("제외 키워드 포함 → 스킵: %s", item["title"])
                    continue

                # 상세 페이지 파싱
                detail = self._parse_article(item["url"])
                if detail:
                    # 상세 페이지 정보로 보강 (리스트 정보를 fallback으로 유지)
                    item["title"] = detail["title"] or item["title"]
                    item["body"] = detail["body"]
                    item["date"] = detail["date"] or item["date"]
                    item["section"] = detail["section"] or item["section"]
                else:
                    item["body"] = ""

                articles.append(item)
                pbar.update(1)

                if len(articles) >= max_results:
                    break

            page += 1

        pbar.close()
        logger.info("총 %d건 수집 완료.", len(articles))
        return articles

    def crawl_and_save(
        self,
        filename: str = "kookmin_press",
        max_results: int = 500,
    ) -> int:
        """크롤링 후 data/raw/ 에 JSONL로 저장. 수집 건수를 반환."""
        articles = self.crawl(max_results=max_results)
        if articles:
            path = save_jsonl(articles, filename)
            logger.info("저장 완료: %s (%d건)", path, len(articles))
        return len(articles)
