"""네이버 뉴스 웹 크롤러 (무한스크롤 API 기반).

Phase 1: search.naver.com 뉴스 검색 결과를 월별로 수집 → CSV
Phase 2: CSV의 URL에서 기사 본문 추출 → content 컬럼 추가
"""

from __future__ import annotations

import asyncio
import calendar
import csv
import logging
import random
import re
from datetime import date
from pathlib import Path

import aiohttp
import httpx
import pandas as pd
import trafilatura
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

logger = logging.getLogger(__name__)

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

_SKIP_PATTERN = re.compile(
    r"국민대중|국민대출|국민대상|국민대통합|국민대책|국민대다수|국민대화|국민대축제"
)

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

SEARCH_URL = "https://search.naver.com/search.naver"
MORE_API_URL = "https://s.search.naver.com/p/newssearch/3/api/tab/more"


def _clean_text(text: str) -> str:
    """HTML 태그 제거 및 공백 정리."""
    clean = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", clean).strip()


class NaverNewsWebCrawler:
    """네이버 뉴스 검색 웹 크롤러.

    Parameters
    ----------
    query : str
        검색 키워드. Default "국민대".
    delay : tuple[float, float]
        요청 간 대기 시간(초). Default (1.0, 2.0).
    body_concurrency : int
        본문 추출 동시 요청 수. Default 10.
    body_delay : tuple[float, float]
        본문 요청 간 대기 시간(초). Default (0.3, 0.8).
    sort : str
        정렬 방식. "0"=관련도순, "1"=최신순. Default "1".
    max_pages_per_month : int | None
        월별 최대 페이지 수 (None=무제한). Default None.
    """

    def __init__(
        self,
        query: str = "국민대",
        delay: tuple[float, float] = (1.0, 2.0),
        body_concurrency: int = 10,
        body_delay: tuple[float, float] = (0.3, 0.8),
        skip_filter: bool = False,
        sort: str = "1",
        max_pages_per_month: int | None = None,
    ):
        self.query = query
        self.delay = delay
        self.body_concurrency = body_concurrency
        self.body_delay = body_delay
        self.skip_filter = skip_filter
        self.sort = sort
        self.max_pages_per_month = max_pages_per_month

    # -- date range helpers ----------------------------------------------------

    @staticmethod
    def _generate_month_ranges(
        start_year: int, start_month: int, end_year: int, end_month: int
    ) -> list[tuple[str, str]]:
        """월별 (ds, de) 문자열 쌍 생성."""
        ranges = []
        y, m = start_year, start_month
        while (y, m) <= (end_year, end_month):
            last_day = calendar.monthrange(y, m)[1]
            ds = f"{y}.{m:02d}.01"
            de = f"{y}.{m:02d}.{last_day:02d}"
            ranges.append((ds, de))
            m += 1
            if m > 12:
                m = 1
                y += 1
        return ranges

    # -- HTTP helpers ----------------------------------------------------------

    def _build_search_params(self, ds: str, de: str, start: int = 1) -> dict:
        """검색 파라미터 생성."""
        return {
            "ssc": "tab.news.all",
            "query": self.query,
            "sort": self.sort,  # "0"=관련도순, "1"=최신순
            "pd": "3",
            "ds": ds,
            "de": de,
            "start": str(start),
            "field": "0",
            "is_dts": "0",
            "is_sug_officeid": "0",
            "mynews": "0",
            "office_type": "0",
            "office_section_code": "0",
            "office_category": "0",
            "service_area": "0",
            "photo": "0",
        }

    async def _fetch_page(
        self, client: httpx.AsyncClient, ds: str, de: str, start: int
    ) -> str:
        """검색 결과 HTML을 가져온다. start=1이면 본체, 이후는 무한스크롤 API(JSON→HTML)."""
        params = self._build_search_params(ds, de, start)
        headers = {
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
            "Referer": "https://search.naver.com/",
        }

        url = SEARCH_URL if start == 1 else MORE_API_URL
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        await asyncio.sleep(random.uniform(*self.delay))

        if start == 1:
            return resp.text

        # 무한스크롤 API는 JSON 반환 → collection[0].html 추출
        data = resp.json()
        collections = data.get("collection", [])
        if collections and "html" in collections[0]:
            return collections[0]["html"]
        return ""

    # -- HTML parsing ----------------------------------------------------------

    def _parse_items(self, html: str) -> list[dict]:
        """검색 결과 HTML에서 기사 메타데이터 추출."""
        soup = BeautifulSoup(html, "html.parser")
        items: list[dict] = []

        # 기사 컨테이너: data-heatmap-target=".tit" 링크를 포함하는 블록
        for title_a in soup.select('a[data-heatmap-target=".tit"]'):
            # 제목
            title_span = title_a.select_one("span")
            if not title_span:
                continue
            title = _clean_text(title_span.get_text())
            if not title:
                continue

            # URL
            url = title_a.get("href", "")
            if not url:
                continue

            # 상위 컨테이너 찾기
            container = title_a.parent
            while container and container.parent:
                # Ci1iD8ML5xKVKTbxwSzZ 또는 비슷한 최상위 기사 블록
                if container.parent and container.parent.name == "div":
                    sibling_profile = container.parent.select_one(
                        '[data-sds-comp="Profile"]'
                    )
                    if sibling_profile:
                        container = container.parent
                        break
                container = container.parent

            # 언론사
            press = ""
            prof_links = container.select('a[data-heatmap-target=".prof"]') if container else []
            for pl in prof_links:
                text = pl.get_text(strip=True)
                if text and len(text) > 1:
                    press = text
                    break

            # 날짜
            date_text = ""
            subtexts = container.select("span.sds-comps-profile-info-subtext") if container else []
            for st in subtexts:
                t = st.get_text(strip=True)
                if t and ("전" in t or "." in t):
                    date_text = t
                    break

            # 요약문
            description = ""
            body_a = container.select_one('a[data-heatmap-target=".body"]') if container else None
            if body_a:
                desc_span = body_a.select_one("span")
                if desc_span:
                    description = _clean_text(desc_span.get_text())

            # 네이버뉴스 링크
            naver_url = ""
            nav_a = container.select_one('a[data-heatmap-target=".nav"]') if container else None
            if nav_a:
                naver_url = nav_a.get("href", "")

            items.append({
                "title": title,
                "press": press,
                "date": date_text,
                "description": description,
                "url": url,
                "naver_url": naver_url,
            })

        return items

    def _filter_items(self, items: list[dict]) -> list[dict]:
        """국민대학교 관련 기사만 필터. skip_filter=True면 필터링 건너뜀."""
        if self.skip_filter:
            return items
        filtered = []
        for item in items:
            title = item["title"]
            if "국민대" not in title:
                logger.debug("국민대 미포함 스킵: %s", title)
                continue
            if _SKIP_PATTERN.search(title):
                logger.debug("skip_pattern 스킵: %s", title)
                continue
            filtered.append(item)
        return filtered

    # -- CSV I/O ---------------------------------------------------------------

    @staticmethod
    def _save_to_csv(
        items: list[dict], path: Path, keyword: str | None = None
    ) -> None:
        """DataFrame을 CSV에 append. keyword가 주어지면 컬럼 추가."""
        if not items:
            return
        if keyword:
            for item in items:
                item["keyword"] = keyword
        df = pd.DataFrame(items)
        # keyword 컬럼이 있으면 맨 앞으로 이동
        if "keyword" in df.columns:
            cols = ["keyword"] + [c for c in df.columns if c != "keyword"]
            df = df[cols]
        write_header = not path.exists() or path.stat().st_size == 0
        df.to_csv(
            path,
            mode="a",
            index=False,
            header=write_header,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL,
        )

    # -- Phase 1: list crawl ---------------------------------------------------

    async def crawl_list(
        self,
        output: str = "kookmin_esg_raw",
        start_year: int = 2023,
        start_month: int = 1,
        end_year: int = 2026,
        end_month: int = 2,
        append: bool = False,
    ) -> int:
        """Phase 1: 월별 뉴스 리스트 수집 → CSV.

        Parameters
        ----------
        append : bool
            True면 기존 파일에 추가, False면 새로 생성.

        Returns
        -------
        int
            수집된 기사 수.
        """
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = DATA_RAW_DIR / f"{output}.csv"

        # append=False면 기존 파일 삭제
        if not append and csv_path.exists():
            csv_path.unlink()

        month_ranges = self._generate_month_ranges(
            start_year, start_month, end_year, end_month
        )
        total_collected = 0

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30), follow_redirects=True
        ) as client:
            for ds, de in tqdm(month_ranges, desc="월별 리스트 수집", unit="월"):
                month_items: list[dict] = []
                start = 1
                consecutive_empty = 0

                # 월별 최대 페이지 제한 (None이면 2000)
                max_start = (self.max_pages_per_month * 10) if self.max_pages_per_month else 2000

                while start <= max_start:
                    try:
                        html = await self._fetch_page(client, ds, de, start)
                    except Exception as exc:
                        logger.warning("페이지 요청 실패 (%s, start=%d): %s", ds, start, exc)
                        break

                    raw_items = self._parse_items(html)
                    if not raw_items:
                        consecutive_empty += 1
                        if consecutive_empty >= 2:
                            break
                        start += 10
                        continue

                    consecutive_empty = 0
                    filtered = self._filter_items(raw_items)
                    month_items.extend(filtered)
                    start += 10

                if month_items:
                    self._save_to_csv(month_items, csv_path, keyword=self.query)
                    total_collected += len(month_items)

                logger.info(
                    "%s ~ %s: %d건 수집 (누적 %d건)",
                    ds, de, len(month_items), total_collected,
                )

        logger.info("Phase 1 완료: 총 %d건 → %s", total_collected, csv_path)
        return total_collected

    async def _crawl_single_month(
        self,
        client: httpx.AsyncClient,
        ds: str,
        de: str,
    ) -> list[dict]:
        """단일 월의 기사 목록 크롤링. URL 포함하여 반환."""
        month_items: list[dict] = []
        start = 1
        consecutive_empty = 0

        # 월별 최대 페이지 제한 (None이면 2000)
        max_start = (self.max_pages_per_month * 10) if self.max_pages_per_month else 2000

        while start <= max_start:
            try:
                html = await self._fetch_page(client, ds, de, start)
            except Exception as exc:
                logger.warning("페이지 요청 실패 (%s, start=%d): %s", ds, start, exc)
                break

            raw_items = self._parse_items(html)
            if not raw_items:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
                start += 10
                continue

            consecutive_empty = 0
            filtered = self._filter_items(raw_items)
            month_items.extend(filtered)
            start += 10

        return month_items

    async def crawl_list_parallel(
        self,
        output: str = "kookmin_esg_raw",
        start_year: int = 2023,
        start_month: int = 1,
        end_year: int = 2026,
        end_month: int = 2,
        append: bool = False,
        month_concurrency: int = 6,
    ) -> int:
        """Phase 1: 월별 병렬 크롤링 → CSV.

        Parameters
        ----------
        month_concurrency : int
            동시에 크롤링할 월 수 (기본: 6).

        Returns
        -------
        int
            수집된 기사 수.
        """
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = DATA_RAW_DIR / f"{output}.csv"

        if not append and csv_path.exists():
            csv_path.unlink()

        month_ranges = self._generate_month_ranges(
            start_year, start_month, end_year, end_month
        )
        total_collected = 0

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30), follow_redirects=True
        ) as client:
            # 월을 배치로 나눠서 병렬 실행
            for i in range(0, len(month_ranges), month_concurrency):
                batch = month_ranges[i : i + month_concurrency]
                batch_desc = f"{batch[0][0]}~{batch[-1][1]}"

                # 병렬로 월별 크롤링 실행
                tasks = [
                    self._crawl_single_month(client, ds, de)
                    for ds, de in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 결과 수집 및 저장
                batch_items: list[dict] = []
                for (ds, de), result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.error("월 크롤링 실패 %s~%s: %s", ds, de, result)
                        continue
                    batch_items.extend(result)

                if batch_items:
                    self._save_to_csv(batch_items, csv_path, keyword=self.query)
                    total_collected += len(batch_items)

                logger.info(
                    "배치 %s: %d건 수집 (누적 %d건)",
                    batch_desc, len(batch_items), total_collected,
                )

        logger.info("Phase 1 (병렬) 완료: 총 %d건 → %s", total_collected, csv_path)
        return total_collected

    # -- Phase 2: body extraction ----------------------------------------------

    def _detect_encoding(self, raw_bytes: bytes, content_type: str | None) -> str:
        """HTML 인코딩 감지. 한국 언론사는 EUC-KR/CP949 많이 사용."""
        # 1. Content-Type 헤더에서 charset 확인
        if content_type:
            for part in content_type.split(";"):
                if "charset=" in part.lower():
                    return part.split("=")[-1].strip()

        # 2. HTML meta 태그에서 charset 확인
        head_bytes = raw_bytes[:2048]
        try:
            head_str = head_bytes.decode("ascii", errors="ignore")
            # <meta charset="...">
            import re
            match = re.search(r'charset=["\']?([^"\'\s>]+)', head_str, re.I)
            if match:
                return match.group(1)
        except Exception:
            pass

        # 3. 기본값: utf-8 시도, 실패시 cp949(한국어)
        return "utf-8"

    def _decode_html(self, raw_bytes: bytes, content_type: str | None) -> str:
        """바이트를 문자열로 디코딩. 여러 인코딩 시도."""
        detected = self._detect_encoding(raw_bytes, content_type)
        encodings = [detected, "utf-8", "cp949", "euc-kr", "latin-1"]

        for enc in encodings:
            try:
                return raw_bytes.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue

        # 최후의 수단: 에러 무시
        return raw_bytes.decode("utf-8", errors="ignore")

    async def _fetch_body(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        url: str,
    ) -> str:
        """기사 본문 추출."""
        async with semaphore:
            headers = {"User-Agent": random.choice(_USER_AGENTS)}
            try:
                async with session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=15),
                    ssl=False,  # SSL 검증 비활성화 (일부 사이트 인증서 문제)
                ) as resp:
                    resp.raise_for_status()
                    raw_bytes = await resp.read()
                    content_type = resp.headers.get("Content-Type", "")
                    html = self._decode_html(raw_bytes, content_type)
            except Exception as exc:
                logger.warning("본문 요청 실패 %s: %s", url, exc)
                return ""

            await asyncio.sleep(random.uniform(*self.body_delay))

            # 네이버 뉴스
            if "news.naver.com" in url:
                soup = BeautifulSoup(html, "html.parser")
                article = soup.select_one(
                    "#newsct_article, #articeBody, #articleBodyContents"
                )
                if article:
                    for tag in article.select("script, style, span.end_photo_org"):
                        tag.decompose()
                    return article.get_text(separator="\n", strip=True)

            # 외부 링크 → trafilatura
            result = trafilatura.extract(html)
            return result or ""

    async def extract_bodies(
        self,
        csv_path: str | Path | None = None,
        output: str = "kookmin_esg_raw",
    ) -> int:
        """Phase 2: CSV에서 URL 로드 → 본문 추출 → content 컬럼 추가.

        Returns
        -------
        int
            본문 추출 성공 건수.
        """
        if csv_path is None:
            csv_path = DATA_RAW_DIR / f"{output}.csv"
        csv_path = Path(csv_path)

        if not csv_path.exists():
            logger.error("CSV 파일 없음: %s", csv_path)
            return 0

        df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
        df.fillna("", inplace=True)
        if "content" not in df.columns:
            df["content"] = ""

        # 본문이 비어있는 행만 추출
        mask = df["content"].isna() | (df["content"] == "")
        todo_indices = df[mask].index.tolist()

        if not todo_indices:
            logger.info("모든 기사에 본문이 이미 있습니다.")
            return 0

        logger.info("본문 추출 대상: %d건", len(todo_indices))
        semaphore = asyncio.Semaphore(self.body_concurrency)
        success = 0

        async with aiohttp.ClientSession() as session:
            pbar = async_tqdm(total=len(todo_indices), desc="본문 추출", unit="건")

            async def process(idx: int) -> None:
                nonlocal success
                row = df.loc[idx]
                # 네이버 URL 우선, 없으면 원문 URL
                naver_url = row.get("naver_url", "")
                orig_url = row.get("url", "")
                url = naver_url if naver_url else orig_url
                if not url:
                    pbar.update(1)
                    return
                body = await self._fetch_body(session, semaphore, url)

                # 네이버 URL 실패 시 원문으로 재시도
                if not body and naver_url and orig_url and orig_url != url:
                    body = await self._fetch_body(session, semaphore, orig_url)

                if body:
                    df.at[idx, "content"] = body
                    success += 1
                pbar.update(1)

            tasks = [asyncio.create_task(process(idx)) for idx in todo_indices]
            await asyncio.gather(*tasks, return_exceptions=True)
            pbar.close()

        # CSV 덮어쓰기
        df.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL,
        )
        logger.info("Phase 2 완료: %d/%d건 본문 추출 → %s", success, len(todo_indices), csv_path)
        return success
