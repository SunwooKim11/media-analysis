"""Async base crawler using httpx."""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod

import httpx
from bs4 import BeautifulSoup

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]


class AsyncCrawler(ABC):
    """Abstract async base crawler with httpx.

    Parameters
    ----------
    concurrency : int
        Max concurrent requests. Default 5.
    delay : tuple[float, float]
        Min/max seconds to sleep between requests. Default (0.5, 1.5).
    timeout : int
        Request timeout in seconds. Default 30.
    """

    def __init__(
        self,
        concurrency: int = 5,
        delay: tuple[float, float] = (0.5, 1.5),
        timeout: int = 30,
    ):
        self.concurrency = concurrency
        self.delay = delay
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            headers={"User-Agent": random.choice(_USER_AGENTS)},
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()

    def _random_ua(self) -> str:
        return random.choice(_USER_AGENTS)

    async def _random_delay(self) -> None:
        await asyncio.sleep(random.uniform(*self.delay))

    async def fetch(self, url: str) -> httpx.Response:
        """Send an async GET request, bounded by semaphore."""
        async with self._semaphore:
            headers = {"User-Agent": self._random_ua()}
            response = await self._client.get(url, headers=headers)
            response.raise_for_status()
            await self._random_delay()
            return response

    async def fetch_soup(self, url: str) -> BeautifulSoup:
        """Fetch page and return BeautifulSoup object."""
        resp = await self.fetch(url)
        return BeautifulSoup(resp.text, "html.parser")

    @abstractmethod
    async def crawl(self, **kwargs) -> int:
        """Run the async crawl. Returns number of articles collected."""
        ...
