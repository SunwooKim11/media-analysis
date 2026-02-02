"""Base crawler with random User-Agent rotation and rate limiting."""

import random
import time
from abc import ABC, abstractmethod

import requests
from bs4 import BeautifulSoup

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]


class Crawler(ABC):
    """Abstract base crawler.

    Parameters
    ----------
    delay : tuple[float, float]
        Min/max seconds to sleep between requests. Default (1.0, 3.0).
    timeout : int
        Request timeout in seconds. Default 30.
    """

    def __init__(self, delay: tuple[float, float] = (1.0, 3.0), timeout: int = 30):
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()

    def _random_ua(self) -> str:
        return random.choice(_USER_AGENTS)

    def _sleep(self) -> None:
        time.sleep(random.uniform(*self.delay))

    def fetch(self, url: str) -> requests.Response:
        """Send a GET request with a random User-Agent, then sleep."""
        headers = {"User-Agent": self._random_ua()}
        response = self.session.get(url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        self._sleep()
        return response

    def fetch_soup(self, url: str) -> BeautifulSoup:
        """Fetch a page and return a BeautifulSoup object."""
        resp = self.fetch(url)
        return BeautifulSoup(resp.text, "html.parser")

    @abstractmethod
    def crawl(self, query: str, max_results: int = 100) -> list[dict]:
        """Run the crawl and return a list of article dicts."""
