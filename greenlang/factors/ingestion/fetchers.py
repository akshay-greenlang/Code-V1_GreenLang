# -*- coding: utf-8 -*-
"""HTTP / local file fetchers (D2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen


class BaseFetcher(ABC):
    @abstractmethod
    def fetch(self, url: str) -> bytes:
        raise NotImplementedError


class HttpFetcher(BaseFetcher):
    def __init__(self, timeout_s: float = 30.0, user_agent: str = "GreenLang-Factors/1.0"):
        self.timeout_s = timeout_s
        self.user_agent = user_agent

    def fetch(self, url: str) -> bytes:
        req = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(req, timeout=self.timeout_s) as resp:  # nosec B310 — controlled registry URLs
            return resp.read()


class FileFetcher(BaseFetcher):
    def fetch(self, url: str) -> bytes:
        p = Path(url)
        if not p.is_file():
            raise FileNotFoundError(url)
        return p.read_bytes()


def head_exists(url: str, timeout_s: float = 10.0) -> bool:
    """Best-effort GET reachability check for source watch (U1); False on failure."""
    try:
        req = Request(url, headers={"User-Agent": "GreenLang-Factors-Watch/1.0"})
        with urlopen(req, timeout=timeout_s) as resp:  # nosec B310
            code = getattr(resp, "status", resp.getcode())
            return 200 <= int(code) < 400
    except Exception:
        return False
