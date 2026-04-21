# -*- coding: utf-8 -*-
"""Cursor + offset paginators for list/search responses.

Usage (sync)::

    for factor in client.paginate_search("diesel", page_size=50):
        print(factor.factor_id)

Usage (async)::

    async for factor in aclient.paginate_search("diesel", page_size=50):
        print(factor.factor_id)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PageInfo:
    """Summary of pagination progress."""

    page: int
    page_size: int
    total_count: Optional[int]
    next_cursor: Optional[str] = None
    items_yielded: int = 0


class OffsetPaginator(Iterator[T]):
    """Generic offset/limit iterator.

    The fetcher callable receives ``(offset, limit)`` and must return
    ``(items, total_count_or_none)``.  Iteration stops when:

        * ``items`` is empty, OR
        * ``offset + len(items) >= total_count`` (if total known), OR
        * user-supplied ``max_items`` was reached.
    """

    def __init__(
        self,
        fetcher: Callable[[int, int], "tuple[List[T], Optional[int]]"],
        *,
        page_size: int = 100,
        start_offset: int = 0,
        max_items: Optional[int] = None,
    ) -> None:
        self._fetcher = fetcher
        self._page_size = max(1, int(page_size))
        self._offset = int(start_offset)
        self._max_items = max_items
        self._buffer: List[T] = []
        self._total: Optional[int] = None
        self._exhausted = False
        self._yielded = 0

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self._max_items is not None and self._yielded >= self._max_items:
            raise StopIteration
        if not self._buffer and not self._exhausted:
            self._fill()
        if not self._buffer:
            raise StopIteration
        item = self._buffer.pop(0)
        self._yielded += 1
        return item

    def _fill(self) -> None:
        limit = self._page_size
        if self._max_items is not None:
            limit = min(limit, self._max_items - self._yielded)
            if limit <= 0:
                self._exhausted = True
                return
        items, total = self._fetcher(self._offset, limit)
        self._buffer.extend(items)
        self._offset += len(items)
        if total is not None:
            self._total = total
        if not items or (total is not None and self._offset >= total):
            self._exhausted = True

    @property
    def total_count(self) -> Optional[int]:
        return self._total


class CursorPaginator(Iterator[T]):
    """Generic cursor-based iterator.

    The fetcher callable receives ``cursor`` (None on first call) and
    must return ``(items, next_cursor)``.  Iteration stops when
    ``next_cursor`` is ``None`` or user-supplied ``max_items`` reached.
    """

    def __init__(
        self,
        fetcher: Callable[[Optional[str]], "tuple[List[T], Optional[str]]"],
        *,
        max_items: Optional[int] = None,
    ) -> None:
        self._fetcher = fetcher
        self._max_items = max_items
        self._buffer: List[T] = []
        self._cursor: Optional[str] = None
        self._started = False
        self._exhausted = False
        self._yielded = 0

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self._max_items is not None and self._yielded >= self._max_items:
            raise StopIteration
        if not self._buffer and not self._exhausted:
            self._fill()
        if not self._buffer:
            raise StopIteration
        item = self._buffer.pop(0)
        self._yielded += 1
        return item

    def _fill(self) -> None:
        if self._started and self._cursor is None:
            self._exhausted = True
            return
        items, next_cursor = self._fetcher(self._cursor)
        self._started = True
        self._buffer.extend(items)
        self._cursor = next_cursor
        if not items or next_cursor is None:
            self._exhausted = True

    @property
    def cursor(self) -> Optional[str]:
        return self._cursor


class AsyncOffsetPaginator:
    """Async sibling of :class:`OffsetPaginator`."""

    def __init__(
        self,
        fetcher: Callable[[int, int], Awaitable["tuple[List[T], Optional[int]]"]],
        *,
        page_size: int = 100,
        start_offset: int = 0,
        max_items: Optional[int] = None,
    ) -> None:
        self._fetcher = fetcher
        self._page_size = max(1, int(page_size))
        self._offset = int(start_offset)
        self._max_items = max_items
        self._buffer: List[T] = []
        self._total: Optional[int] = None
        self._exhausted = False
        self._yielded = 0

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        if self._max_items is not None and self._yielded >= self._max_items:
            raise StopAsyncIteration
        if not self._buffer and not self._exhausted:
            await self._fill()
        if not self._buffer:
            raise StopAsyncIteration
        item = self._buffer.pop(0)
        self._yielded += 1
        return item

    async def _fill(self) -> None:
        limit = self._page_size
        if self._max_items is not None:
            limit = min(limit, self._max_items - self._yielded)
            if limit <= 0:
                self._exhausted = True
                return
        items, total = await self._fetcher(self._offset, limit)
        self._buffer.extend(items)
        self._offset += len(items)
        if total is not None:
            self._total = total
        if not items or (total is not None and self._offset >= total):
            self._exhausted = True


class AsyncCursorPaginator:
    """Async sibling of :class:`CursorPaginator`."""

    def __init__(
        self,
        fetcher: Callable[[Optional[str]], Awaitable["tuple[List[T], Optional[str]]"]],
        *,
        max_items: Optional[int] = None,
    ) -> None:
        self._fetcher = fetcher
        self._max_items = max_items
        self._buffer: List[T] = []
        self._cursor: Optional[str] = None
        self._started = False
        self._exhausted = False
        self._yielded = 0

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        if self._max_items is not None and self._yielded >= self._max_items:
            raise StopAsyncIteration
        if not self._buffer and not self._exhausted:
            await self._fill()
        if not self._buffer:
            raise StopAsyncIteration
        item = self._buffer.pop(0)
        self._yielded += 1
        return item

    async def _fill(self) -> None:
        if self._started and self._cursor is None:
            self._exhausted = True
            return
        items, next_cursor = await self._fetcher(self._cursor)
        self._started = True
        self._buffer.extend(items)
        self._cursor = next_cursor
        if not items or next_cursor is None:
            self._exhausted = True


def extract_items(payload: Any, key: str = "factors") -> List[Dict[str, Any]]:
    """Pull ``key`` out of a response payload regardless of wrapper shape."""
    if isinstance(payload, dict):
        inner = payload.get(key)
        if isinstance(inner, list):
            return inner
    if isinstance(payload, list):
        return payload
    return []


__all__ = [
    "PageInfo",
    "OffsetPaginator",
    "CursorPaginator",
    "AsyncOffsetPaginator",
    "AsyncCursorPaginator",
    "extract_items",
]
