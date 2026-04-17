# -*- coding: utf-8 -*-
"""Parser harness (D3): uniform parse status + row counts for ingest_runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ParserContext:
    artifact_id: str
    source_id: str
    parser_id: str


@dataclass
class ParserResult:
    status: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    error: str = ""


def run_parser(
    ctx: ParserContext,
    raw: bytes,
    fn: Callable[[ParserContext, bytes], ParserResult],
) -> ParserResult:
    logger.debug("Running parser=%s source=%s artifact=%s", ctx.parser_id, ctx.source_id, ctx.artifact_id)
    try:
        result = fn(ctx, raw)
        logger.info("Parser %s completed status=%s rows=%d", ctx.parser_id, result.status, len(result.rows))
        return result
    except Exception as exc:  # pragma: no cover — defensive
        logger.error("Parser %s failed: %s", ctx.parser_id, exc)
        return ParserResult(status="failed", error=str(exc))
