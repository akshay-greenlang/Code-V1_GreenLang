# -*- coding: utf-8 -*-
"""Daily source watch dry-run (U1) + thin entry point to the full pipeline.

The dry-run probe remains for backward compatibility with callers (CLI
``validate-registry``, CI health checks).  For real scheduled polling use
``greenlang.factors.watch.scheduler.run_watch`` or the integrated pipeline
in ``greenlang.factors.watch.pipeline.run_regulatory_watch_cycle``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from greenlang.factors.ingestion.fetchers import head_exists
from greenlang.factors.source_registry import load_source_registry

logger = logging.getLogger(__name__)


def dry_run_registry_urls(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return reachability for every registry entry without persistence.

    Kept as a stateless probe used by CI and the ``validate-registry`` CLI.
    Changes are *not* detected here — use ``scheduler.run_watch`` or the
    integrated pipeline for that.
    """
    rows = load_source_registry(path)
    out: List[Dict[str, Any]] = []
    for e in rows:
        url = e.watch_url
        ok: Optional[bool] = None
        if url and e.watch_mechanism in ("http_head", "http_get", "html_diff"):
            ok = head_exists(str(url))
        out.append(
            {
                "source_id": e.source_id,
                "watch_mechanism": e.watch_mechanism,
                "url": url,
                "reachable": ok,
            }
        )
    logger.info("Watch dry-run completed: %d sources checked", len(out))
    return out
