# -*- coding: utf-8 -*-
"""Legacy QA review queue primitive — superseded by ``review_workflow``.

Historically this module wrote raw rows into a ``qa_reviews`` SQLite
table.  The full review lifecycle (assignment, checklist, SLA timer,
consensus voting, promotion) now lives in
:mod:`greenlang.factors.quality.review_workflow`, which owns its own
schema and is the canonical path for every new caller.

``enqueue_review`` is preserved as a thin compatibility shim so that
older migrations and one-off scripts still import cleanly.  New code
should call ``review_workflow.create_review`` +
``save_review_to_sqlite`` instead.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
import warnings
from typing import Any, Dict


def enqueue_review(
    conn: sqlite3.Connection,
    *,
    edition_id: str,
    factor_id: str,
    status: str,
    payload: Dict[str, Any],
) -> str:
    """Deprecated — use :func:`review_workflow.create_review`.

    Kept to avoid breaking historical migrations that still import
    this function.  Emits a ``DeprecationWarning`` on call.
    """
    warnings.warn(
        "greenlang.factors.quality.review_queue.enqueue_review is deprecated; "
        "use greenlang.factors.quality.review_workflow.create_review + "
        "save_review_to_sqlite instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    rid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO qa_reviews (review_id, edition_id, factor_id, status, payload_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (rid, edition_id, factor_id, status, json.dumps(payload, sort_keys=True, default=str)),
    )
    return rid


__all__ = ["enqueue_review"]
