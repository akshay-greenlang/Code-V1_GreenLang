# -*- coding: utf-8 -*-
"""Audit export bundle shape (Q5) — JSON bundle; zip left to packaging."""

from __future__ import annotations

from typing import Any, Dict, Optional

from greenlang.data.emission_factor_record import EmissionFactorRecord


def build_audit_bundle_dict(
    record: EmissionFactorRecord,
    *,
    raw_pointer: Optional[str] = None,
    parser_log: Optional[str] = None,
    qa_errors: Optional[list] = None,
    reviewer_decision: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "factor_id": record.factor_id,
        "content_hash": record.content_hash,
        "normalized": record.to_dict(),
        "raw_artifact_uri": raw_pointer,
        "parser_log": parser_log,
        "qa_errors": qa_errors or [],
        "reviewer_decision": reviewer_decision,
    }
