# -*- coding: utf-8 -*-
"""QA gates for factor payloads prior to catalog insert."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def validate_factor_dict(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Return (ok, errors) for a dict shaped like EmissionFactorRecord.to_dict() output.
    """
    errors: List[str] = []
    fid = data.get("factor_id")
    if not fid or not str(fid).startswith("EF:"):
        errors.append("factor_id must exist and start with EF:")
    vectors = data.get("vectors") or {}
    for gas in ("CO2", "CH4", "N2O"):
        if gas not in vectors:
            errors.append(f"vectors.{gas} is required")
            continue
        try:
            v = float(vectors[gas])
            if v < 0:
                errors.append(f"vectors.{gas} must be non-negative")
        except (TypeError, ValueError):
            errors.append(f"vectors.{gas} must be numeric")
    gwp = data.get("gwp_100yr") or {}
    if "co2e_total" in gwp:
        try:
            if float(gwp["co2e_total"]) < 0:
                errors.append("gwp_100yr.co2e_total must be non-negative")
        except (TypeError, ValueError):
            errors.append("gwp_100yr.co2e_total must be numeric")
    if errors:
        logger.debug("QA validation failed for %s: %s", data.get("factor_id", "?"), "; ".join(errors))
    return (len(errors) == 0, errors)
