# -*- coding: utf-8 -*-
"""Extended validators (Q1–Q2): schema + sanity."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Tuple

from greenlang.factors.etl.qa import validate_factor_dict
from greenlang.factors.ontology.geography import is_iso3166_alpha2
from greenlang.factors.ontology.units import is_known_activity_unit

logger = logging.getLogger(__name__)


def validate_canonical_row(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    ok, errs = validate_factor_dict(data)
    errors = list(errs)
    co2e = 0.0
    try:
        gwp = data.get("gwp_100yr") or {}
        co2e = float(gwp.get("co2e_total") or 0)
    except (TypeError, ValueError):
        errors.append("gwp_100yr.co2e_total not numeric")
    if co2e > 1e7:
        errors.append("outlier: co2e_total extremely high")
    vf = data.get("valid_from")
    if vf:
        try:
            y = date.fromisoformat(str(vf)[:10]).year
            if y < 1990 or y > 2040:
                errors.append("valid_from year outside plausible window")
        except ValueError:
            errors.append("valid_from not parseable")
    geo = str(data.get("geography") or "").strip().upper()
    if len(geo) == 2 and geo not in {"EU"} and not is_iso3166_alpha2(geo):
        errors.append("geography looks like ISO2 but invalid code")
    unit = str(data.get("unit") or "")
    if unit and not is_known_activity_unit(unit):
        errors.append(f"unit {unit!r} not in known denominator ontology subset")
    st = str(data.get("factor_status") or "certified")
    if st not in ("certified", "preview", "connector_only", "deprecated"):
        errors.append("factor_status invalid")
    if errors:
        logger.debug("Canonical row validation failed for %s: %d issues", data.get("factor_id", "?"), len(errors))
    return (len(errors) == 0, errors)
