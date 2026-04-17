# -*- coding: utf-8 -*-
"""Canonical normalizer (D3–D4): dict row → EmissionFactorRecord + lineage sidecar."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from greenlang.data.emission_factor_record import EmissionFactorRecord

logger = logging.getLogger(__name__)


class CanonicalNormalizer:
    """Wraps EmissionFactorRecord.from_dict with lineage preservation."""

    def normalize(self, row: Dict[str, Any], raw_fragment: Dict[str, Any]) -> Tuple[EmissionFactorRecord, Dict[str, Any]]:
        rec = EmissionFactorRecord.from_dict(dict(row))
        lineage = {"raw": raw_fragment, "normalized_factor_id": rec.factor_id}
        logger.debug("Normalized factor_id=%s", rec.factor_id)
        return rec, lineage
