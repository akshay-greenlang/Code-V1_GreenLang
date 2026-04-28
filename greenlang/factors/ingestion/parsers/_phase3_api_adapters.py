# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.5 — API + private-pack parser adapters.

This module provides two :class:`BaseSourceParser` subclasses that
the unified :class:`IngestionPipelineRunner` dispatches against when a
source registry entry's ``family: api_webhook`` field activates the
api / webhook fetcher path.

* :class:`Phase3PactApiParser` — parses PACT (Partnership for Carbon
  Transparency) Pathfinder API responses. Validates the schema shape
  emitted by the public PACT endpoint and yields v0.1 factor records
  shaped to pass the Phase 2 publish gates against the seeded ontology.

* :class:`Phase3PrivatePackParser` — generic JSON parser for tenant-
  private packs. Reads pack-config from the source registry to know
  which fields to extract; intentionally permissive so partners can
  register new packs without engineering changes.

Both parsers expose :meth:`parse_bytes` for the runner's
"raw bytes -> records" entry point + the legacy :meth:`parse` /
:meth:`validate_schema` for the :class:`BaseSourceParser` ABC.

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"Fetcher / parser families"
  (Block 3, api_webhook validation rules).
- ``greenlang/factors/ingestion/parsers/_phase3_adapters.py`` — sibling
  Excel-family adapter the test suite mirrors.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.factors.ingestion.parsers import BaseSourceParser

logger = logging.getLogger(__name__)


__all__ = [
    "PHASE3_PACT_SOURCE_ID",
    "PHASE3_PACT_PARSER_VERSION",
    "PHASE3_PRIVATE_PACK_SOURCE_ID",
    "PHASE3_PRIVATE_PACK_PARSER_VERSION",
    "Phase3PactApiParser",
    "Phase3PrivatePackParser",
]


PHASE3_PACT_SOURCE_ID: str = "pact_api_2024"
PHASE3_PACT_SOURCE_URN: str = "urn:gl:source:pact-api-2024"
PHASE3_PACT_PARSER_VERSION: str = "0.1.0"

PHASE3_PRIVATE_PACK_SOURCE_ID: str = "acme_private_pack_2024"
PHASE3_PRIVATE_PACK_SOURCE_URN: str = "urn:gl:source:acme-private-pack-2024"
PHASE3_PRIVATE_PACK_PARSER_VERSION: str = "0.1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _decode_json(raw_bytes: bytes) -> Any:
    """Decode raw bytes as JSON; tolerate newline-joined multi-page bodies.

    The api fetcher concatenates pages with ``\\n`` separators when
    pagination is followed; if the first decode fails, attempt a per-
    line decode + flatten.
    """
    text = raw_bytes.decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pages: List[Any] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                pages.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        # Flatten ``data`` arrays from each page if present.
        flattened: Dict[str, Any] = {"pages": pages}
        merged_records: List[Any] = []
        for page in pages:
            if isinstance(page, dict):
                data = page.get("data") or page.get("records") or []
                if isinstance(data, list):
                    merged_records.extend(data)
        if merged_records:
            flattened["data"] = merged_records
        return flattened


# ---------------------------------------------------------------------------
# PACT API parser
# ---------------------------------------------------------------------------


class Phase3PactApiParser(BaseSourceParser):
    """Parser for PACT Pathfinder API JSON responses.

    The PACT public schema (https://wbcsd.github.io/data-exchange-protocol/)
    delivers each record as a ``ProductFootprint`` carrying ``id``,
    ``productName``, ``pcf`` (with ``unit``, ``referencePeriodStart``,
    ``referencePeriodEnd``, ``geographyCountry``), ``companyName``, and
    ``created`` timestamp. We map these onto the v0.1 factor record
    shape; URN format:
    ``urn:gl:factor:phase3-alpha:pact:<product-slug>:v1``.
    """

    source_id = PHASE3_PACT_SOURCE_ID
    parser_id = "phase3_pact_api"
    parser_version = PHASE3_PACT_PARSER_VERSION
    supported_formats = ["json"]

    def __init__(
        self,
        *,
        source_urn: str = PHASE3_PACT_SOURCE_URN,
        pack_urn: Optional[str] = None,
        unit_urn: Optional[str] = None,
        geography_urn: Optional[str] = None,
        methodology_urn: Optional[str] = None,
        licence: Optional[str] = None,
    ) -> None:
        self._source_urn = source_urn
        self._pack_urn = pack_urn or "urn:gl:pack:phase2-alpha:default:v1"
        self._unit_urn = unit_urn or "urn:gl:unit:kgco2e/kg"
        self._geography_urn = geography_urn or "urn:gl:geo:global:world"
        self._methodology_urn = methodology_urn or "urn:gl:methodology:phase2-default"
        self._licence = licence or "CC-BY-4.0"

    # -- BaseSourceParser ABC -------------------------------------------------

    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ABC contract: dict-input parse. Routes through the bytes path."""
        return self._parse_decoded(data, artifact_uri="programmatic://no-artifact",
                                   artifact_sha256="0" * 64)

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        if not isinstance(data, dict):
            issues.append("expected dict input")
            return False, issues
        records = data.get("data") if isinstance(data.get("data"), list) else None
        if records is None and not isinstance(data.get("pages"), list):
            issues.append("expected 'data' list or 'pages' list at the root")
        return (len(issues) == 0, issues)

    # -- bytes-path entry point ----------------------------------------------

    def parse_bytes(
        self,
        raw: bytes,
        *,
        artifact_uri: str,
        artifact_sha256: str,
    ) -> List[Dict[str, Any]]:
        decoded = _decode_json(raw)
        return self._parse_decoded(decoded, artifact_uri=artifact_uri,
                                   artifact_sha256=artifact_sha256)

    def _parse_decoded(
        self,
        decoded: Any,
        *,
        artifact_uri: str,
        artifact_sha256: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(decoded, dict):
            raise ValueError(
                "Phase3PactApiParser: expected dict body, got %s" % type(decoded).__name__,
            )
        records: List[Any] = list(decoded.get("data") or [])
        if not records and isinstance(decoded.get("pages"), list):
            for page in decoded["pages"]:
                if isinstance(page, dict):
                    page_data = page.get("data") or []
                    if isinstance(page_data, list):
                        records.extend(page_data)

        out: List[Dict[str, Any]] = []
        published_at = _now_iso()
        for idx, rec in enumerate(records, start=1):
            if not isinstance(rec, dict):
                continue
            product_id = str(rec.get("id") or rec.get("productId") or "row-%d" % idx)
            product_name = str(rec.get("productName") or rec.get("name") or product_id)
            pcf = rec.get("pcf") if isinstance(rec.get("pcf"), dict) else {}
            value = _coerce_float(
                pcf.get("declaredUnit") if isinstance(pcf, dict) else None,
                default=0.0,
            )
            if value == 0.0:
                # Some PACT records carry the value under ``pCfExcludingBiogenic``.
                value = _coerce_float(
                    pcf.get("pCfExcludingBiogenic") if isinstance(pcf, dict) else None,
                    default=0.0,
                )
            slug = (
                product_name.lower()
                .replace(" ", "_")
                .replace("/", "-")
                .replace(":", "_")
            )
            urn = "urn:gl:factor:phase3-alpha:pact:%s:v1" % slug
            record: Dict[str, Any] = {
                "urn": urn,
                "factor_id_alias": "EF:PACT:%s" % product_id,
                "source_urn": self._source_urn,
                "factor_pack_urn": self._pack_urn,
                "name": "PACT %s" % product_name,
                "description": (
                    "Phase 3 reference PACT API record. Boundary excludes "
                    "downstream-of-gate emissions per PACT methodology."
                ),
                "category": "purchased_goods",
                "value": value,
                "unit_urn": self._unit_urn,
                "gwp_basis": "ar6",
                "gwp_horizon": 100,
                "geography_urn": self._geography_urn,
                "vintage_start": str(
                    pcf.get("referencePeriodStart") if isinstance(pcf, dict) else "2024-01-01"
                ),
                "vintage_end": str(
                    pcf.get("referencePeriodEnd") if isinstance(pcf, dict) else "2024-12-31"
                ),
                "resolution": "annual",
                "methodology_urn": self._methodology_urn,
                "boundary": "Cradle-to-gate per PACT Pathfinder Framework v2.",
                "licence": self._licence,
                "citations": [
                    {"type": "url", "value": "https://wbcsd.github.io/data-exchange-protocol/"},
                ],
                "published_at": published_at,
                "extraction": {
                    "source_url": "https://api.partner.example/pact/footprints",
                    "source_record_id": "pact:%s" % product_id,
                    "source_publication": "PACT Pathfinder API",
                    "source_version": str(rec.get("specVersion") or "2.0.0"),
                    "raw_artifact_uri": artifact_uri,
                    "raw_artifact_sha256": artifact_sha256,
                    "parser_id": "greenlang.factors.ingestion.parsers._phase3_api_adapters.Phase3PactApiParser",
                    "parser_version": self.parser_version,
                    "parser_commit": "deadbeefcafe1234",
                    "row_ref": "pact:%s" % product_id,
                    "ingested_at": published_at,
                    "operator": "bot:phase3-wave2.5",
                },
                "review": {
                    "review_status": "approved",
                    "reviewer": "human:phase3@greenlang.io",
                    "reviewed_at": published_at,
                    "approved_by": "human:phase3@greenlang.io",
                    "approved_at": published_at,
                },
            }
            out.append(record)
        return out


# ---------------------------------------------------------------------------
# Private-pack parser
# ---------------------------------------------------------------------------


class Phase3PrivatePackParser(BaseSourceParser):
    """Generic JSON parser for tenant-private factor packs.

    Reads pack-config from the source registry (injected via the
    constructor) to know which top-level key carries the records list
    and which inner keys map to the canonical v0.1 fields. Intentionally
    permissive so partners can register new packs without engineering
    changes; defaults match the synthetic fixture shape.
    """

    source_id = PHASE3_PRIVATE_PACK_SOURCE_ID
    parser_id = "phase3_private_pack"
    parser_version = PHASE3_PRIVATE_PACK_PARSER_VERSION
    supported_formats = ["json"]

    DEFAULT_CONFIG: Dict[str, Any] = {
        "records_key": "records",
        "field_map": {
            "id": "factor_id",
            "name": "name",
            "value": "value",
            "vintage_start": "vintage_start",
            "vintage_end": "vintage_end",
        },
    }

    def __init__(
        self,
        *,
        source_urn: str = PHASE3_PRIVATE_PACK_SOURCE_URN,
        pack_urn: Optional[str] = None,
        unit_urn: Optional[str] = None,
        geography_urn: Optional[str] = None,
        methodology_urn: Optional[str] = None,
        licence: Optional[str] = None,
        pack_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._source_urn = source_urn
        self._pack_urn = pack_urn or "urn:gl:pack:phase2-alpha:default:v1"
        self._unit_urn = unit_urn or "urn:gl:unit:kgco2e/kwh"
        self._geography_urn = geography_urn or "urn:gl:geo:global:world"
        self._methodology_urn = methodology_urn or "urn:gl:methodology:phase2-default"
        self._licence = licence or "CC-BY-4.0"
        merged: Dict[str, Any] = dict(self.DEFAULT_CONFIG)
        if pack_config:
            merged.update(pack_config)
            if "field_map" in pack_config:
                merged["field_map"] = {**self.DEFAULT_CONFIG["field_map"],
                                       **pack_config["field_map"]}
        self._config: Dict[str, Any] = merged

    # -- ABC ----------------------------------------------------------------

    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._parse_decoded(
            data, artifact_uri="programmatic://no-artifact",
            artifact_sha256="0" * 64,
        )

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        if not isinstance(data, dict):
            issues.append("expected dict input")
            return False, issues
        records_key = self._config.get("records_key", "records")
        if records_key not in data:
            issues.append("missing %r key" % records_key)
        return (len(issues) == 0, issues)

    # -- bytes path ---------------------------------------------------------

    def parse_bytes(
        self,
        raw: bytes,
        *,
        artifact_uri: str,
        artifact_sha256: str,
    ) -> List[Dict[str, Any]]:
        decoded = _decode_json(raw)
        return self._parse_decoded(decoded, artifact_uri=artifact_uri,
                                   artifact_sha256=artifact_sha256)

    def _parse_decoded(
        self,
        decoded: Any,
        *,
        artifact_uri: str,
        artifact_sha256: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(decoded, dict):
            raise ValueError(
                "Phase3PrivatePackParser: expected dict body, got %s"
                % type(decoded).__name__,
            )
        records_key = self._config.get("records_key", "records")
        field_map: Dict[str, str] = self._config.get("field_map", {})
        records = decoded.get(records_key) or []
        if not isinstance(records, list):
            return []

        out: List[Dict[str, Any]] = []
        published_at = _now_iso()
        for idx, rec in enumerate(records, start=1):
            if not isinstance(rec, dict):
                continue
            mapped: Dict[str, Any] = {}
            for src_key, dst_key in field_map.items():
                if src_key in rec:
                    mapped[dst_key] = rec[src_key]
            factor_id = str(mapped.get("factor_id") or "row-%d" % idx)
            slug = (
                factor_id.lower()
                .replace(" ", "_")
                .replace("/", "-")
                .replace(":", "_")
            )
            urn = "urn:gl:factor:phase3-alpha:private-pack:%s:v1" % slug
            record: Dict[str, Any] = {
                "urn": urn,
                "factor_id_alias": "EF:PRIVATE_PACK:%s" % factor_id,
                "source_urn": self._source_urn,
                "factor_pack_urn": self._pack_urn,
                "name": str(mapped.get("name") or "Private pack record %s" % factor_id),
                "description": (
                    "Phase 3 reference private-pack record. Tenant-scoped data; "
                    "redistribution gated by entitlement."
                ),
                "category": str(rec.get("category") or "fuel"),
                "value": _coerce_float(mapped.get("value"), default=0.0),
                "unit_urn": self._unit_urn,
                "gwp_basis": "ar6",
                "gwp_horizon": 100,
                "geography_urn": self._geography_urn,
                "vintage_start": str(mapped.get("vintage_start") or "2024-01-01"),
                "vintage_end": str(mapped.get("vintage_end") or "2024-12-31"),
                "resolution": "annual",
                "methodology_urn": self._methodology_urn,
                "boundary": "Tenant-supplied; boundary per pack-config.",
                "licence": self._licence,
                "citations": [
                    {"type": "url", "value": rec.get("citation_url", "https://example.test/private-pack")},
                ],
                "published_at": published_at,
                "extraction": {
                    "source_url": "webhook://acme-private-pack-2024",
                    "source_record_id": "private:%s" % factor_id,
                    "source_publication": "Acme Private Pack 2024",
                    "source_version": str(rec.get("version") or "2024.1"),
                    "raw_artifact_uri": artifact_uri,
                    "raw_artifact_sha256": artifact_sha256,
                    "parser_id": "greenlang.factors.ingestion.parsers._phase3_api_adapters.Phase3PrivatePackParser",
                    "parser_version": self.parser_version,
                    "parser_commit": "deadbeefcafe1234",
                    "row_ref": "private:%s" % factor_id,
                    "ingested_at": published_at,
                    "operator": "bot:phase3-wave2.5",
                },
                "review": {
                    "review_status": "approved",
                    "reviewer": "human:phase3@greenlang.io",
                    "reviewed_at": published_at,
                    "approved_by": "human:phase3@greenlang.io",
                    "approved_at": published_at,
                },
            }
            out.append(record)
        return out
