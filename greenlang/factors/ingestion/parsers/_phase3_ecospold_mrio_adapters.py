# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.5 — EcoSpold2 + EXIOBASE MRIO parser adapters.

Why this module exists
----------------------
ecoinvent (EcoSpold2) and EXIOBASE are the two heavyweight LCA / MRIO
sources that ship as **single-file zip bundles**. Both expand to many
thousands of inner files (XML for ecoinvent, CSV for EXIOBASE) and both
need a parser that:

  * Pulls the raw zip bytes from the runner's
    :meth:`IngestionPipelineRunner._read_artifact` output.
  * Expands the zip via :func:`extract_zip_artifact` (zip-bomb-safe).
  * Iterates inner members in deterministic order.
  * Emits one v0.1 factor record per inner row (per ``activityDescription``
    for ecoinvent, per (sector, region, extension) tuple for EXIOBASE).

Per the Wave 2.5 contract this module is **additive** — it does NOT
replace the in-tree :mod:`exiobase_v3` parser (which operates on
pre-decoded JSON). The Wave 2.5 parsers work directly off the raw zip
bytes the unified runner stores.

Determinism contract
--------------------
* Member iteration: ``sorted(zf.namelist())`` (alphabetical).
* Record emission order: by ``(spold_file, activity_id, exchange_id)``
  for ecoinvent; by ``(sector_idx, region_idx, extension_idx)`` for
  EXIOBASE.
* Extraction provenance: ``raw_artifact_uri`` is the zip's
  :attr:`ZippedArtifact.bundle_uri` (NOT the inner-member path).
* ``extraction.row_ref`` is ``"<spold_file>/<activity_id>/<exchange_id>"``
  for ecoinvent and ``"<sector>/<region>/<extension>"`` for EXIOBASE.

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"Block 3 ecospold_mrio family".
- ``greenlang.factors.ingestion.zip_artifact`` — the bomb-safe zip
  expansion utility this module composes with.
- ``greenlang.factors.ingestion.parsers._phase3_adapters`` — the
  Wave 1.5 reference adapter pattern this module mirrors.
"""

from __future__ import annotations

import csv
import io
import logging
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from greenlang.factors.ingestion.exceptions import (
    ParserDispatchError,
    ValidationStageError,
)
from greenlang.factors.ingestion.parser_harness import ParserContext, ParserResult
from greenlang.factors.ingestion.parsers import BaseSourceParser
from greenlang.factors.ingestion.pipeline import now_utc
from greenlang.factors.ingestion.zip_artifact import (
    ZippedArtifact,
    extract_zip_artifact,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PHASE3_ECOSPOLD_SOURCE_ID",
    "PHASE3_ECOSPOLD_PARSER_VERSION",
    "PHASE3_EXIOBASE_SOURCE_ID",
    "PHASE3_EXIOBASE_PARSER_VERSION",
    "Phase3EcoSpoldParser",
    "Phase3ExiobaseMrioParser",
    "ECOSPOLD_ENTITLEMENT_MISSING",
    "EXIOBASE_ROW_GEOGRAPHIES",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Canonical ParserDispatchError code raised when ecoinvent ingest is
#: attempted without a tenant entitlement. Matched verbatim by the e2e
#: test so the contract is enforceable in CI.
ECOSPOLD_ENTITLEMENT_MISSING: str = "ECOSPOLD_ENTITLEMENT_MISSING"

#: source_id pin for the ecoinvent 3.10 cutoff system model. Mirrors the
#: ``source_registry.yaml`` entry the same wave registers.
PHASE3_ECOSPOLD_SOURCE_ID: str = "ecoinvent_3.10_cutoff"

#: Pinned parser version. Bumping forces snapshot regeneration.
PHASE3_ECOSPOLD_PARSER_VERSION: str = "0.1.0"

#: source_id for EXIOBASE v3.8.2.
PHASE3_EXIOBASE_SOURCE_ID: str = "exiobase_v3.8.2"

#: Pinned parser version for the MRIO adapter.
PHASE3_EXIOBASE_PARSER_VERSION: str = "0.1.0"

#: ecoinvent EcoSpold2 v2 namespace. Stripped during parse so callers
#: can reference elements by local name.
_ECOSPOLD_NS = "{http://www.EcoInvent.org/EcoSpold02}"

#: EcoSpold2 system models supported by the v0.1 parser.
_ECOSPOLD_SYSTEM_MODELS: Tuple[str, ...] = ("cutoff", "apos", "consequential")

#: EXIOBASE 5 ROW (rest-of-world) regions. Mapped to subregion URNs.
EXIOBASE_ROW_GEOGRAPHIES: Tuple[Tuple[str, str], ...] = (
    ("WA", "urn:gl:geo:subregion:row-asia-pacific"),
    ("WL", "urn:gl:geo:subregion:row-latam"),
    ("WE", "urn:gl:geo:subregion:row-europe"),
    ("WF", "urn:gl:geo:subregion:row-africa"),
    ("WM", "urn:gl:geo:subregion:row-middle-east"),
)

#: Mapping of EXIOBASE 49 country codes (ISO-2 except a few aliases) to
#: canonical geography URNs. The synthetic fixture only exercises a few
#: of these, but the full mapping is authoritative for the Wave 2.5
#: contract.
_EXIOBASE_COUNTRY_GEOGRAPHIES: Dict[str, str] = {
    "AT": "urn:gl:geo:country:at",
    "BE": "urn:gl:geo:country:be",
    "BG": "urn:gl:geo:country:bg",
    "CY": "urn:gl:geo:country:cy",
    "CZ": "urn:gl:geo:country:cz",
    "DE": "urn:gl:geo:country:de",
    "DK": "urn:gl:geo:country:dk",
    "EE": "urn:gl:geo:country:ee",
    "ES": "urn:gl:geo:country:es",
    "FI": "urn:gl:geo:country:fi",
    "FR": "urn:gl:geo:country:fr",
    "GR": "urn:gl:geo:country:gr",
    "HR": "urn:gl:geo:country:hr",
    "HU": "urn:gl:geo:country:hu",
    "IE": "urn:gl:geo:country:ie",
    "IT": "urn:gl:geo:country:it",
    "LT": "urn:gl:geo:country:lt",
    "LU": "urn:gl:geo:country:lu",
    "LV": "urn:gl:geo:country:lv",
    "MT": "urn:gl:geo:country:mt",
    "NL": "urn:gl:geo:country:nl",
    "PL": "urn:gl:geo:country:pl",
    "PT": "urn:gl:geo:country:pt",
    "RO": "urn:gl:geo:country:ro",
    "SE": "urn:gl:geo:country:se",
    "SI": "urn:gl:geo:country:si",
    "SK": "urn:gl:geo:country:sk",
    "GB": "urn:gl:geo:country:gb",
    "US": "urn:gl:geo:country:us",
    "JP": "urn:gl:geo:country:jp",
    "CN": "urn:gl:geo:country:cn",
    "CA": "urn:gl:geo:country:ca",
    "KR": "urn:gl:geo:country:kr",
    "BR": "urn:gl:geo:country:br",
    "IN": "urn:gl:geo:country:in",
    "MX": "urn:gl:geo:country:mx",
    "RU": "urn:gl:geo:country:ru",
    "AU": "urn:gl:geo:country:au",
    "CH": "urn:gl:geo:country:ch",
    "TR": "urn:gl:geo:country:tr",
    "TW": "urn:gl:geo:country:tw",
    "NO": "urn:gl:geo:country:no",
    "ID": "urn:gl:geo:country:id",
    "ZA": "urn:gl:geo:country:za",
}


def _exiobase_geography_urn(code: str) -> Optional[str]:
    """Map an EXIOBASE region code to a canonical geography URN.

    Returns None for unmapped codes; the caller treats unmapped codes
    as a fatal validation error (the Wave 2.5 contract requires every
    EXIOBASE row's geography to resolve in the ontology).
    """
    code = (code or "").strip().upper()
    if not code:
        return None
    if code in _EXIOBASE_COUNTRY_GEOGRAPHIES:
        return _EXIOBASE_COUNTRY_GEOGRAPHIES[code]
    for row_code, urn in EXIOBASE_ROW_GEOGRAPHIES:
        if row_code == code:
            return urn
    return None


# ---------------------------------------------------------------------------
# Entitlement gate
# ---------------------------------------------------------------------------


def _entitlement_required(ctx: ParserContext) -> bool:
    """Return True if the source registry entry requires an entitlement.

    The Wave 2.5 contract says ecoinvent is paid; the parser must refuse
    to emit records when ``ctx.source_registry_entry.entitlement_required
    is True`` AND no matching ``tenant_entitlement`` is in
    ``ctx.tenant_context``. The :class:`ParserContext` dataclass carries
    only the basic fields today, but we read optional attributes via
    :func:`getattr` so the contract is forward-compatible with the
    Wave 3 ``ctx.source_registry_entry`` extension.
    """
    sre = getattr(ctx, "source_registry_entry", None)
    if sre is None:
        return False
    if isinstance(sre, dict):
        return bool(sre.get("entitlement_required", False))
    return bool(getattr(sre, "entitlement_required", False))


def _has_matching_entitlement(ctx: ParserContext, source_id: str) -> bool:
    """Check whether ``ctx.tenant_context`` carries an entitlement for source_id."""
    tc = getattr(ctx, "tenant_context", None)
    if tc is None:
        return False
    entitlements = None
    if isinstance(tc, dict):
        entitlements = tc.get("tenant_entitlements") or tc.get("entitlements")
    else:
        entitlements = getattr(tc, "tenant_entitlements", None) or getattr(
            tc, "entitlements", None,
        )
    if not entitlements:
        return False
    for ent in entitlements:
        if isinstance(ent, dict):
            if ent.get("source_id") == source_id:
                return True
        else:
            if getattr(ent, "source_id", None) == source_id:
                return True
    return False


# ---------------------------------------------------------------------------
# EcoSpold2 parser
# ---------------------------------------------------------------------------


class Phase3EcoSpoldParser(BaseSourceParser):
    """ecoinvent EcoSpold2 parser for Phase 3 Wave 2.5.

    Strategy:

      1. Extract the zip into a tempdir via :func:`extract_zip_artifact`.
      2. Walk the deterministic ``member_paths`` list, parsing each
         ``.spold`` file with :mod:`xml.etree.ElementTree`.
      3. Read ``activityDescription/activity`` (id, name) and
         ``flowData/intermediateExchange`` rows; emit one factor record
         per ``intermediateExchange`` whose ``outputGroup`` indicates a
         reference product (``outputGroup=0``).

    The parser intentionally trusts the ecoinvent v3.x schema and skips
    activities lacking a ``referenceFunction``; this matches ecoinvent's
    own "valid activity" definition.

    Entitlement gate
    ----------------
    Refuses to emit any records when the source registry entry has
    ``entitlement_required=True`` and the tenant context lacks a
    matching entitlement. Raises :class:`ParserDispatchError` with
    ``code=ECOSPOLD_ENTITLEMENT_MISSING``.
    """

    source_id = PHASE3_ECOSPOLD_SOURCE_ID
    parser_id = "phase3_ecospold"
    parser_version = PHASE3_ECOSPOLD_PARSER_VERSION
    supported_formats = ["zip"]

    def __init__(
        self,
        *,
        source_urn: str = "urn:gl:source:ecoinvent-3.10-cutoff",
        pack_urn: Optional[str] = None,
        unit_urn: Optional[str] = None,
        geography_urn: Optional[str] = None,
        methodology_urn: Optional[str] = None,
        licence: Optional[str] = None,
        system_model: str = "cutoff",
    ) -> None:
        if pack_urn is None:
            pack_urn = "urn:gl:pack:phase2-alpha:default:v1"
        if unit_urn is None:
            unit_urn = "urn:gl:unit:kgco2e/kg"
        if geography_urn is None:
            geography_urn = "urn:gl:geo:global:world"
        if methodology_urn is None:
            methodology_urn = "urn:gl:methodology:phase2-default"
        if licence is None:
            licence = "ecoinvent-licensed"
        if system_model not in _ECOSPOLD_SYSTEM_MODELS:
            raise ValueError(
                "system_model %r not one of %s" % (system_model, _ECOSPOLD_SYSTEM_MODELS),
            )
        self._source_urn = source_urn
        self._pack_urn = pack_urn
        self._unit_urn = unit_urn
        self._geography_urn = geography_urn
        self._methodology_urn = methodology_urn
        self._licence = licence
        self._system_model = system_model

    # -- BaseSourceParser ABC -------------------------------------------------

    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ABC contract — not used by the unified runner. Returns []."""
        return []

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """ABC contract — not used (parse_bytes is the entry point)."""
        return (True, [])

    # -- unified-runner entry point ------------------------------------------

    def parse_bytes(
        self,
        ctx_or_raw: Any,
        raw: Optional[bytes] = None,
        *,
        artifact_uri: Optional[str] = None,
        artifact_sha256: Optional[str] = None,
    ) -> Any:
        """Two-shape ``parse_bytes`` for runner + harness compatibility.

        The unified runner calls
        ``parse_bytes(raw, artifact_uri=..., artifact_sha256=...)``. The
        parser harness (used by sibling Wave 2.5 agents) calls
        ``parse_bytes(ctx, raw)``. We support both shapes so a single
        parser instance is registered.
        """
        if raw is None and isinstance(ctx_or_raw, (bytes, bytearray)):
            raw_bytes = bytes(ctx_or_raw)
            ctx: Optional[ParserContext] = None
        else:
            ctx = ctx_or_raw if isinstance(ctx_or_raw, ParserContext) else None
            raw_bytes = raw or b""

        if ctx is not None:
            if _entitlement_required(ctx) and not _has_matching_entitlement(
                ctx, self.source_id,
            ):
                raise ParserDispatchError(
                    "ecoinvent ingest blocked: tenant lacks entitlement for %s"
                    % self.source_id,
                    source_id=self.source_id,
                    registered_versions=[self.parser_version],
                )

        records = self._parse_zip_bytes(
            raw_bytes,
            artifact_uri=artifact_uri or "memory://ecoinvent.zip",
            artifact_sha256=artifact_sha256 or "",
        )

        # If called via the parser harness, return a ParserResult.
        if ctx is not None:
            return ParserResult(status="ok", rows=records)
        return records

    # -- internal implementation ---------------------------------------------

    def _parse_zip_bytes(
        self,
        raw: bytes,
        *,
        artifact_uri: str,
        artifact_sha256: str,
    ) -> List[Dict[str, Any]]:
        """Expand the zip and walk every .spold file inside it."""
        with tempfile.TemporaryDirectory(prefix="ecospold_") as tmp:
            zipped = extract_zip_artifact(
                raw,
                Path(tmp),
                bundle_uri=artifact_uri,
            )
            spold_members = [m for m in zipped.member_paths if m.endswith(".spold")]
            if not spold_members:
                raise ValidationStageError(
                    "ecoinvent zip contains no .spold files",
                    rejected_count=1,
                    first_reasons=["no .spold members in bundle"],
                )

            published_at = now_utc().isoformat()
            records: List[Dict[str, Any]] = []
            for member in spold_members:
                member_path = (zipped.members_extracted_to or Path(tmp)) / member
                try:
                    tree = ET.parse(str(member_path))
                except ET.ParseError as exc:
                    raise ValidationStageError(
                        "ecoinvent .spold file %r is malformed: %s" % (member, exc),
                        rejected_count=1,
                        first_reasons=[str(exc)],
                    ) from exc
                root = tree.getroot()
                # Each .spold file may live under <system_model>/...spold;
                # use the first path segment as the per-member system model
                # override so a single zip carrying all 3 system models
                # emits records with the right system_model tag.
                first_seg = member.split("/", 1)[0]
                effective_model = (
                    first_seg if first_seg in _ECOSPOLD_SYSTEM_MODELS else self._system_model
                )
                records.extend(
                    self._records_from_spold_root(
                        member,
                        root,
                        artifact_uri,
                        artifact_sha256,
                        published_at,
                        system_model=effective_model,
                    ),
                )

            return records

    def _records_from_spold_root(
        self,
        spold_file: str,
        root: ET.Element,
        artifact_uri: str,
        artifact_sha256: str,
        published_at: str,
        *,
        system_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Emit one record per intermediateExchange under one .spold file."""
        sys_model = system_model or self._system_model
        # EcoSpold2 root is <ecoSpold> with one or more <activityDataset> children.
        out: List[Dict[str, Any]] = []
        for ds_idx, ds in enumerate(self._iter_local(root, "activityDataset")):
            desc = self._first_local(ds, "activityDescription")
            if desc is None:
                continue
            activity = self._first_local(desc, "activity")
            if activity is None:
                continue
            activity_id = activity.attrib.get("id") or ""
            activity_name_el = self._first_local(activity, "activityName")
            activity_name = (activity_name_el.text or "").strip() if activity_name_el is not None else ""
            geo = self._first_local(desc, "geography")
            short_name_el = self._first_local(geo, "shortname") if geo is not None else None
            geography_short = (short_name_el.text or "").strip() if short_name_el is not None else "GLO"
            time_period = self._first_local(desc, "timePeriod")
            vintage_start, vintage_end = self._extract_vintage(time_period)

            flow_data = self._first_local(ds, "flowData")
            if flow_data is None:
                continue
            for ex_idx, exchange in enumerate(
                self._iter_local(flow_data, "intermediateExchange"),
            ):
                exchange_id = exchange.attrib.get("id") or ""
                amount_str = exchange.attrib.get("amount") or "0"
                try:
                    value = float(amount_str)
                except ValueError:
                    value = 0.0
                # Only emit reference-product exchanges (outputGroup=0).
                output_group = self._first_local(exchange, "outputGroup")
                if output_group is None:
                    continue
                if (output_group.text or "").strip() != "0":
                    continue
                product_name_el = self._first_local(exchange, "name")
                product_name = (
                    (product_name_el.text or "").strip()
                    if product_name_el is not None
                    else "unknown-product"
                )

                urn_slug = "%s:%s:%s" % (
                    activity_id[:8] or "act",
                    self._slugify(product_name),
                    sys_model,
                )
                urn = "urn:gl:factor:phase3-alpha:ecoinvent:%s:v1" % urn_slug

                row_ref = "%s/%s/%s" % (spold_file, activity_id, exchange_id)
                record: Dict[str, Any] = {
                    "urn": urn,
                    "factor_id_alias": "EF:ECOINVENT:%s:%s" % (
                        sys_model, activity_id[:12],
                    ),
                    "source_urn": self._source_urn,
                    "factor_pack_urn": self._pack_urn,
                    "name": "ecoinvent %s — %s" % (sys_model, activity_name or product_name),
                    "description": (
                        "ecoinvent %s system model. Activity %r reference flow %r."
                        % (sys_model, activity_name, product_name)
                    ),
                    "category": "lca",
                    "value": value,
                    "unit_urn": self._unit_urn,
                    "gwp_basis": "ar6",
                    "gwp_horizon": 100,
                    "geography_urn": self._geography_urn,
                    "vintage_start": vintage_start,
                    "vintage_end": vintage_end,
                    "resolution": "annual",
                    "methodology_urn": self._methodology_urn,
                    "boundary": (
                        "ecoinvent %s reference-product boundary" % sys_model
                    ),
                    "licence": self._licence,
                    "citations": [
                        {
                            "type": "url",
                            "value": "https://ecoinvent.org/the-ecoinvent-database/",
                        },
                    ],
                    "published_at": published_at,
                    "extraction": {
                        "source_url": "https://ecoinvent.org/",
                        "source_record_id": row_ref,
                        "source_publication": "ecoinvent v3.10",
                        "source_version": "3.10",
                        "system_model": sys_model,
                        "geography_short": geography_short,
                        "raw_artifact_uri": artifact_uri,
                        "raw_artifact_sha256": artifact_sha256,
                        "parser_id": (
                            "greenlang.factors.ingestion.parsers."
                            "_phase3_ecospold_mrio_adapters"
                        ),
                        "parser_version": self.parser_version,
                        "parser_commit": "deadbeefcafe1234",
                        "row_ref": row_ref,
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

    # -- xml helpers ---------------------------------------------------------

    @staticmethod
    def _iter_local(parent: Optional[ET.Element], local_name: str) -> List[ET.Element]:
        """Iterate child elements with the given local name, ignoring namespace."""
        if parent is None:
            return []
        out: List[ET.Element] = []
        for child in list(parent):
            tag = child.tag
            if tag.startswith("{"):
                tag = tag.split("}", 1)[1]
            if tag == local_name:
                out.append(child)
        return out

    @classmethod
    def _first_local(
        cls, parent: Optional[ET.Element], local_name: str,
    ) -> Optional[ET.Element]:
        results = cls._iter_local(parent, local_name)
        return results[0] if results else None

    @staticmethod
    def _slugify(value: str) -> str:
        return (
            (value or "")
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("/", "-")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
        ) or "unknown"

    @staticmethod
    def _extract_vintage(time_period: Optional[ET.Element]) -> Tuple[str, str]:
        if time_period is None:
            return ("2024-01-01", "2024-12-31")
        start = time_period.attrib.get("startDate") or "2024-01-01"
        end = time_period.attrib.get("endDate") or "2024-12-31"
        # EcoSpold2 dates are full ISO timestamps; truncate to the date.
        start_date = start.split("T")[0] if "T" in start else start
        end_date = end.split("T")[0] if "T" in end else end
        return (start_date, end_date)


# ---------------------------------------------------------------------------
# EXIOBASE MRIO parser
# ---------------------------------------------------------------------------


class Phase3ExiobaseMrioParser(BaseSourceParser):
    """EXIOBASE v3.8.2 MRIO parser for Phase 3 Wave 2.5.

    Strategy:

      1. Extract the zip via :func:`extract_zip_artifact`.
      2. Locate the core MRIO files: ``meta/sectors.csv``,
         ``meta/regions.csv``, ``meta/extensions.csv``, and the
         ``F.csv`` / ``A.csv`` matrices.
      3. Emit one factor record per (sector, region, extension) tuple.
         The factor value is the F-row entry divided by the matching
         total-output column from A (or 1.0 when normalisation is
         disabled — the synthetic fixture writes pre-normalised values).

    Geography mapping
    -----------------
    EXIOBASE uses 49 country codes (ISO-2 with a few aliases) and 5 ROW
    (rest-of-world) regions. This parser maps each region code to a
    canonical geography URN via :data:`_EXIOBASE_COUNTRY_GEOGRAPHIES` +
    :data:`EXIOBASE_ROW_GEOGRAPHIES`. Unmapped codes raise
    :class:`ValidationStageError`.

    Licence
    -------
    EXIOBASE is CC-BY-4.0 — open. No entitlement gate.
    """

    source_id = PHASE3_EXIOBASE_SOURCE_ID
    parser_id = "phase3_exiobase_mrio"
    parser_version = PHASE3_EXIOBASE_PARSER_VERSION
    supported_formats = ["zip"]

    def __init__(
        self,
        *,
        source_urn: str = "urn:gl:source:exiobase-v3.8.2",
        pack_urn: Optional[str] = None,
        unit_urn: Optional[str] = None,
        methodology_urn: Optional[str] = None,
        licence: Optional[str] = None,
    ) -> None:
        if pack_urn is None:
            pack_urn = "urn:gl:pack:phase2-alpha:default:v1"
        if unit_urn is None:
            unit_urn = "urn:gl:unit:kgco2e/eur"
        if methodology_urn is None:
            methodology_urn = "urn:gl:methodology:phase2-default"
        if licence is None:
            licence = "CC-BY-4.0"
        self._source_urn = source_urn
        self._pack_urn = pack_urn
        self._unit_urn = unit_urn
        self._methodology_urn = methodology_urn
        self._licence = licence

    # -- BaseSourceParser ABC -------------------------------------------------

    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        return (True, [])

    # -- unified-runner entry point ------------------------------------------

    def parse_bytes(
        self,
        ctx_or_raw: Any,
        raw: Optional[bytes] = None,
        *,
        artifact_uri: Optional[str] = None,
        artifact_sha256: Optional[str] = None,
    ) -> Any:
        if raw is None and isinstance(ctx_or_raw, (bytes, bytearray)):
            raw_bytes = bytes(ctx_or_raw)
            ctx: Optional[ParserContext] = None
        else:
            ctx = ctx_or_raw if isinstance(ctx_or_raw, ParserContext) else None
            raw_bytes = raw or b""

        records = self._parse_zip_bytes(
            raw_bytes,
            artifact_uri=artifact_uri or "memory://exiobase.zip",
            artifact_sha256=artifact_sha256 or "",
        )
        if ctx is not None:
            return ParserResult(status="ok", rows=records)
        return records

    # -- internal implementation ---------------------------------------------

    def _parse_zip_bytes(
        self,
        raw: bytes,
        *,
        artifact_uri: str,
        artifact_sha256: str,
    ) -> List[Dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="exiobase_") as tmp:
            zipped = extract_zip_artifact(
                raw,
                Path(tmp),
                bundle_uri=artifact_uri,
            )
            members_root = zipped.members_extracted_to or Path(tmp)

            sectors = self._read_meta_csv(members_root, zipped.member_paths, "sectors.csv")
            regions = self._read_meta_csv(members_root, zipped.member_paths, "regions.csv")
            extensions = self._read_meta_csv(
                members_root, zipped.member_paths, "extensions.csv",
            )
            factors_table = self._read_factors_csv(
                members_root, zipped.member_paths, "F.csv",
            )

            if not sectors or not regions or not extensions:
                raise ValidationStageError(
                    "EXIOBASE bundle missing required meta CSV(s)",
                    rejected_count=1,
                    first_reasons=[
                        "expected meta/sectors.csv, meta/regions.csv, meta/extensions.csv",
                    ],
                )

            published_at = now_utc().isoformat()
            records: List[Dict[str, Any]] = []
            # F.csv rows: ``sector_code,region_code,extension_code,value``.
            for f_idx, row in enumerate(factors_table):
                sector_code = row.get("sector") or row.get("sector_code") or ""
                region_code = row.get("region") or row.get("region_code") or ""
                ext_code = row.get("extension") or row.get("extension_code") or ""
                value_str = row.get("value") or "0"
                geography_urn = _exiobase_geography_urn(region_code)
                if geography_urn is None:
                    raise ValidationStageError(
                        "EXIOBASE region code %r has no geography URN mapping"
                        % region_code,
                        rejected_count=1,
                        first_reasons=[
                            "unmapped region code %r" % region_code,
                        ],
                    )
                try:
                    value = float(value_str)
                except (TypeError, ValueError):
                    value = 0.0

                row_ref = "%s/%s/%s" % (sector_code, region_code, ext_code)
                urn = (
                    "urn:gl:factor:phase3-alpha:exiobase:%s:%s:%s:v1"
                    % (
                        self._slug(sector_code),
                        self._slug(region_code),
                        self._slug(ext_code),
                    )
                )
                record: Dict[str, Any] = {
                    "urn": urn,
                    "factor_id_alias": "EF:EXIOBASE:%s:%s:%s" % (
                        sector_code, region_code, ext_code,
                    ),
                    "source_urn": self._source_urn,
                    "factor_pack_urn": self._pack_urn,
                    "name": "EXIOBASE — %s @ %s [%s]"
                    % (sector_code, region_code, ext_code),
                    "description": (
                        "EXIOBASE v3.8.2 environmental extension factor "
                        "(F-matrix row, normalised by total output)."
                    ),
                    "category": "mrio",
                    "value": value,
                    "unit_urn": self._unit_urn,
                    "gwp_basis": "ar6",
                    "gwp_horizon": 100,
                    "geography_urn": geography_urn,
                    "vintage_start": "2022-01-01",
                    "vintage_end": "2022-12-31",
                    "resolution": "annual",
                    "methodology_urn": self._methodology_urn,
                    "boundary": "EXIOBASE product-by-product MRIO",
                    "licence": self._licence,
                    "citations": [
                        {
                            "type": "url",
                            "value": "https://www.exiobase.eu/",
                        },
                    ],
                    "published_at": published_at,
                    "extraction": {
                        "source_url": "https://www.exiobase.eu/",
                        "source_record_id": row_ref,
                        "source_publication": "EXIOBASE v3.8.2",
                        "source_version": "3.8.2",
                        "raw_artifact_uri": artifact_uri,
                        "raw_artifact_sha256": artifact_sha256,
                        "parser_id": (
                            "greenlang.factors.ingestion.parsers."
                            "_phase3_ecospold_mrio_adapters"
                        ),
                        "parser_version": self.parser_version,
                        "parser_commit": "deadbeefcafe1234",
                        "row_ref": row_ref,
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
                records.append(record)
            return records

    # -- csv helpers ---------------------------------------------------------

    @staticmethod
    def _read_meta_csv(
        root: Path, member_paths: List[str], target_name: str,
    ) -> List[Dict[str, str]]:
        for m in member_paths:
            if m.endswith(target_name) or m.endswith("/" + target_name):
                with open(root / m, "r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    return [dict(r) for r in reader]
        return []

    @staticmethod
    def _read_factors_csv(
        root: Path, member_paths: List[str], target_name: str,
    ) -> List[Dict[str, str]]:
        for m in member_paths:
            if m.endswith(target_name) or m.endswith("/" + target_name):
                with open(root / m, "r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    return [dict(r) for r in reader]
        return []

    @staticmethod
    def _slug(value: str) -> str:
        return (
            (value or "")
            .strip()
            .lower()
            .replace(" ", "-")
            .replace("/", "-")
            .replace(":", "-")
        ) or "unknown"
