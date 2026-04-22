# -*- coding: utf-8 -*-
"""
PACT (Partnership for Carbon Transparency) product-data parser.

PACT defines a Pathfinder Framework data-exchange spec for Product
Carbon Footprints (PCF). Each exchange ``ProductFootprint`` object maps
naturally to a GreenLang material-embodied factor.

This parser is backward compatible with PACT v2 (existing F9 callers)
and additionally understands the PACT v3 schema extensions
(WBCSD / PACT Steering, published 2026) that add:

* ``assurance`` block (``coverage``, ``level`` of none / limited /
  reasonable, ``provider``) per PACT v3 Appendix A §4.3
* ``dataQualityIndicator`` object with 1-5 representativeness scores
  (``geographicalRepresentativeness``, ``temporalRepresentativeness``,
  ``technologicalRepresentativeness``) per PACT v3 §4.2.3
* ``primaryDataShare`` float (0.0-1.0) replacing the older
  ``supplierPrimaryDataShare`` in v2
* ``crossSectoralStandardsUsed`` list + ``productOrSectorSpecificRules``
  list for the PCF-methodology cross-reference per PACT v3 §3.2

v2 input (still supported)::

    {
        "id": "urn:gl:pact:product:ACME-STEEL-HRC-001",
        "productName": "Hot-rolled steel coil",
        "productCategoryCpc": "41237",
        "pcf": {
            "declaredUnit": "kg",
            "unitaryProductAmount": "1.0",
            "pCfExcludingBiogenic": "2.34",
            "pCfIncludingBiogenic": "2.35",
            "fossilGhgEmissions": "2.30",
            "biogenicCarbonEmissions": "0.05",
            "geographyCountrySubdivision": "DE"
        },
        "companyName": "Acme Steel",
        "periodCoveredStart": "2024-01-01",
        "periodCoveredEnd": "2024-12-31",
        "version": 2,
        "pcfSpec": "2.0.0"
    }

v3 input (superset — all v2 keys remain valid)::

    {
        "id": "urn:gl:pact:product:ACME-STEEL-HRC-001",
        "productName": "Hot-rolled steel coil",
        "productCategoryCpc": "41237",
        "pcf": {
            "declaredUnit": "kg",
            "pCfExcludingBiogenic": "2.34",
            "biogenicCarbonEmissions": "0.05",
            "geographyCountrySubdivision": "DE",
            "primaryDataShare": 0.72,
            "dataQualityIndicator": {
                "coveragePercent": 95.0,
                "geographicalRepresentativeness": 4,
                "temporalRepresentativeness": 5,
                "technologicalRepresentativeness": 4,
                "dataQualityRating": 4
            },
            "crossSectoralStandardsUsed": ["GHG Protocol Product Standard"],
            "productOrSectorSpecificRules": [
                {"operator": "PEF", "ruleNames": ["Steel PEFCR"], "version": "3.1"}
            ]
        },
        "assurance": {
            "coverage": "product line",
            "level": "reasonable",
            "provider": "TUV Rheinland"
        },
        "companyName": "Acme Steel",
        "periodCoveredStart": "2024-01-01",
        "periodCoveredEnd": "2024-12-31",
        "version": 1,
        "pcfSpec": "3.0.0"
    }

Reference: WBCSD PACT, "Pathfinder Framework v3: Guidance for the
Accounting and Exchange of Product Life Cycle Emissions" (2026-Q1).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from greenlang.data.canonical_v2 import (
    ActivitySchema,
    Explainability,
    FactorFamily,
    FactorParameters,
    FormulaType,
    Jurisdiction,
    MethodProfile,
    PrimaryDataFlag,
    RedistributionClass,
    Verification,
    VerificationStatus,
)
from greenlang.data.emission_factor_record import (
    Boundary,
    DataQualityScore,
    EmissionFactorRecord,
    GeographyLevel,
    GHGVectors,
    GWPSet,
    GWPValues,
    LicenseInfo,
    Methodology,
    Scope,
    SourceProvenance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PACT v3 schema enums and helpers
# ---------------------------------------------------------------------------


class PACTAssuranceLevel(str, Enum):
    """PACT v3 Appendix A §4.3 assurance classifications.

    The three values mirror ISAE 3000 terminology used by the PACT
    steering committee:

    * ``NONE`` — no third-party assurance, self-declared data only
    * ``LIMITED`` — limited assurance engagement (ISAE 3000 (Revised)
      "limited assurance" conclusion)
    * ``REASONABLE`` — reasonable assurance engagement (higher bar,
      equivalent to financial-audit-level assurance)
    """

    NONE = "none"
    LIMITED = "limited"
    REASONABLE = "reasonable"

    @classmethod
    def parse(cls, raw: Any) -> "PACTAssuranceLevel":
        """Parse any v3 assurance-level payload to the enum.

        Unknown or missing values default to ``NONE``. Accepts raw
        strings of any case plus common PACT synonyms.
        """
        if raw is None:
            return cls.NONE
        token = str(raw).strip().lower()
        if token in {"", "none", "unassured", "self_declared", "self-declared"}:
            return cls.NONE
        if token in {"limited", "limited_assurance", "limited-assurance"}:
            return cls.LIMITED
        if token in {"reasonable", "reasonable_assurance", "reasonable-assurance"}:
            return cls.REASONABLE
        raise ValueError(f"Unknown PACT assurance level: {raw!r}")


@dataclass(frozen=True)
class PACTDataQualityIndicatorV3:
    """Structured PACT v3 §4.2.3 Data Quality Indicator (DQI) payload.

    The four component scores (``geographical``, ``temporal``,
    ``technological``, ``dataQualityRating``) are 1-5 per PACT v3
    pedigree matrix; ``coverage_percent`` is 0-100.
    """

    coverage_percent: Optional[float] = None
    geographical: Optional[int] = None
    temporal: Optional[int] = None
    technological: Optional[int] = None
    data_quality_rating: Optional[int] = None


def _parse_dqi_v3(raw: Any) -> Optional[PACTDataQualityIndicatorV3]:
    """Parse a raw dataQualityIndicator block into the structured form.

    Returns ``None`` when no v3 DQI is present. Silently tolerates
    missing sub-fields — v3 only requires ``dataQualityRating``.
    """
    if not isinstance(raw, dict) or not raw:
        return None

    def _as_int(v: Any) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def _as_float(v: Any) -> Optional[float]:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return PACTDataQualityIndicatorV3(
        coverage_percent=_as_float(raw.get("coveragePercent")),
        geographical=_as_int(raw.get("geographicalRepresentativeness")),
        temporal=_as_int(raw.get("temporalRepresentativeness")),
        technological=_as_int(raw.get("technologicalRepresentativeness")),
        data_quality_rating=_as_int(raw.get("dataQualityRating")),
    )


def _detect_spec_version(row: Dict[str, Any]) -> str:
    """Return the pcfSpec / version string; defaults to '2.0.0' for v2 back-compat."""
    spec = str(row.get("pcfSpec") or "2.0.0")
    return spec


def _is_v3(row: Dict[str, Any]) -> bool:
    """Detect whether a row follows PACT v3 (pcfSpec 3.x or v3 fields present)."""
    spec = _detect_spec_version(row)
    if spec.startswith("3"):
        return True
    pcf = row.get("pcf") or {}
    v3_keys = {
        "primaryDataShare",
        "dataQualityIndicator",
        "crossSectoralStandardsUsed",
        "productOrSectorSpecificRules",
    }
    if any(key in pcf for key in v3_keys):
        return True
    if "assurance" in row:
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_pact_rows(rows: Iterable[Dict[str, Any]]) -> List[EmissionFactorRecord]:
    """Parse a collection of PACT Product Footprint rows.

    Handles both v2 and v3 schemas transparently. v3 rows are upgraded
    to carry assurance + DQI + primaryDataShare via the canonical
    ``FactorParameters.uncertainty_*`` and ``Verification`` slots plus
    ``explainability.assumptions`` text — no schema migration is
    required on the consuming side.
    """
    out: List[EmissionFactorRecord] = []
    for i, row in enumerate(rows):
        try:
            out.append(_row_to_record(row))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping PACT row %d: %s", i, exc)
    logger.info("Parsed %d PACT product footprints", len(out))
    return out


def _row_to_record(row: Dict[str, Any]) -> EmissionFactorRecord:
    pcf = row.get("pcf") or {}
    product_id = row["id"]
    product_name = row.get("productName", product_id)
    country = str(pcf.get("geographyCountrySubdivision", "GLOBAL"))[:2].upper()

    # PACT values are strings per the spec - coerce to float.
    fossil_co2e = float(pcf.get("pCfExcludingBiogenic", pcf.get("fossilGhgEmissions", 0.0)))
    biogenic_co2 = float(pcf.get("biogenicCarbonEmissions", 0.0))
    declared_unit = str(pcf.get("declaredUnit", "kg"))

    period_start = str(row.get("periodCoveredStart", "2024-01-01"))
    period_end = str(row.get("periodCoveredEnd", "2024-12-31"))

    spec_version = _detect_spec_version(row)
    is_v3 = _is_v3(row)

    factor_id = f"EF:PACT:{product_id}"

    # ------------------------------------------------------------------
    # v2 fields (always parsed)
    # ------------------------------------------------------------------
    supplier_primary_data_share_v2 = _as_optional_float(
        pcf.get("supplierPrimaryDataShare")
    )

    # ------------------------------------------------------------------
    # v3 fields (optional; populated only when present)
    # ------------------------------------------------------------------
    assurance_raw = row.get("assurance") or {}
    assurance_level = PACTAssuranceLevel.parse(assurance_raw.get("level"))
    assurance_coverage = assurance_raw.get("coverage")
    assurance_provider = assurance_raw.get("provider")

    dqi_v3 = _parse_dqi_v3(pcf.get("dataQualityIndicator"))

    # PACT v3 renames supplierPrimaryDataShare -> primaryDataShare.  We
    # prefer the v3 name when both are given.
    primary_data_share_v3 = _as_optional_float(pcf.get("primaryDataShare"))
    if primary_data_share_v3 is None:
        primary_data_share_v3 = supplier_primary_data_share_v2

    cross_sectoral_rules = pcf.get("crossSectoralStandardsUsed") or []
    if not isinstance(cross_sectoral_rules, list):
        cross_sectoral_rules = [str(cross_sectoral_rules)]
    cross_sectoral_version = _extract_cross_sectoral_version(
        pcf.get("productOrSectorSpecificRules") or []
    )

    geographic_rep_v3 = dqi_v3.geographical if dqi_v3 else None
    temporal_rep_v3 = dqi_v3.temporal if dqi_v3 else None
    technological_rep_v3 = dqi_v3.technological if dqi_v3 else None
    data_quality_indicator_v3 = dqi_v3.data_quality_rating if dqi_v3 else None

    # ------------------------------------------------------------------
    # Map assurance -> Verification.status
    #   reasonable -> REGULATOR_APPROVED-equivalent strength
    #   limited    -> EXTERNAL_VERIFIED
    #   none       -> UNVERIFIED (v3) / EXTERNAL_VERIFIED (v2 default)
    # ------------------------------------------------------------------
    if assurance_level is PACTAssuranceLevel.REASONABLE:
        verif_status = VerificationStatus.REGULATOR_APPROVED
    elif assurance_level is PACTAssuranceLevel.LIMITED:
        verif_status = VerificationStatus.EXTERNAL_VERIFIED
    elif is_v3 and assurance_level is PACTAssuranceLevel.NONE:
        verif_status = VerificationStatus.UNVERIFIED
    else:
        # v2 default: PACT spec calls for external verification by the
        # contributing member
        verif_status = VerificationStatus.EXTERNAL_VERIFIED

    verifier = (
        assurance_provider
        or row.get("verifiedBy")
        or "PACT-conformant external verifier"
    )

    # ------------------------------------------------------------------
    # DQS — prefer v3 pedigree scores when present, otherwise keep the
    # v2 curator defaults
    # ------------------------------------------------------------------
    dqs = DataQualityScore(
        temporal=_clamp_1_5(temporal_rep_v3, default=5),
        geographical=_clamp_1_5(geographic_rep_v3, default=4),
        technological=_clamp_1_5(technological_rep_v3, default=4),
        representativeness=_clamp_1_5(data_quality_indicator_v3, default=4),
        methodological=5,
    )

    # ------------------------------------------------------------------
    # Assumptions — carry v3 fields explicitly for the /explain endpoint
    # ------------------------------------------------------------------
    assumptions: List[str] = [
        "Boundary: cradle-to-gate unless the PACT object says otherwise.",
        "Biogenic carbon reported separately per PACT + GHG Protocol Product Standard.",
    ]
    if is_v3:
        assumptions.append(f"PACT Pathfinder Framework v3 — spec {spec_version}.")
        assumptions.append(
            f"Assurance level: {assurance_level.value}"
            + (f" ({assurance_coverage})" if assurance_coverage else "")
            + (f" by {assurance_provider}" if assurance_provider else "")
            + "."
        )
        if primary_data_share_v3 is not None:
            assumptions.append(
                f"Primary data share (v3 §4.2.2): {primary_data_share_v3:.2f}."
            )
        if data_quality_indicator_v3 is not None:
            assumptions.append(
                f"PACT v3 DQI composite score: {data_quality_indicator_v3}/5."
            )
        if cross_sectoral_rules:
            assumptions.append(
                "Cross-sectoral standards used: "
                + ", ".join(str(s) for s in cross_sectoral_rules)
                + "."
            )
        if cross_sectoral_version:
            assumptions.append(
                f"Product/sector-specific rules version: {cross_sectoral_version}."
            )

    # ------------------------------------------------------------------
    # FactorParameters — pack v3 primary_data_share into uncertainty_low
    # slot hint? No — we use supplier_specific + biogenic_share and
    # stash the v3 DQI bundle in the explainability/extras instead so
    # we don't change the FactorParameters schema.
    # ------------------------------------------------------------------
    parameters = FactorParameters(
        scope_applicability=["scope3"],
        biogenic_share=(
            (biogenic_co2 / (fossil_co2e + biogenic_co2))
            if (fossil_co2e + biogenic_co2) > 0
            else 0.0
        ),
        supplier_specific=primary_data_share_v3 is not None
        and primary_data_share_v3 >= 0.5,
    )

    activity_tags = ["product_carbon", "pact"]
    if is_v3:
        activity_tags.append("pact_v3")
    tags = ["pact", product_name]
    if is_v3:
        tags.append("pact_v3")
        tags.append(f"assurance_{assurance_level.value}")

    compliance_frameworks = ["PACT", "GHG_Protocol_Product", "ISO_14067"]
    if is_v3:
        compliance_frameworks.append("PACT_v3")

    explainability = Explainability(
        assumptions=assumptions,
        fallback_rank=2,  # supplier-specific
        rationale=(
            f"PACT v{spec_version} product footprint for {product_name}."
            if is_v3
            else f"PACT product footprint for {product_name}."
        ),
    )

    return EmissionFactorRecord(
        factor_id=factor_id,
        fuel_type=product_name,
        unit=declared_unit,
        geography=country,
        geography_level=GeographyLevel.COUNTRY,
        vectors=GHGVectors(
            CO2=fossil_co2e,
            CH4=0.0,
            N2O=0.0,
            biogenic_CO2=biogenic_co2,
        ),
        gwp_100yr=GWPValues(gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273),
        scope=Scope.SCOPE_3,
        boundary=Boundary.CRADLE_TO_GATE,
        provenance=SourceProvenance(
            source_org=row.get("companyName", "PACT exchange"),
            source_publication="PACT Pathfinder Framework " + spec_version,
            source_year=int(period_start[:4]),
            methodology=Methodology.LCA,
        ),
        valid_from=date.fromisoformat(period_start),
        valid_to=date.fromisoformat(period_end),
        uncertainty_95ci=0.15,
        dqs=dqs,
        license_info=LicenseInfo(
            license="PACT Pathfinder terms",
            redistribution_allowed=True,
            commercial_use_allowed=True,
            attribution_required=True,
        ),
        source_id="pact_pathfinder" if is_v3 else "pact_exchange",
        source_release=spec_version,
        source_record_id=product_id,
        release_version="pact-" + str(row.get("version", "1")),
        license_class="registry_terms",
        compliance_frameworks=compliance_frameworks,
        activity_tags=activity_tags,
        sector_tags=[row.get("productCategoryCpc", "unknown_cpc")],
        tags=tags,
        factor_family=FactorFamily.MATERIAL_EMBODIED.value,
        factor_name=product_name,
        method_profile=MethodProfile.PRODUCT_CARBON.value,
        factor_version="1.0.0",
        formula_type=FormulaType.LCA.value,
        jurisdiction=Jurisdiction(country=country),
        activity_schema=ActivitySchema(
            category="product_carbon_footprint",
            sub_category=row.get("productCategoryCpc"),
            classification_codes=[f"CPC:{row.get('productCategoryCpc', '')}"],
        ),
        parameters=parameters,
        verification=Verification(
            status=verif_status,
            verified_by=verifier,
            verification_reference=(
                f"PACT v3 assurance={assurance_level.value}" if is_v3 else None
            ),
        ),
        explainability=explainability,
        primary_data_flag=(
            PrimaryDataFlag.PRIMARY.value
            if (primary_data_share_v3 is not None and primary_data_share_v3 >= 0.8)
            else PrimaryDataFlag.PRIMARY_MODELED.value
        ),
        redistribution_class=RedistributionClass.RESTRICTED.value,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_optional_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _clamp_1_5(v: Optional[int], default: int) -> int:
    """Clamp a PACT v3 pedigree score to 1-5, falling back to ``default``."""
    if v is None:
        return default
    if v < 1:
        return 1
    if v > 5:
        return 5
    return v


def _extract_cross_sectoral_version(rules: Any) -> Optional[str]:
    """Pick out the first 'version' string from a productOrSectorSpecificRules list."""
    if not isinstance(rules, list):
        return None
    for entry in rules:
        if isinstance(entry, dict):
            version = entry.get("version")
            if version:
                return str(version)
    return None


__all__ = [
    "parse_pact_rows",
    "PACTAssuranceLevel",
    "PACTDataQualityIndicatorV3",
]
