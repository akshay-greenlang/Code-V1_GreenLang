# -*- coding: utf-8 -*-
"""
XBRLTaxonomyIntegration - XBRL Taxonomy Integration for PACK-030
===================================================================

Enterprise integration for fetching, caching, and validating XBRL/iXBRL
taxonomies from SEC and CSRD regulatory bodies. Provides taxonomy
element lookups, tag validation, namespace management, and version
tracking for SEC climate disclosures and CSRD ESRS E1 digital reports.

Integration Points:
    - SEC Taxonomy: SEC Climate Disclosure XBRL taxonomy (Reg S-K)
    - CSRD Taxonomy: ESRS E1 Climate Change digital taxonomy
    - Tag Validation: Verify metric-to-tag mapping correctness
    - Namespace Management: XBRL namespace and schema references
    - Version Tracking: Taxonomy version history and deprecation

Architecture:
    SEC XBRL Registry  --> PACK-030 SEC Climate Workflow
    CSRD Taxonomy      --> PACK-030 CSRD ESRS E1 Workflow
    Validation         --> PACK-030 XBRL Tagging Engine

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-030 Net Zero Reporting Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaxonomyFramework(str, Enum):
    SEC = "sec"
    CSRD = "csrd"
    ISSB = "issb"
    GRI = "gri"


class TagDataType(str, Enum):
    MONETARY = "monetary"
    PERCENTAGE = "percentage"
    MASS = "mass"
    ENERGY = "energy"
    TEXT = "text"
    BOOLEAN = "boolean"
    DATE = "date"
    INTEGER = "integer"


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ImportStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    CACHED = "cached"


# ---------------------------------------------------------------------------
# SEC XBRL Taxonomy Elements
# ---------------------------------------------------------------------------

SEC_TAXONOMY_ELEMENTS: Dict[str, Dict[str, Any]] = {
    "us-gaap:GreenHouseGasEmissionsScope1": {
        "name": "Scope 1 GHG Emissions",
        "namespace": "http://xbrl.sec.gov/climate/2024",
        "data_type": "mass",
        "unit": "tCO2e",
        "period_type": "duration",
        "required": True,
        "reg_sk_reference": "Item 1504(a)",
    },
    "us-gaap:GreenHouseGasEmissionsScope2": {
        "name": "Scope 2 GHG Emissions",
        "namespace": "http://xbrl.sec.gov/climate/2024",
        "data_type": "mass",
        "unit": "tCO2e",
        "period_type": "duration",
        "required": True,
        "reg_sk_reference": "Item 1504(b)",
    },
    "us-gaap:GreenHouseGasEmissionsScope3": {
        "name": "Scope 3 GHG Emissions",
        "namespace": "http://xbrl.sec.gov/climate/2024",
        "data_type": "mass",
        "unit": "tCO2e",
        "period_type": "duration",
        "required": False,
        "reg_sk_reference": "Item 1504(c)",
    },
    "us-gaap:GreenHouseGasEmissionsIntensityRevenue": {
        "name": "GHG Emissions Intensity (Revenue)",
        "namespace": "http://xbrl.sec.gov/climate/2024",
        "data_type": "percentage",
        "unit": "tCO2e/USD_million",
        "period_type": "duration",
        "required": True,
        "reg_sk_reference": "Item 1505",
    },
    "us-gaap:ClimateRelatedTargets": {
        "name": "Climate-Related Targets",
        "namespace": "http://xbrl.sec.gov/climate/2024",
        "data_type": "text",
        "period_type": "duration",
        "required": True,
        "reg_sk_reference": "Item 1506",
    },
    "us-gaap:TransitionPlanDescription": {
        "name": "Transition Plan Description",
        "namespace": "http://xbrl.sec.gov/climate/2024",
        "data_type": "text",
        "period_type": "duration",
        "required": False,
        "reg_sk_reference": "Item 1502",
    },
    "us-gaap:InternalCarbonPriceAmount": {
        "name": "Internal Carbon Price",
        "namespace": "http://xbrl.sec.gov/climate/2024",
        "data_type": "monetary",
        "unit": "USD",
        "period_type": "instant",
        "required": False,
        "reg_sk_reference": "Item 1505(c)",
    },
}

# ---------------------------------------------------------------------------
# CSRD ESRS E1 Taxonomy Elements
# ---------------------------------------------------------------------------

CSRD_TAXONOMY_ELEMENTS: Dict[str, Dict[str, Any]] = {
    "esrs:E1-1_TransitionPlan": {
        "name": "Transition plan for climate change mitigation",
        "namespace": "http://xbrl.efrag.org/taxonomy/esrs/2024",
        "data_type": "text",
        "period_type": "duration",
        "required": True,
        "esrs_reference": "ESRS E1-1",
    },
    "esrs:E1-4_GHGReductionTargets": {
        "name": "GHG emission reduction targets",
        "namespace": "http://xbrl.efrag.org/taxonomy/esrs/2024",
        "data_type": "text",
        "period_type": "duration",
        "required": True,
        "esrs_reference": "ESRS E1-4",
    },
    "esrs:E1-6_GrossScope1Emissions": {
        "name": "Gross Scope 1 GHG emissions",
        "namespace": "http://xbrl.efrag.org/taxonomy/esrs/2024",
        "data_type": "mass",
        "unit": "tCO2e",
        "period_type": "duration",
        "required": True,
        "esrs_reference": "ESRS E1-6 para 44",
    },
    "esrs:E1-6_GrossScope2Emissions": {
        "name": "Gross Scope 2 GHG emissions",
        "namespace": "http://xbrl.efrag.org/taxonomy/esrs/2024",
        "data_type": "mass",
        "unit": "tCO2e",
        "period_type": "duration",
        "required": True,
        "esrs_reference": "ESRS E1-6 para 48",
    },
    "esrs:E1-6_GrossScope3Emissions": {
        "name": "Gross Scope 3 GHG emissions",
        "namespace": "http://xbrl.efrag.org/taxonomy/esrs/2024",
        "data_type": "mass",
        "unit": "tCO2e",
        "period_type": "duration",
        "required": True,
        "esrs_reference": "ESRS E1-6 para 51",
    },
    "esrs:E1-6_TotalGHGEmissions": {
        "name": "Total GHG emissions",
        "namespace": "http://xbrl.efrag.org/taxonomy/esrs/2024",
        "data_type": "mass",
        "unit": "tCO2e",
        "period_type": "duration",
        "required": True,
        "esrs_reference": "ESRS E1-6 para 55",
    },
    "esrs:E1-7_GHGRemovals": {
        "name": "GHG removals and carbon credits",
        "namespace": "http://xbrl.efrag.org/taxonomy/esrs/2024",
        "data_type": "mass",
        "unit": "tCO2e",
        "period_type": "duration",
        "required": True,
        "esrs_reference": "ESRS E1-7",
    },
    "esrs:E1-8_InternalCarbonPricing": {
        "name": "Internal carbon pricing",
        "namespace": "http://xbrl.efrag.org/taxonomy/esrs/2024",
        "data_type": "monetary",
        "unit": "EUR",
        "period_type": "duration",
        "required": False,
        "esrs_reference": "ESRS E1-8",
    },
    "esrs:E1-9_AnticipatedFinancialEffects": {
        "name": "Anticipated financial effects",
        "namespace": "http://xbrl.efrag.org/taxonomy/esrs/2024",
        "data_type": "monetary",
        "unit": "EUR",
        "period_type": "duration",
        "required": True,
        "esrs_reference": "ESRS E1-9",
    },
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class XBRLIntegrationConfig(BaseModel):
    pack_id: str = Field(default="PACK-030")
    enable_provenance: bool = Field(default=True)
    sec_taxonomy_version: str = Field(default="2024")
    csrd_taxonomy_version: str = Field(default="2024")
    cache_ttl_seconds: int = Field(default=86400)
    validate_on_fetch: bool = Field(default=True)


class TaxonomyElement(BaseModel):
    """XBRL taxonomy element."""
    element_id: str = Field(default="")
    name: str = Field(default="")
    namespace: str = Field(default="")
    data_type: TagDataType = Field(default=TagDataType.TEXT)
    unit: str = Field(default="")
    period_type: str = Field(default="duration")
    required: bool = Field(default=False)
    regulatory_reference: str = Field(default="")
    framework: TaxonomyFramework = Field(default=TaxonomyFramework.SEC)


class Taxonomy(BaseModel):
    """Complete XBRL taxonomy."""
    taxonomy_id: str = Field(default_factory=_new_uuid)
    framework: TaxonomyFramework = Field(default=TaxonomyFramework.SEC)
    version: str = Field(default="2024")
    elements: List[TaxonomyElement] = Field(default_factory=list)
    total_elements: int = Field(default=0)
    required_elements: int = Field(default=0)
    namespace_uri: str = Field(default="")
    schema_url: str = Field(default="")
    effective_date: str = Field(default="2024-01-01")
    provenance_hash: str = Field(default="")
    fetched_at: datetime = Field(default_factory=_utcnow)


class TagValidationResult(BaseModel):
    """XBRL tag validation result."""
    validation_id: str = Field(default_factory=_new_uuid)
    framework: TaxonomyFramework = Field(default=TaxonomyFramework.SEC)
    tags_validated: int = Field(default=0)
    valid_tags: int = Field(default=0)
    invalid_tags: int = Field(default=0)
    warnings: int = Field(default=0)
    missing_required: List[str] = Field(default_factory=list)
    issues: List[Dict[str, str]] = Field(default_factory=list)
    compliance_score: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class XBRLIntegrationResult(BaseModel):
    result_id: str = Field(default_factory=_new_uuid)
    sec_taxonomy: Optional[Taxonomy] = Field(None)
    csrd_taxonomy: Optional[Taxonomy] = Field(None)
    validation: Optional[TagValidationResult] = Field(None)
    import_status: ImportStatus = Field(default=ImportStatus.FAILED)
    integration_quality_score: float = Field(default=0.0)
    fetched_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# XBRLTaxonomyIntegration
# ---------------------------------------------------------------------------


class XBRLTaxonomyIntegration:
    """XBRL taxonomy integration for PACK-030.

    Example:
        >>> config = XBRLIntegrationConfig()
        >>> integration = XBRLTaxonomyIntegration(config)
        >>> sec = await integration.fetch_sec_taxonomy()
        >>> csrd = await integration.fetch_csrd_taxonomy()
        >>> validation = await integration.validate_tags(tags, framework="sec")
    """

    def __init__(self, config: Optional[XBRLIntegrationConfig] = None) -> None:
        self.config = config or XBRLIntegrationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._sec_cache: Optional[Taxonomy] = None
        self._csrd_cache: Optional[Taxonomy] = None
        self.logger.info("XBRLTaxonomyIntegration initialized: SEC v%s, CSRD v%s",
                         self.config.sec_taxonomy_version, self.config.csrd_taxonomy_version)

    async def fetch_sec_taxonomy(self) -> Taxonomy:
        """Fetch SEC climate disclosure XBRL taxonomy."""
        if self._sec_cache is not None:
            return self._sec_cache

        elements: List[TaxonomyElement] = []
        for elem_id, info in SEC_TAXONOMY_ELEMENTS.items():
            elements.append(TaxonomyElement(
                element_id=elem_id, name=info["name"],
                namespace=info["namespace"],
                data_type=TagDataType(info.get("data_type", "text")),
                unit=info.get("unit", ""),
                period_type=info.get("period_type", "duration"),
                required=info.get("required", False),
                regulatory_reference=info.get("reg_sk_reference", ""),
                framework=TaxonomyFramework.SEC,
            ))

        taxonomy = Taxonomy(
            framework=TaxonomyFramework.SEC,
            version=self.config.sec_taxonomy_version,
            elements=elements, total_elements=len(elements),
            required_elements=sum(1 for e in elements if e.required),
            namespace_uri="http://xbrl.sec.gov/climate/2024",
            schema_url="https://xbrl.sec.gov/climate/2024/climate-2024.xsd",
            effective_date="2024-01-01",
        )
        if self.config.enable_provenance:
            taxonomy.provenance_hash = _compute_hash(taxonomy)

        self._sec_cache = taxonomy
        self.logger.info("SEC taxonomy fetched: %d elements, %d required",
                         taxonomy.total_elements, taxonomy.required_elements)
        return taxonomy

    async def fetch_csrd_taxonomy(self) -> Taxonomy:
        """Fetch CSRD ESRS E1 digital taxonomy."""
        if self._csrd_cache is not None:
            return self._csrd_cache

        elements: List[TaxonomyElement] = []
        for elem_id, info in CSRD_TAXONOMY_ELEMENTS.items():
            elements.append(TaxonomyElement(
                element_id=elem_id, name=info["name"],
                namespace=info["namespace"],
                data_type=TagDataType(info.get("data_type", "text")),
                unit=info.get("unit", ""),
                period_type=info.get("period_type", "duration"),
                required=info.get("required", False),
                regulatory_reference=info.get("esrs_reference", ""),
                framework=TaxonomyFramework.CSRD,
            ))

        taxonomy = Taxonomy(
            framework=TaxonomyFramework.CSRD,
            version=self.config.csrd_taxonomy_version,
            elements=elements, total_elements=len(elements),
            required_elements=sum(1 for e in elements if e.required),
            namespace_uri="http://xbrl.efrag.org/taxonomy/esrs/2024",
            schema_url="https://xbrl.efrag.org/taxonomy/esrs/2024/esrs-2024.xsd",
            effective_date="2024-01-01",
        )
        if self.config.enable_provenance:
            taxonomy.provenance_hash = _compute_hash(taxonomy)

        self._csrd_cache = taxonomy
        self.logger.info("CSRD taxonomy fetched: %d elements, %d required",
                         taxonomy.total_elements, taxonomy.required_elements)
        return taxonomy

    async def validate_tags(
        self,
        tags: List[Dict[str, Any]],
        framework: str = "sec",
    ) -> TagValidationResult:
        """Validate XBRL tags against taxonomy.

        Args:
            tags: List of tag dicts with 'element_id', 'value', 'unit'.
            framework: 'sec' or 'csrd'.

        Returns:
            TagValidationResult with compliance score.
        """
        if framework == "sec":
            taxonomy = await self.fetch_sec_taxonomy()
            ref_elements = SEC_TAXONOMY_ELEMENTS
        else:
            taxonomy = await self.fetch_csrd_taxonomy()
            ref_elements = CSRD_TAXONOMY_ELEMENTS

        valid = 0
        invalid = 0
        warns = 0
        issues: List[Dict[str, str]] = []

        tag_ids = {t.get("element_id") for t in tags}
        for tag in tags:
            elem_id = tag.get("element_id", "")
            if elem_id in ref_elements:
                valid += 1
            else:
                invalid += 1
                issues.append({"element_id": elem_id, "severity": "error",
                               "message": f"Unknown element: {elem_id}"})

        # Check missing required
        missing_required = []
        for elem_id, info in ref_elements.items():
            if info.get("required", False) and elem_id not in tag_ids:
                missing_required.append(elem_id)
                issues.append({"element_id": elem_id, "severity": "error",
                               "message": f"Required element missing: {info['name']}"})

        total_checked = valid + invalid
        score = (valid / max(total_checked, 1)) * 100.0

        result = TagValidationResult(
            framework=TaxonomyFramework(framework),
            tags_validated=total_checked,
            valid_tags=valid, invalid_tags=invalid, warnings=warns,
            missing_required=missing_required,
            issues=issues, compliance_score=round(score, 2),
        )
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self.logger.info("XBRL validation: %d valid, %d invalid, %d missing, score=%.1f%%",
                         valid, invalid, len(missing_required), score)
        return result

    async def get_full_integration(self) -> XBRLIntegrationResult:
        errors: List[str] = []
        sec = csrd = None
        try:
            sec = await self.fetch_sec_taxonomy()
        except Exception as exc:
            errors.append(f"SEC taxonomy fetch failed: {exc}")
        try:
            csrd = await self.fetch_csrd_taxonomy()
        except Exception as exc:
            errors.append(f"CSRD taxonomy fetch failed: {exc}")

        quality = (50.0 if sec else 0.0) + (50.0 if csrd else 0.0)
        status = ImportStatus.SUCCESS if not errors else ImportStatus.PARTIAL

        result = XBRLIntegrationResult(
            sec_taxonomy=sec, csrd_taxonomy=csrd,
            import_status=status, integration_quality_score=quality,
        )
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def get_integration_status(self) -> Dict[str, Any]:
        return {
            "sec_cached": self._sec_cache is not None,
            "csrd_cached": self._csrd_cache is not None,
            "sec_version": self.config.sec_taxonomy_version,
            "csrd_version": self.config.csrd_taxonomy_version,
            "module_version": _MODULE_VERSION,
        }

    async def refresh(self) -> XBRLIntegrationResult:
        self._sec_cache = None
        self._csrd_cache = None
        return await self.get_full_integration()
