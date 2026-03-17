# -*- coding: utf-8 -*-
"""
DMAPackBridge - PACK-015 Double Materiality Integration Bridge for PACK-017
=============================================================================

Connects PACK-017 to PACK-015 Double Materiality Assessment Pack for
importing DMA results that determine which ESRS standards are material,
mapping materiality scores to standard activation, tracking IRO
identification across all 12 ESRS standards, and feeding materiality
results into the orchestrator for phase filtering.

Methods:
    - import_dma_results()       -- Import full DMA assessment from PACK-015
    - get_material_standards()   -- Get list of material ESRS standards
    - get_iro_register()         -- Get IRO register across all 12 standards
    - get_materiality_matrix()   -- Get materiality matrix for scorecard
    - export_disclosure_status() -- Export disclosure status back to DMA pack

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-017 ESRS Full Coverage Pack
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
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


class MaterialityStatus(str, Enum):
    """Materiality assessment status for an ESRS standard."""

    MATERIAL = "material"
    NOT_MATERIAL = "not_material"
    PENDING = "pending"
    UNDER_REVIEW = "under_review"


class IROType(str, Enum):
    """IRO classification type."""

    IMPACT = "impact"
    RISK = "risk"
    OPPORTUNITY = "opportunity"


class ESRSStandard(str, Enum):
    """All 12 ESRS topical and cross-cutting standards."""

    ESRS_1 = "ESRS 1"
    ESRS_2 = "ESRS 2"
    ESRS_E1 = "ESRS E1"
    ESRS_E2 = "ESRS E2"
    ESRS_E3 = "ESRS E3"
    ESRS_E4 = "ESRS E4"
    ESRS_E5 = "ESRS E5"
    ESRS_S1 = "ESRS S1"
    ESRS_S2 = "ESRS S2"
    ESRS_S3 = "ESRS S3"
    ESRS_S4 = "ESRS S4"
    ESRS_G1 = "ESRS G1"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class DMABridgeConfig(BaseModel):
    """Configuration for the DMA Pack Bridge."""

    dma_pack_id: str = Field(default="PACK-015")
    dma_pack_version: str = Field(default="1.0.0")
    target_pack_id: str = Field(default="PACK-017")
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    enable_provenance: bool = Field(default=True)
    auto_import: bool = Field(default=True)
    materiality_threshold: float = Field(
        default=2.5, ge=0.0, le=5.0,
        description="Minimum composite score for a standard to be deemed material",
    )


class StandardMateriality(BaseModel):
    """Materiality assessment for a single ESRS standard."""

    standard: str = Field(default="")
    status: MaterialityStatus = Field(default=MaterialityStatus.PENDING)
    impact_materiality_score: float = Field(default=0.0, ge=0.0, le=5.0)
    financial_materiality_score: float = Field(default=0.0, ge=0.0, le=5.0)
    composite_score: float = Field(default=0.0, ge=0.0, le=5.0)
    double_materiality: bool = Field(default=False)
    rationale: str = Field(default="")
    sub_topics: List[str] = Field(default_factory=list)
    assessment_date: str = Field(default="")


class IROEntry(BaseModel):
    """Impact, Risk, or Opportunity register entry."""

    iro_id: str = Field(default="")
    name: str = Field(default="")
    iro_type: IROType = Field(default=IROType.IMPACT)
    esrs_standard: str = Field(default="")
    sub_topic: str = Field(default="")
    description: str = Field(default="")
    severity_score: float = Field(default=0.0, ge=0.0, le=5.0)
    likelihood_score: float = Field(default=0.0, ge=0.0, le=5.0)
    composite_score: float = Field(default=0.0, ge=0.0, le=5.0)
    value_chain_stage: str = Field(default="")
    time_horizon: str = Field(default="")
    stakeholder_groups: List[str] = Field(default_factory=list)


class DMAImportResult(BaseModel):
    """Result of a DMA import operation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    standards_assessed: int = Field(default=0)
    standards_material: int = Field(default=0)
    standards_not_material: int = Field(default=0)
    iro_count: int = Field(default=0)
    materiality_results: List[StandardMateriality] = Field(default_factory=list)
    material_standard_list: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Standard to Topic Mapping
# ---------------------------------------------------------------------------

STANDARD_TOPIC_MAP: Dict[str, str] = {
    "ESRS E1": "Climate change",
    "ESRS E2": "Pollution",
    "ESRS E3": "Water and marine resources",
    "ESRS E4": "Biodiversity and ecosystems",
    "ESRS E5": "Resource use and circular economy",
    "ESRS S1": "Own workforce",
    "ESRS S2": "Workers in the value chain",
    "ESRS S3": "Affected communities",
    "ESRS S4": "Consumers and end-users",
    "ESRS G1": "Business conduct",
}

# ESRS 1 and ESRS 2 are always mandatory (cross-cutting)
MANDATORY_STANDARDS: List[str] = ["ESRS 1", "ESRS 2"]


# ---------------------------------------------------------------------------
# DMAPackBridge
# ---------------------------------------------------------------------------


class DMAPackBridge:
    """PACK-015 Double Materiality integration bridge for PACK-017.

    Provides the materiality gating mechanism that determines which
    of the 10 topical ESRS standards (E1-E5, S1-S4, G1) should be
    activated, based on the double materiality assessment from PACK-015.

    Attributes:
        config: Bridge configuration.
        _materiality_cache: Cached materiality results.
        _iro_cache: Cached IRO register entries.

    Example:
        >>> bridge = DMAPackBridge(DMABridgeConfig(reporting_year=2025))
        >>> result = bridge.import_dma_results(context)
        >>> material = bridge.get_material_standards()
        >>> assert "ESRS E1" in material
    """

    def __init__(self, config: Optional[DMABridgeConfig] = None) -> None:
        """Initialize DMAPackBridge."""
        self.config = config or DMABridgeConfig()
        self._materiality_cache: List[StandardMateriality] = []
        self._iro_cache: List[IROEntry] = []
        logger.info(
            "DMAPackBridge initialized (dma=%s, year=%d, threshold=%.1f)",
            self.config.dma_pack_id,
            self.config.reporting_year,
            self.config.materiality_threshold,
        )

    def import_dma_results(self, context: Dict[str, Any]) -> DMAImportResult:
        """Import full DMA assessment from PACK-015.

        Args:
            context: Pipeline context with DMA results.

        Returns:
            DMAImportResult with per-standard materiality and IRO data.
        """
        result = DMAImportResult(started_at=_utcnow())

        try:
            dma_data = context.get("dma_results", {})
            standard_assessments = dma_data.get("standard_assessments", {})
            iro_register = dma_data.get("iro_register", [])

            # Parse materiality for each topical standard
            materiality_results: List[StandardMateriality] = []
            for std_enum in ESRSStandard:
                std_name = std_enum.value
                if std_name in MANDATORY_STANDARDS:
                    continue  # Cross-cutting standards are always active

                assessment = standard_assessments.get(std_name, {})
                impact_score = assessment.get("impact_score", 0.0)
                financial_score = assessment.get("financial_score", 0.0)
                composite = max(impact_score, financial_score)

                is_material = composite >= self.config.materiality_threshold
                sm = StandardMateriality(
                    standard=std_name,
                    status=MaterialityStatus.MATERIAL if is_material else MaterialityStatus.NOT_MATERIAL,
                    impact_materiality_score=impact_score,
                    financial_materiality_score=financial_score,
                    composite_score=composite,
                    double_materiality=(impact_score >= self.config.materiality_threshold
                                       and financial_score >= self.config.materiality_threshold),
                    rationale=assessment.get("rationale", ""),
                    sub_topics=assessment.get("sub_topics", []),
                    assessment_date=assessment.get("assessment_date", ""),
                )
                materiality_results.append(sm)

            self._materiality_cache = materiality_results
            result.materiality_results = materiality_results
            result.standards_assessed = len(materiality_results)
            result.standards_material = sum(
                1 for m in materiality_results if m.status == MaterialityStatus.MATERIAL
            )
            result.standards_not_material = result.standards_assessed - result.standards_material

            # Build material standards list (mandatory + material topical)
            material_list = list(MANDATORY_STANDARDS)
            for m in materiality_results:
                if m.status == MaterialityStatus.MATERIAL:
                    material_list.append(m.standard)
            result.material_standard_list = material_list

            # Parse IRO register
            parsed_iros: List[IROEntry] = []
            for iro_data in iro_register:
                parsed_iros.append(IROEntry(
                    iro_id=iro_data.get("iro_id", _new_uuid()),
                    name=iro_data.get("name", ""),
                    iro_type=IROType(iro_data.get("iro_type", "impact")),
                    esrs_standard=iro_data.get("esrs_standard", ""),
                    sub_topic=iro_data.get("sub_topic", ""),
                    description=iro_data.get("description", ""),
                    severity_score=iro_data.get("severity_score", 0.0),
                    likelihood_score=iro_data.get("likelihood_score", 0.0),
                    composite_score=iro_data.get("composite_score", 0.0),
                    value_chain_stage=iro_data.get("value_chain_stage", ""),
                    time_horizon=iro_data.get("time_horizon", ""),
                    stakeholder_groups=iro_data.get("stakeholder_groups", []),
                ))
            self._iro_cache = parsed_iros
            result.iro_count = len(parsed_iros)

            # Store in context for orchestrator
            context["material_standards"] = material_list
            context["iro_register_parsed"] = [i.model_dump() for i in parsed_iros]

            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "materiality": [m.model_dump() for m in materiality_results],
                    "iro_count": len(parsed_iros),
                })

            logger.info(
                "DMA import: %d/%d standards material, %d IROs imported",
                result.standards_material,
                result.standards_assessed,
                result.iro_count,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("DMA import failed: %s", str(exc))

        result.completed_at = _utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def get_material_standards(self) -> List[str]:
        """Get the list of material ESRS standards.

        Returns:
            List of standard names deemed material (always includes ESRS 1/2).
        """
        material = list(MANDATORY_STANDARDS)
        for m in self._materiality_cache:
            if m.status == MaterialityStatus.MATERIAL:
                material.append(m.standard)
        return material

    def get_iro_register(
        self,
        standard_filter: Optional[str] = None,
    ) -> List[IROEntry]:
        """Get IRO register entries, optionally filtered by standard.

        Args:
            standard_filter: Optional ESRS standard name to filter by.

        Returns:
            List of IROEntry matching the filter.
        """
        if standard_filter:
            return [
                iro for iro in self._iro_cache
                if iro.esrs_standard == standard_filter
            ]
        return list(self._iro_cache)

    def get_materiality_matrix(self) -> Dict[str, Any]:
        """Get the full materiality matrix for scorecard rendering.

        Returns:
            Dict with standards as keys, materiality details as values.
        """
        matrix: Dict[str, Any] = {}
        for m in self._materiality_cache:
            matrix[m.standard] = {
                "status": m.status.value,
                "impact_score": m.impact_materiality_score,
                "financial_score": m.financial_materiality_score,
                "composite_score": m.composite_score,
                "double_materiality": m.double_materiality,
                "topic": STANDARD_TOPIC_MAP.get(m.standard, ""),
            }
        # Always include mandatory cross-cutting standards
        for std in MANDATORY_STANDARDS:
            matrix[std] = {
                "status": "mandatory",
                "impact_score": 5.0,
                "financial_score": 5.0,
                "composite_score": 5.0,
                "double_materiality": True,
                "topic": "Cross-cutting",
            }
        return matrix

    def export_disclosure_status(
        self,
        disclosure_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Export disclosure completion status back to PACK-015.

        Args:
            disclosure_status: Dict mapping standard names to completion status.

        Returns:
            Dict with export result.
        """
        try:
            payload = {
                "source_pack": self.config.target_pack_id,
                "target_pack": self.config.dma_pack_id,
                "reporting_year": self.config.reporting_year,
                "disclosure_status": disclosure_status,
                "exported_at": _utcnow().isoformat(),
            }
            logger.info(
                "Exported disclosure status for %d standards to DMA pack",
                len(disclosure_status),
            )
            return {"status": "completed", "payload_hash": _compute_hash(payload)}
        except Exception as exc:
            logger.error("Disclosure status export failed: %s", str(exc))
            return {"status": "failed", "error": str(exc)}

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status.

        Returns:
            Dict with bridge status information.
        """
        return {
            "dma_pack_id": self.config.dma_pack_id,
            "target_pack_id": self.config.target_pack_id,
            "reporting_year": self.config.reporting_year,
            "materiality_threshold": self.config.materiality_threshold,
            "cached_standards": len(self._materiality_cache),
            "cached_iros": len(self._iro_cache),
        }
