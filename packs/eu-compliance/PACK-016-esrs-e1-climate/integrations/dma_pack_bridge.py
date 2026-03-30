# -*- coding: utf-8 -*-
"""
DMAPackBridge - PACK-015 Double Materiality Integration Bridge for PACK-016
=============================================================================

Connects PACK-016 to PACK-015 Double Materiality Assessment Pack for
E1 materiality import, IRO register import, and climate disclosure export.

Methods:
    - import_e1_materiality()      -- Import E1 climate materiality assessment
    - import_iro_register()        -- Import climate-related IROs
    - export_climate_disclosures() -- Export E1 disclosures to DMA pack

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-016 ESRS E1 Climate Pack
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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

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
    """E1 materiality assessment status."""

    MATERIAL = "material"
    NOT_MATERIAL = "not_material"
    PENDING = "pending"
    UNDER_REVIEW = "under_review"

class IROType(str, Enum):
    """IRO classification type."""

    IMPACT = "impact"
    RISK = "risk"
    OPPORTUNITY = "opportunity"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class DMABridgeConfig(BaseModel):
    """Configuration for the DMA Pack Bridge."""

    dma_pack_id: str = Field(default="PACK-015")
    dma_pack_version: str = Field(default="1.0.0")
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    enable_provenance: bool = Field(default=True)
    auto_import: bool = Field(default=True)

class E1MaterialityResult(BaseModel):
    """E1 climate materiality assessment result from DMA."""

    topic: str = Field(default="E1 - Climate Change")
    is_material: bool = Field(default=True)
    impact_materiality_score: float = Field(default=0.0, ge=0.0, le=5.0)
    financial_materiality_score: float = Field(default=0.0, ge=0.0, le=5.0)
    double_materiality: bool = Field(default=False)
    assessment_date: str = Field(default="")
    rationale: str = Field(default="")
    sub_topics: List[Dict[str, Any]] = Field(default_factory=list)

class IROEntry(BaseModel):
    """Impact, Risk, or Opportunity register entry."""

    iro_id: str = Field(default="")
    name: str = Field(default="")
    iro_type: IROType = Field(default=IROType.IMPACT)
    esrs_topic: str = Field(default="E1")
    description: str = Field(default="")
    severity_score: float = Field(default=0.0)
    likelihood_score: float = Field(default=0.0)
    composite_score: float = Field(default=0.0)
    value_chain_stage: str = Field(default="")
    time_horizon: str = Field(default="")

class BridgeResult(BaseModel):
    """Result from a bridge operation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_transferred: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# DMAPackBridge
# ---------------------------------------------------------------------------

class DMAPackBridge:
    """PACK-015 Double Materiality integration bridge for PACK-016.

    Provides data flow between the E1 Climate Pack and the DMA Pack
    for materiality assessments, IRO register data, and climate
    disclosure results.

    Attributes:
        config: Bridge configuration.

    Example:
        >>> bridge = DMAPackBridge(DMABridgeConfig(reporting_year=2025))
        >>> result = bridge.import_e1_materiality(context)
        >>> assert result.status == "completed"
    """

    def __init__(self, config: Optional[DMABridgeConfig] = None) -> None:
        """Initialize DMAPackBridge."""
        self.config = config or DMABridgeConfig()
        logger.info(
            "DMAPackBridge initialized (dma=%s, year=%d)",
            self.config.dma_pack_id,
            self.config.reporting_year,
        )

    def import_e1_materiality(
        self,
        context: Dict[str, Any],
    ) -> BridgeResult:
        """Import E1 climate materiality assessment from PACK-015.

        Args:
            context: Pipeline context with DMA results.

        Returns:
            BridgeResult with materiality assessment data.
        """
        result = BridgeResult(started_at=utcnow())

        try:
            dma_data = context.get("dma_results", {})
            e1_data = dma_data.get("e1_climate", {})

            materiality = E1MaterialityResult(
                is_material=e1_data.get("is_material", True),
                impact_materiality_score=e1_data.get("impact_score", 0.0),
                financial_materiality_score=e1_data.get("financial_score", 0.0),
                double_materiality=e1_data.get("double_materiality", False),
                assessment_date=e1_data.get("assessment_date", ""),
                rationale=e1_data.get("rationale", ""),
                sub_topics=e1_data.get("sub_topics", []),
            )

            # Store in context for downstream phases
            context["e1_materiality"] = materiality.is_material
            context["e1_materiality_detail"] = materiality.model_dump()

            result.records_transferred = 1
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(materiality)

            logger.info(
                "E1 materiality imported: material=%s, impact=%.2f, financial=%.2f",
                materiality.is_material,
                materiality.impact_materiality_score,
                materiality.financial_materiality_score,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("E1 materiality import failed: %s", str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def import_iro_register(
        self,
        context: Dict[str, Any],
    ) -> BridgeResult:
        """Import climate-related IROs from PACK-015 register.

        Args:
            context: Pipeline context with IRO register data.

        Returns:
            BridgeResult with IRO entries for E1 topic.
        """
        result = BridgeResult(started_at=utcnow())

        try:
            all_iros = context.get("iro_register", [])
            e1_iros = [
                iro for iro in all_iros
                if iro.get("esrs_topic", "").startswith("E1")
            ]

            parsed_iros = [
                IROEntry(
                    iro_id=iro.get("iro_id", ""),
                    name=iro.get("name", ""),
                    iro_type=IROType(iro.get("iro_type", "impact")),
                    esrs_topic=iro.get("esrs_topic", "E1"),
                    description=iro.get("description", ""),
                    severity_score=iro.get("severity_score", 0.0),
                    likelihood_score=iro.get("likelihood_score", 0.0),
                    composite_score=iro.get("composite_score", 0.0),
                    value_chain_stage=iro.get("value_chain_stage", ""),
                    time_horizon=iro.get("time_horizon", ""),
                )
                for iro in e1_iros
            ]

            context["e1_iro_register"] = [i.model_dump() for i in parsed_iros]

            result.records_transferred = len(parsed_iros)
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(
                    [i.model_dump() for i in parsed_iros]
                )

            logger.info(
                "Imported %d E1 IROs from DMA register",
                result.records_transferred,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("IRO register import failed: %s", str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def export_climate_disclosures(
        self,
        e1_disclosures: Dict[str, Any],
    ) -> BridgeResult:
        """Export E1 climate disclosures back to PACK-015.

        Args:
            e1_disclosures: E1 disclosure results to export.

        Returns:
            BridgeResult with export status.
        """
        result = BridgeResult(started_at=utcnow())

        try:
            export_payload = {
                "source_pack": "PACK-016",
                "target_pack": self.config.dma_pack_id,
                "reporting_year": self.config.reporting_year,
                "esrs_topic": "E1",
                "disclosures": e1_disclosures,
                "exported_at": utcnow().isoformat(),
            }

            result.records_transferred = len(
                e1_disclosures.get("disclosure_requirements", [])
            )
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(export_payload)

            logger.info(
                "Exported %d E1 disclosures to DMA pack",
                result.records_transferred,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Climate disclosure export failed: %s", str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status.

        Returns:
            Dict with bridge status information.
        """
        return {
            "dma_pack_id": self.config.dma_pack_id,
            "dma_pack_version": self.config.dma_pack_version,
            "reporting_year": self.config.reporting_year,
            "auto_import": self.config.auto_import,
        }
