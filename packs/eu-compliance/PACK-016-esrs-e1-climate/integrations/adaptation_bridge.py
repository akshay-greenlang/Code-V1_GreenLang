# -*- coding: utf-8 -*-
"""
AdaptationBridge - GL-ADAPT Agent Integration Bridge for PACK-016
===================================================================

Connects PACK-016 to climate adaptation agents for physical risk data,
climate scenario imports, and resilience score calculations.

Methods:
    - import_physical_risks()     -- Import physical risk assessments
    - import_scenarios()          -- Import climate scenarios (RCP/SSP)
    - import_resilience_scores()  -- Import resilience and adaptation scores

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

class HazardCategory(str, Enum):
    """Climate hazard categories per ESRS."""

    ACUTE_TEMPERATURE = "acute_temperature"
    ACUTE_WIND = "acute_wind"
    ACUTE_WATER = "acute_water"
    ACUTE_MASS_MOVEMENT = "acute_mass_movement"
    CHRONIC_TEMPERATURE = "chronic_temperature"
    CHRONIC_WATER = "chronic_water"
    CHRONIC_WIND = "chronic_wind"
    CHRONIC_SOLID_MASS = "chronic_solid_mass"

class ScenarioFramework(str, Enum):
    """Climate scenario frameworks."""

    RCP_2_6 = "RCP2.6"
    RCP_4_5 = "RCP4.5"
    RCP_8_5 = "RCP8.5"
    SSP1_2_6 = "SSP1-2.6"
    SSP2_4_5 = "SSP2-4.5"
    SSP5_8_5 = "SSP5-8.5"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class AdaptBridgeConfig(BaseModel):
    """Configuration for the Adaptation Bridge."""

    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    enable_provenance: bool = Field(default=True)
    default_scenario: str = Field(default="SSP2-4.5")
    time_horizons: List[str] = Field(
        default_factory=lambda: ["2030", "2050", "2100"]
    )

class PhysicalRisk(BaseModel):
    """Physical climate risk assessment entry."""

    hazard: str = Field(default="")
    category: str = Field(default="")
    location: str = Field(default="")
    likelihood: str = Field(default="")
    severity: str = Field(default="")
    exposure_eur: float = Field(default=0.0)
    affected_assets: List[str] = Field(default_factory=list)
    time_horizon: str = Field(default="")
    adaptation_measures: List[str] = Field(default_factory=list)

class ClimateScenario(BaseModel):
    """Climate scenario data."""

    name: str = Field(default="")
    framework: str = Field(default="")
    temperature_outcome: str = Field(default="")
    time_horizon: str = Field(default="")
    physical_risk_factors: Dict[str, float] = Field(default_factory=dict)
    transition_risk_factors: Dict[str, float] = Field(default_factory=dict)

class ResilienceScore(BaseModel):
    """Climate resilience assessment score."""

    location: str = Field(default="")
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    physical_resilience: float = Field(default=0.0, ge=0.0, le=100.0)
    adaptive_capacity: float = Field(default=0.0, ge=0.0, le=100.0)
    vulnerability_index: float = Field(default=0.0, ge=0.0, le=100.0)
    adaptation_gap: float = Field(default=0.0, ge=0.0, le=100.0)

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
# AdaptationBridge
# ---------------------------------------------------------------------------

class AdaptationBridge:
    """Climate adaptation agent integration bridge for PACK-016.

    Provides data flow from climate adaptation and resilience agents
    into the E1 Climate Pack for physical risk assessment, climate
    scenario modeling, and resilience scoring.

    Attributes:
        config: Bridge configuration.

    Example:
        >>> bridge = AdaptationBridge(AdaptBridgeConfig(reporting_year=2025))
        >>> result = bridge.import_physical_risks(context)
        >>> assert result.status == "completed"
    """

    def __init__(self, config: Optional[AdaptBridgeConfig] = None) -> None:
        """Initialize AdaptationBridge."""
        self.config = config or AdaptBridgeConfig()
        logger.info(
            "AdaptationBridge initialized (year=%d, scenario=%s)",
            self.config.reporting_year,
            self.config.default_scenario,
        )

    def import_physical_risks(
        self,
        context: Dict[str, Any],
    ) -> BridgeResult:
        """Import physical risk assessments from adaptation agents.

        Args:
            context: Pipeline context with physical risk data.

        Returns:
            BridgeResult with physical risk import status.
        """
        result = BridgeResult(started_at=utcnow())

        try:
            raw_risks = context.get("physical_risks", [])
            risks = [
                PhysicalRisk(
                    hazard=r.get("hazard", ""),
                    category=r.get("category", ""),
                    location=r.get("location", ""),
                    likelihood=r.get("likelihood", ""),
                    severity=r.get("severity", ""),
                    exposure_eur=r.get("exposure_eur", 0.0),
                    affected_assets=r.get("affected_assets", []),
                    time_horizon=r.get("time_horizon", ""),
                    adaptation_measures=r.get("adaptation_measures", []),
                )
                for r in raw_risks
            ]

            context["physical_risks_parsed"] = [
                r.model_dump() for r in risks
            ]

            result.records_transferred = len(risks)
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(
                    [r.model_dump() for r in risks]
                )

            total_exposure = sum(r.exposure_eur for r in risks)
            logger.info(
                "Imported %d physical risks (exposure: EUR %.2f)",
                len(risks),
                total_exposure,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Physical risk import failed: %s", str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def import_scenarios(
        self,
        context: Dict[str, Any],
    ) -> BridgeResult:
        """Import climate scenarios (RCP/SSP frameworks).

        Args:
            context: Pipeline context with climate scenario data.

        Returns:
            BridgeResult with scenario import status.
        """
        result = BridgeResult(started_at=utcnow())

        try:
            raw_scenarios = context.get("climate_scenarios", [])
            scenarios = [
                ClimateScenario(
                    name=s.get("name", ""),
                    framework=s.get("framework", ""),
                    temperature_outcome=s.get("temperature_outcome", ""),
                    time_horizon=s.get("time_horizon", ""),
                    physical_risk_factors=s.get("physical_risk_factors", {}),
                    transition_risk_factors=s.get("transition_risk_factors", {}),
                )
                for s in raw_scenarios
            ]

            context["climate_scenarios_parsed"] = [
                s.model_dump() for s in scenarios
            ]

            result.records_transferred = len(scenarios)
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(
                    [s.model_dump() for s in scenarios]
                )

            logger.info("Imported %d climate scenarios", len(scenarios))

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Scenario import failed: %s", str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def import_resilience_scores(
        self,
        context: Dict[str, Any],
    ) -> BridgeResult:
        """Import resilience and adaptation scores.

        Args:
            context: Pipeline context with resilience score data.

        Returns:
            BridgeResult with resilience score import status.
        """
        result = BridgeResult(started_at=utcnow())

        try:
            raw_scores = context.get("resilience_scores", [])
            scores = [
                ResilienceScore(
                    location=s.get("location", ""),
                    overall_score=s.get("overall_score", 0.0),
                    physical_resilience=s.get("physical_resilience", 0.0),
                    adaptive_capacity=s.get("adaptive_capacity", 0.0),
                    vulnerability_index=s.get("vulnerability_index", 0.0),
                    adaptation_gap=s.get("adaptation_gap", 0.0),
                )
                for s in raw_scores
            ]

            context["resilience_scores_parsed"] = [
                s.model_dump() for s in scores
            ]

            result.records_transferred = len(scores)
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(
                    [s.model_dump() for s in scores]
                )

            avg_score = (
                sum(s.overall_score for s in scores) / len(scores)
                if scores
                else 0.0
            )
            logger.info(
                "Imported %d resilience scores (avg: %.1f)",
                len(scores),
                avg_score,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Resilience score import failed: %s", str(exc))

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
            "default_scenario": self.config.default_scenario,
            "time_horizons": self.config.time_horizons,
            "reporting_year": self.config.reporting_year,
        }
