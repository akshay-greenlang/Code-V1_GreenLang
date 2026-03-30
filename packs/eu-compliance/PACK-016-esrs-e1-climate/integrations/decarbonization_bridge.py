# -*- coding: utf-8 -*-
"""
DecarbonizationBridge - GL-DECARB Agent Integration Bridge for PACK-016
=========================================================================

Connects PACK-016 to decarbonization planning agents for transition plan
data, abatement options, and pathway scenario imports.

Methods:
    - import_transition_plan()      -- Import transition plan from decarb agents
    - import_abatement_options()    -- Import marginal abatement curve data
    - import_pathway_scenarios()    -- Import decarbonization pathway scenarios

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
# Data Models
# ---------------------------------------------------------------------------

class DecarbBridgeConfig(BaseModel):
    """Configuration for the Decarbonization Bridge."""

    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    enable_provenance: bool = Field(default=True)
    scenario_provider: str = Field(default="IEA")
    sector_pathway: str = Field(default="")

class AbatementOption(BaseModel):
    """Marginal abatement curve option."""

    name: str = Field(default="")
    category: str = Field(default="")
    abatement_tco2e: float = Field(default=0.0)
    cost_per_tco2e_eur: float = Field(default=0.0)
    total_cost_eur: float = Field(default=0.0)
    implementation_year: int = Field(default=0)
    readiness_level: str = Field(default="")
    payback_years: float = Field(default=0.0)

class PathwayScenario(BaseModel):
    """Decarbonization pathway scenario."""

    name: str = Field(default="")
    provider: str = Field(default="")
    temperature_target: str = Field(default="")
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    annual_reduction_rate_pct: float = Field(default=0.0)
    sector_specific: bool = Field(default=False)

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
# DecarbonizationBridge
# ---------------------------------------------------------------------------

class DecarbonizationBridge:
    """Decarbonization agent integration bridge for PACK-016.

    Provides data flow from decarbonization planning agents into the
    E1 Climate Pack for transition plan assessment, abatement curve
    analysis, and pathway scenario modeling.

    Attributes:
        config: Bridge configuration.

    Example:
        >>> bridge = DecarbonizationBridge(DecarbBridgeConfig(reporting_year=2025))
        >>> result = bridge.import_transition_plan(context)
        >>> assert result.status == "completed"
    """

    def __init__(self, config: Optional[DecarbBridgeConfig] = None) -> None:
        """Initialize DecarbonizationBridge."""
        self.config = config or DecarbBridgeConfig()
        logger.info(
            "DecarbonizationBridge initialized (year=%d, provider=%s)",
            self.config.reporting_year,
            self.config.scenario_provider,
        )

    def import_transition_plan(
        self,
        context: Dict[str, Any],
    ) -> BridgeResult:
        """Import transition plan from decarbonization agents.

        Args:
            context: Pipeline context with transition plan data.

        Returns:
            BridgeResult with transition plan import status.
        """
        result = BridgeResult(started_at=utcnow())

        try:
            plan = context.get("transition_plan", {})
            levers = context.get("decarbonization_levers", [])

            import_data = {
                "plan": plan,
                "levers": levers,
                "total_abatement_tco2e": sum(
                    lv.get("abatement_tco2e", 0.0) for lv in levers
                ),
            }

            context["transition_plan_import"] = import_data

            result.records_transferred = 1 + len(levers)
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(import_data)

            logger.info(
                "Transition plan imported with %d levers", len(levers)
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Transition plan import failed: %s", str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def import_abatement_options(
        self,
        context: Dict[str, Any],
    ) -> BridgeResult:
        """Import marginal abatement curve options.

        Args:
            context: Pipeline context with abatement data.

        Returns:
            BridgeResult with abatement options import status.
        """
        result = BridgeResult(started_at=utcnow())

        try:
            raw_options = context.get("abatement_options", [])
            options = [
                AbatementOption(
                    name=opt.get("name", ""),
                    category=opt.get("category", ""),
                    abatement_tco2e=opt.get("abatement_tco2e", 0.0),
                    cost_per_tco2e_eur=opt.get("cost_per_tco2e_eur", 0.0),
                    total_cost_eur=opt.get("total_cost_eur", 0.0),
                    implementation_year=opt.get("implementation_year", 0),
                    readiness_level=opt.get("readiness_level", ""),
                    payback_years=opt.get("payback_years", 0.0),
                )
                for opt in raw_options
            ]

            # Sort by cost-effectiveness (MAC curve order)
            options.sort(key=lambda o: o.cost_per_tco2e_eur)

            context["abatement_options_parsed"] = [
                o.model_dump() for o in options
            ]

            result.records_transferred = len(options)
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(
                    [o.model_dump() for o in options]
                )

            logger.info("Imported %d abatement options", len(options))

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Abatement options import failed: %s", str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def import_pathway_scenarios(
        self,
        context: Dict[str, Any],
    ) -> BridgeResult:
        """Import decarbonization pathway scenarios.

        Args:
            context: Pipeline context with scenario data.

        Returns:
            BridgeResult with pathway scenario import status.
        """
        result = BridgeResult(started_at=utcnow())

        try:
            raw_scenarios = context.get("pathway_scenarios", [])
            scenarios = [
                PathwayScenario(
                    name=s.get("name", ""),
                    provider=s.get("provider", self.config.scenario_provider),
                    temperature_target=s.get("temperature_target", ""),
                    milestones=s.get("milestones", []),
                    annual_reduction_rate_pct=s.get("annual_reduction_rate_pct", 0.0),
                    sector_specific=s.get("sector_specific", False),
                )
                for s in raw_scenarios
            ]

            context["pathway_scenarios_parsed"] = [
                s.model_dump() for s in scenarios
            ]

            result.records_transferred = len(scenarios)
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(
                    [s.model_dump() for s in scenarios]
                )

            logger.info("Imported %d pathway scenarios", len(scenarios))

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Pathway scenario import failed: %s", str(exc))

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
            "scenario_provider": self.config.scenario_provider,
            "sector_pathway": self.config.sector_pathway,
            "reporting_year": self.config.reporting_year,
        }
