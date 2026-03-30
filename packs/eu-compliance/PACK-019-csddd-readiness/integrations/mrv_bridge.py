# -*- coding: utf-8 -*-
"""
MRVBridge - AGENT-MRV Emission Data Bridge for PACK-019 CSDDD Climate Transition
===================================================================================

This module routes AGENT-MRV emission data to substantiate the climate transition
plan required by CSDDD Article 22. It provides Scope 1/2/3 emission baselines,
validates reduction targets against measured emission data, and calculates
year-over-year reduction progress for Paris Agreement alignment verification.

Legal References:
    - Directive (EU) 2024/1760 (CSDDD), Article 22 - Climate transition plan
    - Paris Agreement Art 2.1(a) - 1.5C temperature goal
    - GHG Protocol Corporate Standard - Scope 1/2/3 classifications
    - Science Based Targets initiative (SBTi) - target validation methodology

MRV Agent Routing:
    Scope 1: MRV-001 through MRV-008 (Stationary/Mobile/Process/Fugitive/etc.)
    Scope 2: MRV-009 through MRV-013 (Location/Market/Steam/Cooling/Dual)
    Scope 3: MRV-014 through MRV-028 (Categories 1-15)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-019 CSDDD Readiness Pack
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

class EmissionScope(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"

class TargetValidationStatus(str, Enum):
    """Validation status for a climate target against MRV data."""

    VALIDATED = "validated"
    ON_TRACK = "on_track"
    BEHIND = "behind"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_VALIDATED = "not_validated"

class ReductionTrajectory(str, Enum):
    """Emission reduction trajectory classification."""

    ALIGNED_15C = "aligned_1.5c"
    ALIGNED_2C = "aligned_2c"
    NOT_ALIGNED = "not_aligned"
    INSUFFICIENT_DATA = "insufficient_data"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class MRVBridgeConfig(BaseModel):
    """Configuration for the MRV Bridge."""

    pack_id: str = Field(default="PACK-019")
    enable_provenance: bool = Field(default=True)
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    gwp_source: str = Field(default="IPCC AR6")
    required_reduction_rate_15c: float = Field(
        default=4.2, description="Required annual reduction rate (%) for 1.5C alignment"
    )
    required_reduction_rate_2c: float = Field(
        default=2.5, description="Required annual reduction rate (%) for 2C alignment"
    )

class EmissionDataPoint(BaseModel):
    """A single emission data point from MRV agents."""

    scope: EmissionScope = Field(default=EmissionScope.SCOPE_1)
    category: str = Field(default="")
    tco2e: float = Field(default=0.0, ge=0.0)
    year: int = Field(default=2025)
    source_agent: str = Field(default="")
    methodology: str = Field(default="")

class ScopeEmissions(BaseModel):
    """Aggregated emissions for a single scope."""

    scope: EmissionScope = Field(default=EmissionScope.SCOPE_1)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    categories: List[EmissionDataPoint] = Field(default_factory=list)
    year: int = Field(default=2025)
    agent_count: int = Field(default=0)

class TargetValidation(BaseModel):
    """Result of validating a climate target against MRV data."""

    target_id: str = Field(default="")
    target_description: str = Field(default="")
    target_year: int = Field(default=2030)
    target_reduction_pct: float = Field(default=0.0)
    baseline_tco2e: float = Field(default=0.0)
    current_tco2e: float = Field(default=0.0)
    achieved_reduction_pct: float = Field(default=0.0)
    remaining_reduction_pct: float = Field(default=0.0)
    status: TargetValidationStatus = Field(default=TargetValidationStatus.NOT_VALIDATED)
    trajectory: ReductionTrajectory = Field(default=ReductionTrajectory.INSUFFICIENT_DATA)

class ReductionProgress(BaseModel):
    """Year-over-year emission reduction progress calculation."""

    baseline_year: int = Field(default=2019)
    baseline_tco2e: float = Field(default=0.0)
    current_year: int = Field(default=2025)
    current_tco2e: float = Field(default=0.0)
    absolute_reduction_tco2e: float = Field(default=0.0)
    reduction_pct: float = Field(default=0.0)
    annual_reduction_rate_pct: float = Field(default=0.0)
    trajectory: ReductionTrajectory = Field(default=ReductionTrajectory.INSUFFICIENT_DATA)
    years_elapsed: int = Field(default=0)
    provenance_hash: str = Field(default="")

class BridgeResult(BaseModel):
    """Result of an MRV bridge operation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# MRV Agent Routing Map
# ---------------------------------------------------------------------------

SCOPE1_AGENTS: Dict[str, str] = {
    "MRV-001": "Stationary Combustion",
    "MRV-002": "Refrigerants & F-Gas",
    "MRV-003": "Mobile Combustion",
    "MRV-004": "Process Emissions",
    "MRV-005": "Fugitive Emissions",
    "MRV-006": "Land Use Emissions",
    "MRV-007": "Waste Treatment Emissions",
    "MRV-008": "Agricultural Emissions",
}

SCOPE2_AGENTS: Dict[str, str] = {
    "MRV-009": "Scope 2 Location-Based",
    "MRV-010": "Scope 2 Market-Based",
    "MRV-011": "Steam/Heat Purchase",
    "MRV-012": "Cooling Purchase",
    "MRV-013": "Dual Reporting Reconciliation",
}

SCOPE3_AGENTS: Dict[str, str] = {
    f"MRV-{i + 13:03d}": f"Scope 3 Category {i}"
    for i in range(1, 16)
}

# ---------------------------------------------------------------------------
# MRVBridge
# ---------------------------------------------------------------------------

class MRVBridge:
    """AGENT-MRV emission data bridge for PACK-019 CSDDD climate transition.

    Routes MRV emission data for climate transition plan substantiation,
    validates reduction targets against measured data, and calculates
    Paris-aligned reduction progress. All calculations are deterministic
    (zero-hallucination).

    Attributes:
        config: Bridge configuration.
        _emission_cache: Cached emission data by company.

    Example:
        >>> bridge = MRVBridge(MRVBridgeConfig(reporting_year=2025))
        >>> data = bridge.get_emission_data("company_123")
        >>> assert data.status == "completed"
    """

    def __init__(self, config: Optional[MRVBridgeConfig] = None) -> None:
        """Initialize MRVBridge."""
        self.config = config or MRVBridgeConfig()
        self._emission_cache: Dict[str, Dict[str, ScopeEmissions]] = {}
        logger.info(
            "MRVBridge initialized (year=%d, gwp=%s)",
            self.config.reporting_year,
            self.config.gwp_source,
        )

    def get_emission_data(
        self,
        company_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> BridgeResult:
        """Get all emission data for a company from MRV agents.

        Args:
            company_id: Company identifier.
            context: Optional pipeline context with pre-loaded data.

        Returns:
            BridgeResult with status and records processed.
        """
        result = BridgeResult(started_at=utcnow())
        ctx = context or {}

        try:
            scope1 = self.get_scope1_data(ctx)
            scope2 = self.get_scope2_data(ctx)
            scope3 = self.get_scope3_data(ctx)

            self._emission_cache[company_id] = {
                EmissionScope.SCOPE_1.value: scope1,
                EmissionScope.SCOPE_2.value: scope2,
                EmissionScope.SCOPE_3.value: scope3,
            }

            total_records = (
                len(scope1.categories) + len(scope2.categories) + len(scope3.categories)
            )
            result.records_processed = total_records
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "company_id": company_id,
                    "scope1_tco2e": scope1.total_tco2e,
                    "scope2_tco2e": scope2.total_tco2e,
                    "scope3_tco2e": scope3.total_tco2e,
                })

            logger.info(
                "Emission data loaded for %s: S1=%.2f, S2=%.2f, S3=%.2f tCO2e",
                company_id,
                scope1.total_tco2e,
                scope2.total_tco2e,
                scope3.total_tco2e,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Emission data retrieval failed: %s", str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def validate_targets_against_mrv(
        self,
        targets: List[Dict[str, Any]],
        mrv_data: Dict[str, Any],
    ) -> List[TargetValidation]:
        """Validate climate targets against MRV emission data.

        Uses deterministic arithmetic only (zero-hallucination).

        Args:
            targets: List of climate target dicts with keys:
                target_id, description, target_year, reduction_pct,
                baseline_year, baseline_tco2e, scope.
            mrv_data: MRV emission data dict with current emissions.

        Returns:
            List of TargetValidation results.
        """
        validations: List[TargetValidation] = []

        for target in targets:
            baseline_tco2e = target.get("baseline_tco2e", 0.0)
            target_reduction = target.get("reduction_pct", 0.0)
            target_year = target.get("target_year", 2030)
            baseline_year = target.get("baseline_year", 2019)
            current_tco2e = mrv_data.get("current_tco2e", 0.0)

            # Deterministic reduction calculation
            if baseline_tco2e > 0:
                achieved_reduction = round(
                    (1.0 - current_tco2e / baseline_tco2e) * 100.0, 2
                )
            else:
                achieved_reduction = 0.0

            remaining = round(target_reduction - achieved_reduction, 2)

            # Determine status
            if baseline_tco2e <= 0 or current_tco2e <= 0:
                status = TargetValidationStatus.INSUFFICIENT_DATA
            elif achieved_reduction >= target_reduction:
                status = TargetValidationStatus.VALIDATED
            else:
                years_total = target_year - baseline_year
                years_elapsed = self.config.reporting_year - baseline_year
                if years_total > 0 and years_elapsed > 0:
                    expected_progress = (years_elapsed / years_total) * target_reduction
                    if achieved_reduction >= expected_progress * 0.9:
                        status = TargetValidationStatus.ON_TRACK
                    else:
                        status = TargetValidationStatus.BEHIND
                else:
                    status = TargetValidationStatus.NOT_VALIDATED

            # Determine trajectory
            years_elapsed = max(self.config.reporting_year - baseline_year, 1)
            annual_rate = achieved_reduction / years_elapsed if years_elapsed > 0 else 0.0
            if annual_rate >= self.config.required_reduction_rate_15c:
                trajectory = ReductionTrajectory.ALIGNED_15C
            elif annual_rate >= self.config.required_reduction_rate_2c:
                trajectory = ReductionTrajectory.ALIGNED_2C
            elif baseline_tco2e > 0:
                trajectory = ReductionTrajectory.NOT_ALIGNED
            else:
                trajectory = ReductionTrajectory.INSUFFICIENT_DATA

            validations.append(TargetValidation(
                target_id=target.get("target_id", _new_uuid()),
                target_description=target.get("description", ""),
                target_year=target_year,
                target_reduction_pct=target_reduction,
                baseline_tco2e=baseline_tco2e,
                current_tco2e=current_tco2e,
                achieved_reduction_pct=achieved_reduction,
                remaining_reduction_pct=max(remaining, 0.0),
                status=status,
                trajectory=trajectory,
            ))

        logger.info("Validated %d climate targets", len(validations))
        return validations

    def get_scope1_data(self, context: Dict[str, Any]) -> ScopeEmissions:
        """Get aggregated Scope 1 emissions from MRV agents 001-008.

        Args:
            context: Pipeline context with pre-loaded emission data.

        Returns:
            ScopeEmissions for Scope 1.
        """
        emissions = context.get("scope1_emissions", [])
        categories = [
            EmissionDataPoint(
                scope=EmissionScope.SCOPE_1,
                category=e.get("category", ""),
                tco2e=e.get("tco2e", 0.0),
                year=e.get("year", self.config.reporting_year),
                source_agent=e.get("source_agent", ""),
            )
            for e in emissions
        ]
        total = round(sum(c.tco2e for c in categories), 2)

        return ScopeEmissions(
            scope=EmissionScope.SCOPE_1,
            total_tco2e=total,
            categories=categories,
            year=self.config.reporting_year,
            agent_count=len(SCOPE1_AGENTS),
        )

    def get_scope2_data(self, context: Dict[str, Any]) -> ScopeEmissions:
        """Get aggregated Scope 2 emissions from MRV agents 009-013.

        Args:
            context: Pipeline context with pre-loaded emission data.

        Returns:
            ScopeEmissions for Scope 2.
        """
        location = context.get("scope2_location_tco2e", 0.0)
        market = context.get("scope2_market_tco2e", 0.0)

        categories = [
            EmissionDataPoint(
                scope=EmissionScope.SCOPE_2,
                category="location_based",
                tco2e=location,
                year=self.config.reporting_year,
                source_agent="MRV-009",
            ),
            EmissionDataPoint(
                scope=EmissionScope.SCOPE_2,
                category="market_based",
                tco2e=market,
                year=self.config.reporting_year,
                source_agent="MRV-010",
            ),
        ]

        return ScopeEmissions(
            scope=EmissionScope.SCOPE_2,
            total_tco2e=round(location, 2),
            categories=categories,
            year=self.config.reporting_year,
            agent_count=len(SCOPE2_AGENTS),
        )

    def get_scope3_data(self, context: Dict[str, Any]) -> ScopeEmissions:
        """Get aggregated Scope 3 emissions from MRV agents 014-028.

        Args:
            context: Pipeline context with pre-loaded emission data.

        Returns:
            ScopeEmissions for Scope 3.
        """
        scope3_cats = context.get("scope3_categories", [])
        categories = [
            EmissionDataPoint(
                scope=EmissionScope.SCOPE_3,
                category=c.get("category", f"category_{i + 1}"),
                tco2e=c.get("tco2e", 0.0),
                year=c.get("year", self.config.reporting_year),
                source_agent=c.get("source_agent", f"MRV-{i + 14:03d}"),
            )
            for i, c in enumerate(scope3_cats)
        ]
        total = round(sum(c.tco2e for c in categories), 2)

        return ScopeEmissions(
            scope=EmissionScope.SCOPE_3,
            total_tco2e=total,
            categories=categories,
            year=self.config.reporting_year,
            agent_count=len(SCOPE3_AGENTS),
        )

    def calculate_reduction_progress(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> ReductionProgress:
        """Calculate year-over-year emission reduction progress.

        Deterministic arithmetic only (zero-hallucination).

        Args:
            baseline: Dict with keys: year, tco2e (baseline emissions).
            current: Dict with keys: year, tco2e (current emissions).

        Returns:
            ReductionProgress with absolute/percentage reduction and trajectory.
        """
        baseline_year = baseline.get("year", 2019)
        baseline_tco2e = baseline.get("tco2e", 0.0)
        current_year = current.get("year", self.config.reporting_year)
        current_tco2e = current.get("tco2e", 0.0)

        years_elapsed = max(current_year - baseline_year, 1)
        absolute_reduction = round(baseline_tco2e - current_tco2e, 2)

        if baseline_tco2e > 0:
            reduction_pct = round(
                (1.0 - current_tco2e / baseline_tco2e) * 100.0, 2
            )
        else:
            reduction_pct = 0.0

        annual_rate = round(reduction_pct / years_elapsed, 2) if years_elapsed > 0 else 0.0

        if annual_rate >= self.config.required_reduction_rate_15c:
            trajectory = ReductionTrajectory.ALIGNED_15C
        elif annual_rate >= self.config.required_reduction_rate_2c:
            trajectory = ReductionTrajectory.ALIGNED_2C
        elif baseline_tco2e > 0:
            trajectory = ReductionTrajectory.NOT_ALIGNED
        else:
            trajectory = ReductionTrajectory.INSUFFICIENT_DATA

        progress = ReductionProgress(
            baseline_year=baseline_year,
            baseline_tco2e=baseline_tco2e,
            current_year=current_year,
            current_tco2e=current_tco2e,
            absolute_reduction_tco2e=absolute_reduction,
            reduction_pct=reduction_pct,
            annual_reduction_rate_pct=annual_rate,
            trajectory=trajectory,
            years_elapsed=years_elapsed,
        )
        progress.provenance_hash = _compute_hash(progress)

        logger.info(
            "Reduction progress: %.2f%% over %d years (trajectory=%s)",
            reduction_pct,
            years_elapsed,
            trajectory.value,
        )
        return progress
