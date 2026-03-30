# -*- coding: utf-8 -*-
"""
NetZeroBridge - Integration with Net Zero Packs for PACK-041
==============================================================

This module provides integration with Net Zero Packs (PACK-021 through
PACK-030) to share baseline inventory data, import emission reduction
targets, track progress toward net zero goals, and assess SBTi alignment.

Pack Integrations:
    PACK-021: Net Zero Starter (baseline provision)
    PACK-022: Net Zero Acceleration (progress tracking)
    PACK-023: SBTi Alignment (target assessment)
    PACK-024-030: Carbon Neutral, Race to Zero, SME/Enterprise, etc.

Key Capabilities:
    - Provide verified Scope 1-2 baseline to net zero planning
    - Import and validate emission reduction targets
    - Track year-over-year progress against targets
    - Assess alignment with SBTi 1.5C and well-below 2C pathways

Zero-Hallucination:
    All progress calculations, pathway alignment checks, and target
    gap analysis use deterministic arithmetic only.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-041 Scope 1-2 Complete
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
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

class TargetType(str, Enum):
    """Emission reduction target types."""

    ABSOLUTE = "absolute"
    INTENSITY = "intensity"
    SECTOR_BASED = "sector_based"

class TargetScope(str, Enum):
    """Target emission scope coverage."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_1_2_3 = "scope_1_2_3"

class SBTiPathway(str, Enum):
    """SBTi approved temperature pathways."""

    WELL_BELOW_2C = "well_below_2c"
    CELSIUS_1_5 = "1.5c"

class ProgressStatus(str, Enum):
    """Progress tracking status."""

    ON_TRACK = "on_track"
    BEHIND = "behind"
    AT_RISK = "at_risk"
    EXCEEDED = "exceeded"

# ---------------------------------------------------------------------------
# SBTi linear annual reduction rates (% per year)
# ---------------------------------------------------------------------------

SBTI_ANNUAL_REDUCTION_RATES: Dict[str, Dict[str, float]] = {
    "1.5c": {
        "scope_1_2_absolute": 4.2,
        "scope_1_2_intensity": 7.0,
    },
    "well_below_2c": {
        "scope_1_2_absolute": 2.5,
        "scope_1_2_intensity": 4.0,
    },
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class BaselineData(BaseModel):
    """Baseline GHG inventory data provided to net zero packs."""

    baseline_id: str = Field(default_factory=_new_uuid)
    organization_name: str = Field(default="")
    base_year: int = Field(default=2019)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    total_scope1_2_tco2e: float = Field(default=0.0, ge=0.0)
    intensity_metrics: Dict[str, float] = Field(default_factory=dict)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    verification_status: str = Field(default="unverified")
    provenance_hash: str = Field(default="")
    timestamp: datetime = Field(default_factory=utcnow)

class EmissionTarget(BaseModel):
    """Emission reduction target from net zero packs."""

    target_id: str = Field(default_factory=_new_uuid)
    target_name: str = Field(default="")
    target_type: TargetType = Field(default=TargetType.ABSOLUTE)
    target_scope: TargetScope = Field(default=TargetScope.SCOPE_1_2)
    base_year: int = Field(default=2019)
    target_year: int = Field(default=2030)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    target_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    sbti_aligned: bool = Field(default=False)
    pathway: Optional[SBTiPathway] = Field(None)
    source_pack: str = Field(default="")
    provenance_hash: str = Field(default="")

class ProgressReport(BaseModel):
    """Progress report against emission reduction targets."""

    report_id: str = Field(default_factory=_new_uuid)
    reporting_year: int = Field(default=2025)
    target_id: str = Field(default="")
    target_name: str = Field(default="")
    base_year_emissions_tco2e: float = Field(default=0.0)
    target_emissions_tco2e: float = Field(default=0.0)
    current_emissions_tco2e: float = Field(default=0.0)
    expected_emissions_tco2e: float = Field(default=0.0)
    reduction_achieved_pct: float = Field(default=0.0)
    reduction_required_pct: float = Field(default=0.0)
    gap_tco2e: float = Field(default=0.0)
    status: ProgressStatus = Field(default=ProgressStatus.ON_TRACK)
    years_remaining: int = Field(default=0)
    annual_reduction_needed_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    timestamp: datetime = Field(default_factory=utcnow)

class SBTiAlignment(BaseModel):
    """SBTi alignment assessment result."""

    assessment_id: str = Field(default_factory=_new_uuid)
    pathway: SBTiPathway = Field(default=SBTiPathway.CELSIUS_1_5)
    aligned: bool = Field(default=False)
    required_annual_reduction_pct: float = Field(default=0.0)
    actual_annual_reduction_pct: float = Field(default=0.0)
    gap_pct: float = Field(default=0.0)
    base_year: int = Field(default=2019)
    reporting_year: int = Field(default=2025)
    target_year: int = Field(default=2030)
    base_emissions_tco2e: float = Field(default=0.0)
    current_emissions_tco2e: float = Field(default=0.0)
    recommendation: str = Field(default="")
    provenance_hash: str = Field(default="")
    timestamp: datetime = Field(default_factory=utcnow)

# ---------------------------------------------------------------------------
# NetZeroBridge
# ---------------------------------------------------------------------------

class NetZeroBridge:
    """Integration with Net Zero Packs (PACK-021 through PACK-030).

    Provides verified Scope 1-2 baseline data to net zero planning packs,
    imports emission reduction targets, tracks year-over-year progress,
    and assesses alignment with SBTi pathways.

    Attributes:
        _baselines: Provided baseline data.
        _targets: Imported emission targets.

    Example:
        >>> bridge = NetZeroBridge()
        >>> baseline = bridge.provide_baseline(inventory)
        >>> targets = bridge.import_targets(pack_data)
        >>> progress = bridge.track_progress(inventory, targets)
    """

    def __init__(self) -> None:
        """Initialize NetZeroBridge."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._baselines: Dict[str, BaselineData] = {}
        self._targets: Dict[str, EmissionTarget] = {}

        self.logger.info("NetZeroBridge initialized")

    # -------------------------------------------------------------------------
    # Baseline Provision
    # -------------------------------------------------------------------------

    def provide_baseline(
        self,
        inventory: Dict[str, Any],
    ) -> BaselineData:
        """Provide verified Scope 1-2 baseline to net zero packs.

        Args:
            inventory: GHG inventory data with scope1/scope2 totals.

        Returns:
            BaselineData for net zero planning.
        """
        scope1 = float(inventory.get("scope1_tco2e", 0.0))
        scope2_loc = float(inventory.get("scope2_location_tco2e", 0.0))
        scope2_mkt = float(inventory.get("scope2_market_tco2e", 0.0))

        baseline = BaselineData(
            organization_name=inventory.get("organization_name", ""),
            base_year=inventory.get("base_year", 2019),
            scope1_tco2e=scope1,
            scope2_location_tco2e=scope2_loc,
            scope2_market_tco2e=scope2_mkt,
            total_scope1_2_tco2e=scope1 + scope2_loc,
            intensity_metrics=inventory.get("intensity_metrics", {}),
            data_quality_score=inventory.get("data_quality_score", 0.0),
            verification_status=inventory.get("verification_status", "unverified"),
        )
        baseline.provenance_hash = _compute_hash(baseline)
        self._baselines[baseline.baseline_id] = baseline

        self.logger.info(
            "Baseline provided: year=%d, scope1=%.1f, scope2=%.1f, total=%.1f tCO2e",
            baseline.base_year, scope1, scope2_loc, baseline.total_scope1_2_tco2e,
        )
        return baseline

    # -------------------------------------------------------------------------
    # Target Import
    # -------------------------------------------------------------------------

    def import_targets(
        self,
        pack_data: Dict[str, Any],
    ) -> List[EmissionTarget]:
        """Import emission reduction targets from net zero packs.

        Args:
            pack_data: Dict with targets list from net zero packs.

        Returns:
            List of imported EmissionTarget records.
        """
        targets_data = pack_data.get("targets", [])
        imported: List[EmissionTarget] = []

        for t in targets_data:
            target = EmissionTarget(
                target_name=t.get("target_name", ""),
                target_type=TargetType(t.get("target_type", "absolute")),
                target_scope=TargetScope(t.get("target_scope", "scope_1_2")),
                base_year=t.get("base_year", 2019),
                target_year=t.get("target_year", 2030),
                base_year_emissions_tco2e=t.get("base_year_emissions_tco2e", 0.0),
                target_emissions_tco2e=t.get("target_emissions_tco2e", 0.0),
                reduction_pct=t.get("reduction_pct", 0.0),
                sbti_aligned=t.get("sbti_aligned", False),
                pathway=SBTiPathway(t["pathway"]) if t.get("pathway") else None,
                source_pack=t.get("source_pack", ""),
            )
            target.provenance_hash = _compute_hash(target)
            self._targets[target.target_id] = target
            imported.append(target)

        self.logger.info("Imported %d emission targets", len(imported))
        return imported

    # -------------------------------------------------------------------------
    # Progress Tracking
    # -------------------------------------------------------------------------

    def track_progress(
        self,
        inventory: Dict[str, Any],
        targets: List[EmissionTarget],
    ) -> List[ProgressReport]:
        """Track progress against emission reduction targets.

        Uses linear interpolation to determine expected emissions for
        the current reporting year and compares with actual.

        Args:
            inventory: Current year GHG inventory with totals.
            targets: List of emission targets to track against.

        Returns:
            List of ProgressReport records.
        """
        reporting_year = inventory.get("reporting_year", 2025)
        current_emissions = float(inventory.get("total_scope1_2_tco2e", 0.0))
        reports: List[ProgressReport] = []

        for target in targets:
            base = Decimal(str(target.base_year_emissions_tco2e))
            target_val = Decimal(str(target.target_emissions_tco2e))
            total_years = max(1, target.target_year - target.base_year)
            elapsed_years = reporting_year - target.base_year
            years_remaining = max(0, target.target_year - reporting_year)

            # Linear interpolation for expected emissions
            annual_reduction = (base - target_val) / Decimal(str(total_years))
            expected = base - (annual_reduction * Decimal(str(elapsed_years)))
            expected_float = float(expected.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

            # Reduction achieved
            reduction_achieved = float(
                ((base - Decimal(str(current_emissions))) / base * Decimal("100")).quantize(
                    Decimal("0.1"), rounding=ROUND_HALF_UP
                )
            ) if base > 0 else 0.0

            # Gap
            gap = current_emissions - expected_float

            # Status determination
            if current_emissions <= expected_float * 0.95:
                status = ProgressStatus.EXCEEDED
            elif current_emissions <= expected_float:
                status = ProgressStatus.ON_TRACK
            elif current_emissions <= expected_float * 1.10:
                status = ProgressStatus.BEHIND
            else:
                status = ProgressStatus.AT_RISK

            # Annual reduction needed to reach target
            annual_needed = 0.0
            if years_remaining > 0 and current_emissions > 0:
                remaining_reduction = current_emissions - float(target_val)
                annual_needed = round(
                    remaining_reduction / current_emissions / years_remaining * 100, 1
                )

            report = ProgressReport(
                reporting_year=reporting_year,
                target_id=target.target_id,
                target_name=target.target_name,
                base_year_emissions_tco2e=float(base),
                target_emissions_tco2e=float(target_val),
                current_emissions_tco2e=current_emissions,
                expected_emissions_tco2e=expected_float,
                reduction_achieved_pct=reduction_achieved,
                reduction_required_pct=target.reduction_pct,
                gap_tco2e=round(gap, 1),
                status=status,
                years_remaining=years_remaining,
                annual_reduction_needed_pct=annual_needed,
            )
            report.provenance_hash = _compute_hash(report)
            reports.append(report)

            self.logger.info(
                "Progress: target='%s', status=%s, achieved=%.1f%%, gap=%.1f tCO2e",
                target.target_name, status.value, reduction_achieved, gap,
            )

        return reports

    # -------------------------------------------------------------------------
    # SBTi Alignment
    # -------------------------------------------------------------------------

    def assess_sbti_alignment(
        self,
        inventory: Dict[str, Any],
        sbti_targets: Optional[Dict[str, Any]] = None,
    ) -> SBTiAlignment:
        """Assess alignment with SBTi pathways.

        Compares actual annual reduction rate against SBTi-required rates
        for 1.5C and well-below 2C pathways.

        Args:
            inventory: Current GHG inventory with base year and current totals.
            sbti_targets: Optional SBTi target parameters.

        Returns:
            SBTiAlignment assessment result.
        """
        start_time = time.monotonic()

        base_year = inventory.get("base_year", 2019)
        reporting_year = inventory.get("reporting_year", 2025)
        target_year = inventory.get("target_year", 2030)
        base_emissions = Decimal(str(inventory.get("base_year_emissions_tco2e", 0.0)))
        current_emissions = Decimal(str(inventory.get("total_scope1_2_tco2e", 0.0)))
        pathway_str = inventory.get("pathway", "1.5c")

        pathway = SBTiPathway.CELSIUS_1_5
        if pathway_str == "well_below_2c":
            pathway = SBTiPathway.WELL_BELOW_2C

        rates = SBTI_ANNUAL_REDUCTION_RATES.get(pathway.value, {})
        required_rate = rates.get("scope_1_2_absolute", 4.2)

        # Actual annual reduction rate
        elapsed_years = max(1, reporting_year - base_year)
        if base_emissions > 0:
            total_reduction_pct = float(
                (base_emissions - current_emissions) / base_emissions * Decimal("100")
            )
            actual_annual_rate = round(total_reduction_pct / elapsed_years, 2)
        else:
            actual_annual_rate = 0.0

        gap = round(required_rate - actual_annual_rate, 2)
        aligned = actual_annual_rate >= required_rate

        recommendation = ""
        if aligned:
            recommendation = (
                f"On track for {pathway.value} pathway. "
                f"Actual rate ({actual_annual_rate}%/yr) meets requirement ({required_rate}%/yr)."
            )
        else:
            recommendation = (
                f"Behind {pathway.value} pathway by {gap}%/yr. "
                f"Need to increase annual reduction from {actual_annual_rate}%/yr "
                f"to {required_rate}%/yr. Consider additional abatement measures."
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        assessment = SBTiAlignment(
            pathway=pathway,
            aligned=aligned,
            required_annual_reduction_pct=required_rate,
            actual_annual_reduction_pct=actual_annual_rate,
            gap_pct=gap,
            base_year=base_year,
            reporting_year=reporting_year,
            target_year=target_year,
            base_emissions_tco2e=float(base_emissions),
            current_emissions_tco2e=float(current_emissions),
            recommendation=recommendation,
        )
        assessment.provenance_hash = _compute_hash(assessment)

        self.logger.info(
            "SBTi alignment: pathway=%s, aligned=%s, actual=%.2f%%/yr, required=%.2f%%/yr",
            pathway.value, aligned, actual_annual_rate, required_rate,
        )
        return assessment
