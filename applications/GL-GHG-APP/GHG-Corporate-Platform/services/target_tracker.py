"""
Target Tracker -- Emission Reduction Targets and SBTi Alignment

Tracks absolute and intensity-based emission reduction targets,
validates alignment with the Science Based Targets initiative (SBTi),
and provides progress monitoring and trajectory forecasting.

SBTi requirements implemented:
  - Near-term targets: 5-year horizon, minimum 4.2%/yr (1.5C) or 2.5%/yr (2C)
  - Long-term targets: 15-year horizon, minimum 90% reduction
  - Annual reduction rate validation
  - Gap-to-target analysis

Example:
    >>> tracker = TargetTracker(config)
    >>> target = tracker.set_target("org-1", SetTargetRequest(...))
    >>> progress = tracker.calculate_progress(target.id)
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import GHGAppConfig, Scope
from .models import (
    GHGInventory,
    SetTargetRequest,
    Target,
    _new_id,
    _now,
)

logger = logging.getLogger(__name__)


class TargetTracker:
    """
    Tracks emission reduction targets and SBTi alignment.

    Provides target setting, progress calculation, trajectory
    forecasting, and gap analysis.
    """

    SBTI_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
        "near_term": {
            "timeframe_years": 5,
            "min_reduction_1_5C": Decimal("4.2"),
            "min_reduction_2C": Decimal("2.5"),
            "description": "Near-term SBTi targets (5-year horizon)",
        },
        "long_term": {
            "timeframe_years": 15,
            "min_reduction": Decimal("90.0"),
            "description": "Long-term SBTi targets (net-zero by 2050)",
        },
    }

    # Valid SBTi pathways
    VALID_PATHWAYS = ["1.5C", "well-below-2C"]

    def __init__(
        self,
        config: Optional[GHGAppConfig] = None,
        inventory_store: Optional[Dict[str, GHGInventory]] = None,
    ) -> None:
        """
        Initialize TargetTracker.

        Args:
            config: Application configuration.
            inventory_store: Shared reference to inventory storage.
        """
        self.config = config or GHGAppConfig()
        self._inventory_store = inventory_store if inventory_store is not None else {}
        self._targets: Dict[str, Target] = {}
        logger.info("TargetTracker initialized")

    # ------------------------------------------------------------------
    # Target CRUD
    # ------------------------------------------------------------------

    def set_target(
        self,
        org_id: str,
        request: SetTargetRequest,
    ) -> Target:
        """
        Set an emission reduction target.

        Args:
            org_id: Organization ID.
            request: Target parameters.

        Returns:
            Created Target.

        Raises:
            ValueError: If target parameters are invalid.
        """
        if request.target_year <= request.base_year:
            raise ValueError("Target year must be after base year")

        target = Target(
            org_id=org_id,
            name=request.name,
            target_type=request.target_type,
            scope=request.scope,
            base_year=request.base_year,
            base_year_emissions=request.base_year_emissions,
            target_year=request.target_year,
            reduction_pct=request.reduction_pct,
            sbti_aligned=request.sbti_aligned,
            sbti_pathway=request.sbti_pathway,
        )

        self._targets[target.id] = target

        logger.info(
            "Set target '%s' for org '%s': %.1f%% reduction by %d (scope=%s)",
            target.name or target.id,
            org_id,
            request.reduction_pct,
            request.target_year,
            request.scope.value,
        )
        return target

    def get_targets(self, org_id: str) -> List[Target]:
        """Get all targets for an organization."""
        return [t for t in self._targets.values() if t.org_id == org_id]

    def get_target(self, target_id: str) -> Optional[Target]:
        """Get a specific target by ID."""
        return self._targets.get(target_id)

    def update_target_progress(
        self,
        target_id: str,
        current_emissions: Decimal,
        current_year: int,
    ) -> Target:
        """
        Update target with current emissions data.

        Args:
            target_id: Target ID.
            current_emissions: Current year emissions.
            current_year: Current reporting year.

        Returns:
            Updated Target.
        """
        target = self._get_target_or_raise(target_id)
        target.current_emissions = current_emissions
        target.current_year = current_year
        target.updated_at = _now()

        logger.info(
            "Updated target %s: current emissions=%.2f tCO2e (year %d), progress=%.1f%%",
            target_id,
            current_emissions,
            current_year,
            target.current_progress_pct,
        )
        return target

    # ------------------------------------------------------------------
    # Progress Calculation
    # ------------------------------------------------------------------

    def calculate_progress(
        self,
        target_id: str,
    ) -> Dict[str, Any]:
        """
        Calculate detailed progress toward a target.

        Args:
            target_id: Target ID.

        Returns:
            Dict with progress metrics.
        """
        target = self._get_target_or_raise(target_id)

        target_reduction = target.base_year_emissions * (target.reduction_pct / Decimal("100"))
        target_absolute = target.base_year_emissions - target_reduction

        actual_reduction = Decimal("0")
        if target.current_emissions is not None:
            actual_reduction = target.base_year_emissions - target.current_emissions

        progress_pct = target.current_progress_pct

        # Time progress
        total_years = target.target_year - target.base_year
        elapsed_years = 0
        if target.current_year:
            elapsed_years = target.current_year - target.base_year
        time_progress_pct = Decimal("0")
        if total_years > 0:
            time_progress_pct = (
                Decimal(str(elapsed_years)) / Decimal(str(total_years)) * 100
            ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        # On track assessment
        on_track = progress_pct >= time_progress_pct if time_progress_pct > 0 else True

        result = {
            "target_id": target.id,
            "target_name": target.name,
            "scope": target.scope.value,
            "target_type": target.target_type.value,
            "base_year": target.base_year,
            "base_year_emissions": str(target.base_year_emissions),
            "target_year": target.target_year,
            "target_reduction_pct": str(target.reduction_pct),
            "target_absolute_tco2e": str(target_absolute.quantize(Decimal("0.01"))),
            "current_year": target.current_year,
            "current_emissions": str(target.current_emissions) if target.current_emissions else None,
            "actual_reduction_tco2e": str(actual_reduction.quantize(Decimal("0.01"))),
            "progress_pct": str(progress_pct.quantize(Decimal("0.1"))),
            "time_progress_pct": str(time_progress_pct),
            "on_track": on_track,
            "remaining_reduction_tco2e": str(
                (target_reduction - actual_reduction).quantize(Decimal("0.01"))
            ),
            "years_remaining": target.target_year - (target.current_year or target.base_year),
        }

        return result

    # ------------------------------------------------------------------
    # Trajectory Forecasting
    # ------------------------------------------------------------------

    def forecast_trajectory(
        self,
        target_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Forecast the required emissions trajectory to meet the target.

        Produces a linear pathway from current emissions to the target,
        showing required annual reductions.

        Args:
            target_id: Target ID.

        Returns:
            List of annual trajectory points.
        """
        target = self._get_target_or_raise(target_id)

        start_year = target.current_year or target.base_year
        start_emissions = target.current_emissions or target.base_year_emissions

        target_emissions = target.base_year_emissions * (
            1 - target.reduction_pct / Decimal("100")
        )

        remaining_years = target.target_year - start_year
        if remaining_years <= 0:
            return [
                {
                    "year": target.target_year,
                    "required_tco2e": str(target_emissions.quantize(Decimal("0.01"))),
                    "note": "Target year has been reached or passed",
                }
            ]

        annual_reduction = (start_emissions - target_emissions) / Decimal(str(remaining_years))
        trajectory: List[Dict[str, Any]] = []

        for i in range(remaining_years + 1):
            year = start_year + i
            required = start_emissions - (annual_reduction * Decimal(str(i)))
            required = max(required, Decimal("0"))

            point: Dict[str, Any] = {
                "year": year,
                "required_tco2e": str(required.quantize(Decimal("0.01"))),
                "reduction_from_base_pct": str(
                    ((target.base_year_emissions - required) / target.base_year_emissions * 100).quantize(
                        Decimal("0.1")
                    )
                    if target.base_year_emissions > 0
                    else "0"
                ),
            }

            # Add actual data if available
            actual = self._find_emissions_for_year(target.org_id, year, target.scope)
            if actual is not None:
                point["actual_tco2e"] = str(actual.quantize(Decimal("0.01")))
                point["variance_tco2e"] = str(
                    (actual - required).quantize(Decimal("0.01"))
                )
                point["on_track"] = actual <= required

            trajectory.append(point)

        return trajectory

    # ------------------------------------------------------------------
    # SBTi Alignment
    # ------------------------------------------------------------------

    def check_sbti_alignment(
        self,
        target_id: str,
    ) -> Dict[str, Any]:
        """
        Check whether a target meets SBTi requirements.

        Validates:
          1. Timeframe appropriateness
          2. Annual reduction rate
          3. Pathway alignment (1.5C or well-below-2C)
          4. Scope coverage

        Args:
            target_id: Target ID.

        Returns:
            Dict with alignment assessment.
        """
        target = self._get_target_or_raise(target_id)

        timeframe = target.target_year - target.base_year
        annual_reduction = self.get_required_annual_reduction(target_id)

        checks: Dict[str, Any] = {
            "target_id": target.id,
            "aligned": True,
            "pathway": target.sbti_pathway,
            "checks": [],
        }

        # Check 1: Valid timeframe
        if timeframe <= self.SBTI_REQUIREMENTS["near_term"]["timeframe_years"]:
            target_type = "near_term"
        elif timeframe <= self.SBTI_REQUIREMENTS["long_term"]["timeframe_years"]:
            target_type = "long_term"
        else:
            target_type = "beyond_long_term"

        timeframe_ok = timeframe >= 5 and timeframe <= 15
        checks["checks"].append({
            "name": "timeframe",
            "description": "Target timeframe is 5-15 years",
            "passed": timeframe_ok,
            "detail": f"Timeframe: {timeframe} years ({target_type})",
        })
        if not timeframe_ok:
            checks["aligned"] = False

        # Check 2: Reduction rate
        if target.sbti_pathway == "1.5C":
            min_rate = self.SBTI_REQUIREMENTS["near_term"]["min_reduction_1_5C"]
            rate_ok = annual_reduction >= min_rate
            checks["checks"].append({
                "name": "reduction_rate",
                "description": f"Annual reduction >= {min_rate}%/yr for 1.5C",
                "passed": rate_ok,
                "detail": f"Annual rate: {annual_reduction.quantize(Decimal('0.1'))}%/yr",
            })
        elif target.sbti_pathway == "well-below-2C":
            min_rate = self.SBTI_REQUIREMENTS["near_term"]["min_reduction_2C"]
            rate_ok = annual_reduction >= min_rate
            checks["checks"].append({
                "name": "reduction_rate",
                "description": f"Annual reduction >= {min_rate}%/yr for well-below-2C",
                "passed": rate_ok,
                "detail": f"Annual rate: {annual_reduction.quantize(Decimal('0.1'))}%/yr",
            })
        else:
            rate_ok = False
            checks["checks"].append({
                "name": "reduction_rate",
                "description": "Valid SBTi pathway required (1.5C or well-below-2C)",
                "passed": False,
                "detail": f"Pathway: {target.sbti_pathway}",
            })

        if not rate_ok:
            checks["aligned"] = False

        # Check 3: Scope coverage (Scope 1+2 mandatory, Scope 3 if >40%)
        scope_ok = target.scope in (
            Scope.SCOPE_1,
            Scope.SCOPE_2_LOCATION,
            Scope.SCOPE_2_MARKET,
        )
        checks["checks"].append({
            "name": "scope_coverage",
            "description": "Target covers mandatory scopes (Scope 1 and/or 2)",
            "passed": scope_ok or target.scope == Scope.SCOPE_3,
            "detail": f"Target scope: {target.scope.value}",
        })

        # Check 4: Total reduction (long-term minimum 90%)
        if target_type == "long_term":
            reduction_ok = target.reduction_pct >= Decimal("90")
            checks["checks"].append({
                "name": "long_term_reduction",
                "description": "Long-term target requires >= 90% reduction",
                "passed": reduction_ok,
                "detail": f"Target reduction: {target.reduction_pct}%",
            })
            if not reduction_ok:
                checks["aligned"] = False

        checks["summary"] = (
            "Target meets SBTi requirements"
            if checks["aligned"]
            else "Target does not meet SBTi requirements -- see failed checks"
        )

        return checks

    # ------------------------------------------------------------------
    # Gap Analysis
    # ------------------------------------------------------------------

    def get_gap_to_target(
        self,
        target_id: str,
    ) -> Dict[str, Any]:
        """
        Calculate the gap between current trajectory and target.

        Args:
            target_id: Target ID.

        Returns:
            Dict with gap analysis.
        """
        target = self._get_target_or_raise(target_id)

        if target.current_emissions is None or target.current_year is None:
            return {
                "target_id": target.id,
                "error": "Current emissions data not available",
            }

        target_emissions = target.base_year_emissions * (
            1 - target.reduction_pct / Decimal("100")
        )
        remaining = target.current_emissions - target_emissions
        years_left = target.target_year - target.current_year

        required_annual = Decimal("0")
        if years_left > 0:
            required_annual = (remaining / Decimal(str(years_left))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        # Historical annual reduction (if we have data)
        historical_rate = self._calculate_historical_rate(target)

        # Will we meet the target at current rate?
        will_meet = False
        if historical_rate and historical_rate > 0 and years_left > 0:
            projected_reduction = historical_rate * Decimal(str(years_left))
            projected_emissions = target.current_emissions - projected_reduction
            will_meet = projected_emissions <= target_emissions

        return {
            "target_id": target.id,
            "target_name": target.name,
            "current_emissions": str(target.current_emissions),
            "target_emissions": str(target_emissions.quantize(Decimal("0.01"))),
            "gap_tco2e": str(remaining.quantize(Decimal("0.01"))),
            "years_remaining": years_left,
            "required_annual_reduction_tco2e": str(required_annual),
            "historical_annual_reduction_tco2e": (
                str(historical_rate.quantize(Decimal("0.01")))
                if historical_rate
                else None
            ),
            "on_track_at_current_rate": will_meet,
            "acceleration_needed": (
                not will_meet and historical_rate is not None and historical_rate > 0
            ),
        }

    def get_required_annual_reduction(
        self,
        target_id: str,
    ) -> Decimal:
        """
        Calculate the required annual reduction rate (%).

        This is the compound annual reduction needed to reach the
        target from the base year.

        Args:
            target_id: Target ID.

        Returns:
            Annual reduction percentage.
        """
        target = self._get_target_or_raise(target_id)
        timeframe = target.target_year - target.base_year

        if timeframe <= 0:
            return Decimal("0")

        annual_rate = (target.reduction_pct / Decimal(str(timeframe))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        return annual_rate

    # ------------------------------------------------------------------
    # Dashboard Support
    # ------------------------------------------------------------------

    def get_target_summary(
        self,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Get a summary of all targets for dashboard display.

        Args:
            org_id: Organization ID.

        Returns:
            Dict with target summaries.
        """
        targets = self.get_targets(org_id)
        if not targets:
            return {"org_id": org_id, "targets": [], "count": 0}

        summaries = []
        for t in targets:
            summaries.append({
                "id": t.id,
                "name": t.name,
                "scope": t.scope.value,
                "type": t.target_type.value,
                "base_year": t.base_year,
                "target_year": t.target_year,
                "reduction_pct": str(t.reduction_pct),
                "progress_pct": str(t.current_progress_pct.quantize(Decimal("0.1"))),
                "sbti_aligned": t.sbti_aligned,
                "sbti_pathway": t.sbti_pathway,
            })

        return {
            "org_id": org_id,
            "count": len(summaries),
            "targets": summaries,
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_target_or_raise(self, target_id: str) -> Target:
        """Retrieve target or raise ValueError."""
        target = self._targets.get(target_id)
        if target is None:
            raise ValueError(f"Target not found: {target_id}")
        return target

    def _find_emissions_for_year(
        self,
        org_id: str,
        year: int,
        scope: Scope,
    ) -> Optional[Decimal]:
        """Find emissions for a specific org/year/scope."""
        for inv in self._inventory_store.values():
            if inv.org_id != org_id or inv.year != year:
                continue

            if scope == Scope.SCOPE_1 and inv.scope1:
                return inv.scope1.total_tco2e
            if scope == Scope.SCOPE_2_LOCATION and inv.scope2_location:
                return inv.scope2_location.total_tco2e
            if scope == Scope.SCOPE_2_MARKET and inv.scope2_market:
                return inv.scope2_market.total_tco2e
            if scope == Scope.SCOPE_3 and inv.scope3:
                return inv.scope3.total_tco2e

            return inv.grand_total_tco2e

        return None

    def _calculate_historical_rate(
        self,
        target: Target,
    ) -> Optional[Decimal]:
        """Calculate historical annual reduction rate from inventory data."""
        if target.current_emissions is None or target.current_year is None:
            return None

        elapsed = target.current_year - target.base_year
        if elapsed <= 0:
            return None

        total_reduction = target.base_year_emissions - target.current_emissions
        return (total_reduction / Decimal(str(elapsed))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
