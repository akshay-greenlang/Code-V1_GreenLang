"""
Base Year Manager -- GHG Protocol Chapter 6 Implementation

Manages base year selection, locking, recalculation triggers, and
year-over-year trend analysis.  Per the GHG Protocol Corporate Standard,
the base year is the reference point for emissions tracking and must be
recalculated when structural changes exceed the significance threshold.

Structural change triggers (GHG Protocol Ch 6):
  - Mergers and acquisitions
  - Divestitures
  - Outsourcing / insourcing
  - Methodology changes
  - Error corrections
  - Changes in reporting boundary

Example:
    >>> mgr = BaseYearManager(config)
    >>> by = mgr.set_base_year("org-1", SetBaseYearRequest(...))
    >>> mgr.lock_base_year("org-1")
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional

from .config import GHGAppConfig, Scope
from .models import (
    BaseYear,
    GHGInventory,
    Recalculation,
    SetBaseYearRequest,
    RecalculateBaseYearRequest,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


class BaseYearManager:
    """
    Manages base year selection and recalculation per GHG Protocol Ch 6.

    Key rules implemented:
      1. Base year must be representative of typical emissions.
      2. Recalculation is triggered when structural changes exceed the
         significance threshold (default 5%).
      3. Once locked, the base year can only be changed through the
         formal recalculation process.
      4. A complete history of recalculations is maintained for audit.
    """

    STRUCTURAL_CHANGE_TRIGGERS = [
        "merger",
        "acquisition",
        "divestiture",
        "outsourcing",
        "insourcing",
        "methodology_change",
        "error_correction",
        "boundary_change",
    ]

    def __init__(
        self,
        config: Optional[GHGAppConfig] = None,
        inventory_store: Optional[Dict[str, GHGInventory]] = None,
    ) -> None:
        """
        Initialize BaseYearManager.

        Args:
            config: Application configuration.
            inventory_store: Shared reference to inventory storage (for trend queries).
        """
        self.config = config or GHGAppConfig()
        self.significance_threshold = self.config.base_year_significance_threshold
        self._base_years: Dict[str, BaseYear] = {}
        self._inventory_store = inventory_store if inventory_store is not None else {}
        logger.info(
            "BaseYearManager initialized (threshold=%.1f%%)",
            self.significance_threshold,
        )

    # ------------------------------------------------------------------
    # Base Year CRUD
    # ------------------------------------------------------------------

    def set_base_year(
        self,
        org_id: str,
        request: SetBaseYearRequest,
    ) -> BaseYear:
        """
        Set or replace the base year for an organization.

        Args:
            org_id: Organization ID.
            request: Base year data.

        Returns:
            Created BaseYear.

        Raises:
            ValueError: If base year already locked (use recalculate instead).
        """
        start = datetime.utcnow()

        existing = self._base_years.get(org_id)
        if existing and existing.locked:
            raise ValueError(
                "Base year is locked. Use recalculate_base_year() for changes."
            )

        total = (
            request.scope1_emissions
            + request.scope2_location_emissions
            + request.scope3_emissions
        )

        base_year = BaseYear(
            org_id=org_id,
            year=request.year,
            scope1_emissions=request.scope1_emissions,
            scope2_location_emissions=request.scope2_location_emissions,
            scope2_market_emissions=request.scope2_market_emissions,
            scope3_emissions=request.scope3_emissions,
            total_emissions=total,
            justification=request.justification,
        )
        self._base_years[org_id] = base_year

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Set base year %d for org '%s' (total=%.2f tCO2e) in %.1f ms",
            request.year,
            org_id,
            total,
            elapsed_ms,
        )
        return base_year

    def get_base_year(self, org_id: str) -> Optional[BaseYear]:
        """Retrieve the base year for an organization."""
        return self._base_years.get(org_id)

    def lock_base_year(self, org_id: str) -> BaseYear:
        """
        Lock the base year, preventing casual modification.

        After locking, changes require formal recalculation with
        documented structural change triggers.

        Args:
            org_id: Organization ID.

        Returns:
            Locked BaseYear.

        Raises:
            ValueError: If no base year exists.
        """
        base_year = self._get_base_year_or_raise(org_id)
        base_year.locked = True
        base_year.updated_at = _now()

        logger.info("Locked base year %d for org '%s'", base_year.year, org_id)
        return base_year

    def unlock_base_year(self, org_id: str) -> BaseYear:
        """
        Unlock the base year (admin operation).

        Args:
            org_id: Organization ID.

        Returns:
            Unlocked BaseYear.
        """
        base_year = self._get_base_year_or_raise(org_id)
        base_year.locked = False
        base_year.updated_at = _now()

        logger.warning("Unlocked base year %d for org '%s'", base_year.year, org_id)
        return base_year

    # ------------------------------------------------------------------
    # Recalculation Logic
    # ------------------------------------------------------------------

    def check_recalculation_needed(
        self,
        org_id: str,
        current_year_emissions: Decimal,
        trigger: str,
    ) -> bool:
        """
        Determine whether a base year recalculation is needed.

        Per GHG Protocol Ch 6, recalculation is triggered when:
          1. The trigger is a recognized structural change, AND
          2. The impact exceeds the significance threshold.

        Args:
            org_id: Organization ID.
            current_year_emissions: Emissions that would result from the change.
            trigger: Structural change type.

        Returns:
            True if recalculation is needed.
        """
        if trigger not in self.STRUCTURAL_CHANGE_TRIGGERS:
            logger.warning("Unrecognized trigger '%s' -- not a structural change", trigger)
            return False

        base_year = self._base_years.get(org_id)
        if base_year is None:
            logger.info("No base year set for org '%s' -- recalculation not applicable", org_id)
            return False

        reference = base_year.current_total
        if reference == 0:
            return current_year_emissions > 0

        change_pct = abs(
            (current_year_emissions - reference) / reference * 100
        )

        needed = change_pct >= self.significance_threshold

        logger.info(
            "Recalculation check for org '%s': trigger=%s, change=%.2f%%, "
            "threshold=%.1f%%, needed=%s",
            org_id,
            trigger,
            change_pct,
            self.significance_threshold,
            needed,
        )
        return needed

    def recalculate_base_year(
        self,
        org_id: str,
        request: RecalculateBaseYearRequest,
    ) -> Recalculation:
        """
        Perform a formal base year recalculation.

        Creates a Recalculation record and updates the base year
        emissions.  The original values are preserved in the
        recalculation history.

        Args:
            org_id: Organization ID.
            request: Recalculation details.

        Returns:
            Created Recalculation record.

        Raises:
            ValueError: If trigger not recognized or base year not set.
        """
        start = datetime.utcnow()

        if request.trigger not in self.STRUCTURAL_CHANGE_TRIGGERS:
            raise ValueError(
                f"Invalid trigger '{request.trigger}'. "
                f"Valid triggers: {self.STRUCTURAL_CHANGE_TRIGGERS}"
            )

        base_year = self._get_base_year_or_raise(org_id)

        original_total = base_year.current_total
        new_total = (
            request.new_scope1
            + request.new_scope2_location
            + request.new_scope3
        )

        recalc = Recalculation(
            trigger=request.trigger,
            original_value=original_total,
            new_value=new_total,
            reason=request.reason,
            affected_scopes=self._determine_affected_scopes(request, base_year),
        )

        # Update base year emissions
        base_year.scope1_emissions = request.new_scope1
        base_year.scope2_location_emissions = request.new_scope2_location
        base_year.scope2_market_emissions = request.new_scope2_market
        base_year.scope3_emissions = request.new_scope3
        base_year.total_emissions = new_total
        base_year.recalculations.append(recalc)
        base_year.updated_at = _now()

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Recalculated base year for org '%s': trigger=%s, "
            "original=%.2f -> new=%.2f (change=%.2f%%) in %.1f ms",
            org_id,
            request.trigger,
            original_total,
            new_total,
            recalc.change_pct,
            elapsed_ms,
        )
        return recalc

    def get_recalculation_history(self, org_id: str) -> List[Recalculation]:
        """
        Get the complete recalculation history for an organization.

        Args:
            org_id: Organization ID.

        Returns:
            List of Recalculation records ordered by date.
        """
        base_year = self._base_years.get(org_id)
        if base_year is None:
            return []
        return sorted(
            base_year.recalculations,
            key=lambda r: r.recalculated_at,
        )

    # ------------------------------------------------------------------
    # Year-over-Year Analysis
    # ------------------------------------------------------------------

    def calculate_yoy_change(
        self,
        org_id: str,
        year: int,
    ) -> Dict:
        """
        Calculate year-over-year emissions change.

        Compares the given year to the prior year. If no prior year
        inventory exists, compares to the base year.

        Args:
            org_id: Organization ID.
            year: Current reporting year.

        Returns:
            Dict with absolute and percentage changes by scope.
        """
        current = self._find_inventory(org_id, year)
        previous = self._find_inventory(org_id, year - 1)
        base_year = self._base_years.get(org_id)

        if current is None:
            return {"error": f"No inventory found for year {year}"}

        # Determine reference
        if previous is not None:
            ref_total = previous.grand_total_tco2e
            ref_year = year - 1
            ref_label = "prior_year"
        elif base_year is not None:
            ref_total = base_year.current_total
            ref_year = base_year.year
            ref_label = "base_year"
        else:
            return {
                "year": year,
                "total_tco2e": str(current.grand_total_tco2e),
                "change_absolute": None,
                "change_pct": None,
                "reference": "none",
            }

        absolute_change = current.grand_total_tco2e - ref_total
        pct_change = Decimal("0")
        if ref_total != 0:
            pct_change = (absolute_change / ref_total * 100).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        result = {
            "year": year,
            "reference_year": ref_year,
            "reference_type": ref_label,
            "total_tco2e": str(current.grand_total_tco2e),
            "reference_tco2e": str(ref_total),
            "change_absolute": str(absolute_change),
            "change_pct": str(pct_change),
            "direction": "decrease" if absolute_change < 0 else "increase",
        }

        # Per-scope breakdown
        scope_changes = self._compute_scope_changes(current, previous, base_year)
        result["by_scope"] = scope_changes

        logger.info(
            "YoY change for org '%s' year %d: %.2f%% vs %s (%d)",
            org_id,
            year,
            pct_change,
            ref_label,
            ref_year,
        )
        return result

    def get_trend_data(
        self,
        org_id: str,
        start_year: int,
        end_year: int,
    ) -> List[Dict]:
        """
        Build a multi-year trend dataset.

        Returns one entry per year with total emissions and
        per-scope breakdowns, including base year reference.

        Args:
            org_id: Organization ID.
            start_year: First year (inclusive).
            end_year: Last year (inclusive).

        Returns:
            List of yearly trend data points.
        """
        trends: List[Dict] = []
        base_year = self._base_years.get(org_id)

        for year in range(start_year, end_year + 1):
            inv = self._find_inventory(org_id, year)
            if inv is None:
                trends.append({
                    "year": year,
                    "total_tco2e": None,
                    "has_data": False,
                })
                continue

            entry: Dict = {
                "year": year,
                "has_data": True,
                "total_tco2e": str(inv.grand_total_tco2e),
                "scope1_tco2e": str(inv.scope1.total_tco2e) if inv.scope1 else "0",
                "scope2_location_tco2e": str(
                    inv.scope2_location.total_tco2e
                ) if inv.scope2_location else "0",
                "scope2_market_tco2e": str(
                    inv.scope2_market.total_tco2e
                ) if inv.scope2_market else "0",
                "scope3_tco2e": str(inv.scope3.total_tco2e) if inv.scope3 else "0",
            }

            # Change from base year
            if base_year and base_year.current_total > 0:
                change_from_base = (
                    (inv.grand_total_tco2e - base_year.current_total)
                    / base_year.current_total
                    * 100
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                entry["change_from_base_pct"] = str(change_from_base)

            trends.append(entry)

        logger.info(
            "Generated trend data for org '%s': %d-%d (%d points)",
            org_id,
            start_year,
            end_year,
            len(trends),
        )
        return trends

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_base_year_or_raise(self, org_id: str) -> BaseYear:
        """Retrieve base year or raise ValueError."""
        base_year = self._base_years.get(org_id)
        if base_year is None:
            raise ValueError(f"No base year set for organization: {org_id}")
        return base_year

    def _find_inventory(self, org_id: str, year: int) -> Optional[GHGInventory]:
        """Find inventory for org-year in the shared store."""
        for inv in self._inventory_store.values():
            if inv.org_id == org_id and inv.year == year:
                return inv
        return None

    def _determine_affected_scopes(
        self,
        request: RecalculateBaseYearRequest,
        base_year: BaseYear,
    ) -> List[Scope]:
        """Determine which scopes are affected by the recalculation."""
        affected: List[Scope] = []
        if request.new_scope1 != base_year.scope1_emissions:
            affected.append(Scope.SCOPE_1)
        if request.new_scope2_location != base_year.scope2_location_emissions:
            affected.append(Scope.SCOPE_2_LOCATION)
        if request.new_scope2_market != base_year.scope2_market_emissions:
            affected.append(Scope.SCOPE_2_MARKET)
        if request.new_scope3 != base_year.scope3_emissions:
            affected.append(Scope.SCOPE_3)
        return affected

    def _compute_scope_changes(
        self,
        current: GHGInventory,
        previous: Optional[GHGInventory],
        base_year: Optional[BaseYear],
    ) -> Dict:
        """Compute per-scope changes between current and reference."""
        scope_map: Dict[str, Dict[str, Decimal]] = {}

        current_scopes = {
            "scope1": current.scope1.total_tco2e if current.scope1 else Decimal("0"),
            "scope2_location": current.scope2_location.total_tco2e if current.scope2_location else Decimal("0"),
            "scope2_market": current.scope2_market.total_tco2e if current.scope2_market else Decimal("0"),
            "scope3": current.scope3.total_tco2e if current.scope3 else Decimal("0"),
        }

        if previous is not None:
            ref_scopes = {
                "scope1": previous.scope1.total_tco2e if previous.scope1 else Decimal("0"),
                "scope2_location": previous.scope2_location.total_tco2e if previous.scope2_location else Decimal("0"),
                "scope2_market": previous.scope2_market.total_tco2e if previous.scope2_market else Decimal("0"),
                "scope3": previous.scope3.total_tco2e if previous.scope3 else Decimal("0"),
            }
        elif base_year is not None:
            ref_scopes = {
                "scope1": base_year.scope1_emissions,
                "scope2_location": base_year.scope2_location_emissions,
                "scope2_market": base_year.scope2_market_emissions,
                "scope3": base_year.scope3_emissions,
            }
        else:
            return {}

        for scope_name in current_scopes:
            cur = current_scopes[scope_name]
            ref = ref_scopes[scope_name]
            absolute = cur - ref
            pct = Decimal("0")
            if ref != 0:
                pct = (absolute / ref * 100).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            scope_map[scope_name] = {
                "current": str(cur),
                "reference": str(ref),
                "change_absolute": str(absolute),
                "change_pct": str(pct),
            }

        return scope_map
