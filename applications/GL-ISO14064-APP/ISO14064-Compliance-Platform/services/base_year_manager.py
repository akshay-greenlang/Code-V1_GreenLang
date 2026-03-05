"""
Base Year Manager -- ISO 14064-1:2018 Clause 5.3 / 7.3 Implementation

Manages base year selection, locking, recalculation triggers, and
historical audit trail per ISO 14064-1:2018.

Recalculation triggers:
  - Structural changes (acquisition, divestiture, outsourcing, insourcing)
  - Methodology changes (new EFs, revised calculation approach)
  - Error corrections exceeding significance threshold
  - Boundary changes

Supports fixed and rolling base year approaches.

Example:
    >>> mgr = BaseYearManager(config)
    >>> base = mgr.set_base_year("org-1", 2020, total=Decimal("10000"),
    ...     justification="Representative year with complete data")
    >>> mgr.lock_base_year("org-1")
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    ISO14064AppConfig,
    ISOCategory,
)
from .models import (
    BaseYear,
    Recalculation,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


class BaseYearManager:
    """
    Manages base year selection and recalculation per ISO 14064-1.

    Key rules:
      1. Base year must be representative of typical emissions.
      2. Recalculation triggered when changes exceed threshold (default 5%).
      3. Locked base year requires formal recalculation with trigger.
      4. Complete recalculation history maintained for audit.
      5. Supports fixed and rolling base year approaches.
    """

    RECOGNIZED_TRIGGERS: List[str] = [
        "structural_change",
        "methodology_change",
        "error_correction",
        "boundary_change",
        "acquisition",
        "divestiture",
        "outsourcing",
        "insourcing",
    ]

    def __init__(
        self,
        config: Optional[ISO14064AppConfig] = None,
    ) -> None:
        """
        Initialize BaseYearManager.

        Args:
            config: Application configuration.
        """
        self.config = config or ISO14064AppConfig()
        self._threshold = self.config.recalculation_threshold_percent
        self._base_years: Dict[str, BaseYear] = {}
        logger.info("BaseYearManager initialized (threshold=%.1f%%)", self._threshold)

    # ------------------------------------------------------------------
    # Base Year CRUD
    # ------------------------------------------------------------------

    def set_base_year(
        self,
        org_id: str,
        year: int,
        total_emissions: Decimal = Decimal("0"),
        total_removals: Decimal = Decimal("0"),
        justification: str = "Representative year selected per ISO 14064-1 Clause 5.3",
        category_emissions: Optional[Dict[str, Decimal]] = None,
    ) -> BaseYear:
        """
        Set or replace the base year for an organization.

        Args:
            org_id: Organization ID.
            year: Base year.
            total_emissions: Total emissions (tCO2e).
            total_removals: Total removals (tCO2e).
            justification: Reason for selecting this year.
            category_emissions: Optional per-category breakdown.

        Returns:
            Created BaseYear.

        Raises:
            ValueError: If base year is locked.
        """
        existing = self._base_years.get(org_id)
        if existing and existing.locked:
            raise ValueError(
                "Base year is locked. Use recalculate_base_year() for changes."
            )

        cat_fields: Dict[str, Decimal] = {}
        if category_emissions:
            cat_fields["category_1_emissions"] = category_emissions.get(
                ISOCategory.CATEGORY_1_DIRECT.value, Decimal("0"),
            )
            cat_fields["category_2_emissions"] = category_emissions.get(
                ISOCategory.CATEGORY_2_ENERGY.value, Decimal("0"),
            )
            cat_fields["category_3_emissions"] = category_emissions.get(
                ISOCategory.CATEGORY_3_TRANSPORT.value, Decimal("0"),
            )
            cat_fields["category_4_emissions"] = category_emissions.get(
                ISOCategory.CATEGORY_4_PRODUCTS_USED.value, Decimal("0"),
            )
            cat_fields["category_5_emissions"] = category_emissions.get(
                ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG.value, Decimal("0"),
            )
            cat_fields["category_6_emissions"] = category_emissions.get(
                ISOCategory.CATEGORY_6_OTHER.value, Decimal("0"),
            )

        net = total_emissions - total_removals

        record = BaseYear(
            org_id=org_id,
            year=year,
            total_emissions=total_emissions,
            total_removals=total_removals,
            net_emissions=net,
            justification=justification,
            **cat_fields,
        )
        self._base_years[org_id] = record

        logger.info(
            "Set base year %d for org '%s' (%.2f tCO2e)",
            year, org_id, total_emissions,
        )
        return record

    def get_base_year(self, org_id: str) -> Optional[BaseYear]:
        """Retrieve the base year record for an organization."""
        return self._base_years.get(org_id)

    def lock_base_year(self, org_id: str) -> BaseYear:
        """Lock the base year, preventing casual modification."""
        record = self._get_record_or_raise(org_id)
        record.locked = True
        record.updated_at = _now()
        logger.info("Locked base year %d for org '%s'", record.year, org_id)
        return record

    def unlock_base_year(self, org_id: str) -> BaseYear:
        """Unlock the base year (admin operation)."""
        record = self._get_record_or_raise(org_id)
        record.locked = False
        record.updated_at = _now()
        logger.warning("Unlocked base year %d for org '%s'", record.year, org_id)
        return record

    def is_locked(self, org_id: str) -> bool:
        """Check if base year is locked."""
        record = self._base_years.get(org_id)
        if record is None:
            return False
        return record.locked

    # ------------------------------------------------------------------
    # Recalculation
    # ------------------------------------------------------------------

    def check_recalculation_needed(
        self,
        org_id: str,
        new_total: Decimal,
        trigger_type: str,
    ) -> bool:
        """
        Determine whether base year recalculation is needed.

        Args:
            org_id: Organization ID.
            new_total: Emissions after the change.
            trigger_type: Type of change.

        Returns:
            True if recalculation is needed.
        """
        if trigger_type not in self.RECOGNIZED_TRIGGERS:
            logger.warning("Unrecognized trigger '%s'", trigger_type)
            return False

        record = self._base_years.get(org_id)
        if record is None:
            return False

        reference = record.current_total
        if reference == 0:
            return new_total > 0

        change_pct = abs((new_total - reference) / reference * 100)
        needed = change_pct >= self._threshold

        logger.info(
            "Recalculation check: trigger=%s, change=%.2f%%, threshold=%.1f%%, needed=%s",
            trigger_type, change_pct, self._threshold, needed,
        )
        return needed

    def recalculate_base_year(
        self,
        org_id: str,
        trigger: str,
        new_total: Decimal,
        reason: str,
        new_by_category: Optional[Dict[str, Decimal]] = None,
        affected_categories: Optional[List[ISOCategory]] = None,
        approved_by: Optional[str] = None,
    ) -> Recalculation:
        """
        Perform a formal base year recalculation.

        Args:
            org_id: Organization ID.
            trigger: Recalculation trigger type.
            new_total: New total emissions.
            reason: Detailed justification (min 10 chars).
            new_by_category: New per-category breakdown.
            affected_categories: Which categories were affected.
            approved_by: Approver ID.

        Returns:
            Created Recalculation record.

        Raises:
            ValueError: If trigger not recognized or base year not set.
        """
        if trigger not in self.RECOGNIZED_TRIGGERS:
            raise ValueError(
                f"Invalid trigger '{trigger}'. Valid: {self.RECOGNIZED_TRIGGERS}"
            )

        record = self._get_record_or_raise(org_id)
        reference = record.current_total

        # Build original by-category snapshot
        original_by_cat: Dict[str, Decimal] = {
            ISOCategory.CATEGORY_1_DIRECT.value: record.category_1_emissions,
            ISOCategory.CATEGORY_2_ENERGY.value: record.category_2_emissions,
            ISOCategory.CATEGORY_3_TRANSPORT.value: record.category_3_emissions,
            ISOCategory.CATEGORY_4_PRODUCTS_USED.value: record.category_4_emissions,
            ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG.value: record.category_5_emissions,
            ISOCategory.CATEGORY_6_OTHER.value: record.category_6_emissions,
        }

        recalc = Recalculation(
            trigger=trigger,
            original_total=reference,
            new_total=new_total,
            original_by_category=original_by_cat,
            new_by_category=new_by_category or {},
            reason=reason,
            affected_categories=affected_categories or [],
            approved_by=approved_by,
        )

        record.recalculations.append(recalc)

        # Update category-level emissions if provided
        if new_by_category:
            for cat_val, val in new_by_category.items():
                if cat_val == ISOCategory.CATEGORY_1_DIRECT.value:
                    record.category_1_emissions = val
                elif cat_val == ISOCategory.CATEGORY_2_ENERGY.value:
                    record.category_2_emissions = val
                elif cat_val == ISOCategory.CATEGORY_3_TRANSPORT.value:
                    record.category_3_emissions = val
                elif cat_val == ISOCategory.CATEGORY_4_PRODUCTS_USED.value:
                    record.category_4_emissions = val
                elif cat_val == ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG.value:
                    record.category_5_emissions = val
                elif cat_val == ISOCategory.CATEGORY_6_OTHER.value:
                    record.category_6_emissions = val

        record.updated_at = _now()

        logger.info(
            "Recalculated base year for org '%s': %.2f -> %.2f (%.2f%%)",
            org_id, reference, new_total, recalc.change_pct,
        )
        return recalc

    def get_recalculation_history(self, org_id: str) -> List[Recalculation]:
        """Get the complete recalculation audit trail."""
        record = self._base_years.get(org_id)
        if record is None:
            return []
        return sorted(record.recalculations, key=lambda r: r.recalculated_at)

    # ------------------------------------------------------------------
    # Year-over-Year
    # ------------------------------------------------------------------

    def calculate_yoy_change(
        self,
        org_id: str,
        current_emissions: Decimal,
    ) -> Dict[str, Any]:
        """Calculate year-over-year change from base year."""
        record = self._base_years.get(org_id)
        if record is None:
            return {"error": "No base year set"}

        reference = record.current_total
        absolute_change = current_emissions - reference
        pct_change = Decimal("0")
        if reference != 0:
            pct_change = (absolute_change / reference * 100).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            )

        return {
            "base_year": record.year,
            "base_year_tco2e": str(reference),
            "current_tco2e": str(current_emissions),
            "change_absolute_tco2e": str(absolute_change),
            "change_pct": str(pct_change),
            "direction": "decrease" if absolute_change < 0 else "increase",
        }

    def get_trend_data(
        self,
        org_id: str,
        yearly_emissions: Dict[int, Decimal],
    ) -> List[Dict[str, Any]]:
        """
        Build multi-year trend relative to base year.

        Args:
            org_id: Organization ID.
            yearly_emissions: Dict of year -> total emissions.

        Returns:
            List of yearly trend data points.
        """
        record = self._base_years.get(org_id)
        reference = Decimal("0")
        if record:
            reference = record.current_total

        trends: List[Dict[str, Any]] = []
        for year in sorted(yearly_emissions.keys()):
            total = yearly_emissions[year]
            entry: Dict[str, Any] = {
                "year": year,
                "total_tco2e": str(total),
            }
            if reference > 0:
                change_pct = ((total - reference) / reference * 100).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )
                entry["change_from_base_pct"] = str(change_pct)
            trends.append(entry)

        return trends

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_base_year_summary(self, org_id: str) -> Dict[str, Any]:
        """Get a summary of base year configuration and history."""
        record = self._base_years.get(org_id)
        if record is None:
            return {"message": "No base year set", "org_id": org_id}

        return {
            "org_id": org_id,
            "base_year": record.year,
            "locked": record.locked,
            "original_total_emissions": str(record.total_emissions),
            "current_total": str(record.current_total),
            "total_removals": str(record.total_removals),
            "net_emissions": str(record.net_emissions),
            "recalculation_count": len(record.recalculations),
            "threshold_pct": str(self._threshold),
            "justification": record.justification,
            "by_category": {
                "category_1": str(record.category_1_emissions),
                "category_2": str(record.category_2_emissions),
                "category_3": str(record.category_3_emissions),
                "category_4": str(record.category_4_emissions),
                "category_5": str(record.category_5_emissions),
                "category_6": str(record.category_6_emissions),
            },
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_record_or_raise(self, org_id: str) -> BaseYear:
        """Retrieve base year record or raise ValueError."""
        record = self._base_years.get(org_id)
        if record is None:
            raise ValueError(f"No base year set for organization: {org_id}")
        return record
