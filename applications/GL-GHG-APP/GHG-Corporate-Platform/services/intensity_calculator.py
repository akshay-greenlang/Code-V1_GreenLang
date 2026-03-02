"""
Intensity Calculator -- GHG Protocol Chapter 12 Implementation

Calculates GHG intensity metrics (emissions per unit of activity) to
allow meaningful comparison across time periods, entities, and sectors.

Supported denominator types:
  - Revenue (tCO2e per million USD)
  - Employees (tCO2e per FTE)
  - Production units (tCO2e per unit)
  - Floor area (tCO2e per m2)
  - Custom denominators

Includes sector benchmark data for major industries.

Example:
    >>> calc = IntensityCalculator(config)
    >>> metric = calc.calculate_revenue_intensity(Decimal("10000"), Decimal("50"))
    >>> print(metric.intensity_value)  # 200.0 tCO2e/M USD
"""

from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    GHGAppConfig,
    IntensityDenominator,
    Scope,
    SECTOR_BENCHMARKS,
)
from .models import (
    GHGInventory,
    IntensityMetric,
    _new_id,
    _now,
)

logger = logging.getLogger(__name__)


class IntensityCalculator:
    """
    Calculates GHG intensity metrics per GHG Protocol Ch 12.

    Intensity metrics normalize emissions against business metrics,
    enabling performance tracking independent of absolute growth.
    """

    def __init__(
        self,
        config: Optional[GHGAppConfig] = None,
        inventory_store: Optional[Dict[str, GHGInventory]] = None,
    ) -> None:
        """
        Initialize IntensityCalculator.

        Args:
            config: Application configuration.
            inventory_store: Shared reference to inventory storage.
        """
        self.config = config or GHGAppConfig()
        self._inventory_store = inventory_store if inventory_store is not None else {}
        logger.info("IntensityCalculator initialized")

    # ------------------------------------------------------------------
    # Core Intensity Calculations
    # ------------------------------------------------------------------

    def calculate_revenue_intensity(
        self,
        total_tco2e: Decimal,
        revenue_million_usd: Decimal,
        scope: Optional[Scope] = None,
    ) -> IntensityMetric:
        """
        Calculate revenue-based intensity (tCO2e per million USD).

        Args:
            total_tco2e: Total emissions in tCO2e.
            revenue_million_usd: Revenue in million USD.
            scope: Optional scope filter.

        Returns:
            IntensityMetric with revenue intensity.

        Raises:
            ValueError: If revenue is zero or negative.
        """
        self._validate_positive(revenue_million_usd, "revenue")
        intensity = self._safe_divide(total_tco2e, revenue_million_usd)

        metric = IntensityMetric(
            denominator=IntensityDenominator.REVENUE,
            denominator_value=revenue_million_usd,
            denominator_unit="million USD",
            intensity_value=intensity,
            scope=scope,
            total_tco2e=total_tco2e,
            unit="tCO2e/M USD",
        )

        logger.info(
            "Revenue intensity: %.2f tCO2e / %.2f M USD = %.4f tCO2e/M USD",
            total_tco2e,
            revenue_million_usd,
            intensity,
        )
        return metric

    def calculate_employee_intensity(
        self,
        total_tco2e: Decimal,
        fte_count: int,
        scope: Optional[Scope] = None,
    ) -> IntensityMetric:
        """
        Calculate employee-based intensity (tCO2e per FTE).

        Args:
            total_tco2e: Total emissions in tCO2e.
            fte_count: Number of full-time equivalents.
            scope: Optional scope filter.

        Returns:
            IntensityMetric with employee intensity.

        Raises:
            ValueError: If FTE count is zero or negative.
        """
        if fte_count <= 0:
            raise ValueError(f"FTE count must be positive, got {fte_count}")

        fte_decimal = Decimal(str(fte_count))
        intensity = self._safe_divide(total_tco2e, fte_decimal)

        metric = IntensityMetric(
            denominator=IntensityDenominator.EMPLOYEES,
            denominator_value=fte_decimal,
            denominator_unit="FTE",
            intensity_value=intensity,
            scope=scope,
            total_tco2e=total_tco2e,
            unit="tCO2e/FTE",
        )

        logger.info(
            "Employee intensity: %.2f tCO2e / %d FTE = %.4f tCO2e/FTE",
            total_tco2e,
            fte_count,
            intensity,
        )
        return metric

    def calculate_production_intensity(
        self,
        total_tco2e: Decimal,
        units: Decimal,
        unit_name: str,
        scope: Optional[Scope] = None,
    ) -> IntensityMetric:
        """
        Calculate production-based intensity (tCO2e per production unit).

        Args:
            total_tco2e: Total emissions in tCO2e.
            units: Number of production units.
            unit_name: Name of the production unit (e.g. MWh, tonnes).
            scope: Optional scope filter.

        Returns:
            IntensityMetric with production intensity.
        """
        self._validate_positive(units, "production units")
        intensity = self._safe_divide(total_tco2e, units)

        metric = IntensityMetric(
            denominator=IntensityDenominator.PRODUCTION_UNITS,
            denominator_value=units,
            denominator_unit=unit_name,
            intensity_value=intensity,
            scope=scope,
            total_tco2e=total_tco2e,
            unit=f"tCO2e/{unit_name}",
        )

        logger.info(
            "Production intensity: %.2f tCO2e / %.2f %s = %.4f tCO2e/%s",
            total_tco2e,
            units,
            unit_name,
            intensity,
            unit_name,
        )
        return metric

    def calculate_floor_area_intensity(
        self,
        total_tco2e: Decimal,
        area_m2: Decimal,
        scope: Optional[Scope] = None,
    ) -> IntensityMetric:
        """
        Calculate floor area intensity (tCO2e per m2).

        Args:
            total_tco2e: Total emissions in tCO2e.
            area_m2: Total floor area in square meters.
            scope: Optional scope filter.

        Returns:
            IntensityMetric with floor area intensity.
        """
        self._validate_positive(area_m2, "floor area")
        intensity = self._safe_divide(total_tco2e, area_m2)

        metric = IntensityMetric(
            denominator=IntensityDenominator.FLOOR_AREA,
            denominator_value=area_m2,
            denominator_unit="m2",
            intensity_value=intensity,
            scope=scope,
            total_tco2e=total_tco2e,
            unit="tCO2e/m2",
        )

        logger.info(
            "Floor area intensity: %.2f tCO2e / %.2f m2 = %.6f tCO2e/m2",
            total_tco2e,
            area_m2,
            intensity,
        )
        return metric

    def calculate_custom_intensity(
        self,
        total_tco2e: Decimal,
        denominator_name: str,
        denominator_value: Decimal,
        unit: str,
        scope: Optional[Scope] = None,
    ) -> IntensityMetric:
        """
        Calculate a custom intensity metric.

        Args:
            total_tco2e: Total emissions in tCO2e.
            denominator_name: Description of the denominator.
            denominator_value: Value of the denominator.
            unit: Full unit string (e.g. tCO2e/passenger-km).
            scope: Optional scope filter.

        Returns:
            IntensityMetric with custom intensity.
        """
        self._validate_positive(denominator_value, denominator_name)
        intensity = self._safe_divide(total_tco2e, denominator_value)

        metric = IntensityMetric(
            denominator=IntensityDenominator.CUSTOM,
            denominator_value=denominator_value,
            denominator_unit=denominator_name,
            intensity_value=intensity,
            scope=scope,
            total_tco2e=total_tco2e,
            unit=unit,
        )

        logger.info(
            "Custom intensity (%s): %.2f tCO2e / %.2f = %.4f %s",
            denominator_name,
            total_tco2e,
            denominator_value,
            intensity,
            unit,
        )
        return metric

    # ------------------------------------------------------------------
    # Batch / Inventory-Level Calculations
    # ------------------------------------------------------------------

    def calculate_all_intensities(
        self,
        inventory: GHGInventory,
        org_revenue_million_usd: Optional[Decimal] = None,
        org_fte_count: Optional[int] = None,
        org_floor_area_m2: Optional[Decimal] = None,
        org_production_units: Optional[Decimal] = None,
        org_production_unit_name: Optional[str] = None,
    ) -> List[IntensityMetric]:
        """
        Calculate all applicable intensity metrics for an inventory.

        Only calculates metrics where denominator data is available.

        Args:
            inventory: GHG inventory with populated emissions.
            org_revenue_million_usd: Organization revenue.
            org_fte_count: Organization FTE count.
            org_floor_area_m2: Total floor area.
            org_production_units: Production volume.
            org_production_unit_name: Production unit name.

        Returns:
            List of calculated IntensityMetrics.
        """
        metrics: List[IntensityMetric] = []
        total = inventory.grand_total_tco2e

        if total <= 0:
            logger.warning("Inventory %s has zero emissions; no intensity metrics", inventory.id)
            return metrics

        if org_revenue_million_usd and org_revenue_million_usd > 0:
            metrics.append(
                self.calculate_revenue_intensity(total, org_revenue_million_usd)
            )

        if org_fte_count and org_fte_count > 0:
            metrics.append(
                self.calculate_employee_intensity(total, org_fte_count)
            )

        if org_floor_area_m2 and org_floor_area_m2 > 0:
            metrics.append(
                self.calculate_floor_area_intensity(total, org_floor_area_m2)
            )

        if (
            org_production_units
            and org_production_units > 0
            and org_production_unit_name
        ):
            metrics.append(
                self.calculate_production_intensity(
                    total,
                    org_production_units,
                    org_production_unit_name,
                )
            )

        # Scope-specific intensities (revenue only for brevity)
        if org_revenue_million_usd and org_revenue_million_usd > 0:
            scope_pairs = [
                (Scope.SCOPE_1, inventory.scope1),
                (Scope.SCOPE_2_MARKET, inventory.scope2_market),
                (Scope.SCOPE_3, inventory.scope3),
            ]
            for scope, scope_data in scope_pairs:
                if scope_data and scope_data.total_tco2e > 0:
                    metrics.append(
                        self.calculate_revenue_intensity(
                            scope_data.total_tco2e,
                            org_revenue_million_usd,
                            scope=scope,
                        )
                    )

        logger.info(
            "Calculated %d intensity metrics for inventory %s",
            len(metrics),
            inventory.id,
        )
        return metrics

    # ------------------------------------------------------------------
    # Year-over-Year Comparison
    # ------------------------------------------------------------------

    def compare_yoy_intensity(
        self,
        org_id: str,
        metric_type: IntensityDenominator,
        year1: int,
        year2: int,
    ) -> Dict[str, Any]:
        """
        Compare intensity metrics between two years.

        Args:
            org_id: Organization ID.
            metric_type: Denominator type to compare.
            year1: First year.
            year2: Second year.

        Returns:
            Dict with comparison results.
        """
        inv1 = self._find_inventory(org_id, year1)
        inv2 = self._find_inventory(org_id, year2)

        if inv1 is None or inv2 is None:
            missing = []
            if inv1 is None:
                missing.append(str(year1))
            if inv2 is None:
                missing.append(str(year2))
            return {
                "error": f"Missing inventory for year(s): {', '.join(missing)}",
            }

        metrics1 = [
            m for m in inv1.intensity_metrics
            if m.denominator == metric_type
        ]
        metrics2 = [
            m for m in inv2.intensity_metrics
            if m.denominator == metric_type
        ]

        if not metrics1 or not metrics2:
            return {
                "error": f"No {metric_type.value} intensity metrics found for comparison",
            }

        m1 = metrics1[0]
        m2 = metrics2[0]
        change = m2.intensity_value - m1.intensity_value
        change_pct = Decimal("0")
        if m1.intensity_value > 0:
            change_pct = (change / m1.intensity_value * 100).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        return {
            "metric_type": metric_type.value,
            "year1": year1,
            "year1_intensity": str(m1.intensity_value),
            "year1_unit": m1.unit,
            "year2": year2,
            "year2_intensity": str(m2.intensity_value),
            "year2_unit": m2.unit,
            "change_absolute": str(change),
            "change_pct": str(change_pct),
            "direction": "improved" if change < 0 else "worsened",
        }

    # ------------------------------------------------------------------
    # Sector Benchmarks
    # ------------------------------------------------------------------

    def get_sector_benchmark(
        self,
        industry: str,
        metric_type: str,
    ) -> Optional[Decimal]:
        """
        Get sector benchmark intensity value.

        Args:
            industry: Industry sector (e.g. energy, manufacturing).
            metric_type: Benchmark type (revenue, employees).

        Returns:
            Benchmark value or None if not available.
        """
        industry_lower = industry.strip().lower()
        benchmarks = SECTOR_BENCHMARKS.get(industry_lower)
        if benchmarks is None:
            logger.warning("No benchmarks available for industry: %s", industry)
            return None

        value = benchmarks.get(metric_type.strip().lower())
        if value is None:
            logger.warning(
                "No %s benchmark for industry: %s",
                metric_type,
                industry,
            )
            return None

        return value

    def compare_to_benchmark(
        self,
        industry: str,
        metric_type: str,
        actual_intensity: Decimal,
    ) -> Dict[str, Any]:
        """
        Compare actual intensity against sector benchmark.

        Args:
            industry: Industry sector.
            metric_type: Benchmark type.
            actual_intensity: Actual intensity value.

        Returns:
            Dict with benchmark comparison.
        """
        benchmark = self.get_sector_benchmark(industry, metric_type)
        if benchmark is None:
            return {
                "error": f"No benchmark available for {industry}/{metric_type}",
            }

        diff = actual_intensity - benchmark
        diff_pct = Decimal("0")
        if benchmark > 0:
            diff_pct = (diff / benchmark * 100).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        return {
            "industry": industry,
            "metric_type": metric_type,
            "actual": str(actual_intensity),
            "benchmark": str(benchmark),
            "difference": str(diff),
            "difference_pct": str(diff_pct),
            "above_benchmark": diff > 0,
            "rating": self._intensity_rating(diff_pct),
        }

    def get_all_benchmarks(self) -> Dict[str, Dict[str, str]]:
        """Return all available sector benchmarks."""
        result: Dict[str, Dict[str, str]] = {}
        for industry, metrics in SECTOR_BENCHMARKS.items():
            result[industry] = {k: str(v) for k, v in metrics.items()}
        return result

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
        """Perform safe division with rounding."""
        if denominator == 0:
            return Decimal("0")
        return (numerator / denominator).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

    @staticmethod
    def _validate_positive(value: Decimal, name: str) -> None:
        """Validate that a value is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    @staticmethod
    def _intensity_rating(diff_pct: Decimal) -> str:
        """Rate intensity relative to benchmark."""
        if diff_pct <= Decimal("-25"):
            return "excellent"
        if diff_pct <= Decimal("-10"):
            return "good"
        if diff_pct <= Decimal("10"):
            return "average"
        if diff_pct <= Decimal("25"):
            return "below_average"
        return "poor"

    def _find_inventory(self, org_id: str, year: int) -> Optional[GHGInventory]:
        """Find inventory for org-year in the shared store."""
        for inv in self._inventory_store.values():
            if inv.org_id == org_id and inv.year == year:
                return inv
        return None
