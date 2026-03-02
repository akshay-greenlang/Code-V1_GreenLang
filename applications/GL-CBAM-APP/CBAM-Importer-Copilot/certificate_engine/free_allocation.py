# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Free Allocation Engine v1.1

Manages EU ETS free allocation benchmarks and the CBAM phase-out schedule.
Computes free allocation deductions that reduce the gross CBAM certificate
obligation for EU importers.

Per EU CBAM Regulation 2023/956:
  - Article 31: Free allocation to EU producers is phased out as CBAM
    is phased in (2026-2034, declining to zero)
  - Recital 24: CBAM should mirror EU ETS free allocation to prevent
    double protection
  - Annex V: Phase-out schedule percentages

The CBAM free allocation deduction represents the emissions that would
have been covered by free EU ETS allowances if the goods were produced
in the EU. As CBAM phases in and free allocation phases out, the
deduction shrinks year over year.

Phase-out schedule (per Article 31):
    2025: 100% (pre-CBAM definitive period)
    2026: 97.5%   2027: 95.0%   2028: 90.0%
    2029: 77.5%   2030: 51.5%   2031: 39.0%
    2032: 26.5%   2033: 14.0%   2034+: 0.0%

Product benchmarks are based on EU ETS Benchmark Decision 2021/927:
    Cement clinker:  0.766 tCO2e/t
    Hot metal:       1.328 tCO2e/t
    Aluminium:       1.514 tCO2e/t
    Ammonia:         1.619 tCO2e/t
    Hydrogen:        8.850 tCO2e/t

All calculations use Decimal with ROUND_HALF_UP (ZERO HALLUCINATION).

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import logging
import threading
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .models import (
    AllocationPhase,
    FreeAllocationFactor,
    PHASE_OUT_SCHEDULE,
    PRODUCT_BENCHMARKS,
    quantize_decimal,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ALLOCATION PHASE MAPPING
# ============================================================================

_YEAR_TO_PHASE: Dict[int, AllocationPhase] = {
    2025: AllocationPhase.PHASE1_100PCT,
    2026: AllocationPhase.PHASE2_97_5PCT,
    2027: AllocationPhase.PHASE3_95PCT,
    2028: AllocationPhase.PHASE4_90PCT,
    2029: AllocationPhase.PHASE5_77_5PCT,
    2030: AllocationPhase.PHASE6_51_5PCT,
    2031: AllocationPhase.PHASE7_39PCT,
    2032: AllocationPhase.PHASE8_26_5PCT,
    2033: AllocationPhase.PHASE9_14PCT,
}


def _get_phase_for_year(year: int) -> AllocationPhase:
    """Get the allocation phase for a given year."""
    if year <= 2025:
        return AllocationPhase.PHASE1_100PCT
    if year >= 2034:
        return AllocationPhase.PHASE_OUT_0PCT
    return _YEAR_TO_PHASE.get(year, AllocationPhase.PHASE_OUT_0PCT)


def _get_allocation_pct_for_year(year: int) -> Decimal:
    """Get the free allocation percentage for a given year."""
    if year <= 2025:
        return Decimal("100")
    if year >= 2034:
        return Decimal("0")
    return PHASE_OUT_SCHEDULE.get(year, Decimal("0"))


class FreeAllocationEngine:
    """
    Engine for computing EU ETS free allocation deductions for CBAM.

    Manages product benchmarks and the declining free allocation schedule.
    Computes the deduction that reduces the gross CBAM certificate obligation.

    Thread Safety:
        Uses threading.RLock for mutable state access.

    Example:
        >>> engine = FreeAllocationEngine()
        >>> factor = engine.get_allocation_factor("25231000", 2026)
        >>> print(f"Benchmark: {factor.benchmark_value_tCO2e} tCO2e/t")
        >>> print(f"Allocation: {factor.allocation_percentage}%")
        >>> print(f"Effective: {factor.effective_allocation_tCO2e} tCO2e/t")
        >>>
        >>> deduction = engine.calculate_free_allocation_deduction(
        ...     gross_certs=Decimal("3000"),
        ...     cn_code="25231000",
        ...     year=2026,
        ... )
        >>> print(f"Deduction: {deduction} tCO2e")
    """

    def __init__(self) -> None:
        """Initialize the free allocation engine."""
        self._lock = threading.RLock()
        # cn_code -> {year -> FreeAllocationFactor}
        self._factor_cache: Dict[str, Dict[int, FreeAllocationFactor]] = {}
        # Custom benchmark overrides: cn_code -> {year -> benchmark_value}
        self._custom_benchmarks: Dict[str, Dict[int, Decimal]] = {}
        self._load_default_benchmarks()
        logger.info(
            "FreeAllocationEngine initialized with %d product benchmarks",
            len(PRODUCT_BENCHMARKS),
        )

    def _load_default_benchmarks(self) -> None:
        """Pre-cache default benchmark factors for all known CN codes and years."""
        for cn_code, benchmark_data in PRODUCT_BENCHMARKS.items():
            for year in range(2026, 2035):
                self._build_factor(cn_code, year, benchmark_data)

    def _build_factor(
        self,
        cn_code: str,
        year: int,
        benchmark_data: Dict[str, Any],
    ) -> FreeAllocationFactor:
        """Build and cache a FreeAllocationFactor."""
        phase = _get_phase_for_year(year)
        alloc_pct = _get_allocation_pct_for_year(year)
        benchmark_value = Decimal(str(benchmark_data["benchmark_value"]))

        factor = FreeAllocationFactor(
            product_benchmark=benchmark_data["product_name"],
            benchmark_value_tCO2e=benchmark_value,
            allocation_percentage=alloc_pct,
            phase=phase,
            year=year,
            cn_codes=[cn_code],
        )

        if cn_code not in self._factor_cache:
            self._factor_cache[cn_code] = {}
        self._factor_cache[cn_code][year] = factor

        return factor

    # ========================================================================
    # ALLOCATION FACTOR RETRIEVAL
    # ========================================================================

    def get_allocation_factor(
        self,
        cn_code: str,
        year: int,
    ) -> Optional[FreeAllocationFactor]:
        """
        Get the free allocation factor for a product and year.

        Looks up the EU ETS benchmark value for the CN code and applies
        the phase-out percentage for the given year.

        Args:
            cn_code: Combined Nomenclature code (6-10 digits).
            year: Reference year for the allocation percentage.

        Returns:
            FreeAllocationFactor if a benchmark exists, None otherwise.
        """
        with self._lock:
            # Check cache first
            cached = self._factor_cache.get(cn_code, {}).get(year)
            if cached is not None:
                return cached

            # Check if there is custom benchmark data
            custom = self._custom_benchmarks.get(cn_code, {}).get(year)
            if custom is not None:
                benchmark_data = {
                    "product_name": f"Custom benchmark for {cn_code}",
                    "benchmark_value": custom,
                }
                return self._build_factor(cn_code, year, benchmark_data)

            # Check if the CN code maps to a known product benchmark
            if cn_code in PRODUCT_BENCHMARKS:
                return self._build_factor(cn_code, year, PRODUCT_BENCHMARKS[cn_code])

            # Try matching by prefix (first 6 digits)
            prefix = cn_code[:6]
            for known_cn, benchmark_data in PRODUCT_BENCHMARKS.items():
                if known_cn[:6] == prefix:
                    logger.info(
                        "CN code %s matched to benchmark %s via 6-digit prefix %s",
                        cn_code, known_cn, prefix,
                    )
                    return self._build_factor(cn_code, year, benchmark_data)

            logger.debug(
                "No benchmark found for CN code %s in year %d", cn_code, year
            )
            return None

    # ========================================================================
    # DEDUCTION CALCULATION
    # ========================================================================

    def calculate_free_allocation_deduction(
        self,
        gross_certs: Decimal,
        cn_code: str,
        year: int,
    ) -> Decimal:
        """
        Calculate the free allocation deduction for a product.

        deduction = min(benchmark_value x allocation_pct / 100, gross_certs)

        The deduction represents the portion of emissions that would have
        been covered by free EU ETS allowances if the product were produced
        in the EU.

        This is a ZERO-HALLUCINATION deterministic computation.

        Args:
            gross_certs: Gross certificates for this product (tCO2e).
            cn_code: Combined Nomenclature code.
            year: Reference year for allocation percentage.

        Returns:
            Free allocation deduction in tCO2e.
        """
        factor = self.get_allocation_factor(cn_code, year)

        if factor is None:
            logger.info(
                "No benchmark for CN %s -- free allocation deduction is 0", cn_code
            )
            return Decimal("0")

        # Effective deduction per tonne of product
        effective = factor.effective_allocation_tCO2e

        # The deduction cannot exceed gross certificates
        deduction = min(effective, gross_certs)
        deduction = quantize_decimal(deduction, places=3)

        logger.debug(
            "Free allocation: CN=%s, year=%d, benchmark=%.3f, alloc_pct=%.1f%%, "
            "effective=%.6f, gross=%.3f, deduction=%.3f",
            cn_code, year, factor.benchmark_value_tCO2e,
            factor.allocation_percentage, effective,
            gross_certs, deduction,
        )

        return deduction

    # ========================================================================
    # PHASE-OUT SCHEDULE
    # ========================================================================

    def get_phase_out_schedule(self) -> Dict[int, Dict[str, Any]]:
        """
        Get the full free allocation phase-out schedule.

        Returns the declining allocation percentage from 2026 to 2034,
        along with phase identifiers and labels.

        Returns:
            Dict mapping year to allocation details.
        """
        schedule = {}
        for year in range(2025, 2036):
            phase = _get_phase_for_year(year)
            alloc_pct = _get_allocation_pct_for_year(year)
            cbam_pct = Decimal("100") - alloc_pct

            schedule[year] = {
                "year": year,
                "free_allocation_pct": str(alloc_pct),
                "cbam_coverage_pct": str(cbam_pct),
                "phase": phase.value,
                "phase_label": phase.label,
            }

        return schedule

    # ========================================================================
    # BENCHMARK VALUES
    # ========================================================================

    def get_benchmark_values(self) -> List[FreeAllocationFactor]:
        """
        Get all product benchmark values for the current year.

        Returns:
            List of FreeAllocationFactor for all known products.
        """
        from datetime import date as date_type
        current_year = date_type.today().year

        results: List[FreeAllocationFactor] = []
        for cn_code in PRODUCT_BENCHMARKS:
            factor = self.get_allocation_factor(cn_code, current_year)
            if factor is not None:
                results.append(factor)

        return results

    # ========================================================================
    # BENCHMARK UPDATES
    # ========================================================================

    def update_benchmark(
        self,
        cn_code: str,
        year: int,
        value: Decimal,
    ) -> FreeAllocationFactor:
        """
        Update or set a custom benchmark value for a product and year.

        This is an admin operation for overriding default benchmarks.

        Args:
            cn_code: Combined Nomenclature code.
            year: Reference year.
            value: New benchmark value in tCO2e per tonne.

        Returns:
            Updated FreeAllocationFactor.
        """
        with self._lock:
            if cn_code not in self._custom_benchmarks:
                self._custom_benchmarks[cn_code] = {}
            self._custom_benchmarks[cn_code][year] = Decimal(str(value))

            # Invalidate cache for this cn_code/year
            if cn_code in self._factor_cache and year in self._factor_cache[cn_code]:
                del self._factor_cache[cn_code][year]

        benchmark_data = {
            "product_name": f"Custom benchmark for {cn_code}",
            "benchmark_value": value,
        }
        factor = self._build_factor(cn_code, year, benchmark_data)

        logger.info(
            "Benchmark updated: CN=%s, year=%d, value=%.3f tCO2e/t",
            cn_code, year, value,
        )

        return factor

    # ========================================================================
    # FULL PRODUCT LISTING
    # ========================================================================

    def get_all_products(self) -> List[Dict[str, Any]]:
        """
        Get a list of all known CBAM products with benchmark values.

        Returns:
            List of product details dicts.
        """
        products = []
        for cn_code, data in sorted(PRODUCT_BENCHMARKS.items()):
            products.append({
                "cn_code": cn_code,
                "product_name": data["product_name"],
                "benchmark_value_tCO2e": str(data["benchmark_value"]),
                "unit": data["unit"],
                "source": data["source"],
            })
        return products

    # ========================================================================
    # YEAR-OVER-YEAR COMPARISON
    # ========================================================================

    def compare_allocation_years(
        self,
        cn_code: str,
        year_from: int,
        year_to: int,
    ) -> Dict[str, Any]:
        """
        Compare free allocation between two years for a given CN code.

        Useful for showing importers the year-over-year increase in their
        CBAM obligation as free allocation declines.

        Args:
            cn_code: Combined Nomenclature code.
            year_from: Starting year.
            year_to: Ending year.

        Returns:
            Dict with comparison details.
        """
        factor_from = self.get_allocation_factor(cn_code, year_from)
        factor_to = self.get_allocation_factor(cn_code, year_to)

        if factor_from is None or factor_to is None:
            return {
                "cn_code": cn_code,
                "error": "No benchmark available for one or both years",
            }

        alloc_from = factor_from.allocation_percentage
        alloc_to = factor_to.allocation_percentage
        change = alloc_to - alloc_from

        return {
            "cn_code": cn_code,
            "product_name": factor_from.product_benchmark,
            "year_from": year_from,
            "year_to": year_to,
            "allocation_pct_from": str(alloc_from),
            "allocation_pct_to": str(alloc_to),
            "allocation_change_pct": str(change),
            "effective_deduction_from": str(factor_from.effective_allocation_tCO2e),
            "effective_deduction_to": str(factor_to.effective_allocation_tCO2e),
            "cbam_coverage_from": str(Decimal("100") - alloc_from),
            "cbam_coverage_to": str(Decimal("100") - alloc_to),
        }
