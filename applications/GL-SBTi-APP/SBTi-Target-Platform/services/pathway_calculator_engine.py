"""
Pathway Calculator Engine -- SBTi Emissions Reduction Pathway Computation

Implements the SBTi-approved methodologies for calculating emissions reduction
pathways:
  - Absolute Contraction Approach (ACA): 4.2%/yr for 1.5C, 2.5%/yr for WB2C
  - Sectoral Decarbonization Approach (SDA): sector-specific convergence
  - Economic Intensity: revenue-based intensity pathways
  - Physical Intensity: physical output-based intensity pathways
  - Supplier Engagement: Scope 3 supplier coverage pathways
  - FLAG Commodity: per-commodity intensity convergence
  - FLAG Sector: 3.03%/yr absolute reduction

All calculations use deterministic formulas (zero-hallucination).
Formula: target_emissions = base_emissions * (1 - annual_rate)^years

Reference:
    - SBTi Criteria and Recommendations v5.1, Section 4: Methods
    - SBTi Corporate Net-Zero Standard v1.2
    - SBTi FLAG Guidance v1.1
    - SBTi Sector-Specific Guidance

Example:
    >>> engine = PathwayCalculatorEngine(config)
    >>> pathway = engine.calculate_aca_pathway(100000, 2020, 2030, "1.5c")
    >>> len(pathway.milestones)
    11
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    ACA_ANNUAL_RATES,
    FLAG_COMMODITY_BENCHMARKS,
    FLAG_SECTOR_ANNUAL_RATE,
    SECTOR_INTENSITY_PATHWAYS,
    UNCERTAINTY_BAND_WIDTHS,
    FLAGCommodity,
    SBTiAppConfig,
    SBTiSector,
    TargetMethod,
)
from .models import (
    Pathway,
    PathwayMilestone,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pathway Comparison and Uncertainty Data Classes
# ---------------------------------------------------------------------------

class PathwayComparison:
    """Result of comparing multiple calculated pathways."""

    def __init__(
        self,
        pathways: List[Dict[str, Any]],
        most_ambitious: str,
        least_ambitious: str,
        summary: Dict[str, Any],
    ) -> None:
        self.pathways = pathways
        self.most_ambitious = most_ambitious
        self.least_ambitious = least_ambitious
        self.summary = summary


class UncertaintyBands:
    """Uncertainty bands around a pathway at a given confidence level."""

    def __init__(
        self,
        confidence_pct: int,
        upper_band: List[Dict[str, float]],
        lower_band: List[Dict[str, float]],
        band_width_pct: float,
    ) -> None:
        self.confidence_pct = confidence_pct
        self.upper_band = upper_band
        self.lower_band = lower_band
        self.band_width_pct = band_width_pct


class PathwayCalculatorEngine:
    """
    SBTi Pathway Calculator Engine.

    Calculates emissions reduction pathways for all SBTi-approved methods:
    ACA, SDA, economic/physical intensity, supplier engagement, and FLAG.
    Generates annual milestones and supports uncertainty band computation.

    Attributes:
        config: Application configuration.
        _pathways: In-memory store of calculated pathways keyed by pathway ID.
    """

    def __init__(self, config: Optional[SBTiAppConfig] = None) -> None:
        """
        Initialize PathwayCalculatorEngine.

        Args:
            config: Application configuration instance.
        """
        self.config = config or SBTiAppConfig()
        self._pathways: Dict[str, Pathway] = {}
        logger.info("PathwayCalculatorEngine initialized")

    # ------------------------------------------------------------------
    # Absolute Contraction Approach (ACA)
    # ------------------------------------------------------------------

    def calculate_aca_pathway(
        self,
        base_emissions: float,
        base_year: int,
        target_year: int,
        alignment: str = "1.5c",
    ) -> Pathway:
        """
        Calculate an Absolute Contraction Approach (ACA) pathway.

        Uses compound annual reduction:
            emissions(y) = base_emissions * (1 - annual_rate)^(y - base_year)

        Args:
            base_emissions: Base year emissions in tCO2e.
            base_year: Base year (>= 2015).
            target_year: Target year.
            alignment: Temperature alignment ("1.5c", "well_below_2c", "below_2c").

        Returns:
            Pathway with annual milestones.

        Raises:
            ValueError: If alignment is not recognized or parameters invalid.
        """
        start = datetime.utcnow()

        # Resolve annual rate
        rate_map = {"1.5c": 0.042, "well_below_2c": 0.025, "below_2c": 0.015}
        annual_rate = rate_map.get(alignment)
        if annual_rate is None:
            raise ValueError(f"Unknown alignment: {alignment}. Use: {list(rate_map.keys())}")

        if base_emissions <= 0:
            raise ValueError("Base emissions must be positive")
        if target_year <= base_year:
            raise ValueError("Target year must be after base year")

        # Generate milestones
        milestones: List[PathwayMilestone] = []
        cumulative_budget = Decimal("0")

        for year in range(base_year, target_year + 1):
            years_elapsed = year - base_year
            expected = base_emissions * ((1 - annual_rate) ** years_elapsed)
            cumulative_reduction = (1 - expected / base_emissions) * 100

            milestone = PathwayMilestone(
                year=year,
                expected_emissions_tco2e=Decimal(str(round(expected, 2))),
                cumulative_reduction_pct=Decimal(str(round(cumulative_reduction, 2))),
                annual_budget_tco2e=Decimal(str(round(expected, 2))),
            )
            milestones.append(milestone)
            cumulative_budget += Decimal(str(round(expected, 2)))

        target_emissions = base_emissions * ((1 - annual_rate) ** (target_year - base_year))

        pathway = Pathway(
            tenant_id="default",
            org_id="",
            target_id="",
            method=TargetMethod.ABSOLUTE_CONTRACTION,
            base_year=base_year,
            target_year=target_year,
            base_emissions_tco2e=Decimal(str(base_emissions)),
            target_emissions_tco2e=Decimal(str(round(target_emissions, 2))),
            annual_reduction_rate=Decimal(str(annual_rate * 100)),
            milestones=milestones,
            cumulative_budget_tco2e=cumulative_budget,
        )

        self._pathways[pathway.id] = pathway

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "ACA pathway: base=%.0f target_yr=%d rate=%.1f%%/yr "
            "target_emissions=%.0f milestones=%d in %.1f ms",
            base_emissions, target_year, annual_rate * 100,
            target_emissions, len(milestones), elapsed_ms,
        )
        return pathway

    # ------------------------------------------------------------------
    # Sectoral Decarbonization Approach (SDA)
    # ------------------------------------------------------------------

    def calculate_sda_pathway(
        self,
        sector: SBTiSector,
        base_intensity: float,
        base_year: int,
        target_year: int,
        activity_data: Optional[Dict[str, float]] = None,
    ) -> Pathway:
        """
        Calculate a Sectoral Decarbonization Approach (SDA) pathway.

        Interpolates the sector-specific convergence pathway to determine
        company-specific intensity targets.

        Args:
            sector: SBTi sector classification.
            base_intensity: Company base year intensity value.
            base_year: Base year.
            target_year: Target year.
            activity_data: Optional activity data (production volume, etc.).

        Returns:
            Pathway with intensity-based milestones.

        Raises:
            ValueError: If sector has no SDA pathway available.
        """
        start = datetime.utcnow()
        sector_data = SECTOR_INTENSITY_PATHWAYS.get(sector)
        if sector_data is None:
            raise ValueError(f"No SDA pathway available for sector {sector.value}")

        pathway_points = sector_data["pathway_points"]
        unit = sector_data["unit"]

        # Interpolate sector pathway for each year
        milestones: List[PathwayMilestone] = []
        sector_years = sorted(pathway_points.keys())
        base_production = (activity_data or {}).get("production_volume", 1.0)
        cumulative_budget = Decimal("0")

        for year in range(base_year, target_year + 1):
            # Interpolate sector intensity at this year
            sector_intensity = self._interpolate_pathway(sector_years, pathway_points, year)

            # Company intensity converges toward sector pathway
            # Linear interpolation between company base and sector target
            total_years = target_year - base_year
            years_elapsed = year - base_year
            if total_years > 0:
                progress = years_elapsed / total_years
            else:
                progress = 1.0

            company_intensity = base_intensity + (sector_intensity - base_intensity) * progress

            # Compute absolute emissions if production data available
            expected_emissions = company_intensity * base_production
            cumulative_reduction = 0.0
            if base_intensity > 0:
                cumulative_reduction = (1 - company_intensity / base_intensity) * 100

            milestone = PathwayMilestone(
                year=year,
                expected_emissions_tco2e=Decimal(str(round(expected_emissions, 2))),
                expected_intensity_value=Decimal(str(round(company_intensity, 4))),
                cumulative_reduction_pct=Decimal(str(round(max(cumulative_reduction, 0), 2))),
                annual_budget_tco2e=Decimal(str(round(expected_emissions, 2))),
            )
            milestones.append(milestone)
            cumulative_budget += Decimal(str(round(expected_emissions, 2)))

        # Compute overall annual rate from intensity change
        final_intensity = float(milestones[-1].expected_intensity_value or base_intensity)
        total_years = target_year - base_year
        if base_intensity > 0 and total_years > 0:
            total_reduction_pct = (1 - final_intensity / base_intensity) * 100
            annual_rate_pct = total_reduction_pct / total_years
        else:
            annual_rate_pct = 0.0

        pathway = Pathway(
            tenant_id="default",
            org_id="",
            target_id="",
            method=TargetMethod.SECTORAL_DECARBONIZATION,
            base_year=base_year,
            target_year=target_year,
            base_emissions_tco2e=Decimal(str(round(base_intensity * base_production, 2))),
            target_emissions_tco2e=Decimal(str(round(final_intensity * base_production, 2))),
            annual_reduction_rate=Decimal(str(round(annual_rate_pct, 2))),
            milestones=milestones,
            cumulative_budget_tco2e=cumulative_budget,
            sector=sector,
        )

        self._pathways[pathway.id] = pathway

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "SDA pathway: sector=%s base_intensity=%.4f target_yr=%d "
            "final_intensity=%.4f rate=%.2f%%/yr in %.1f ms",
            sector.value, base_intensity, target_year,
            final_intensity, annual_rate_pct, elapsed_ms,
        )
        return pathway

    # ------------------------------------------------------------------
    # Economic Intensity Pathway
    # ------------------------------------------------------------------

    def calculate_economic_intensity_pathway(
        self,
        base_intensity: float,
        base_year: int,
        target_year: int,
        growth_rate: float = 0.03,
    ) -> Pathway:
        """
        Calculate an economic intensity reduction pathway.

        Intensity is tCO2e per unit of economic output (revenue, GDP, etc.).
        The pathway accounts for expected growth in economic activity.

        Args:
            base_intensity: Base year intensity (tCO2e/$M).
            base_year: Base year.
            target_year: Target year.
            growth_rate: Expected annual economic growth rate (default 3%).

        Returns:
            Pathway with intensity milestones.
        """
        start = datetime.utcnow()
        total_years = target_year - base_year
        if total_years <= 0:
            raise ValueError("Target year must be after base year")

        # For 1.5C alignment, absolute reduction must be >= 4.2%/yr
        # With growth, intensity must decrease faster:
        # intensity_rate = absolute_rate + growth_rate
        absolute_rate = 0.042
        intensity_rate = absolute_rate + growth_rate

        milestones: List[PathwayMilestone] = []
        cumulative_budget = Decimal("0")

        for year in range(base_year, target_year + 1):
            years_elapsed = year - base_year
            expected_intensity = base_intensity * ((1 - intensity_rate) ** years_elapsed)
            expected_intensity = max(expected_intensity, 0)

            # Estimate absolute emissions using growth
            economic_output = 1.0 * ((1 + growth_rate) ** years_elapsed)
            expected_emissions = expected_intensity * economic_output

            cumulative_reduction = 0.0
            if base_intensity > 0:
                cumulative_reduction = (1 - expected_intensity / base_intensity) * 100

            milestone = PathwayMilestone(
                year=year,
                expected_emissions_tco2e=Decimal(str(round(expected_emissions, 2))),
                expected_intensity_value=Decimal(str(round(expected_intensity, 4))),
                cumulative_reduction_pct=Decimal(str(round(max(cumulative_reduction, 0), 2))),
                annual_budget_tco2e=Decimal(str(round(expected_emissions, 2))),
            )
            milestones.append(milestone)
            cumulative_budget += Decimal(str(round(expected_emissions, 2)))

        final_intensity = base_intensity * ((1 - intensity_rate) ** total_years)

        pathway = Pathway(
            tenant_id="default",
            org_id="",
            target_id="",
            method=TargetMethod.ECONOMIC_INTENSITY,
            base_year=base_year,
            target_year=target_year,
            base_emissions_tco2e=Decimal(str(round(base_intensity, 2))),
            target_emissions_tco2e=Decimal(str(round(max(final_intensity, 0), 2))),
            annual_reduction_rate=Decimal(str(round(intensity_rate * 100, 2))),
            milestones=milestones,
            cumulative_budget_tco2e=cumulative_budget,
        )

        self._pathways[pathway.id] = pathway

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Economic intensity pathway: base=%.4f growth=%.1f%% rate=%.1f%%/yr in %.1f ms",
            base_intensity, growth_rate * 100, intensity_rate * 100, elapsed_ms,
        )
        return pathway

    # ------------------------------------------------------------------
    # Physical Intensity Pathway
    # ------------------------------------------------------------------

    def calculate_physical_intensity_pathway(
        self,
        sector: SBTiSector,
        base_intensity: float,
        base_year: int,
        target_year: int,
    ) -> Pathway:
        """
        Calculate a physical intensity reduction pathway.

        Uses sector-specific intensity convergence pathways (SDA).
        Physical intensity is tCO2e per physical unit (tonne, MWh, m2, etc.).

        Args:
            sector: SBTi sector.
            base_intensity: Company base year physical intensity.
            base_year: Base year.
            target_year: Target year.

        Returns:
            Pathway with physical intensity milestones.
        """
        # Delegate to SDA since physical intensity uses sector convergence
        return self.calculate_sda_pathway(
            sector=sector,
            base_intensity=base_intensity,
            base_year=base_year,
            target_year=target_year,
        )

    # ------------------------------------------------------------------
    # Supplier Engagement Pathway
    # ------------------------------------------------------------------

    def calculate_supplier_engagement_pathway(
        self,
        current_pct: float,
        target_pct: float,
        years: int,
    ) -> Pathway:
        """
        Calculate a supplier engagement coverage pathway.

        Linear increase in percentage of suppliers with SBTi targets.

        Args:
            current_pct: Current % of suppliers (by emissions) with SBTi targets.
            target_pct: Target % of suppliers with SBTi targets.
            years: Timeframe in years.

        Returns:
            Pathway showing annual supplier engagement milestones.
        """
        start = datetime.utcnow()
        if years <= 0:
            raise ValueError("Years must be positive")

        annual_increase = (target_pct - current_pct) / years
        base_year = self.config.reporting_year
        target_year = base_year + years

        milestones: List[PathwayMilestone] = []
        for year in range(base_year, target_year + 1):
            years_elapsed = year - base_year
            expected_pct = current_pct + annual_increase * years_elapsed
            expected_pct = min(expected_pct, 100.0)

            milestone = PathwayMilestone(
                year=year,
                expected_emissions_tco2e=Decimal("0"),
                expected_intensity_value=Decimal(str(round(expected_pct, 2))),
                cumulative_reduction_pct=Decimal(str(round(expected_pct - current_pct, 2))),
            )
            milestones.append(milestone)

        pathway = Pathway(
            tenant_id="default",
            org_id="",
            target_id="",
            method=TargetMethod.SUPPLIER_ENGAGEMENT,
            base_year=base_year,
            target_year=target_year,
            base_emissions_tco2e=Decimal(str(round(current_pct, 2))),
            target_emissions_tco2e=Decimal(str(round(target_pct, 2))),
            annual_reduction_rate=Decimal(str(round(annual_increase, 2))),
            milestones=milestones,
        )

        self._pathways[pathway.id] = pathway

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Supplier engagement pathway: %.0f%% -> %.0f%% over %d years (%.1f%%/yr) in %.1f ms",
            current_pct, target_pct, years, annual_increase, elapsed_ms,
        )
        return pathway

    # ------------------------------------------------------------------
    # FLAG Commodity Pathway
    # ------------------------------------------------------------------

    def calculate_flag_commodity_pathway(
        self,
        commodity: FLAGCommodity,
        base_data: Dict[str, float],
    ) -> Pathway:
        """
        Calculate a FLAG commodity-specific intensity pathway.

        Uses SBTi FLAG benchmark intensities for each commodity
        to establish a convergence pathway.

        Args:
            commodity: FLAG commodity type.
            base_data: Dict with keys "base_intensity", "base_year",
                       "target_year", "production_volume".

        Returns:
            Pathway with commodity intensity milestones.

        Raises:
            ValueError: If commodity benchmarks not found.
        """
        start = datetime.utcnow()
        benchmarks = FLAG_COMMODITY_BENCHMARKS.get(commodity)
        if benchmarks is None:
            raise ValueError(f"No FLAG benchmarks for commodity {commodity.value}")

        base_intensity = base_data.get("base_intensity", benchmarks["base_intensity"])
        base_year = int(base_data.get("base_year", 2020))
        target_year = int(base_data.get("target_year", 2030))
        production = base_data.get("production_volume", 1.0)

        target_2030 = benchmarks["target_2030"]
        target_2050 = benchmarks["target_2050"]

        milestones: List[PathwayMilestone] = []
        cumulative_budget = Decimal("0")
        total_years = target_year - base_year

        for year in range(base_year, target_year + 1):
            years_elapsed = year - base_year
            # Linear interpolation toward benchmark
            if total_years > 0:
                progress = years_elapsed / total_years
            else:
                progress = 1.0

            # Determine benchmark target for this year
            if target_year <= 2030:
                target_intensity = target_2030
            elif target_year <= 2050:
                # Interpolate between 2030 and 2050 benchmarks
                years_to_2050 = 2050 - 2030
                years_past_2030 = min(target_year - 2030, years_to_2050)
                target_intensity = target_2030 + (target_2050 - target_2030) * (years_past_2030 / years_to_2050)
            else:
                target_intensity = target_2050

            expected_intensity = base_intensity + (target_intensity - base_intensity) * progress
            expected_emissions = expected_intensity * production

            cumulative_reduction = 0.0
            if base_intensity > 0:
                cumulative_reduction = (1 - expected_intensity / base_intensity) * 100

            milestone = PathwayMilestone(
                year=year,
                expected_emissions_tco2e=Decimal(str(round(expected_emissions, 2))),
                expected_intensity_value=Decimal(str(round(expected_intensity, 4))),
                cumulative_reduction_pct=Decimal(str(round(max(cumulative_reduction, 0), 2))),
            )
            milestones.append(milestone)
            cumulative_budget += Decimal(str(round(expected_emissions, 2)))

        final_intensity = float(milestones[-1].expected_intensity_value or 0)
        annual_rate = 0.0
        if base_intensity > 0 and total_years > 0:
            annual_rate = ((1 - final_intensity / base_intensity) * 100) / total_years

        pathway = Pathway(
            tenant_id="default",
            org_id="",
            target_id="",
            method=TargetMethod.FLAG_COMMODITY,
            base_year=base_year,
            target_year=target_year,
            base_emissions_tco2e=Decimal(str(round(base_intensity * production, 2))),
            target_emissions_tco2e=Decimal(str(round(final_intensity * production, 2))),
            annual_reduction_rate=Decimal(str(round(annual_rate, 2))),
            milestones=milestones,
            cumulative_budget_tco2e=cumulative_budget,
        )

        self._pathways[pathway.id] = pathway

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "FLAG commodity pathway: %s base=%.2f target=%.2f rate=%.2f%%/yr in %.1f ms",
            commodity.value, base_intensity, final_intensity, annual_rate, elapsed_ms,
        )
        return pathway

    # ------------------------------------------------------------------
    # FLAG Sector Pathway
    # ------------------------------------------------------------------

    def calculate_flag_sector_pathway(
        self,
        base_emissions: float,
        base_year: int,
        target_year: int,
    ) -> Pathway:
        """
        Calculate a FLAG sector-level absolute reduction pathway.

        Uses the SBTi FLAG sector annual reduction rate of 3.03%/yr.

        Args:
            base_emissions: Base year FLAG emissions in tCO2e.
            base_year: Base year.
            target_year: Target year.

        Returns:
            Pathway with annual milestones at 3.03%/yr reduction.
        """
        start = datetime.utcnow()
        rate = FLAG_SECTOR_ANNUAL_RATE

        milestones: List[PathwayMilestone] = []
        cumulative_budget = Decimal("0")

        for year in range(base_year, target_year + 1):
            years_elapsed = year - base_year
            expected = base_emissions * ((1 - rate) ** years_elapsed)
            cumulative_reduction = (1 - expected / base_emissions) * 100 if base_emissions > 0 else 0

            milestone = PathwayMilestone(
                year=year,
                expected_emissions_tco2e=Decimal(str(round(expected, 2))),
                cumulative_reduction_pct=Decimal(str(round(cumulative_reduction, 2))),
                annual_budget_tco2e=Decimal(str(round(expected, 2))),
            )
            milestones.append(milestone)
            cumulative_budget += Decimal(str(round(expected, 2)))

        target_emissions = base_emissions * ((1 - rate) ** (target_year - base_year))

        pathway = Pathway(
            tenant_id="default",
            org_id="",
            target_id="",
            method=TargetMethod.FLAG_SECTOR,
            base_year=base_year,
            target_year=target_year,
            base_emissions_tco2e=Decimal(str(base_emissions)),
            target_emissions_tco2e=Decimal(str(round(target_emissions, 2))),
            annual_reduction_rate=Decimal(str(round(rate * 100, 2))),
            milestones=milestones,
            cumulative_budget_tco2e=cumulative_budget,
        )

        self._pathways[pathway.id] = pathway

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "FLAG sector pathway: base=%.0f rate=%.2f%%/yr target=%.0f in %.1f ms",
            base_emissions, rate * 100, target_emissions, elapsed_ms,
        )
        return pathway

    # ------------------------------------------------------------------
    # Pathway Comparison
    # ------------------------------------------------------------------

    def compare_pathways(self, pathways: List[Pathway]) -> PathwayComparison:
        """
        Compare multiple calculated pathways side by side.

        Args:
            pathways: List of Pathway instances to compare.

        Returns:
            PathwayComparison with ranked results.
        """
        if not pathways:
            return PathwayComparison(
                pathways=[], most_ambitious="", least_ambitious="",
                summary={"count": 0},
            )

        pathway_data: List[Dict[str, Any]] = []
        for p in pathways:
            total_reduction = 0.0
            if float(p.base_emissions_tco2e) > 0:
                total_reduction = (
                    1 - float(p.target_emissions_tco2e) / float(p.base_emissions_tco2e)
                ) * 100

            pathway_data.append({
                "id": p.id,
                "method": p.method.value,
                "annual_rate_pct": float(p.annual_reduction_rate),
                "total_reduction_pct": round(total_reduction, 2),
                "target_emissions": float(p.target_emissions_tco2e),
                "cumulative_budget": float(p.cumulative_budget_tco2e or 0),
                "milestones_count": len(p.milestones),
            })

        # Sort by total reduction (descending = most ambitious first)
        pathway_data.sort(key=lambda x: x["total_reduction_pct"], reverse=True)

        most_ambitious = pathway_data[0]["id"]
        least_ambitious = pathway_data[-1]["id"]

        avg_rate = sum(d["annual_rate_pct"] for d in pathway_data) / len(pathway_data)
        avg_reduction = sum(d["total_reduction_pct"] for d in pathway_data) / len(pathway_data)

        summary = {
            "count": len(pathway_data),
            "avg_annual_rate_pct": round(avg_rate, 2),
            "avg_total_reduction_pct": round(avg_reduction, 2),
            "most_ambitious_method": pathway_data[0]["method"],
            "least_ambitious_method": pathway_data[-1]["method"],
        }

        logger.info(
            "Compared %d pathways: most_ambitious=%s least=%s",
            len(pathways), most_ambitious, least_ambitious,
        )
        return PathwayComparison(
            pathways=pathway_data,
            most_ambitious=most_ambitious,
            least_ambitious=least_ambitious,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Milestone Generation
    # ------------------------------------------------------------------

    def generate_milestones(self, pathway: Pathway) -> List[PathwayMilestone]:
        """
        Generate or retrieve milestones for a pathway.

        If milestones already exist on the pathway, returns them.
        Otherwise recalculates based on the pathway parameters.

        Args:
            pathway: Pathway instance.

        Returns:
            List of PathwayMilestone instances.
        """
        if pathway.milestones:
            return pathway.milestones

        # Recalculate based on pathway parameters
        rate = float(pathway.annual_reduction_rate) / 100
        base = float(pathway.base_emissions_tco2e)

        milestones: List[PathwayMilestone] = []
        for year in range(pathway.base_year, pathway.target_year + 1):
            years_elapsed = year - pathway.base_year
            expected = base * ((1 - rate) ** years_elapsed)
            cumulative_reduction = (1 - expected / base) * 100 if base > 0 else 0

            milestones.append(PathwayMilestone(
                year=year,
                expected_emissions_tco2e=Decimal(str(round(expected, 2))),
                cumulative_reduction_pct=Decimal(str(round(cumulative_reduction, 2))),
                annual_budget_tco2e=Decimal(str(round(expected, 2))),
            ))

        return milestones

    # ------------------------------------------------------------------
    # Uncertainty Bands
    # ------------------------------------------------------------------

    def calculate_uncertainty_bands(
        self,
        pathway: Pathway,
        confidence: int = 90,
    ) -> UncertaintyBands:
        """
        Calculate uncertainty bands around a pathway.

        Uses configurable band widths as a percentage of the pathway
        value at each milestone year.

        Args:
            pathway: Pathway to compute bands for.
            confidence: Confidence level (50, 60, 70, 80, or 90%).

        Returns:
            UncertaintyBands with upper and lower band values.
        """
        band_width = UNCERTAINTY_BAND_WIDTHS.get(str(confidence), 0.20)

        upper_band: List[Dict[str, float]] = []
        lower_band: List[Dict[str, float]] = []

        for milestone in pathway.milestones:
            value = float(milestone.expected_emissions_tco2e)
            delta = value * band_width
            upper_band.append({"year": milestone.year, "value": round(value + delta, 2)})
            lower_band.append({"year": milestone.year, "value": round(max(value - delta, 0), 2)})

        logger.info(
            "Uncertainty bands: confidence=%d%% width=%.0f%% milestones=%d",
            confidence, band_width * 100, len(pathway.milestones),
        )
        return UncertaintyBands(
            confidence_pct=confidence,
            upper_band=upper_band,
            lower_band=lower_band,
            band_width_pct=band_width * 100,
        )

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate_pathway(
        years: List[int],
        points: Dict[int, float],
        target_year: int,
    ) -> float:
        """
        Linearly interpolate a pathway value for a given year.

        Args:
            years: Sorted list of pathway years.
            points: Year -> value mapping.
            target_year: Year to interpolate.

        Returns:
            Interpolated value.
        """
        if target_year in points:
            return points[target_year]

        if target_year <= years[0]:
            return points[years[0]]
        if target_year >= years[-1]:
            return points[years[-1]]

        # Find bounding years
        for i in range(len(years) - 1):
            if years[i] <= target_year <= years[i + 1]:
                y0, y1 = years[i], years[i + 1]
                v0, v1 = points[y0], points[y1]
                fraction = (target_year - y0) / (y1 - y0)
                return v0 + (v1 - v0) * fraction

        return points[years[-1]]
