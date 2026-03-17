"""
PACK-013 CSRD Manufacturing Pack - EU ETS Bridge.

Integration with the EU Emissions Trading System for installations
covered by EU ETS.  Calculates compliance obligations, free-allocation
shortfalls, benchmark comparisons, and estimated compliance costs.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants: ETS product benchmarks (tCO2e per tonne of product)
# Updated to Phase IV (2021-2030) values
# ---------------------------------------------------------------------------

PRODUCT_BENCHMARKS: Dict[str, float] = {
    # Metals
    "hot_metal": 1.328,
    "sintered_ore": 0.171,
    "iron_casting": 0.325,
    "eaf_carbon_steel": 0.283,
    "eaf_high_alloy_steel": 0.352,
    "coke": 0.286,
    # Cement and lime
    "grey_clinker": 0.766,
    "white_clinker": 0.987,
    "lime": 0.954,
    "dolite": 1.072,
    # Glass
    "float_glass": 0.453,
    "container_glass": 0.382,
    # Ceramics
    "bricks": 0.139,
    "roof_tiles": 0.144,
    # Chemicals
    "ammonia": 1.619,
    "nitric_acid": 0.302,
    "adipic_acid": 2.790,
    "hydrogen": 8.850,
    "synthesis_gas": 0.242,
    "ethylene_oxide": 0.512,
    "vinyl_chloride": 0.204,
    "styrene": 0.527,
    "phenol": 0.038,
    "soda_ash": 0.843,
    "carbon_black": 1.954,
    # Paper and pulp
    "newsprint": 0.298,
    "uncoated_fine_paper": 0.318,
    "coated_fine_paper": 0.318,
    "tissue": 0.334,
    "testliner": 0.248,
    "short_fibre_kraft_pulp": 0.120,
    "long_fibre_kraft_pulp": 0.120,
    # Refining
    "refinery_products": 0.0295,  # CWT-based
    # Aluminium
    "primary_aluminium": 1.514,
    "pre_bake_anode": 0.324,
}

# Free allocation reduction factor (linear reduction from 2021)
FREE_ALLOCATION_LRF: float = 0.022  # 2.2% per year
FREE_ALLOCATION_BASE_YEAR: int = 2021


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ETSBridgeConfig(BaseModel):
    """Configuration for the EU ETS bridge."""
    installation_id: Optional[str] = Field(default=None)
    phase: str = Field(
        default="IV",
        description="ETS phase (IV = 2021-2030)",
    )
    free_allocation_tracking: bool = Field(default=True)
    carbon_price_eur: float = Field(
        default=90.0, ge=0.0,
        description="Current EUA price in EUR",
    )
    msr_active: bool = Field(
        default=True,
        description="Market Stability Reserve impact",
    )
    carbon_leakage_sector: bool = Field(
        default=False,
        description="Whether installation is in carbon-leakage sector",
    )
    reporting_year: int = Field(default=2025)
    country_code: str = Field(default="DE")


class BenchmarkComparison(BaseModel):
    """Result of comparing installation intensity to EU ETS benchmark."""
    product: str
    installation_intensity: float = Field(
        ge=0.0, description="tCO2e per tonne product"
    )
    benchmark_value: float = Field(ge=0.0)
    ratio: float = Field(
        ge=0.0,
        description="installation_intensity / benchmark_value",
    )
    above_benchmark: bool = Field(default=False)
    gap_tco2e_per_tonne: float = Field(default=0.0)


class ETSComplianceResult(BaseModel):
    """EU ETS compliance obligation result."""
    installation_id: Optional[str] = Field(default=None)
    reporting_year: int = Field(default=2025)
    verified_emissions: float = Field(default=0.0, ge=0.0)
    free_allocation: float = Field(default=0.0, ge=0.0)
    shortfall: float = Field(
        default=0.0,
        description="Allowances to purchase (verified - free)",
    )
    surplus: float = Field(
        default=0.0,
        description="Excess free allocation (free - verified)",
    )
    carbon_price: float = Field(default=0.0, ge=0.0)
    compliance_cost: float = Field(
        default=0.0, ge=0.0,
        description="Estimated cost in EUR for shortfall",
    )
    benchmark_comparisons: List[BenchmarkComparison] = Field(
        default_factory=list
    )
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class EUETSBridge:
    """
    EU Emissions Trading System integration for manufacturing
    installations.

    Provides compliance-obligation calculation, free-allocation tracking,
    benchmark comparison, and cost estimation.
    """

    def __init__(
        self, config: Optional[ETSBridgeConfig] = None
    ) -> None:
        self.config = config or ETSBridgeConfig()

    @staticmethod
    def _compute_hash(data: Any) -> str:
        raw = str(data).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    # -- public API ----------------------------------------------------------

    def calculate_ets_obligation(
        self, verified_emissions: float
    ) -> ETSComplianceResult:
        """
        Calculate the full ETS compliance obligation for the reporting year.

        Args:
            verified_emissions: Verified annual emissions in tCO2e.

        Returns:
            ETSComplianceResult with shortfall, cost estimate, etc.
        """
        free_alloc = self.get_free_allocation(
            self.config.installation_id or "",
            self.config.reporting_year,
        )

        shortfall = max(verified_emissions - free_alloc, 0.0)
        surplus = max(free_alloc - verified_emissions, 0.0)
        cost = self.estimate_compliance_cost(
            shortfall, self.config.carbon_price_eur
        )

        data = {
            "verified": verified_emissions,
            "free": free_alloc,
            "shortfall": shortfall,
            "year": self.config.reporting_year,
        }

        return ETSComplianceResult(
            installation_id=self.config.installation_id,
            reporting_year=self.config.reporting_year,
            verified_emissions=round(verified_emissions, 4),
            free_allocation=round(free_alloc, 4),
            shortfall=round(shortfall, 4),
            surplus=round(surplus, 4),
            carbon_price=self.config.carbon_price_eur,
            compliance_cost=round(cost, 2),
            provenance_hash=self._compute_hash(data),
        )

    def get_free_allocation(
        self, installation_id: str, year: int
    ) -> float:
        """
        Calculate free allocation for the given year.

        Uses a linear-reduction factor from the base year.  Carbon-leakage
        sectors receive 100% of the benchmark allocation; others receive
        a declining share.
        """
        if not self.config.free_allocation_tracking:
            return 0.0

        # Base allocation (simplified: assume 10,000 tCO2e baseline)
        base_allocation = 10_000.0

        years_since_base = max(
            year - FREE_ALLOCATION_BASE_YEAR, 0
        )

        if self.config.carbon_leakage_sector:
            # Carbon-leakage sectors get 100% of (declining) benchmark
            reduction = 1.0 - (FREE_ALLOCATION_LRF * years_since_base)
        else:
            # Non-carbon-leakage: additional reduction
            reduction = max(
                1.0 - (FREE_ALLOCATION_LRF * 1.5 * years_since_base),
                0.0,
            )

        free_alloc = base_allocation * max(reduction, 0.0)

        logger.info(
            "Free allocation for %s year %d: %.2f tCO2e "
            "(reduction factor: %.4f)",
            installation_id, year, free_alloc, reduction,
        )
        return round(free_alloc, 4)

    def compare_benchmark(
        self,
        emissions_intensity: float,
        product_benchmark: str,
    ) -> Dict[str, Any]:
        """
        Compare an installation's emissions intensity against the EU ETS
        product benchmark.

        Args:
            emissions_intensity: tCO2e per tonne of product.
            product_benchmark: Product benchmark key (e.g. ``grey_clinker``).

        Returns:
            Dict with benchmark comparison details.
        """
        benchmark_val = PRODUCT_BENCHMARKS.get(product_benchmark)
        if benchmark_val is None:
            available = list(PRODUCT_BENCHMARKS.keys())
            return {
                "status": "benchmark_not_found",
                "product": product_benchmark,
                "available_benchmarks": available,
            }

        ratio = (
            emissions_intensity / benchmark_val
            if benchmark_val > 0 else 0.0
        )
        above = emissions_intensity > benchmark_val
        gap = max(emissions_intensity - benchmark_val, 0.0)

        comparison = BenchmarkComparison(
            product=product_benchmark,
            installation_intensity=round(emissions_intensity, 6),
            benchmark_value=benchmark_val,
            ratio=round(ratio, 4),
            above_benchmark=above,
            gap_tco2e_per_tonne=round(gap, 6),
        )

        return {
            "comparison": comparison.model_dump(),
            "status": "above_benchmark" if above else "at_or_below",
            "improvement_needed_pct": round(
                max((ratio - 1.0) * 100, 0.0), 2
            ),
        }

    def estimate_compliance_cost(
        self, shortfall: float, carbon_price: float
    ) -> float:
        """
        Estimate the compliance cost for purchasing EUAs.

        Args:
            shortfall: Allowances needed (tCO2e).
            carbon_price: Price per EUA in EUR.

        Returns:
            Estimated cost in EUR.
        """
        if shortfall <= 0:
            return 0.0
        cost = shortfall * carbon_price
        logger.info(
            "ETS compliance cost estimate: %.2f tCO2e x %.2f EUR = %.2f EUR",
            shortfall, carbon_price, cost,
        )
        return round(cost, 2)

    def get_product_benchmarks(self) -> Dict[str, float]:
        """Return the full product benchmark table."""
        return dict(PRODUCT_BENCHMARKS)

    def calculate_cbam_adjustment(
        self,
        verified_emissions: float,
        foreign_carbon_price_paid: float,
    ) -> Dict[str, Any]:
        """
        Calculate CBAM adjustment when the installation also exports
        to non-EU markets or when comparing with CBAM.
        """
        ets_cost = verified_emissions * self.config.carbon_price_eur
        net_cost = max(ets_cost - foreign_carbon_price_paid, 0.0)
        return {
            "ets_cost_eur": round(ets_cost, 2),
            "foreign_carbon_price_paid_eur": round(
                foreign_carbon_price_paid, 2
            ),
            "net_cost_eur": round(net_cost, 2),
            "cbam_relevant": foreign_carbon_price_paid > 0,
        }
