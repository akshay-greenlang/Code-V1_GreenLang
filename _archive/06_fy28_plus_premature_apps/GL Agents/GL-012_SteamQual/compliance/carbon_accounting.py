"""
GL-012 SteamQual - Carbon and Energy Accounting

Comprehensive carbon and energy accounting for steam quality improvements
including energy savings calculations, CO2e impact assessment, and
reconciliation with energy meters for auditable methodology.

Regulatory References:
- EPA 40 CFR Part 98: Mandatory Greenhouse Gas Reporting
- GHG Protocol: Corporate Standard
- ISO 14064: Greenhouse Gas Accounting
- DOE Steam System Assessment Guidelines

This module provides:
1. Energy savings calculations from quality improvements
2. CO2e impact of quality-related fuel savings
3. Reduced blowdown loss calculations
4. Marginal emissions factor application
5. Reconciliation with energy meters for audit trails

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CARBON ACCOUNTING ENUMERATIONS
# =============================================================================

class FuelType(Enum):
    """Fuel types for steam generation with EPA emission factors."""

    NATURAL_GAS = "natural_gas"
    FUEL_OIL_2 = "fuel_oil_no2"
    FUEL_OIL_6 = "fuel_oil_no6"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUBBITUMINOUS = "coal_subbituminous"
    PROPANE = "propane"
    BIOMASS_WOOD = "biomass_wood"
    BIOGAS = "biogas"


class EmissionScope(Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"  # Direct emissions from fuel combustion
    SCOPE_2 = "scope_2"  # Indirect from purchased electricity
    SCOPE_3 = "scope_3"  # Other indirect emissions


class QualityImprovementType(Enum):
    """Types of steam quality improvements affecting energy/emissions."""

    DRYNESS_IMPROVEMENT = "dryness_improvement"
    REDUCED_BLOWDOWN = "reduced_blowdown"
    IMPROVED_CONDENSATE_RETURN = "improved_condensate_return"
    REDUCED_STEAM_LOSSES = "reduced_steam_losses"
    SUPERHEAT_OPTIMIZATION = "superheat_optimization"
    PRESSURE_OPTIMIZATION = "pressure_optimization"


class EmissionFactorSource(Enum):
    """Source of emission factors for traceability."""

    EPA_40CFR98 = "EPA 40 CFR Part 98 Table C-1"
    EPA_EGRID = "EPA eGRID"
    IPCC_2006 = "IPCC 2006 Guidelines"
    GHG_PROTOCOL = "GHG Protocol"
    SITE_SPECIFIC = "Site-Specific Measurement"


# =============================================================================
# EMISSION FACTORS DATABASE
# =============================================================================

@dataclass
class EmissionFactor:
    """
    Emission factor with full regulatory traceability.

    All factors are from official EPA or IPCC sources
    with complete citation for audit purposes.
    """

    fuel_type: FuelType
    co2_factor: Decimal      # kg CO2/MMBtu
    ch4_factor: Decimal      # kg CH4/MMBtu
    n2o_factor: Decimal      # kg N2O/MMBtu
    unit: str
    source: EmissionFactorSource
    source_table: str
    effective_date: str
    hhv_basis: bool = True   # True = Higher Heating Value basis
    uncertainty_pct: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fuel_type": self.fuel_type.value,
            "co2_factor": str(self.co2_factor),
            "ch4_factor": str(self.ch4_factor),
            "n2o_factor": str(self.n2o_factor),
            "unit": self.unit,
            "source": self.source.value,
            "source_table": self.source_table,
            "effective_date": self.effective_date,
            "hhv_basis": self.hhv_basis,
            "uncertainty_pct": str(self.uncertainty_pct) if self.uncertainty_pct else None,
        }


# EPA 40 CFR Part 98 Table C-1 Emission Factors
EPA_EMISSION_FACTORS: Dict[FuelType, EmissionFactor] = {
    FuelType.NATURAL_GAS: EmissionFactor(
        fuel_type=FuelType.NATURAL_GAS,
        co2_factor=Decimal("53.06"),
        ch4_factor=Decimal("0.001"),
        n2o_factor=Decimal("0.0001"),
        unit="kg/MMBtu",
        source=EmissionFactorSource.EPA_40CFR98,
        source_table="Table C-1, Table C-2",
        effective_date="2024-01-01",
        uncertainty_pct=Decimal("1.0"),
    ),
    FuelType.FUEL_OIL_2: EmissionFactor(
        fuel_type=FuelType.FUEL_OIL_2,
        co2_factor=Decimal("73.96"),
        ch4_factor=Decimal("0.003"),
        n2o_factor=Decimal("0.0006"),
        unit="kg/MMBtu",
        source=EmissionFactorSource.EPA_40CFR98,
        source_table="Table C-1, Table C-2",
        effective_date="2024-01-01",
        uncertainty_pct=Decimal("1.0"),
    ),
    FuelType.FUEL_OIL_6: EmissionFactor(
        fuel_type=FuelType.FUEL_OIL_6,
        co2_factor=Decimal("75.10"),
        ch4_factor=Decimal("0.003"),
        n2o_factor=Decimal("0.0006"),
        unit="kg/MMBtu",
        source=EmissionFactorSource.EPA_40CFR98,
        source_table="Table C-1, Table C-2",
        effective_date="2024-01-01",
        uncertainty_pct=Decimal("1.0"),
    ),
    FuelType.COAL_BITUMINOUS: EmissionFactor(
        fuel_type=FuelType.COAL_BITUMINOUS,
        co2_factor=Decimal("93.28"),
        ch4_factor=Decimal("0.011"),
        n2o_factor=Decimal("0.0016"),
        unit="kg/MMBtu",
        source=EmissionFactorSource.EPA_40CFR98,
        source_table="Table C-1, Table C-2",
        effective_date="2024-01-01",
        uncertainty_pct=Decimal("2.0"),
    ),
    FuelType.COAL_SUBBITUMINOUS: EmissionFactor(
        fuel_type=FuelType.COAL_SUBBITUMINOUS,
        co2_factor=Decimal("97.17"),
        ch4_factor=Decimal("0.011"),
        n2o_factor=Decimal("0.0016"),
        unit="kg/MMBtu",
        source=EmissionFactorSource.EPA_40CFR98,
        source_table="Table C-1, Table C-2",
        effective_date="2024-01-01",
        uncertainty_pct=Decimal("2.0"),
    ),
    FuelType.PROPANE: EmissionFactor(
        fuel_type=FuelType.PROPANE,
        co2_factor=Decimal("62.87"),
        ch4_factor=Decimal("0.003"),
        n2o_factor=Decimal("0.0006"),
        unit="kg/MMBtu",
        source=EmissionFactorSource.EPA_40CFR98,
        source_table="Table C-1, Table C-2",
        effective_date="2024-01-01",
        uncertainty_pct=Decimal("1.0"),
    ),
    FuelType.BIOMASS_WOOD: EmissionFactor(
        fuel_type=FuelType.BIOMASS_WOOD,
        co2_factor=Decimal("93.80"),  # Biogenic - may be reported separately
        ch4_factor=Decimal("0.032"),
        n2o_factor=Decimal("0.0042"),
        unit="kg/MMBtu",
        source=EmissionFactorSource.EPA_40CFR98,
        source_table="Table C-1, Table C-2",
        effective_date="2024-01-01",
        uncertainty_pct=Decimal("5.0"),
    ),
}

# Global Warming Potentials (100-year) per EPA/IPCC AR4
GWP_VALUES = {
    "CO2": Decimal("1"),
    "CH4": Decimal("25"),
    "N2O": Decimal("298"),
}


# =============================================================================
# ENERGY SAVINGS CALCULATIONS
# =============================================================================

@dataclass
class EnergySavingsResult:
    """
    Energy savings calculation result with full audit trail.

    Provides complete documentation of energy savings
    from steam quality improvements.
    """

    calculation_id: str
    timestamp: datetime
    improvement_type: QualityImprovementType
    description: str

    # Baseline and improved states
    baseline_value: Decimal
    improved_value: Decimal
    value_unit: str

    # Energy savings
    energy_savings_mmbtu: Decimal
    energy_savings_gj: Decimal
    savings_percentage: Decimal

    # Annualized
    annual_operating_hours: Decimal
    annual_energy_savings_mmbtu: Decimal
    annual_energy_savings_gj: Decimal

    # Financial (if applicable)
    fuel_cost_per_mmbtu: Optional[Decimal]
    annual_cost_savings_usd: Optional[Decimal]

    # Methodology
    calculation_method: str
    assumptions: List[str]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "calculation_id": self.calculation_id,
            "timestamp": self.timestamp.isoformat(),
            "improvement_type": self.improvement_type.value,
            "description": self.description,
            "baseline_value": str(self.baseline_value),
            "improved_value": str(self.improved_value),
            "value_unit": self.value_unit,
            "energy_savings_mmbtu": str(self.energy_savings_mmbtu),
            "energy_savings_gj": str(self.energy_savings_gj),
            "savings_percentage": str(self.savings_percentage),
            "annual_operating_hours": str(self.annual_operating_hours),
            "annual_energy_savings_mmbtu": str(self.annual_energy_savings_mmbtu),
            "annual_energy_savings_gj": str(self.annual_energy_savings_gj),
            "fuel_cost_per_mmbtu": str(self.fuel_cost_per_mmbtu) if self.fuel_cost_per_mmbtu else None,
            "annual_cost_savings_usd": str(self.annual_cost_savings_usd) if self.annual_cost_savings_usd else None,
            "calculation_method": self.calculation_method,
            "assumptions": self.assumptions,
            "provenance_hash": self.provenance_hash,
        }


class EnergySavingsCalculator:
    """
    Calculator for energy savings from steam quality improvements.

    Implements deterministic calculations following DOE methodologies.
    Zero-hallucination: All formulas are physics-based and auditable.

    Example:
        >>> calc = EnergySavingsCalculator()
        >>> result = calc.calculate_dryness_improvement_savings(
        ...     steam_flow_lb_hr=Decimal("10000"),
        ...     baseline_dryness=Decimal("0.95"),
        ...     improved_dryness=Decimal("0.98"),
        ...     operating_hours=Decimal("8000")
        ... )
        >>> print(f"Annual savings: {result.annual_energy_savings_mmbtu} MMBtu")
    """

    VERSION = "1.0.0"

    # Conversion factors
    BTU_PER_LB_LATENT = Decimal("970")  # Approximate latent heat of steam at 100 psig
    GJ_PER_MMBTU = Decimal("1.0551")
    MMBTU_TO_BTU = Decimal("1000000")

    def __init__(self, boiler_efficiency: Decimal = Decimal("0.82")) -> None:
        """
        Initialize energy savings calculator.

        Args:
            boiler_efficiency: Assumed boiler efficiency (default 82%)
        """
        self.boiler_efficiency = boiler_efficiency
        logger.info(f"EnergySavingsCalculator initialized with efficiency={boiler_efficiency}")

    def calculate_dryness_improvement_savings(
        self,
        steam_flow_lb_hr: Union[Decimal, float],
        baseline_dryness: Union[Decimal, float],
        improved_dryness: Union[Decimal, float],
        operating_hours: Union[Decimal, float],
        steam_enthalpy_btu_lb: Union[Decimal, float] = Decimal("1190"),
        fuel_cost_per_mmbtu: Optional[Union[Decimal, float]] = None,
    ) -> EnergySavingsResult:
        """
        Calculate energy savings from steam dryness improvement.

        Physics basis: Higher dryness fraction means more usable
        latent heat delivered per pound of steam generated.

        Formula:
        Energy Savings = Steam Flow * (Improved - Baseline) * Enthalpy / Boiler Eff

        Args:
            steam_flow_lb_hr: Steam flow rate in lb/hr
            baseline_dryness: Baseline dryness fraction (0-1)
            improved_dryness: Improved dryness fraction (0-1)
            operating_hours: Annual operating hours
            steam_enthalpy_btu_lb: Steam enthalpy (default 1190 BTU/lb)
            fuel_cost_per_mmbtu: Optional fuel cost for savings calculation

        Returns:
            EnergySavingsResult with complete documentation
        """
        timestamp = datetime.now(timezone.utc)

        # Convert to Decimal
        flow = Decimal(str(steam_flow_lb_hr))
        baseline = Decimal(str(baseline_dryness))
        improved = Decimal(str(improved_dryness))
        hours = Decimal(str(operating_hours))
        enthalpy = Decimal(str(steam_enthalpy_btu_lb))

        # Validate inputs
        if not (Decimal("0") < baseline < Decimal("1")):
            raise ValueError(f"Invalid baseline dryness: {baseline}")
        if not (Decimal("0") < improved <= Decimal("1")):
            raise ValueError(f"Invalid improved dryness: {improved}")
        if improved <= baseline:
            raise ValueError("Improved dryness must exceed baseline")

        # Calculate dryness improvement
        dryness_improvement = improved - baseline

        # Energy benefit per lb of steam (more usable heat delivered)
        # When dryness improves, we get more latent heat per lb generated
        energy_benefit_btu_lb = enthalpy * dryness_improvement

        # Hourly savings in BTU (this is fuel saved due to less makeup for same useful heat)
        hourly_savings_btu = flow * energy_benefit_btu_lb / self.boiler_efficiency

        # Convert to MMBtu
        hourly_savings_mmbtu = hourly_savings_btu / self.MMBTU_TO_BTU

        # Annual savings
        annual_savings_mmbtu = hourly_savings_mmbtu * hours

        # Convert to GJ
        hourly_savings_gj = hourly_savings_mmbtu * self.GJ_PER_MMBTU
        annual_savings_gj = annual_savings_mmbtu * self.GJ_PER_MMBTU

        # Savings percentage
        baseline_energy = flow * enthalpy / self.boiler_efficiency / self.MMBTU_TO_BTU
        savings_pct = (hourly_savings_mmbtu / baseline_energy) * Decimal("100") if baseline_energy > 0 else Decimal("0")

        # Cost savings
        cost_savings = None
        if fuel_cost_per_mmbtu is not None:
            cost_savings = annual_savings_mmbtu * Decimal(str(fuel_cost_per_mmbtu))

        # Provenance hash
        provenance_data = {
            "steam_flow_lb_hr": str(flow),
            "baseline_dryness": str(baseline),
            "improved_dryness": str(improved),
            "annual_savings_mmbtu": str(annual_savings_mmbtu),
            "timestamp": timestamp.isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return EnergySavingsResult(
            calculation_id=f"ES-DRY-{timestamp.strftime('%Y%m%d%H%M%S')}",
            timestamp=timestamp,
            improvement_type=QualityImprovementType.DRYNESS_IMPROVEMENT,
            description=f"Steam dryness improvement from {baseline} to {improved}",
            baseline_value=baseline,
            improved_value=improved,
            value_unit="fraction",
            energy_savings_mmbtu=hourly_savings_mmbtu.quantize(Decimal("0.001")),
            energy_savings_gj=hourly_savings_gj.quantize(Decimal("0.001")),
            savings_percentage=savings_pct.quantize(Decimal("0.01")),
            annual_operating_hours=hours,
            annual_energy_savings_mmbtu=annual_savings_mmbtu.quantize(Decimal("0.01")),
            annual_energy_savings_gj=annual_savings_gj.quantize(Decimal("0.01")),
            fuel_cost_per_mmbtu=Decimal(str(fuel_cost_per_mmbtu)) if fuel_cost_per_mmbtu else None,
            annual_cost_savings_usd=cost_savings.quantize(Decimal("0.01")) if cost_savings else None,
            calculation_method="DOE Steam Tip Sheet methodology",
            assumptions=[
                f"Boiler efficiency: {self.boiler_efficiency}",
                f"Steam enthalpy: {enthalpy} BTU/lb",
                "Steady-state operation assumed",
            ],
            provenance_hash=provenance_hash,
        )

    def calculate_blowdown_reduction_savings(
        self,
        steam_generation_lb_hr: Union[Decimal, float],
        baseline_blowdown_pct: Union[Decimal, float],
        improved_blowdown_pct: Union[Decimal, float],
        operating_hours: Union[Decimal, float],
        feedwater_temp_f: Union[Decimal, float] = Decimal("180"),
        blowdown_temp_f: Union[Decimal, float] = Decimal("366"),  # ~150 psig sat
        fuel_cost_per_mmbtu: Optional[Union[Decimal, float]] = None,
    ) -> EnergySavingsResult:
        """
        Calculate energy savings from reduced boiler blowdown.

        Physics basis: Blowdown removes hot water from the boiler,
        requiring additional fuel to heat makeup water.

        Formula:
        Blowdown Loss = Blowdown Rate * (h_blowdown - h_feedwater) / Boiler Eff

        Args:
            steam_generation_lb_hr: Steam generation rate in lb/hr
            baseline_blowdown_pct: Baseline blowdown as % of steam flow
            improved_blowdown_pct: Improved blowdown as % of steam flow
            operating_hours: Annual operating hours
            feedwater_temp_f: Feedwater temperature (F)
            blowdown_temp_f: Blowdown water temperature (F)
            fuel_cost_per_mmbtu: Optional fuel cost

        Returns:
            EnergySavingsResult with complete documentation
        """
        timestamp = datetime.now(timezone.utc)

        # Convert to Decimal
        steam_gen = Decimal(str(steam_generation_lb_hr))
        baseline_bd = Decimal(str(baseline_blowdown_pct)) / Decimal("100")
        improved_bd = Decimal(str(improved_blowdown_pct)) / Decimal("100")
        hours = Decimal(str(operating_hours))
        fw_temp = Decimal(str(feedwater_temp_f))
        bd_temp = Decimal(str(blowdown_temp_f))

        # Validate
        if improved_bd >= baseline_bd:
            raise ValueError("Improved blowdown must be less than baseline")

        # Approximate enthalpy (BTU/lb) using temperature
        # h ~ 1 BTU/lb-F relative to 32F for liquid water
        h_feedwater = fw_temp - Decimal("32")
        h_blowdown = bd_temp - Decimal("32")
        enthalpy_loss = h_blowdown - h_feedwater

        # Calculate blowdown rates
        baseline_bd_flow = steam_gen * baseline_bd
        improved_bd_flow = steam_gen * improved_bd
        bd_reduction = baseline_bd_flow - improved_bd_flow

        # Energy saved (BTU/hr)
        hourly_savings_btu = bd_reduction * enthalpy_loss / self.boiler_efficiency

        # Convert to MMBtu
        hourly_savings_mmbtu = hourly_savings_btu / self.MMBTU_TO_BTU
        annual_savings_mmbtu = hourly_savings_mmbtu * hours

        # Convert to GJ
        hourly_savings_gj = hourly_savings_mmbtu * self.GJ_PER_MMBTU
        annual_savings_gj = annual_savings_mmbtu * self.GJ_PER_MMBTU

        # Savings percentage (relative to total boiler fuel)
        total_steam_energy = steam_gen * Decimal("1000") / self.boiler_efficiency / self.MMBTU_TO_BTU
        savings_pct = (hourly_savings_mmbtu / total_steam_energy) * Decimal("100") if total_steam_energy > 0 else Decimal("0")

        # Cost savings
        cost_savings = None
        if fuel_cost_per_mmbtu is not None:
            cost_savings = annual_savings_mmbtu * Decimal(str(fuel_cost_per_mmbtu))

        # Provenance
        provenance_data = {
            "steam_generation": str(steam_gen),
            "baseline_blowdown_pct": str(baseline_bd * 100),
            "improved_blowdown_pct": str(improved_bd * 100),
            "annual_savings_mmbtu": str(annual_savings_mmbtu),
            "timestamp": timestamp.isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return EnergySavingsResult(
            calculation_id=f"ES-BD-{timestamp.strftime('%Y%m%d%H%M%S')}",
            timestamp=timestamp,
            improvement_type=QualityImprovementType.REDUCED_BLOWDOWN,
            description=f"Blowdown reduction from {baseline_bd*100}% to {improved_bd*100}%",
            baseline_value=baseline_bd * Decimal("100"),
            improved_value=improved_bd * Decimal("100"),
            value_unit="percent",
            energy_savings_mmbtu=hourly_savings_mmbtu.quantize(Decimal("0.001")),
            energy_savings_gj=hourly_savings_gj.quantize(Decimal("0.001")),
            savings_percentage=savings_pct.quantize(Decimal("0.01")),
            annual_operating_hours=hours,
            annual_energy_savings_mmbtu=annual_savings_mmbtu.quantize(Decimal("0.01")),
            annual_energy_savings_gj=annual_savings_gj.quantize(Decimal("0.01")),
            fuel_cost_per_mmbtu=Decimal(str(fuel_cost_per_mmbtu)) if fuel_cost_per_mmbtu else None,
            annual_cost_savings_usd=cost_savings.quantize(Decimal("0.01")) if cost_savings else None,
            calculation_method="DOE Steam Tip Sheet #9 methodology",
            assumptions=[
                f"Boiler efficiency: {self.boiler_efficiency}",
                f"Feedwater temperature: {fw_temp} F",
                f"Blowdown temperature: {bd_temp} F",
                "No heat recovery from blowdown",
            ],
            provenance_hash=provenance_hash,
        )


# =============================================================================
# CO2e IMPACT CALCULATIONS
# =============================================================================

@dataclass
class EmissionsImpactResult:
    """
    Emissions impact calculation result with EPA traceability.

    Documents CO2e reductions from energy savings with
    complete regulatory citation.
    """

    calculation_id: str
    timestamp: datetime
    improvement_type: QualityImprovementType

    # Energy basis
    energy_savings_mmbtu: Decimal
    energy_savings_gj: Decimal

    # Fuel and factors
    fuel_type: FuelType
    emission_factor_source: EmissionFactorSource
    co2_factor_kg_mmbtu: Decimal
    ch4_factor_kg_mmbtu: Decimal
    n2o_factor_kg_mmbtu: Decimal

    # Individual GHG reductions
    co2_reduction_kg: Decimal
    ch4_reduction_kg: Decimal
    n2o_reduction_kg: Decimal

    # CO2e (using GWP)
    co2e_reduction_kg: Decimal
    co2e_reduction_mt: Decimal  # Metric tonnes

    # Annualized
    annual_co2e_reduction_mt: Decimal

    # Marginal factor (if applicable)
    marginal_factor_applied: bool
    marginal_factor_source: Optional[str]

    # Provenance
    scope: EmissionScope
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "calculation_id": self.calculation_id,
            "timestamp": self.timestamp.isoformat(),
            "improvement_type": self.improvement_type.value,
            "energy_savings_mmbtu": str(self.energy_savings_mmbtu),
            "energy_savings_gj": str(self.energy_savings_gj),
            "fuel_type": self.fuel_type.value,
            "emission_factor_source": self.emission_factor_source.value,
            "co2_factor_kg_mmbtu": str(self.co2_factor_kg_mmbtu),
            "ch4_factor_kg_mmbtu": str(self.ch4_factor_kg_mmbtu),
            "n2o_factor_kg_mmbtu": str(self.n2o_factor_kg_mmbtu),
            "co2_reduction_kg": str(self.co2_reduction_kg),
            "ch4_reduction_kg": str(self.ch4_reduction_kg),
            "n2o_reduction_kg": str(self.n2o_reduction_kg),
            "co2e_reduction_kg": str(self.co2e_reduction_kg),
            "co2e_reduction_mt": str(self.co2e_reduction_mt),
            "annual_co2e_reduction_mt": str(self.annual_co2e_reduction_mt),
            "marginal_factor_applied": self.marginal_factor_applied,
            "marginal_factor_source": self.marginal_factor_source,
            "scope": self.scope.value,
            "provenance_hash": self.provenance_hash,
        }


class EmissionsImpactCalculator:
    """
    Calculator for CO2e impact of energy savings.

    Uses EPA 40 CFR Part 98 emission factors with
    complete regulatory traceability.

    Zero-hallucination: All factors are from official sources.

    Example:
        >>> calc = EmissionsImpactCalculator()
        >>> result = calc.calculate_emissions_reduction(
        ...     energy_savings_mmbtu=Decimal("1000"),
        ...     fuel_type=FuelType.NATURAL_GAS,
        ...     improvement_type=QualityImprovementType.DRYNESS_IMPROVEMENT
        ... )
        >>> print(f"CO2e reduction: {result.co2e_reduction_mt} MT")
    """

    VERSION = "1.0.0"
    KG_PER_MT = Decimal("1000")

    def __init__(
        self,
        emission_factors: Optional[Dict[FuelType, EmissionFactor]] = None,
        gwp_values: Optional[Dict[str, Decimal]] = None,
    ) -> None:
        """
        Initialize emissions impact calculator.

        Args:
            emission_factors: Custom emission factors (default EPA)
            gwp_values: Custom GWP values (default IPCC AR4)
        """
        self._factors = emission_factors or EPA_EMISSION_FACTORS
        self._gwp = gwp_values or GWP_VALUES
        logger.info("EmissionsImpactCalculator initialized with EPA factors")

    def calculate_emissions_reduction(
        self,
        energy_savings_mmbtu: Union[Decimal, float],
        fuel_type: FuelType,
        improvement_type: QualityImprovementType,
        operating_hours: Optional[Union[Decimal, float]] = None,
        use_marginal_factor: bool = False,
        marginal_factor: Optional[Decimal] = None,
        marginal_source: Optional[str] = None,
    ) -> EmissionsImpactResult:
        """
        Calculate emissions reduction from energy savings.

        Uses EPA 40 CFR Part 98 emission factors for
        complete regulatory traceability.

        Formula:
        CO2e = Energy * (CO2_factor + CH4_factor*GWP_CH4 + N2O_factor*GWP_N2O)

        Args:
            energy_savings_mmbtu: Energy savings in MMBtu
            fuel_type: Type of fuel saved
            improvement_type: Type of quality improvement
            operating_hours: Annual operating hours (for annualization)
            use_marginal_factor: Apply marginal emissions factor
            marginal_factor: Custom marginal factor (kg CO2e/MMBtu)
            marginal_source: Source of marginal factor

        Returns:
            EmissionsImpactResult with complete audit trail
        """
        timestamp = datetime.now(timezone.utc)
        energy = Decimal(str(energy_savings_mmbtu))

        # Get emission factor
        factor = self._factors.get(fuel_type)
        if factor is None:
            raise ValueError(f"No emission factor for fuel type: {fuel_type}")

        # Calculate individual GHG reductions
        co2_reduction = energy * factor.co2_factor
        ch4_reduction = energy * factor.ch4_factor
        n2o_reduction = energy * factor.n2o_factor

        # Calculate CO2e using GWP
        co2e_from_co2 = co2_reduction * self._gwp["CO2"]
        co2e_from_ch4 = ch4_reduction * self._gwp["CH4"]
        co2e_from_n2o = n2o_reduction * self._gwp["N2O"]
        co2e_total = co2e_from_co2 + co2e_from_ch4 + co2e_from_n2o

        # Apply marginal factor if specified
        if use_marginal_factor and marginal_factor:
            co2e_total = energy * marginal_factor

        # Convert to metric tonnes
        co2e_mt = co2e_total / self.KG_PER_MT

        # Annualize if hours provided
        annual_co2e_mt = co2e_mt
        if operating_hours:
            # Assume energy_savings is hourly, multiply by hours
            annual_co2e_mt = co2e_mt * Decimal(str(operating_hours))

        # Convert energy to GJ
        energy_gj = energy * Decimal("1.0551")

        # Provenance hash
        provenance_data = {
            "energy_savings_mmbtu": str(energy),
            "fuel_type": fuel_type.value,
            "co2e_reduction_kg": str(co2e_total),
            "timestamp": timestamp.isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return EmissionsImpactResult(
            calculation_id=f"EI-{timestamp.strftime('%Y%m%d%H%M%S')}",
            timestamp=timestamp,
            improvement_type=improvement_type,
            energy_savings_mmbtu=energy.quantize(Decimal("0.01")),
            energy_savings_gj=energy_gj.quantize(Decimal("0.01")),
            fuel_type=fuel_type,
            emission_factor_source=factor.source,
            co2_factor_kg_mmbtu=factor.co2_factor,
            ch4_factor_kg_mmbtu=factor.ch4_factor,
            n2o_factor_kg_mmbtu=factor.n2o_factor,
            co2_reduction_kg=co2_reduction.quantize(Decimal("0.01")),
            ch4_reduction_kg=ch4_reduction.quantize(Decimal("0.0001")),
            n2o_reduction_kg=n2o_reduction.quantize(Decimal("0.00001")),
            co2e_reduction_kg=co2e_total.quantize(Decimal("0.01")),
            co2e_reduction_mt=co2e_mt.quantize(Decimal("0.0001")),
            annual_co2e_reduction_mt=annual_co2e_mt.quantize(Decimal("0.01")),
            marginal_factor_applied=use_marginal_factor,
            marginal_factor_source=marginal_source,
            scope=EmissionScope.SCOPE_1,
            provenance_hash=provenance_hash,
        )


# =============================================================================
# METER RECONCILIATION
# =============================================================================

@dataclass
class MeterReconciliationResult:
    """
    Result of reconciliation between calculated and metered values.

    Provides auditable verification that calculated savings
    reconcile with actual energy meter readings.
    """

    reconciliation_id: str
    timestamp: datetime
    period_start: datetime
    period_end: datetime

    # Calculated values
    calculated_energy_mmbtu: Decimal
    calculated_savings_mmbtu: Decimal

    # Metered values
    metered_fuel_mmbtu: Decimal
    metered_steam_klb: Decimal

    # Derived metered savings
    baseline_metered_fuel_mmbtu: Decimal
    metered_savings_mmbtu: Decimal

    # Reconciliation
    variance_mmbtu: Decimal
    variance_percent: Decimal
    reconciliation_status: str  # "PASS", "MARGINAL", "FAIL"

    # Adjustments
    adjustments_applied: List[Dict[str, Any]]

    # Provenance
    meter_ids: List[str]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "reconciliation_id": self.reconciliation_id,
            "timestamp": self.timestamp.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "calculated_energy_mmbtu": str(self.calculated_energy_mmbtu),
            "calculated_savings_mmbtu": str(self.calculated_savings_mmbtu),
            "metered_fuel_mmbtu": str(self.metered_fuel_mmbtu),
            "metered_steam_klb": str(self.metered_steam_klb),
            "baseline_metered_fuel_mmbtu": str(self.baseline_metered_fuel_mmbtu),
            "metered_savings_mmbtu": str(self.metered_savings_mmbtu),
            "variance_mmbtu": str(self.variance_mmbtu),
            "variance_percent": str(self.variance_percent),
            "reconciliation_status": self.reconciliation_status,
            "adjustments_applied": self.adjustments_applied,
            "meter_ids": self.meter_ids,
            "provenance_hash": self.provenance_hash,
        }


class MeterReconciler:
    """
    Reconciles calculated savings with energy meter readings.

    Per playbook requirement: methodology must be auditable
    and reconcile with energy meters.

    Example:
        >>> reconciler = MeterReconciler()
        >>> result = reconciler.reconcile(
        ...     calculated_savings=Decimal("500"),
        ...     metered_fuel=Decimal("10000"),
        ...     baseline_fuel=Decimal("10500"),
        ...     steam_production=Decimal("50000"),
        ...     meter_ids=["FM-001", "FM-002"]
        ... )
        >>> print(f"Status: {result.reconciliation_status}")
    """

    VERSION = "1.0.0"

    # Reconciliation thresholds
    PASS_THRESHOLD_PCT = Decimal("5")      # Within 5%
    MARGINAL_THRESHOLD_PCT = Decimal("10")  # Within 10%

    def __init__(self) -> None:
        """Initialize meter reconciler."""
        logger.info("MeterReconciler initialized")

    def reconcile(
        self,
        calculated_savings_mmbtu: Union[Decimal, float],
        metered_fuel_mmbtu: Union[Decimal, float],
        baseline_fuel_mmbtu: Union[Decimal, float],
        steam_production_klb: Union[Decimal, float],
        period_start: datetime,
        period_end: datetime,
        meter_ids: List[str],
        adjustments: Optional[List[Dict[str, Any]]] = None,
    ) -> MeterReconciliationResult:
        """
        Reconcile calculated savings with metered values.

        Args:
            calculated_savings_mmbtu: Calculated energy savings
            metered_fuel_mmbtu: Actual metered fuel consumption
            baseline_fuel_mmbtu: Baseline fuel consumption
            steam_production_klb: Metered steam production
            period_start: Start of reconciliation period
            period_end: End of reconciliation period
            meter_ids: List of meter identifiers used
            adjustments: Optional adjustments applied

        Returns:
            MeterReconciliationResult with status
        """
        timestamp = datetime.now(timezone.utc)

        calc_savings = Decimal(str(calculated_savings_mmbtu))
        metered_fuel = Decimal(str(metered_fuel_mmbtu))
        baseline_fuel = Decimal(str(baseline_fuel_mmbtu))
        steam_prod = Decimal(str(steam_production_klb))

        # Calculate metered savings
        metered_savings = baseline_fuel - metered_fuel

        # Apply adjustments if any
        applied_adjustments = adjustments or []
        adjusted_metered_savings = metered_savings
        for adj in applied_adjustments:
            if "value" in adj:
                adjusted_metered_savings += Decimal(str(adj["value"]))

        # Calculate variance
        variance = calc_savings - adjusted_metered_savings

        # Variance percentage
        if adjusted_metered_savings != Decimal("0"):
            variance_pct = abs(variance / adjusted_metered_savings) * Decimal("100")
        else:
            variance_pct = Decimal("100") if calc_savings != Decimal("0") else Decimal("0")

        # Determine status
        if variance_pct <= self.PASS_THRESHOLD_PCT:
            status = "PASS"
        elif variance_pct <= self.MARGINAL_THRESHOLD_PCT:
            status = "MARGINAL"
        else:
            status = "FAIL"

        # Provenance hash
        provenance_data = {
            "calculated_savings": str(calc_savings),
            "metered_savings": str(adjusted_metered_savings),
            "variance": str(variance),
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "meter_ids": meter_ids,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return MeterReconciliationResult(
            reconciliation_id=f"MR-{timestamp.strftime('%Y%m%d%H%M%S')}",
            timestamp=timestamp,
            period_start=period_start,
            period_end=period_end,
            calculated_energy_mmbtu=calc_savings + baseline_fuel,
            calculated_savings_mmbtu=calc_savings.quantize(Decimal("0.01")),
            metered_fuel_mmbtu=metered_fuel.quantize(Decimal("0.01")),
            metered_steam_klb=steam_prod.quantize(Decimal("0.01")),
            baseline_metered_fuel_mmbtu=baseline_fuel.quantize(Decimal("0.01")),
            metered_savings_mmbtu=adjusted_metered_savings.quantize(Decimal("0.01")),
            variance_mmbtu=variance.quantize(Decimal("0.01")),
            variance_percent=variance_pct.quantize(Decimal("0.01")),
            reconciliation_status=status,
            adjustments_applied=applied_adjustments,
            meter_ids=meter_ids,
            provenance_hash=provenance_hash,
        )


# =============================================================================
# CARBON ACCOUNTING MANAGER
# =============================================================================

class CarbonAccountingManager:
    """
    Central manager for carbon and energy accounting.

    Coordinates energy savings calculations, emissions impact,
    and meter reconciliation for complete audit trail.

    Example:
        >>> manager = CarbonAccountingManager(FuelType.NATURAL_GAS)
        >>> savings = manager.calculate_quality_improvement_savings(...)
        >>> emissions = manager.calculate_emissions_impact(savings)
        >>> reconciliation = manager.reconcile_with_meters(...)
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        primary_fuel: FuelType,
        boiler_efficiency: Decimal = Decimal("0.82"),
    ) -> None:
        """
        Initialize carbon accounting manager.

        Args:
            primary_fuel: Primary fuel type for steam generation
            boiler_efficiency: Boiler efficiency (decimal)
        """
        self.primary_fuel = primary_fuel
        self.boiler_efficiency = boiler_efficiency

        self._energy_calc = EnergySavingsCalculator(boiler_efficiency)
        self._emissions_calc = EmissionsImpactCalculator()
        self._reconciler = MeterReconciler()

        logger.info(
            f"CarbonAccountingManager initialized for {primary_fuel.value}, "
            f"efficiency={boiler_efficiency}"
        )

    def calculate_dryness_improvement_impact(
        self,
        steam_flow_lb_hr: Decimal,
        baseline_dryness: Decimal,
        improved_dryness: Decimal,
        operating_hours: Decimal,
        fuel_cost_per_mmbtu: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """
        Calculate complete energy and emissions impact of dryness improvement.

        Args:
            steam_flow_lb_hr: Steam flow rate
            baseline_dryness: Baseline dryness fraction
            improved_dryness: Improved dryness fraction
            operating_hours: Annual operating hours
            fuel_cost_per_mmbtu: Optional fuel cost

        Returns:
            Complete impact assessment with energy and emissions
        """
        # Calculate energy savings
        energy_result = self._energy_calc.calculate_dryness_improvement_savings(
            steam_flow_lb_hr=steam_flow_lb_hr,
            baseline_dryness=baseline_dryness,
            improved_dryness=improved_dryness,
            operating_hours=operating_hours,
            fuel_cost_per_mmbtu=fuel_cost_per_mmbtu,
        )

        # Calculate emissions impact
        emissions_result = self._emissions_calc.calculate_emissions_reduction(
            energy_savings_mmbtu=energy_result.annual_energy_savings_mmbtu,
            fuel_type=self.primary_fuel,
            improvement_type=QualityImprovementType.DRYNESS_IMPROVEMENT,
        )

        return {
            "energy_savings": energy_result.to_dict(),
            "emissions_impact": emissions_result.to_dict(),
            "summary": {
                "annual_energy_savings_mmbtu": str(energy_result.annual_energy_savings_mmbtu),
                "annual_energy_savings_gj": str(energy_result.annual_energy_savings_gj),
                "annual_co2e_reduction_mt": str(emissions_result.annual_co2e_reduction_mt),
                "annual_cost_savings_usd": str(energy_result.annual_cost_savings_usd) if energy_result.annual_cost_savings_usd else None,
            },
        }

    def calculate_blowdown_reduction_impact(
        self,
        steam_generation_lb_hr: Decimal,
        baseline_blowdown_pct: Decimal,
        improved_blowdown_pct: Decimal,
        operating_hours: Decimal,
        fuel_cost_per_mmbtu: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """
        Calculate complete energy and emissions impact of blowdown reduction.

        Args:
            steam_generation_lb_hr: Steam generation rate
            baseline_blowdown_pct: Baseline blowdown percentage
            improved_blowdown_pct: Improved blowdown percentage
            operating_hours: Annual operating hours
            fuel_cost_per_mmbtu: Optional fuel cost

        Returns:
            Complete impact assessment
        """
        # Calculate energy savings
        energy_result = self._energy_calc.calculate_blowdown_reduction_savings(
            steam_generation_lb_hr=steam_generation_lb_hr,
            baseline_blowdown_pct=baseline_blowdown_pct,
            improved_blowdown_pct=improved_blowdown_pct,
            operating_hours=operating_hours,
            fuel_cost_per_mmbtu=fuel_cost_per_mmbtu,
        )

        # Calculate emissions impact
        emissions_result = self._emissions_calc.calculate_emissions_reduction(
            energy_savings_mmbtu=energy_result.annual_energy_savings_mmbtu,
            fuel_type=self.primary_fuel,
            improvement_type=QualityImprovementType.REDUCED_BLOWDOWN,
        )

        return {
            "energy_savings": energy_result.to_dict(),
            "emissions_impact": emissions_result.to_dict(),
            "summary": {
                "annual_energy_savings_mmbtu": str(energy_result.annual_energy_savings_mmbtu),
                "annual_energy_savings_gj": str(energy_result.annual_energy_savings_gj),
                "annual_co2e_reduction_mt": str(emissions_result.annual_co2e_reduction_mt),
                "annual_cost_savings_usd": str(energy_result.annual_cost_savings_usd) if energy_result.annual_cost_savings_usd else None,
            },
        }

    def reconcile_with_meters(
        self,
        calculated_savings_mmbtu: Decimal,
        metered_fuel_mmbtu: Decimal,
        baseline_fuel_mmbtu: Decimal,
        steam_production_klb: Decimal,
        period_start: datetime,
        period_end: datetime,
        meter_ids: List[str],
    ) -> MeterReconciliationResult:
        """
        Reconcile calculated savings with meter readings.

        Per playbook: methodology must be auditable and
        reconcile with energy meters.

        Args:
            calculated_savings_mmbtu: Calculated savings
            metered_fuel_mmbtu: Actual metered fuel
            baseline_fuel_mmbtu: Baseline fuel consumption
            steam_production_klb: Steam production
            period_start: Period start
            period_end: Period end
            meter_ids: Meter identifiers

        Returns:
            MeterReconciliationResult
        """
        return self._reconciler.reconcile(
            calculated_savings_mmbtu=calculated_savings_mmbtu,
            metered_fuel_mmbtu=metered_fuel_mmbtu,
            baseline_fuel_mmbtu=baseline_fuel_mmbtu,
            steam_production_klb=steam_production_klb,
            period_start=period_start,
            period_end=period_end,
            meter_ids=meter_ids,
        )

    def generate_accounting_report(
        self,
        site_id: str,
        period_start: datetime,
        period_end: datetime,
        improvements: List[Dict[str, Any]],
        reconciliation: Optional[MeterReconciliationResult] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive carbon accounting report.

        Args:
            site_id: Site identifier
            period_start: Reporting period start
            period_end: Reporting period end
            improvements: List of improvement results
            reconciliation: Optional meter reconciliation

        Returns:
            Complete accounting report
        """
        timestamp = datetime.now(timezone.utc)

        # Aggregate totals
        total_energy_mmbtu = Decimal("0")
        total_co2e_mt = Decimal("0")
        total_cost_savings = Decimal("0")

        for imp in improvements:
            summary = imp.get("summary", {})
            if "annual_energy_savings_mmbtu" in summary:
                total_energy_mmbtu += Decimal(summary["annual_energy_savings_mmbtu"])
            if "annual_co2e_reduction_mt" in summary:
                total_co2e_mt += Decimal(summary["annual_co2e_reduction_mt"])
            if summary.get("annual_cost_savings_usd"):
                total_cost_savings += Decimal(summary["annual_cost_savings_usd"])

        # Report hash
        report_data = {
            "site_id": site_id,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "total_energy_mmbtu": str(total_energy_mmbtu),
            "total_co2e_mt": str(total_co2e_mt),
        }
        report_hash = hashlib.sha256(
            json.dumps(report_data, sort_keys=True).encode()
        ).hexdigest()

        return {
            "report_metadata": {
                "report_id": f"CAR-{timestamp.strftime('%Y%m%d%H%M%S')}",
                "site_id": site_id,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "generated_at": timestamp.isoformat(),
                "report_hash": report_hash,
                "generator_version": self.VERSION,
            },
            "summary": {
                "total_energy_savings_mmbtu": str(total_energy_mmbtu.quantize(Decimal("0.01"))),
                "total_energy_savings_gj": str((total_energy_mmbtu * Decimal("1.0551")).quantize(Decimal("0.01"))),
                "total_co2e_reduction_mt": str(total_co2e_mt.quantize(Decimal("0.01"))),
                "total_cost_savings_usd": str(total_cost_savings.quantize(Decimal("0.01"))) if total_cost_savings else None,
                "primary_fuel": self.primary_fuel.value,
                "boiler_efficiency": str(self.boiler_efficiency),
            },
            "improvements": improvements,
            "reconciliation": reconciliation.to_dict() if reconciliation else None,
            "methodology": {
                "energy_calculation": "DOE Steam Tip Sheet methodologies",
                "emission_factors": "EPA 40 CFR Part 98 Table C-1",
                "gwp_values": "IPCC AR4 (100-year)",
                "scope": EmissionScope.SCOPE_1.value,
            },
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_carbon_accounting_manager(
    fuel_type: FuelType,
    boiler_efficiency: Decimal = Decimal("0.82"),
) -> CarbonAccountingManager:
    """
    Factory function to create carbon accounting manager.

    Args:
        fuel_type: Primary fuel type
        boiler_efficiency: Boiler efficiency

    Returns:
        Configured CarbonAccountingManager
    """
    return CarbonAccountingManager(fuel_type, boiler_efficiency)


def get_emission_factor(fuel_type: FuelType) -> Optional[EmissionFactor]:
    """
    Get EPA emission factor for fuel type.

    Args:
        fuel_type: Fuel type

    Returns:
        EmissionFactor or None
    """
    return EPA_EMISSION_FACTORS.get(fuel_type)


def get_all_emission_factors() -> Dict[FuelType, EmissionFactor]:
    """Get all available emission factors."""
    return EPA_EMISSION_FACTORS.copy()
