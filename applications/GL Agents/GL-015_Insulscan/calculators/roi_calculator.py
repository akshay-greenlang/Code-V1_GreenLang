"""
GL-015 INSULSCAN ROI Calculator

ZERO-HALLUCINATION financial analysis calculator for insulation investments.

Features:
    - Annual energy savings calculations
    - Payback period analysis
    - Net Present Value (NPV) calculation
    - CO2 reduction quantification
    - Lifecycle cost analysis
    - Full provenance tracking with SHA-256 hashes

Financial Standards: ISO 15686-5 (Buildings and constructed assets - Life cycle costing)

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json


class EnergyType(Enum):
    """Energy source types for cost and emission calculations."""
    NATURAL_GAS = "natural_gas"
    ELECTRICITY = "electricity"
    FUEL_OIL = "fuel_oil"
    COAL = "coal"
    PROPANE = "propane"
    BIOMASS = "biomass"
    STEAM = "steam"


@dataclass(frozen=True)
class EnergyCostData:
    """
    Energy cost and emission factor data.

    Attributes:
        price_per_kwh: Energy price in USD per kWh
        emission_factor_kg_co2_per_kwh: CO2 emissions per kWh
        price_volatility_annual: Annual price volatility (std dev)
        source: Data source reference
    """
    price_per_kwh: Decimal
    emission_factor_kg_co2_per_kwh: Decimal
    price_volatility_annual: Decimal
    source: str


@dataclass(frozen=True)
class ROIResult:
    """
    Immutable result container for ROI calculations.

    Attributes:
        annual_energy_savings_kwh: Energy saved per year (kWh)
        annual_cost_savings_usd: Cost savings per year (USD)
        simple_payback_years: Simple payback period (years)
        npv_usd: Net Present Value (USD)
        irr_percent: Internal Rate of Return (%)
        co2_reduction_kg: Annual CO2 reduction (kg)
        provenance_hash: SHA-256 hash for audit trail
        calculation_inputs: Dictionary of all input parameters
    """
    annual_energy_savings_kwh: Decimal
    annual_cost_savings_usd: Decimal
    simple_payback_years: Decimal
    npv_usd: Decimal
    irr_percent: Optional[Decimal]
    co2_reduction_kg: Decimal
    provenance_hash: str
    calculation_inputs: Dict[str, Any]


@dataclass(frozen=True)
class LifecycleCostResult:
    """
    Lifecycle cost analysis result.

    Attributes:
        total_lifecycle_cost_usd: Total cost over analysis period
        installation_cost_usd: Initial installation cost
        maintenance_cost_npv_usd: NPV of maintenance costs
        energy_cost_npv_usd: NPV of energy costs
        replacement_cost_npv_usd: NPV of replacement costs
        salvage_value_npv_usd: NPV of end-of-life salvage value
        provenance_hash: SHA-256 hash for audit trail
    """
    total_lifecycle_cost_usd: Decimal
    installation_cost_usd: Decimal
    maintenance_cost_npv_usd: Decimal
    energy_cost_npv_usd: Decimal
    replacement_cost_npv_usd: Decimal
    salvage_value_npv_usd: Decimal
    provenance_hash: str
    calculation_inputs: Dict[str, Any]


class InsulationROICalculator:
    """
    Financial analysis calculator for insulation investments.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations use deterministic financial formulas
    - Energy prices from configurable database
    - Emission factors from EPA/DEFRA sources
    - Complete audit trail with provenance hashing

    Example Usage:
        >>> calc = InsulationROICalculator()
        >>> savings = calc.calculate_annual_energy_savings(
        ...     current_heat_loss_w=5000,
        ...     improved_heat_loss_w=1000,
        ...     operating_hours=8760
        ... )
        >>> float(savings) > 30000
        True

    Determinism Test:
        >>> calc = InsulationROICalculator()
        >>> r1 = calc.calculate_full_roi(5000, 1000, 8760, 10000)
        >>> r2 = calc.calculate_full_roi(5000, 1000, 8760, 10000)
        >>> r1.npv_usd == r2.npv_usd
        True
        >>> r1.provenance_hash == r2.provenance_hash
        True
    """

    # Energy Cost and Emission Factor Database
    # Sources: EIA (US), BEIS (UK), EPA eGRID
    # Prices in USD/kWh, emissions in kg CO2/kWh
    ENERGY_DATABASE: Dict[EnergyType, EnergyCostData] = {
        EnergyType.NATURAL_GAS: EnergyCostData(
            price_per_kwh=Decimal("0.035"),
            emission_factor_kg_co2_per_kwh=Decimal("0.185"),
            price_volatility_annual=Decimal("0.15"),
            source="EIA 2024 Industrial Natural Gas Prices"
        ),
        EnergyType.ELECTRICITY: EnergyCostData(
            price_per_kwh=Decimal("0.085"),
            emission_factor_kg_co2_per_kwh=Decimal("0.417"),
            price_volatility_annual=Decimal("0.08"),
            source="EIA 2024 Industrial Electricity Prices, EPA eGRID"
        ),
        EnergyType.FUEL_OIL: EnergyCostData(
            price_per_kwh=Decimal("0.055"),
            emission_factor_kg_co2_per_kwh=Decimal("0.267"),
            price_volatility_annual=Decimal("0.25"),
            source="EIA 2024 No. 2 Fuel Oil Prices"
        ),
        EnergyType.COAL: EnergyCostData(
            price_per_kwh=Decimal("0.025"),
            emission_factor_kg_co2_per_kwh=Decimal("0.340"),
            price_volatility_annual=Decimal("0.10"),
            source="EIA 2024 Industrial Coal Prices"
        ),
        EnergyType.PROPANE: EnergyCostData(
            price_per_kwh=Decimal("0.065"),
            emission_factor_kg_co2_per_kwh=Decimal("0.215"),
            price_volatility_annual=Decimal("0.20"),
            source="EIA 2024 Propane Prices"
        ),
        EnergyType.BIOMASS: EnergyCostData(
            price_per_kwh=Decimal("0.045"),
            emission_factor_kg_co2_per_kwh=Decimal("0.039"),  # Biogenic, often counted as zero
            price_volatility_annual=Decimal("0.12"),
            source="EIA 2024 Biomass Prices"
        ),
        EnergyType.STEAM: EnergyCostData(
            price_per_kwh=Decimal("0.050"),
            emission_factor_kg_co2_per_kwh=Decimal("0.200"),  # Depends on generation source
            price_volatility_annual=Decimal("0.10"),
            source="Industrial steam cost estimates"
        ),
    }

    # Default financial parameters
    DEFAULT_DISCOUNT_RATE = Decimal("0.08")  # 8% real discount rate
    DEFAULT_ANALYSIS_YEARS = 20
    DEFAULT_INFLATION_RATE = Decimal("0.025")  # 2.5% general inflation
    DEFAULT_ENERGY_ESCALATION = Decimal("0.03")  # 3% energy price escalation

    def __init__(
        self,
        precision: int = 2,
        custom_energy_prices: Optional[Dict[EnergyType, Decimal]] = None
    ):
        """
        Initialize calculator with specified precision and optional custom prices.

        Args:
            precision: Number of decimal places for financial outputs
            custom_energy_prices: Optional custom energy prices (USD/kWh)
        """
        self.precision = precision
        self._quantize_str = "0." + "0" * precision

        # Apply custom prices if provided
        if custom_energy_prices:
            for energy_type, price in custom_energy_prices.items():
                if energy_type in self.ENERGY_DATABASE:
                    original = self.ENERGY_DATABASE[energy_type]
                    self.ENERGY_DATABASE[energy_type] = EnergyCostData(
                        price_per_kwh=Decimal(str(price)),
                        emission_factor_kg_co2_per_kwh=original.emission_factor_kg_co2_per_kwh,
                        price_volatility_annual=original.price_volatility_annual,
                        source=f"Custom price override ({original.source})"
                    )

    def calculate_annual_energy_savings(
        self,
        current_heat_loss_w: float,
        improved_heat_loss_w: float,
        operating_hours: float
    ) -> Decimal:
        """
        Calculate annual energy savings in kWh.

        Formula: savings_kwh = (current_loss - improved_loss) * hours / 1000

        Args:
            current_heat_loss_w: Current heat loss in watts
            improved_heat_loss_w: Heat loss after improvement in watts
            operating_hours: Annual operating hours

        Returns:
            Annual energy savings in kWh

        Example:
            >>> calc = InsulationROICalculator()
            >>> savings = calc.calculate_annual_energy_savings(
            ...     current_heat_loss_w=5000,
            ...     improved_heat_loss_w=1000,
            ...     operating_hours=8760
            ... )
            >>> float(savings)
            35040.0

        Example - Zero Improvement:
            >>> calc = InsulationROICalculator()
            >>> savings = calc.calculate_annual_energy_savings(1000, 1000, 8760)
            >>> float(savings)
            0.0
        """
        current = Decimal(str(current_heat_loss_w))
        improved = Decimal(str(improved_heat_loss_w))
        hours = Decimal(str(operating_hours))

        # Validate inputs
        self._validate_non_negative("current_heat_loss_w", current)
        self._validate_non_negative("improved_heat_loss_w", improved)
        self._validate_positive("operating_hours", hours)

        if improved > current:
            raise ValueError(
                f"Improved heat loss ({improved}W) cannot exceed current ({current}W)"
            )

        # Calculate savings: (W * hours) / 1000 = kWh
        savings_kwh = (current - improved) * hours / Decimal("1000")

        return self._apply_precision(savings_kwh)

    def calculate_annual_cost_savings(
        self,
        energy_savings_kwh: float,
        energy_type: EnergyType = EnergyType.NATURAL_GAS,
        boiler_efficiency: float = 0.85
    ) -> Decimal:
        """
        Calculate annual cost savings in USD.

        Formula: cost_savings = energy_savings / efficiency * price_per_kwh

        Args:
            energy_savings_kwh: Annual energy savings in kWh
            energy_type: Type of energy source
            boiler_efficiency: Boiler/heating system efficiency (0-1)

        Returns:
            Annual cost savings in USD

        Example - Natural Gas:
            >>> calc = InsulationROICalculator()
            >>> savings = calc.calculate_annual_cost_savings(
            ...     energy_savings_kwh=35040,
            ...     energy_type=EnergyType.NATURAL_GAS,
            ...     boiler_efficiency=0.85
            ... )
            >>> 1000 < float(savings) < 2000
            True

        Example - Electricity:
            >>> calc = InsulationROICalculator()
            >>> savings = calc.calculate_annual_cost_savings(
            ...     energy_savings_kwh=35040,
            ...     energy_type=EnergyType.ELECTRICITY,
            ...     boiler_efficiency=1.0
            ... )
            >>> float(savings) > 2500
            True
        """
        savings = Decimal(str(energy_savings_kwh))
        efficiency = Decimal(str(boiler_efficiency))

        # Validate inputs
        self._validate_non_negative("energy_savings_kwh", savings)
        self._validate_range("boiler_efficiency", efficiency, Decimal("0.5"), Decimal("1.0"))

        # Get energy price
        energy_data = self.ENERGY_DATABASE[energy_type]

        # Calculate fuel required (accounting for efficiency)
        fuel_required_kwh = savings / efficiency

        # Calculate cost savings
        cost_savings = fuel_required_kwh * energy_data.price_per_kwh

        return self._apply_precision(cost_savings)

    def calculate_payback_period(
        self,
        installation_cost: float,
        annual_savings: float
    ) -> Decimal:
        """
        Calculate simple payback period in years.

        Formula: payback = installation_cost / annual_savings

        Args:
            installation_cost: Total installation cost in USD
            annual_savings: Annual cost savings in USD

        Returns:
            Simple payback period in years

        Example:
            >>> calc = InsulationROICalculator()
            >>> payback = calc.calculate_payback_period(
            ...     installation_cost=10000,
            ...     annual_savings=2500
            ... )
            >>> float(payback)
            4.0

        Example - Short Payback:
            >>> calc = InsulationROICalculator()
            >>> payback = calc.calculate_payback_period(5000, 5000)
            >>> float(payback)
            1.0
        """
        cost = Decimal(str(installation_cost))
        savings = Decimal(str(annual_savings))

        # Validate inputs
        self._validate_non_negative("installation_cost", cost)
        self._validate_positive("annual_savings", savings)

        payback = cost / savings

        return self._apply_precision(payback)

    def calculate_npv(
        self,
        installation_cost: float,
        annual_savings: float,
        years: int = 20,
        discount_rate: float = 0.08,
        energy_escalation_rate: float = 0.03
    ) -> Decimal:
        """
        Calculate Net Present Value of insulation investment.

        NPV = -C_0 + sum(CF_t / (1+r)^t) for t=1 to n
        Where CF_t = annual_savings * (1 + escalation)^t

        Args:
            installation_cost: Initial investment in USD
            annual_savings: First year savings in USD
            years: Analysis period in years
            discount_rate: Real discount rate (decimal)
            energy_escalation_rate: Annual energy price escalation (decimal)

        Returns:
            Net Present Value in USD

        Example - Positive NPV:
            >>> calc = InsulationROICalculator()
            >>> npv = calc.calculate_npv(
            ...     installation_cost=10000,
            ...     annual_savings=2000,
            ...     years=20,
            ...     discount_rate=0.08
            ... )
            >>> float(npv) > 10000
            True

        Example - Break-even Analysis:
            >>> calc = InsulationROICalculator()
            >>> npv = calc.calculate_npv(10000, 500, 20, 0.08)
            >>> float(npv) < 0
            True
        """
        C_0 = Decimal(str(installation_cost))
        CF_base = Decimal(str(annual_savings))
        r = Decimal(str(discount_rate))
        g = Decimal(str(energy_escalation_rate))

        # Validate inputs
        self._validate_non_negative("installation_cost", C_0)
        self._validate_non_negative("annual_savings", CF_base)
        self._validate_range("discount_rate", r, Decimal("0"), Decimal("0.30"))
        self._validate_range("years", Decimal(str(years)), Decimal("1"), Decimal("50"))

        # Calculate NPV
        npv = -C_0

        for t in range(1, years + 1):
            # Cash flow with escalation
            CF_t = CF_base * ((Decimal("1") + g) ** t)
            # Present value
            PV_t = CF_t / ((Decimal("1") + r) ** t)
            npv += PV_t

        return self._apply_precision(npv)

    def calculate_irr(
        self,
        installation_cost: float,
        annual_savings: float,
        years: int = 20,
        energy_escalation_rate: float = 0.03,
        tolerance: float = 0.0001,
        max_iterations: int = 100
    ) -> Optional[Decimal]:
        """
        Calculate Internal Rate of Return using Newton-Raphson method.

        IRR is the discount rate that makes NPV = 0

        Args:
            installation_cost: Initial investment in USD
            annual_savings: First year savings in USD
            years: Analysis period
            energy_escalation_rate: Annual energy price escalation
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations

        Returns:
            IRR as decimal (or None if no convergence)

        Example:
            >>> calc = InsulationROICalculator()
            >>> irr = calc.calculate_irr(10000, 2000, 20)
            >>> irr is not None
            True
            >>> 0.15 < float(irr) < 0.25
            True
        """
        C_0 = Decimal(str(installation_cost))
        CF_base = Decimal(str(annual_savings))
        g = Decimal(str(energy_escalation_rate))

        # Initial guess based on simple payback
        if CF_base <= 0:
            return None

        simple_return = CF_base / C_0
        irr_guess = float(simple_return)

        # Newton-Raphson iteration
        for _ in range(max_iterations):
            npv = -float(C_0)
            npv_derivative = Decimal("0")

            for t in range(1, years + 1):
                CF_t = float(CF_base) * ((1 + float(g)) ** t)
                discount = (1 + irr_guess) ** t

                npv += CF_t / discount
                npv_derivative -= t * CF_t / ((1 + irr_guess) ** (t + 1))

            if abs(npv_derivative) < 1e-10:
                break

            delta = npv / float(npv_derivative)
            irr_guess -= delta

            if abs(delta) < tolerance:
                return self._apply_precision(Decimal(str(irr_guess)))

            # Bounds check
            if irr_guess < -0.99 or irr_guess > 10:
                return None

        return None

    def calculate_co2_reduction(
        self,
        energy_savings_kwh: float,
        energy_type: EnergyType = EnergyType.NATURAL_GAS,
        boiler_efficiency: float = 0.85
    ) -> Decimal:
        """
        Calculate annual CO2 emission reduction.

        Formula: CO2_kg = fuel_kwh * emission_factor

        Args:
            energy_savings_kwh: Annual energy savings (kWh thermal)
            energy_type: Type of energy source
            boiler_efficiency: Heating system efficiency

        Returns:
            Annual CO2 reduction in kg

        Example - Natural Gas:
            >>> calc = InsulationROICalculator()
            >>> co2 = calc.calculate_co2_reduction(
            ...     energy_savings_kwh=35040,
            ...     energy_type=EnergyType.NATURAL_GAS,
            ...     boiler_efficiency=0.85
            ... )
            >>> float(co2) > 7000
            True

        Example - Coal (Higher Emissions):
            >>> calc = InsulationROICalculator()
            >>> co2_gas = calc.calculate_co2_reduction(35040, EnergyType.NATURAL_GAS)
            >>> co2_coal = calc.calculate_co2_reduction(35040, EnergyType.COAL)
            >>> float(co2_coal) > float(co2_gas)
            True
        """
        savings = Decimal(str(energy_savings_kwh))
        efficiency = Decimal(str(boiler_efficiency))

        # Validate inputs
        self._validate_non_negative("energy_savings_kwh", savings)
        self._validate_range("boiler_efficiency", efficiency, Decimal("0.5"), Decimal("1.0"))

        # Get emission factor
        energy_data = self.ENERGY_DATABASE[energy_type]

        # Calculate fuel required
        fuel_required_kwh = savings / efficiency

        # Calculate CO2 reduction
        co2_kg = fuel_required_kwh * energy_data.emission_factor_kg_co2_per_kwh

        return self._apply_precision(co2_kg)

    def calculate_full_roi(
        self,
        current_heat_loss_w: float,
        improved_heat_loss_w: float,
        operating_hours: float,
        installation_cost: float,
        energy_type: EnergyType = EnergyType.NATURAL_GAS,
        boiler_efficiency: float = 0.85,
        analysis_years: int = 20,
        discount_rate: float = 0.08,
        energy_escalation_rate: float = 0.03
    ) -> ROIResult:
        """
        Perform complete ROI analysis with provenance tracking.

        Args:
            current_heat_loss_w: Current heat loss in watts
            improved_heat_loss_w: Heat loss after improvement in watts
            operating_hours: Annual operating hours
            installation_cost: Total installation cost in USD
            energy_type: Type of energy source
            boiler_efficiency: Heating system efficiency
            analysis_years: NPV analysis period
            discount_rate: Real discount rate
            energy_escalation_rate: Energy price escalation rate

        Returns:
            ROIResult with complete analysis and provenance

        Example:
            >>> calc = InsulationROICalculator()
            >>> result = calc.calculate_full_roi(
            ...     current_heat_loss_w=5000,
            ...     improved_heat_loss_w=1000,
            ...     operating_hours=8760,
            ...     installation_cost=10000,
            ...     energy_type=EnergyType.NATURAL_GAS
            ... )
            >>> float(result.annual_energy_savings_kwh) > 30000
            True
            >>> float(result.simple_payback_years) < 10
            True
            >>> float(result.npv_usd) > 0
            True
        """
        # Calculate energy savings
        energy_savings = self.calculate_annual_energy_savings(
            current_heat_loss_w, improved_heat_loss_w, operating_hours
        )

        # Calculate cost savings
        cost_savings = self.calculate_annual_cost_savings(
            float(energy_savings), energy_type, boiler_efficiency
        )

        # Calculate payback period
        if cost_savings > 0:
            payback = self.calculate_payback_period(installation_cost, float(cost_savings))
        else:
            payback = Decimal("999")  # Infinite payback

        # Calculate NPV
        npv = self.calculate_npv(
            installation_cost, float(cost_savings), analysis_years,
            discount_rate, energy_escalation_rate
        )

        # Calculate IRR
        irr = self.calculate_irr(
            installation_cost, float(cost_savings), analysis_years,
            energy_escalation_rate
        )

        # Calculate CO2 reduction
        co2_reduction = self.calculate_co2_reduction(
            float(energy_savings), energy_type, boiler_efficiency
        )

        # Build provenance
        inputs = {
            "current_heat_loss_w": str(current_heat_loss_w),
            "improved_heat_loss_w": str(improved_heat_loss_w),
            "operating_hours": str(operating_hours),
            "installation_cost": str(installation_cost),
            "energy_type": energy_type.value,
            "boiler_efficiency": str(boiler_efficiency),
            "analysis_years": analysis_years,
            "discount_rate": str(discount_rate),
            "energy_escalation_rate": str(energy_escalation_rate),
            "energy_price_usd_kwh": str(self.ENERGY_DATABASE[energy_type].price_per_kwh),
            "emission_factor_kg_co2_kwh": str(
                self.ENERGY_DATABASE[energy_type].emission_factor_kg_co2_per_kwh
            ),
        }

        provenance_hash = self._calculate_provenance_hash(
            "full_roi_analysis", inputs,
            f"npv={npv},payback={payback},co2={co2_reduction}"
        )

        return ROIResult(
            annual_energy_savings_kwh=energy_savings,
            annual_cost_savings_usd=cost_savings,
            simple_payback_years=payback,
            npv_usd=npv,
            irr_percent=irr * Decimal("100") if irr else None,
            co2_reduction_kg=co2_reduction,
            provenance_hash=provenance_hash,
            calculation_inputs=inputs
        )

    def calculate_lifecycle_cost(
        self,
        installation_cost: float,
        annual_maintenance_cost: float,
        annual_energy_cost: float,
        analysis_years: int = 20,
        discount_rate: float = 0.08,
        replacement_year: Optional[int] = None,
        replacement_cost: Optional[float] = None,
        salvage_value: float = 0.0
    ) -> LifecycleCostResult:
        """
        Perform lifecycle cost analysis per ISO 15686-5.

        LCC = C_initial + PV(maintenance) + PV(energy) + PV(replacement) - PV(salvage)

        Args:
            installation_cost: Initial installation cost (USD)
            annual_maintenance_cost: Annual maintenance cost (USD)
            annual_energy_cost: Annual energy cost (USD)
            analysis_years: Analysis period (years)
            discount_rate: Real discount rate
            replacement_year: Year of major replacement (optional)
            replacement_cost: Cost of replacement (USD)
            salvage_value: End-of-life salvage value (USD)

        Returns:
            LifecycleCostResult with cost breakdown

        Example:
            >>> calc = InsulationROICalculator()
            >>> result = calc.calculate_lifecycle_cost(
            ...     installation_cost=10000,
            ...     annual_maintenance_cost=200,
            ...     annual_energy_cost=1500,
            ...     analysis_years=20,
            ...     discount_rate=0.08
            ... )
            >>> float(result.total_lifecycle_cost_usd) > 20000
            True
            >>> float(result.installation_cost_usd) == 10000
            True
        """
        C_0 = Decimal(str(installation_cost))
        M = Decimal(str(annual_maintenance_cost))
        E = Decimal(str(annual_energy_cost))
        r = Decimal(str(discount_rate))
        S = Decimal(str(salvage_value))

        # Validate inputs
        self._validate_non_negative("installation_cost", C_0)
        self._validate_non_negative("annual_maintenance_cost", M)
        self._validate_non_negative("annual_energy_cost", E)

        # Calculate NPV of maintenance costs
        maintenance_npv = Decimal("0")
        for t in range(1, analysis_years + 1):
            maintenance_npv += M / ((Decimal("1") + r) ** t)

        # Calculate NPV of energy costs (with 3% escalation)
        energy_escalation = Decimal("0.03")
        energy_npv = Decimal("0")
        for t in range(1, analysis_years + 1):
            E_t = E * ((Decimal("1") + energy_escalation) ** t)
            energy_npv += E_t / ((Decimal("1") + r) ** t)

        # Calculate NPV of replacement costs
        replacement_npv = Decimal("0")
        if replacement_year and replacement_cost:
            R = Decimal(str(replacement_cost))
            t = replacement_year
            replacement_npv = R / ((Decimal("1") + r) ** t)

        # Calculate NPV of salvage value
        salvage_npv = S / ((Decimal("1") + r) ** analysis_years)

        # Total lifecycle cost
        total_lcc = C_0 + maintenance_npv + energy_npv + replacement_npv - salvage_npv

        # Apply precision
        total_lcc = self._apply_precision(total_lcc)
        maintenance_npv = self._apply_precision(maintenance_npv)
        energy_npv = self._apply_precision(energy_npv)
        replacement_npv = self._apply_precision(replacement_npv)
        salvage_npv = self._apply_precision(salvage_npv)

        # Build provenance
        inputs = {
            "installation_cost": str(installation_cost),
            "annual_maintenance_cost": str(annual_maintenance_cost),
            "annual_energy_cost": str(annual_energy_cost),
            "analysis_years": analysis_years,
            "discount_rate": str(discount_rate),
            "replacement_year": replacement_year,
            "replacement_cost": str(replacement_cost) if replacement_cost else None,
            "salvage_value": str(salvage_value),
        }

        provenance_hash = self._calculate_provenance_hash(
            "lifecycle_cost_analysis", inputs, str(total_lcc)
        )

        return LifecycleCostResult(
            total_lifecycle_cost_usd=total_lcc,
            installation_cost_usd=self._apply_precision(C_0),
            maintenance_cost_npv_usd=maintenance_npv,
            energy_cost_npv_usd=energy_npv,
            replacement_cost_npv_usd=replacement_npv,
            salvage_value_npv_usd=salvage_npv,
            provenance_hash=provenance_hash,
            calculation_inputs=inputs
        )

    def compare_alternatives(
        self,
        alternatives: List[Dict[str, Any]],
        operating_hours: float,
        energy_type: EnergyType = EnergyType.NATURAL_GAS,
        analysis_years: int = 20
    ) -> List[Tuple[str, ROIResult]]:
        """
        Compare multiple insulation alternatives.

        Args:
            alternatives: List of alternative specifications with keys:
                - name: Alternative name
                - current_heat_loss_w: Current heat loss
                - improved_heat_loss_w: Improved heat loss
                - installation_cost: Cost in USD
            operating_hours: Annual operating hours
            energy_type: Energy source type
            analysis_years: Analysis period

        Returns:
            List of (name, ROIResult) tuples sorted by NPV (best first)

        Example:
            >>> calc = InsulationROICalculator()
            >>> alternatives = [
            ...     {"name": "Option A", "current_heat_loss_w": 5000,
            ...      "improved_heat_loss_w": 1000, "installation_cost": 8000},
            ...     {"name": "Option B", "current_heat_loss_w": 5000,
            ...      "improved_heat_loss_w": 500, "installation_cost": 15000}
            ... ]
            >>> results = calc.compare_alternatives(alternatives, 8760)
            >>> len(results) == 2
            True
            >>> results[0][1].npv_usd >= results[1][1].npv_usd
            True
        """
        results = []

        for alt in alternatives:
            roi = self.calculate_full_roi(
                current_heat_loss_w=alt["current_heat_loss_w"],
                improved_heat_loss_w=alt["improved_heat_loss_w"],
                operating_hours=operating_hours,
                installation_cost=alt["installation_cost"],
                energy_type=energy_type,
                analysis_years=analysis_years
            )
            results.append((alt["name"], roi))

        # Sort by NPV (highest first)
        results.sort(key=lambda x: x[1].npv_usd, reverse=True)

        return results

    def _validate_positive(self, name: str, value: Decimal) -> None:
        """Validate value is positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    def _validate_non_negative(self, name: str, value: Decimal) -> None:
        """Validate value is non-negative."""
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    def _validate_range(
        self,
        name: str,
        value: Decimal,
        min_val: Decimal,
        max_val: Decimal
    ) -> None:
        """Validate value is within range."""
        if value < min_val or value > max_val:
            raise ValueError(
                f"{name} must be between {min_val} and {max_val}, got {value}"
            )

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply configured precision using ROUND_HALF_UP."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance_hash(
        self,
        calculation_type: str,
        inputs: Dict[str, Any],
        result: str
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        provenance_data = {
            "calculator": "InsulationROICalculator",
            "version": "1.0.0",
            "standard": "ISO 15686-5",
            "calculation_type": calculation_type,
            "inputs": inputs,
            "result": result,
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
