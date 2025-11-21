# -*- coding: utf-8 -*-
"""
ROI Calculator for GL-006 HeatRecoveryMaximizer

Comprehensive financial analysis including:
- Capital cost estimation
- Operating cost calculation
- Annual energy savings
- Simple payback period
- Return on Investment (ROI)
- Net Present Value (NPV)
- Internal Rate of Return (IRR)
- Sensitivity analysis

Zero-hallucination design using standard financial formulas.

References:
- Engineering Economics and Financial Accounting, Park & Sharp-Bette
- ASHRAE Simple Payback and Energy Savings Calculations
- EPA Energy Cost Savings Guide
- IRS MACRS Depreciation Tables

Author: GreenLang AI Agent Factory
Created: 2025-11-19
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class EquipmentType(str, Enum):
    """Heat recovery equipment types"""
    SHELL_TUBE_HX = "shell_tube"
    PLATE_HX = "plate"
    SPIRAL_HX = "spiral"
    AIR_PREHEATER = "air_preheater"
    ECONOMIZER = "economizer"
    HEAT_PIPE = "heat_pipe"
    RUN_AROUND_COIL = "run_around_coil"


class DepreciationMethod(str, Enum):
    """Depreciation methods"""
    STRAIGHT_LINE = "straight_line"
    MACRS_5_YEAR = "macrs_5"
    MACRS_7_YEAR = "macrs_7"
    DOUBLE_DECLINING = "double_declining"


@dataclass
class EquipmentCostFactors:
    """Cost factors for different equipment types"""
    # Base cost per kW (USD/kW installed)
    cost_per_kw: Dict[EquipmentType, Tuple[float, float]] = None  # (min, max)

    # Installation multipliers
    installation_factor: float = 1.25  # 25% installation labor
    piping_factor: float = 0.15  # 15% piping/valves
    controls_factor: float = 0.10  # 10% controls/instrumentation
    engineering_factor: float = 0.08  # 8% engineering
    contingency_factor: float = 0.10  # 10% contingency

    def __post_init__(self):
        if self.cost_per_kw is None:
            self.cost_per_kw = {
                EquipmentType.SHELL_TUBE_HX: (400, 800),
                EquipmentType.PLATE_HX: (300, 600),
                EquipmentType.SPIRAL_HX: (500, 900),
                EquipmentType.AIR_PREHEATER: (600, 1000),
                EquipmentType.ECONOMIZER: (700, 1200),
                EquipmentType.HEAT_PIPE: (800, 1400),
                EquipmentType.RUN_AROUND_COIL: (500, 900),
            }


class CapitalCostInput(BaseModel):
    """Input for capital cost estimation"""
    equipment_type: EquipmentType = Field(..., description="Equipment type")
    heat_capacity_kw: float = Field(..., gt=0, description="Heat transfer capacity kW")
    heat_exchanger_area_m2: Optional[float] = Field(None, gt=0, description="Heat exchanger area m²")
    pressure_rating_bar: Optional[float] = Field(10.0, ge=0, description="Design pressure bar")
    material: str = Field("carbon_steel", description="Material: carbon_steel, stainless_steel, titanium")
    include_installation: bool = Field(True, description="Include installation costs")
    include_auxiliary: bool = Field(True, description="Include auxiliary costs")
    location_factor: float = Field(1.0, ge=0.5, le=2.0, description="Geographic cost multiplier")


class OperatingCostInput(BaseModel):
    """Input for operating cost calculation"""
    maintenance_percent_of_capital: float = Field(2.0, ge=0, le=10, description="Annual maintenance %")
    operating_hours_per_year: float = Field(8000, ge=0, le=8760, description="Operating hours/year")
    pumping_power_kw: Optional[float] = Field(None, ge=0, description="Pumping power kW")
    electricity_cost_usd_per_kwh: float = Field(0.10, gt=0, description="Electricity cost $/kWh")
    water_treatment_cost_usd_per_year: Optional[float] = Field(None, ge=0, description="Water treatment $/year")
    insurance_percent: float = Field(0.5, ge=0, le=2, description="Insurance % of capital")


class EnergySavingsInput(BaseModel):
    """Input for energy savings calculation"""
    heat_recovery_kw: float = Field(..., gt=0, description="Heat recovery kW")
    operating_hours_per_year: float = Field(8000, ge=0, le=8760, description="Operating hours/year")
    energy_cost_usd_per_kwh: float = Field(0.08, gt=0, description="Energy cost $/kWh")
    energy_cost_escalation_percent: float = Field(3.0, ge=0, le=15, description="Annual cost escalation %")
    avoided_fuel: str = Field("natural_gas", description="Avoided fuel type")
    system_efficiency: float = Field(0.85, ge=0.5, le=1.0, description="System efficiency")


class FinancialParameters(BaseModel):
    """Financial analysis parameters"""
    discount_rate_percent: float = Field(10.0, ge=0, le=30, description="Discount rate %")
    analysis_period_years: int = Field(15, ge=1, le=30, description="Analysis period years")
    inflation_rate_percent: float = Field(2.5, ge=0, le=10, description="General inflation %")
    depreciation_method: DepreciationMethod = Field(DepreciationMethod.MACRS_7_YEAR)
    tax_rate_percent: float = Field(25.0, ge=0, le=50, description="Corporate tax rate %")
    residual_value_percent: float = Field(10.0, ge=0, le=30, description="Equipment residual value %")


class ROIResult(BaseModel):
    """ROI calculation results"""
    # Capital costs
    equipment_cost_usd: float = Field(..., description="Equipment cost $")
    installation_cost_usd: float = Field(..., description="Installation cost $")
    total_capital_cost_usd: float = Field(..., description="Total capital investment $")

    # Operating costs
    annual_maintenance_cost_usd: float = Field(..., description="Annual maintenance $")
    annual_energy_cost_usd: float = Field(..., description="Annual energy (pumping) $")
    total_annual_operating_cost_usd: float = Field(..., description="Total annual operating $")

    # Savings
    annual_energy_savings_kwh: float = Field(..., description="Annual energy savings kWh")
    annual_energy_cost_savings_usd: float = Field(..., description="Annual cost savings $")
    net_annual_savings_usd: float = Field(..., description="Net annual savings (savings - operating) $")

    # Simple metrics
    simple_payback_years: float = Field(..., description="Simple payback period years")
    simple_roi_percent: float = Field(..., description="Simple ROI %")

    # Advanced metrics
    npv_usd: float = Field(..., description="Net Present Value $")
    irr_percent: float = Field(..., description="Internal Rate of Return %")
    profitability_index: float = Field(..., description="Profitability Index")
    discounted_payback_years: float = Field(..., description="Discounted payback years")

    # Lifecycle metrics
    total_lifecycle_savings_usd: float = Field(..., description="Total lifecycle savings $")
    total_lifecycle_cost_usd: float = Field(..., description="Total lifecycle cost $")
    lifecycle_savings_to_investment_ratio: float = Field(..., description="Lifecycle S/I ratio")

    # Environmental
    lifetime_co2_reduction_tonnes: float = Field(..., description="Lifetime CO2 reduction tonnes")
    cost_per_tonne_co2_avoided_usd: float = Field(..., description="Cost per tonne CO2 avoided $")


class SensitivityResult(BaseModel):
    """Sensitivity analysis results"""
    parameter_name: str = Field(..., description="Parameter being varied")
    base_value: float = Field(..., description="Base case value")
    variation_percent: float = Field(..., description="Variation %")
    npv_change_percent: float = Field(..., description="NPV change %")
    roi_change_percent: float = Field(..., description="ROI change %")
    payback_change_percent: float = Field(..., description="Payback change %")


class ROICalculator:
    """
    Calculate comprehensive financial metrics for heat recovery investments.

    Zero-hallucination approach:
    - Standard engineering economics formulas
    - ASHRAE cost estimation methods
    - EPA energy savings calculations
    - IRS depreciation schedules

    Performance target: <1s for complete analysis
    """

    # MACRS depreciation percentages
    MACRS_5_YEAR = [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576]
    MACRS_7_YEAR = [0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446]

    # Material cost multipliers (relative to carbon steel baseline)
    MATERIAL_MULTIPLIERS = {
        'carbon_steel': 1.0,
        'stainless_steel_304': 2.0,
        'stainless_steel_316': 2.5,
        'titanium': 5.0,
        'copper_alloy': 3.0,
        'nickel_alloy': 4.0
    }

    # CO2 emission factors (kg CO2/kWh) by fuel type
    CO2_FACTORS = {
        'natural_gas': 0.18,
        'fuel_oil': 0.27,
        'coal': 0.34,
        'electricity_grid': 0.50,
        'biomass': 0.05
    }

    def __init__(self):
        """Initialize ROI calculator"""
        self.logger = logging.getLogger(__name__)
        self.cost_factors = EquipmentCostFactors()

    def calculate_roi(
        self,
        capital_input: CapitalCostInput,
        operating_input: OperatingCostInput,
        savings_input: EnergySavingsInput,
        financial_params: FinancialParameters
    ) -> ROIResult:
        """
        Calculate complete ROI analysis.

        Args:
            capital_input: Capital cost inputs
            operating_input: Operating cost inputs
            savings_input: Energy savings inputs
            financial_params: Financial parameters

        Returns:
            ROIResult with all financial metrics
        """
        self.logger.info("Starting ROI calculation")

        # 1. Calculate capital costs
        capital_costs = self._calculate_capital_costs(capital_input)

        # 2. Calculate annual operating costs
        operating_costs = self._calculate_annual_operating_costs(
            operating_input,
            capital_costs['total_capital_cost']
        )

        # 3. Calculate annual savings
        savings = self._calculate_annual_savings(savings_input)

        # 4. Calculate net annual benefit
        net_annual_savings = savings['annual_cost_savings'] - operating_costs['total_annual_cost']

        # 5. Simple payback and ROI
        simple_payback = (capital_costs['total_capital_cost'] / net_annual_savings
                         if net_annual_savings > 0 else 999)
        simple_roi = (net_annual_savings / capital_costs['total_capital_cost'] * 100
                     if capital_costs['total_capital_cost'] > 0 else 0)

        # 6. NPV calculation
        npv = self._calculate_npv(
            capital_costs['total_capital_cost'],
            net_annual_savings,
            financial_params,
            savings_input.energy_cost_escalation_percent
        )

        # 7. IRR calculation
        irr = self._calculate_irr(
            capital_costs['total_capital_cost'],
            net_annual_savings,
            financial_params.analysis_period_years
        )

        # 8. Profitability index
        pi = (npv + capital_costs['total_capital_cost']) / capital_costs['total_capital_cost'] if capital_costs['total_capital_cost'] > 0 else 0

        # 9. Discounted payback
        discounted_payback = self._calculate_discounted_payback(
            capital_costs['total_capital_cost'],
            net_annual_savings,
            financial_params.discount_rate_percent
        )

        # 10. Lifecycle calculations
        total_lifecycle_savings = net_annual_savings * financial_params.analysis_period_years
        total_lifecycle_cost = capital_costs['total_capital_cost'] + (
            operating_costs['total_annual_cost'] * financial_params.analysis_period_years
        )
        lifecycle_ratio = total_lifecycle_savings / total_lifecycle_cost if total_lifecycle_cost > 0 else 0

        # 11. Environmental impact
        co2_factor = self.CO2_FACTORS.get(savings_input.avoided_fuel, 0.18)
        lifetime_co2_reduction = (
            savings['annual_kwh_savings'] * financial_params.analysis_period_years * co2_factor / 1000
        )  # tonnes
        cost_per_tonne_co2 = (
            capital_costs['total_capital_cost'] / lifetime_co2_reduction
            if lifetime_co2_reduction > 0 else 0
        )

        result = ROIResult(
            equipment_cost_usd=capital_costs['equipment_cost'],
            installation_cost_usd=capital_costs['installation_cost'],
            total_capital_cost_usd=capital_costs['total_capital_cost'],
            annual_maintenance_cost_usd=operating_costs['maintenance_cost'],
            annual_energy_cost_usd=operating_costs['energy_cost'],
            total_annual_operating_cost_usd=operating_costs['total_annual_cost'],
            annual_energy_savings_kwh=savings['annual_kwh_savings'],
            annual_energy_cost_savings_usd=savings['annual_cost_savings'],
            net_annual_savings_usd=net_annual_savings,
            simple_payback_years=simple_payback,
            simple_roi_percent=simple_roi,
            npv_usd=npv,
            irr_percent=irr,
            profitability_index=pi,
            discounted_payback_years=discounted_payback,
            total_lifecycle_savings_usd=total_lifecycle_savings,
            total_lifecycle_cost_usd=total_lifecycle_cost,
            lifecycle_savings_to_investment_ratio=lifecycle_ratio,
            lifetime_co2_reduction_tonnes=lifetime_co2_reduction,
            cost_per_tonne_co2_avoided_usd=cost_per_tonne_co2
        )

        self.logger.info(f"ROI calculation complete: NPV=${npv:,.0f}, IRR={irr:.1f}%, Payback={simple_payback:.1f}yr")

        return result

    def _calculate_capital_costs(self, inputs: CapitalCostInput) -> Dict[str, float]:
        """Calculate total capital costs"""
        # Base equipment cost
        cost_range = self.cost_factors.cost_per_kw[inputs.equipment_type]
        base_cost_per_kw = (cost_range[0] + cost_range[1]) / 2  # Use midpoint

        equipment_cost = inputs.heat_capacity_kw * base_cost_per_kw

        # Material multiplier
        material_multiplier = self.MATERIAL_MULTIPLIERS.get(inputs.material, 1.0)
        equipment_cost *= material_multiplier

        # Pressure rating adjustment (higher pressure = higher cost)
        if inputs.pressure_rating_bar and inputs.pressure_rating_bar > 10:
            pressure_factor = 1.0 + ((inputs.pressure_rating_bar - 10) / 100)
            equipment_cost *= pressure_factor

        # Location factor
        equipment_cost *= inputs.location_factor

        # Installation and auxiliary costs
        installation_cost = 0
        if inputs.include_installation:
            installation_cost = equipment_cost * (
                self.cost_factors.installation_factor - 1.0  # -1 because base is 1.0
            )

        auxiliary_cost = 0
        if inputs.include_auxiliary:
            auxiliary_cost = equipment_cost * (
                self.cost_factors.piping_factor +
                self.cost_factors.controls_factor +
                self.cost_factors.engineering_factor +
                self.cost_factors.contingency_factor
            )

        total_capital = equipment_cost + installation_cost + auxiliary_cost

        return {
            'equipment_cost': equipment_cost,
            'installation_cost': installation_cost + auxiliary_cost,
            'total_capital_cost': total_capital
        }

    def _calculate_annual_operating_costs(
        self,
        inputs: OperatingCostInput,
        capital_cost: float
    ) -> Dict[str, float]:
        """Calculate annual operating costs"""
        # Maintenance cost (% of capital)
        maintenance_cost = capital_cost * (inputs.maintenance_percent_of_capital / 100)

        # Energy cost for pumping/auxiliaries
        energy_cost = 0
        if inputs.pumping_power_kw:
            annual_kwh = inputs.pumping_power_kw * inputs.operating_hours_per_year
            energy_cost = annual_kwh * inputs.electricity_cost_usd_per_kwh

        # Water treatment (if applicable)
        water_treatment = inputs.water_treatment_cost_usd_per_year or 0

        # Insurance
        insurance_cost = capital_cost * (inputs.insurance_percent / 100)

        total_annual_cost = maintenance_cost + energy_cost + water_treatment + insurance_cost

        return {
            'maintenance_cost': maintenance_cost,
            'energy_cost': energy_cost,
            'water_treatment_cost': water_treatment,
            'insurance_cost': insurance_cost,
            'total_annual_cost': total_annual_cost
        }

    def _calculate_annual_savings(self, inputs: EnergySavingsInput) -> Dict[str, float]:
        """Calculate annual energy savings"""
        # Annual kWh savings (accounting for system efficiency)
        annual_kwh = (inputs.heat_recovery_kw *
                     inputs.operating_hours_per_year *
                     inputs.system_efficiency)

        # Annual cost savings
        annual_cost_savings = annual_kwh * inputs.energy_cost_usd_per_kwh

        return {
            'annual_kwh_savings': annual_kwh,
            'annual_cost_savings': annual_cost_savings
        }

    def _calculate_npv(
        self,
        capital_cost: float,
        annual_savings: float,
        financial_params: FinancialParameters,
        escalation_rate: float
    ) -> float:
        """
        Calculate Net Present Value.

        NPV = -Initial Investment + Σ(Cash Flow_t / (1 + discount_rate)^t)
        """
        discount_rate = financial_params.discount_rate_percent / 100
        npv = -capital_cost  # Initial investment (negative cash flow)

        for year in range(1, financial_params.analysis_period_years + 1):
            # Escalate savings
            escalated_savings = annual_savings * ((1 + escalation_rate / 100) ** (year - 1))

            # Discount to present value
            pv = escalated_savings / ((1 + discount_rate) ** year)
            npv += pv

        # Add residual value (discounted)
        residual_value = capital_cost * (financial_params.residual_value_percent / 100)
        residual_pv = residual_value / ((1 + discount_rate) ** financial_params.analysis_period_years)
        npv += residual_pv

        return npv

    def _calculate_irr(
        self,
        capital_cost: float,
        annual_savings: float,
        analysis_period: int
    ) -> float:
        """
        Calculate Internal Rate of Return using Newton-Raphson method.

        IRR is the discount rate where NPV = 0.
        """
        # Initial guess
        irr = 0.10  # 10%

        # Newton-Raphson iteration
        for iteration in range(50):  # Max 50 iterations
            npv = -capital_cost
            npv_derivative = 0

            for year in range(1, analysis_period + 1):
                discount_factor = (1 + irr) ** year
                npv += annual_savings / discount_factor
                npv_derivative -= year * annual_savings / ((1 + irr) ** (year + 1))

            # Check convergence
            if abs(npv) < 0.01:  # $0.01 tolerance
                break

            # Newton-Raphson update
            if npv_derivative != 0:
                irr = irr - npv / npv_derivative
            else:
                break

            # Bounds check
            if irr < -0.99:
                irr = -0.99
            elif irr > 10.0:
                irr = 10.0
                break

        return irr * 100  # Convert to percentage

    def _calculate_discounted_payback(
        self,
        capital_cost: float,
        annual_savings: float,
        discount_rate: float
    ) -> float:
        """Calculate discounted payback period"""
        discount_rate_decimal = discount_rate / 100
        cumulative_pv = 0

        for year in range(1, 51):  # Max 50 years
            pv = annual_savings / ((1 + discount_rate_decimal) ** year)
            cumulative_pv += pv

            if cumulative_pv >= capital_cost:
                # Interpolate for fractional year
                previous_cumulative = cumulative_pv - pv
                fraction = (capital_cost - previous_cumulative) / pv
                return year - 1 + fraction

        return 999  # No payback within 50 years

    def perform_sensitivity_analysis(
        self,
        base_case: ROIResult,
        capital_input: CapitalCostInput,
        operating_input: OperatingCostInput,
        savings_input: EnergySavingsInput,
        financial_params: FinancialParameters,
        variation_percent: float = 20.0
    ) -> List[SensitivityResult]:
        """
        Perform sensitivity analysis on key parameters.

        Args:
            base_case: Base case ROI result
            capital_input: Base capital inputs
            operating_input: Base operating inputs
            savings_input: Base savings inputs
            financial_params: Base financial parameters
            variation_percent: % variation for sensitivity (+/-)

        Returns:
            List of sensitivity results for each parameter
        """
        sensitivity_results = []

        # Parameters to analyze
        parameters = [
            ('energy_cost', savings_input.energy_cost_usd_per_kwh, 'energy cost'),
            ('capital_cost_multiplier', 1.0, 'capital cost'),
            ('discount_rate', financial_params.discount_rate_percent, 'discount rate'),
            ('operating_hours', savings_input.operating_hours_per_year, 'operating hours'),
        ]

        for param_name, base_value, display_name in parameters:
            # Vary parameter up
            varied_input_up = self._vary_parameter(
                capital_input, operating_input, savings_input, financial_params,
                param_name, base_value * (1 + variation_percent / 100)
            )
            result_up = self.calculate_roi(*varied_input_up)

            # Vary parameter down
            varied_input_down = self._vary_parameter(
                capital_input, operating_input, savings_input, financial_params,
                param_name, base_value * (1 - variation_percent / 100)
            )
            result_down = self.calculate_roi(*varied_input_down)

            # Calculate sensitivity
            npv_change_up = ((result_up.npv_usd - base_case.npv_usd) / base_case.npv_usd * 100) if base_case.npv_usd != 0 else 0
            roi_change_up = ((result_up.simple_roi_percent - base_case.simple_roi_percent) / base_case.simple_roi_percent * 100) if base_case.simple_roi_percent != 0 else 0

            sensitivity = SensitivityResult(
                parameter_name=display_name,
                base_value=base_value,
                variation_percent=variation_percent,
                npv_change_percent=npv_change_up,
                roi_change_percent=roi_change_up,
                payback_change_percent=0  # Simplified
            )

            sensitivity_results.append(sensitivity)

        return sensitivity_results

    def _vary_parameter(
        self,
        capital_input: CapitalCostInput,
        operating_input: OperatingCostInput,
        savings_input: EnergySavingsInput,
        financial_params: FinancialParameters,
        param_name: str,
        new_value: float
    ) -> Tuple:
        """Helper to vary a specific parameter"""
        # Create copies
        capital = capital_input.copy(deep=True)
        operating = operating_input.copy(deep=True)
        savings = savings_input.copy(deep=True)
        financial = financial_params.copy(deep=True)

        # Modify the specified parameter
        if param_name == 'energy_cost':
            savings.energy_cost_usd_per_kwh = new_value
        elif param_name == 'capital_cost_multiplier':
            capital.location_factor = new_value
        elif param_name == 'discount_rate':
            financial.discount_rate_percent = new_value
        elif param_name == 'operating_hours':
            savings.operating_hours_per_year = new_value

        return (capital, operating, savings, financial)


# Example usage
if __name__ == "__main__":
    calculator = ROICalculator()

    # Example: Shell-and-tube heat exchanger
    capital = CapitalCostInput(
        equipment_type=EquipmentType.SHELL_TUBE_HX,
        heat_capacity_kw=500.0,
        material="stainless_steel_304",
        location_factor=1.0
    )

    operating = OperatingCostInput(
        maintenance_percent_of_capital=2.0,
        operating_hours_per_year=8000,
        pumping_power_kw=5.0,
        electricity_cost_usd_per_kwh=0.10
    )

    savings = EnergySavingsInput(
        heat_recovery_kw=500.0,
        operating_hours_per_year=8000,
        energy_cost_usd_per_kwh=0.08,
        energy_cost_escalation_percent=3.0,
        system_efficiency=0.85
    )

    financial = FinancialParameters(
        discount_rate_percent=10.0,
        analysis_period_years=15,
        inflation_rate_percent=2.5
    )

    result = calculator.calculate_roi(capital, operating, savings, financial)

    print(f"ROI Analysis Results:")
    print(f"  Total Capital Cost: ${result.total_capital_cost_usd:,.0f}")
    print(f"  Annual Savings: ${result.net_annual_savings_usd:,.0f}")
    print(f"  Simple Payback: {result.simple_payback_years:.1f} years")
    print(f"  Simple ROI: {result.simple_roi_percent:.1f}%")
    print(f"  NPV: ${result.npv_usd:,.0f}")
    print(f"  IRR: {result.irr_percent:.1f}%")
    print(f"  Profitability Index: {result.profitability_index:.2f}")
    print(f"  Lifetime CO2 Reduction: {result.lifetime_co2_reduction_tonnes:,.0f} tonnes")
