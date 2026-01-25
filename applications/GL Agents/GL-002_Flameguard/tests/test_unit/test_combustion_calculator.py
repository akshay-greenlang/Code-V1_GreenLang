"""
GL-002 Flameguard - Combustion Calculator Unit Tests

Comprehensive unit tests for combustion calculations including:
- Stoichiometric air calculations
- Excess air from O2 measurement
- Combustion product mass flows
- Heat release calculations
- CO/NOx formation correlations

Reference Standards:
- ASME PTC 4.1 (Fired Steam Generators)
- EPA Method 19 (SO2, NOx, PM)
- Babcock & Wilcox Steam Tables

Target Coverage: 85%+

Author: GL-BackendDeveloper
Version: 1.0.0
"""

import pytest
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, patch
import hashlib
import json
import math

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# COMBUSTION CALCULATOR IMPLEMENTATION
# =============================================================================

@dataclass
class FuelAnalysis:
    """Ultimate analysis of fuel (weight percent, dry basis)."""

    carbon: float  # C
    hydrogen: float  # H
    oxygen: float  # O
    nitrogen: float  # N
    sulfur: float  # S
    ash: float  # Ash
    moisture: float  # As-fired moisture

    def validate(self) -> bool:
        """Validate that analysis sums to ~100%."""
        total = self.carbon + self.hydrogen + self.oxygen + self.nitrogen + self.sulfur + self.ash
        return 99.0 <= total <= 101.0


@dataclass
class CombustionInput:
    """Input data for combustion calculations."""

    fuel_type: str
    fuel_flow_rate: float  # lb/hr or scfh
    fuel_flow_unit: str  # "lb_hr" or "scfh"
    flue_gas_o2_percent: float
    flue_gas_co_ppm: float = 0.0
    flue_gas_temp_f: float = 400.0
    ambient_temp_f: float = 77.0
    combustion_air_temp_f: float = 80.0
    relative_humidity_percent: float = 60.0
    barometric_pressure_psia: float = 14.696
    fuel_analysis: Optional[FuelAnalysis] = None
    fuel_hhv_btu_lb: Optional[float] = None


@dataclass
class CombustionResult:
    """Results of combustion calculations."""

    calculation_id: str
    timestamp: datetime

    # Stoichiometric values
    stoichiometric_air_lb_lb_fuel: float
    theoretical_air_lb_hr: float
    actual_air_lb_hr: float
    excess_air_percent: float

    # Combustion products (lb/hr)
    co2_mass_flow_lb_hr: float
    h2o_mass_flow_lb_hr: float
    so2_mass_flow_lb_hr: float
    n2_mass_flow_lb_hr: float
    o2_mass_flow_lb_hr: float
    total_flue_gas_lb_hr: float

    # Heat release
    heat_input_mmbtu_hr: float
    heat_release_rate_btu_hr: float

    # Efficiency-related
    adiabatic_flame_temp_f: float
    flue_gas_temp_f: float

    # Provenance
    input_hash: str
    output_hash: str
    formula_version: str = "ASME_PTC_4.1_2013"


class CombustionCalculator:
    """
    ASME PTC 4.1 compliant combustion calculator.

    Calculates:
    - Stoichiometric and actual air requirements
    - Excess air from O2 measurement
    - Combustion product mass flows
    - Heat release rates
    - Adiabatic flame temperature

    All calculations are DETERMINISTIC with full provenance tracking.

    Example:
        >>> calc = CombustionCalculator()
        >>> input_data = CombustionInput(
        ...     fuel_type="natural_gas",
        ...     fuel_flow_rate=10000,
        ...     fuel_flow_unit="scfh",
        ...     flue_gas_o2_percent=3.0,
        ... )
        >>> result = calc.calculate(input_data)
    """

    VERSION = "1.0.0"

    # Default fuel properties
    DEFAULT_FUELS = {
        "natural_gas": {
            "hhv_btu_lb": 23875.0,
            "lhv_btu_lb": 21500.0,
            "carbon": 75.0,
            "hydrogen": 25.0,
            "oxygen": 0.0,
            "nitrogen": 0.0,
            "sulfur": 0.0,
            "ash": 0.0,
            "moisture": 0.0,
            "stoichiometric_air_ratio": 17.2,
            "density_lb_scf": 0.042,
        },
        "fuel_oil_no2": {
            "hhv_btu_lb": 19500.0,
            "lhv_btu_lb": 18300.0,
            "carbon": 86.5,
            "hydrogen": 12.5,
            "oxygen": 0.1,
            "nitrogen": 0.1,
            "sulfur": 0.3,
            "ash": 0.01,
            "moisture": 0.5,
            "stoichiometric_air_ratio": 14.1,
        },
        "coal_bituminous": {
            "hhv_btu_lb": 12500.0,
            "lhv_btu_lb": 11800.0,
            "carbon": 70.0,
            "hydrogen": 5.0,
            "oxygen": 7.5,
            "nitrogen": 1.5,
            "sulfur": 2.0,
            "ash": 9.0,
            "moisture": 5.0,
            "stoichiometric_air_ratio": 9.5,
        },
    }

    # Molecular weights
    MW = {
        "C": 12.01,
        "H": 1.008,
        "O": 16.00,
        "N": 14.01,
        "S": 32.07,
        "CO2": 44.01,
        "H2O": 18.02,
        "SO2": 64.07,
        "N2": 28.01,
        "O2": 32.00,
        "air": 28.97,
    }

    def __init__(self) -> None:
        """Initialize combustion calculator."""
        pass

    def calculate(self, inp: CombustionInput) -> CombustionResult:
        """
        Calculate combustion parameters.

        Args:
            inp: Combustion input data

        Returns:
            CombustionResult with all calculated values
        """
        start_time = datetime.now(timezone.utc)

        # Get fuel properties
        fuel_props = self.DEFAULT_FUELS.get(inp.fuel_type, self.DEFAULT_FUELS["natural_gas"])

        # Convert fuel flow to lb/hr
        if inp.fuel_flow_unit == "scfh" and "density_lb_scf" in fuel_props:
            fuel_mass_flow_lb_hr = inp.fuel_flow_rate * fuel_props["density_lb_scf"]
        else:
            fuel_mass_flow_lb_hr = inp.fuel_flow_rate

        # Get fuel analysis
        if inp.fuel_analysis:
            analysis = inp.fuel_analysis
        else:
            analysis = FuelAnalysis(
                carbon=fuel_props["carbon"],
                hydrogen=fuel_props["hydrogen"],
                oxygen=fuel_props["oxygen"],
                nitrogen=fuel_props["nitrogen"],
                sulfur=fuel_props["sulfur"],
                ash=fuel_props["ash"],
                moisture=fuel_props["moisture"],
            )

        # Calculate stoichiometric air
        stoich_air = self._calculate_stoichiometric_air(analysis)
        theoretical_air = fuel_mass_flow_lb_hr * stoich_air

        # Calculate excess air from O2
        excess_air_pct = self._o2_to_excess_air(inp.flue_gas_o2_percent)
        actual_air = theoretical_air * (1 + excess_air_pct / 100)

        # Calculate combustion products
        products = self._calculate_combustion_products(
            fuel_mass_flow_lb_hr, analysis, excess_air_pct
        )

        # Calculate heat input
        hhv = inp.fuel_hhv_btu_lb or fuel_props["hhv_btu_lb"]
        heat_release = fuel_mass_flow_lb_hr * hhv
        heat_input_mmbtu = heat_release / 1e6

        # Calculate adiabatic flame temperature
        aft = self._calculate_adiabatic_flame_temp(
            hhv, stoich_air, excess_air_pct
        )

        # Compute provenance
        input_hash = self._compute_hash({
            "fuel_type": inp.fuel_type,
            "fuel_flow_rate": inp.fuel_flow_rate,
            "o2_percent": inp.flue_gas_o2_percent,
        })
        output_hash = self._compute_hash({
            "excess_air": excess_air_pct,
            "heat_input": heat_input_mmbtu,
        })

        return CombustionResult(
            calculation_id=f"COMB-{start_time.strftime('%Y%m%d%H%M%S')}",
            timestamp=start_time,
            stoichiometric_air_lb_lb_fuel=round(stoich_air, 2),
            theoretical_air_lb_hr=round(theoretical_air, 0),
            actual_air_lb_hr=round(actual_air, 0),
            excess_air_percent=round(excess_air_pct, 1),
            co2_mass_flow_lb_hr=round(products["co2"], 1),
            h2o_mass_flow_lb_hr=round(products["h2o"], 1),
            so2_mass_flow_lb_hr=round(products["so2"], 3),
            n2_mass_flow_lb_hr=round(products["n2"], 0),
            o2_mass_flow_lb_hr=round(products["o2"], 1),
            total_flue_gas_lb_hr=round(products["total"], 0),
            heat_input_mmbtu_hr=round(heat_input_mmbtu, 3),
            heat_release_rate_btu_hr=round(heat_release, 0),
            adiabatic_flame_temp_f=round(aft, 0),
            flue_gas_temp_f=inp.flue_gas_temp_f,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def _calculate_stoichiometric_air(self, analysis: FuelAnalysis) -> float:
        """
        Calculate stoichiometric air requirement per lb fuel.

        Based on:
        - C + O2 -> CO2
        - H + 1/4 O2 -> 1/2 H2O
        - S + O2 -> SO2

        Air is 23.2% O2 by weight.
        """
        c = analysis.carbon / 100
        h = analysis.hydrogen / 100
        o = analysis.oxygen / 100
        s = analysis.sulfur / 100

        # O2 required (lb/lb fuel)
        o2_required = (
            c * (self.MW["O2"] / self.MW["C"]) +  # C combustion
            h * (self.MW["O2"] / (4 * self.MW["H"])) +  # H combustion
            s * (self.MW["O2"] / self.MW["S"]) -  # S combustion
            o  # O2 already in fuel
        )

        # Convert to air (air is 23.2% O2 by weight)
        air_required = o2_required / 0.232

        return max(0, air_required)

    def _o2_to_excess_air(self, o2_percent: float) -> float:
        """
        Convert flue gas O2 percentage to excess air percentage.

        Based on: O2% = 21 * EA / (1 + EA)
        Where EA = excess air fraction

        Rearranged: EA = O2% / (21 - O2%)
        """
        if o2_percent >= 21:
            return 500.0  # Max cap
        if o2_percent <= 0:
            return 0.0

        excess_air_fraction = o2_percent / (21 - o2_percent)
        return excess_air_fraction * 100

    def _calculate_combustion_products(
        self,
        fuel_mass_lb_hr: float,
        analysis: FuelAnalysis,
        excess_air_pct: float,
    ) -> Dict[str, float]:
        """Calculate mass flow of combustion products."""
        c = analysis.carbon / 100
        h = analysis.hydrogen / 100
        o = analysis.oxygen / 100
        n = analysis.nitrogen / 100
        s = analysis.sulfur / 100
        moisture = analysis.moisture / 100

        # CO2 from carbon combustion
        co2 = fuel_mass_lb_hr * c * (self.MW["CO2"] / self.MW["C"])

        # H2O from hydrogen combustion + fuel moisture
        h2o_comb = fuel_mass_lb_hr * h * (self.MW["H2O"] / (2 * self.MW["H"]))
        h2o_fuel = fuel_mass_lb_hr * moisture
        h2o = h2o_comb + h2o_fuel

        # SO2 from sulfur combustion
        so2 = fuel_mass_lb_hr * s * (self.MW["SO2"] / self.MW["S"])

        # Stoichiometric air
        stoich_air = self._calculate_stoichiometric_air(analysis)
        actual_air = fuel_mass_lb_hr * stoich_air * (1 + excess_air_pct / 100)

        # N2 from air (79% by volume, ~77% by weight) + fuel nitrogen
        n2_from_air = actual_air * 0.768
        n2_from_fuel = fuel_mass_lb_hr * n
        n2 = n2_from_air + n2_from_fuel

        # Excess O2
        o2 = actual_air * 0.232 - (
            fuel_mass_lb_hr * c * (self.MW["O2"] / self.MW["C"]) +
            fuel_mass_lb_hr * h * (self.MW["O2"] / (4 * self.MW["H"])) +
            fuel_mass_lb_hr * s * (self.MW["O2"] / self.MW["S"]) -
            fuel_mass_lb_hr * o
        )
        o2 = max(0, o2)

        total = co2 + h2o + so2 + n2 + o2

        return {
            "co2": co2,
            "h2o": h2o,
            "so2": so2,
            "n2": n2,
            "o2": o2,
            "total": total,
        }

    def _calculate_adiabatic_flame_temp(
        self,
        hhv_btu_lb: float,
        stoich_air_ratio: float,
        excess_air_pct: float,
    ) -> float:
        """
        Estimate adiabatic flame temperature.

        Simplified correlation based on fuel HHV and excess air.
        """
        # Base flame temp for stoichiometric combustion
        # Natural gas: ~3500F, oil: ~3400F, coal: ~3200F
        base_temp = 3200 + hhv_btu_lb * 0.01

        # Reduce for excess air (each 10% EA reduces temp by ~100F)
        temp_reduction = excess_air_pct * 10

        aft = base_temp - temp_reduction

        return max(1500, min(4000, aft))

    def _compute_hash(self, data: Dict) -> str:
        """Compute SHA-256 hash for provenance."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def combustion_calculator() -> CombustionCalculator:
    """Create combustion calculator instance."""
    return CombustionCalculator()


@pytest.fixture
def natural_gas_input() -> CombustionInput:
    """Sample natural gas combustion input."""
    return CombustionInput(
        fuel_type="natural_gas",
        fuel_flow_rate=25000.0,  # scfh
        fuel_flow_unit="scfh",
        flue_gas_o2_percent=3.0,
        flue_gas_temp_f=400.0,
        ambient_temp_f=77.0,
    )


@pytest.fixture
def fuel_oil_input() -> CombustionInput:
    """Sample fuel oil combustion input."""
    return CombustionInput(
        fuel_type="fuel_oil_no2",
        fuel_flow_rate=1000.0,  # lb/hr
        fuel_flow_unit="lb_hr",
        flue_gas_o2_percent=4.0,
        flue_gas_temp_f=450.0,
    )


@pytest.fixture
def coal_input() -> CombustionInput:
    """Sample coal combustion input."""
    return CombustionInput(
        fuel_type="coal_bituminous",
        fuel_flow_rate=5000.0,  # lb/hr
        fuel_flow_unit="lb_hr",
        flue_gas_o2_percent=5.0,
        flue_gas_temp_f=350.0,
    )


@pytest.fixture
def custom_fuel_analysis() -> FuelAnalysis:
    """Custom fuel analysis for testing."""
    return FuelAnalysis(
        carbon=72.0,
        hydrogen=5.5,
        oxygen=8.0,
        nitrogen=1.5,
        sulfur=3.0,
        ash=10.0,
        moisture=4.0,
    )


# =============================================================================
# STOICHIOMETRIC AIR TESTS
# =============================================================================

class TestStoichiometricAir:
    """Test stoichiometric air calculations."""

    def test_natural_gas_stoich_air(self, combustion_calculator):
        """
        Validate stoichiometric air for natural gas.

        Natural gas (CH4 dominant) requires ~17.2 lb air/lb fuel.
        """
        analysis = FuelAnalysis(
            carbon=75.0, hydrogen=25.0, oxygen=0.0,
            nitrogen=0.0, sulfur=0.0, ash=0.0, moisture=0.0,
        )

        stoich_air = combustion_calculator._calculate_stoichiometric_air(analysis)

        # Natural gas: approximately 17 lb air / lb fuel
        assert 15.0 <= stoich_air <= 19.0, (
            f"Natural gas stoichiometric air ratio {stoich_air} "
            f"outside expected range 15-19"
        )

    def test_fuel_oil_stoich_air(self, combustion_calculator):
        """
        Validate stoichiometric air for fuel oil.

        Fuel oil #2 requires ~14.1 lb air/lb fuel.
        """
        analysis = FuelAnalysis(
            carbon=86.5, hydrogen=12.5, oxygen=0.1,
            nitrogen=0.1, sulfur=0.3, ash=0.01, moisture=0.5,
        )

        stoich_air = combustion_calculator._calculate_stoichiometric_air(analysis)

        # Fuel oil: approximately 14 lb air / lb fuel
        assert 12.0 <= stoich_air <= 16.0, (
            f"Fuel oil stoichiometric air ratio {stoich_air} "
            f"outside expected range 12-16"
        )

    def test_coal_stoich_air(self, combustion_calculator):
        """
        Validate stoichiometric air for coal.

        Bituminous coal requires ~9.5 lb air/lb fuel.
        """
        analysis = FuelAnalysis(
            carbon=70.0, hydrogen=5.0, oxygen=7.5,
            nitrogen=1.5, sulfur=2.0, ash=9.0, moisture=5.0,
        )

        stoich_air = combustion_calculator._calculate_stoichiometric_air(analysis)

        # Coal: approximately 9-10 lb air / lb fuel
        assert 8.0 <= stoich_air <= 12.0, (
            f"Coal stoichiometric air ratio {stoich_air} "
            f"outside expected range 8-12"
        )

    def test_stoich_air_accounts_for_fuel_oxygen(self, combustion_calculator):
        """
        Validate that fuel oxygen reduces air requirement.

        Oxygen in fuel displaces combustion air requirement.
        """
        # Fuel without oxygen
        analysis_no_o2 = FuelAnalysis(
            carbon=80.0, hydrogen=10.0, oxygen=0.0,
            nitrogen=5.0, sulfur=0.0, ash=5.0, moisture=0.0,
        )

        # Same fuel with oxygen
        analysis_with_o2 = FuelAnalysis(
            carbon=75.0, hydrogen=10.0, oxygen=5.0,
            nitrogen=5.0, sulfur=0.0, ash=5.0, moisture=0.0,
        )

        stoich_no_o2 = combustion_calculator._calculate_stoichiometric_air(analysis_no_o2)
        stoich_with_o2 = combustion_calculator._calculate_stoichiometric_air(analysis_with_o2)

        assert stoich_with_o2 < stoich_no_o2, (
            "Fuel oxygen should reduce air requirement"
        )


# =============================================================================
# EXCESS AIR TESTS
# =============================================================================

class TestExcessAir:
    """Test excess air calculations from O2 measurement."""

    @pytest.mark.parametrize("o2_percent,expected_excess_air", [
        (0.0, 0.0),
        (3.0, 16.7),  # 3/(21-3) * 100 = 16.67%
        (5.0, 31.25),  # 5/(21-5) * 100 = 31.25%
        (7.0, 50.0),   # 7/(21-7) * 100 = 50%
        (10.0, 90.9),  # 10/(21-10) * 100 = 90.9%
    ])
    def test_o2_to_excess_air_conversion(
        self,
        combustion_calculator,
        o2_percent: float,
        expected_excess_air: float,
    ):
        """
        Validate O2 to excess air conversion.

        Formula: EA% = O2% / (21 - O2%) * 100
        """
        result = combustion_calculator._o2_to_excess_air(o2_percent)

        assert abs(result - expected_excess_air) < 0.5, (
            f"O2 {o2_percent}% -> excess air {result}%, "
            f"expected ~{expected_excess_air}%"
        )

    def test_high_o2_caps_excess_air(self, combustion_calculator):
        """Test that high O2 values are capped."""
        result = combustion_calculator._o2_to_excess_air(20.5)

        assert result <= 500, "Excess air should be capped at high O2 levels"

    def test_negative_o2_returns_zero(self, combustion_calculator):
        """Test that negative O2 returns zero excess air."""
        result = combustion_calculator._o2_to_excess_air(-1.0)

        assert result == 0.0, "Negative O2 should return zero excess air"


# =============================================================================
# COMBUSTION PRODUCTS TESTS
# =============================================================================

class TestCombustionProducts:
    """Test combustion product calculations."""

    def test_co2_production(self, combustion_calculator):
        """
        Validate CO2 production from carbon combustion.

        C + O2 -> CO2
        12 lb C + 32 lb O2 -> 44 lb CO2
        """
        analysis = FuelAnalysis(
            carbon=75.0, hydrogen=25.0, oxygen=0.0,
            nitrogen=0.0, sulfur=0.0, ash=0.0, moisture=0.0,
        )

        products = combustion_calculator._calculate_combustion_products(
            fuel_mass_lb_hr=1000.0,
            analysis=analysis,
            excess_air_pct=20.0,
        )

        # 750 lb C * (44/12) = 2750 lb CO2
        expected_co2 = 750 * (44.01 / 12.01)

        assert abs(products["co2"] - expected_co2) < 10, (
            f"CO2 production {products['co2']} lb/hr, expected ~{expected_co2}"
        )

    def test_h2o_production(self, combustion_calculator):
        """
        Validate H2O production from hydrogen combustion.

        2H + 1/2 O2 -> H2O
        2 lb H + 16 lb O2 -> 18 lb H2O
        """
        analysis = FuelAnalysis(
            carbon=75.0, hydrogen=25.0, oxygen=0.0,
            nitrogen=0.0, sulfur=0.0, ash=0.0, moisture=0.0,
        )

        products = combustion_calculator._calculate_combustion_products(
            fuel_mass_lb_hr=1000.0,
            analysis=analysis,
            excess_air_pct=20.0,
        )

        # 250 lb H * (18/2) = 2250 lb H2O
        expected_h2o = 250 * (18.02 / (2 * 1.008))

        assert abs(products["h2o"] - expected_h2o) < 50, (
            f"H2O production {products['h2o']} lb/hr, expected ~{expected_h2o}"
        )

    def test_so2_production(self, combustion_calculator):
        """
        Validate SO2 production from sulfur combustion.

        S + O2 -> SO2
        32 lb S + 32 lb O2 -> 64 lb SO2
        """
        analysis = FuelAnalysis(
            carbon=70.0, hydrogen=5.0, oxygen=7.5,
            nitrogen=1.5, sulfur=2.0, ash=9.0, moisture=5.0,
        )

        products = combustion_calculator._calculate_combustion_products(
            fuel_mass_lb_hr=1000.0,
            analysis=analysis,
            excess_air_pct=30.0,
        )

        # 20 lb S * (64/32) = 40 lb SO2
        expected_so2 = 20 * (64.07 / 32.07)

        assert abs(products["so2"] - expected_so2) < 1, (
            f"SO2 production {products['so2']} lb/hr, expected ~{expected_so2}"
        )

    def test_mass_balance(self, combustion_calculator, natural_gas_input):
        """
        Validate mass balance: fuel + air = products.

        Total flue gas mass should equal fuel + air mass.
        """
        result = combustion_calculator.calculate(natural_gas_input)

        # Fuel mass
        fuel_mass = natural_gas_input.fuel_flow_rate * 0.042  # lb/hr

        # Total products should approximately equal fuel + air
        total_input = fuel_mass + result.actual_air_lb_hr
        total_output = result.total_flue_gas_lb_hr

        # Allow 5% tolerance for ash and other minor components
        tolerance = total_input * 0.05

        assert abs(total_output - total_input) < tolerance, (
            f"Mass balance error: input {total_input} lb/hr, "
            f"output {total_output} lb/hr"
        )


# =============================================================================
# HEAT RELEASE TESTS
# =============================================================================

class TestHeatRelease:
    """Test heat release calculations."""

    def test_natural_gas_heat_input(self, combustion_calculator, natural_gas_input):
        """
        Validate heat input calculation for natural gas.

        Heat Input = Fuel Flow * HHV
        """
        result = combustion_calculator.calculate(natural_gas_input)

        # Expected: 25000 scfh * 0.042 lb/scf * 23875 Btu/lb / 1e6
        expected_mmbtu = 25000 * 0.042 * 23875 / 1e6

        assert abs(result.heat_input_mmbtu_hr - expected_mmbtu) < 0.5, (
            f"Heat input {result.heat_input_mmbtu_hr} MMBTU/hr, "
            f"expected ~{expected_mmbtu}"
        )

    def test_fuel_oil_heat_input(self, combustion_calculator, fuel_oil_input):
        """
        Validate heat input calculation for fuel oil.
        """
        result = combustion_calculator.calculate(fuel_oil_input)

        # Expected: 1000 lb/hr * 19500 Btu/lb / 1e6
        expected_mmbtu = 1000 * 19500 / 1e6

        assert abs(result.heat_input_mmbtu_hr - expected_mmbtu) < 0.5, (
            f"Heat input {result.heat_input_mmbtu_hr} MMBTU/hr, "
            f"expected ~{expected_mmbtu}"
        )

    def test_adiabatic_flame_temp_decreases_with_excess_air(
        self,
        combustion_calculator,
    ):
        """
        Validate that adiabatic flame temperature decreases with excess air.

        Higher excess air dilutes products and reduces flame temperature.
        """
        hhv = 23875  # Natural gas
        stoich_air = 17.2

        aft_10pct = combustion_calculator._calculate_adiabatic_flame_temp(
            hhv, stoich_air, 10.0
        )
        aft_30pct = combustion_calculator._calculate_adiabatic_flame_temp(
            hhv, stoich_air, 30.0
        )
        aft_50pct = combustion_calculator._calculate_adiabatic_flame_temp(
            hhv, stoich_air, 50.0
        )

        assert aft_10pct > aft_30pct > aft_50pct, (
            "Adiabatic flame temp should decrease with excess air"
        )


# =============================================================================
# PROVENANCE AND AUDIT TESTS
# =============================================================================

class TestProvenance:
    """Test provenance tracking and audit trail."""

    def test_calculation_has_provenance_hash(
        self,
        combustion_calculator,
        natural_gas_input,
    ):
        """Validate that calculations include provenance hashes."""
        result = combustion_calculator.calculate(natural_gas_input)

        assert len(result.input_hash) == 16, "Input hash should be 16 characters"
        assert len(result.output_hash) == 16, "Output hash should be 16 characters"
        assert result.input_hash != result.output_hash, "Hashes should differ"

    def test_deterministic_calculation(
        self,
        combustion_calculator,
        natural_gas_input,
    ):
        """Validate that same inputs produce same outputs."""
        result1 = combustion_calculator.calculate(natural_gas_input)
        result2 = combustion_calculator.calculate(natural_gas_input)

        assert result1.excess_air_percent == result2.excess_air_percent
        assert result1.heat_input_mmbtu_hr == result2.heat_input_mmbtu_hr
        assert result1.input_hash == result2.input_hash

    def test_calculation_id_format(
        self,
        combustion_calculator,
        natural_gas_input,
    ):
        """Validate calculation ID format."""
        result = combustion_calculator.calculate(natural_gas_input)

        assert result.calculation_id.startswith("COMB-")
        assert len(result.calculation_id) == 19  # COMB-YYYYMMDDHHMMSS


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_fuel_flow(self, combustion_calculator):
        """Test handling of zero fuel flow."""
        input_data = CombustionInput(
            fuel_type="natural_gas",
            fuel_flow_rate=0.0,
            fuel_flow_unit="scfh",
            flue_gas_o2_percent=3.0,
        )

        result = combustion_calculator.calculate(input_data)

        assert result.heat_input_mmbtu_hr == 0.0
        assert result.total_flue_gas_lb_hr == 0.0

    def test_very_high_o2(self, combustion_calculator):
        """Test handling of very high O2 (near 21%)."""
        input_data = CombustionInput(
            fuel_type="natural_gas",
            fuel_flow_rate=10000.0,
            fuel_flow_unit="scfh",
            flue_gas_o2_percent=20.0,  # Very high - almost no combustion
        )

        result = combustion_calculator.calculate(input_data)

        # Should have very high excess air
        assert result.excess_air_percent > 400

    def test_custom_fuel_analysis(
        self,
        combustion_calculator,
        custom_fuel_analysis,
    ):
        """Test with custom fuel analysis."""
        input_data = CombustionInput(
            fuel_type="custom",
            fuel_flow_rate=2000.0,
            fuel_flow_unit="lb_hr",
            flue_gas_o2_percent=4.0,
            fuel_analysis=custom_fuel_analysis,
            fuel_hhv_btu_lb=11500.0,
        )

        result = combustion_calculator.calculate(input_data)

        # Custom coal-like fuel
        assert result.heat_input_mmbtu_hr > 0
        assert result.so2_mass_flow_lb_hr > 0  # 3% sulfur

    def test_fuel_analysis_validation(self, custom_fuel_analysis):
        """Test fuel analysis validation."""
        assert custom_fuel_analysis.validate() is True

        # Invalid analysis (doesn't sum to 100%)
        invalid_analysis = FuelAnalysis(
            carbon=50.0, hydrogen=10.0, oxygen=5.0,
            nitrogen=1.0, sulfur=1.0, ash=5.0, moisture=2.0,
        )  # Sums to 74%

        assert invalid_analysis.validate() is False
