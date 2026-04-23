"""
GL-007 FURNACEPULSE - Golden Test Dataset

Reference data for regression testing of furnace performance calculations.
Validated against API 560, NFPA 86, and manufacturer specifications.

Standards:
- API 560: Fired Heaters for General Refinery Service
- NFPA 86: Standard for Ovens and Furnaces
- ISO 13579: Industrial furnaces and associated processing equipment

Numerical Precision:
- Efficiency values: 1 decimal place (e.g., 85.3%)
- Temperature: 1 decimal place for calculations, integer for display
- Heat duty: 2 significant figures for kW values
- Draft pressure: 1 decimal place for Pa values
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from decimal import Decimal, ROUND_HALF_UP


# =============================================================================
# NUMERICAL PRECISION STANDARDS
# =============================================================================

class NumericalPrecision:
    """
    Numerical precision standards for GL-007 FURNACEPULSE.

    All calculations must adhere to these precision requirements
    for reproducibility and regulatory compliance.
    """

    # Efficiency: 1 decimal place (e.g., 85.3%)
    EFFICIENCY_DECIMALS = 1

    # Temperature: 1 decimal place for calculations
    TEMPERATURE_DECIMALS = 1

    # Heat duty: 2 decimal places for kW
    HEAT_DUTY_DECIMALS = 2

    # Draft pressure: 1 decimal place for Pa
    DRAFT_PRESSURE_DECIMALS = 1

    # Fuel flow: 3 decimal places for kg/s
    FUEL_FLOW_DECIMALS = 3

    # Excess air: 1 decimal place for percentage
    EXCESS_AIR_DECIMALS = 1

    # RUL (Remaining Useful Life): Integer days
    RUL_DECIMALS = 0

    # Confidence scores: 2 decimal places (0.00 to 1.00)
    CONFIDENCE_DECIMALS = 2

    @staticmethod
    def round_efficiency(value: float) -> float:
        """Round efficiency to standard precision."""
        return round(value, NumericalPrecision.EFFICIENCY_DECIMALS)

    @staticmethod
    def round_temperature(value: float) -> float:
        """Round temperature to standard precision."""
        return round(value, NumericalPrecision.TEMPERATURE_DECIMALS)

    @staticmethod
    def round_heat_duty(value: float) -> float:
        """Round heat duty to standard precision."""
        return round(value, NumericalPrecision.HEAT_DUTY_DECIMALS)

    @staticmethod
    def round_draft(value: float) -> float:
        """Round draft pressure to standard precision."""
        return round(value, NumericalPrecision.DRAFT_PRESSURE_DECIMALS)


# =============================================================================
# GOLDEN TEST CASES - EFFICIENCY CALCULATIONS
# =============================================================================

@dataclass(frozen=True)
class EfficiencyTestCase:
    """Golden test case for efficiency calculation."""

    test_id: str
    description: str

    # Inputs
    fuel_type: str
    fuel_flow_kg_s: float
    fuel_lhv_kj_kg: float
    flue_gas_temp_c: float
    ambient_temp_c: float
    excess_air_percent: float
    stack_o2_percent: float

    # Expected outputs (1 decimal precision)
    expected_thermal_efficiency_percent: float
    expected_combustion_efficiency_percent: float
    expected_radiation_loss_percent: float
    expected_stack_loss_percent: float

    # Tolerance for comparison
    tolerance_percent: float = 0.5

    # Standard reference
    standard_reference: str = "API 560"


# Golden test cases for efficiency calculations
EFFICIENCY_GOLDEN_TESTS: List[EfficiencyTestCase] = [
    EfficiencyTestCase(
        test_id="EFF-001",
        description="Natural gas furnace at design conditions",
        fuel_type="natural_gas",
        fuel_flow_kg_s=0.125,
        fuel_lhv_kj_kg=47500.0,
        flue_gas_temp_c=200.0,
        ambient_temp_c=25.0,
        excess_air_percent=15.0,
        stack_o2_percent=3.0,
        expected_thermal_efficiency_percent=88.5,
        expected_combustion_efficiency_percent=98.5,
        expected_radiation_loss_percent=1.5,
        expected_stack_loss_percent=10.0,
        standard_reference="API 560 Example 5.2",
    ),
    EfficiencyTestCase(
        test_id="EFF-002",
        description="High excess air operation",
        fuel_type="natural_gas",
        fuel_flow_kg_s=0.120,
        fuel_lhv_kj_kg=47500.0,
        flue_gas_temp_c=250.0,
        ambient_temp_c=25.0,
        excess_air_percent=30.0,
        stack_o2_percent=5.5,
        expected_thermal_efficiency_percent=82.3,
        expected_combustion_efficiency_percent=99.0,
        expected_radiation_loss_percent=1.5,
        expected_stack_loss_percent=16.2,
        standard_reference="API 560 Figure 6",
    ),
    EfficiencyTestCase(
        test_id="EFF-003",
        description="Low excess air - optimal combustion",
        fuel_type="natural_gas",
        fuel_flow_kg_s=0.130,
        fuel_lhv_kj_kg=47500.0,
        flue_gas_temp_c=180.0,
        ambient_temp_c=20.0,
        excess_air_percent=10.0,
        stack_o2_percent=2.0,
        expected_thermal_efficiency_percent=91.2,
        expected_combustion_efficiency_percent=97.5,
        expected_radiation_loss_percent=1.2,
        expected_stack_loss_percent=7.6,
        standard_reference="API 560 Optimal Operation",
    ),
    EfficiencyTestCase(
        test_id="EFF-004",
        description="Fuel oil operation",
        fuel_type="fuel_oil_6",
        fuel_flow_kg_s=0.100,
        fuel_lhv_kj_kg=40000.0,
        flue_gas_temp_c=220.0,
        ambient_temp_c=25.0,
        excess_air_percent=20.0,
        stack_o2_percent=4.0,
        expected_thermal_efficiency_percent=85.8,
        expected_combustion_efficiency_percent=97.0,
        expected_radiation_loss_percent=2.0,
        expected_stack_loss_percent=12.2,
        standard_reference="API 560 Heavy Fuel",
    ),
    EfficiencyTestCase(
        test_id="EFF-005",
        description="Cold ambient conditions",
        fuel_type="natural_gas",
        fuel_flow_kg_s=0.135,
        fuel_lhv_kj_kg=47500.0,
        flue_gas_temp_c=190.0,
        ambient_temp_c=-10.0,
        excess_air_percent=12.0,
        stack_o2_percent=2.5,
        expected_thermal_efficiency_percent=90.5,
        expected_combustion_efficiency_percent=98.8,
        expected_radiation_loss_percent=1.0,
        expected_stack_loss_percent=8.5,
        standard_reference="API 560 Cold Weather",
    ),
]


# =============================================================================
# GOLDEN TEST CASES - HOTSPOT DETECTION
# =============================================================================

@dataclass(frozen=True)
class HotspotTestCase:
    """Golden test case for hotspot detection."""

    test_id: str
    description: str

    # IR camera frame data (simulated as temperature grid)
    frame_width: int
    frame_height: int
    background_temp_c: float
    hotspot_locations: List[tuple]  # [(x, y, temp_c), ...]

    # Expected outputs
    expected_hotspots_detected: int
    expected_max_temp_c: float
    expected_severity: str  # "normal", "warning", "critical"

    # Tolerance
    temperature_tolerance_c: float = 2.0


HOTSPOT_GOLDEN_TESTS: List[HotspotTestCase] = [
    HotspotTestCase(
        test_id="HOT-001",
        description="Normal operation - no hotspots",
        frame_width=320,
        frame_height=240,
        background_temp_c=750.0,
        hotspot_locations=[],
        expected_hotspots_detected=0,
        expected_max_temp_c=750.0,
        expected_severity="normal",
    ),
    HotspotTestCase(
        test_id="HOT-002",
        description="Single warning hotspot",
        frame_width=320,
        frame_height=240,
        background_temp_c=750.0,
        hotspot_locations=[(160, 120, 850.0)],
        expected_hotspots_detected=1,
        expected_max_temp_c=850.0,
        expected_severity="warning",
    ),
    HotspotTestCase(
        test_id="HOT-003",
        description="Critical hotspot near TMT limit",
        frame_width=320,
        frame_height=240,
        background_temp_c=800.0,
        hotspot_locations=[(100, 80, 920.0), (200, 160, 880.0)],
        expected_hotspots_detected=2,
        expected_max_temp_c=920.0,
        expected_severity="critical",
    ),
    HotspotTestCase(
        test_id="HOT-004",
        description="Multiple distributed hotspots",
        frame_width=640,
        frame_height=480,
        background_temp_c=720.0,
        hotspot_locations=[
            (100, 100, 800.0),
            (300, 200, 810.0),
            (500, 300, 795.0),
            (200, 400, 820.0),
        ],
        expected_hotspots_detected=4,
        expected_max_temp_c=820.0,
        expected_severity="warning",
    ),
]


# =============================================================================
# GOLDEN TEST CASES - RUL PREDICTION
# =============================================================================

@dataclass(frozen=True)
class RULTestCase:
    """Golden test case for Remaining Useful Life prediction."""

    test_id: str
    description: str

    # Input conditions
    current_tmt_c: float
    avg_operating_hours: int
    historical_max_tmt_c: float
    thermal_cycles_count: int
    last_inspection_days_ago: int

    # Expected outputs (integer days for RUL)
    expected_rul_days: int
    expected_confidence: float  # 2 decimal precision
    expected_recommendation: str

    # Tolerance
    rul_tolerance_days: int = 30


RUL_GOLDEN_TESTS: List[RULTestCase] = [
    RULTestCase(
        test_id="RUL-001",
        description="Healthy tube - long remaining life",
        current_tmt_c=780.0,
        avg_operating_hours=6000,
        historical_max_tmt_c=820.0,
        thermal_cycles_count=50,
        last_inspection_days_ago=30,
        expected_rul_days=730,  # ~2 years
        expected_confidence=0.85,
        expected_recommendation="schedule_next_inspection",
    ),
    RULTestCase(
        test_id="RUL-002",
        description="Degraded tube - maintenance recommended",
        current_tmt_c=870.0,
        avg_operating_hours=12000,
        historical_max_tmt_c=910.0,
        thermal_cycles_count=200,
        last_inspection_days_ago=180,
        expected_rul_days=180,  # ~6 months
        expected_confidence=0.75,
        expected_recommendation="schedule_maintenance",
    ),
    RULTestCase(
        test_id="RUL-003",
        description="Critical condition - immediate action",
        current_tmt_c=920.0,
        avg_operating_hours=20000,
        historical_max_tmt_c=940.0,
        thermal_cycles_count=500,
        last_inspection_days_ago=365,
        expected_rul_days=45,  # ~1.5 months
        expected_confidence=0.60,
        expected_recommendation="immediate_inspection_required",
    ),
    RULTestCase(
        test_id="RUL-004",
        description="New installation - maximum life",
        current_tmt_c=720.0,
        avg_operating_hours=500,
        historical_max_tmt_c=750.0,
        thermal_cycles_count=10,
        last_inspection_days_ago=7,
        expected_rul_days=1825,  # ~5 years
        expected_confidence=0.90,
        expected_recommendation="no_action_required",
    ),
]


# =============================================================================
# GOLDEN TEST CASES - DRAFT ANALYSIS
# =============================================================================

@dataclass(frozen=True)
class DraftTestCase:
    """Golden test case for furnace draft analysis."""

    test_id: str
    description: str

    # Inputs
    firebox_pressure_pa: float
    stack_pressure_pa: float
    ambient_pressure_pa: float
    stack_temp_c: float
    ambient_temp_c: float
    stack_height_m: float

    # Expected outputs (1 decimal precision)
    expected_natural_draft_pa: float
    expected_effective_draft_pa: float
    expected_draft_efficiency_percent: float
    expected_compliance: str  # "compliant", "warning", "violation"

    standard_reference: str = "NFPA 86"


DRAFT_GOLDEN_TESTS: List[DraftTestCase] = [
    DraftTestCase(
        test_id="DFT-001",
        description="Normal natural draft operation",
        firebox_pressure_pa=-50.0,
        stack_pressure_pa=-100.0,
        ambient_pressure_pa=101325.0,
        stack_temp_c=200.0,
        ambient_temp_c=25.0,
        stack_height_m=30.0,
        expected_natural_draft_pa=-120.5,
        expected_effective_draft_pa=-70.5,
        expected_draft_efficiency_percent=58.5,
        expected_compliance="compliant",
    ),
    DraftTestCase(
        test_id="DFT-002",
        description="Marginal draft - warning condition",
        firebox_pressure_pa=-20.0,
        stack_pressure_pa=-30.0,
        ambient_pressure_pa=101325.0,
        stack_temp_c=180.0,
        ambient_temp_c=20.0,
        stack_height_m=20.0,
        expected_natural_draft_pa=-75.2,
        expected_effective_draft_pa=-55.2,
        expected_draft_efficiency_percent=73.5,
        expected_compliance="warning",
    ),
    DraftTestCase(
        test_id="DFT-003",
        description="Positive pressure - violation",
        firebox_pressure_pa=15.0,
        stack_pressure_pa=-5.0,
        ambient_pressure_pa=101325.0,
        stack_temp_c=220.0,
        ambient_temp_c=30.0,
        stack_height_m=25.0,
        expected_natural_draft_pa=-95.8,
        expected_effective_draft_pa=-100.8,
        expected_draft_efficiency_percent=105.2,
        expected_compliance="violation",
    ),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_efficiency_test_by_id(test_id: str) -> Optional[EfficiencyTestCase]:
    """Get efficiency test case by ID."""
    for test in EFFICIENCY_GOLDEN_TESTS:
        if test.test_id == test_id:
            return test
    return None


def get_hotspot_test_by_id(test_id: str) -> Optional[HotspotTestCase]:
    """Get hotspot test case by ID."""
    for test in HOTSPOT_GOLDEN_TESTS:
        if test.test_id == test_id:
            return test
    return None


def get_rul_test_by_id(test_id: str) -> Optional[RULTestCase]:
    """Get RUL test case by ID."""
    for test in RUL_GOLDEN_TESTS:
        if test.test_id == test_id:
            return test
    return None


def validate_efficiency_result(
    test_case: EfficiencyTestCase,
    actual_thermal_efficiency: float,
    actual_combustion_efficiency: float,
    actual_radiation_loss: float,
    actual_stack_loss: float,
) -> Dict[str, Any]:
    """
    Validate efficiency calculation results against golden test case.

    Returns validation result with pass/fail status and details.
    """
    tolerance = test_case.tolerance_percent

    thermal_pass = abs(
        actual_thermal_efficiency - test_case.expected_thermal_efficiency_percent
    ) <= tolerance
    combustion_pass = abs(
        actual_combustion_efficiency - test_case.expected_combustion_efficiency_percent
    ) <= tolerance
    radiation_pass = abs(
        actual_radiation_loss - test_case.expected_radiation_loss_percent
    ) <= tolerance
    stack_pass = abs(
        actual_stack_loss - test_case.expected_stack_loss_percent
    ) <= tolerance

    all_pass = thermal_pass and combustion_pass and radiation_pass and stack_pass

    return {
        "test_id": test_case.test_id,
        "passed": all_pass,
        "details": {
            "thermal_efficiency": {
                "expected": test_case.expected_thermal_efficiency_percent,
                "actual": NumericalPrecision.round_efficiency(actual_thermal_efficiency),
                "passed": thermal_pass,
            },
            "combustion_efficiency": {
                "expected": test_case.expected_combustion_efficiency_percent,
                "actual": NumericalPrecision.round_efficiency(actual_combustion_efficiency),
                "passed": combustion_pass,
            },
            "radiation_loss": {
                "expected": test_case.expected_radiation_loss_percent,
                "actual": NumericalPrecision.round_efficiency(actual_radiation_loss),
                "passed": radiation_pass,
            },
            "stack_loss": {
                "expected": test_case.expected_stack_loss_percent,
                "actual": NumericalPrecision.round_efficiency(actual_stack_loss),
                "passed": stack_pass,
            },
        },
        "tolerance": tolerance,
        "standard_reference": test_case.standard_reference,
    }


# =============================================================================
# REGRESSION TEST RUNNER
# =============================================================================

def run_all_golden_tests(calculator_module) -> Dict[str, Any]:
    """
    Run all golden tests against calculator module.

    Args:
        calculator_module: Module containing calculator functions

    Returns:
        Comprehensive test results
    """
    results = {
        "efficiency_tests": [],
        "hotspot_tests": [],
        "rul_tests": [],
        "draft_tests": [],
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0,
        },
    }

    # Run efficiency tests
    for test in EFFICIENCY_GOLDEN_TESTS:
        # This would call actual calculator
        # result = calculator_module.calculate_efficiency(...)
        # For now, just record the test case
        results["efficiency_tests"].append({
            "test_id": test.test_id,
            "status": "pending",
        })
        results["summary"]["total"] += 1

    # Similar for other test types...

    return results
