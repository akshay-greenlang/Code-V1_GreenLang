"""
GL-017 CONDENSYNC Agent - Test Suite

Comprehensive test coverage for condenser optimization agent including:
- Unit tests for all calculator modules
- Integration tests for the optimizer agent
- Performance benchmarks
- Compliance tests for HEI Standards

Target: 85%+ code coverage

Standards Reference:
    - HEI Standards for Steam Surface Condensers, 12th Edition
    - CTI Standards for Cooling Towers
"""

from typing import Dict, Any

__all__ = [
    "create_test_config",
    "create_test_input",
    "DESIGN_VALUES",
    "OPERATING_RANGES",
]

# Design values for testing (based on typical 500MW unit)
DESIGN_VALUES: Dict[str, Any] = {
    "condenser_id": "TEST-C-001",
    "design_steam_flow_lb_hr": 500000.0,
    "design_backpressure_inhga": 1.5,
    "design_duty_btu_hr": 500_000_000.0,
    "design_cw_flow_gpm": 100000.0,
    "design_inlet_temp_f": 70.0,
    "design_outlet_temp_f": 95.0,
    "design_surface_area_ft2": 150000.0,
    "design_cleanliness_factor": 0.85,
    "design_ttd_f": 5.0,
}

# Operating ranges for parameterized tests
OPERATING_RANGES: Dict[str, tuple] = {
    "load_pct": (30.0, 110.0),
    "cw_inlet_temp_f": (50.0, 100.0),
    "vacuum_inhga": (0.8, 5.0),
    "cleanliness_factor": (0.5, 1.0),
    "cycles_of_concentration": (1.5, 10.0),
}


def create_test_config(
    condenser_id: str = "TEST-C-001",
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Create a test configuration dictionary.

    Args:
        condenser_id: Condenser identifier
        **overrides: Override specific config values

    Returns:
        Configuration dictionary
    """
    config = {
        "condenser_id": condenser_id,
        **DESIGN_VALUES,
        **overrides,
    }
    return config


def create_test_input(
    load_pct: float = 85.0,
    cw_inlet_temp_f: float = 75.0,
    vacuum_inhga: float = 1.5,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Create test input data.

    Args:
        load_pct: Load percentage
        cw_inlet_temp_f: Cooling water inlet temperature
        vacuum_inhga: Condenser vacuum
        **overrides: Override specific input values

    Returns:
        Input data dictionary
    """
    steam_flow = DESIGN_VALUES["design_steam_flow_lb_hr"] * (load_pct / 100.0)

    input_data = {
        "condenser_id": "TEST-C-001",
        "load_pct": load_pct,
        "exhaust_steam_flow_lb_hr": steam_flow,
        "exhaust_steam_pressure_psia": 1.2,
        "condenser_vacuum_inhga": vacuum_inhga,
        "saturation_temperature_f": 101.0,
        "hotwell_temperature_f": 100.5,
        "cw_inlet_temperature_f": cw_inlet_temp_f,
        "cw_outlet_temperature_f": cw_inlet_temp_f + 20.0,
        "cw_inlet_flow_gpm": DESIGN_VALUES["design_cw_flow_gpm"] * 0.9,
        **overrides,
    }
    return input_data
