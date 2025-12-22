"""
GL-006 HEATRECLAIM - Test Fixtures

Pytest fixtures for testing heat recovery optimization.
"""

import pytest
from typing import List

from ..core.config import (
    HeatReclaimConfig,
    StreamType,
    Phase,
    ExchangerType,
    OptimizationObjective,
)
from ..core.schemas import (
    HeatStream,
    HeatExchanger,
    HENDesign,
    UtilityCost,
)


@pytest.fixture
def default_config() -> HeatReclaimConfig:
    """Default heat reclaim configuration."""
    return HeatReclaimConfig()


@pytest.fixture
def simple_hot_streams() -> List[HeatStream]:
    """Simple set of hot streams for testing."""
    return [
        HeatStream(
            stream_id="H1",
            stream_name="Hot Stream 1",
            stream_type=StreamType.HOT,
            fluid_name="Water",
            phase=Phase.LIQUID,
            T_supply_C=150.0,
            T_target_C=60.0,
            m_dot_kg_s=2.0,
            Cp_kJ_kgK=4.18,
        ),
        HeatStream(
            stream_id="H2",
            stream_name="Hot Stream 2",
            stream_type=StreamType.HOT,
            fluid_name="Oil",
            phase=Phase.LIQUID,
            T_supply_C=90.0,
            T_target_C=60.0,
            m_dot_kg_s=3.0,
            Cp_kJ_kgK=2.0,
        ),
    ]


@pytest.fixture
def simple_cold_streams() -> List[HeatStream]:
    """Simple set of cold streams for testing."""
    return [
        HeatStream(
            stream_id="C1",
            stream_name="Cold Stream 1",
            stream_type=StreamType.COLD,
            fluid_name="Water",
            phase=Phase.LIQUID,
            T_supply_C=20.0,
            T_target_C=135.0,
            m_dot_kg_s=1.5,
            Cp_kJ_kgK=4.18,
        ),
        HeatStream(
            stream_id="C2",
            stream_name="Cold Stream 2",
            stream_type=StreamType.COLD,
            fluid_name="Water",
            phase=Phase.LIQUID,
            T_supply_C=80.0,
            T_target_C=140.0,
            m_dot_kg_s=2.0,
            Cp_kJ_kgK=4.18,
        ),
    ]


@pytest.fixture
def textbook_hot_streams() -> List[HeatStream]:
    """Classic textbook example - 4 stream problem."""
    return [
        HeatStream(
            stream_id="H1",
            stream_name="Hot 1",
            stream_type=StreamType.HOT,
            fluid_name="Process",
            phase=Phase.LIQUID,
            T_supply_C=180.0,
            T_target_C=60.0,
            m_dot_kg_s=1.0,
            Cp_kJ_kgK=3.0,
        ),
        HeatStream(
            stream_id="H2",
            stream_name="Hot 2",
            stream_type=StreamType.HOT,
            fluid_name="Process",
            phase=Phase.LIQUID,
            T_supply_C=150.0,
            T_target_C=30.0,
            m_dot_kg_s=1.0,
            Cp_kJ_kgK=1.0,
        ),
    ]


@pytest.fixture
def textbook_cold_streams() -> List[HeatStream]:
    """Classic textbook example - 4 stream problem."""
    return [
        HeatStream(
            stream_id="C1",
            stream_name="Cold 1",
            stream_type=StreamType.COLD,
            fluid_name="Process",
            phase=Phase.LIQUID,
            T_supply_C=20.0,
            T_target_C=135.0,
            m_dot_kg_s=1.0,
            Cp_kJ_kgK=2.0,
        ),
        HeatStream(
            stream_id="C2",
            stream_name="Cold 2",
            stream_type=StreamType.COLD,
            fluid_name="Process",
            phase=Phase.LIQUID,
            T_supply_C=80.0,
            T_target_C=140.0,
            m_dot_kg_s=1.0,
            Cp_kJ_kgK=4.0,
        ),
    ]


@pytest.fixture
def industrial_hot_streams() -> List[HeatStream]:
    """Industrial refinery-scale problem."""
    return [
        HeatStream(
            stream_id="H1",
            stream_name="Crude Distillate",
            stream_type=StreamType.HOT,
            fluid_name="Crude",
            phase=Phase.LIQUID,
            T_supply_C=320.0,
            T_target_C=150.0,
            m_dot_kg_s=50.0,
            Cp_kJ_kgK=2.3,
        ),
        HeatStream(
            stream_id="H2",
            stream_name="Product Cooler",
            stream_type=StreamType.HOT,
            fluid_name="Product",
            phase=Phase.LIQUID,
            T_supply_C=250.0,
            T_target_C=80.0,
            m_dot_kg_s=30.0,
            Cp_kJ_kgK=2.5,
        ),
        HeatStream(
            stream_id="H3",
            stream_name="Reactor Effluent",
            stream_type=StreamType.HOT,
            fluid_name="Effluent",
            phase=Phase.GAS,
            T_supply_C=400.0,
            T_target_C=200.0,
            m_dot_kg_s=20.0,
            Cp_kJ_kgK=1.8,
        ),
    ]


@pytest.fixture
def industrial_cold_streams() -> List[HeatStream]:
    """Industrial refinery-scale problem."""
    return [
        HeatStream(
            stream_id="C1",
            stream_name="Feed Preheater",
            stream_type=StreamType.COLD,
            fluid_name="Feed",
            phase=Phase.LIQUID,
            T_supply_C=50.0,
            T_target_C=280.0,
            m_dot_kg_s=45.0,
            Cp_kJ_kgK=2.4,
        ),
        HeatStream(
            stream_id="C2",
            stream_name="Reboiler",
            stream_type=StreamType.COLD,
            fluid_name="Bottoms",
            phase=Phase.TWO_PHASE,
            T_supply_C=180.0,
            T_target_C=220.0,
            m_dot_kg_s=25.0,
            Cp_kJ_kgK=3.0,
        ),
        HeatStream(
            stream_id="C3",
            stream_name="Steam Generator",
            stream_type=StreamType.COLD,
            fluid_name="Water",
            phase=Phase.LIQUID,
            T_supply_C=105.0,
            T_target_C=180.0,
            m_dot_kg_s=15.0,
            Cp_kJ_kgK=4.2,
        ),
    ]


@pytest.fixture
def utility_costs() -> UtilityCost:
    """Standard utility costs."""
    return UtilityCost(
        hot_utility_cost_usd_kWh=0.05,
        cold_utility_cost_usd_kWh=0.02,
        electricity_cost_usd_kWh=0.10,
        hot_utility_name="HP Steam",
        cold_utility_name="Cooling Water",
        hot_utility_T_C=250.0,
        cold_utility_T_in_C=25.0,
        cold_utility_T_out_C=35.0,
    )


@pytest.fixture
def sample_exchangers() -> List[HeatExchanger]:
    """Sample heat exchangers for testing."""
    return [
        HeatExchanger(
            exchanger_id="HX-001",
            exchanger_name="Process HX 1",
            exchanger_type=ExchangerType.SHELL_AND_TUBE,
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=500.0,
            hot_inlet_T_C=150.0,
            hot_outlet_T_C=90.0,
            cold_inlet_T_C=20.0,
            cold_outlet_T_C=80.0,
        ),
        HeatExchanger(
            exchanger_id="HX-002",
            exchanger_name="Process HX 2",
            exchanger_type=ExchangerType.SHELL_AND_TUBE,
            hot_stream_id="H2",
            cold_stream_id="C2",
            duty_kW=300.0,
            hot_inlet_T_C=90.0,
            hot_outlet_T_C=60.0,
            cold_inlet_T_C=80.0,
            cold_outlet_T_C=110.0,
        ),
    ]


@pytest.fixture
def sample_design(sample_exchangers) -> HENDesign:
    """Sample HEN design for testing."""
    from ..core.config import OptimizationMode

    return HENDesign(
        design_name="Test Design",
        mode=OptimizationMode.GRASSROOTS,
        exchangers=sample_exchangers,
        total_heat_recovered_kW=800.0,
        hot_utility_required_kW=200.0,
        cold_utility_required_kW=150.0,
        exchanger_count=2,
        new_exchanger_count=2,
    )


# Test data for determinism verification
DETERMINISTIC_TEST_CASES = [
    {
        "name": "simple_2x2",
        "hot": [
            {"id": "H1", "T_in": 150, "T_out": 60, "FCp": 8.36},
            {"id": "H2", "T_in": 90, "T_out": 60, "FCp": 6.0},
        ],
        "cold": [
            {"id": "C1", "T_in": 20, "T_out": 135, "FCp": 6.27},
            {"id": "C2", "T_in": 80, "T_out": 140, "FCp": 8.36},
        ],
        "delta_t_min": 10.0,
        "expected_pinch": 90.0,  # Approximate
    },
    {
        "name": "textbook_4stream",
        "hot": [
            {"id": "H1", "T_in": 180, "T_out": 60, "FCp": 3.0},
            {"id": "H2", "T_in": 150, "T_out": 30, "FCp": 1.0},
        ],
        "cold": [
            {"id": "C1", "T_in": 20, "T_out": 135, "FCp": 2.0},
            {"id": "C2", "T_in": 80, "T_out": 140, "FCp": 4.0},
        ],
        "delta_t_min": 10.0,
        "expected_pinch": 90.0,
    },
]
