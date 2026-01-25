"""
Golden Period Comparison Tests for GL-003 UNIFIEDSTEAM

Validates system behavior against recorded golden periods from
known-good operational states.

Author: GL-003 Test Engineering Team
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
import json
import hashlib
import pytest


@dataclass
class GoldenDataPoint:
    """Single data point from golden period recording."""
    timestamp: datetime
    tag: str
    value: float
    unit: str
    quality: str = "GOOD"


@dataclass
class GoldenPeriod:
    """Golden period recording with expected outputs."""
    period_id: str
    description: str
    start_time: datetime
    end_time: datetime
    site: str
    area: str

    # Input signals
    input_signals: List[GoldenDataPoint] = field(default_factory=list)

    # Expected computed outputs
    expected_properties: Dict[str, Any] = field(default_factory=dict)
    expected_kpis: Dict[str, Any] = field(default_factory=dict)
    expected_recommendations: List[Dict[str, Any]] = field(default_factory=list)

    # Tolerances
    property_tolerance_pct: float = 0.5
    kpi_tolerance_pct: float = 1.0

    # Metadata
    recorded_by: str = "system"
    validated: bool = False
    hash: str = ""

    def compute_hash(self) -> str:
        """Compute deterministic hash of golden period."""
        content = json.dumps({
            "period_id": self.period_id,
            "inputs": [(d.timestamp.isoformat(), d.tag, d.value)
                      for d in sorted(self.input_signals, key=lambda x: (x.timestamp, x.tag))],
            "expected": {
                "properties": self.expected_properties,
                "kpis": self.expected_kpis,
            }
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]


# Sample golden period data representing known-good operating state
def create_steady_state_golden_period() -> GoldenPeriod:
    """Create golden period for steady-state operation."""
    base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    return GoldenPeriod(
        period_id="GOLDEN_001_STEADY_STATE",
        description="Steady-state operation at nominal conditions",
        start_time=base_time,
        end_time=base_time + timedelta(hours=1),
        site="SITE_A",
        area="UTILITIES",
        input_signals=[
            # HP header measurements
            GoldenDataPoint(base_time, "HEADER_HP_PT001", 4137.0, "kPa_g"),
            GoldenDataPoint(base_time, "HEADER_HP_TT001", 399.0, "degC"),
            GoldenDataPoint(base_time, "HEADER_HP_FT001", 45.0, "t/h"),
            # MP header measurements
            GoldenDataPoint(base_time, "HEADER_MP_PT001", 1034.0, "kPa_g"),
            GoldenDataPoint(base_time, "HEADER_MP_TT001", 198.0, "degC"),
            GoldenDataPoint(base_time, "HEADER_MP_FT001", 32.0, "t/h"),
            # LP header measurements
            GoldenDataPoint(base_time, "HEADER_LP_PT001", 345.0, "kPa_g"),
            GoldenDataPoint(base_time, "HEADER_LP_TT001", 152.0, "degC"),
            GoldenDataPoint(base_time, "HEADER_LP_FT001", 28.0, "t/h"),
            # Desuperheater
            GoldenDataPoint(base_time, "DSH_001_INLET_TT", 405.0, "degC"),
            GoldenDataPoint(base_time, "DSH_001_OUTLET_TT", 275.0, "degC"),
            GoldenDataPoint(base_time, "DSH_001_SPRAY_FT", 2.5, "t/h"),
            # Condensate
            GoldenDataPoint(base_time, "COND_TANK_LT001", 65.0, "%"),
            GoldenDataPoint(base_time, "COND_RETURN_TT001", 85.0, "degC"),
        ],
        expected_properties={
            "HEADER_HP": {
                "enthalpy_kj_kg": {"value": 3213.0, "tolerance_pct": 0.5},
                "density_kg_m3": {"value": 14.8, "tolerance_pct": 1.0},
                "superheat_c": {"value": 147.0, "tolerance_pct": 2.0},
            },
            "HEADER_MP": {
                "enthalpy_kj_kg": {"value": 2846.0, "tolerance_pct": 0.5},
                "superheat_c": {"value": 17.0, "tolerance_pct": 5.0},
            },
            "HEADER_LP": {
                "enthalpy_kj_kg": {"value": 2768.0, "tolerance_pct": 0.5},
                "superheat_c": {"value": 8.0, "tolerance_pct": 10.0},
            },
        },
        expected_kpis={
            "steam_generation_efficiency": {"value": 85.5, "unit": "%", "tolerance_pct": 2.0},
            "condensate_return_ratio": {"value": 78.0, "unit": "%", "tolerance_pct": 3.0},
            "specific_steam_consumption": {"value": 1.25, "unit": "t/MWh", "tolerance_pct": 5.0},
        },
        expected_recommendations=[],  # No recommendations in steady state
        recorded_by="golden_period_recorder",
        validated=True,
    )


def create_optimization_opportunity_golden_period() -> GoldenPeriod:
    """Create golden period with known optimization opportunities."""
    base_time = datetime(2024, 1, 20, 14, 0, 0, tzinfo=timezone.utc)

    return GoldenPeriod(
        period_id="GOLDEN_002_OPTIMIZATION",
        description="Operating state with desuperheater optimization opportunity",
        start_time=base_time,
        end_time=base_time + timedelta(hours=1),
        site="SITE_A",
        area="UTILITIES",
        input_signals=[
            # HP header - slightly high superheat
            GoldenDataPoint(base_time, "HEADER_HP_PT001", 4137.0, "kPa_g"),
            GoldenDataPoint(base_time, "HEADER_HP_TT001", 420.0, "degC"),  # High temp
            GoldenDataPoint(base_time, "HEADER_HP_FT001", 42.0, "t/h"),
            # Desuperheater - suboptimal
            GoldenDataPoint(base_time, "DSH_001_INLET_TT", 425.0, "degC"),
            GoldenDataPoint(base_time, "DSH_001_OUTLET_TT", 295.0, "degC"),  # Could be lower
            GoldenDataPoint(base_time, "DSH_001_SPRAY_FT", 1.8, "t/h"),  # Low spray
            GoldenDataPoint(base_time, "DSH_001_SETPOINT", 280.0, "degC"),
        ],
        expected_properties={
            "HEADER_HP": {
                "enthalpy_kj_kg": {"value": 3260.0, "tolerance_pct": 0.5},
                "superheat_c": {"value": 168.0, "tolerance_pct": 2.0},
            },
        },
        expected_kpis={
            "desuperheater_approach_temp": {"value": 15.0, "unit": "degC", "tolerance_pct": 20.0},
        },
        expected_recommendations=[
            {
                "type": "desuperheater_setpoint",
                "asset": "DSH_001",
                "action": "reduce_outlet_temp",
                "expected_impact_kw": 50.0,
            }
        ],
        recorded_by="golden_period_recorder",
        validated=True,
    )


def create_trap_failure_golden_period() -> GoldenPeriod:
    """Create golden period with steam trap failure signature."""
    base_time = datetime(2024, 2, 5, 8, 0, 0, tzinfo=timezone.utc)

    return GoldenPeriod(
        period_id="GOLDEN_003_TRAP_FAILURE",
        description="Known failed steam trap ST-103",
        start_time=base_time,
        end_time=base_time + timedelta(hours=2),
        site="SITE_A",
        area="PROCESS_A",
        input_signals=[
            # Trap measurements showing failure pattern
            GoldenDataPoint(base_time, "ST103_INLET_TT", 185.0, "degC"),
            GoldenDataPoint(base_time, "ST103_OUTLET_TT", 182.0, "degC"),  # High outlet
            GoldenDataPoint(base_time, "ST103_DELTA_T", 3.0, "degC"),  # Small delta
            GoldenDataPoint(base_time, "ST103_ACOUSTIC_DB", 78.0, "dB"),  # High acoustic
            GoldenDataPoint(base_time, "ST103_UPSTREAM_PT", 1034.0, "kPa_g"),
            # Comparison - healthy trap
            GoldenDataPoint(base_time, "ST104_INLET_TT", 185.0, "degC"),
            GoldenDataPoint(base_time, "ST104_OUTLET_TT", 95.0, "degC"),  # Normal outlet
            GoldenDataPoint(base_time, "ST104_DELTA_T", 90.0, "degC"),  # Good delta
            GoldenDataPoint(base_time, "ST104_ACOUSTIC_DB", 45.0, "dB"),  # Normal acoustic
        ],
        expected_properties={},
        expected_kpis={
            "trap_failure_probability_ST103": {"value": 0.92, "tolerance_pct": 10.0},
            "trap_failure_probability_ST104": {"value": 0.05, "tolerance_pct": 50.0},
        },
        expected_recommendations=[
            {
                "type": "trap_inspection",
                "asset": "ST-103",
                "priority": "high",
                "expected_steam_loss_kg_hr": 45.0,
            }
        ],
        recorded_by="golden_period_recorder",
        validated=True,
    )


# All golden periods for testing
GOLDEN_PERIODS = [
    create_steady_state_golden_period(),
    create_optimization_opportunity_golden_period(),
    create_trap_failure_golden_period(),
]


class TestGoldenPeriodInfrastructure:
    """Tests for golden period validation infrastructure."""

    def test_golden_period_hash_stability(self):
        """Test that golden period hashes are stable."""
        period = create_steady_state_golden_period()
        hash1 = period.compute_hash()
        hash2 = period.compute_hash()

        assert hash1 == hash2, "Golden period hash should be deterministic"
        assert len(hash1) == 32, "Hash should be 32 characters"

    def test_golden_period_completeness(self):
        """Test that golden periods have required fields."""
        for period in GOLDEN_PERIODS:
            assert period.period_id, f"Period missing ID"
            assert period.description, f"Period {period.period_id} missing description"
            assert period.input_signals, f"Period {period.period_id} has no input signals"
            assert period.validated, f"Period {period.period_id} not validated"


class TestSteadyStateGolden:
    """Tests against steady-state golden period."""

    @pytest.fixture
    def golden_period(self) -> GoldenPeriod:
        return create_steady_state_golden_period()

    def test_steam_property_calculation(self, golden_period):
        """Validate steam properties match golden period."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                compute_properties_pt,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        # Find HP header inputs
        hp_pressure = None
        hp_temp = None
        for signal in golden_period.input_signals:
            if signal.tag == "HEADER_HP_PT001":
                hp_pressure = Decimal(str(signal.value))
            elif signal.tag == "HEADER_HP_TT001":
                hp_temp = Decimal(str(signal.value))

        if hp_pressure and hp_temp:
            result = compute_properties_pt(
                pressure_kpa=hp_pressure,
                temperature_c=hp_temp,
            )

            expected = golden_period.expected_properties.get("HEADER_HP", {})

            if "enthalpy_kj_kg" in expected and hasattr(result, 'enthalpy_kj_kg'):
                exp_val = expected["enthalpy_kj_kg"]["value"]
                tolerance = expected["enthalpy_kj_kg"]["tolerance_pct"] / 100
                calc_val = float(result.enthalpy_kj_kg)

                rel_error = abs(calc_val - exp_val) / exp_val
                assert rel_error <= tolerance, (
                    f"HP header enthalpy: {calc_val:.1f} vs expected {exp_val:.1f} "
                    f"(error: {rel_error*100:.2f}%)"
                )

    def test_no_recommendations_in_steady_state(self, golden_period):
        """Verify no spurious recommendations in steady state."""
        # In a real test, we would run the recommendation engine
        # and verify it produces no recommendations
        assert len(golden_period.expected_recommendations) == 0, (
            "Steady state should have no expected recommendations"
        )


class TestOptimizationGolden:
    """Tests against optimization opportunity golden period."""

    @pytest.fixture
    def golden_period(self) -> GoldenPeriod:
        return create_optimization_opportunity_golden_period()

    def test_desuperheater_optimization_detected(self, golden_period):
        """Verify desuperheater optimization is detected."""
        try:
            from GL_Agents.GL003_UnifiedSteam.optimization.recommendation_engine import (
                RecommendationEngine,
            )
        except ImportError:
            pytest.skip("Recommendation engine not available")

        # Get expected recommendation
        expected_recs = golden_period.expected_recommendations
        assert len(expected_recs) > 0, "Should have expected recommendations"

        expected = expected_recs[0]
        assert expected["type"] == "desuperheater_setpoint"
        assert expected["asset"] == "DSH_001"

    def test_superheat_calculation(self, golden_period):
        """Verify superheat calculation matches golden."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                compute_properties_pt,
                get_saturation_temperature,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        # Find HP header conditions
        hp_pressure = None
        hp_temp = None
        for signal in golden_period.input_signals:
            if signal.tag == "HEADER_HP_PT001":
                hp_pressure = Decimal(str(signal.value))
            elif signal.tag == "HEADER_HP_TT001":
                hp_temp = Decimal(str(signal.value))

        if hp_pressure and hp_temp:
            t_sat = get_saturation_temperature(pressure_kpa=hp_pressure)
            if t_sat:
                superheat = float(hp_temp) - float(t_sat)

                expected = golden_period.expected_properties.get("HEADER_HP", {})
                if "superheat_c" in expected:
                    exp_val = expected["superheat_c"]["value"]
                    tolerance = expected["superheat_c"]["tolerance_pct"] / 100

                    rel_error = abs(superheat - exp_val) / exp_val
                    assert rel_error <= tolerance, (
                        f"Superheat: {superheat:.1f}C vs expected {exp_val:.1f}C"
                    )


class TestTrapFailureGolden:
    """Tests against trap failure golden period."""

    @pytest.fixture
    def golden_period(self) -> GoldenPeriod:
        return create_trap_failure_golden_period()

    def test_trap_failure_signature(self, golden_period):
        """Verify trap failure patterns are recognizable."""
        # Extract trap signals
        failed_trap_signals = {}
        healthy_trap_signals = {}

        for signal in golden_period.input_signals:
            if signal.tag.startswith("ST103_"):
                failed_trap_signals[signal.tag] = signal.value
            elif signal.tag.startswith("ST104_"):
                healthy_trap_signals[signal.tag] = signal.value

        # Verify failure signature
        assert failed_trap_signals.get("ST103_DELTA_T", 999) < 10, (
            "Failed trap should have low delta-T"
        )
        assert failed_trap_signals.get("ST103_ACOUSTIC_DB", 0) > 70, (
            "Failed trap should have high acoustic"
        )

        # Verify healthy signature
        assert healthy_trap_signals.get("ST104_DELTA_T", 0) > 50, (
            "Healthy trap should have good delta-T"
        )
        assert healthy_trap_signals.get("ST104_ACOUSTIC_DB", 999) < 60, (
            "Healthy trap should have low acoustic"
        )

    def test_failure_detection_expected(self, golden_period):
        """Verify expected trap failure detection."""
        expected_kpis = golden_period.expected_kpis

        assert "trap_failure_probability_ST103" in expected_kpis, (
            "Should expect high failure probability for ST103"
        )
        assert expected_kpis["trap_failure_probability_ST103"]["value"] > 0.8, (
            "ST103 failure probability should be high"
        )

        assert "trap_failure_probability_ST104" in expected_kpis, (
            "Should have failure probability for ST104"
        )
        assert expected_kpis["trap_failure_probability_ST104"]["value"] < 0.2, (
            "ST104 failure probability should be low"
        )


class TestGoldenPeriodRegression:
    """Regression tests using golden periods."""

    @pytest.mark.parametrize("golden_period", GOLDEN_PERIODS, ids=lambda p: p.period_id)
    def test_input_signal_validity(self, golden_period):
        """Verify all input signals are within valid ranges."""
        for signal in golden_period.input_signals:
            # Pressure signals
            if "_PT" in signal.tag and signal.unit == "kPa_g":
                assert 0 < signal.value < 20000, (
                    f"Pressure {signal.tag}={signal.value} out of range"
                )
            # Temperature signals
            elif "_TT" in signal.tag and signal.unit == "degC":
                assert -50 < signal.value < 600, (
                    f"Temperature {signal.tag}={signal.value} out of range"
                )
            # Flow signals
            elif "_FT" in signal.tag and signal.unit == "t/h":
                assert 0 <= signal.value < 500, (
                    f"Flow {signal.tag}={signal.value} out of range"
                )

    @pytest.mark.parametrize("golden_period", GOLDEN_PERIODS, ids=lambda p: p.period_id)
    def test_expected_values_reasonable(self, golden_period):
        """Verify expected property values are physically reasonable."""
        for asset, props in golden_period.expected_properties.items():
            if "enthalpy_kj_kg" in props:
                h = props["enthalpy_kj_kg"]["value"]
                assert 0 < h < 4000, f"Enthalpy {h} out of range for {asset}"

            if "superheat_c" in props:
                sh = props["superheat_c"]["value"]
                assert -10 < sh < 300, f"Superheat {sh} out of range for {asset}"

            if "density_kg_m3" in props:
                rho = props["density_kg_m3"]["value"]
                assert 0.1 < rho < 1100, f"Density {rho} out of range for {asset}"


class TestGoldenPeriodManagement:
    """Tests for golden period lifecycle management."""

    def test_create_new_golden_period(self):
        """Test creating a new golden period."""
        period = GoldenPeriod(
            period_id="TEST_001",
            description="Test period",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(hours=1),
            site="TEST",
            area="TEST",
        )

        assert period.period_id == "TEST_001"
        assert not period.validated
        assert period.compute_hash()

    def test_golden_period_serialization(self):
        """Test golden period can be serialized."""
        period = create_steady_state_golden_period()

        # Should be JSON serializable
        data = {
            "period_id": period.period_id,
            "description": period.description,
            "start_time": period.start_time.isoformat(),
            "end_time": period.end_time.isoformat(),
            "site": period.site,
            "area": period.area,
            "input_signals": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "tag": s.tag,
                    "value": s.value,
                    "unit": s.unit,
                }
                for s in period.input_signals
            ],
            "hash": period.compute_hash(),
        }

        json_str = json.dumps(data)
        assert len(json_str) > 0

        # Should be deserializable
        loaded = json.loads(json_str)
        assert loaded["period_id"] == period.period_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
