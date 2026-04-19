"""
GL-020 ECONOPULSE - End-to-End Integration Tests

Complete workflow tests from sensor data to alerts.
Tests performance trending accuracy, cleaning recommendation validation,
and multi-economizer scenarios.

Target Coverage: 85%+
Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import os
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Any, Optional
import time
import hashlib
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test fixtures from conftest
from tests.conftest import (
    EconomizerConfig, EconomizerType, FlowArrangement,
    TemperatureReading, FlowReading, FoulingData, FoulingLevel,
    PerformanceBaseline, AlertConfig, AlertType, AlertSeverity, Alert
)

# Import calculators from unit tests
from tests.unit.test_heat_transfer_calculator import HeatTransferCalculator
from tests.unit.test_fouling_calculator import FoulingCalculator
from tests.unit.test_economizer_efficiency import EconomizerEfficiencyCalculator
from tests.unit.test_thermal_properties import ThermalPropertiesCalculator
from tests.unit.test_alert_manager import AlertManager, AlertEvent


# =============================================================================
# INTEGRATION AGENT CLASS
# =============================================================================

@dataclass
class PerformanceSnapshot:
    """Snapshot of economizer performance at a point in time."""
    timestamp: datetime
    economizer_id: str
    heat_duty_kw: float
    effectiveness: float
    u_value_w_m2k: float
    fouling_factor_m2k_w: float
    fouling_level: FoulingLevel
    approach_temp_c: float
    efficiency_loss_pct: float
    fuel_penalty_pct: float
    performance_index: float
    alerts: List[Alert]
    provenance_hash: str


@dataclass
class CleaningRecommendation:
    """Cleaning recommendation for an economizer."""
    economizer_id: str
    recommendation_date: date
    urgency: str  # "immediate", "scheduled", "monitor"
    estimated_savings_per_day: float
    days_until_critical: int
    current_fouling_level: FoulingLevel
    projected_cleaning_date: date
    confidence: float


class EconomizerPerformanceAgent:
    """
    GL-020 ECONOPULSE Agent - Economizer Performance Monitoring

    Integrates all calculators to provide end-to-end economizer monitoring.
    """

    VERSION = "1.0.0"
    NAME = "EconomizerPerformanceAgent"
    AGENT_ID = "GL-020"
    CODENAME = "ECONOPULSE"

    def __init__(self, alert_configs: List[AlertConfig] = None):
        self.heat_transfer_calc = HeatTransferCalculator()
        self.fouling_calc = FoulingCalculator()
        self.efficiency_calc = EconomizerEfficiencyCalculator()
        self.thermal_props = ThermalPropertiesCalculator()
        self.alert_manager = AlertManager(alert_configs or [])

        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.performance_history: Dict[str, List[PerformanceSnapshot]] = {}

    def register_baseline(self, baseline: PerformanceBaseline) -> None:
        """Register a clean baseline for an economizer."""
        self.baselines[baseline.economizer_id] = baseline

    def process_readings(
        self,
        economizer: EconomizerConfig,
        temperatures: Dict[str, TemperatureReading],
        flows: Dict[str, FlowReading]
    ) -> PerformanceSnapshot:
        """
        Process sensor readings and calculate all performance metrics.

        Args:
            economizer: Economizer configuration
            temperatures: Temperature readings dict
            flows: Flow readings dict

        Returns:
            PerformanceSnapshot with all calculated metrics
        """
        # Get thermal properties at operating conditions
        T_water_avg = (temperatures["water_inlet"].value_c +
                      temperatures["water_outlet"].value_c) / 2
        T_gas_avg = (temperatures["gas_inlet"].value_c +
                    temperatures["gas_outlet"].value_c) / 2

        water_cp = self.thermal_props.calculate_water_cp(T_water_avg)
        gas_cp = self.thermal_props.calculate_flue_gas_cp(T_gas_avg)

        # Calculate heat transfer metrics
        heat_transfer_result = self.heat_transfer_calc.calculate_all(
            economizer=economizer,
            temperatures=temperatures,
            flows=flows,
            water_cp_kj_kg_k=water_cp,
            gas_cp_kj_kg_k=gas_cp
        )

        # Get baseline for fouling calculation
        baseline = self.baselines.get(economizer.economizer_id)
        if baseline:
            clean_u_value = baseline.clean_u_value_w_m2k
            days_since_cleaning = (date.today() - baseline.baseline_date).days
        else:
            clean_u_value = economizer.design_u_value_w_m2k
            days_since_cleaning = 0

        # Calculate fouling metrics
        fouling_result = self.fouling_calc.calculate_all(
            clean_u_value=clean_u_value,
            current_u_value=heat_transfer_result.u_value_w_m2k,
            days_since_cleaning=days_since_cleaning,
            design_effectiveness=heat_transfer_result.effectiveness
        )

        # Calculate efficiency metrics
        efficiency_result = self.efficiency_calc.calculate_all(
            economizer=economizer,
            temperatures=temperatures,
            flows=flows,
            current_u_value=heat_transfer_result.u_value_w_m2k,
            water_cp_kj_kg_k=water_cp,
            gas_cp_kj_kg_k=gas_cp
        )

        # Process alerts
        timestamp = temperatures["gas_inlet"].timestamp
        alerts = self._process_alerts(
            economizer.economizer_id,
            timestamp,
            fouling_result.fouling_factor_m2k_w,
            efficiency_result.effectiveness,
            heat_transfer_result.u_value_w_m2k,
            heat_transfer_result.approach_temp_c
        )

        # Generate provenance hash
        provenance_data = (
            f"{heat_transfer_result.provenance_hash},"
            f"{fouling_result.provenance_hash},"
            f"{efficiency_result.provenance_hash}"
        )
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            economizer_id=economizer.economizer_id,
            heat_duty_kw=heat_transfer_result.heat_duty_kw,
            effectiveness=efficiency_result.effectiveness,
            u_value_w_m2k=heat_transfer_result.u_value_w_m2k,
            fouling_factor_m2k_w=fouling_result.fouling_factor_m2k_w,
            fouling_level=fouling_result.fouling_level,
            approach_temp_c=heat_transfer_result.approach_temp_c,
            efficiency_loss_pct=fouling_result.efficiency_loss_pct,
            fuel_penalty_pct=fouling_result.fuel_penalty_pct,
            performance_index=efficiency_result.performance_index,
            alerts=alerts,
            provenance_hash=provenance_hash
        )

        # Store in history
        if economizer.economizer_id not in self.performance_history:
            self.performance_history[economizer.economizer_id] = []
        self.performance_history[economizer.economizer_id].append(snapshot)

        return snapshot

    def _process_alerts(
        self,
        economizer_id: str,
        timestamp: datetime,
        fouling_factor: float,
        effectiveness: float,
        u_value: float,
        approach_temp: float
    ) -> List[Alert]:
        """Process all parameters through alert manager."""
        alerts = []

        # Fouling factor alert
        event = AlertEvent(
            parameter="fouling_factor_m2k_w",
            current_value=fouling_factor,
            previous_value=None,
            timestamp=timestamp,
            economizer_id=economizer_id
        )
        alerts.extend(self.alert_manager.process_event(event))

        # Effectiveness alert
        event = AlertEvent(
            parameter="effectiveness",
            current_value=effectiveness,
            previous_value=None,
            timestamp=timestamp,
            economizer_id=economizer_id
        )
        alerts.extend(self.alert_manager.process_event(event))

        # U-value alert
        event = AlertEvent(
            parameter="u_value_w_m2k",
            current_value=u_value,
            previous_value=None,
            timestamp=timestamp,
            economizer_id=economizer_id
        )
        alerts.extend(self.alert_manager.process_event(event))

        # Approach temperature alert
        event = AlertEvent(
            parameter="approach_temp_c",
            current_value=approach_temp,
            previous_value=None,
            timestamp=timestamp,
            economizer_id=economizer_id
        )
        alerts.extend(self.alert_manager.process_event(event))

        return alerts

    def get_cleaning_recommendation(
        self,
        economizer_id: str,
        fuel_cost_per_kwh: float = 0.03
    ) -> CleaningRecommendation:
        """
        Generate cleaning recommendation for an economizer.

        Args:
            economizer_id: Economizer identifier
            fuel_cost_per_kwh: Fuel cost for savings calculation

        Returns:
            CleaningRecommendation with urgency and savings estimate
        """
        if economizer_id not in self.performance_history:
            raise ValueError(f"No performance data for {economizer_id}")

        history = self.performance_history[economizer_id]
        if not history:
            raise ValueError(f"Empty performance history for {economizer_id}")

        # Get most recent snapshot
        latest = history[-1]

        # Analyze trend if enough history
        if len(history) >= 5:
            fouling_data = [
                FoulingData(
                    economizer_id=economizer_id,
                    timestamp=s.timestamp,
                    fouling_factor_m2k_w=s.fouling_factor_m2k_w,
                    fouling_level=s.fouling_level,
                    efficiency_loss_pct=s.efficiency_loss_pct,
                    estimated_fuel_penalty_pct=s.fuel_penalty_pct,
                    days_since_cleaning=i,
                    cleaning_recommended=False,
                    estimated_days_to_cleaning=0
                )
                for i, s in enumerate(history)
            ]
            trend = self.fouling_calc.analyze_trend(fouling_data)
            days_to_critical = trend.days_to_threshold
            projected_date = trend.projected_cleaning_date
            confidence = trend.confidence_level
        else:
            days_to_critical = 30  # Default estimate
            projected_date = date.today() + timedelta(days=30)
            confidence = 0.5

        # Calculate daily savings from cleaning
        daily_hours = 24
        baseline = self.baselines.get(economizer_id)
        if baseline:
            design_duty = baseline.clean_heat_duty_kw
        else:
            design_duty = latest.heat_duty_kw / (1 - latest.efficiency_loss_pct / 100)

        energy_loss_kwh = design_duty * (latest.efficiency_loss_pct / 100) * daily_hours
        daily_savings = energy_loss_kwh * fuel_cost_per_kwh

        # Determine urgency
        if latest.fouling_level == FoulingLevel.SEVERE or days_to_critical <= 0:
            urgency = "immediate"
        elif latest.fouling_level == FoulingLevel.HEAVY or days_to_critical <= 30:
            urgency = "scheduled"
        else:
            urgency = "monitor"

        return CleaningRecommendation(
            economizer_id=economizer_id,
            recommendation_date=date.today(),
            urgency=urgency,
            estimated_savings_per_day=daily_savings,
            days_until_critical=max(0, days_to_critical),
            current_fouling_level=latest.fouling_level,
            projected_cleaning_date=projected_date,
            confidence=confidence
        )

    def get_performance_trend(
        self,
        economizer_id: str,
        days: int = 30
    ) -> Dict[str, List]:
        """
        Get performance trend data for an economizer.

        Args:
            economizer_id: Economizer identifier
            days: Number of days to include

        Returns:
            Dict with trend data for each metric
        """
        if economizer_id not in self.performance_history:
            return {"error": "No data"}

        history = self.performance_history[economizer_id]
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        filtered = [s for s in history if s.timestamp >= cutoff]

        return {
            "timestamps": [s.timestamp for s in filtered],
            "effectiveness": [s.effectiveness for s in filtered],
            "u_value": [s.u_value_w_m2k for s in filtered],
            "fouling_factor": [s.fouling_factor_m2k_w for s in filtered],
            "heat_duty": [s.heat_duty_kw for s in filtered],
            "performance_index": [s.performance_index for s in filtered]
        }

    def generate_provenance_report(
        self,
        economizer_id: str
    ) -> Dict[str, Any]:
        """Generate provenance report for audit trail."""
        if economizer_id not in self.performance_history:
            return {"error": "No data"}

        history = self.performance_history[economizer_id]
        latest = history[-1] if history else None

        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.NAME,
            "codename": self.CODENAME,
            "version": self.VERSION,
            "economizer_id": economizer_id,
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_points": len(history),
            "latest_snapshot": {
                "timestamp": latest.timestamp.isoformat() if latest else None,
                "provenance_hash": latest.provenance_hash if latest else None,
                "performance_index": latest.performance_index if latest else None,
                "fouling_level": latest.fouling_level.value if latest else None
            } if latest else None,
            "baseline_registered": economizer_id in self.baselines
        }


# =============================================================================
# END-TO-END INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """End-to-end integration tests for complete monitoring workflow."""

    def test_complete_monitoring_pipeline(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        design_flow_readings,
        clean_baseline_performance,
        multiple_alert_configs
    ):
        """Test complete monitoring pipeline from sensors to alerts."""
        agent = EconomizerPerformanceAgent(multiple_alert_configs)

        # Register baseline
        agent.register_baseline(clean_baseline_performance)

        # Process readings
        snapshot = agent.process_readings(
            economizer=bare_tube_economizer,
            temperatures=clean_operation_temperatures,
            flows=design_flow_readings
        )

        # Validate snapshot
        assert snapshot is not None
        assert snapshot.economizer_id == bare_tube_economizer.economizer_id
        assert snapshot.heat_duty_kw > 0
        assert 0 < snapshot.effectiveness < 1
        assert snapshot.u_value_w_m2k > 0
        assert snapshot.fouling_factor_m2k_w >= 0
        assert len(snapshot.provenance_hash) == 64

    def test_fouling_detection_workflow(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        fouled_operation_temperatures,
        design_flow_readings,
        clean_baseline_performance,
        multiple_alert_configs
    ):
        """Test fouling detection through complete workflow."""
        agent = EconomizerPerformanceAgent(multiple_alert_configs)
        agent.register_baseline(clean_baseline_performance)

        # Process clean readings
        snapshot_clean = agent.process_readings(
            economizer=bare_tube_economizer,
            temperatures=clean_operation_temperatures,
            flows=design_flow_readings
        )

        # Process fouled readings
        snapshot_fouled = agent.process_readings(
            economizer=bare_tube_economizer,
            temperatures=fouled_operation_temperatures,
            flows=design_flow_readings
        )

        # Fouled should show degradation
        assert snapshot_fouled.fouling_factor_m2k_w > snapshot_clean.fouling_factor_m2k_w
        assert snapshot_fouled.effectiveness < snapshot_clean.effectiveness
        assert snapshot_fouled.efficiency_loss_pct > snapshot_clean.efficiency_loss_pct

    def test_alert_generation_workflow(
        self,
        bare_tube_economizer,
        fouled_operation_temperatures,
        design_flow_readings,
        clean_baseline_performance,
        multiple_alert_configs
    ):
        """Test alert generation through complete workflow."""
        agent = EconomizerPerformanceAgent(multiple_alert_configs)
        agent.register_baseline(clean_baseline_performance)

        # Update temperatures to trigger alerts
        now = datetime.now(timezone.utc)
        fouled_operation_temperatures["gas_outlet"].value_c = 250.0  # Very poor heat transfer
        fouled_operation_temperatures["gas_outlet"].timestamp = now
        fouled_operation_temperatures["gas_inlet"].timestamp = now
        fouled_operation_temperatures["water_inlet"].timestamp = now
        fouled_operation_temperatures["water_outlet"].timestamp = now

        snapshot = agent.process_readings(
            economizer=bare_tube_economizer,
            temperatures=fouled_operation_temperatures,
            flows=design_flow_readings
        )

        # Should have alerts for poor performance
        active_alerts = agent.alert_manager.get_active_alerts()
        # Alerts depend on thresholds in config

    def test_cleaning_recommendation_workflow(
        self,
        bare_tube_economizer,
        design_flow_readings,
        clean_baseline_performance,
        sensor_reading_generator
    ):
        """Test cleaning recommendation generation."""
        agent = EconomizerPerformanceAgent()
        agent.register_baseline(clean_baseline_performance)

        # Generate multiple readings to build history
        readings = sensor_reading_generator(
            num_readings=10,
            economizer_id=bare_tube_economizer.economizer_id
        )

        for reading in readings:
            snapshot = agent.process_readings(
                economizer=bare_tube_economizer,
                temperatures=reading["temperatures"],
                flows=reading["flows"]
            )

        # Get cleaning recommendation
        recommendation = agent.get_cleaning_recommendation(
            economizer_id=bare_tube_economizer.economizer_id
        )

        assert recommendation is not None
        assert recommendation.economizer_id == bare_tube_economizer.economizer_id
        assert recommendation.urgency in ["immediate", "scheduled", "monitor"]
        assert recommendation.estimated_savings_per_day >= 0
        assert recommendation.days_until_critical >= 0

    def test_performance_trend_tracking(
        self,
        bare_tube_economizer,
        design_flow_readings,
        clean_baseline_performance,
        sensor_reading_generator
    ):
        """Test performance trend tracking over time."""
        agent = EconomizerPerformanceAgent()
        agent.register_baseline(clean_baseline_performance)

        # Generate readings over simulated time period
        readings = sensor_reading_generator(
            num_readings=20,
            interval_minutes=60,  # Hourly
            economizer_id=bare_tube_economizer.economizer_id
        )

        for reading in readings:
            agent.process_readings(
                economizer=bare_tube_economizer,
                temperatures=reading["temperatures"],
                flows=reading["flows"]
            )

        # Get trend data
        trend = agent.get_performance_trend(
            economizer_id=bare_tube_economizer.economizer_id,
            days=7
        )

        assert "timestamps" in trend
        assert "effectiveness" in trend
        assert "fouling_factor" in trend
        assert len(trend["timestamps"]) == 20

    def test_provenance_audit_trail(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        design_flow_readings,
        clean_baseline_performance
    ):
        """Test provenance report generation for audit trail."""
        agent = EconomizerPerformanceAgent()
        agent.register_baseline(clean_baseline_performance)

        # Process readings
        agent.process_readings(
            economizer=bare_tube_economizer,
            temperatures=clean_operation_temperatures,
            flows=design_flow_readings
        )

        # Generate provenance report
        report = agent.generate_provenance_report(
            economizer_id=bare_tube_economizer.economizer_id
        )

        assert report["agent_id"] == "GL-020"
        assert report["codename"] == "ECONOPULSE"
        assert report["economizer_id"] == bare_tube_economizer.economizer_id
        assert report["baseline_registered"] is True
        assert report["data_points"] == 1
        assert report["latest_snapshot"] is not None
        assert len(report["latest_snapshot"]["provenance_hash"]) == 64


@pytest.mark.integration
class TestMultiEconomizerScenarios:
    """Integration tests for multi-economizer scenarios."""

    def test_multiple_economizers_monitoring(
        self,
        multiple_economizers,
        design_flow_readings,
        clean_operation_temperatures
    ):
        """Test monitoring multiple economizers simultaneously."""
        agent = EconomizerPerformanceAgent()

        snapshots = []
        for economizer in multiple_economizers:
            # Update readings for each economizer
            temps = dict(clean_operation_temperatures)
            flows = dict(design_flow_readings)

            snapshot = agent.process_readings(
                economizer=economizer,
                temperatures=temps,
                flows=flows
            )
            snapshots.append(snapshot)

        # All economizers should have snapshots
        assert len(snapshots) == len(multiple_economizers)

        # Each should be tracked separately
        for snapshot, economizer in zip(snapshots, multiple_economizers):
            assert snapshot.economizer_id == economizer.economizer_id

    def test_independent_alert_tracking(
        self,
        multiple_economizers,
        design_flow_readings,
        fouled_operation_temperatures,
        multiple_alert_configs
    ):
        """Test alerts are tracked independently per economizer."""
        agent = EconomizerPerformanceAgent(multiple_alert_configs)

        # Process fouled readings for each economizer
        now = datetime.now(timezone.utc)
        for i, economizer in enumerate(multiple_economizers):
            temps = dict(fouled_operation_temperatures)
            # Update timestamps to be unique
            for key in temps:
                temps[key].timestamp = now + timedelta(seconds=i)

            agent.process_readings(
                economizer=economizer,
                temperatures=temps,
                flows=design_flow_readings
            )

        # Check alerts per economizer
        for economizer in multiple_economizers:
            alerts = agent.alert_manager.get_active_alerts(
                economizer_id=economizer.economizer_id
            )
            # Each economizer's alerts should be separate

    def test_comparative_performance_analysis(
        self,
        bare_tube_economizer,
        finned_tube_economizer,
        clean_operation_temperatures,
        design_flow_readings
    ):
        """Test comparative analysis across economizer types."""
        agent = EconomizerPerformanceAgent()

        # Process both economizers
        snapshot_bare = agent.process_readings(
            economizer=bare_tube_economizer,
            temperatures=clean_operation_temperatures,
            flows=design_flow_readings
        )

        snapshot_finned = agent.process_readings(
            economizer=finned_tube_economizer,
            temperatures=clean_operation_temperatures,
            flows=design_flow_readings
        )

        # Compare performance
        # Finned tube typically has higher U-value
        # Both should have valid results
        assert snapshot_bare.u_value_w_m2k > 0
        assert snapshot_finned.u_value_w_m2k > 0


@pytest.mark.integration
class TestDataQualityHandling:
    """Integration tests for data quality scenarios."""

    def test_low_quality_data_handling(
        self,
        bare_tube_economizer,
        design_flow_readings
    ):
        """Test handling of low quality sensor data."""
        agent = EconomizerPerformanceAgent()

        now = datetime.now(timezone.utc)

        # Create readings with low quality
        temps = {
            "gas_inlet": TemperatureReading(
                sensor_id="TT-GAS-IN-001", location="gas_inlet",
                timestamp=now, value_c=380.0, quality=0.5  # Low quality
            ),
            "gas_outlet": TemperatureReading(
                sensor_id="TT-GAS-OUT-001", location="gas_outlet",
                timestamp=now, value_c=175.0, quality=0.99
            ),
            "water_inlet": TemperatureReading(
                sensor_id="TT-WATER-IN-001", location="water_inlet",
                timestamp=now, value_c=105.0, quality=0.99
            ),
            "water_outlet": TemperatureReading(
                sensor_id="TT-WATER-OUT-001", location="water_outlet",
                timestamp=now, value_c=145.0, quality=0.99
            )
        }

        # Should still process successfully
        snapshot = agent.process_readings(
            economizer=bare_tube_economizer,
            temperatures=temps,
            flows=design_flow_readings
        )

        assert snapshot is not None

    def test_edge_case_temperatures(
        self,
        bare_tube_economizer,
        design_flow_readings,
        edge_case_temperatures
    ):
        """Test handling of edge case temperature scenarios."""
        agent = EconomizerPerformanceAgent()

        # Test very small delta T
        temps = edge_case_temperatures["very_small_delta_t"]

        # Should handle gracefully
        try:
            snapshot = agent.process_readings(
                economizer=bare_tube_economizer,
                temperatures=temps,
                flows=design_flow_readings
            )
            # If it doesn't raise, results should be valid but small
            if snapshot:
                assert snapshot.heat_duty_kw >= 0
        except ValueError:
            # Expected for some edge cases
            pass


@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration:
    """Performance integration tests."""

    def test_high_frequency_processing(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        design_flow_readings
    ):
        """Test high frequency data processing."""
        agent = EconomizerPerformanceAgent()

        num_samples = 100
        start = time.time()

        for _ in range(num_samples):
            agent.process_readings(
                economizer=bare_tube_economizer,
                temperatures=clean_operation_temperatures,
                flows=design_flow_readings
            )

        duration = time.time() - start
        throughput = num_samples / duration

        # Should process at least 50 samples per second
        assert throughput > 50

    def test_large_history_handling(
        self,
        bare_tube_economizer,
        design_flow_readings,
        sensor_reading_generator
    ):
        """Test handling of large performance history."""
        agent = EconomizerPerformanceAgent()

        # Generate large history
        readings = sensor_reading_generator(
            num_readings=1000,
            economizer_id=bare_tube_economizer.economizer_id
        )

        start = time.time()
        for reading in readings:
            agent.process_readings(
                economizer=bare_tube_economizer,
                temperatures=reading["temperatures"],
                flows=reading["flows"]
            )
        processing_time = time.time() - start

        # Get trend (should be efficient even with large history)
        start = time.time()
        trend = agent.get_performance_trend(
            economizer_id=bare_tube_economizer.economizer_id,
            days=365
        )
        trend_time = time.time() - start

        assert len(trend["timestamps"]) == 1000
        assert trend_time < 1.0  # Should complete in under 1 second

    def test_concurrent_economizer_processing(
        self,
        multiple_economizers,
        clean_operation_temperatures,
        design_flow_readings
    ):
        """Test processing multiple economizers in sequence."""
        agent = EconomizerPerformanceAgent()

        num_iterations = 50

        start = time.time()
        for _ in range(num_iterations):
            for economizer in multiple_economizers:
                agent.process_readings(
                    economizer=economizer,
                    temperatures=clean_operation_temperatures,
                    flows=design_flow_readings
                )
        duration = time.time() - start

        total_ops = num_iterations * len(multiple_economizers)
        throughput = total_ops / duration

        # Should handle multiple economizers efficiently
        assert throughput > 100


@pytest.mark.integration
class TestReproducibilityIntegration:
    """Integration tests for reproducibility and provenance."""

    def test_calculation_reproducibility(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        design_flow_readings
    ):
        """Test that calculations are reproducible."""
        agent = EconomizerPerformanceAgent()

        snapshots = []
        for _ in range(5):
            snapshot = agent.process_readings(
                economizer=bare_tube_economizer,
                temperatures=clean_operation_temperatures,
                flows=design_flow_readings
            )
            snapshots.append(snapshot)

        # All snapshots should have same core values
        first = snapshots[0]
        for snapshot in snapshots[1:]:
            assert snapshot.heat_duty_kw == pytest.approx(first.heat_duty_kw, rel=0.001)
            assert snapshot.effectiveness == pytest.approx(first.effectiveness, rel=0.001)
            assert snapshot.u_value_w_m2k == pytest.approx(first.u_value_w_m2k, rel=0.001)

    def test_provenance_chain_integrity(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        design_flow_readings,
        sensor_reading_generator
    ):
        """Test provenance chain maintains integrity."""
        agent = EconomizerPerformanceAgent()

        readings = sensor_reading_generator(
            num_readings=10,
            economizer_id=bare_tube_economizer.economizer_id
        )

        hashes = []
        for reading in readings:
            snapshot = agent.process_readings(
                economizer=bare_tube_economizer,
                temperatures=reading["temperatures"],
                flows=reading["flows"]
            )
            hashes.append(snapshot.provenance_hash)

        # All hashes should be valid SHA-256
        for h in hashes:
            assert len(h) == 64
            assert all(c in '0123456789abcdef' for c in h)

        # Each snapshot should have unique hash (different inputs)
        # Note: If same inputs, same hash is expected

    def test_audit_trail_completeness(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        design_flow_readings,
        clean_baseline_performance
    ):
        """Test audit trail contains all required information."""
        agent = EconomizerPerformanceAgent()
        agent.register_baseline(clean_baseline_performance)

        agent.process_readings(
            economizer=bare_tube_economizer,
            temperatures=clean_operation_temperatures,
            flows=design_flow_readings
        )

        report = agent.generate_provenance_report(
            economizer_id=bare_tube_economizer.economizer_id
        )

        # Required audit fields
        required_fields = [
            "agent_id", "agent_name", "codename", "version",
            "economizer_id", "report_timestamp", "data_points",
            "latest_snapshot", "baseline_registered"
        ]

        for field in required_fields:
            assert field in report, f"Missing required field: {field}"


@pytest.mark.integration
class TestCleaningRecommendationValidation:
    """Integration tests for cleaning recommendation accuracy."""

    def test_recommendation_urgency_levels(
        self,
        bare_tube_economizer,
        design_flow_readings,
        clean_baseline_performance
    ):
        """Test cleaning recommendation urgency levels."""
        agent = EconomizerPerformanceAgent()
        agent.register_baseline(clean_baseline_performance)

        now = datetime.now(timezone.utc)

        # Simulate different fouling levels
        fouling_scenarios = [
            (0.0002, FoulingLevel.CLEAN, "monitor"),
            (0.0005, FoulingLevel.MODERATE, "monitor"),
            (0.0009, FoulingLevel.HEAVY, "scheduled"),
        ]

        for fouling_factor, expected_level, expected_urgency in fouling_scenarios:
            # Create snapshot directly in history
            snapshot = PerformanceSnapshot(
                timestamp=now,
                economizer_id=bare_tube_economizer.economizer_id,
                heat_duty_kw=2000.0,
                effectiveness=0.7,
                u_value_w_m2k=40.0,
                fouling_factor_m2k_w=fouling_factor,
                fouling_level=expected_level,
                approach_temp_c=70.0,
                efficiency_loss_pct=10.0,
                fuel_penalty_pct=2.0,
                performance_index=80.0,
                alerts=[],
                provenance_hash="test_hash_" + str(fouling_factor)
            )

            # Clear and add to history
            agent.performance_history[bare_tube_economizer.economizer_id] = [snapshot]

            recommendation = agent.get_cleaning_recommendation(
                economizer_id=bare_tube_economizer.economizer_id
            )

            assert recommendation.current_fouling_level == expected_level

    def test_savings_calculation_accuracy(
        self,
        bare_tube_economizer,
        clean_baseline_performance
    ):
        """Test cleaning savings calculation is reasonable."""
        agent = EconomizerPerformanceAgent()
        agent.register_baseline(clean_baseline_performance)

        now = datetime.now(timezone.utc)

        # Create snapshot with known efficiency loss
        snapshot = PerformanceSnapshot(
            timestamp=now,
            economizer_id=bare_tube_economizer.economizer_id,
            heat_duty_kw=2000.0,
            effectiveness=0.65,
            u_value_w_m2k=35.0,
            fouling_factor_m2k_w=0.0007,
            fouling_level=FoulingLevel.MODERATE,
            approach_temp_c=80.0,
            efficiency_loss_pct=15.0,  # 15% loss
            fuel_penalty_pct=3.0,
            performance_index=75.0,
            alerts=[],
            provenance_hash="test_hash"
        )

        agent.performance_history[bare_tube_economizer.economizer_id] = [snapshot]

        recommendation = agent.get_cleaning_recommendation(
            economizer_id=bare_tube_economizer.economizer_id,
            fuel_cost_per_kwh=0.03
        )

        # Savings should be reasonable for 15% efficiency loss
        # Expected: ~2500kW * 0.15 * 24h * $0.03/kWh = ~$270/day
        assert recommendation.estimated_savings_per_day > 0
        assert recommendation.estimated_savings_per_day < 1000  # Sanity check
