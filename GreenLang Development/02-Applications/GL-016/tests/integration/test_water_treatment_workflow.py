# -*- coding: utf-8 -*-
"""
Integration Tests for Water Treatment Workflow - GL-016 WATERGUARD

Comprehensive integration test suite covering:
- Complete water treatment analysis workflow
- Integration with SCADA connectors
- Alert generation for out-of-spec conditions
- Report generation

Target: 95%+ coverage for integration scenarios
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.water_chemistry_calculator import WaterChemistryCalculator, WaterSample
from calculators.scale_formation_calculator import ScaleFormationCalculator, ScaleConditions
from calculators.corrosion_rate_calculator import CorrosionRateCalculator, CorrosionConditions
from calculators.provenance import ProvenanceTracker


# ============================================================================
# Alert System Simulation
# ============================================================================

class AlertLevel:
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class WaterTreatmentAlert:
    """Alert for out-of-spec water conditions."""

    def __init__(
        self,
        alert_id: str,
        level: str,
        parameter: str,
        current_value: float,
        threshold: float,
        message: str,
        timestamp: datetime = None
    ):
        self.alert_id = alert_id
        self.level = level
        self.parameter = parameter
        self.current_value = current_value
        self.threshold = threshold
        self.message = message
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.acknowledged = False
        self.resolved = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'level': self.level,
            'parameter': self.parameter,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'resolved': self.resolved
        }


class AlertManager:
    """Manages water treatment alerts."""

    def __init__(self):
        self.alerts: List[WaterTreatmentAlert] = []
        self.alert_counter = 0

        # Alert thresholds
        self.thresholds = {
            'ph_low': 8.0,
            'ph_high': 11.0,
            'conductivity_high': 5000.0,
            'dissolved_oxygen_high': 0.01,
            'chloride_high': 200.0,
            'silica_high': 150.0,
            'iron_high': 0.1,
            'lsi_scaling': 1.0,
            'lsi_corrosive': -1.0,
            'corrosion_rate_high': 10.0,  # mpy
        }

    def check_water_chemistry(self, analysis_result: Dict[str, Any]) -> List[WaterTreatmentAlert]:
        """Check water chemistry results and generate alerts."""
        new_alerts = []

        water_chem = analysis_result.get('water_chemistry', {})

        # Check pH
        ph = water_chem.get('ph', 7.0)
        if ph < self.thresholds['ph_low']:
            new_alerts.append(self._create_alert(
                AlertLevel.WARNING,
                'pH',
                ph,
                self.thresholds['ph_low'],
                f"pH {ph} is below minimum {self.thresholds['ph_low']}"
            ))
        elif ph > self.thresholds['ph_high']:
            new_alerts.append(self._create_alert(
                AlertLevel.WARNING,
                'pH',
                ph,
                self.thresholds['ph_high'],
                f"pH {ph} is above maximum {self.thresholds['ph_high']}"
            ))

        # Check scaling indices
        scaling = analysis_result.get('scaling_indices', {})
        lsi = scaling.get('lsi', {}).get('value', 0)

        if lsi > self.thresholds['lsi_scaling']:
            new_alerts.append(self._create_alert(
                AlertLevel.WARNING,
                'LSI',
                lsi,
                self.thresholds['lsi_scaling'],
                f"LSI {lsi:.2f} indicates scaling tendency"
            ))
        elif lsi < self.thresholds['lsi_corrosive']:
            new_alerts.append(self._create_alert(
                AlertLevel.WARNING,
                'LSI',
                lsi,
                self.thresholds['lsi_corrosive'],
                f"LSI {lsi:.2f} indicates corrosive tendency"
            ))

        self.alerts.extend(new_alerts)
        return new_alerts

    def check_corrosion(self, corrosion_result: Dict[str, Any]) -> List[WaterTreatmentAlert]:
        """Check corrosion analysis and generate alerts."""
        new_alerts = []

        total_rate = corrosion_result.get('total_corrosion_rate', {})
        rate_mpy = total_rate.get('total_corrosion_rate_mpy', 0)

        if rate_mpy > self.thresholds['corrosion_rate_high']:
            new_alerts.append(self._create_alert(
                AlertLevel.CRITICAL,
                'Corrosion Rate',
                rate_mpy,
                self.thresholds['corrosion_rate_high'],
                f"Corrosion rate {rate_mpy:.1f} mpy exceeds threshold"
            ))

        self.alerts.extend(new_alerts)
        return new_alerts

    def _create_alert(
        self,
        level: str,
        parameter: str,
        value: float,
        threshold: float,
        message: str
    ) -> WaterTreatmentAlert:
        """Create a new alert."""
        self.alert_counter += 1
        return WaterTreatmentAlert(
            alert_id=f"ALERT-{self.alert_counter:05d}",
            level=level,
            parameter=parameter,
            current_value=value,
            threshold=threshold,
            message=message
        )

    def get_active_alerts(self) -> List[WaterTreatmentAlert]:
        """Get all active (unresolved) alerts."""
        return [a for a in self.alerts if not a.resolved]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
        return False


# ============================================================================
# Report Generator
# ============================================================================

class WaterTreatmentReport:
    """Generate water treatment analysis reports."""

    def __init__(self, boiler_id: str, report_period: str = "daily"):
        self.boiler_id = boiler_id
        self.report_period = report_period
        self.timestamp = datetime.now(timezone.utc)
        self.sections = {}

    def add_water_chemistry_section(self, analysis: Dict[str, Any]) -> None:
        """Add water chemistry analysis section."""
        self.sections['water_chemistry'] = {
            'title': 'Water Chemistry Analysis',
            'data': analysis.get('water_chemistry', {}),
            'scaling_indices': analysis.get('scaling_indices', {}),
            'provenance_hash': analysis.get('provenance', {}).get('provenance_hash', 'N/A')
        }

    def add_scale_analysis_section(self, analysis: Dict[str, Any]) -> None:
        """Add scale formation analysis section."""
        self.sections['scale_formation'] = {
            'title': 'Scale Formation Analysis',
            'total_scale': analysis.get('total_scale_prediction', {}),
            'cleaning_schedule': analysis.get('cleaning_schedule', {}),
            'provenance_hash': analysis.get('provenance', {}).get('provenance_hash', 'N/A')
        }

    def add_corrosion_section(self, analysis: Dict[str, Any]) -> None:
        """Add corrosion analysis section."""
        self.sections['corrosion'] = {
            'title': 'Corrosion Analysis',
            'total_rate': analysis.get('total_corrosion_rate', {}),
            'remaining_life': analysis.get('remaining_life_analysis', {}),
            'provenance_hash': analysis.get('provenance', {}).get('provenance_hash', 'N/A')
        }

    def add_alerts_section(self, alerts: List[WaterTreatmentAlert]) -> None:
        """Add alerts section."""
        self.sections['alerts'] = {
            'title': 'Active Alerts',
            'count': len(alerts),
            'alerts': [a.to_dict() for a in alerts]
        }

    def add_recommendations_section(self, recommendations: List[str]) -> None:
        """Add recommendations section."""
        self.sections['recommendations'] = {
            'title': 'Treatment Recommendations',
            'items': recommendations
        }

    def generate(self) -> Dict[str, Any]:
        """Generate the complete report."""
        return {
            'report_type': 'Water Treatment Analysis',
            'boiler_id': self.boiler_id,
            'report_period': self.report_period,
            'generated_at': self.timestamp.isoformat(),
            'sections': self.sections
        }


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def water_chemistry_calculator():
    """Water chemistry calculator instance."""
    return WaterChemistryCalculator(version="1.0.0")


@pytest.fixture
def scale_calculator():
    """Scale formation calculator instance."""
    return ScaleFormationCalculator(version="1.0.0")


@pytest.fixture
def corrosion_calculator():
    """Corrosion rate calculator instance."""
    return CorrosionRateCalculator(version="1.0.0")


@pytest.fixture
def alert_manager():
    """Alert manager instance."""
    return AlertManager()


@pytest.fixture
def standard_water_sample():
    """Standard water sample for testing."""
    return WaterSample(
        temperature_c=85.0,
        ph=8.5,
        conductivity_us_cm=1200.0,
        calcium_mg_l=50.0,
        magnesium_mg_l=30.0,
        sodium_mg_l=100.0,
        potassium_mg_l=10.0,
        chloride_mg_l=150.0,
        sulfate_mg_l=100.0,
        bicarbonate_mg_l=200.0,
        carbonate_mg_l=10.0,
        hydroxide_mg_l=0.0,
        silica_mg_l=25.0,
        iron_mg_l=0.05,
        copper_mg_l=0.01,
        phosphate_mg_l=15.0,
        dissolved_oxygen_mg_l=0.02,
        total_alkalinity_mg_l_caco3=250.0,
        total_hardness_mg_l_caco3=180.0
    )


@pytest.fixture
def out_of_spec_water_sample():
    """Out-of-spec water sample for alert testing."""
    return WaterSample(
        temperature_c=95.0,
        ph=6.5,  # Low pH - corrosive
        conductivity_us_cm=6000.0,  # High conductivity
        calcium_mg_l=200.0,  # High calcium
        magnesium_mg_l=100.0,
        sodium_mg_l=500.0,
        potassium_mg_l=30.0,
        chloride_mg_l=500.0,  # High chloride
        sulfate_mg_l=300.0,
        bicarbonate_mg_l=50.0,
        carbonate_mg_l=0.0,
        hydroxide_mg_l=0.0,
        silica_mg_l=200.0,  # High silica
        iron_mg_l=0.5,  # High iron
        copper_mg_l=0.1,
        phosphate_mg_l=5.0,
        dissolved_oxygen_mg_l=0.1,  # High DO
        total_alkalinity_mg_l_caco3=50.0,
        total_hardness_mg_l_caco3=800.0  # Very hard
    )


@pytest.fixture
def standard_scale_conditions():
    """Standard scale conditions for testing."""
    return ScaleConditions(
        temperature_c=85.0,
        pressure_bar=10.0,
        flow_velocity_m_s=2.0,
        surface_roughness_um=10.0,
        operating_time_hours=1000.0,
        cycles_of_concentration=5.0,
        calcium_mg_l=50.0,
        magnesium_mg_l=30.0,
        sulfate_mg_l=100.0,
        silica_mg_l=25.0,
        iron_mg_l=0.05,
        copper_mg_l=0.01,
        ph=8.5,
        alkalinity_mg_l_caco3=250.0
    )


@pytest.fixture
def standard_corrosion_conditions():
    """Standard corrosion conditions for testing."""
    return CorrosionConditions(
        temperature_c=85.0,
        pressure_bar=10.0,
        flow_velocity_m_s=2.0,
        ph=8.5,
        dissolved_oxygen_mg_l=0.02,
        carbon_dioxide_mg_l=5.0,
        chloride_mg_l=150.0,
        sulfate_mg_l=100.0,
        ammonia_mg_l=0.5,
        conductivity_us_cm=1200.0,
        material_type='carbon_steel',
        surface_finish='machined',
        operating_time_hours=1000.0,
        stress_level_mpa=100.0
    )


# ============================================================================
# Complete Workflow Tests
# ============================================================================

@pytest.mark.integration
class TestCompleteWorkflow:
    """Test complete water treatment analysis workflow."""

    def test_full_water_analysis_workflow(
        self,
        water_chemistry_calculator,
        scale_calculator,
        corrosion_calculator,
        standard_water_sample,
        standard_scale_conditions,
        standard_corrosion_conditions
    ):
        """Test complete water analysis workflow."""
        # Step 1: Water chemistry analysis
        water_chem_result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            standard_water_sample
        )

        assert water_chem_result is not None
        assert 'water_chemistry' in water_chem_result
        assert 'scaling_indices' in water_chem_result
        assert 'provenance' in water_chem_result

        # Step 2: Scale formation analysis
        scale_result = scale_calculator.calculate_comprehensive_scale_analysis(
            standard_scale_conditions
        )

        assert scale_result is not None
        assert 'total_scale_prediction' in scale_result
        assert 'cleaning_schedule' in scale_result

        # Step 3: Corrosion analysis
        corrosion_result = corrosion_calculator.calculate_comprehensive_corrosion_analysis(
            standard_corrosion_conditions
        )

        assert corrosion_result is not None
        assert 'total_corrosion_rate' in corrosion_result
        assert 'remaining_life_analysis' in corrosion_result

        # Step 4: Generate report
        report = WaterTreatmentReport(boiler_id="BOILER-001")
        report.add_water_chemistry_section(water_chem_result)
        report.add_scale_analysis_section(scale_result)
        report.add_corrosion_section(corrosion_result)

        final_report = report.generate()

        assert final_report['boiler_id'] == "BOILER-001"
        assert 'water_chemistry' in final_report['sections']
        assert 'scale_formation' in final_report['sections']
        assert 'corrosion' in final_report['sections']

    def test_workflow_with_alerts(
        self,
        water_chemistry_calculator,
        corrosion_calculator,
        out_of_spec_water_sample,
        alert_manager
    ):
        """Test workflow with out-of-spec conditions generating alerts."""
        # Analyze out-of-spec water
        water_result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            out_of_spec_water_sample
        )

        # Check for alerts
        water_alerts = alert_manager.check_water_chemistry(water_result)

        # Should generate alerts for low pH
        assert len(water_alerts) > 0

        # Get active alerts
        active = alert_manager.get_active_alerts()
        assert len(active) > 0

        # Acknowledge an alert
        if active:
            assert alert_manager.acknowledge_alert(active[0].alert_id)
            assert active[0].acknowledged

    def test_workflow_provenance_chain(
        self,
        water_chemistry_calculator,
        scale_calculator,
        standard_water_sample,
        standard_scale_conditions
    ):
        """Test provenance chain through workflow."""
        provenance_hashes = []

        # Step 1
        water_result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            standard_water_sample
        )
        provenance_hashes.append(water_result['provenance']['provenance_hash'])

        # Step 2
        scale_result = scale_calculator.calculate_comprehensive_scale_analysis(
            standard_scale_conditions
        )
        provenance_hashes.append(scale_result['provenance']['provenance_hash'])

        # All hashes should be unique
        assert len(set(provenance_hashes)) == len(provenance_hashes)

        # All hashes should be valid SHA-256
        for h in provenance_hashes:
            assert len(h) == 64
            assert all(c in '0123456789abcdef' for c in h)


# ============================================================================
# SCADA Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.scada
class TestSCADAIntegration:
    """Test SCADA system integration."""

    @pytest.mark.asyncio
    async def test_scada_data_acquisition(self, mock_scada_server, mock_scada_data):
        """Test acquiring data from SCADA system."""
        # Configure mock
        mock_scada_server.read_multiple_tags.return_value = mock_scada_data['tags']

        # Connect
        connected = await mock_scada_server.connect()
        assert connected

        # Read tags
        tags = await mock_scada_server.read_multiple_tags()
        assert 'BOILER_PRESSURE' in tags
        assert 'CONDUCTIVITY' in tags
        assert 'PH_SENSOR' in tags

        # Verify data quality
        for tag_name, tag_data in tags.items():
            assert tag_data['quality'] == 'GOOD'

        # Disconnect
        await mock_scada_server.disconnect()

    @pytest.mark.asyncio
    async def test_scada_to_analysis_workflow(
        self,
        water_chemistry_calculator,
        mock_scada_server,
        mock_scada_data
    ):
        """Test workflow from SCADA data to analysis."""
        mock_scada_server.read_multiple_tags.return_value = mock_scada_data['tags']

        # Get SCADA data
        await mock_scada_server.connect()
        tags = await mock_scada_server.read_multiple_tags()
        await mock_scada_server.disconnect()

        # Convert SCADA data to WaterSample
        sample = WaterSample(
            temperature_c=tags['FEEDWATER_TEMP']['value'],
            ph=tags['PH_SENSOR']['value'],
            conductivity_us_cm=tags['CONDUCTIVITY']['value'],
            calcium_mg_l=50.0,  # Would come from analyzer
            magnesium_mg_l=30.0,
            sodium_mg_l=100.0,
            potassium_mg_l=10.0,
            chloride_mg_l=tags.get('CHLORIDE_SENSOR', {'value': 150.0})['value'],
            sulfate_mg_l=100.0,
            bicarbonate_mg_l=200.0,
            carbonate_mg_l=10.0,
            hydroxide_mg_l=0.0,
            silica_mg_l=tags.get('SILICA_SENSOR', {'value': 25.0})['value'],
            iron_mg_l=0.05,
            copper_mg_l=0.01,
            phosphate_mg_l=15.0,
            dissolved_oxygen_mg_l=tags.get('DO_SENSOR', {'value': 0.02})['value'],
            total_alkalinity_mg_l_caco3=250.0,
            total_hardness_mg_l_caco3=180.0
        )

        # Analyze
        result = water_chemistry_calculator.calculate_water_chemistry_analysis(sample)

        assert result is not None
        assert 'provenance' in result

    @pytest.mark.asyncio
    async def test_scada_write_setpoint(self, mock_scada_server):
        """Test writing setpoint to SCADA."""
        await mock_scada_server.connect()

        # Write a setpoint
        success = await mock_scada_server.write_tag('BLOWDOWN_SP', 2.5)
        assert success

        await mock_scada_server.disconnect()


# ============================================================================
# Alert Generation Tests
# ============================================================================

@pytest.mark.integration
class TestAlertGeneration:
    """Test alert generation for out-of-spec conditions."""

    def test_low_ph_alert(self, water_chemistry_calculator, alert_manager):
        """Test alert generation for low pH."""
        sample = WaterSample(
            temperature_c=85.0,
            ph=6.5,  # Low pH
            conductivity_us_cm=1200.0,
            calcium_mg_l=50.0,
            magnesium_mg_l=30.0,
            sodium_mg_l=100.0,
            potassium_mg_l=10.0,
            chloride_mg_l=150.0,
            sulfate_mg_l=100.0,
            bicarbonate_mg_l=50.0,
            carbonate_mg_l=0.0,
            hydroxide_mg_l=0.0,
            silica_mg_l=25.0,
            iron_mg_l=0.05,
            copper_mg_l=0.01,
            phosphate_mg_l=15.0,
            dissolved_oxygen_mg_l=0.02,
            total_alkalinity_mg_l_caco3=50.0,
            total_hardness_mg_l_caco3=180.0
        )

        result = water_chemistry_calculator.calculate_water_chemistry_analysis(sample)
        alerts = alert_manager.check_water_chemistry(result)

        # Should have pH alert
        ph_alerts = [a for a in alerts if a.parameter == 'pH']
        assert len(ph_alerts) > 0

    def test_high_ph_alert(self, water_chemistry_calculator, alert_manager):
        """Test alert generation for high pH."""
        sample = WaterSample(
            temperature_c=85.0,
            ph=12.0,  # High pH
            conductivity_us_cm=5000.0,
            calcium_mg_l=5.0,
            magnesium_mg_l=2.0,
            sodium_mg_l=1000.0,
            potassium_mg_l=50.0,
            chloride_mg_l=100.0,
            sulfate_mg_l=50.0,
            bicarbonate_mg_l=0.0,
            carbonate_mg_l=200.0,
            hydroxide_mg_l=500.0,
            silica_mg_l=25.0,
            iron_mg_l=0.01,
            copper_mg_l=0.005,
            phosphate_mg_l=50.0,
            dissolved_oxygen_mg_l=0.01,
            total_alkalinity_mg_l_caco3=1000.0,
            total_hardness_mg_l_caco3=20.0
        )

        result = water_chemistry_calculator.calculate_water_chemistry_analysis(sample)
        alerts = alert_manager.check_water_chemistry(result)

        ph_alerts = [a for a in alerts if a.parameter == 'pH']
        assert len(ph_alerts) > 0

    def test_alert_lifecycle(self, alert_manager):
        """Test alert lifecycle: create, acknowledge, resolve."""
        # Create alert
        alert = alert_manager._create_alert(
            AlertLevel.WARNING,
            'pH',
            7.5,
            8.0,
            "pH below target"
        )
        alert_manager.alerts.append(alert)

        # Check it's active
        active = alert_manager.get_active_alerts()
        assert len(active) == 1
        assert not active[0].acknowledged
        assert not active[0].resolved

        # Acknowledge
        assert alert_manager.acknowledge_alert(alert.alert_id)
        assert active[0].acknowledged
        assert not active[0].resolved

        # Resolve
        assert alert_manager.resolve_alert(alert.alert_id)
        assert active[0].resolved

        # No longer active
        still_active = alert_manager.get_active_alerts()
        assert len(still_active) == 0

    def test_multiple_alerts(self, water_chemistry_calculator, alert_manager, out_of_spec_water_sample):
        """Test multiple alerts for severely out-of-spec water."""
        result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            out_of_spec_water_sample
        )

        alerts = alert_manager.check_water_chemistry(result)

        # Should generate multiple alerts
        assert len(alerts) >= 1

        # Check alert details
        for alert in alerts:
            assert alert.alert_id is not None
            assert alert.level in [AlertLevel.NORMAL, AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EMERGENCY]
            assert alert.parameter is not None
            assert alert.message is not None


# ============================================================================
# Report Generation Tests
# ============================================================================

@pytest.mark.integration
class TestReportGeneration:
    """Test report generation functionality."""

    def test_basic_report_generation(
        self,
        water_chemistry_calculator,
        standard_water_sample
    ):
        """Test basic report generation."""
        result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            standard_water_sample
        )

        report = WaterTreatmentReport(boiler_id="BOILER-001")
        report.add_water_chemistry_section(result)

        generated = report.generate()

        assert generated['boiler_id'] == "BOILER-001"
        assert 'water_chemistry' in generated['sections']
        assert 'provenance_hash' in generated['sections']['water_chemistry']

    def test_comprehensive_report(
        self,
        water_chemistry_calculator,
        scale_calculator,
        corrosion_calculator,
        alert_manager,
        standard_water_sample,
        standard_scale_conditions,
        standard_corrosion_conditions
    ):
        """Test comprehensive report with all sections."""
        # Run analyses
        water_result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            standard_water_sample
        )
        scale_result = scale_calculator.calculate_comprehensive_scale_analysis(
            standard_scale_conditions
        )
        corrosion_result = corrosion_calculator.calculate_comprehensive_corrosion_analysis(
            standard_corrosion_conditions
        )

        # Check for alerts
        alerts = alert_manager.check_water_chemistry(water_result)
        alerts.extend(alert_manager.check_corrosion(corrosion_result))

        # Generate report
        report = WaterTreatmentReport(boiler_id="BOILER-001", report_period="weekly")
        report.add_water_chemistry_section(water_result)
        report.add_scale_analysis_section(scale_result)
        report.add_corrosion_section(corrosion_result)
        report.add_alerts_section(alert_manager.get_active_alerts())
        report.add_recommendations_section([
            "Maintain current pH control",
            "Monitor silica levels",
            "Schedule chemical cleaning in 3 months"
        ])

        generated = report.generate()

        # Verify all sections
        assert len(generated['sections']) == 5
        assert 'water_chemistry' in generated['sections']
        assert 'scale_formation' in generated['sections']
        assert 'corrosion' in generated['sections']
        assert 'alerts' in generated['sections']
        assert 'recommendations' in generated['sections']

    def test_report_json_serialization(
        self,
        water_chemistry_calculator,
        standard_water_sample
    ):
        """Test report can be serialized to JSON."""
        result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            standard_water_sample
        )

        report = WaterTreatmentReport(boiler_id="BOILER-001")
        report.add_water_chemistry_section(result)

        generated = report.generate()

        # Should be JSON serializable
        json_str = json.dumps(generated, default=str)
        assert len(json_str) > 0

        # Should be deserializable
        loaded = json.loads(json_str)
        assert loaded['boiler_id'] == "BOILER-001"


# ============================================================================
# Data Pipeline Tests
# ============================================================================

@pytest.mark.integration
class TestDataPipeline:
    """Test data pipeline integration."""

    def test_sequential_analyses(
        self,
        water_chemistry_calculator,
        scale_calculator,
        corrosion_calculator,
        standard_water_sample,
        standard_scale_conditions,
        standard_corrosion_conditions
    ):
        """Test sequential analysis pipeline."""
        results = []

        # Run water chemistry
        water_result = water_chemistry_calculator.calculate_water_chemistry_analysis(
            standard_water_sample
        )
        results.append(('water_chemistry', water_result))

        # Use water chemistry to inform scale conditions
        # (In real system, would update scale conditions based on water chemistry)
        scale_result = scale_calculator.calculate_comprehensive_scale_analysis(
            standard_scale_conditions
        )
        results.append(('scale_formation', scale_result))

        # Corrosion analysis
        corrosion_result = corrosion_calculator.calculate_comprehensive_corrosion_analysis(
            standard_corrosion_conditions
        )
        results.append(('corrosion', corrosion_result))

        # Verify all results have provenance
        for name, result in results:
            assert 'provenance' in result, f"{name} missing provenance"
            assert 'provenance_hash' in result['provenance'], f"{name} missing hash"

    def test_batch_analysis(self, water_chemistry_calculator):
        """Test batch analysis of multiple samples."""
        samples = []

        # Create batch of samples with varying conditions
        for i in range(5):
            sample = WaterSample(
                temperature_c=25.0 + i * 10,
                ph=7.0 + i * 0.5,
                conductivity_us_cm=500.0 + i * 100,
                calcium_mg_l=50.0 + i * 10,
                magnesium_mg_l=25.0 + i * 5,
                sodium_mg_l=50.0,
                potassium_mg_l=5.0,
                chloride_mg_l=50.0,
                sulfate_mg_l=50.0,
                bicarbonate_mg_l=100.0 + i * 20,
                carbonate_mg_l=5.0,
                hydroxide_mg_l=0.0,
                silica_mg_l=10.0,
                iron_mg_l=0.05,
                copper_mg_l=0.01,
                phosphate_mg_l=0.0,
                dissolved_oxygen_mg_l=8.0 - i * 1.0,
                total_alkalinity_mg_l_caco3=100.0 + i * 20,
                total_hardness_mg_l_caco3=200.0 + i * 40
            )
            samples.append(sample)

        # Analyze all samples
        results = [
            water_chemistry_calculator.calculate_water_chemistry_analysis(s)
            for s in samples
        ]

        # All should have results
        assert len(results) == 5

        # All hashes should be different (different inputs)
        hashes = [r['provenance']['provenance_hash'] for r in results]
        assert len(set(hashes)) == 5


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in integration scenarios."""

    @pytest.mark.asyncio
    async def test_scada_connection_failure(self, mock_scada_server):
        """Test handling of SCADA connection failure."""
        mock_scada_server.connect.return_value = False

        connected = await mock_scada_server.connect()
        assert not connected

    def test_analysis_with_missing_data(self, water_chemistry_calculator):
        """Test analysis handles edge case data."""
        # Minimal sample (near-zero values)
        sample = WaterSample(
            temperature_c=25.0,
            ph=7.0,
            conductivity_us_cm=10.0,
            calcium_mg_l=0.1,
            magnesium_mg_l=0.1,
            sodium_mg_l=1.0,
            potassium_mg_l=0.1,
            chloride_mg_l=1.0,
            sulfate_mg_l=1.0,
            bicarbonate_mg_l=5.0,
            carbonate_mg_l=0.0,
            hydroxide_mg_l=0.0,
            silica_mg_l=0.1,
            iron_mg_l=0.01,
            copper_mg_l=0.001,
            phosphate_mg_l=0.0,
            dissolved_oxygen_mg_l=10.0,
            total_alkalinity_mg_l_caco3=5.0,
            total_hardness_mg_l_caco3=1.0
        )

        result = water_chemistry_calculator.calculate_water_chemistry_analysis(sample)

        # Should complete without error
        assert result is not None
        assert 'provenance' in result

    def test_alert_manager_no_alerts(self, alert_manager):
        """Test alert manager with no alerts generated."""
        # Check water with acceptable values
        mock_result = {
            'water_chemistry': {
                'ph': 9.0  # Within acceptable range
            },
            'scaling_indices': {
                'lsi': {'value': 0.5}  # Slight scaling tendency but within range
            }
        }

        alerts = alert_manager.check_water_chemistry(mock_result)

        # No critical alerts expected
        critical = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical) == 0
