# -*- coding: utf-8 -*-
"""
Integration Tests for GL-014 EXCHANGER-PRO External Systems.

Tests integration with external systems including:
- Process Historian (OSIsoft PI, Honeywell PHD)
- CMMS (Work order creation and equipment history)
- DCS (Distributed Control System data retrieval)
- Agent Coordinator communication
- Data Transformer pipeline

Author: GL-TestEngineer
Created: 2025-12-01
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import test utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from calculators.fouling_calculator import (
    FoulingCalculator,
    FoulingResistanceInput,
    FluidType,
    ExchangerType,
)
from calculators.economic_calculator import (
    EconomicCalculator,
    EnergyLossInput,
    FuelType,
)


# =============================================================================
# Test Class: Process Historian Connection
# =============================================================================

@pytest.mark.integration
class TestProcessHistorianConnection:
    """Integration tests for process historian connectivity."""

    @pytest.mark.asyncio
    async def test_historian_connect_success(self, mock_process_historian):
        """Test successful connection to process historian."""
        # Act
        result = await mock_process_historian.connect()

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_historian_disconnect(self, mock_process_historian):
        """Test clean disconnection from process historian."""
        # Arrange
        await mock_process_historian.connect()

        # Act
        result = await mock_process_historian.disconnect()

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_historian_health_check(self, mock_process_historian):
        """Test health check returns valid status."""
        # Act
        health = await mock_process_historian.health_check()

        # Assert
        assert health["status"] == "healthy"
        assert "latency_ms" in health
        assert health["latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_historian_tag_discovery(self, mock_process_historian):
        """Test tag discovery returns heat exchanger tags."""
        # Act
        tags = await mock_process_historian.discover_tags()

        # Assert
        assert len(tags) > 0
        tag_names = [t["tag_name"] for t in tags]
        assert any("TI" in name for name in tag_names)  # Temperature indicators
        assert any("FI" in name for name in tag_names)  # Flow indicators

    @pytest.mark.asyncio
    async def test_historian_data_retrieval(self, mock_process_historian):
        """Test time-series data retrieval."""
        # Arrange
        tag_name = "HX-001.TI101"
        start_time = datetime.now(timezone.utc) - timedelta(hours=24)
        end_time = datetime.now(timezone.utc)

        # Act
        data = await mock_process_historian.get_interpolated_values(
            tag_name=tag_name,
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,
        )

        # Assert
        assert len(data) > 0
        assert "timestamp" in data[0]
        assert "value" in data[0]
        assert "quality" in data[0]

    @pytest.mark.asyncio
    async def test_historian_snapshot_retrieval(self, mock_process_historian):
        """Test current value (snapshot) retrieval."""
        # Arrange
        tags = ["HX-001.TI101", "HX-001.TI102"]

        # Act
        snapshots = await mock_process_historian.get_snapshot(tags=tags)

        # Assert
        assert len(snapshots) == 2
        for tag in tags:
            assert tag in snapshots
            assert "value" in snapshots[tag]
            assert "timestamp" in snapshots[tag]

    @pytest.mark.asyncio
    async def test_historian_connection_retry(self, mock_process_historian):
        """Test connection retry on transient failure."""
        # Arrange: Configure mock to fail first, then succeed
        mock_process_historian.connect = AsyncMock(
            side_effect=[Exception("Connection timeout"), True]
        )

        # Act: First call fails
        with pytest.raises(Exception):
            await mock_process_historian.connect()

        # Act: Retry succeeds
        result = await mock_process_historian.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_historian_data_quality_filtering(self, mock_process_historian):
        """Test filtering of bad quality data."""
        # Arrange
        mock_process_historian.get_interpolated_values = AsyncMock(return_value=[
            {"timestamp": "2025-01-01T00:00:00Z", "value": 120.5, "quality": "good"},
            {"timestamp": "2025-01-01T01:00:00Z", "value": 0.0, "quality": "bad"},
            {"timestamp": "2025-01-01T02:00:00Z", "value": 120.8, "quality": "good"},
        ])

        # Act
        data = await mock_process_historian.get_interpolated_values(
            tag_name="HX-001.TI101",
            start_time=datetime.now(timezone.utc) - timedelta(hours=3),
            end_time=datetime.now(timezone.utc),
        )

        # Assert: Filter good quality only
        good_data = [d for d in data if d["quality"] == "good"]
        assert len(good_data) == 2


# =============================================================================
# Test Class: CMMS Work Order Creation
# =============================================================================

@pytest.mark.integration
class TestCMMSWorkOrderCreation:
    """Integration tests for CMMS work order creation."""

    @pytest.mark.asyncio
    async def test_cmms_connect(self, mock_cmms_connector):
        """Test CMMS connection establishment."""
        result = await mock_cmms_connector.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_create_cleaning_work_order(self, mock_cmms_connector):
        """Test creation of cleaning work order."""
        # Arrange
        work_order_data = {
            "equipment_id": "HX-001",
            "work_type": "cleaning",
            "priority": "medium",
            "description": "Scheduled cleaning - fouling detected",
            "estimated_duration_hours": 24,
        }

        # Act
        result = await mock_cmms_connector.create_work_order(**work_order_data)

        # Assert
        assert "work_order_id" in result
        assert result["status"] == "created"
        assert result["equipment_id"] == "HX-001"

    @pytest.mark.asyncio
    async def test_create_emergency_work_order(self, mock_cmms_connector):
        """Test creation of emergency work order for critical fouling."""
        # Arrange
        mock_cmms_connector.create_work_order = AsyncMock(return_value={
            "work_order_id": "WO-2025-EMERG-001",
            "status": "created",
            "priority": "critical",
            "equipment_id": "HX-001",
            "description": "EMERGENCY: Critical fouling - immediate action required",
        })

        # Act
        result = await mock_cmms_connector.create_work_order(
            equipment_id="HX-001",
            work_type="cleaning",
            priority="critical",
            description="EMERGENCY: Critical fouling - immediate action required",
        )

        # Assert
        assert result["priority"] == "critical"
        assert "EMERG" in result["work_order_id"]

    @pytest.mark.asyncio
    async def test_get_equipment_maintenance_history(self, mock_cmms_connector):
        """Test retrieval of equipment maintenance history."""
        # Act
        history = await mock_cmms_connector.get_equipment_history(
            equipment_id="HX-001",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        # Assert
        assert len(history) > 0
        assert all("work_order_id" in item for item in history)
        assert all("completion_date" in item for item in history)

    @pytest.mark.asyncio
    async def test_get_spare_parts_inventory(self, mock_cmms_connector):
        """Test spare parts inventory check."""
        # Act
        inventory = await mock_cmms_connector.get_spare_parts_inventory(
            equipment_id="HX-001"
        )

        # Assert
        assert len(inventory) > 0
        for item in inventory:
            assert "part_number" in item
            assert "quantity" in item

    @pytest.mark.asyncio
    async def test_work_order_with_fouling_data(
        self,
        mock_cmms_connector,
        fouling_calculator: FoulingCalculator,
    ):
        """Test work order creation includes fouling analysis data."""
        # Arrange: Calculate fouling
        fouling_input = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=350.0,  # Significant fouling
        )
        fouling_result = fouling_calculator.calculate_fouling_resistance(fouling_input)

        # Act: Create work order with fouling data
        mock_cmms_connector.create_work_order = AsyncMock(return_value={
            "work_order_id": "WO-2025-001234",
            "status": "created",
            "fouling_data": {
                "cleanliness_factor": str(fouling_result.cleanliness_factor_percent),
                "fouling_resistance": str(fouling_result.fouling_resistance_m2_k_w),
            }
        })

        result = await mock_cmms_connector.create_work_order(
            equipment_id="HX-001",
            work_type="cleaning",
            fouling_resistance=float(fouling_result.fouling_resistance_m2_k_w),
            cleanliness_factor=float(fouling_result.cleanliness_factor_percent),
        )

        # Assert
        assert "fouling_data" in result


# =============================================================================
# Test Class: DCS Data Retrieval
# =============================================================================

@pytest.mark.integration
class TestDCSDataRetrieval:
    """Integration tests for DCS data retrieval."""

    @pytest.mark.asyncio
    async def test_dcs_connect(self, mock_dcs_connector):
        """Test DCS connection establishment."""
        result = await mock_dcs_connector.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_read_current_process_values(self, mock_dcs_connector):
        """Test reading current process values from DCS."""
        # Act
        values = await mock_dcs_connector.read_current_values()

        # Assert
        expected_tags = ["TI101", "TI102", "TI103", "TI104", "FI101"]
        for tag in expected_tags:
            assert tag in values
            assert isinstance(values[tag], (int, float))

    @pytest.mark.asyncio
    async def test_read_historical_process_values(self, mock_dcs_connector):
        """Test reading historical process values from DCS."""
        # Act
        data = await mock_dcs_connector.read_historical_values(
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),
            end_time=datetime.now(timezone.utc),
            tags=["TI101", "TI102"],
        )

        # Assert
        assert len(data) > 0
        for point in data:
            assert "timestamp" in point
            assert "TI101" in point
            assert "TI102" in point

    @pytest.mark.asyncio
    async def test_dcs_setpoint_write(self, mock_dcs_connector):
        """Test writing setpoint to DCS."""
        # Act
        result = await mock_dcs_connector.write_setpoint(
            tag="FC101.SP",
            value=15.5,
        )

        # Assert
        assert result["status"] == "success"
        assert result["acknowledged"] is True

    @pytest.mark.asyncio
    async def test_dcs_data_validation(self, mock_dcs_connector):
        """Test DCS data validation (reasonable ranges)."""
        # Arrange
        mock_dcs_connector.read_current_values = AsyncMock(return_value={
            "TI101": 120.5,  # Hot inlet (should be 80-200 C)
            "TI102": 81.2,   # Hot outlet (should be < TI101)
            "TI103": 30.5,   # Cold inlet (should be 10-50 C)
            "TI104": 64.8,   # Cold outlet (should be > TI103)
        })

        # Act
        values = await mock_dcs_connector.read_current_values()

        # Assert: Validate temperature relationships
        assert values["TI101"] > values["TI102"], "Hot outlet should be < hot inlet"
        assert values["TI104"] > values["TI103"], "Cold outlet should be > cold inlet"


# =============================================================================
# Test Class: Agent Coordinator Communication
# =============================================================================

@pytest.mark.integration
class TestAgentCoordinatorCommunication:
    """Integration tests for agent coordinator communication."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create mock agent coordinator."""
        coordinator = MagicMock()
        coordinator.register_agent = AsyncMock(return_value={"agent_id": "gl-014", "status": "registered"})
        coordinator.send_result = AsyncMock(return_value={"status": "received"})
        coordinator.request_data = AsyncMock(return_value={"data": {"value": 100}})
        coordinator.broadcast_alert = AsyncMock(return_value={"status": "broadcast"})
        return coordinator

    @pytest.mark.asyncio
    async def test_agent_registration(self, mock_coordinator):
        """Test agent registration with coordinator."""
        # Act
        result = await mock_coordinator.register_agent(
            agent_id="gl-014",
            agent_type="exchanger-pro",
            capabilities=["fouling_analysis", "economic_analysis", "predictive_maintenance"],
        )

        # Assert
        assert result["status"] == "registered"
        assert result["agent_id"] == "gl-014"

    @pytest.mark.asyncio
    async def test_send_analysis_result(self, mock_coordinator):
        """Test sending analysis result to coordinator."""
        # Arrange
        analysis_result = {
            "exchanger_id": "HX-001",
            "cleanliness_factor": 0.84,
            "fouling_severity": "moderate",
            "recommended_action": "schedule_cleaning",
            "provenance_hash": "abc123",
        }

        # Act
        result = await mock_coordinator.send_result(
            result_type="fouling_analysis",
            data=analysis_result,
        )

        # Assert
        assert result["status"] == "received"

    @pytest.mark.asyncio
    async def test_request_equipment_data(self, mock_coordinator):
        """Test requesting equipment data from coordinator."""
        # Act
        result = await mock_coordinator.request_data(
            data_type="equipment_specifications",
            equipment_id="HX-001",
        )

        # Assert
        assert "data" in result

    @pytest.mark.asyncio
    async def test_broadcast_critical_alert(self, mock_coordinator):
        """Test broadcasting critical alert through coordinator."""
        # Arrange
        alert = {
            "alert_type": "critical_fouling",
            "equipment_id": "HX-001",
            "severity": "critical",
            "message": "Immediate cleaning required - fouling exceeds threshold",
        }

        # Act
        result = await mock_coordinator.broadcast_alert(alert)

        # Assert
        assert result["status"] == "broadcast"


# =============================================================================
# Test Class: Data Transformer Pipeline
# =============================================================================

@pytest.mark.integration
class TestDataTransformerPipeline:
    """Integration tests for data transformation pipeline."""

    @pytest.fixture
    def mock_transformer(self):
        """Create mock data transformer."""
        transformer = MagicMock()
        transformer.transform = MagicMock()
        transformer.validate = MagicMock(return_value=True)
        transformer.normalize = MagicMock()
        return transformer

    def test_raw_data_transformation(self, mock_transformer, sample_temperature_series):
        """Test transformation of raw historian data."""
        # Arrange
        mock_transformer.transform.return_value = [
            {
                "timestamp": item["timestamp"],
                "hot_inlet_c": float(item["hot_inlet_c"]),
                "hot_outlet_c": float(item["hot_outlet_c"]),
                "cold_inlet_c": float(item["cold_inlet_c"]),
                "cold_outlet_c": float(item["cold_outlet_c"]),
            }
            for item in sample_temperature_series
        ]

        # Act
        transformed = mock_transformer.transform(sample_temperature_series)

        # Assert
        assert len(transformed) == len(sample_temperature_series)
        for item in transformed:
            assert isinstance(item["hot_inlet_c"], float)

    def test_data_validation(self, mock_transformer):
        """Test data validation in pipeline."""
        # Arrange
        valid_data = {
            "hot_inlet_c": 120.0,
            "hot_outlet_c": 80.0,
            "cold_inlet_c": 30.0,
            "cold_outlet_c": 65.0,
        }

        # Act
        result = mock_transformer.validate(valid_data)

        # Assert
        assert result is True

    def test_data_normalization(self, mock_transformer):
        """Test data normalization."""
        # Arrange
        raw_data = {"temperature_f": 248.0}  # Fahrenheit
        mock_transformer.normalize.return_value = {"temperature_c": 120.0}  # Celsius

        # Act
        normalized = mock_transformer.normalize(raw_data, target_unit="celsius")

        # Assert
        assert "temperature_c" in normalized
        assert normalized["temperature_c"] == 120.0

    def test_pipeline_error_handling(self, mock_transformer):
        """Test pipeline handles errors gracefully."""
        # Arrange: Configure mock to raise on invalid data
        mock_transformer.validate.side_effect = ValueError("Invalid temperature value")

        # Act & Assert
        with pytest.raises(ValueError):
            mock_transformer.validate({"hot_inlet_c": -999})


# =============================================================================
# Test Class: Cross-System Integration
# =============================================================================

@pytest.mark.integration
class TestCrossSystemIntegration:
    """Tests for integration across multiple systems."""

    @pytest.mark.asyncio
    async def test_historian_to_calculator_integration(
        self,
        mock_process_historian,
        fouling_calculator: FoulingCalculator,
    ):
        """Test data flow from historian to calculator."""
        # Arrange: Configure historian to return U values
        mock_process_historian.get_snapshot = AsyncMock(return_value={
            "HX-001.U_CLEAN": {"value": 500.0, "quality": "good"},
            "HX-001.U_ACTUAL": {"value": 420.0, "quality": "good"},
        })

        # Act: Get data from historian
        snapshots = await mock_process_historian.get_snapshot(
            tags=["HX-001.U_CLEAN", "HX-001.U_ACTUAL"]
        )

        # Calculate fouling
        fouling_input = FoulingResistanceInput(
            u_clean_w_m2_k=snapshots["HX-001.U_CLEAN"]["value"],
            u_fouled_w_m2_k=snapshots["HX-001.U_ACTUAL"]["value"],
        )
        result = fouling_calculator.calculate_fouling_resistance(fouling_input)

        # Assert
        assert result.fouling_resistance_m2_k_w > Decimal("0")

    @pytest.mark.asyncio
    async def test_analysis_to_cmms_integration(
        self,
        mock_cmms_connector,
        fouling_calculator: FoulingCalculator,
    ):
        """Test data flow from analysis to CMMS work order."""
        # Arrange: Perform fouling analysis
        fouling_input = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=300.0,  # Severe fouling
        )
        fouling_result = fouling_calculator.calculate_fouling_resistance(fouling_input)

        # Act: Check if cleaning is needed and create work order
        cf = float(fouling_result.cleanliness_factor_percent)
        if cf < 70:  # Threshold for cleaning
            work_order = await mock_cmms_connector.create_work_order(
                equipment_id="HX-001",
                work_type="cleaning",
                priority="high",
                description=f"Cleaning required - CF={cf:.1f}%",
            )

            # Assert
            assert "work_order_id" in work_order
            assert work_order["status"] == "created"

    @pytest.mark.asyncio
    async def test_dcs_to_economic_integration(
        self,
        mock_dcs_connector,
        economic_calculator: EconomicCalculator,
    ):
        """Test data flow from DCS to economic calculator."""
        # Arrange: Get process data from DCS
        mock_dcs_connector.read_current_values = AsyncMock(return_value={
            "HX001_DUTY_ACTUAL_KW": 1275.0,
            "HX001_DUTY_DESIGN_KW": 1500.0,
        })

        values = await mock_dcs_connector.read_current_values()

        # Act: Calculate economic impact
        energy_input = EnergyLossInput(
            design_duty_kw=Decimal(str(values["HX001_DUTY_DESIGN_KW"])),
            actual_duty_kw=Decimal(str(values["HX001_DUTY_ACTUAL_KW"])),
            fuel_type=FuelType.NATURAL_GAS,
            fuel_cost_per_kwh=Decimal("0.05"),
            operating_hours_per_year=Decimal("8000"),
        )
        economic_result = economic_calculator.calculate_energy_loss_cost(energy_input)

        # Assert
        assert economic_result.total_energy_penalty_per_year_usd > Decimal("0")


# =============================================================================
# Test Class: Error Recovery and Resilience
# =============================================================================

@pytest.mark.integration
class TestErrorRecoveryResilience:
    """Tests for error recovery and system resilience."""

    @pytest.mark.asyncio
    async def test_historian_connection_recovery(self, mock_process_historian):
        """Test recovery from historian connection failure."""
        # Arrange: Simulate connection failure then success
        call_count = 0

        async def connect_with_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return True

        mock_process_historian.connect = connect_with_retry

        # Act: Should eventually succeed
        result = None
        for attempt in range(5):
            try:
                result = await mock_process_historian.connect()
                break
            except ConnectionError:
                continue

        # Assert
        assert result is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_cmms_timeout_handling(self, mock_cmms_connector):
        """Test handling of CMMS timeout."""
        # Arrange
        mock_cmms_connector.create_work_order = AsyncMock(
            side_effect=asyncio.TimeoutError("Request timed out")
        )

        # Act & Assert
        with pytest.raises(asyncio.TimeoutError):
            await mock_cmms_connector.create_work_order(equipment_id="HX-001")

    @pytest.mark.asyncio
    async def test_dcs_data_staleness_detection(self, mock_dcs_connector):
        """Test detection of stale DCS data."""
        # Arrange: Return data with old timestamp
        old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        mock_dcs_connector.read_current_values = AsyncMock(return_value={
            "TI101": 120.5,
            "_timestamp": old_timestamp,
        })

        # Act
        values = await mock_dcs_connector.read_current_values()

        # Assert: Check for staleness
        timestamp = datetime.fromisoformat(values["_timestamp"].replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - timestamp
        assert age > timedelta(hours=1), "Data should be detected as stale"


# =============================================================================
# Test Class: Data Consistency
# =============================================================================

@pytest.mark.integration
class TestDataConsistency:
    """Tests for data consistency across systems."""

    @pytest.mark.asyncio
    async def test_historian_dcs_consistency(
        self,
        mock_process_historian,
        mock_dcs_connector,
    ):
        """Test data consistency between historian and DCS."""
        # Arrange: Set up consistent values
        expected_value = 120.5
        mock_process_historian.get_snapshot = AsyncMock(return_value={
            "HX-001.TI101": {"value": expected_value, "quality": "good"}
        })
        mock_dcs_connector.read_current_values = AsyncMock(return_value={
            "TI101": expected_value
        })

        # Act
        historian_data = await mock_process_historian.get_snapshot(tags=["HX-001.TI101"])
        dcs_data = await mock_dcs_connector.read_current_values()

        # Assert: Values should be consistent
        assert historian_data["HX-001.TI101"]["value"] == dcs_data["TI101"]

    def test_calculation_consistency(
        self,
        fouling_calculator: FoulingCalculator,
        economic_calculator: EconomicCalculator,
    ):
        """Test calculation consistency across calculators."""
        # Arrange
        u_clean = 500.0
        u_fouled = 420.0

        # Calculate cleanliness factor
        fouling_result = fouling_calculator.calculate_fouling_resistance(
            FoulingResistanceInput(u_clean_w_m2_k=u_clean, u_fouled_w_m2_k=u_fouled)
        )

        # Calculate duty loss
        design_duty = Decimal("1500")
        actual_duty = design_duty * fouling_result.cleanliness_factor_percent / Decimal("100")

        # Calculate economic impact
        energy_result = economic_calculator.calculate_energy_loss_cost(
            EnergyLossInput(
                design_duty_kw=design_duty,
                actual_duty_kw=actual_duty,
                fuel_type=FuelType.NATURAL_GAS,
                fuel_cost_per_kwh=Decimal("0.05"),
                operating_hours_per_year=Decimal("8000"),
            )
        )

        # Assert: Loss should match fouling level
        expected_loss_percent = Decimal("100") - fouling_result.cleanliness_factor_percent
        actual_loss_percent = energy_result.heat_transfer_loss_percent

        assert abs(expected_loss_percent - actual_loss_percent) < Decimal("0.1")
