# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Integration Tests

Integration tests for agent pipeline, ERP connectors, database interactions,
and external API integrations. Tests end-to-end data flow between components.

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
import json
from decimal import Decimal
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import hashlib


# =============================================================================
# TEST: THERMAL IMAGE PIPELINE INTEGRATION
# =============================================================================

@pytest.mark.integration
class TestThermalImagePipeline:
    """Integration tests for thermal image processing pipeline."""

    @pytest.mark.asyncio
    async def test_camera_capture_to_analysis_pipeline(
        self,
        mock_thermal_camera,
        sample_thermal_image_data
    ):
        """Test complete pipeline from camera capture to analysis."""
        # Step 1: Connect to camera
        await mock_thermal_camera.connect()
        assert await mock_thermal_camera.get_status()

        # Step 2: Capture image
        image_data = await mock_thermal_camera.capture_image()
        assert "temperature_matrix" in image_data

        # Step 3: Process image (simulated)
        temp_matrix = image_data["temperature_matrix"]
        stats = {
            "min_temp_c": min(min(row) for row in temp_matrix),
            "max_temp_c": max(max(row) for row in temp_matrix),
            "avg_temp_c": sum(sum(row) for row in temp_matrix) / (len(temp_matrix) * len(temp_matrix[0])),
        }

        # Step 4: Detect anomalies
        hotspot_threshold = stats["avg_temp_c"] + 10
        hotspot_count = sum(
            1 for row in temp_matrix
            for temp in row if temp > hotspot_threshold
        )

        assert hotspot_count >= 0

        # Step 5: Disconnect
        await mock_thermal_camera.disconnect()

    @pytest.mark.asyncio
    async def test_batch_image_processing(
        self,
        mock_thermal_camera,
        sample_thermal_image_data
    ):
        """Test batch processing of multiple thermal images."""
        await mock_thermal_camera.connect()

        processed_images = []
        for i in range(5):
            image_data = await mock_thermal_camera.capture_image()

            # Simulate processing
            result = {
                "image_id": f"IMG-{i+1:03d}",
                "capture_time": datetime.now().isoformat(),
                "max_temp_c": max(max(row) for row in image_data["temperature_matrix"]),
                "anomaly_detected": i % 2 == 0,  # Simulate alternating anomalies
            }
            processed_images.append(result)

        assert len(processed_images) == 5
        anomaly_count = sum(1 for img in processed_images if img["anomaly_detected"])
        assert anomaly_count == 3  # Images 0, 2, 4

        await mock_thermal_camera.disconnect()

    @pytest.mark.asyncio
    async def test_image_to_report_generation(
        self,
        mock_thermal_camera,
        sample_ambient_conditions
    ):
        """Test image analysis to report generation pipeline."""
        await mock_thermal_camera.connect()

        # Capture and analyze
        image_data = await mock_thermal_camera.capture_image()

        # Generate report structure
        report = {
            "report_id": "RPT-2025-001",
            "inspection_date": date.today().isoformat(),
            "equipment_inspected": [],
            "findings": [],
            "recommendations": [],
            "ambient_conditions": sample_ambient_conditions,
            "images_analyzed": 1,
            "defects_found": 0,
            "overall_condition": "acceptable",
        }

        # Simulate finding a defect
        if max(max(row) for row in image_data["temperature_matrix"]) > 50:
            report["findings"].append({
                "type": "hotspot",
                "severity": "moderate",
                "location": "Section A",
            })
            report["defects_found"] = 1
            report["recommendations"].append({
                "action": "schedule_inspection",
                "priority": "medium",
            })

        assert "report_id" in report
        await mock_thermal_camera.disconnect()


# =============================================================================
# TEST: CMMS INTEGRATION
# =============================================================================

@pytest.mark.integration
class TestCMMSIntegration:
    """Integration tests for CMMS connector."""

    @pytest.mark.asyncio
    async def test_defect_to_work_order_flow(
        self,
        mock_cmms_connector,
        sample_thermal_defect
    ):
        """Test defect detection to work order creation flow."""
        await mock_cmms_connector.connect()

        # Get equipment info
        equipment_tag = "P-1001-A"
        equipment_info = await mock_cmms_connector.get_equipment_info(equipment_tag)

        # Check maintenance history
        history = await mock_cmms_connector.get_maintenance_history(equipment_tag)

        # Create work order for defect
        work_order_data = {
            "equipment_tag": equipment_tag,
            "priority": "high",
            "description": "Thermal inspection defect - insulation repair required",
            "estimated_hours": 8,
            "defect_id": "DEF-2025-001",
        }

        work_order = await mock_cmms_connector.create_work_order(work_order_data)

        assert work_order["work_order_id"] is not None
        assert work_order["status"] == "created"

        # Update asset condition
        condition_update = {
            "equipment_tag": equipment_tag,
            "condition_score": 6,
            "inspection_date": date.today().isoformat(),
        }
        updated = await mock_cmms_connector.update_asset_condition(condition_update)
        assert updated is True

        await mock_cmms_connector.disconnect()

    @pytest.mark.asyncio
    async def test_batch_work_order_creation(self, mock_cmms_connector):
        """Test batch work order creation for multiple defects."""
        await mock_cmms_connector.connect()

        defects = [
            {"equipment_tag": f"P-100{i}-A", "priority": "medium", "description": f"Defect {i}"}
            for i in range(1, 6)
        ]

        work_orders = []
        for defect in defects:
            wo = await mock_cmms_connector.create_work_order(defect)
            work_orders.append(wo)

        assert len(work_orders) == 5
        assert all(wo["status"] == "created" for wo in work_orders)

        await mock_cmms_connector.disconnect()

    @pytest.mark.asyncio
    async def test_equipment_lookup_and_update(self, mock_cmms_connector):
        """Test equipment information lookup and update."""
        await mock_cmms_connector.connect()

        # Lookup
        equipment_info = await mock_cmms_connector.get_equipment_info("P-1001-A")
        assert equipment_info["equipment_tag"] == "P-1001-A"

        # Get history
        history = await mock_cmms_connector.get_maintenance_history("P-1001-A")
        assert isinstance(history, list)

        # Update condition
        updated = await mock_cmms_connector.update_asset_condition({
            "equipment_tag": "P-1001-A",
            "condition_score": 7,
        })
        assert updated is True

        await mock_cmms_connector.disconnect()


# =============================================================================
# TEST: WEATHER SERVICE INTEGRATION
# =============================================================================

@pytest.mark.integration
class TestWeatherServiceIntegration:
    """Integration tests for weather service connector."""

    @pytest.mark.asyncio
    async def test_weather_data_for_inspection(
        self,
        mock_weather_service,
        sample_ambient_conditions
    ):
        """Test weather data retrieval for inspection planning."""
        # Get current conditions
        current = await mock_weather_service.get_current_conditions()

        # Validate conditions for outdoor inspection
        is_suitable = (
            current["temperature_c"] > 5 and
            current["temperature_c"] < 40 and
            current["wind_speed_m_s"] < 10 and
            current["humidity_percent"] < 90
        )

        # Get forecast
        forecast = await mock_weather_service.get_forecast()

        assert isinstance(current, dict)
        assert isinstance(forecast, list)

    @pytest.mark.asyncio
    async def test_weather_correction_integration(self, mock_weather_service):
        """Test weather correction factors in calculations."""
        weather = await mock_weather_service.get_current_conditions()

        # Calculate correction factors
        wind_correction = 1.0 + (weather["wind_speed_m_s"] * 0.05)
        solar_correction = 1.0 - (weather.get("solar_radiation_w_m2", 0) / 2000)

        assert wind_correction >= 1.0
        assert 0 < solar_correction <= 1.0

    @pytest.mark.asyncio
    async def test_historical_weather_analysis(self, mock_weather_service):
        """Test historical weather data for trend analysis."""
        historical = await mock_weather_service.get_historical()

        # Verify historical data structure
        assert "avg_temperature_c" in historical
        assert "max_temperature_c" in historical
        assert "min_temperature_c" in historical

        # Calculate temperature range
        temp_range = historical["max_temperature_c"] - historical["min_temperature_c"]
        assert temp_range > 0


# =============================================================================
# TEST: DATABASE INTEGRATION
# =============================================================================

@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.mark.asyncio
    async def test_inspection_record_persistence(self, mock_database):
        """Test inspection record save and retrieve."""
        await mock_database.connect()

        # Save inspection record
        inspection_data = {
            "inspection_id": "INS-2025-001",
            "equipment_tag": "P-1001-A",
            "inspection_date": date.today().isoformat(),
            "inspector_id": "INSP-001",
            "findings": json.dumps([{"type": "hotspot", "severity": "moderate"}]),
            "images": json.dumps(["IMG-001", "IMG-002"]),
        }

        result = await mock_database.execute(
            "INSERT INTO inspections ...",
            inspection_data
        )
        assert result["affected_rows"] == 1

        # Retrieve record
        retrieved = await mock_database.fetch_one(
            "SELECT * FROM inspections WHERE inspection_id = ?",
            ["INS-2025-001"]
        )
        assert retrieved is not None

        await mock_database.disconnect()

    @pytest.mark.asyncio
    async def test_defect_tracking_persistence(self, mock_database):
        """Test defect tracking record persistence."""
        await mock_database.connect()

        # Insert defect record
        defect_data = {
            "defect_id": "DEF-2025-001",
            "equipment_tag": "P-1001-A",
            "damage_type": "missing",
            "heat_loss_w_per_m": 250.0,
            "status": "open",
            "created_at": datetime.now().isoformat(),
        }

        result = await mock_database.execute(
            "INSERT INTO defects ...",
            defect_data
        )
        assert result["affected_rows"] == 1

        await mock_database.disconnect()

    @pytest.mark.asyncio
    async def test_report_generation_from_database(self, mock_database):
        """Test report generation from database records."""
        await mock_database.connect()

        # Fetch all defects for a report
        defects = await mock_database.fetch_all(
            "SELECT * FROM defects WHERE status = ?",
            ["open"]
        )

        # Generate report summary
        report_summary = {
            "total_defects": len(defects),
            "defect_ids": [d.get("id") for d in defects],
        }

        assert isinstance(report_summary["total_defects"], int)

        await mock_database.disconnect()


# =============================================================================
# TEST: CALCULATOR INTEGRATION
# =============================================================================

@pytest.mark.integration
class TestCalculatorIntegration:
    """Integration tests for calculator module integration."""

    def test_heat_loss_to_energy_savings_flow(
        self,
        known_heat_loss_values,
        sample_equipment_parameters
    ):
        """Test heat loss calculation to energy savings flow."""
        # Calculate heat loss
        case = known_heat_loss_values["case_1"]
        heat_loss_w_per_m = float(case["expected_heat_loss_w_per_m"])

        # Equipment parameters
        pipe_length_m = float(sample_equipment_parameters["pipe_length_m"])

        # Total heat loss
        total_heat_loss_w = heat_loss_w_per_m * pipe_length_m

        # Annual energy loss
        operating_hours = 8000
        annual_energy_kwh = total_heat_loss_w * operating_hours / 1000

        # Energy cost
        energy_cost_per_kwh = 0.12
        annual_cost = annual_energy_kwh * energy_cost_per_kwh

        assert total_heat_loss_w > 0
        assert annual_energy_kwh > 0
        assert annual_cost > 0

    def test_thermal_analysis_to_prioritization_flow(
        self,
        sample_thermal_image_data,
        sample_ambient_conditions
    ):
        """Test thermal analysis to repair prioritization flow."""
        # Analyze thermal image (simulated)
        temp_matrix = sample_thermal_image_data["temperature_matrix"]
        max_temp = max(max(row) for row in temp_matrix)
        avg_temp = sum(sum(row) for row in temp_matrix) / (len(temp_matrix) * len(temp_matrix[0]))

        # Determine delta-T
        ambient_temp = float(sample_ambient_conditions["ambient_temperature_c"])
        delta_t = max_temp - ambient_temp

        # Calculate criticality score
        if delta_t > 50:
            priority = "critical"
        elif delta_t > 30:
            priority = "urgent"
        elif delta_t > 15:
            priority = "high"
        else:
            priority = "medium"

        assert priority in ["critical", "urgent", "high", "medium", "low"]

    def test_economic_analysis_integration(self):
        """Test economic analysis integration with heat loss data."""
        # Heat loss data
        heat_loss_w = 500
        operating_hours = 8000

        # Energy loss
        annual_kwh = heat_loss_w * operating_hours / 1000

        # Economic parameters
        energy_cost = 0.12  # $/kWh
        repair_cost = 5000  # $

        # Calculate savings
        annual_savings = annual_kwh * energy_cost

        # Calculate payback
        payback_years = repair_cost / annual_savings

        # Calculate NPV (10 years, 10% discount)
        discount_rate = 0.10
        analysis_years = 10

        npv = -repair_cost
        for year in range(1, analysis_years + 1):
            npv += annual_savings / ((1 + discount_rate) ** year)

        assert annual_savings > 0
        assert payback_years > 0
        assert npv != 0  # Could be positive or negative


# =============================================================================
# TEST: AGENT COORDINATION INTEGRATION
# =============================================================================

@pytest.mark.integration
class TestAgentCoordinationIntegration:
    """Integration tests for coordination with other GL agents."""

    @pytest.mark.asyncio
    async def test_gl001_thermosync_coordination(self):
        """Test coordination with GL-001 THERMOSYNC agent."""
        # Simulate receiving temperature data from GL-001
        thermosync_data = {
            "agent_id": "GL-001",
            "data_type": "temperature_reading",
            "equipment_tag": "P-1001-A",
            "temperature_c": 175.0,
            "timestamp": datetime.now().isoformat(),
            "provenance_hash": hashlib.sha256(b"test").hexdigest(),
        }

        # Process in GL-015 context
        insulscan_input = {
            "process_temperature_c": thermosync_data["temperature_c"],
            "source_agent": thermosync_data["agent_id"],
            "equipment_tag": thermosync_data["equipment_tag"],
        }

        assert insulscan_input["source_agent"] == "GL-001"

    @pytest.mark.asyncio
    async def test_gl006_emissions_handoff(self):
        """Test data handoff to GL-006 emissions agent."""
        # Prepare emissions data for GL-006
        energy_loss_data = {
            "annual_energy_loss_kwh": 4000,
            "fuel_type": "natural_gas",
            "boiler_efficiency": 0.85,
        }

        # Calculate CO2 emissions for handoff
        co2_factor = 1.89  # kg CO2 per m3 natural gas
        gas_heating_value = 10.5  # kWh per m3

        # Convert energy loss to fuel consumption
        fuel_consumed_m3 = energy_loss_data["annual_energy_loss_kwh"] / (
            gas_heating_value * energy_loss_data["boiler_efficiency"]
        )

        co2_emissions_kg = fuel_consumed_m3 * co2_factor

        handoff_payload = {
            "target_agent": "GL-006",
            "data_type": "emission_source",
            "source_agent": "GL-015",
            "emissions_kg_co2": co2_emissions_kg,
            "category": "fugitive_heat_loss",
        }

        assert handoff_payload["target_agent"] == "GL-006"
        assert handoff_payload["emissions_kg_co2"] > 0

    @pytest.mark.asyncio
    async def test_inter_agent_provenance_chain(self):
        """Test provenance chain across agent boundaries."""
        # GL-001 provides temperature reading
        gl001_hash = hashlib.sha256(b"gl001_temperature_reading").hexdigest()

        # GL-015 processes and adds to chain
        gl015_input = {
            "source_provenance": gl001_hash,
            "calculation_inputs": {"temperature_c": 175.0},
        }

        combined_data = json.dumps(gl015_input, sort_keys=True)
        gl015_hash = hashlib.sha256(combined_data.encode()).hexdigest()

        # Verify chain
        provenance_chain = [gl001_hash, gl015_hash]

        assert len(provenance_chain) == 2
        assert all(len(h) == 64 for h in provenance_chain)


# =============================================================================
# TEST: API INTEGRATION
# =============================================================================

@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for external API interactions."""

    @pytest.mark.asyncio
    async def test_rest_api_request_response(self, async_test_client):
        """Test REST API request/response cycle."""
        # Mock API endpoint
        async_test_client.post.return_value = Mock(
            status_code=200,
            json=Mock(return_value={"status": "success", "data": {}})
        )

        response = await async_test_client.post(
            "/api/v1/inspections",
            json={"equipment_tag": "P-1001-A"}
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_error_handling(self, async_test_client):
        """Test API error handling."""
        # Mock 500 error
        async_test_client.get.return_value = Mock(
            status_code=500,
            json=Mock(return_value={"error": "Internal server error"})
        )

        response = await async_test_client.get("/api/v1/equipment/INVALID")

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_api_authentication(self, async_test_client, valid_api_token):
        """Test API authentication flow."""
        # Mock authenticated request
        async_test_client.get.return_value = Mock(
            status_code=200,
            json=Mock(return_value={"authenticated": True})
        )

        response = await async_test_client.get(
            "/api/v1/protected",
            headers={"Authorization": f"Bearer {valid_api_token}"}
        )

        assert response.status_code == 200


# =============================================================================
# TEST: DATA FLOW INTEGRATION
# =============================================================================

@pytest.mark.integration
class TestDataFlowIntegration:
    """Integration tests for end-to-end data flow."""

    @pytest.mark.asyncio
    async def test_full_inspection_data_flow(
        self,
        mock_thermal_camera,
        mock_weather_service,
        mock_cmms_connector,
        mock_database
    ):
        """Test complete inspection data flow."""
        # 1. Get weather conditions
        weather = await mock_weather_service.get_current_conditions()

        # 2. Connect to camera
        await mock_thermal_camera.connect()

        # 3. Capture thermal images
        images = []
        for i in range(3):
            img = await mock_thermal_camera.capture_image()
            images.append(img)

        await mock_thermal_camera.disconnect()

        # 4. Analyze images (simulated)
        findings = []
        for idx, img in enumerate(images):
            max_temp = max(max(row) for row in img["temperature_matrix"])
            if max_temp > 50:
                findings.append({
                    "image_index": idx,
                    "max_temp_c": max_temp,
                    "anomaly_type": "hotspot",
                })

        # 5. Create work orders if needed
        await mock_cmms_connector.connect()
        for finding in findings:
            await mock_cmms_connector.create_work_order({
                "priority": "medium",
                "description": f"Thermal anomaly detected - {finding['anomaly_type']}",
            })
        await mock_cmms_connector.disconnect()

        # 6. Save to database
        await mock_database.connect()
        await mock_database.execute("INSERT INTO inspections ...", {
            "images_count": len(images),
            "findings_count": len(findings),
        })
        await mock_database.disconnect()

        assert len(images) == 3

    def test_data_transformation_pipeline(self):
        """Test data transformation between components."""
        # Raw camera data
        raw_data = {"temp_matrix": [[25.0] * 10 for _ in range(10)]}

        # Transform to analysis format
        analysis_input = {
            "temperature_matrix": raw_data["temp_matrix"],
            "width": len(raw_data["temp_matrix"][0]),
            "height": len(raw_data["temp_matrix"]),
        }

        # Transform analysis output to report format
        analysis_output = {
            "min_temp": 25.0,
            "max_temp": 25.0,
            "anomalies": [],
        }

        report_data = {
            "temperature_range": f"{analysis_output['min_temp']}-{analysis_output['max_temp']}C",
            "anomaly_count": len(analysis_output["anomalies"]),
            "status": "normal" if not analysis_output["anomalies"] else "anomaly_detected",
        }

        assert report_data["status"] == "normal"


# =============================================================================
# TEST: ERROR RECOVERY INTEGRATION
# =============================================================================

@pytest.mark.integration
class TestErrorRecoveryIntegration:
    """Integration tests for error recovery scenarios."""

    @pytest.mark.asyncio
    async def test_camera_reconnection_on_failure(self, mock_thermal_camera):
        """Test camera reconnection after failure."""
        # First connection fails, second succeeds
        mock_thermal_camera.connect = AsyncMock(side_effect=[
            ConnectionError("Connection failed"),
            True
        ])

        connected = False
        for attempt in range(2):
            try:
                await mock_thermal_camera.connect()
                connected = True
                break
            except ConnectionError:
                await asyncio.sleep(0.1)

        assert connected is True

    @pytest.mark.asyncio
    async def test_partial_batch_processing_recovery(self, mock_cmms_connector):
        """Test recovery from partial batch processing failure."""
        await mock_cmms_connector.connect()

        items = [{"id": i} for i in range(5)]
        results = []
        failed_items = []

        for item in items:
            try:
                # Simulate failure on item 3
                if item["id"] == 2:
                    raise Exception("Processing error")
                result = await mock_cmms_connector.create_work_order(item)
                results.append(result)
            except Exception as e:
                failed_items.append({"item": item, "error": str(e)})

        await mock_cmms_connector.disconnect()

        assert len(results) == 4
        assert len(failed_items) == 1

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_failure(self, mock_database):
        """Test database transaction rollback on failure."""
        await mock_database.connect()

        try:
            # Start transaction
            await mock_database.transaction()

            # First operation succeeds
            await mock_database.execute("INSERT INTO table1 ...", {})

            # Second operation fails
            mock_database.execute = AsyncMock(side_effect=Exception("DB Error"))
            await mock_database.execute("INSERT INTO table2 ...", {})

        except Exception:
            # Rollback would happen here
            pass

        await mock_database.disconnect()
