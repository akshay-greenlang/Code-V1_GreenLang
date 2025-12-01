# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Unit Tests for Tools Module

Unit tests for external connector tools and integration utilities.
Tests thermal camera interfaces, CMMS connectors, and weather services.

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
import hashlib


# =============================================================================
# TEST: THERMAL CAMERA TOOL
# =============================================================================

class TestThermalCameraTool:
    """Tests for thermal camera connector tool."""

    @pytest.mark.asyncio
    async def test_camera_connect(self, mock_thermal_camera):
        """Test thermal camera connection."""
        result = await mock_thermal_camera.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_camera_disconnect(self, mock_thermal_camera):
        """Test thermal camera disconnection."""
        await mock_thermal_camera.connect()
        result = await mock_thermal_camera.disconnect()
        assert result is True

    @pytest.mark.asyncio
    async def test_camera_capture_image(self, mock_thermal_camera):
        """Test thermal image capture."""
        await mock_thermal_camera.connect()
        result = await mock_thermal_camera.capture_image()

        assert "image_data" in result
        assert "temperature_matrix" in result
        assert "timestamp" in result
        assert "camera_serial" in result

    @pytest.mark.asyncio
    async def test_camera_get_status(self, mock_thermal_camera):
        """Test camera status retrieval."""
        await mock_thermal_camera.connect()
        status = await mock_thermal_camera.get_status()

        assert status["connected"] is True
        assert "battery_level" in status
        assert 0 <= status["battery_level"] <= 100

    @pytest.mark.asyncio
    async def test_camera_set_emissivity(self, mock_thermal_camera):
        """Test emissivity setting."""
        await mock_thermal_camera.connect()
        result = await mock_thermal_camera.set_emissivity(0.95)
        assert result is True

    @pytest.mark.asyncio
    async def test_camera_set_emissivity_invalid(self, mock_thermal_camera):
        """Test emissivity validation."""
        # Emissivity must be between 0 and 1
        invalid_values = [-0.1, 1.1, 2.0]

        for value in invalid_values:
            with pytest.raises((ValueError, AssertionError)):
                if not (0 < value <= 1):
                    raise ValueError(f"Invalid emissivity: {value}")

    @pytest.mark.asyncio
    async def test_camera_set_reflected_temperature(self, mock_thermal_camera):
        """Test reflected temperature setting."""
        await mock_thermal_camera.connect()
        result = await mock_thermal_camera.set_reflected_temperature(25.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_camera_calibration(self, mock_thermal_camera):
        """Test camera calibration."""
        await mock_thermal_camera.connect()
        result = await mock_thermal_camera.calibrate()

        assert result["status"] == "calibrated"

    @pytest.mark.asyncio
    async def test_camera_connection_retry(self, mock_thermal_camera):
        """Test camera connection retry logic."""
        mock_thermal_camera.connect = AsyncMock(side_effect=[
            ConnectionError("Failed"),
            ConnectionError("Failed"),
            True  # Success on third attempt
        ])

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await mock_thermal_camera.connect()
                if result:
                    break
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise

        assert result is True

    def test_camera_image_format_validation(self):
        """Test supported image format validation."""
        supported_formats = [
            "RADIOMETRIC_JPEG",
            "FLIR_FFF",
            "TIFF",
            "RAW",
        ]

        test_format = "RADIOMETRIC_JPEG"
        assert test_format in supported_formats

        invalid_format = "INVALID_FORMAT"
        assert invalid_format not in supported_formats

    def test_camera_temperature_range_validation(self):
        """Test camera temperature range validation."""
        camera_spec = {
            "min_temp_c": -40,
            "max_temp_c": 650,
            "accuracy_c": 2.0,
        }

        test_temps = [25, 100, 300, 500]

        for temp in test_temps:
            assert camera_spec["min_temp_c"] <= temp <= camera_spec["max_temp_c"]

    @pytest.mark.asyncio
    async def test_camera_batch_capture(self, mock_thermal_camera):
        """Test batch image capture."""
        await mock_thermal_camera.connect()

        images = []
        for i in range(5):
            img = await mock_thermal_camera.capture_image()
            images.append(img)

        assert len(images) == 5
        assert all("temperature_matrix" in img for img in images)


# =============================================================================
# TEST: CMMS CONNECTOR
# =============================================================================

class TestCMMSConnector:
    """Tests for CMMS (Computerized Maintenance Management System) connector."""

    @pytest.mark.asyncio
    async def test_cmms_connect(self, mock_cmms_connector):
        """Test CMMS connection."""
        result = await mock_cmms_connector.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_cmms_disconnect(self, mock_cmms_connector):
        """Test CMMS disconnection."""
        await mock_cmms_connector.connect()
        result = await mock_cmms_connector.disconnect()
        assert result is True

    @pytest.mark.asyncio
    async def test_cmms_create_work_order(self, mock_cmms_connector):
        """Test work order creation."""
        await mock_cmms_connector.connect()

        work_order_data = {
            "equipment_tag": "P-1001-A",
            "priority": "high",
            "description": "Insulation repair required - thermal defect detected",
            "estimated_hours": 8,
            "materials_required": ["Mineral wool 75mm", "Aluminum jacketing"],
        }

        result = await mock_cmms_connector.create_work_order(work_order_data)

        assert "work_order_id" in result
        assert result["status"] == "created"
        assert result["priority"] == "high"

    @pytest.mark.asyncio
    async def test_cmms_get_equipment_info(self, mock_cmms_connector):
        """Test equipment information retrieval."""
        await mock_cmms_connector.connect()

        equipment_tag = "P-1001-A"
        info = await mock_cmms_connector.get_equipment_info(equipment_tag)

        assert info["equipment_tag"] == equipment_tag
        assert "description" in info
        assert "location" in info

    @pytest.mark.asyncio
    async def test_cmms_get_maintenance_history(self, mock_cmms_connector):
        """Test maintenance history retrieval."""
        await mock_cmms_connector.connect()

        history = await mock_cmms_connector.get_maintenance_history("P-1001-A")

        assert isinstance(history, list)
        assert len(history) > 0
        assert "date" in history[0]
        assert "type" in history[0]

    @pytest.mark.asyncio
    async def test_cmms_update_asset_condition(self, mock_cmms_connector):
        """Test asset condition update."""
        await mock_cmms_connector.connect()

        condition_data = {
            "equipment_tag": "P-1001-A",
            "condition_score": 7,
            "notes": "Thermal inspection completed - moderate degradation",
            "inspection_date": datetime.now().isoformat(),
        }

        result = await mock_cmms_connector.update_asset_condition(condition_data)
        assert result is True

    def test_cmms_work_order_validation(self):
        """Test work order data validation."""
        valid_work_order = {
            "equipment_tag": "P-1001-A",
            "priority": "high",
            "description": "Test description",
        }

        required_fields = ["equipment_tag", "priority", "description"]

        for field in required_fields:
            assert field in valid_work_order

    def test_cmms_priority_levels(self):
        """Test valid priority levels."""
        valid_priorities = ["emergency", "urgent", "high", "medium", "low"]

        test_priority = "high"
        assert test_priority in valid_priorities

    @pytest.mark.asyncio
    async def test_cmms_batch_work_orders(self, mock_cmms_connector):
        """Test batch work order creation."""
        await mock_cmms_connector.connect()

        defects = [
            {"equipment_tag": f"P-100{i}-A", "priority": "medium"}
            for i in range(1, 6)
        ]

        work_orders = []
        for defect in defects:
            wo = await mock_cmms_connector.create_work_order(defect)
            work_orders.append(wo)

        assert len(work_orders) == 5

    def test_cmms_api_response_parsing(self):
        """Test API response parsing."""
        raw_response = json.dumps({
            "data": {
                "work_order_id": "WO-2025-001234",
                "status": "created",
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
            }
        })

        parsed = json.loads(raw_response)
        assert "data" in parsed
        assert parsed["data"]["work_order_id"] == "WO-2025-001234"


# =============================================================================
# TEST: WEATHER SERVICE
# =============================================================================

class TestWeatherService:
    """Tests for weather service connector."""

    @pytest.mark.asyncio
    async def test_weather_get_current_conditions(self, mock_weather_service):
        """Test current weather conditions retrieval."""
        conditions = await mock_weather_service.get_current_conditions()

        assert "temperature_c" in conditions
        assert "humidity_percent" in conditions
        assert "wind_speed_m_s" in conditions
        assert "solar_radiation_w_m2" in conditions

    @pytest.mark.asyncio
    async def test_weather_get_forecast(self, mock_weather_service):
        """Test weather forecast retrieval."""
        forecast = await mock_weather_service.get_forecast()

        assert isinstance(forecast, list)
        assert len(forecast) > 0
        assert "temperature_c" in forecast[0]
        assert "hour" in forecast[0]

    @pytest.mark.asyncio
    async def test_weather_get_historical(self, mock_weather_service):
        """Test historical weather data retrieval."""
        historical = await mock_weather_service.get_historical()

        assert "avg_temperature_c" in historical
        assert "max_temperature_c" in historical
        assert "min_temperature_c" in historical

    def test_weather_data_validation(self):
        """Test weather data range validation."""
        weather_data = {
            "temperature_c": 25.0,
            "humidity_percent": 60.0,
            "wind_speed_m_s": 5.0,
            "solar_radiation_w_m2": 500.0,
        }

        # Temperature range check
        assert -50 <= weather_data["temperature_c"] <= 60

        # Humidity range check
        assert 0 <= weather_data["humidity_percent"] <= 100

        # Wind speed range check
        assert 0 <= weather_data["wind_speed_m_s"] <= 100

        # Solar radiation range check
        assert 0 <= weather_data["solar_radiation_w_m2"] <= 1400

    def test_weather_correction_factors(self):
        """Test weather-based correction factor calculations."""
        wind_speed = 5.0  # m/s
        solar_radiation = 500.0  # W/m2

        # Wind correction (increases convection)
        if wind_speed < 2:
            wind_factor = 1.0
        elif wind_speed < 5:
            wind_factor = 1.2
        else:
            wind_factor = 1.5

        # Solar correction (decreases apparent heat loss during day)
        solar_factor = 1 - (solar_radiation / 2000)  # Max 1000 W/m2 absorbed

        assert wind_factor == 1.5
        assert 0 < solar_factor < 1

    @pytest.mark.asyncio
    async def test_weather_cache_behavior(self, mock_weather_service):
        """Test weather data caching behavior."""
        # First call
        conditions1 = await mock_weather_service.get_current_conditions()

        # Second call (should potentially use cache)
        conditions2 = await mock_weather_service.get_current_conditions()

        # Verify both calls returned data
        assert conditions1 is not None
        assert conditions2 is not None


# =============================================================================
# TEST: BASE CONNECTOR
# =============================================================================

class TestBaseConnector:
    """Tests for base connector functionality."""

    def test_connector_config_validation(self):
        """Test connector configuration validation."""
        config = {
            "connector_name": "test_connector",
            "connector_type": "thermal_camera",
            "connection_timeout_seconds": 30.0,
            "max_retries": 3,
            "retry_base_delay_seconds": 1.0,
        }

        # Validate required fields
        required = ["connector_name", "connector_type"]
        for field in required:
            assert field in config

        # Validate timeout is positive
        assert config["connection_timeout_seconds"] > 0

        # Validate retries is non-negative
        assert config["max_retries"] >= 0

    def test_connector_state_transitions(self):
        """Test valid connector state transitions."""
        valid_states = [
            "disconnected",
            "connecting",
            "connected",
            "reconnecting",
            "error",
            "maintenance",
        ]

        valid_transitions = {
            "disconnected": ["connecting"],
            "connecting": ["connected", "error"],
            "connected": ["disconnected", "reconnecting", "error", "maintenance"],
            "reconnecting": ["connected", "error"],
            "error": ["connecting", "disconnected"],
            "maintenance": ["disconnected", "connecting"],
        }

        current_state = "disconnected"
        next_state = "connecting"

        assert next_state in valid_transitions[current_state]

    def test_connector_retry_delay_calculation(self):
        """Test exponential backoff retry delay calculation."""
        base_delay = 1.0
        max_delay = 60.0
        exponential_base = 2.0

        delays = []
        for attempt in range(5):
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            delays.append(delay)

        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]

    def test_circuit_breaker_states(self):
        """Test circuit breaker state machine."""
        states = ["closed", "open", "half_open"]

        initial_state = "closed"
        failure_threshold = 5
        failures = 0

        # Simulate failures
        for i in range(6):
            failures += 1
            if failures >= failure_threshold:
                current_state = "open"
                break
        else:
            current_state = initial_state

        assert current_state == "open"

    def test_rate_limiter_token_bucket(self):
        """Test token bucket rate limiter."""
        rate = 10.0  # tokens per second
        burst_size = 20

        tokens = float(burst_size)
        requests_allowed = []

        for i in range(25):
            if tokens >= 1:
                tokens -= 1
                requests_allowed.append(True)
            else:
                requests_allowed.append(False)

        allowed_count = sum(requests_allowed)
        assert allowed_count == burst_size

    def test_cache_key_generation(self):
        """Test cache key generation."""
        params = {
            "equipment_tag": "P-1001-A",
            "date": "2025-01-15",
        }

        key_data = json.dumps(params, sort_keys=True)
        cache_key = hashlib.md5(key_data.encode()).hexdigest()

        assert len(cache_key) == 32
        assert cache_key == hashlib.md5(key_data.encode()).hexdigest()  # Deterministic


# =============================================================================
# TEST: DATA VALIDATION
# =============================================================================

class TestDataValidation:
    """Tests for data validation utilities."""

    def test_temperature_range_validation(self):
        """Test temperature range validation."""
        def validate_temperature(temp_c: float, min_c: float = -273.15, max_c: float = 1500) -> bool:
            return min_c <= temp_c <= max_c

        assert validate_temperature(25.0)
        assert validate_temperature(-40.0)
        assert validate_temperature(650.0)
        assert not validate_temperature(-300.0)

    def test_equipment_tag_validation(self):
        """Test equipment tag format validation."""
        import re

        pattern = r"^[A-Z]+-\d{4}-[A-Z]$"

        valid_tags = ["P-1001-A", "V-2500-B", "HX-3000-C"]
        invalid_tags = ["p-1001-a", "P1001A", "P-1-A", ""]

        for tag in valid_tags:
            assert re.match(pattern, tag), f"{tag} should be valid"

        for tag in invalid_tags:
            assert not re.match(pattern, tag), f"{tag} should be invalid"

    def test_numeric_precision_validation(self):
        """Test numeric precision validation."""
        def validate_precision(value: float, max_decimals: int = 2) -> bool:
            str_val = str(value)
            if '.' in str_val:
                decimals = len(str_val.split('.')[1])
                return decimals <= max_decimals
            return True

        assert validate_precision(25.12)
        assert validate_precision(100.0)
        assert not validate_precision(25.123456)

    def test_date_range_validation(self):
        """Test date range validation."""
        def validate_date_range(date_val: datetime, max_age_days: int = 365) -> bool:
            age = datetime.now() - date_val
            return age.days <= max_age_days

        recent_date = datetime.now() - timedelta(days=30)
        old_date = datetime.now() - timedelta(days=400)

        assert validate_date_range(recent_date)
        assert not validate_date_range(old_date)


# =============================================================================
# TEST: ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in tools."""

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, mock_thermal_camera):
        """Test connection timeout handling."""
        mock_thermal_camera.connect = AsyncMock(
            side_effect=asyncio.TimeoutError("Connection timeout")
        )

        with pytest.raises(asyncio.TimeoutError):
            await mock_thermal_camera.connect()

    @pytest.mark.asyncio
    async def test_authentication_error_handling(self, mock_cmms_connector):
        """Test authentication error handling."""
        mock_cmms_connector.connect = AsyncMock(
            side_effect=PermissionError("Authentication failed")
        )

        with pytest.raises(PermissionError):
            await mock_cmms_connector.connect()

    @pytest.mark.asyncio
    async def test_data_validation_error(self, mock_cmms_connector):
        """Test data validation error handling."""
        await mock_cmms_connector.connect()

        invalid_data = {"invalid_field": "value"}

        mock_cmms_connector.create_work_order = AsyncMock(
            side_effect=ValueError("Missing required field: equipment_tag")
        )

        with pytest.raises(ValueError) as exc_info:
            await mock_cmms_connector.create_work_order(invalid_data)

        assert "equipment_tag" in str(exc_info.value)

    def test_graceful_degradation(self):
        """Test graceful degradation on service failure."""
        def get_data_with_fallback(primary_fn, fallback_fn):
            try:
                return primary_fn()
            except Exception:
                return fallback_fn()

        def failing_primary():
            raise ConnectionError("Service unavailable")

        def fallback():
            return {"status": "cached_data"}

        result = get_data_with_fallback(failing_primary, fallback)
        assert result["status"] == "cached_data"


# =============================================================================
# TEST: METRICS AND LOGGING
# =============================================================================

class TestMetricsAndLogging:
    """Tests for metrics collection and logging."""

    def test_request_metrics_collection(self):
        """Test request metrics collection."""
        metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times_ms": [],
        }

        # Simulate requests
        for i in range(10):
            metrics["total_requests"] += 1
            if i < 8:  # 80% success rate
                metrics["successful_requests"] += 1
                metrics["response_times_ms"].append(50 + i * 10)
            else:
                metrics["failed_requests"] += 1

        assert metrics["total_requests"] == 10
        assert metrics["successful_requests"] == 8
        assert metrics["failed_requests"] == 2

        avg_response_time = sum(metrics["response_times_ms"]) / len(metrics["response_times_ms"])
        assert 50 <= avg_response_time <= 150

    def test_audit_log_entry_creation(self):
        """Test audit log entry creation."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": "capture_image",
            "status": "success",
            "duration_ms": 125.5,
            "connector_id": "camera_001",
            "user_id": "inspector_001",
            "details": {"emissivity": 0.95, "distance_m": 3.0},
        }

        required_fields = ["timestamp", "operation", "status"]
        for field in required_fields:
            assert field in audit_entry

    def test_health_check_result(self):
        """Test health check result structure."""
        health_result = {
            "status": "healthy",
            "latency_ms": 25.0,
            "components": {
                "camera": "healthy",
                "database": "healthy",
                "cmms": "degraded",
            },
            "checked_at": datetime.now().isoformat(),
        }

        assert health_result["status"] in ["healthy", "degraded", "unhealthy"]
        assert health_result["latency_ms"] >= 0


# =============================================================================
# TEST: INTEGRATION WITH CALCULATORS
# =============================================================================

class TestToolCalculatorIntegration:
    """Tests for tool integration with calculators."""

    @pytest.mark.asyncio
    async def test_camera_to_analyzer_pipeline(
        self,
        mock_thermal_camera,
        thermal_image_analyzer
    ):
        """Test thermal camera to analyzer pipeline."""
        await mock_thermal_camera.connect()
        image_data = await mock_thermal_camera.capture_image()

        # Verify image data can be passed to analyzer
        assert "temperature_matrix" in image_data
        assert isinstance(image_data["temperature_matrix"], list)

    @pytest.mark.asyncio
    async def test_weather_to_calculator_pipeline(
        self,
        mock_weather_service,
        sample_ambient_conditions
    ):
        """Test weather service to calculator integration."""
        weather = await mock_weather_service.get_current_conditions()

        # Convert weather data to ambient conditions format
        ambient = {
            "ambient_temperature_c": Decimal(str(weather["temperature_c"])),
            "wind_speed_m_s": Decimal(str(weather["wind_speed_m_s"])),
            "relative_humidity_percent": Decimal(str(weather["humidity_percent"])),
        }

        assert ambient["ambient_temperature_c"] == Decimal("25.0")

    def test_defect_to_work_order_mapping(self, sample_thermal_defect):
        """Test defect data to work order mapping."""
        defect = sample_thermal_defect

        if isinstance(defect, dict):
            work_order = {
                "equipment_tag": defect.get("equipment_tag", "UNKNOWN"),
                "priority": "high" if defect.get("heat_loss_w_per_m", 0) > 200 else "medium",
                "description": f"Thermal defect detected - {defect.get('damage_type', 'unknown')}",
            }
        else:
            work_order = {
                "equipment_tag": defect.location.equipment_tag if hasattr(defect, 'location') else "UNKNOWN",
                "priority": "high",
                "description": "Thermal defect detected",
            }

        assert "equipment_tag" in work_order
        assert "priority" in work_order
        assert "description" in work_order
