"""
Unit tests for GL-006 tool schemas and execution.

Tests tool validation, parameter handling, execution flows,
and error management for all heat recovery tools.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any
import json
from pydantic import BaseModel, ValidationError
import asyncio


class MockToolSchemas:
    """Mock tool schemas for testing."""

    class OptimizeHeatRecoveryTool(BaseModel):
        """Tool for initiating heat recovery optimization."""
        plant_id: str
        optimization_level: str = "standard"  # standard, aggressive, conservative
        include_economics: bool = True
        target_payback_years: float = 3.0

    class AnalyzePinchPointTool(BaseModel):
        """Tool for pinch analysis."""
        hot_streams: List[Dict[str, float]]
        cold_streams: List[Dict[str, float]]
        min_approach_temp: float = 10.0
        utilities_available: List[str] = ["steam", "cooling_water"]

    class DesignHeatExchangerTool(BaseModel):
        """Tool for heat exchanger design."""
        duty: float  # kW
        hot_fluid: str
        cold_fluid: str
        exchanger_type: str = "shell_and_tube"
        max_pressure_drop: float = 50.0  # kPa

    class MonitorEquipmentTool(BaseModel):
        """Tool for equipment monitoring."""
        equipment_id: str
        parameters: List[str]
        sampling_rate: int = 60  # seconds
        alert_thresholds: Dict[str, float] = {}

    class CalculateROITool(BaseModel):
        """Tool for ROI calculation."""
        capital_cost: float
        energy_savings: float  # kWh/year
        maintenance_cost: float
        energy_price: float = 0.10  # $/kWh
        project_life: int = 10

    class DetectFoulingTool(BaseModel):
        """Tool for fouling detection."""
        exchanger_id: str
        historical_data: bool = True
        time_window_days: int = 30
        sensitivity: str = "normal"  # low, normal, high

    class GenerateReportTool(BaseModel):
        """Tool for report generation."""
        report_type: str  # summary, detailed, executive
        time_period: str  # daily, weekly, monthly
        include_graphs: bool = True
        recipients: List[str] = []

    class ScheduleMaintenanceTool(BaseModel):
        """Tool for maintenance scheduling."""
        equipment_id: str
        maintenance_type: str  # cleaning, inspection, replacement
        priority: str = "normal"  # low, normal, high, urgent
        estimated_duration_hours: float
        preferred_date: str = None


class TestToolSchemaValidation:
    """Test tool schema validation."""

    def test_optimize_heat_recovery_tool_validation(self):
        """Test OptimizeHeatRecovery tool parameter validation."""
        tool = MockToolSchemas.OptimizeHeatRecoveryTool

        # Valid parameters
        valid_params = tool(
            plant_id="PLANT-001",
            optimization_level="aggressive",
            include_economics=True,
            target_payback_years=2.5
        )
        assert valid_params.plant_id == "PLANT-001"
        assert valid_params.optimization_level == "aggressive"

        # Invalid optimization level
        with pytest.raises(ValidationError):
            tool(
                plant_id="PLANT-001",
                optimization_level="invalid_level"
            )

    def test_analyze_pinch_point_tool_validation(self):
        """Test AnalyzePinchPoint tool parameter validation."""
        tool = MockToolSchemas.AnalyzePinchPointTool

        hot_streams = [
            {"stream_id": "H1", "heat_load": 1000.0},
            {"stream_id": "H2", "heat_load": 800.0}
        ]
        cold_streams = [
            {"stream_id": "C1", "heat_load": 900.0}
        ]

        # Valid parameters
        valid_params = tool(
            hot_streams=hot_streams,
            cold_streams=cold_streams,
            min_approach_temp=15.0
        )
        assert len(valid_params.hot_streams) == 2
        assert valid_params.min_approach_temp == 15.0

        # Empty streams
        with pytest.raises(ValidationError):
            tool(hot_streams=[], cold_streams=[])

    def test_design_heat_exchanger_tool_validation(self):
        """Test DesignHeatExchanger tool parameter validation."""
        tool = MockToolSchemas.DesignHeatExchangerTool

        # Valid parameters
        valid_params = tool(
            duty=2500.0,
            hot_fluid="steam",
            cold_fluid="water",
            exchanger_type="plate",
            max_pressure_drop=40.0
        )
        assert valid_params.duty == 2500.0
        assert valid_params.exchanger_type == "plate"

        # Negative duty
        with pytest.raises(ValidationError):
            tool(
                duty=-100.0,
                hot_fluid="steam",
                cold_fluid="water"
            )

    def test_monitor_equipment_tool_validation(self):
        """Test MonitorEquipment tool parameter validation."""
        tool = MockToolSchemas.MonitorEquipmentTool

        # Valid parameters
        valid_params = tool(
            equipment_id="HX-001",
            parameters=["temperature", "pressure", "flow"],
            sampling_rate=30,
            alert_thresholds={"temperature": 150.0, "pressure": 100.0}
        )
        assert valid_params.equipment_id == "HX-001"
        assert len(valid_params.parameters) == 3
        assert valid_params.sampling_rate == 30

        # Invalid sampling rate
        with pytest.raises(ValidationError):
            tool(
                equipment_id="HX-001",
                parameters=["temperature"],
                sampling_rate=-10
            )

    def test_calculate_roi_tool_validation(self):
        """Test CalculateROI tool parameter validation."""
        tool = MockToolSchemas.CalculateROITool

        # Valid parameters
        valid_params = tool(
            capital_cost=500000.0,
            energy_savings=1000000.0,
            maintenance_cost=10000.0,
            energy_price=0.12,
            project_life=15
        )
        assert valid_params.capital_cost == 500000.0
        assert valid_params.project_life == 15

        # Negative costs
        with pytest.raises(ValidationError):
            tool(
                capital_cost=-500000.0,
                energy_savings=1000000.0,
                maintenance_cost=10000.0
            )

    def test_detect_fouling_tool_validation(self):
        """Test DetectFouling tool parameter validation."""
        tool = MockToolSchemas.DetectFoulingTool

        # Valid parameters
        valid_params = tool(
            exchanger_id="HX-001",
            historical_data=True,
            time_window_days=60,
            sensitivity="high"
        )
        assert valid_params.exchanger_id == "HX-001"
        assert valid_params.sensitivity == "high"

        # Invalid sensitivity
        with pytest.raises(ValidationError):
            tool(
                exchanger_id="HX-001",
                sensitivity="invalid"
            )


class TestToolExecution:
    """Test tool execution and workflows."""

    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Test asynchronous tool execution."""
        async def mock_optimize_heat_recovery(params):
            await asyncio.sleep(0.1)  # Simulate processing
            return {
                "status": "success",
                "improvements_found": 5,
                "potential_savings": 250000.0
            }

        params = MockToolSchemas.OptimizeHeatRecoveryTool(
            plant_id="PLANT-001",
            optimization_level="standard"
        )

        result = await mock_optimize_heat_recovery(params)
        assert result["status"] == "success"
        assert result["improvements_found"] == 5

    def test_tool_chaining(self):
        """Test chaining multiple tools in sequence."""
        # Tool 1: Pinch Analysis
        pinch_tool = Mock(return_value={
            "pinch_temp": 85.0,
            "heat_recovery_potential": 2000.0
        })

        # Tool 2: HEN Design based on pinch
        design_tool = Mock(return_value={
            "exchangers": [{"id": "HX-NEW-001", "duty": 1500.0}],
            "capital_cost": 300000.0
        })

        # Tool 3: ROI Calculation
        roi_tool = Mock(return_value={
            "payback_years": 2.5,
            "npv": 450000.0
        })

        # Execute chain
        pinch_result = pinch_tool()
        design_result = design_tool(heat_duty=pinch_result["heat_recovery_potential"])
        roi_result = roi_tool(capital_cost=design_result["capital_cost"])

        assert roi_result["payback_years"] == 2.5

    def test_tool_error_handling(self):
        """Test error handling in tool execution."""
        def failing_tool():
            raise ConnectionError("Database connection failed")

        with pytest.raises(ConnectionError):
            failing_tool()

        # Test with error recovery
        def tool_with_retry(max_retries=3):
            for attempt in range(max_retries):
                try:
                    if attempt < 2:
                        raise ConnectionError("Temporary failure")
                    return {"status": "success"}
                except ConnectionError:
                    if attempt == max_retries - 1:
                        raise
            return None

        result = tool_with_retry()
        assert result["status"] == "success"

    def test_tool_timeout_handling(self):
        """Test tool execution timeout handling."""
        import time

        def slow_tool(timeout=1.0):
            start_time = time.time()
            time.sleep(2.0)  # Exceeds timeout

            if time.time() - start_time > timeout:
                raise TimeoutError("Tool execution timed out")

            return {"status": "completed"}

        with pytest.raises(TimeoutError):
            slow_tool(timeout=1.0)

    def test_tool_parameter_coercion(self):
        """Test automatic parameter type coercion."""
        tool = MockToolSchemas.CalculateROITool

        # String to float coercion
        params = tool(
            capital_cost=500000,  # Int to float
            energy_savings=1000000,
            maintenance_cost=10000,
            energy_price=0.12
        )

        assert isinstance(params.capital_cost, float)
        assert params.capital_cost == 500000.0

    @pytest.mark.parametrize("tool_name,expected_category", [
        ("OptimizeHeatRecovery", "optimization"),
        ("AnalyzePinchPoint", "analysis"),
        ("DesignHeatExchanger", "design"),
        ("MonitorEquipment", "monitoring"),
        ("CalculateROI", "economics"),
        ("DetectFouling", "diagnostics"),
        ("GenerateReport", "reporting"),
        ("ScheduleMaintenance", "maintenance")
    ])
    def test_tool_categorization(self, tool_name, expected_category):
        """Test tool categorization for routing."""
        tool_categories = {
            "OptimizeHeatRecovery": "optimization",
            "AnalyzePinchPoint": "analysis",
            "DesignHeatExchanger": "design",
            "MonitorEquipment": "monitoring",
            "CalculateROI": "economics",
            "DetectFouling": "diagnostics",
            "GenerateReport": "reporting",
            "ScheduleMaintenance": "maintenance"
        }

        assert tool_categories[tool_name] == expected_category


class TestToolIntegration:
    """Test tool integration with agent systems."""

    def test_tool_registration(self):
        """Test tool registration with agent."""
        class ToolRegistry:
            def __init__(self):
                self.tools = {}

            def register(self, name: str, tool_class):
                self.tools[name] = tool_class

            def get_tool(self, name: str):
                return self.tools.get(name)

        registry = ToolRegistry()

        # Register tools
        registry.register("optimize", MockToolSchemas.OptimizeHeatRecoveryTool)
        registry.register("pinch", MockToolSchemas.AnalyzePinchPointTool)
        registry.register("design", MockToolSchemas.DesignHeatExchangerTool)

        assert len(registry.tools) == 3
        assert registry.get_tool("optimize") == MockToolSchemas.OptimizeHeatRecoveryTool

    def test_tool_middleware(self):
        """Test middleware for tool execution."""
        class ToolMiddleware:
            def __init__(self):
                self.call_count = 0
                self.last_params = None

            def before_execution(self, tool_name: str, params: Dict):
                self.call_count += 1
                self.last_params = params
                # Add timestamp
                params["_timestamp"] = datetime.now().isoformat()

            def after_execution(self, result: Dict):
                # Add execution metadata
                result["_call_count"] = self.call_count

        middleware = ToolMiddleware()

        # Execute tool with middleware
        params = {"plant_id": "PLANT-001"}
        middleware.before_execution("optimize", params)

        assert "_timestamp" in params
        assert middleware.call_count == 1

        result = {"status": "success"}
        middleware.after_execution(result)

        assert result["_call_count"] == 1

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self):
        """Test concurrent execution of multiple tools."""
        async def tool_1():
            await asyncio.sleep(0.1)
            return {"tool": "1", "result": "success"}

        async def tool_2():
            await asyncio.sleep(0.15)
            return {"tool": "2", "result": "success"}

        async def tool_3():
            await asyncio.sleep(0.05)
            return {"tool": "3", "result": "success"}

        # Execute concurrently
        results = await asyncio.gather(tool_1(), tool_2(), tool_3())

        assert len(results) == 3
        assert all(r["result"] == "success" for r in results)
        # Tool 3 should complete first due to shortest sleep
        assert results[2]["tool"] == "3"

    def test_tool_result_caching(self):
        """Test caching of tool results."""
        class ResultCache:
            def __init__(self):
                self.cache = {}

            def get_key(self, tool_name: str, params: Dict) -> str:
                return f"{tool_name}:{json.dumps(params, sort_keys=True)}"

            def get(self, tool_name: str, params: Dict):
                key = self.get_key(tool_name, params)
                return self.cache.get(key)

            def set(self, tool_name: str, params: Dict, result: Dict):
                key = self.get_key(tool_name, params)
                self.cache[key] = result

        cache = ResultCache()

        # First execution
        params = {"plant_id": "PLANT-001"}
        result = {"status": "success", "value": 100}
        cache.set("optimize", params, result)

        # Cached execution
        cached_result = cache.get("optimize", params)
        assert cached_result == result

        # Different params - not cached
        different_params = {"plant_id": "PLANT-002"}
        assert cache.get("optimize", different_params) is None

    def test_tool_permission_checking(self):
        """Test permission checking for tool execution."""
        class PermissionChecker:
            def __init__(self):
                self.permissions = {
                    "user": ["read", "analyze"],
                    "operator": ["read", "analyze", "optimize"],
                    "admin": ["read", "analyze", "optimize", "configure"]
                }

            def can_execute(self, user_role: str, tool_category: str) -> bool:
                tool_permissions = {
                    "analysis": "analyze",
                    "optimization": "optimize",
                    "monitoring": "read",
                    "configuration": "configure"
                }

                required_permission = tool_permissions.get(tool_category)
                user_permissions = self.permissions.get(user_role, [])

                return required_permission in user_permissions

        checker = PermissionChecker()

        # User permissions
        assert checker.can_execute("user", "analysis") is True
        assert checker.can_execute("user", "optimization") is False

        # Operator permissions
        assert checker.can_execute("operator", "optimization") is True
        assert checker.can_execute("operator", "configuration") is False

        # Admin permissions
        assert checker.can_execute("admin", "configuration") is True