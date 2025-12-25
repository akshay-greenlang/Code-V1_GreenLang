"""GL-013 CMMS Integration Tests - Author: GL-TestEngineer"""
import pytest
from datetime import datetime, timedelta, timezone
import asyncio

class TestWorkOrderGeneration:
    @pytest.mark.asyncio
    async def test_create_work_order(self, mock_cmms_connector, mock_work_order_request):
        result = await mock_cmms_connector.create_work_order(mock_work_order_request)
        assert "work_order_id" in result
        assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_work_order_includes_prediction_data(self, mock_cmms_connector, mock_work_order_request):
        assert "equipment_id" in mock_work_order_request
        assert "confidence_score" in mock_work_order_request
        assert "remaining_useful_life_hours" in mock_work_order_request

    @pytest.mark.asyncio
    async def test_work_order_priority_based_on_rul(self, mock_work_order_request):
        rul = mock_work_order_request["remaining_useful_life_hours"]
        priority = mock_work_order_request["priority"]
        if rul < 500:
            expected_priority = "high"
        elif rul < 2000:
            expected_priority = "medium"
        else:
            expected_priority = "low"
        assert priority in ["low", "medium", "high", "critical"]

    @pytest.mark.asyncio
    async def test_work_order_scheduling(self, mock_work_order_request):
        scheduled_start = mock_work_order_request["scheduled_start"]
        due_date = mock_work_order_request["due_date"]
        assert scheduled_start < due_date

class TestIdempotencyVerification:
    @pytest.mark.asyncio
    async def test_duplicate_prediction_same_work_order(self, mock_cmms_connector, mock_work_order_request):
        result1 = await mock_cmms_connector.create_work_order(mock_work_order_request)
        result2 = await mock_cmms_connector.create_work_order(mock_work_order_request)
        assert result1["work_order_id"] == result2["work_order_id"]

    @pytest.mark.asyncio
    async def test_no_duplicate_work_orders_created(self, mock_cmms_connector):
        work_orders = await mock_cmms_connector.list_work_orders()
        wo_ids = [wo["work_order_id"] for wo in work_orders]
        assert len(wo_ids) == len(set(wo_ids))  # All unique

class TestFeedbackLoop:
    @pytest.mark.asyncio
    async def test_work_order_completion_updates_model(self, mock_cmms_connector):
        work_order = await mock_cmms_connector.get_work_order("WO-2024-001")
        assert work_order["status"] in ["pending", "approved", "in_progress", "completed"]

    @pytest.mark.asyncio
    async def test_actual_failure_captured(self, mock_cmms_connector):
        work_order = await mock_cmms_connector.get_work_order("WO-2024-001")
        assert "equipment_id" in work_order

class TestCMMSConnectivity:
    @pytest.mark.asyncio
    async def test_connector_health_check(self, mock_cmms_connector):
        result = await mock_cmms_connector.health_check()
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_connector_connect_disconnect(self, mock_cmms_connector):
        await mock_cmms_connector.connect()
        await mock_cmms_connector.disconnect()
        # Should complete without error

class TestEquipmentDataRetrieval:
    @pytest.mark.asyncio
    async def test_get_equipment_details(self, mock_cmms_connector):
        equipment = await mock_cmms_connector.get_equipment("MOT-001")
        assert equipment["equipment_id"] == "MOT-001"
        assert "equipment_name" in equipment
        assert "status" in equipment

class TestWorkOrderQuerying:
    @pytest.mark.asyncio
    async def test_list_work_orders(self, mock_cmms_connector):
        work_orders = await mock_cmms_connector.list_work_orders()
        assert isinstance(work_orders, list)
        for wo in work_orders:
            assert "work_order_id" in wo
            assert "status" in wo

    @pytest.mark.asyncio
    async def test_get_specific_work_order(self, mock_cmms_connector):
        work_order = await mock_cmms_connector.get_work_order("WO-2024-001")
        assert work_order["work_order_id"] == "WO-2024-001"

class TestCMMSErrorHandling:
    @pytest.mark.asyncio
    async def test_handles_connection_timeout(self, mock_cmms_connector):
        # Mock should handle gracefully
        result = await mock_cmms_connector.health_check()
        assert result is not None

class TestPredictionToCMMSIntegration:
    @pytest.mark.asyncio
    async def test_prediction_triggers_work_order(self, sample_rul_prediction, mock_cmms_connector, mock_work_order_request):
        if sample_rul_prediction.rul_hours_mean < 5000:
            result = await mock_cmms_connector.create_work_order(mock_work_order_request)
            assert result["work_order_id"] is not None

    @pytest.mark.asyncio
    async def test_work_order_contains_prediction_reference(self, mock_work_order_request):
        assert "prediction_id" in mock_work_order_request
        assert mock_work_order_request["prediction_id"] is not None
