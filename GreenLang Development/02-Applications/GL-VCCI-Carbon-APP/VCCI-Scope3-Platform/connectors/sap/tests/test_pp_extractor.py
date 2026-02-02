# -*- coding: utf-8 -*-
"""
Tests for SAP PP (Production Planning) Extractor

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch

from connectors.sap.extractors.pp_extractor import (
    PPExtractor,
    ProductionOrderData,
    ProductionOrderOperationData,
    PlannedOrderData
)
from connectors.sap.extractors.base import ExtractionConfig


@pytest.fixture
def mock_sap_client():
    """Create a mock SAP client."""
    client = Mock()
    client.get = MagicMock()
    client.get_by_key = MagicMock()
    return client


@pytest.fixture
def pp_extractor(mock_sap_client):
    """Create a PP extractor instance."""
    config = ExtractionConfig(batch_size=100, enable_delta=False)
    return PPExtractor(mock_sap_client, config)


@pytest.fixture
def sample_production_order():
    """Sample production order data."""
    return {
        "ManufacturingOrder": "1000000001",
        "ManufacturingOrderType": "YP01",
        "Material": "MAT-12345",
        "MaterialName": "Finished Product A",
        "ProductionPlant": "1000",
        "ProductionVersion": "0001",
        "MfgOrderPlannedTotalQty": 1000.0,
        "ProductionUnit": "EA",
        "MfgOrderPlannedStartDate": "2024-01-15T08:00:00Z",
        "MfgOrderPlannedEndDate": "2024-01-20T17:00:00Z",
        "MfgOrderActualStartDate": "2024-01-15T08:30:00Z",
        "MfgOrderActualEndDate": "2024-01-20T16:45:00Z",
        "MfgOrderConfirmedYieldQty": 980.0,
        "MfgOrderScrapQty": 20.0,
        "OrderIsCreated": True,
        "OrderIsReleased": True,
        "OrderIsConfirmed": True,
        "OrderIsClosed": False,
        "ProductionSupervisor": "USER001",
        "CreatedByUser": "ADMIN",
        "CreationDate": "2024-01-10",
        "ChangedOn": "2024-01-20T16:45:00Z"
    }


@pytest.fixture
def sample_production_operations():
    """Sample production order operations."""
    return [
        {
            "ManufacturingOrder": "1000000001",
            "ManufacturingOrderOperation": "0010",
            "WorkCenter": "WC-PRESS-01",
            "WorkCenterTypeCode": "MACH",
            "OperationText": "Pressing",
            "Plant": "1000",
            "OpPlannedTotalQuantity": 1000.0,
            "OpTotalConfirmedYieldQty": 980.0,
            "OpPlannedMachineTime": 40.0,
            "OpActualMachineTime": 38.5,
            "OpPlannedLaborTime": 20.0,
            "OpActualLaborTime": 19.0,
        },
        {
            "ManufacturingOrder": "1000000001",
            "ManufacturingOrderOperation": "0020",
            "WorkCenter": "WC-ASSEM-01",
            "WorkCenterTypeCode": "ASBL",
            "OperationText": "Assembly",
            "Plant": "1000",
            "OpPlannedTotalQuantity": 1000.0,
            "OpTotalConfirmedYieldQty": 980.0,
            "OpPlannedMachineTime": 20.0,
            "OpActualMachineTime": 21.5,
            "OpPlannedLaborTime": 40.0,
            "OpActualLaborTime": 42.0,
        }
    ]


class TestPPExtractor:
    """Tests for PP Extractor."""

    def test_initialization(self, mock_sap_client):
        """Test extractor initialization."""
        extractor = PPExtractor(mock_sap_client)
        assert extractor.service_name == "PP"
        assert extractor.client == mock_sap_client
        assert extractor.get_entity_set_name() == "A_ProductionOrder"
        assert extractor.get_changed_on_field() == "ChangedOn"

    def test_extract_production_orders(self, pp_extractor, mock_sap_client, sample_production_order):
        """Test extracting production orders."""
        mock_sap_client.get.return_value = {
            "value": [sample_production_order]
        }

        results = list(pp_extractor.extract_production_orders(
            plant="1000",
            start_date_from="2024-01-01",
            start_date_to="2024-12-31"
        ))

        assert len(results) == 1
        assert results[0]["ManufacturingOrder"] == "1000000001"
        assert results[0]["ProductionPlant"] == "1000"

        # Verify filter was applied
        call_args = mock_sap_client.get.call_args
        params = call_args[1]["params"]
        assert "ProductionPlant eq '1000'" in params["$filter"]

    def test_extract_production_orders_with_status_filter(self, pp_extractor, mock_sap_client, sample_production_order):
        """Test extracting production orders with status filter."""
        mock_sap_client.get.return_value = {
            "value": [sample_production_order]
        }

        results = list(pp_extractor.extract_production_orders(
            plant="1000",
            status_filter="RELEASED"
        ))

        assert len(results) == 1

        # Verify status filter was applied
        call_args = mock_sap_client.get.call_args
        params = call_args[1]["params"]
        assert "OrderIsReleased eq true" in params["$filter"]

    def test_extract_production_order_operations(self, pp_extractor, mock_sap_client, sample_production_operations):
        """Test extracting production order operations."""
        mock_sap_client.get.return_value = {
            "value": sample_production_operations
        }

        results = list(pp_extractor.extract_production_order_operations(
            manufacturing_order="1000000001"
        ))

        assert len(results) == 2
        assert results[0]["ManufacturingOrderOperation"] == "0010"
        assert results[0]["WorkCenter"] == "WC-PRESS-01"
        assert results[1]["ManufacturingOrderOperation"] == "0020"

        # Verify entity set changed
        assert pp_extractor.get_entity_set_name() == "A_ProductionOrderOperation"

    def test_extract_production_order_with_details(
        self,
        pp_extractor,
        mock_sap_client,
        sample_production_order,
        sample_production_operations
    ):
        """Test extracting complete production order with details."""
        # Mock header
        mock_sap_client.get_by_key.return_value = sample_production_order

        # Mock components and operations
        def get_side_effect(entity_set, params):
            if "Component" in entity_set:
                return {"value": []}
            elif "Operation" in entity_set:
                return {"value": sample_production_operations}
            return {"value": []}

        mock_sap_client.get.side_effect = get_side_effect

        result = pp_extractor.extract_production_order_with_details("1000000001")

        assert result["header"] is not None
        assert result["header"]["ManufacturingOrder"] == "1000000001"
        assert len(result["operations"]) == 2
        assert isinstance(result["components"], list)

    def test_extract_manufacturing_emissions_data(
        self,
        pp_extractor,
        mock_sap_client,
        sample_production_order
    ):
        """Test extracting production orders optimized for emissions."""
        mock_sap_client.get.return_value = {
            "value": [sample_production_order]
        }

        results = list(pp_extractor.extract_manufacturing_emissions_data(
            plant="1000",
            date_from="2024-01-01",
            date_to="2024-12-31"
        ))

        assert len(results) == 1
        assert results[0]["ManufacturingOrder"] == "1000000001"

        # Verify specific fields were selected
        assert pp_extractor.config.select_fields is not None
        assert "ManufacturingOrder" in pp_extractor.config.select_fields
        assert "MfgOrderConfirmedYieldQty" in pp_extractor.config.select_fields

    def test_delta_extraction(self, pp_extractor, mock_sap_client, sample_production_order):
        """Test delta extraction."""
        pp_extractor.config.enable_delta = True
        pp_extractor.config.last_sync_timestamp = "2024-01-19T00:00:00Z"

        mock_sap_client.get.return_value = {
            "value": [sample_production_order]
        }

        results = list(pp_extractor.get_delta(
            last_sync_timestamp="2024-01-19T00:00:00Z"
        ))

        assert len(results) == 1

        # Verify delta filter was applied
        call_args = mock_sap_client.get.call_args
        params = call_args[1]["params"]
        assert "$filter" in params
        assert "ChangedOn gt datetime'2024-01-19T00:00:00Z'" in params["$filter"]

    def test_pagination(self, pp_extractor, mock_sap_client, sample_production_order):
        """Test pagination handling."""
        # First batch returns records, second returns empty
        mock_sap_client.get.side_effect = [
            {"value": [sample_production_order] * 100},  # Full batch
            {"value": []}  # End of data
        ]

        results = list(pp_extractor.extract_production_orders(plant="1000"))

        assert len(results) == 100
        assert mock_sap_client.get.call_count == 2

        # Verify pagination parameters
        first_call = mock_sap_client.get.call_args_list[0]
        assert first_call[1]["params"]["$skip"] == 0
        assert first_call[1]["params"]["$top"] == 100

    def test_error_handling(self, pp_extractor, mock_sap_client):
        """Test error handling."""
        mock_sap_client.get.side_effect = Exception("Connection error")

        with pytest.raises(Exception) as exc_info:
            list(pp_extractor.extract_production_orders())

        assert "Connection error" in str(exc_info.value)

    def test_extract_planned_orders(self, pp_extractor, mock_sap_client):
        """Test extracting planned orders."""
        sample_planned_order = {
            "PlannedOrder": "2000000001",
            "Material": "MAT-12345",
            "ProductionPlant": "1000",
            "TotalQuantity": 500.0,
            "ProductionUnit": "EA",
            "PlannedOrderPlannedStartDate": "2024-02-01",
            "IsConvertedToProductionOrder": False
        }

        mock_sap_client.get.return_value = {
            "value": [sample_planned_order]
        }

        results = list(pp_extractor.extract_planned_orders(
            plant="1000",
            converted_only=False
        ))

        assert len(results) == 1
        assert results[0]["PlannedOrder"] == "2000000001"
        assert pp_extractor.get_entity_set_name() == "A_PlannedOrder"


class TestProductionOrderData:
    """Tests for Production Order data models."""

    def test_production_order_data_model(self, sample_production_order):
        """Test ProductionOrderData model."""
        data = ProductionOrderData(**sample_production_order)

        assert data.ManufacturingOrder == "1000000001"
        assert data.Material == "MAT-12345"
        assert data.MfgOrderPlannedTotalQty == 1000.0
        assert data.ProductionUnit == "EA"

    def test_production_order_operation_data_model(self, sample_production_operations):
        """Test ProductionOrderOperationData model."""
        data = ProductionOrderOperationData(**sample_production_operations[0])

        assert data.ManufacturingOrder == "1000000001"
        assert data.ManufacturingOrderOperation == "0010"
        assert data.WorkCenter == "WC-PRESS-01"
        assert data.OpActualMachineTime == 38.5


@pytest.mark.integration
class TestPPExtractorIntegration:
    """Integration tests for PP extractor (requires SAP sandbox)."""

    @pytest.mark.skip(reason="Requires SAP sandbox environment")
    def test_real_extraction(self):
        """Test real extraction from SAP sandbox."""
        # This would require actual SAP credentials and sandbox
        pass
