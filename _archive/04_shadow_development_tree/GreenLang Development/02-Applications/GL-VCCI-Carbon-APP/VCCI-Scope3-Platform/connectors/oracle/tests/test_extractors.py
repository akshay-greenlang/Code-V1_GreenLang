# -*- coding: utf-8 -*-
"""
Oracle Connector Extractors Tests
GL-VCCI Scope 3 Platform

Tests for data extractors including base extractor, procurement,
SCM, and financials extractors with delta extraction logic.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Test Count: 25
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone
from typing import Dict, Any, List

from connectors.oracle.extractors.base import (
    BaseExtractor,
    ExtractionConfig,
    ExtractionResult
)
from connectors.oracle.extractors.procurement_extractor import ProcurementExtractor
from connectors.oracle.extractors.scm_extractor import SCMExtractor
from connectors.oracle.extractors.financials_extractor import FinancialsExtractor


class TestExtractionConfig:
    """Tests for ExtractionConfig."""

    def test_config_defaults(self):
        """Test extraction config default values."""
        config = ExtractionConfig()

        assert config.batch_size == 500
        assert config.max_retries == 3
        assert config.timeout_seconds == 300
        assert config.enable_delta is True
        assert config.select_fields is None

    def test_config_custom_values(self):
        """Test extraction config with custom values."""
        config = ExtractionConfig(
            batch_size=1000,
            max_retries=5,
            enable_delta=False,
            select_fields=["POHeaderId", "OrderNumber"]
        )

        assert config.batch_size == 1000
        assert config.max_retries == 5
        assert config.enable_delta is False
        assert len(config.select_fields) == 2

    def test_config_validation_batch_size(self):
        """Test batch size validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ExtractionConfig(batch_size=0)  # Too small

        with pytest.raises(ValidationError):
            ExtractionConfig(batch_size=10000)  # Too large

    def test_config_timestamp_validation(self):
        """Test timestamp format validation."""
        from pydantic import ValidationError

        # Valid ISO 8601 timestamp
        config = ExtractionConfig(last_sync_timestamp="2024-01-01T00:00:00Z")
        assert config.last_sync_timestamp == "2024-01-01T00:00:00Z"

        # Invalid timestamp
        with pytest.raises(ValidationError):
            ExtractionConfig(last_sync_timestamp="invalid-date")


class TestBaseExtractor:
    """Tests for BaseExtractor."""

    def create_mock_extractor(self, client=None):
        """Create a mock extractor for testing."""
        class MockExtractor(BaseExtractor):
            def get_resource_path(self):
                return "/purchaseOrders"

            def get_changed_on_field(self):
                return "LastUpdateDate"

        return MockExtractor(client or Mock(), ExtractionConfig())

    def test_extractor_initialization(self):
        """Test base extractor initialization."""
        client = Mock()
        config = ExtractionConfig(batch_size=1000)
        extractor = self.create_mock_extractor(client)

        assert extractor.client == client
        assert extractor.config is not None

    def test_build_query_params_basic(self):
        """Test building basic query parameters."""
        extractor = self.create_mock_extractor()

        params = extractor._build_query_params()

        assert "limit" in params
        assert params["limit"] == 500  # Default batch size

    def test_build_query_params_with_delta(self):
        """Test building query with delta extraction."""
        config = ExtractionConfig(
            enable_delta=True,
            last_sync_timestamp="2024-01-01T00:00:00Z"
        )
        extractor = self.create_mock_extractor()
        extractor.config = config

        params = extractor._build_query_params()

        assert "q" in params
        assert "LastUpdateDate >= '2024-01-01T00:00:00Z'" in params["q"]

    def test_build_query_params_with_fields(self):
        """Test building query with field selection."""
        config = ExtractionConfig(
            select_fields=["POHeaderId", "OrderNumber", "SupplierId"]
        )
        extractor = self.create_mock_extractor()
        extractor.config = config

        params = extractor._build_query_params()

        assert "fields" in params
        assert "POHeaderId" in params["fields"]
        assert "OrderNumber" in params["fields"]

    def test_build_query_params_with_filters(self):
        """Test building query with additional filters."""
        extractor = self.create_mock_extractor()

        params = extractor._build_query_params(
            additional_filters=["Status='APPROVED'", "BU='US01'"]
        )

        assert "q" in params
        assert "Status='APPROVED'" in params["q"]
        assert "BU='US01'" in params["q"]

    def test_extract_next_link(self):
        """Test extracting next page link."""
        extractor = self.create_mock_extractor()

        response = {
            "hasMore": True,
            "links": [
                {"rel": "self", "href": "https://test.com/api?offset=0"},
                {"rel": "next", "href": "https://test.com/api?offset=100"}
            ]
        }

        next_link = extractor._extract_next_link(response)
        assert next_link == "https://test.com/api?offset=100"

    def test_extract_next_link_no_more(self):
        """Test extracting next link when no more pages."""
        extractor = self.create_mock_extractor()

        response = {
            "hasMore": False,
            "links": [{"rel": "self", "href": "https://test.com/api"}]
        }

        next_link = extractor._extract_next_link(response)
        assert next_link is None

    def test_get_all_records(self):
        """Test extracting all records with pagination."""
        mock_client = Mock()
        mock_client.get.side_effect = [
            {
                "items": [{"POHeaderId": 1}, {"POHeaderId": 2}],
                "hasMore": True,
                "links": [{"rel": "next", "href": "https://test.com/api?offset=2"}]
            },
            {
                "items": [{"POHeaderId": 3}],
                "hasMore": False,
                "links": []
            }
        ]

        extractor = self.create_mock_extractor(mock_client)
        records = list(extractor.get_all())

        assert len(records) == 3
        assert records[0]["POHeaderId"] == 1
        assert records[2]["POHeaderId"] == 3

    def test_get_delta_records(self):
        """Test delta extraction."""
        mock_client = Mock()
        mock_client.get.return_value = {
            "items": [
                {"POHeaderId": 1, "LastUpdateDate": "2024-01-15T00:00:00Z"},
                {"POHeaderId": 2, "LastUpdateDate": "2024-01-16T00:00:00Z"}
            ],
            "hasMore": False,
            "links": []
        }

        extractor = self.create_mock_extractor(mock_client)
        records = list(extractor.get_delta("2024-01-01T00:00:00Z"))

        assert len(records) == 2
        # Verify delta filter was applied
        call_args = mock_client.get.call_args
        assert "q" in call_args[1]["params"]

    def test_get_by_id(self):
        """Test fetching single record by ID."""
        mock_client = Mock()
        mock_client.get.return_value = {"POHeaderId": 123, "OrderNumber": "PO-001"}

        extractor = self.create_mock_extractor(mock_client)
        record = extractor.get_by_id("123")

        assert record["POHeaderId"] == 123
        mock_client.get.assert_called_once()

    def test_get_by_id_not_found(self):
        """Test fetching non-existent record."""
        mock_client = Mock()
        mock_client.get.side_effect = Exception("Not found")

        extractor = self.create_mock_extractor(mock_client)
        record = extractor.get_by_id("999")

        assert record is None

    def test_extract_success(self):
        """Test successful extraction with result."""
        mock_client = Mock()
        mock_client.get.return_value = {
            "items": [
                {"POHeaderId": 1, "LastUpdateDate": "2024-01-15T00:00:00Z"},
                {"POHeaderId": 2, "LastUpdateDate": "2024-01-16T00:00:00Z"}
            ],
            "hasMore": False,
            "links": []
        }

        extractor = self.create_mock_extractor(mock_client)
        result = extractor.extract()

        assert result.success is True
        assert result.records_extracted == 2
        assert result.last_record_timestamp == "2024-01-16T00:00:00Z"
        assert len(result.errors) == 0

    def test_extract_failure(self):
        """Test extraction failure handling."""
        mock_client = Mock()
        mock_client.get.side_effect = Exception("Connection failed")

        extractor = self.create_mock_extractor(mock_client)
        result = extractor.extract()

        assert result.success is False
        assert result.records_extracted == 0
        assert len(result.errors) > 0


class TestProcurementExtractor:
    """Tests for ProcurementExtractor."""

    def test_procurement_extractor_initialization(self):
        """Test procurement extractor initialization."""
        from connectors.oracle.extractors.procurement_extractor import ProcurementExtractor

        client = Mock()
        extractor = ProcurementExtractor(client)

        assert extractor.client == client

    def test_get_purchase_orders(self):
        """Test extracting purchase orders."""
        from connectors.oracle.extractors.procurement_extractor import ProcurementExtractor

        mock_client = Mock()
        mock_client.get.return_value = {
            "items": [
                {"POHeaderId": 1, "OrderNumber": "PO-001"},
                {"POHeaderId": 2, "OrderNumber": "PO-002"}
            ],
            "hasMore": False,
            "links": []
        }

        extractor = ProcurementExtractor(mock_client)
        records = list(extractor.get_purchase_orders())

        assert len(records) == 2
        assert records[0]["OrderNumber"] == "PO-001"

    def test_get_requisitions(self):
        """Test extracting purchase requisitions."""
        from connectors.oracle.extractors.procurement_extractor import ProcurementExtractor

        mock_client = Mock()
        mock_client.get.return_value = {
            "items": [{"RequisitionHeaderId": 1, "RequisitionNumber": "REQ-001"}],
            "hasMore": False,
            "links": []
        }

        extractor = ProcurementExtractor(mock_client)
        records = list(extractor.get_requisitions())

        assert len(records) == 1
        assert records[0]["RequisitionNumber"] == "REQ-001"

    def test_get_suppliers(self):
        """Test extracting suppliers."""
        from connectors.oracle.extractors.procurement_extractor import ProcurementExtractor

        mock_client = Mock()
        mock_client.get.return_value = {
            "items": [{"SupplierId": 1001, "SupplierName": "Acme Corp"}],
            "hasMore": False,
            "links": []
        }

        extractor = ProcurementExtractor(mock_client)
        records = list(extractor.get_suppliers())

        assert len(records) == 1
        assert records[0]["SupplierName"] == "Acme Corp"


class TestSCMExtractor:
    """Tests for SCMExtractor."""

    def test_scm_extractor_initialization(self):
        """Test SCM extractor initialization."""
        from connectors.oracle.extractors.scm_extractor import SCMExtractor

        client = Mock()
        extractor = SCMExtractor(client)

        assert extractor.client == client

    def test_get_shipments(self):
        """Test extracting shipments."""
        from connectors.oracle.extractors.scm_extractor import SCMExtractor

        mock_client = Mock()
        mock_client.get.return_value = {
            "items": [{"ShipmentId": 1, "ShipmentNumber": "SHIP-001"}],
            "hasMore": False,
            "links": []
        }

        extractor = SCMExtractor(mock_client)
        records = list(extractor.get_shipments())

        assert len(records) == 1
        assert records[0]["ShipmentNumber"] == "SHIP-001"

    def test_get_transport_orders(self):
        """Test extracting transportation orders."""
        from connectors.oracle.extractors.scm_extractor import SCMExtractor

        mock_client = Mock()
        mock_client.get.return_value = {
            "items": [{"TransportOrderId": 1, "OrderNumber": "TRANS-001"}],
            "hasMore": False,
            "links": []
        }

        extractor = SCMExtractor(mock_client)
        records = list(extractor.get_transport_orders())

        assert len(records) == 1
        assert records[0]["OrderNumber"] == "TRANS-001"


class TestFinancialsExtractor:
    """Tests for FinancialsExtractor."""

    def test_financials_extractor_initialization(self):
        """Test financials extractor initialization."""
        from connectors.oracle.extractors.financials_extractor import FinancialsExtractor

        client = Mock()
        extractor = FinancialsExtractor(client)

        assert extractor.client == client

    def test_get_fixed_assets(self):
        """Test extracting fixed assets."""
        from connectors.oracle.extractors.financials_extractor import FinancialsExtractor

        mock_client = Mock()
        mock_client.get.return_value = {
            "items": [{"AssetId": 1, "AssetNumber": "ASSET-001"}],
            "hasMore": False,
            "links": []
        }

        extractor = FinancialsExtractor(mock_client)
        records = list(extractor.get_fixed_assets())

        assert len(records) == 1
        assert records[0]["AssetNumber"] == "ASSET-001"
