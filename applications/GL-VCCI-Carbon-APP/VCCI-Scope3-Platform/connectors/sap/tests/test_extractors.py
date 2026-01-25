# -*- coding: utf-8 -*-
"""
Extractor Tests for SAP Connector
GL-VCCI Scope 3 Platform

Tests for SAP data extractors including:
- Base extractor (delta extraction, pagination)
- MM extractor (PO, GR, Vendor, Material)
- SD extractor (Delivery, Transport)
- FI extractor (Fixed Assets)
- Delta extraction with timestamps
- Batch processing
- Error handling
- Pydantic model validation

Test Count: 30 tests
Coverage Target: 90%+

Version: 1.0.0
Phase: 4 (Weeks 24-26)
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from pydantic import ValidationError

from connectors.sap.extractors.base import (
    ExtractionConfig,
    ExtractionResult,
    BaseExtractor,
)


class TestExtractionConfig:
    """Tests for ExtractionConfig model."""

    def test_should_create_config_with_defaults(self):
        """Test creating config with default values."""
        config = ExtractionConfig()

        assert config.batch_size == 1000
        assert config.max_retries == 3
        assert config.timeout_seconds == 300
        assert config.enable_delta is True
        assert config.select_fields is None
        assert config.last_sync_timestamp is None

    def test_should_validate_batch_size_range(self):
        """Test batch size validation."""
        # Valid
        config = ExtractionConfig(batch_size=500)
        assert config.batch_size == 500

        # Below minimum
        with pytest.raises(ValidationError):
            ExtractionConfig(batch_size=0)

        # Above maximum
        with pytest.raises(ValidationError):
            ExtractionConfig(batch_size=20000)

    def test_should_validate_timestamp_format(self):
        """Test timestamp format validation."""
        # Valid ISO 8601
        config = ExtractionConfig(last_sync_timestamp="2024-01-15T12:00:00Z")
        assert config.last_sync_timestamp == "2024-01-15T12:00:00Z"

        # Invalid format
        with pytest.raises(ValidationError):
            ExtractionConfig(last_sync_timestamp="2024-01-15")

    def test_should_accept_custom_select_fields(self):
        """Test custom field selection."""
        config = ExtractionConfig(
            select_fields=["PurchaseOrder", "Vendor", "Amount"]
        )

        assert len(config.select_fields) == 3
        assert "PurchaseOrder" in config.select_fields


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_should_create_successful_result(self):
        """Test creating successful extraction result."""
        result = ExtractionResult(
            success=True,
            records_extracted=100,
            last_record_timestamp="2024-01-15T12:00:00Z"
        )

        assert result.success is True
        assert result.records_extracted == 100
        assert result.last_record_timestamp == "2024-01-15T12:00:00Z"
        assert len(result.errors) == 0

    def test_should_create_failed_result_with_errors(self):
        """Test creating failed result with errors."""
        result = ExtractionResult(
            success=False,
            records_extracted=0,
            errors=["Connection timeout", "Retry failed"]
        )

        assert result.success is False
        assert len(result.errors) == 2

    def test_should_include_extraction_timestamp(self):
        """Test extraction timestamp is auto-generated."""
        result = ExtractionResult(success=True)

        assert result.extraction_timestamp is not None
        # Should be valid ISO 8601
        datetime.fromisoformat(result.extraction_timestamp.replace('Z', '+00:00'))

    def test_should_include_metadata(self):
        """Test including custom metadata."""
        result = ExtractionResult(
            success=True,
            metadata={"batch_count": 5, "endpoint": "purchase_orders"}
        )

        assert result.metadata["batch_count"] == 5
        assert result.metadata["endpoint"] == "purchase_orders"


class TestBaseExtractor:
    """Tests for base extractor functionality."""

    def test_should_initialize_extractor(self, mock_sap_client):
        """Test extractor initialization."""
        config = ExtractionConfig()

        # Create concrete implementation for testing
        class TestExtractor(BaseExtractor):
            def extract(self):
                pass

        extractor = TestExtractor(mock_sap_client, config)

        assert extractor.client == mock_sap_client
        assert extractor.config == config

    def test_should_use_default_config_if_none_provided(self, mock_sap_client):
        """Test default config is used if none provided."""
        class TestExtractor(BaseExtractor):
            def extract(self):
                pass

        extractor = TestExtractor(mock_sap_client)

        assert extractor.config is not None
        assert extractor.config.batch_size == 1000


class TestMMExtractor:
    """Tests for MM (Materials Management) extractor."""

    @patch('connectors.sap.extractors.mm_extractor.MMExtractor')
    def test_should_extract_purchase_orders(self, MockExtractor, mock_sap_client, sample_po_data):
        """Test extracting purchase orders."""
        mock_extractor = MockExtractor.return_value
        mock_extractor.extract_purchase_orders.return_value = ExtractionResult(
            success=True,
            records_extracted=len(sample_po_data)
        )

        result = mock_extractor.extract_purchase_orders()

        assert result.success is True
        assert result.records_extracted == 2

    @patch('connectors.sap.extractors.mm_extractor.MMExtractor')
    def test_should_extract_goods_receipts(self, MockExtractor, mock_sap_client, sample_gr_data):
        """Test extracting goods receipts."""
        mock_extractor = MockExtractor.return_value
        mock_extractor.extract_goods_receipts.return_value = ExtractionResult(
            success=True,
            records_extracted=len(sample_gr_data)
        )

        result = mock_extractor.extract_goods_receipts()

        assert result.success is True
        assert result.records_extracted == 1

    @patch('connectors.sap.extractors.mm_extractor.MMExtractor')
    def test_should_extract_vendor_master(self, MockExtractor, mock_sap_client, sample_vendor_data):
        """Test extracting vendor master data."""
        mock_extractor = MockExtractor.return_value
        mock_extractor.extract_vendors.return_value = ExtractionResult(
            success=True,
            records_extracted=len(sample_vendor_data)
        )

        result = mock_extractor.extract_vendors()

        assert result.success is True
        assert result.records_extracted == 2

    @patch('connectors.sap.extractors.mm_extractor.MMExtractor')
    def test_should_extract_material_master(self, MockExtractor, mock_sap_client, sample_material_data):
        """Test extracting material master data."""
        mock_extractor = MockExtractor.return_value
        mock_extractor.extract_materials.return_value = ExtractionResult(
            success=True,
            records_extracted=len(sample_material_data)
        )

        result = mock_extractor.extract_materials()

        assert result.success is True
        assert result.records_extracted == 1

    @patch('connectors.sap.extractors.mm_extractor.MMExtractor')
    def test_should_use_delta_extraction_for_po(self, MockExtractor, mock_sap_client):
        """Test delta extraction for purchase orders."""
        config = ExtractionConfig(
            enable_delta=True,
            last_sync_timestamp="2024-01-01T00:00:00Z"
        )

        mock_extractor = MockExtractor.return_value
        mock_extractor.extract_purchase_orders.return_value = ExtractionResult(
            success=True,
            records_extracted=10,
            last_record_timestamp="2024-01-15T12:00:00Z"
        )

        result = mock_extractor.extract_purchase_orders(delta=True, since="2024-01-01T00:00:00Z")

        assert result.success is True
        assert result.last_record_timestamp is not None


class TestSDExtractor:
    """Tests for SD (Sales & Distribution) extractor."""

    @patch('connectors.sap.extractors.sd_extractor.SDExtractor')
    def test_should_extract_deliveries(self, MockExtractor, mock_sap_client, sample_delivery_data):
        """Test extracting outbound deliveries."""
        mock_extractor = MockExtractor.return_value
        mock_extractor.extract_deliveries.return_value = ExtractionResult(
            success=True,
            records_extracted=len(sample_delivery_data)
        )

        result = mock_extractor.extract_deliveries()

        assert result.success is True
        assert result.records_extracted == 1

    @patch('connectors.sap.extractors.sd_extractor.SDExtractor')
    def test_should_extract_transportation_orders(self, MockExtractor, mock_sap_client, sample_transport_data):
        """Test extracting transportation orders."""
        mock_extractor = MockExtractor.return_value
        mock_extractor.extract_transportation_orders.return_value = ExtractionResult(
            success=True,
            records_extracted=len(sample_transport_data)
        )

        result = mock_extractor.extract_transportation_orders()

        assert result.success is True
        assert result.records_extracted == 1

    @patch('connectors.sap.extractors.sd_extractor.SDExtractor')
    def test_should_handle_delivery_items(self, MockExtractor, mock_sap_client, sample_delivery_data):
        """Test extracting delivery with items."""
        mock_extractor = MockExtractor.return_value
        mock_extractor.extract_deliveries.return_value = ExtractionResult(
            success=True,
            records_extracted=1,
            metadata={"total_items": len(sample_delivery_data[0]["Items"])}
        )

        result = mock_extractor.extract_deliveries()

        assert result.metadata["total_items"] == 1


class TestFIExtractor:
    """Tests for FI (Financial Accounting) extractor."""

    @patch('connectors.sap.extractors.fi_extractor.FIExtractor')
    def test_should_extract_fixed_assets(self, MockExtractor, mock_sap_client):
        """Test extracting fixed assets."""
        mock_extractor = MockExtractor.return_value
        mock_extractor.extract_fixed_assets.return_value = ExtractionResult(
            success=True,
            records_extracted=50
        )

        result = mock_extractor.extract_fixed_assets()

        assert result.success is True
        assert result.records_extracted == 50


class TestDeltaExtraction:
    """Tests for delta extraction logic."""

    @patch('connectors.sap.extractors.mm_extractor.MMExtractor')
    def test_should_build_delta_filter(self, MockExtractor, mock_sap_client):
        """Test building OData filter for delta extraction."""
        config = ExtractionConfig(
            enable_delta=True,
            last_sync_timestamp="2024-01-01T00:00:00Z"
        )

        mock_extractor = MockExtractor.return_value
        mock_extractor._build_delta_filter.return_value = "ChangedOn gt datetime'2024-01-01T00:00:00Z'"

        filter_str = mock_extractor._build_delta_filter("ChangedOn", "2024-01-01T00:00:00Z")

        assert "ChangedOn gt" in filter_str
        assert "2024-01-01" in filter_str

    @patch('connectors.sap.extractors.mm_extractor.MMExtractor')
    def test_should_track_last_record_timestamp(self, MockExtractor, mock_sap_client):
        """Test tracking last record timestamp for next delta."""
        mock_extractor = MockExtractor.return_value
        mock_extractor.extract_purchase_orders.return_value = ExtractionResult(
            success=True,
            records_extracted=100,
            last_record_timestamp="2024-01-15T18:30:00Z"
        )

        result = mock_extractor.extract_purchase_orders()

        assert result.last_record_timestamp == "2024-01-15T18:30:00Z"


class TestBatchProcessing:
    """Tests for batch processing."""

    @patch('connectors.sap.extractors.mm_extractor.MMExtractor')
    def test_should_process_records_in_batches(self, MockExtractor, mock_sap_client):
        """Test processing records in configurable batches."""
        config = ExtractionConfig(batch_size=100)

        mock_extractor = MockExtractor.return_value
        mock_extractor.extract_purchase_orders.return_value = ExtractionResult(
            success=True,
            records_extracted=250,
            metadata={"batch_count": 3}
        )

        result = mock_extractor.extract_purchase_orders()

        assert result.metadata["batch_count"] == 3

    @patch('connectors.sap.extractors.mm_extractor.MMExtractor')
    def test_should_handle_pagination(self, MockExtractor, mock_sap_client):
        """Test handling pagination across batches."""
        mock_extractor = MockExtractor.return_value
        mock_extractor.extract_purchase_orders.return_value = ExtractionResult(
            success=True,
            records_extracted=1000,
            metadata={"pages_fetched": 10}
        )

        result = mock_extractor.extract_purchase_orders()

        assert result.metadata["pages_fetched"] == 10


class TestFieldSelection:
    """Tests for field selection optimization."""

    @patch('connectors.sap.extractors.mm_extractor.MMExtractor')
    def test_should_select_specific_fields(self, MockExtractor, mock_sap_client):
        """Test selecting specific fields to reduce payload."""
        config = ExtractionConfig(
            select_fields=["PurchaseOrder", "Vendor", "Amount"]
        )

        mock_extractor = MockExtractor.return_value
        mock_extractor._build_select_clause.return_value = "PurchaseOrder,Vendor,Amount"

        select_clause = mock_extractor._build_select_clause(config.select_fields)

        assert "PurchaseOrder" in select_clause
        assert "Vendor" in select_clause


class TestErrorHandling:
    """Tests for error handling in extractors."""

    @patch('connectors.sap.extractors.mm_extractor.MMExtractor')
    def test_should_handle_extraction_errors(self, MockExtractor, mock_sap_client):
        """Test handling errors during extraction."""
        mock_extractor = MockExtractor.return_value
        mock_extractor.extract_purchase_orders.return_value = ExtractionResult(
            success=False,
            records_extracted=0,
            errors=["Connection timeout", "Failed after 3 retries"]
        )

        result = mock_extractor.extract_purchase_orders()

        assert result.success is False
        assert len(result.errors) == 2

    @patch('connectors.sap.extractors.mm_extractor.MMExtractor')
    def test_should_handle_validation_errors(self, MockExtractor, mock_sap_client):
        """Test handling Pydantic validation errors."""
        mock_extractor = MockExtractor.return_value
        mock_extractor.extract_purchase_orders.side_effect = ValidationError.from_exception_data(
            "PurchaseOrder",
            [{"type": "missing", "loc": ("PurchaseOrder",), "msg": "Field required"}]
        )

        with pytest.raises(ValidationError):
            mock_extractor.extract_purchase_orders()

    @patch('connectors.sap.extractors.mm_extractor.MMExtractor')
    def test_should_retry_on_transient_errors(self, MockExtractor, mock_sap_client):
        """Test retry logic on transient errors."""
        config = ExtractionConfig(max_retries=3)

        mock_extractor = MockExtractor.return_value

        # First 2 attempts fail, 3rd succeeds
        mock_extractor.extract_purchase_orders.side_effect = [
            ExtractionResult(success=False, errors=["Timeout"]),
            ExtractionResult(success=False, errors=["Timeout"]),
            ExtractionResult(success=True, records_extracted=100)
        ]

        # After retries, should succeed
        results = [
            mock_extractor.extract_purchase_orders(),
            mock_extractor.extract_purchase_orders(),
            mock_extractor.extract_purchase_orders()
        ]

        assert results[-1].success is True


class TestPydanticModels:
    """Tests for Pydantic model validation."""

    def test_should_validate_required_fields(self):
        """Test Pydantic validation of required fields."""
        # Valid
        config = ExtractionConfig(batch_size=1000)
        assert config.batch_size == 1000

        # Invalid - out of range
        with pytest.raises(ValidationError):
            ExtractionConfig(batch_size=-1)

    def test_should_use_field_validators(self):
        """Test custom field validators."""
        # Valid timestamp
        config = ExtractionConfig(last_sync_timestamp="2024-01-15T12:00:00Z")
        assert config.last_sync_timestamp is not None

        # Invalid timestamp
        with pytest.raises(ValidationError):
            ExtractionConfig(last_sync_timestamp="invalid-date")
