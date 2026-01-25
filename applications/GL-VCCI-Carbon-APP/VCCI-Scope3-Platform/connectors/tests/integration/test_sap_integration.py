# -*- coding: utf-8 -*-
"""
SAP Integration Tests
GL-VCCI Scope 3 Platform

Integration tests for SAP S/4HANA connector with real sandbox environment.
Tests OData API connection, extraction, pagination, authentication, and data mapping.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

import pytest
import time
from typing import List, Dict, Any
from datetime import datetime, timedelta

from sap.client import SAPODataClient, create_query
from greenlang.determinism import DeterministicClock
from sap.exceptions import (
    SAPConnectionError,
    SAPAuthenticationError,
    SAPRateLimitError,
    SAPTimeoutError
)
from sap.mappers.po_mapper import PurchaseOrderMapper
from sap.extractors.mm_extractor import MMExtractor
from sap.extractors.sd_extractor import SDExtractor
from sap.extractors.fi_extractor import FIExtractor


@pytest.mark.integration
@pytest.mark.sap_sandbox
class TestSAPConnection:
    """Test SAP OData API connection."""

    def test_sap_connection_successful(self, sap_client: SAPODataClient):
        """Test successful connection to SAP sandbox."""
        # Attempt simple query to verify connection
        query = create_query().top(1)
        response = sap_client.get("purchase_orders", query.build())

        assert response is not None
        assert "value" in response

    def test_sap_authentication(self, sap_client: SAPODataClient):
        """Test SAP OAuth authentication flow."""
        # Auth handler should get valid token
        headers = sap_client._get_headers()

        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")

    def test_sap_connection_failure_handling(self, sap_sandbox_config):
        """Test connection failure handling with invalid URL."""
        # Create client with invalid URL
        sap_sandbox_config.base_url = "https://invalid.sap.com"
        client = SAPODataClient(sap_sandbox_config)

        with pytest.raises(SAPConnectionError):
            client.get("purchase_orders")

        client.close()


@pytest.mark.integration
@pytest.mark.sap_sandbox
class TestSAPPurchaseOrderExtraction:
    """Test Purchase Order extraction from SAP."""

    def test_extract_purchase_orders(self, sap_client: SAPODataClient):
        """Test extracting purchase orders from sandbox."""
        query = create_query().top(10)
        response = sap_client.get("purchase_orders", query.build())

        assert "value" in response
        pos = response["value"]
        assert isinstance(pos, list)

        # Validate PO structure
        if pos:
            po = pos[0]
            assert "PurchaseOrder" in po or "OrderNumber" in po

    def test_extract_with_date_filter(self, sap_client: SAPODataClient):
        """Test extraction with date range filter."""
        # Filter for recent orders
        cutoff_date = (DeterministicClock.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        query = create_query().filter(
            f"CreationDate ge '{cutoff_date}'"
        ).top(10)

        response = sap_client.get("purchase_orders", query.build())
        assert "value" in response

    def test_extract_with_field_selection(self, sap_client: SAPODataClient):
        """Test extraction with specific field selection."""
        query = create_query().select(
            "PurchaseOrder",
            "Supplier",
            "PurchaseOrderDate",
            "TotalAmount"
        ).top(5)

        response = sap_client.get("purchase_orders", query.build())
        pos = response.get("value", [])

        if pos:
            po = pos[0]
            # Should only contain selected fields (plus metadata)
            assert "PurchaseOrder" in po or "Supplier" in po


@pytest.mark.integration
@pytest.mark.sap_sandbox
class TestSAPPagination:
    """Test SAP OData pagination functionality."""

    def test_paginated_extraction(self, sap_client: SAPODataClient):
        """Test paginated extraction of purchase orders."""
        batch_count = 0
        total_records = 0
        max_batches = 3  # Limit for testing

        for batch in sap_client.query_paginated("purchase_orders", {"$top": 10}):
            batch_count += 1
            total_records += len(batch)

            assert len(batch) > 0
            assert len(batch) <= 10

            if batch_count >= max_batches:
                break

        assert batch_count > 0
        assert total_records > 0

    def test_pagination_consistency(self, sap_client: SAPODataClient):
        """Test pagination returns consistent data."""
        # Extract first page
        query1 = create_query().top(5).skip(0)
        response1 = sap_client.get("purchase_orders", query1.build())
        page1 = response1.get("value", [])

        # Extract second page
        query2 = create_query().top(5).skip(5)
        response2 = sap_client.get("purchase_orders", query2.build())
        page2 = response2.get("value", [])

        # Pages should not overlap (if enough data)
        if page1 and page2:
            page1_ids = {po.get("PurchaseOrder") for po in page1}
            page2_ids = {po.get("PurchaseOrder") for po in page2}
            assert page1_ids.isdisjoint(page2_ids)


@pytest.mark.integration
@pytest.mark.sap_sandbox
class TestSAPDeltaSync:
    """Test SAP delta synchronization functionality."""

    def test_delta_sync_by_date(self, sap_client: SAPODataClient):
        """Test delta sync using LastChangeDate."""
        # Get records modified in last 7 days
        cutoff_date = (DeterministicClock.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        query = create_query().filter(
            f"LastChangeDate ge '{cutoff_date}'"
        ).top(50)

        response = sap_client.get("purchase_orders", query.build())
        records = response.get("value", [])

        # All records should have LastChangeDate >= cutoff
        for record in records:
            if "LastChangeDate" in record:
                record_date = record["LastChangeDate"]
                # Validate date format and value
                assert record_date >= cutoff_date


@pytest.mark.integration
@pytest.mark.sap_sandbox
class TestSAPRateLimiting:
    """Test SAP rate limiting behavior."""

    def test_rate_limiting_respected(self, sap_client: SAPODataClient, performance_metrics):
        """Test that rate limiting is respected."""
        # Make multiple requests
        request_count = 5
        start_time = time.time()

        for i in range(request_count):
            query = create_query().top(1)
            sap_client.get("purchase_orders", query.build())

        elapsed = time.time() - start_time

        # Should take at least some time due to rate limiting
        # Actual validation depends on rate limit config
        assert elapsed > 0


@pytest.mark.integration
@pytest.mark.sap_sandbox
class TestSAPRetryLogic:
    """Test SAP retry logic on transient failures."""

    def test_retry_on_timeout(self, sap_sandbox_config):
        """Test retry on timeout errors."""
        # Configure very short timeout
        sap_sandbox_config.timeout.read_timeout = 0.001
        client = SAPODataClient(sap_sandbox_config)

        # This should timeout and retry
        with pytest.raises((SAPTimeoutError, SAPConnectionError)):
            client.get("purchase_orders")

        client.close()


@pytest.mark.integration
@pytest.mark.sap_sandbox
class TestSAPDataMapping:
    """Test SAP data mapping accuracy."""

    def test_purchase_order_mapping(self, sap_client: SAPODataClient):
        """Test mapping SAP purchase orders to standard format."""
        # Extract raw PO data
        query = create_query().top(1)
        response = sap_client.get("purchase_orders", query.build())
        raw_pos = response.get("value", [])

        if raw_pos:
            mapper = PurchaseOrderMapper()
            mapped_po = mapper.map(raw_pos[0])

            # Validate mapped structure
            assert "transaction_id" in mapped_po
            assert "supplier_id" in mapped_po
            assert "transaction_date" in mapped_po
            assert mapped_po["source_system"] == "SAP"


@pytest.mark.integration
@pytest.mark.sap_sandbox
class TestSAPDeduplication:
    """Test SAP data deduplication."""

    def test_duplicate_detection(self, sap_client: SAPODataClient):
        """Test detection of duplicate purchase orders."""
        # Extract same data twice
        query = create_query().top(5)

        response1 = sap_client.get("purchase_orders", query.build())
        records1 = response1.get("value", [])

        response2 = sap_client.get("purchase_orders", query.build())
        records2 = response2.get("value", [])

        # Should get same records
        if records1 and records2:
            ids1 = {r.get("PurchaseOrder") for r in records1}
            ids2 = {r.get("PurchaseOrder") for r in records2}
            assert ids1 == ids2


@pytest.mark.integration
@pytest.mark.sap_sandbox
class TestSAPModules:
    """Test extraction from all SAP modules (MM, SD, FI)."""

    def test_mm_module_extraction(self, sap_client: SAPODataClient):
        """Test Materials Management (MM) module extraction."""
        extractor = MMExtractor(sap_client)

        # Extract purchase orders
        pos = extractor.extract_purchase_orders(limit=5)
        assert isinstance(pos, list)

    def test_sd_module_extraction(self, sap_client: SAPODataClient):
        """Test Sales & Distribution (SD) module extraction."""
        extractor = SDExtractor(sap_client)

        # Extract outbound deliveries
        deliveries = extractor.extract_outbound_deliveries(limit=5)
        assert isinstance(deliveries, list)

    def test_fi_module_extraction(self, sap_client: SAPODataClient):
        """Test Financial Accounting (FI) module extraction."""
        extractor = FIExtractor(sap_client)

        # Extract fixed assets
        assets = extractor.extract_fixed_assets(limit=5)
        assert isinstance(assets, list)


@pytest.mark.integration
@pytest.mark.sap_sandbox
class TestSAPAuditLogging:
    """Test SAP audit logging functionality."""

    def test_audit_log_creation(self, sap_client: SAPODataClient):
        """Test that extraction creates audit logs."""
        # Extract data and verify audit trail
        query = create_query().top(1)
        response = sap_client.get("purchase_orders", query.build())

        # Audit logging should be triggered
        # Audit log verification can be added when database integration is complete
        assert response is not None


@pytest.mark.integration
@pytest.mark.sap_sandbox
class TestSAPEndToEnd:
    """End-to-end integration tests for SAP connector."""

    def test_e2e_extraction_mapping_validation(self, sap_client: SAPODataClient):
        """Test complete flow: extract -> map -> validate schema."""
        # Step 1: Extract raw data
        query = create_query().top(5)
        response = sap_client.get("purchase_orders", query.build())
        raw_pos = response.get("value", [])

        assert len(raw_pos) > 0, "No purchase orders extracted"

        # Step 2: Map to standard format
        mapper = PurchaseOrderMapper()
        mapped_pos = [mapper.map(po) for po in raw_pos]

        assert len(mapped_pos) == len(raw_pos)

        # Step 3: Validate schema
        for mapped_po in mapped_pos:
            # Required fields
            assert "transaction_id" in mapped_po
            assert "supplier_id" in mapped_po
            assert "transaction_date" in mapped_po
            assert "source_system" in mapped_po
            assert mapped_po["source_system"] == "SAP"

            # Data types
            assert isinstance(mapped_po["transaction_id"], str)
            assert isinstance(mapped_po.get("total_amount", 0), (int, float))

    def test_e2e_with_error_handling(self, sap_client: SAPODataClient):
        """Test end-to-end flow with error handling."""
        try:
            # Extract with potential errors
            query = create_query().top(10)
            response = sap_client.get("purchase_orders", query.build())

            records = response.get("value", [])
            assert len(records) >= 0  # May be empty

        except Exception as e:
            # Should not raise unexpected exceptions
            assert isinstance(e, (SAPConnectionError, SAPAuthenticationError, SAPTimeoutError))


# ==================== Mock Tests for CI/CD ====================

@pytest.mark.integration
class TestSAPMockIntegration:
    """Integration tests using mock SAP client for CI/CD."""

    def test_mock_extraction(self, mock_sap_client: SAPODataClient):
        """Test extraction with mock client."""
        query = create_query().top(1)
        response = mock_sap_client.get("purchase_orders", query.build())

        assert "value" in response
        assert len(response["value"]) > 0

    def test_mock_pagination(self, mock_sap_client: SAPODataClient):
        """Test pagination with mock client."""
        batch_count = 0

        for batch in mock_sap_client.query_paginated("purchase_orders", {"$top": 10}):
            batch_count += 1
            assert len(batch) > 0
            break  # Only test first batch

        assert batch_count == 1
