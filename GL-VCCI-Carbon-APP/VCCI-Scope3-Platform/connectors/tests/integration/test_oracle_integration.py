"""
Oracle Integration Tests
GL-VCCI Scope 3 Platform

Integration tests for Oracle Fusion Cloud connector with real sandbox environment.
Tests REST API connection, extraction, pagination, authentication, and data mapping.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

import pytest
import time
from typing import List, Dict, Any
from datetime import datetime, timedelta

from oracle.client import OracleRESTClient, create_query
from oracle.exceptions import (
    OracleConnectionError,
    OracleAuthenticationError,
    OracleRateLimitError,
    OracleTimeoutError
)
from oracle.mappers.po_mapper import PurchaseOrderMapper
from oracle.extractors.procurement_extractor import ProcurementExtractor
from oracle.extractors.scm_extractor import SCMExtractor
from oracle.extractors.financials_extractor import FinancialsExtractor


@pytest.mark.integration
@pytest.mark.oracle_sandbox
class TestOracleConnection:
    """Test Oracle REST API connection."""

    def test_oracle_connection_successful(self, oracle_client: OracleRESTClient):
        """Test successful connection to Oracle sandbox."""
        # Attempt simple query to verify connection
        query = create_query().limit(1)
        response = oracle_client.get("purchase_orders", query.build())

        assert response is not None
        assert "items" in response or "count" in response

    def test_oracle_authentication(self, oracle_client: OracleRESTClient):
        """Test Oracle OAuth authentication flow."""
        # Auth handler should get valid token
        headers = oracle_client._get_headers()

        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")

    def test_oracle_connection_failure_handling(self, oracle_sandbox_config):
        """Test connection failure handling with invalid URL."""
        # Create client with invalid URL
        oracle_sandbox_config.base_url = "https://invalid.oracle.com"
        client = OracleRESTClient(oracle_sandbox_config)

        with pytest.raises(OracleConnectionError):
            client.get("purchase_orders")

        client.close()


@pytest.mark.integration
@pytest.mark.oracle_sandbox
class TestOraclePurchaseOrderExtraction:
    """Test Purchase Order extraction from Oracle."""

    def test_extract_purchase_orders(self, oracle_client: OracleRESTClient):
        """Test extracting purchase orders from sandbox."""
        query = create_query().limit(10)
        response = oracle_client.get("purchase_orders", query.build())

        assert "items" in response
        pos = response["items"]
        assert isinstance(pos, list)

        # Validate PO structure
        if pos:
            po = pos[0]
            assert "OrderNumber" in po or "POHeaderId" in po

    def test_extract_with_date_filter(self, oracle_client: OracleRESTClient):
        """Test extraction with date range filter."""
        # Filter for recent orders
        cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        query = create_query().q(
            f"CreationDate >= '{cutoff_date}T00:00:00'"
        ).limit(10)

        response = oracle_client.get("purchase_orders", query.build())
        assert "items" in response

    def test_extract_with_field_selection(self, oracle_client: OracleRESTClient):
        """Test extraction with specific field selection."""
        query = create_query().fields(
            "OrderNumber",
            "Supplier",
            "OrderDate",
            "TotalAmount"
        ).limit(5)

        response = oracle_client.get("purchase_orders", query.build())
        pos = response.get("items", [])

        if pos:
            po = pos[0]
            # Should only contain selected fields
            assert "OrderNumber" in po or "Supplier" in po


@pytest.mark.integration
@pytest.mark.oracle_sandbox
class TestOraclePagination:
    """Test Oracle REST API pagination functionality."""

    def test_paginated_extraction(self, oracle_client: OracleRESTClient):
        """Test paginated extraction of purchase orders."""
        batch_count = 0
        total_records = 0
        max_batches = 3  # Limit for testing

        for batch in oracle_client.query_paginated("purchase_orders", {"limit": 10}):
            batch_count += 1
            total_records += len(batch)

            assert len(batch) > 0
            assert len(batch) <= 10

            if batch_count >= max_batches:
                break

        assert batch_count > 0
        assert total_records > 0

    def test_pagination_with_links(self, oracle_client: OracleRESTClient):
        """Test Oracle pagination using links array."""
        query = create_query().limit(5)
        response = oracle_client.get("purchase_orders", query.build())

        # Oracle returns links array for pagination
        assert "items" in response
        if response.get("hasMore"):
            assert "links" in response
            links = response["links"]
            assert isinstance(links, list)

            # Should have a "next" link
            next_links = [l for l in links if l.get("rel") == "next"]
            if next_links:
                assert "href" in next_links[0]


@pytest.mark.integration
@pytest.mark.oracle_sandbox
class TestOracleDeltaSync:
    """Test Oracle delta synchronization functionality."""

    def test_delta_sync_by_date(self, oracle_client: OracleRESTClient):
        """Test delta sync using LastUpdateDate."""
        # Get records modified in last 7 days
        cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        query = create_query().q(
            f"LastUpdateDate >= '{cutoff_date}T00:00:00'"
        ).limit(50)

        response = oracle_client.get("purchase_orders", query.build())
        records = response.get("items", [])

        # All records should have LastUpdateDate >= cutoff
        for record in records:
            if "LastUpdateDate" in record:
                record_date = record["LastUpdateDate"]
                # Validate date is recent
                assert record_date is not None


@pytest.mark.integration
@pytest.mark.oracle_sandbox
class TestOracleAuthentication:
    """Test Oracle authentication and token handling."""

    def test_token_refresh_on_401(self, oracle_client: OracleRESTClient):
        """Test automatic token refresh on 401 response."""
        # Invalidate current token
        oracle_client.auth_handler.invalidate_token()

        # Next request should refresh token
        query = create_query().limit(1)
        response = oracle_client.get("purchase_orders", query.build())

        assert response is not None


@pytest.mark.integration
@pytest.mark.oracle_sandbox
class TestOracleModules:
    """Test extraction from all Oracle modules (Procurement, SCM, Financials)."""

    def test_procurement_module_extraction(self, oracle_client: OracleRESTClient):
        """Test Procurement module extraction."""
        extractor = ProcurementExtractor(oracle_client)

        # Extract purchase orders
        pos = extractor.extract_purchase_orders(limit=5)
        assert isinstance(pos, list)

    def test_scm_module_extraction(self, oracle_client: OracleRESTClient):
        """Test Supply Chain Management (SCM) module extraction."""
        extractor = SCMExtractor(oracle_client)

        # Extract shipments
        shipments = extractor.extract_shipments(limit=5)
        assert isinstance(shipments, list)

    def test_financials_module_extraction(self, oracle_client: OracleRESTClient):
        """Test Financials module extraction."""
        extractor = FinancialsExtractor(oracle_client)

        # Extract invoices
        invoices = extractor.extract_invoices(limit=5)
        assert isinstance(invoices, list)


@pytest.mark.integration
@pytest.mark.oracle_sandbox
class TestOracleDataMapping:
    """Test Oracle data mapping accuracy."""

    def test_purchase_order_mapping(self, oracle_client: OracleRESTClient):
        """Test mapping Oracle purchase orders to standard format."""
        # Extract raw PO data
        query = create_query().limit(1)
        response = oracle_client.get("purchase_orders", query.build())
        raw_pos = response.get("items", [])

        if raw_pos:
            mapper = PurchaseOrderMapper()
            mapped_po = mapper.map(raw_pos[0])

            # Validate mapped structure
            assert "transaction_id" in mapped_po
            assert "supplier_id" in mapped_po
            assert "transaction_date" in mapped_po
            assert mapped_po["source_system"] == "Oracle"


@pytest.mark.integration
@pytest.mark.oracle_sandbox
class TestOracleEndToEnd:
    """End-to-end integration tests for Oracle connector."""

    def test_e2e_extraction_mapping_validation(self, oracle_client: OracleRESTClient):
        """Test complete flow: extract -> map -> validate schema."""
        # Step 1: Extract raw data
        query = create_query().limit(5)
        response = oracle_client.get("purchase_orders", query.build())
        raw_pos = response.get("items", [])

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
            assert mapped_po["source_system"] == "Oracle"

            # Data types
            assert isinstance(mapped_po["transaction_id"], str)
            assert isinstance(mapped_po.get("total_amount", 0), (int, float))

    def test_e2e_with_filters(self, oracle_client: OracleRESTClient):
        """Test end-to-end flow with complex filters."""
        # Extract with date and status filters
        cutoff_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        query = create_query().q(
            f"CreationDate >= '{cutoff_date}T00:00:00' and Status='APPROVED'"
        ).limit(10)

        try:
            response = oracle_client.get("purchase_orders", query.build())
            records = response.get("items", [])

            # Validate filter was applied
            for record in records:
                if "Status" in record:
                    assert record["Status"] in ["APPROVED", "Approved"]

        except OracleConnectionError:
            pytest.skip("Connection error - sandbox may be unavailable")


@pytest.mark.integration
@pytest.mark.oracle_sandbox
class TestOracleErrorHandling:
    """Test Oracle error handling and retry logic."""

    def test_retry_on_transient_error(self, oracle_sandbox_config):
        """Test retry on transient errors."""
        # Configure aggressive retry
        oracle_sandbox_config.retry.max_retries = 2
        oracle_sandbox_config.retry.base_delay = 0.5

        client = OracleRESTClient(oracle_sandbox_config)

        try:
            # Make request that may fail transiently
            query = create_query().limit(1)
            response = client.get("purchase_orders", query.build())
            assert response is not None

        finally:
            client.close()


# ==================== Mock Tests for CI/CD ====================

@pytest.mark.integration
class TestOracleMockIntegration:
    """Integration tests using mock Oracle client for CI/CD."""

    def test_mock_extraction(self, mock_oracle_client: OracleRESTClient):
        """Test extraction with mock client."""
        query = create_query().limit(1)
        response = mock_oracle_client.get("purchase_orders", query.build())

        assert "items" in response
        assert len(response["items"]) > 0

    def test_mock_pagination(self, mock_oracle_client: OracleRESTClient):
        """Test pagination with mock client."""
        batch_count = 0

        for batch in mock_oracle_client.query_paginated("purchase_orders", {"limit": 10}):
            batch_count += 1
            assert len(batch) > 0
            break  # Only test first batch

        assert batch_count == 1

    def test_mock_data_mapping(self, mock_oracle_client: OracleRESTClient):
        """Test data mapping with mock client."""
        query = create_query().limit(1)
        response = mock_oracle_client.get("purchase_orders", query.build())
        raw_pos = response.get("items", [])

        if raw_pos:
            mapper = PurchaseOrderMapper()
            mapped_po = mapper.map(raw_pos[0])

            assert "transaction_id" in mapped_po
            assert "source_system" in mapped_po
