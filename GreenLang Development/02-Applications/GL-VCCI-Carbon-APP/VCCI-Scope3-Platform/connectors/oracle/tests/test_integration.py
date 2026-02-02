# -*- coding: utf-8 -*-
"""
Oracle Connector Integration Tests
GL-VCCI Scope 3 Platform

End-to-end integration tests for Oracle connector including
multi-module extraction, error recovery, and throughput scenarios.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Test Count: 8
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import responses

from connectors.oracle.config import OracleConnectorConfig, OAuth2Config, OracleEnvironment
from connectors.oracle.auth import OracleAuthHandler
from connectors.oracle.client import OracleRESTClient
from connectors.oracle.extractors.procurement_extractor import ProcurementExtractor
from connectors.oracle.extractors.scm_extractor import SCMExtractor
from connectors.oracle.mappers.po_mapper import PurchaseOrderMapper
from connectors.oracle.mappers.shipment_mapper import ShipmentMapper


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @responses.activate
    def test_full_procurement_flow(self, oracle_config, mock_oauth_response):
        """Test complete procurement extraction and mapping flow."""
        # Mock OAuth
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        # Mock PO extraction
        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            json={
                "items": [{
                    "POHeaderId": 123,
                    "OrderNumber": "PO-001",
                    "SupplierId": 1001,
                    "SupplierName": "Test Supplier",
                    "Currency": "USD",
                    "OrderedDate": "2024-01-15T00:00:00Z"
                }],
                "hasMore": False,
                "links": []
            },
            status=200
        )

        # Create client and extractor
        client = OracleRESTClient(oracle_config)
        extractor = ProcurementExtractor(client)

        # Extract data
        pos = list(extractor.get_purchase_orders())

        assert len(pos) == 1
        assert pos[0]["OrderNumber"] == "PO-001"

    @responses.activate
    def test_multi_module_extraction(self, oracle_config, mock_oauth_response):
        """Test extracting from multiple Oracle modules."""
        # Mock OAuth
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        # Mock Procurement endpoint
        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            json={"items": [{"POHeaderId": 1}], "hasMore": False, "links": []},
            status=200
        )

        # Mock SCM endpoint
        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/shipments",
            json={"items": [{"ShipmentId": 1}], "hasMore": False, "links": []},
            status=200
        )

        client = OracleRESTClient(oracle_config)

        # Extract from Procurement
        proc_extractor = ProcurementExtractor(client)
        pos = list(proc_extractor.get_purchase_orders())

        # Extract from SCM
        scm_extractor = SCMExtractor(client)
        shipments = list(scm_extractor.get_shipments())

        assert len(pos) == 1
        assert len(shipments) == 1

    @responses.activate
    def test_extraction_with_mapping(self, oracle_config, mock_oauth_response):
        """Test extraction followed by data mapping."""
        # Mock OAuth
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        # Mock PO data
        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            json={
                "items": [{
                    "POHeaderId": 123,
                    "OrderNumber": "PO-001",
                    "SupplierId": 1001,
                    "SupplierName": "Test Supplier",
                    "Currency": "USD",
                    "OrderedDate": "2024-01-15"
                }],
                "hasMore": False,
                "links": []
            },
            status=200
        )

        # Extract
        client = OracleRESTClient(oracle_config)
        extractor = ProcurementExtractor(client)
        po_headers = list(extractor.get_purchase_orders())

        # Map
        mapper = PurchaseOrderMapper(tenant_id="test")
        po_line = {
            "LineNumber": 1,
            "ItemDescription": "Test Item",
            "Quantity": 10,
            "UOM": "EA",
            "LineAmount": 1000.0
        }

        records = []
        for header in po_headers:
            record = mapper.map_purchase_order(header, po_line)
            records.append(record)

        assert len(records) == 1
        assert records[0].procurement_id == "PROC-123-00001"

    @responses.activate
    def test_error_recovery_with_retry(self, oracle_config, mock_oauth_response):
        """Test error recovery with automatic retry."""
        # Mock OAuth
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        # First request fails with 503
        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            json={"title": "Service Unavailable"},
            status=503
        )

        # Second request succeeds
        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            json={"items": [{"POHeaderId": 1}], "hasMore": False, "links": []},
            status=200
        )

        client = OracleRESTClient(oracle_config)
        result = client.get("purchase_orders")

        # Should succeed after retry
        assert result is not None
        assert "items" in result

    @responses.activate
    def test_throughput_batch_processing(self, oracle_config, mock_oauth_response):
        """Test high-throughput batch processing."""
        # Mock OAuth
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        # Mock multiple pages of data
        for offset in range(0, 1000, 100):
            has_more = offset < 900
            next_offset = offset + 100

            items = [{"POHeaderId": i} for i in range(offset, min(offset + 100, 1000))]

            links = []
            if has_more:
                links.append({
                    "rel": "next",
                    "href": f"https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders?offset={next_offset}"
                })

            responses.add(
                responses.GET,
                f"https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
                json={"items": items, "hasMore": has_more, "links": links},
                status=200
            )

        client = OracleRESTClient(oracle_config)
        extractor = ProcurementExtractor(client)

        # Extract all records
        all_records = list(extractor.get_all())

        # Should handle large dataset
        assert len(all_records) >= 100  # At least first batch

    @pytest.mark.skip(reason="Requires Oracle sandbox credentials")
    def test_oracle_sandbox_connection(self):
        """Test actual connection to Oracle sandbox (requires credentials)."""
        # This test would connect to real Oracle sandbox
        # Only run manually with valid credentials
        config = OracleConnectorConfig.from_env()
        client = OracleRESTClient(config)

        # Test basic connectivity
        result = client.get("purchase_orders", query_params={"limit": 1})

        assert result is not None
        client.close()

    def test_delta_extraction_scenario(self):
        """Test delta extraction from last sync timestamp."""
        mock_client = Mock()

        # Mock response with records after timestamp
        mock_client.get.return_value = {
            "items": [
                {"POHeaderId": 1, "LastUpdateDate": "2024-01-15T10:00:00Z"},
                {"POHeaderId": 2, "LastUpdateDate": "2024-01-16T12:00:00Z"}
            ],
            "hasMore": False,
            "links": []
        }

        extractor = ProcurementExtractor(mock_client)

        # Extract delta since 2024-01-01
        records = list(extractor.get_delta("2024-01-01T00:00:00Z"))

        assert len(records) == 2

        # Verify delta filter was applied
        call_args = mock_client.get.call_args
        assert "params" in call_args[1]
        assert "q" in call_args[1]["params"]

    def test_concurrent_module_extraction(self, oracle_config):
        """Test extracting from multiple modules concurrently (stubbed)."""
        # In a real scenario, this would use threading/asyncio
        # Here we just verify the pattern works

        mock_client = Mock()
        mock_client.get.return_value = {
            "items": [{"id": 1}],
            "hasMore": False,
            "links": []
        }

        # Create extractors for different modules
        proc_extractor = ProcurementExtractor(mock_client)
        scm_extractor = SCMExtractor(mock_client)

        # Extract from both (sequential in this test)
        proc_data = list(proc_extractor.get_purchase_orders())
        scm_data = list(scm_extractor.get_shipments())

        assert len(proc_data) > 0
        assert len(scm_data) > 0


class TestConfigurationScenarios:
    """Tests for different configuration scenarios."""

    def test_multi_environment_configuration(self):
        """Test configuration for different environments."""
        oauth = OAuth2Config(
            client_id="test",
            client_secret="secret",
            token_url="https://test.com/oauth"
        )

        # Sandbox config
        sandbox_config = OracleConnectorConfig(
            environment=OracleEnvironment.SANDBOX,
            base_url="https://sandbox.oraclecloud.com",
            oauth=oauth
        )

        # Production config
        prod_config = OracleConnectorConfig(
            environment=OracleEnvironment.PRODUCTION,
            base_url="https://prod.oraclecloud.com",
            oauth=oauth
        )

        assert sandbox_config.environment == OracleEnvironment.SANDBOX
        assert prod_config.environment == OracleEnvironment.PRODUCTION

    def test_custom_batch_sizes(self):
        """Test configuration with custom batch sizes."""
        oauth = OAuth2Config(
            client_id="test",
            client_secret="secret",
            token_url="https://test.com/oauth"
        )

        config = OracleConnectorConfig(
            environment=OracleEnvironment.SANDBOX,
            base_url="https://test.com",
            oauth=oauth,
            default_batch_size=2000
        )

        # Endpoints should use custom batch size
        assert config.endpoints["purchase_orders"].batch_size == 2000
