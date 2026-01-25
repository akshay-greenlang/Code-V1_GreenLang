# -*- coding: utf-8 -*-
"""
OData Client Tests for SAP Connector
GL-VCCI Scope 3 Platform

Tests for SAP OData client including:
- OData query building ($filter, $select, $top, $skip, $orderby)
- Pagination handling (@odata.nextLink)
- GET/POST operations
- Error handling (HTTP errors, OData errors)
- Rate limiting
- Retry logic
- Timeout handling
- Authentication integration

Test Count: 25 tests
Coverage Target: 95%+

Version: 1.0.0
Phase: 4 (Weeks 24-26)
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import requests

from connectors.sap.client import (
    ODataQueryBuilder,
    RateLimiter,
    SAPODataClient,
    create_query,
)
from connectors.sap.exceptions import (
    SAPConnectionError,
    SAPAuthenticationError,
    SAPRateLimitError,
    SAPDataError,
    SAPTimeoutError,
)


class TestODataQueryBuilder:
    """Tests for OData query builder."""

    def test_should_build_filter_query(self):
        """Test building $filter parameter."""
        builder = ODataQueryBuilder()
        params = builder.filter("PurchaseOrder eq '4500000001'").build()

        assert params["$filter"] == "PurchaseOrder eq '4500000001'"

    def test_should_build_select_query(self):
        """Test building $select parameter."""
        builder = ODataQueryBuilder()
        params = builder.select("PurchaseOrder", "Vendor", "Amount").build()

        assert params["$select"] == "PurchaseOrder,Vendor,Amount"

    def test_should_build_top_query(self):
        """Test building $top parameter."""
        builder = ODataQueryBuilder()
        params = builder.top(100).build()

        assert params["$top"] == 100

    def test_should_build_skip_query(self):
        """Test building $skip parameter."""
        builder = ODataQueryBuilder()
        params = builder.skip(50).build()

        assert params["$skip"] == 50

    def test_should_build_orderby_query_ascending(self):
        """Test building $orderby parameter (ascending)."""
        builder = ODataQueryBuilder()
        params = builder.orderby("PurchaseOrderDate", "PurchaseOrder").build()

        assert params["$orderby"] == "PurchaseOrderDate,PurchaseOrder"

    def test_should_build_orderby_query_descending(self):
        """Test building $orderby parameter (descending)."""
        builder = ODataQueryBuilder()
        params = builder.orderby("PurchaseOrderDate", descending=True).build()

        assert params["$orderby"] == "PurchaseOrderDate desc"

    def test_should_build_expand_query(self):
        """Test building $expand parameter."""
        builder = ODataQueryBuilder()
        params = builder.expand("Items", "Vendor").build()

        assert params["$expand"] == "Items,Vendor"

    def test_should_chain_multiple_query_params(self):
        """Test chaining multiple query parameters."""
        builder = ODataQueryBuilder()
        params = (builder
                  .filter("Amount gt 1000")
                  .select("PurchaseOrder", "Amount")
                  .top(50)
                  .orderby("Amount", descending=True)
                  .build())

        assert params["$filter"] == "Amount gt 1000"
        assert params["$select"] == "PurchaseOrder,Amount"
        assert params["$top"] == 50
        assert params["$orderby"] == "Amount desc"

    def test_should_create_query_builder_via_factory(self):
        """Test creating query builder via factory function."""
        builder = create_query()

        assert isinstance(builder, ODataQueryBuilder)


class TestRateLimiter:
    """Tests for rate limiter."""

    def test_should_allow_request_within_limit(self):
        """Test allowing request within rate limit."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)

        result = limiter.acquire(timeout=1.0)

        assert result is True

    def test_should_enforce_burst_limit(self):
        """Test enforcing burst size limit."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=3)

        # First 3 should succeed (burst)
        assert limiter.acquire(timeout=0.1) is True
        assert limiter.acquire(timeout=0.1) is True
        assert limiter.acquire(timeout=0.1) is True

        # 4th should wait or fail
        start = time.time()
        result = limiter.acquire(timeout=0.1)
        elapsed = time.time() - start

        # Should either wait or return False
        if result:
            assert elapsed > 0.05  # Waited for token

    def test_should_refill_tokens_over_time(self):
        """Test token refill over time."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=2)

        # Consume tokens
        limiter.acquire()
        limiter.acquire()

        # Wait for refill (60 req/min = 1 req/sec)
        time.sleep(1.1)

        # Should have new token
        result = limiter.acquire(timeout=0.1)
        assert result is True


class TestSAPODataClient:
    """Tests for SAP OData client."""

    def test_should_initialize_client(self, sap_config):
        """Test client initialization."""
        client = SAPODataClient(sap_config)

        assert client.config == sap_config
        assert client.auth_handler is not None
        assert client.session is not None
        assert client.rate_limiter is not None

    def test_should_disable_rate_limiter_when_configured(self, sap_config):
        """Test rate limiter can be disabled."""
        sap_config.rate_limit.enabled = False

        client = SAPODataClient(sap_config)

        assert client.rate_limiter is None

    @patch('connectors.sap.client.SAPODataClient._make_request')
    def test_should_execute_get_request(self, mock_request, sap_config, mock_odata_response_collection):
        """Test executing GET request."""
        mock_response = Mock()
        mock_response.json.return_value = mock_odata_response_collection
        mock_request.return_value = mock_response

        client = SAPODataClient(sap_config)
        result = client.get("purchase_orders")

        assert result == mock_odata_response_collection
        assert mock_request.called

    @patch('connectors.sap.client.SAPODataClient._make_request')
    def test_should_execute_get_request_with_entity_id(self, mock_request, sap_config, mock_odata_response_single):
        """Test GET request for single entity."""
        mock_response = Mock()
        mock_response.json.return_value = mock_odata_response_single
        mock_request.return_value = mock_response

        client = SAPODataClient(sap_config)
        result = client.get("purchase_orders", entity_id="4500000001")

        assert result == mock_odata_response_single
        assert "'4500000001'" in str(mock_request.call_args)

    def test_should_raise_error_for_unknown_endpoint(self, sap_config):
        """Test error for unknown endpoint."""
        client = SAPODataClient(sap_config)

        with pytest.raises(ValueError) as exc_info:
            client.get("unknown_endpoint")

        assert "Unknown endpoint" in str(exc_info.value)

    @patch('connectors.sap.client.SAPODataClient._make_request')
    def test_should_handle_pagination(self, mock_request, sap_config, sample_po_data):
        """Test pagination handling with @odata.nextLink."""
        # First page
        mock_response1 = Mock()
        mock_response1.json.return_value = {
            "value": [sample_po_data[0]],
            "@odata.nextLink": "https://api.sap.com/next"
        }

        # Second page (last)
        mock_response2 = Mock()
        mock_response2.json.return_value = {
            "value": [sample_po_data[1]]
        }

        mock_request.side_effect = [mock_response1, mock_response2]

        client = SAPODataClient(sap_config)
        results = list(client.query_paginated("purchase_orders"))

        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1
        assert mock_request.call_count == 2

    @patch('connectors.sap.client.SAPODataClient._make_request')
    def test_should_query_with_builder(self, mock_request, sap_config, sample_po_data):
        """Test querying with OData query builder."""
        mock_response = Mock()
        mock_response.json.return_value = {"value": sample_po_data}
        mock_request.return_value = mock_response

        client = SAPODataClient(sap_config)
        builder = create_query().filter("Amount gt 1000").top(10)

        results = client.query("purchase_orders", builder)

        assert len(results) == 2
        call_params = mock_request.call_args[1]["params"]
        assert "$filter" in call_params
        assert "$top" in call_params

    @patch('requests.Session.request')
    def test_should_handle_401_and_refresh_token(self, mock_session_request, sap_config, mock_oauth_token_response):
        """Test handling 401 by refreshing token."""
        # First request: 401
        mock_response_401 = Mock()
        mock_response_401.status_code = 401

        # Second request: Success
        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {"value": []}

        mock_session_request.side_effect = [mock_response_401, mock_response_200]

        client = SAPODataClient(sap_config)

        with patch('requests.post') as mock_token_post:
            mock_token_response = Mock()
            mock_token_response.status_code = 200
            mock_token_response.json.return_value = mock_oauth_token_response
            mock_token_post.return_value = mock_token_response

            result = client.get("purchase_orders")

            # Should succeed after token refresh
            assert result == {"value": []}
            assert mock_session_request.call_count == 2

    @patch('requests.Session.request')
    def test_should_retry_on_503_error(self, mock_session_request, sap_config):
        """Test retry on 503 Service Unavailable."""
        # First request: 503
        mock_response_503 = Mock()
        mock_response_503.status_code = 503

        # Second request: Success
        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {"value": []}

        mock_session_request.side_effect = [mock_response_503, mock_response_200]

        client = SAPODataClient(sap_config)

        with patch('requests.post') as mock_token_post:
            mock_token_response = Mock()
            mock_token_response.status_code = 200
            mock_token_response.json.return_value = {"access_token": "token", "expires_in": 3600}
            mock_token_post.return_value = mock_token_response

            result = client.get("purchase_orders")

            assert result == {"value": []}
            assert mock_session_request.call_count >= 2

    @patch('requests.Session.request')
    def test_should_handle_timeout_error(self, mock_session_request, sap_config):
        """Test handling timeout errors."""
        mock_session_request.side_effect = requests.exceptions.Timeout("Request timed out")

        client = SAPODataClient(sap_config)

        with patch('requests.post') as mock_token_post:
            mock_token_response = Mock()
            mock_token_response.status_code = 200
            mock_token_response.json.return_value = {"access_token": "token", "expires_in": 3600}
            mock_token_post.return_value = mock_token_response

            with pytest.raises(SAPTimeoutError) as exc_info:
                client.get("purchase_orders")

            assert "timed out" in str(exc_info.value).lower()

    @patch('requests.Session.request')
    def test_should_handle_connection_error(self, mock_session_request, sap_config):
        """Test handling connection errors."""
        mock_session_request.side_effect = requests.exceptions.ConnectionError("Connection refused")

        client = SAPODataClient(sap_config)

        with patch('requests.post') as mock_token_post:
            mock_token_response = Mock()
            mock_token_response.status_code = 200
            mock_token_response.json.return_value = {"access_token": "token", "expires_in": 3600}
            mock_token_post.return_value = mock_token_response

            with pytest.raises(SAPConnectionError):
                client.get("purchase_orders")

    @patch('connectors.sap.client.SAPODataClient._make_request')
    def test_should_handle_rate_limit_error(self, mock_request, sap_config):
        """Test rate limiting enforcement."""
        client = SAPODataClient(sap_config)
        client.rate_limiter = RateLimiter(requests_per_minute=1, burst_size=1)

        # Consume token
        client.rate_limiter.acquire()

        # Next request should fail rate limit
        mock_request.return_value = Mock()

        with pytest.raises(SAPRateLimitError):
            client.get("purchase_orders")

    @patch('connectors.sap.client.SAPODataClient._make_request')
    def test_should_parse_odata_error(self, mock_request, sap_config, mock_odata_error_response):
        """Test parsing OData error response."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = mock_odata_error_response
        mock_request.return_value = mock_response

        client = SAPODataClient(sap_config)

        with pytest.raises(SAPDataError) as exc_info:
            client.get("purchase_orders")

        assert "404" in str(exc_info.value)

    @patch('connectors.sap.client.SAPODataClient._make_request')
    def test_should_execute_post_request(self, mock_request, sap_config):
        """Test executing POST request."""
        mock_response = Mock()
        mock_response.json.return_value = {"PurchaseOrder": "4500000001"}
        mock_request.return_value = mock_response

        client = SAPODataClient(sap_config)
        data = {"Vendor": "VENDOR001", "Amount": 1000}
        result = client.post("purchase_orders", data)

        assert result["PurchaseOrder"] == "4500000001"
        assert mock_request.call_args[0][0] == "POST"

    def test_should_close_session(self, sap_config):
        """Test closing client session."""
        client = SAPODataClient(sap_config)

        with patch.object(client.session, 'close') as mock_close:
            client.close()

            assert mock_close.called
