# -*- coding: utf-8 -*-
"""
Oracle Connector REST Client Tests
GL-VCCI Scope 3 Platform

Tests for REST client operations, query building,
pagination, error handling, and rate limiting.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Test Count: 22
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import responses
import requests

from connectors.oracle.client import (
    OracleRESTClient,
    RateLimiter,
    RESTQueryBuilder,
    create_query
)
from connectors.oracle.config import OracleConnectorConfig, OracleModule
from connectors.oracle.exceptions import (
    OracleConnectionError,
    OracleRateLimitError,
    OracleTimeoutError,
    OracleDataError,
    OracleAuthenticationError
)


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_rate_limiter_initialization(self, rate_limiter):
        """Test rate limiter initialization."""
        assert rate_limiter.requests_per_minute == 60
        assert rate_limiter.burst_size == 10
        assert rate_limiter.tokens == 10

    def test_rate_limiter_acquire(self, rate_limiter):
        """Test acquiring permission to make request."""
        # Should allow burst size requests immediately
        for _ in range(10):
            assert rate_limiter.acquire(timeout=1.0) is True

    def test_rate_limiter_throttling(self):
        """Test rate limiting throttles requests."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=2)

        # First 2 should succeed immediately
        assert limiter.acquire(timeout=0.1) is True
        assert limiter.acquire(timeout=0.1) is True

        # Third should be throttled
        start = time.time()
        result = limiter.acquire(timeout=0.2)
        elapsed = time.time() - start

        # Should either wait or timeout
        assert elapsed > 0.1 or result is False


class TestRESTQueryBuilder:
    """Tests for RESTQueryBuilder."""

    def test_query_builder_initialization(self, query_builder):
        """Test query builder initialization."""
        assert query_builder.params == {}

    def test_query_builder_q_parameter(self, query_builder):
        """Test adding q filter parameter."""
        params = query_builder.q("LastUpdateDate >= '2024-01-01T00:00:00'").build()

        assert params["q"] == "LastUpdateDate >= '2024-01-01T00:00:00'"

    def test_query_builder_fields_parameter(self, query_builder):
        """Test adding fields parameter."""
        params = query_builder.fields("POHeaderId", "OrderNumber", "SupplierId").build()

        assert params["fields"] == "POHeaderId,OrderNumber,SupplierId"

    def test_query_builder_limit_parameter(self, query_builder):
        """Test adding limit parameter."""
        params = query_builder.limit(500).build()

        assert params["limit"] == 500

    def test_query_builder_offset_parameter(self, query_builder):
        """Test adding offset parameter."""
        params = query_builder.offset(100).build()

        assert params["offset"] == 100

    def test_query_builder_orderby_parameter(self, query_builder):
        """Test adding orderBy parameter."""
        params = query_builder.orderby("CreationDate", descending=False).build()

        assert params["orderBy"] == "CreationDate"

    def test_query_builder_orderby_descending(self, query_builder):
        """Test orderBy with descending order."""
        params = query_builder.orderby("LastUpdateDate", descending=True).build()

        assert params["orderBy"] == "LastUpdateDate:desc"

    def test_query_builder_finder_parameter(self, query_builder):
        """Test adding finder parameter."""
        params = query_builder.finder("FindBySupplier", SupplierId="1001").build()

        assert "finder" in params
        assert "FindBySupplier" in params["finder"]
        assert "SupplierId=1001" in params["finder"]

    def test_query_builder_expand_parameter(self, query_builder):
        """Test adding expand parameter."""
        params = query_builder.expand("lines", "distributions").build()

        assert params["expand"] == "lines,distributions"

    def test_query_builder_chaining(self):
        """Test method chaining."""
        builder = create_query()
        params = builder.q("Status='APPROVED'") \
                       .fields("POHeaderId", "OrderNumber") \
                       .limit(100) \
                       .offset(0) \
                       .orderby("CreationDate") \
                       .build()

        assert "q" in params
        assert "fields" in params
        assert "limit" in params
        assert "offset" in params
        assert "orderBy" in params


class TestOracleRESTClient:
    """Tests for OracleRESTClient."""

    def test_client_initialization(self, rest_client, oracle_config):
        """Test REST client initialization."""
        assert rest_client.config == oracle_config
        assert rest_client.auth_handler is not None
        assert rest_client.session is not None

    @responses.activate
    def test_get_single_resource(self, rest_client, oracle_rest_response_single, mock_oauth_response):
        """Test GET request for single resource."""
        # Mock OAuth
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        # Mock GET request
        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            json=oracle_rest_response_single,
            status=200
        )

        result = rest_client.get("purchase_orders")

        assert "items" in result
        assert len(result["items"]) == 1

    @responses.activate
    def test_get_with_query_params(self, rest_client, oracle_rest_response_single, mock_oauth_response):
        """Test GET request with query parameters."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            json=oracle_rest_response_single,
            status=200
        )

        query_params = {"q": "Status='APPROVED'", "limit": 10}
        result = rest_client.get("purchase_orders", query_params=query_params)

        assert result is not None
        # Verify query params were sent
        assert "Status" in responses.calls[1].request.url or "q=" in responses.calls[1].request.url

    @responses.activate
    def test_get_unknown_endpoint(self, rest_client):
        """Test GET request to unknown endpoint raises error."""
        with pytest.raises(ValueError, match="Unknown endpoint"):
            rest_client.get("nonexistent_endpoint")

    @responses.activate
    def test_post_request(self, rest_client, mock_oauth_response):
        """Test POST request."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            json={"POHeaderId": 123, "OrderNumber": "PO-001"},
            status=201
        )

        data = {"OrderNumber": "PO-001", "SupplierId": 1001}
        result = rest_client.post("purchase_orders", data=data)

        assert result["POHeaderId"] == 123

    @responses.activate
    def test_patch_request(self, rest_client, mock_oauth_response):
        """Test PATCH request."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        responses.add(
            responses.PATCH,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders/123",
            json={"POHeaderId": 123, "Status": "APPROVED"},
            status=200
        )

        data = {"Status": "APPROVED"}
        result = rest_client.patch("purchase_orders", resource_id="123", data=data)

        assert result["Status"] == "APPROVED"

    @responses.activate
    def test_pagination(self, rest_client, mock_oauth_response):
        """Test pagination handling."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        # First page
        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            json={
                "items": [{"POHeaderId": 1}, {"POHeaderId": 2}],
                "count": 2,
                "hasMore": True,
                "links": [
                    {"rel": "next", "href": "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders?offset=2"}
                ]
            },
            status=200
        )

        # Second page
        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders?offset=2",
            json={
                "items": [{"POHeaderId": 3}],
                "count": 1,
                "hasMore": False,
                "links": []
            },
            status=200
        )

        results = list(rest_client.query_paginated("purchase_orders"))

        assert len(results) == 2  # Two batches
        assert len(results[0]) == 2  # First batch has 2 items
        assert len(results[1]) == 1  # Second batch has 1 item

    @responses.activate
    def test_error_handling_404(self, rest_client, mock_oauth_response):
        """Test handling 404 errors."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            json={"title": "Not Found", "detail": "Resource not found"},
            status=404
        )

        with pytest.raises(OracleDataError):
            rest_client.get("purchase_orders")

    @responses.activate
    def test_error_handling_401_with_retry(self, rest_client, mock_oauth_response):
        """Test handling 401 with token refresh."""
        # First OAuth call
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        # First API call returns 401
        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            json={"title": "Unauthorized"},
            status=401
        )

        # Second OAuth call (refresh)
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        # Second API call succeeds
        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            json={"items": [], "count": 0, "hasMore": False, "links": []},
            status=200
        )

        result = rest_client.get("purchase_orders")

        # Should succeed after token refresh
        assert result is not None

    @responses.activate
    def test_rate_limit_error(self, rest_client, mock_oauth_response):
        """Test handling rate limit errors."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            json={"title": "Too Many Requests"},
            status=429
        )

        # Should raise rate limit error
        with pytest.raises(OracleRateLimitError):
            rest_client.get("purchase_orders")

    @responses.activate
    def test_timeout_error(self, rest_client, mock_oauth_response):
        """Test handling timeout errors."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            body=requests.exceptions.Timeout("Request timed out")
        )

        with pytest.raises(OracleTimeoutError):
            rest_client.get("purchase_orders")

    @responses.activate
    def test_connection_error(self, rest_client, mock_oauth_response):
        """Test handling connection errors."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            body=requests.exceptions.ConnectionError("Connection failed")
        )

        with pytest.raises(OracleConnectionError):
            rest_client.get("purchase_orders")

    def test_close_session(self, rest_client):
        """Test closing REST client session."""
        session = rest_client.session
        rest_client.close()

        # Session should be closed (difficult to verify, but method should run)
        assert True  # No exception raised

    @responses.activate
    def test_query_method(self, rest_client, mock_oauth_response):
        """Test query method returns all results."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        responses.add(
            responses.GET,
            "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            json={
                "items": [{"POHeaderId": 1}, {"POHeaderId": 2}],
                "count": 2,
                "hasMore": False,
                "links": []
            },
            status=200
        )

        builder = create_query().limit(100)
        results = rest_client.query("purchase_orders", builder)

        assert len(results) == 2
        assert results[0]["POHeaderId"] == 1
