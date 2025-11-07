"""
Pytest Fixtures and Mocks for Oracle Connector Tests
GL-VCCI Scope 3 Platform

Shared fixtures for Oracle REST API mocking, sample data generation,
and test utilities.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock
import responses

from connectors.oracle.config import (
    OracleConnectorConfig,
    OAuth2Config,
    RESTEndpoint,
    OracleModule,
    OracleEnvironment,
    RateLimitConfig,
    RetryConfig,
    TimeoutConfig
)
from connectors.oracle.auth import TokenCache, OracleAuthHandler
from connectors.oracle.client import OracleRESTClient, RateLimiter, RESTQueryBuilder


# ==================== Configuration Fixtures ====================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for Oracle configuration."""
    env_vars = {
        "ORACLE_ENVIRONMENT": "sandbox",
        "ORACLE_BASE_URL": "https://test.oraclecloud.com",
        "ORACLE_CLIENT_ID": "test_client_id",
        "ORACLE_CLIENT_SECRET": "test_client_secret",
        "ORACLE_TOKEN_URL": "https://test.oraclecloud.com/oauth/token",
        "ORACLE_OAUTH_SCOPE": "urn:opc:resource:consumer::all",
        "ORACLE_RATE_LIMIT_RPM": "10",
        "ORACLE_BATCH_SIZE": "1000",
        "ORACLE_DEBUG_MODE": "false"
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def oauth_config():
    """Create test OAuth configuration."""
    return OAuth2Config(
        client_id="test_client_id",
        client_secret="test_client_secret",
        token_url="https://test.oraclecloud.com/oauth/token",
        scope="urn:opc:resource:consumer::all"
    )


@pytest.fixture
def oracle_config(oauth_config):
    """Create test Oracle connector configuration."""
    return OracleConnectorConfig(
        environment=OracleEnvironment.SANDBOX,
        base_url="https://test.oraclecloud.com",
        oauth=oauth_config,
        default_batch_size=1000,
        debug_mode=False
    )


# ==================== Authentication Fixtures ====================

@pytest.fixture
def token_cache():
    """Create fresh token cache."""
    return TokenCache()


@pytest.fixture
def auth_handler(oauth_config, token_cache):
    """Create test auth handler."""
    return OracleAuthHandler(
        oauth_config=oauth_config,
        environment=OracleEnvironment.SANDBOX,
        cache=token_cache
    )


@pytest.fixture
def mock_oauth_response():
    """Mock OAuth token response."""
    return {
        "access_token": "test_access_token_12345",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": "urn:opc:resource:consumer::all"
    }


# ==================== REST Client Fixtures ====================

@pytest.fixture
def rest_client(oracle_config):
    """Create test REST client."""
    return OracleRESTClient(oracle_config)


@pytest.fixture
def rate_limiter():
    """Create test rate limiter."""
    return RateLimiter(requests_per_minute=60, burst_size=10)


@pytest.fixture
def query_builder():
    """Create test query builder."""
    return RESTQueryBuilder()


# ==================== Sample Oracle Data Fixtures ====================

@pytest.fixture
def sample_purchase_order():
    """Sample Oracle Purchase Order data."""
    return {
        "POHeaderId": 300100123456789,
        "OrderNumber": "PO-2024-001234",
        "SupplierId": 1001,
        "SupplierName": "Acme Corp",
        "Currency": "USD",
        "OrderedDate": "2024-01-15T10:30:00Z",
        "BU": "US01",
        "BUName": "US Operations",
        "BuyerId": 5001,
        "BuyerName": "John Smith",
        "DocumentStatus": "APPROVED",
        "PaymentTerms": "NET30",
        "FreightTerms": "FOB",
        "LastUpdateDate": "2024-01-15T10:30:00Z"
    }


@pytest.fixture
def sample_po_line():
    """Sample Oracle PO Line data."""
    return {
        "POLineId": 300100123456790,
        "LineNumber": 1,
        "ItemNumber": "ITEM-12345",
        "ItemDescription": "Widget Type A",
        "Quantity": 100.0,
        "UOM": "EA",
        "LineAmount": 5000.00,
        "CategoryId": 101,
        "CategoryName": "Raw Materials",
        "ShipToLocationCode": "WH-001",
        "NeedByDate": "2024-02-15",
        "PromisedDate": "2024-02-10"
    }


@pytest.fixture
def sample_requisition():
    """Sample Oracle Purchase Requisition data."""
    return {
        "RequisitionHeaderId": 200100123456789,
        "RequisitionNumber": "REQ-2024-005678",
        "RequesterId": 6001,
        "RequesterName": "Jane Doe",
        "RequisitionDate": "2024-01-10T14:20:00Z",
        "BU": "US01",
        "Description": "Office supplies request",
        "TotalAmount": 2500.00,
        "Currency": "USD",
        "Status": "APPROVED",
        "LastUpdateDate": "2024-01-10T14:20:00Z"
    }


@pytest.fixture
def sample_shipment():
    """Sample Oracle Shipment data."""
    return {
        "ShipmentId": 400100123456789,
        "ShipmentNumber": "SHIP-2024-009876",
        "ShipFromLocation": "VENDOR-WH-001",
        "ShipToLocation": "WH-001",
        "ShipmentDate": "2024-01-20T08:00:00Z",
        "DeliveryDate": "2024-01-25T16:00:00Z",
        "Carrier": "FedEx",
        "TrackingNumber": "1234567890",
        "TotalWeight": 500.0,
        "WeightUOM": "LB",
        "Status": "DELIVERED",
        "LastUpdateDate": "2024-01-25T16:00:00Z"
    }


@pytest.fixture
def sample_transport_order():
    """Sample Oracle Transportation Order data."""
    return {
        "TransportOrderId": 500100123456789,
        "OrderNumber": "TRANS-2024-111222",
        "OriginLocation": "NYC-WH",
        "DestinationLocation": "LA-WH",
        "TransportMode": "TRUCK",
        "Carrier": "ABC Logistics",
        "Distance": 2800.0,
        "DistanceUOM": "MI",
        "Weight": 10000.0,
        "WeightUOM": "LB",
        "PickupDate": "2024-02-01T06:00:00Z",
        "DeliveryDate": "2024-02-05T18:00:00Z",
        "Status": "IN_TRANSIT",
        "LastUpdateDate": "2024-02-03T12:00:00Z"
    }


@pytest.fixture
def sample_supplier():
    """Sample Oracle Supplier data."""
    return {
        "SupplierId": 1001,
        "SupplierName": "Acme Corp",
        "SupplierNumber": "SUP-001",
        "Country": "US",
        "State": "CA",
        "City": "San Francisco",
        "Address": "123 Market St",
        "PostalCode": "94102",
        "TaxId": "12-3456789",
        "Status": "ACTIVE",
        "LastUpdateDate": "2024-01-01T00:00:00Z"
    }


# ==================== Oracle REST Response Fixtures ====================

@pytest.fixture
def oracle_rest_response_single():
    """Oracle REST response for single item."""
    return {
        "items": [{
            "POHeaderId": 300100123456789,
            "OrderNumber": "PO-2024-001234"
        }],
        "count": 1,
        "hasMore": False,
        "links": [
            {"rel": "self", "href": "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders"}
        ]
    }


@pytest.fixture
def oracle_rest_response_paginated():
    """Oracle REST response with pagination."""
    return {
        "items": [
            {"POHeaderId": 1, "OrderNumber": "PO-001"},
            {"POHeaderId": 2, "OrderNumber": "PO-002"}
        ],
        "count": 2,
        "hasMore": True,
        "links": [
            {"rel": "self", "href": "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders?limit=2&offset=0"},
            {"rel": "next", "href": "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders?limit=2&offset=2"}
        ]
    }


@pytest.fixture
def oracle_rest_response_empty():
    """Oracle REST response with no items."""
    return {
        "items": [],
        "count": 0,
        "hasMore": False,
        "links": [
            {"rel": "self", "href": "https://test.oraclecloud.com/fscmRestApi/resources/11.13.18.05/purchaseOrders"}
        ]
    }


# ==================== Mock Redis Client ====================

@pytest.fixture
def mock_redis():
    """Mock Redis client for caching tests."""
    redis_mock = MagicMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = True
    redis_mock.exists.return_value = False
    return redis_mock


# ==================== HTTP Mocking Utilities ====================

@pytest.fixture
def mock_responses():
    """Setup responses library for HTTP mocking."""
    with responses.RequestsMock() as rsps:
        yield rsps


# ==================== Test Utilities ====================

def create_oracle_error_response(status_code: int, title: str, detail: str = "") -> Dict[str, Any]:
    """Create Oracle REST error response."""
    return {
        "title": title,
        "detail": detail,
        "status": status_code,
        "o:errorCode": f"ORACLE-{status_code}",
        "o:errorPath": "/fscmRestApi/resources/11.13.18.05/purchaseOrders"
    }


def create_pagination_response(items: List[Dict], offset: int, limit: int, total: int) -> Dict[str, Any]:
    """Create paginated Oracle REST response."""
    has_more = (offset + limit) < total

    links = [
        {"rel": "self", "href": f"https://test.oraclecloud.com/api?limit={limit}&offset={offset}"}
    ]

    if has_more:
        next_offset = offset + limit
        links.append({
            "rel": "next",
            "href": f"https://test.oraclecloud.com/api?limit={limit}&offset={next_offset}"
        })

    return {
        "items": items,
        "count": len(items),
        "hasMore": has_more,
        "links": links
    }
