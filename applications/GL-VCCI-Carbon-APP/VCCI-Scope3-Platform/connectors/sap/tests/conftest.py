# -*- coding: utf-8 -*-
"""
Pytest Configuration and Fixtures for SAP Connector Tests
GL-VCCI Scope 3 Platform

Shared fixtures, mocks, and test data for SAP connector test suite.

Fixtures:
---------
- mock_sap_config: Mock SAP configuration
- mock_oauth_token_response: Mock OAuth token response
- mock_redis_client: Mock Redis client for caching
- mock_celery_app: Mock Celery application
- mock_sap_client: Mock SAP OData client
- sample_po_data: Sample purchase order data
- sample_gr_data: Sample goods receipt data
- sample_delivery_data: Sample delivery data
- sample_transport_data: Sample transport data

Version: 1.0.0
Phase: 4 (Weeks 24-26)
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest
import responses
from freezegun import freeze_time


# Configuration Fixtures
# ----------------------

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for SAP configuration."""
    env_vars = {
        "SAP_ENVIRONMENT": "sandbox",
        "SAP_BASE_URL": "https://sandbox.api.sap.com",
        "SAP_CLIENT_ID": "test-client-id",
        "SAP_CLIENT_SECRET": "test-client-secret",
        "SAP_TOKEN_URL": "https://auth.sap.com/oauth/token",
        "SAP_OAUTH_SCOPE": "API_BUSINESS_PARTNER",
        "SAP_RATE_LIMIT_RPM": "10",
        "SAP_BATCH_SIZE": "1000",
        "SAP_DEBUG_MODE": "true",
        "SAP_MAX_RETRIES": "3",
        "SAP_RETRY_BASE_DELAY": "1.0",
        "SAP_RETRY_MAX_DELAY": "8.0",
        "SAP_CONNECT_TIMEOUT": "10.0",
        "SAP_READ_TIMEOUT": "30.0",
        "SAP_TOTAL_TIMEOUT": "60.0",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def sap_config(mock_env_vars):
    """Create SAP connector configuration with mocked env vars."""
    from connectors.sap.config import SAPConnectorConfig
    return SAPConnectorConfig.from_env()


@pytest.fixture
def oauth_config():
    """Create OAuth configuration."""
    from connectors.sap.config import OAuth2Config
    return OAuth2Config(
        client_id="test-client-id",
        client_secret="test-client-secret",
        token_url="https://auth.sap.com/oauth/token",
        scope="API_BUSINESS_PARTNER",
        token_cache_ttl=3300
    )


# Authentication Fixtures
# -----------------------

@pytest.fixture
def mock_oauth_token_response():
    """Mock OAuth token response."""
    return {
        "access_token": "mock-access-token-12345",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": "API_BUSINESS_PARTNER"
    }


@pytest.fixture
def mock_oauth_error_response():
    """Mock OAuth error response."""
    return {
        "error": "invalid_client",
        "error_description": "Client authentication failed"
    }


@pytest.fixture
def token_cache():
    """Create fresh token cache for testing."""
    from connectors.sap.auth import TokenCache
    return TokenCache()


# Client Fixtures
# ---------------

@pytest.fixture
def mock_sap_client(sap_config):
    """Create mock SAP OData client."""
    from connectors.sap.client import SAPODataClient

    client = Mock(spec=SAPODataClient)
    client.config = sap_config
    client.session = Mock()

    return client


@pytest.fixture
def odata_query_builder():
    """Create OData query builder."""
    from connectors.sap.client import ODataQueryBuilder
    return ODataQueryBuilder()


# Sample SAP Data Fixtures
# -------------------------

@pytest.fixture
def sample_po_data() -> List[Dict[str, Any]]:
    """Sample SAP Purchase Order data."""
    return [
        {
            "PurchaseOrder": "4500000001",
            "PurchaseOrderType": "NB",
            "Vendor": "VENDOR001",
            "VendorName": "Acme Corporation",
            "PurchaseOrderDate": "2024-01-15T00:00:00Z",
            "CompanyCode": "1000",
            "PurchasingOrganization": "1000",
            "PurchasingGroup": "001",
            "Currency": "USD",
            "ExchangeRate": 1.0,
            "Items": [
                {
                    "PurchaseOrderItem": "00010",
                    "Material": "MAT-001",
                    "PurchaseOrderItemText": "Laptop Computer",
                    "OrderQuantity": 10.0,
                    "PurchaseOrderQuantityUnit": "EA",
                    "NetAmount": 15000.0,
                    "Plant": "1000",
                    "StorageLocation": "0001"
                }
            ]
        },
        {
            "PurchaseOrder": "4500000002",
            "PurchaseOrderType": "NB",
            "Vendor": "VENDOR002",
            "VendorName": "Global Supplies Inc",
            "PurchaseOrderDate": "2024-01-20T00:00:00Z",
            "CompanyCode": "1000",
            "Currency": "EUR",
            "ExchangeRate": 1.10,
            "Items": [
                {
                    "PurchaseOrderItem": "00010",
                    "Material": "MAT-002",
                    "PurchaseOrderItemText": "Office Chairs",
                    "OrderQuantity": 50.0,
                    "PurchaseOrderQuantityUnit": "PC",
                    "NetAmount": 5000.0,
                    "Plant": "1000"
                }
            ]
        }
    ]


@pytest.fixture
def sample_gr_data() -> List[Dict[str, Any]]:
    """Sample SAP Goods Receipt data."""
    return [
        {
            "MaterialDocument": "5000000001",
            "MaterialDocumentYear": "2024",
            "PostingDate": "2024-01-20T00:00:00Z",
            "DocumentDate": "2024-01-20T00:00:00Z",
            "MaterialDocumentHeaderText": "GR for PO 4500000001",
            "Items": [
                {
                    "MaterialDocumentItem": "0001",
                    "Material": "MAT-001",
                    "Plant": "1000",
                    "StorageLocation": "0001",
                    "GoodsMovementType": "101",
                    "QuantityInEntryUnit": 10.0,
                    "EntryUnit": "EA",
                    "PurchaseOrder": "4500000001",
                    "PurchaseOrderItem": "00010",
                    "Vendor": "VENDOR001"
                }
            ]
        }
    ]


@pytest.fixture
def sample_delivery_data() -> List[Dict[str, Any]]:
    """Sample SAP Delivery data."""
    return [
        {
            "OutboundDelivery": "8000000001",
            "DeliveryDate": "2024-01-25T00:00:00Z",
            "ActualGoodsMovementDate": "2024-01-25T00:00:00Z",
            "ShipToParty": "CUSTOMER001",
            "ShipToPartyName": "Customer Inc",
            "ShippingPoint": "1000",
            "OverallGrossWeight": 500.0,
            "GrossWeightUnit": "KG",
            "Items": [
                {
                    "DeliveryItem": "000010",
                    "Material": "MAT-001",
                    "ActualDeliveryQuantity": 10.0,
                    "DeliveryQuantityUnit": "EA",
                    "Plant": "1000"
                }
            ]
        }
    ]


@pytest.fixture
def sample_transport_data() -> List[Dict[str, Any]]:
    """Sample SAP Transportation Order data."""
    return [
        {
            "TransportationOrder": "9000000001",
            "TransportationOrderType": "STD",
            "TransportationMode": "02",  # Road
            "TransportationModeCategory": "Road",
            "Carrier": "CARRIER001",
            "CarrierName": "FastShip Logistics",
            "OriginLocation": "1000",
            "DestinationLocation": "2000",
            "PlannedDepartureDate": "2024-01-26T08:00:00Z",
            "PlannedArrivalDate": "2024-01-27T10:00:00Z",
            "TotalDistance": 500.0,
            "DistanceUnit": "KM",
            "TotalWeight": 500.0,
            "WeightUnit": "KG"
        }
    ]


@pytest.fixture
def sample_vendor_data() -> List[Dict[str, Any]]:
    """Sample SAP Vendor Master data."""
    return [
        {
            "Supplier": "VENDOR001",
            "SupplierName": "Acme Corporation",
            "Country": "US",
            "Region": "California",
            "City": "San Francisco",
            "PostalCode": "94105",
            "StreetName": "123 Market Street",
            "PhoneNumber": "+1-415-555-0100",
            "EmailAddress": "contact@acme.example.com"
        },
        {
            "Supplier": "VENDOR002",
            "SupplierName": "Global Supplies Inc",
            "Country": "DE",
            "Region": "Bavaria",
            "City": "Munich",
            "PostalCode": "80331"
        }
    ]


@pytest.fixture
def sample_material_data() -> List[Dict[str, Any]]:
    """Sample SAP Material Master data."""
    return [
        {
            "Material": "MAT-001",
            "MaterialDescription": "Laptop Computer",
            "MaterialType": "FERT",
            "MaterialGroup": "IT Equipment",
            "BaseUnit": "EA",
            "GrossWeight": 2.5,
            "WeightUnit": "KG"
        }
    ]


# OData Response Fixtures
# -----------------------

@pytest.fixture
def mock_odata_response_single():
    """Mock OData response for single entity."""
    return {
        "d": {
            "PurchaseOrder": "4500000001",
            "Vendor": "VENDOR001",
            "VendorName": "Acme Corporation"
        }
    }


@pytest.fixture
def mock_odata_response_collection(sample_po_data):
    """Mock OData response for collection."""
    return {
        "d": {
            "results": sample_po_data
        }
    }


@pytest.fixture
def mock_odata_response_paginated(sample_po_data):
    """Mock OData response with pagination."""
    return {
        "d": {
            "results": sample_po_data[:1],
            "__next": "https://sandbox.api.sap.com/sap/opu/odata/sap/MM_PUR_PO_MAINT_V2_SRV/C_PurchaseOrderTP?$skip=1"
        }
    }


@pytest.fixture
def mock_odata_error_response():
    """Mock OData error response."""
    return {
        "error": {
            "code": "SY/530",
            "message": {
                "lang": "en",
                "value": "Entity not found"
            }
        }
    }


# Redis Mock Fixtures
# -------------------

@pytest.fixture
def mock_redis_client():
    """Mock Redis client for caching."""
    redis_mock = Mock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = False
    redis_mock.ttl.return_value = -1

    return redis_mock


# Celery Mock Fixtures
# --------------------

@pytest.fixture
def mock_celery_app():
    """Mock Celery application."""
    celery_mock = Mock()
    celery_mock.task.return_value = lambda f: f
    celery_mock.send_task.return_value = Mock(id="task-id-123")

    return celery_mock


# Database Mock Fixtures
# ----------------------

@pytest.fixture
def mock_db_session():
    """Mock database session for audit logging."""
    session_mock = Mock()
    session_mock.add.return_value = None
    session_mock.commit.return_value = None
    session_mock.rollback.return_value = None
    session_mock.close.return_value = None

    return session_mock


# Time Fixtures
# -------------

@pytest.fixture
def frozen_time():
    """Freeze time for testing."""
    with freeze_time("2024-01-15 12:00:00"):
        yield datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


# HTTP Mock Fixtures
# ------------------

@pytest.fixture
def mock_responses():
    """Mock HTTP responses using responses library."""
    with responses.RequestsMock() as rsps:
        yield rsps


# Helper Functions
# ----------------

def create_mock_response(status_code: int, json_data: Dict[str, Any] = None, text: str = None):
    """Create mock HTTP response."""
    response = Mock()
    response.status_code = status_code
    response.json.return_value = json_data or {}
    response.text = text or json.dumps(json_data or {})
    response.headers = {"Content-Type": "application/json"}
    return response


# Pytest Configuration
# --------------------

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (slow)"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (fast)"
    )
    config.addinivalue_line(
        "markers", "auth: mark test as authentication test"
    )
    config.addinivalue_line(
        "markers", "client: mark test as client test"
    )
