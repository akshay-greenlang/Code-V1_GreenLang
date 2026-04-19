# -*- coding: utf-8 -*-
"""
Integration tests for SAP S/4HANA OData Connector.

Tests authentication, data extraction, error handling, and rate limiting.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import httpx
from httpx import Response

from ..sap_connector import SAPConnector, SAPConfig


class TestSAPConnector:
    """Test suite for SAP S/4HANA connector."""

    @pytest.fixture
    def mock_credentials(self):
        """Mock SAP credentials."""
        return {
            'base_url': 'https://api.s4hana.example.com',
            'client_id': 'test_client_id',
            'client_secret': 'test_client_secret',
            'oauth_token_url': 'https://auth.s4hana.example.com/oauth/token',
            'company_code': '1000'
        }

    @pytest.fixture
    def mock_oauth_response(self):
        """Mock OAuth token response."""
        return {
            'access_token': 'mock_access_token_123456',
            'token_type': 'Bearer',
            'expires_in': 3600,
            'scope': 'api'
        }

    @pytest.fixture
    def mock_purchase_orders(self):
        """Mock purchase order data from SAP."""
        return {
            'd': {
                'results': [
                    {
                        'PurchaseOrder': '4500000001',
                        'Supplier': '0000100000',
                        'SupplierName': 'Acme Corporation',
                        'CreationDate': '/Date(1640995200000)/',
                        'PurchaseOrderDate': '/Date(1641081600000)/',
                        'TotalNetAmount': '15000.00',
                        'DocumentCurrency': 'USD',
                        'PurchasingProcessingStatus': '02',
                        'CompanyCode': '1000',
                        'Plant': 'PL01',
                        'PurchasingOrganization': 'PO01',
                        'PurchasingGroup': 'PG01',
                        'PaymentTerms': 'NT30',
                        'IncotermsClassification': 'FOB',
                        'to_PurchaseOrderItem': {
                            'results': [
                                {
                                    'PurchaseOrderItem': '00010',
                                    'Material': 'MAT-001',
                                    'PurchaseOrderItemText': 'Raw Material A',
                                    'OrderQuantity': '100',
                                    'OrderPriceUnit': 'EA',
                                    'NetPriceAmount': '15000.00'
                                }
                            ]
                        }
                    },
                    {
                        'PurchaseOrder': '4500000002',
                        'Supplier': '0000100001',
                        'SupplierName': 'Global Supplies Inc',
                        'CreationDate': '/Date(1641081600000)/',
                        'PurchaseOrderDate': '/Date(1641168000000)/',
                        'TotalNetAmount': '25000.00',
                        'DocumentCurrency': 'EUR',
                        'PurchasingProcessingStatus': '03',
                        'CompanyCode': '1000',
                        'Plant': 'PL02',
                        'PurchasingOrganization': 'PO01',
                        'PurchasingGroup': 'PG02',
                        'PaymentTerms': 'NT60',
                        'IncotermsClassification': 'CIF'
                    }
                ]
            }
        }

    @pytest.fixture
    def mock_suppliers(self):
        """Mock supplier data from SAP."""
        return {
            'd': {
                'results': [
                    {
                        'Supplier': '0000100000',
                        'SupplierName': 'Acme Corporation',
                        'TaxNumber1': '12-3456789',
                        'StreetName': '123 Main Street',
                        'CityName': 'New York',
                        'PostalCode': '10001',
                        'Country': 'US',
                        'Region': 'NY',
                        'PhoneNumber1': '+1-212-555-0100',
                        'EmailAddress': 'contact@acme.com',
                        'PaymentTerms': 'NT30',
                        'Industry': 'MANUFACTURING',
                        'CreationDate': '/Date(1577836800000)/'
                    }
                ]
            }
        }

    @patch('httpx.Client')
    def test_connector_initialization(self, mock_client, mock_credentials):
        """Test SAP connector initialization."""
        connector = SAPConnector(mock_credentials)

        assert connector.config.base_url == mock_credentials['base_url']
        assert connector.config.client_id == mock_credentials['client_id']
        assert connector.config.company_code == mock_credentials['company_code']
        assert connector.connected is False
        assert connector.access_token is None

    @patch('httpx.Client')
    def test_authentication_success(self, mock_client_class, mock_credentials, mock_oauth_response):
        """Test successful OAuth2 authentication."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock OAuth token response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_oauth_response
        mock_client.post.return_value = mock_response

        connector = SAPConnector(mock_credentials)
        token = connector._authenticate()

        assert token == 'mock_access_token_123456'
        assert connector.access_token == 'mock_access_token_123456'
        assert connector.token_expires_at is not None

        # Verify OAuth request
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert str(mock_credentials['oauth_token_url']) in str(call_args[0][0])

    @patch('httpx.Client')
    def test_authentication_failure(self, mock_client_class, mock_credentials):
        """Test OAuth2 authentication failure."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock failed OAuth response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = 'Invalid client credentials'
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="401 Unauthorized",
            request=MagicMock(),
            response=mock_response
        )
        mock_client.post.return_value = mock_response

        connector = SAPConnector(mock_credentials)

        with pytest.raises(Exception) as exc_info:
            connector._authenticate()

        assert 'SAP OAuth2 failed' in str(exc_info.value)

    @patch('httpx.Client')
    def test_connection_success(self, mock_client_class, mock_credentials, mock_oauth_response):
        """Test successful connection to SAP."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock OAuth response
        oauth_response = MagicMock()
        oauth_response.status_code = 200
        oauth_response.json.return_value = mock_oauth_response

        # Mock test query response
        test_response = MagicMock()
        test_response.status_code = 200
        test_response.json.return_value = {'d': {'results': []}}

        mock_client.post.return_value = oauth_response
        mock_client.request.return_value = test_response

        connector = SAPConnector(mock_credentials)
        result = connector.connect()

        assert result is True
        assert connector.connected is True

    @patch('httpx.Client')
    def test_query_purchase_orders(self, mock_client_class, mock_credentials,
                                  mock_oauth_response, mock_purchase_orders):
        """Test querying purchase orders from SAP."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock OAuth response
        oauth_response = MagicMock()
        oauth_response.status_code = 200
        oauth_response.json.return_value = mock_oauth_response
        mock_client.post.return_value = oauth_response

        # Mock PO query response
        po_response = MagicMock()
        po_response.status_code = 200
        po_response.json.return_value = mock_purchase_orders
        mock_client.request.return_value = po_response

        connector = SAPConnector(mock_credentials)
        connector.connected = True

        # Query purchase orders
        pos = connector.get_purchase_orders('2022-01-01', '2022-01-31', plant_codes=['PL01', 'PL02'])

        assert len(pos) == 2
        assert pos[0]['document_number'] == '4500000001'
        assert pos[0]['supplier_name'] == 'Acme Corporation'
        assert pos[0]['total_amount'] == 15000.00
        assert pos[0]['currency'] == 'USD'
        assert pos[1]['document_number'] == '4500000002'

    @patch('httpx.Client')
    def test_query_with_pagination(self, mock_client_class, mock_credentials, mock_oauth_response):
        """Test pagination handling for large datasets."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock OAuth response
        oauth_response = MagicMock()
        oauth_response.status_code = 200
        oauth_response.json.return_value = mock_oauth_response
        mock_client.post.return_value = oauth_response

        # Mock paginated responses
        page1 = {
            'd': {
                'results': [
                    {'PurchaseOrder': f'450000000{i}', 'Supplier': f'000010000{i}'}
                    for i in range(1000)
                ]
            }
        }

        page2 = {
            'd': {
                'results': [
                    {'PurchaseOrder': f'450000100{i}', 'Supplier': f'000010100{i}'}
                    for i in range(500)
                ]
            }
        }

        page3 = {
            'd': {'results': []}
        }

        responses = [
            MagicMock(status_code=200, json=MagicMock(return_value=page1)),
            MagicMock(status_code=200, json=MagicMock(return_value=page2)),
            MagicMock(status_code=200, json=MagicMock(return_value=page3))
        ]

        mock_client.request.side_effect = responses

        connector = SAPConnector(mock_credentials)
        connector.connected = True

        query = {
            'entity_type': 'purchase_orders',
            'start_date': '2022-01-01',
            'end_date': '2022-01-31'
        }

        results = connector.query(query)

        assert len(results) == 1500
        assert mock_client.request.call_count == 3

    @patch('httpx.Client')
    @patch('time.sleep')
    def test_rate_limiting(self, mock_sleep, mock_client_class, mock_credentials, mock_oauth_response):
        """Test rate limiting enforcement."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock OAuth response
        oauth_response = MagicMock()
        oauth_response.status_code = 200
        oauth_response.json.return_value = mock_oauth_response
        mock_client.post.return_value = oauth_response

        # Mock API responses
        api_response = MagicMock()
        api_response.status_code = 200
        api_response.json.return_value = {'d': {'results': []}}
        mock_client.request.return_value = api_response

        connector = SAPConnector(mock_credentials)
        connector.connected = True
        connector.config.rate_limit_requests_per_minute = 2  # Low limit for testing

        # Make multiple requests to trigger rate limiting
        for _ in range(3):
            connector._make_request('GET', 'https://api.example.com/test')

        # Verify rate limiting was enforced
        mock_sleep.assert_called()

    @patch('httpx.Client')
    def test_retry_on_failure(self, mock_client_class, mock_credentials, mock_oauth_response):
        """Test retry logic with exponential backoff."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock OAuth response
        oauth_response = MagicMock()
        oauth_response.status_code = 200
        oauth_response.json.return_value = mock_oauth_response
        mock_client.post.return_value = oauth_response

        # Mock API responses - fail twice, then succeed
        fail_response = MagicMock()
        fail_response.status_code = 503
        fail_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="503 Service Unavailable",
            request=MagicMock(),
            response=fail_response
        )

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {'d': {'results': []}}
        success_response.raise_for_status = MagicMock()

        mock_client.request.side_effect = [fail_response, fail_response, success_response]

        connector = SAPConnector(mock_credentials)
        connector.connected = True

        # Should retry and eventually succeed
        with patch('time.sleep'):  # Speed up test
            response = connector._make_request('GET', 'https://api.example.com/test')

        assert response.status_code == 200
        assert mock_client.request.call_count == 3

    @patch('httpx.Client')
    def test_transform_purchase_order(self, mock_client_class, mock_credentials):
        """Test transformation of SAP PO to standard format."""
        connector = SAPConnector(mock_credentials)

        sap_po = {
            'PurchaseOrder': '4500000001',
            'Supplier': '0000100000',
            'SupplierName': 'Acme Corporation',
            'CreationDate': '/Date(1640995200000)/',
            'PurchaseOrderDate': '/Date(1641081600000)/',
            'TotalNetAmount': '15000.00',
            'DocumentCurrency': 'USD',
            'PurchasingProcessingStatus': '02',
            'CompanyCode': '1000',
            'Plant': 'PL01',
            'PurchasingOrganization': 'PO01',
            'PurchasingGroup': 'PG01',
            'PaymentTerms': 'NT30',
            'IncotermsClassification': 'FOB'
        }

        transformed = connector._transform_purchase_order(sap_po)

        assert transformed['id'] == '4500000001'
        assert transformed['type'] == 'purchase_order'
        assert transformed['source_system'] == 'SAP'
        assert transformed['supplier_id'] == '0000100000'
        assert transformed['supplier_name'] == 'Acme Corporation'
        assert transformed['total_amount'] == 15000.00
        assert transformed['currency'] == 'USD'
        assert transformed['plant'] == 'PL01'
        assert 'metadata' in transformed
        assert 'extracted_at' in transformed

    @patch('httpx.Client')
    def test_disconnect(self, mock_client_class, mock_credentials):
        """Test disconnection from SAP."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        connector = SAPConnector(mock_credentials)
        connector.connected = True

        connector.disconnect()

        assert connector.connected is False
        mock_client.close.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])