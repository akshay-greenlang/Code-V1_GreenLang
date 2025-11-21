# -*- coding: utf-8 -*-
"""
Integration tests for Workday REST API and RAAS Connector.

Tests OAuth2 authentication, REST API queries, RAAS reports, and spend analytics.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import httpx

from ..workday_connector import WorkdayConnector, WorkdayConfig


class TestWorkdayConnector:
    """Test suite for Workday connector."""

    @pytest.fixture
    def mock_credentials(self):
        """Mock Workday credentials."""
        return {
            'base_url': 'https://wd2-impl-services1.workday.com',
            'tenant': 'test_tenant',
            'username': 'test_user',
            'password': 'test_password',
            'client_id': 'test_client_id',
            'client_secret': 'test_client_secret',
            'refresh_token': None
        }

    @pytest.fixture
    def mock_oauth_response(self):
        """Mock OAuth token response."""
        return {
            'access_token': 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...',
            'token_type': 'Bearer',
            'expires_in': 3600,
            'refresh_token': 'refresh_token_123456',
            'scope': 'api'
        }

    @pytest.fixture
    def mock_suppliers(self):
        """Mock supplier data from Workday."""
        return {
            'data': [
                {
                    'id': 'SUPPLIER-001',
                    'supplierID': 'SUP001',
                    'supplierName': 'Global Tech Partners',
                    'supplierCategory': 'Technology',
                    'taxID': '98-7654321',
                    'dunsNumber': '987654321',
                    'supplierStatus': 'ACTIVE',
                    'vendorType': 'SUPPLIER',
                    'paymentTerms': {
                        'id': 'NET30',
                        'descriptor': 'Net 30 Days'
                    },
                    'defaultCurrency': 'USD',
                    'addresses': [
                        {
                            'addressID': 'ADDR-001',
                            'addressLine1': '456 Tech Boulevard',
                            'city': 'Seattle',
                            'state': 'WA',
                            'postalCode': '98101',
                            'country': 'USA'
                        }
                    ],
                    'spendCategories': [
                        'IT_HARDWARE',
                        'SOFTWARE_LICENSES'
                    ],
                    'supplierGroup': {
                        'id': 'GROUP-TECH',
                        'descriptor': 'Technology Suppliers'
                    },
                    'createdOn': '2021-01-15T10:00:00Z',
                    'lastUpdated': '2022-01-15T14:30:00Z'
                },
                {
                    'id': 'SUPPLIER-002',
                    'supplierID': 'SUP002',
                    'supplierName': 'Office Essentials Inc',
                    'supplierCategory': 'Office Supplies',
                    'taxID': '12-9876543',
                    'supplierStatus': 'ACTIVE',
                    'vendorType': 'SUPPLIER',
                    'defaultCurrency': 'USD',
                    'createdOn': '2021-02-01T09:00:00Z',
                    'lastUpdated': '2022-01-10T11:00:00Z'
                }
            ],
            'total': 2
        }

    @pytest.fixture
    def mock_purchase_orders(self):
        """Mock purchase order data from Workday."""
        return {
            'data': [
                {
                    'id': 'PO-001',
                    'purchaseOrderNumber': 'PO-2022-0001',
                    'supplier': {
                        'id': 'SUPPLIER-001',
                        'descriptor': 'Global Tech Partners'
                    },
                    'requester': {
                        'id': 'WORKER-001',
                        'descriptor': 'John Smith'
                    },
                    'buyer': {
                        'id': 'BUYER-001',
                        'descriptor': 'Jane Doe'
                    },
                    'purchaseOrderDate': '2022-01-15T10:00:00Z',
                    'dueDate': '2022-02-15T00:00:00Z',
                    'totalAmount': {
                        'value': 75000.00,
                        'currency': 'USD'
                    },
                    'purchaseOrderStatus': 'APPROVED',
                    'company': {
                        'id': 'COMPANY-001',
                        'descriptor': 'Acme Corporation'
                    },
                    'costCenter': {
                        'id': 'CC-IT',
                        'descriptor': 'Information Technology'
                    },
                    'purchaseOrderLines': [
                        {
                            'lineNumber': 1,
                            'itemDescription': 'Dell Laptops',
                            'quantity': 25,
                            'unitCost': 1500.00,
                            'totalCost': 37500.00
                        },
                        {
                            'lineNumber': 2,
                            'itemDescription': 'Software Licenses',
                            'quantity': 25,
                            'unitCost': 1500.00,
                            'totalCost': 37500.00
                        }
                    ],
                    'spendCategory': {
                        'id': 'SPEND-IT',
                        'descriptor': 'IT Equipment and Software'
                    }
                }
            ],
            'total': 1
        }

    @pytest.fixture
    def mock_spend_analytics(self):
        """Mock spend analytics data from Workday."""
        return {
            'data': [
                {
                    'spendCategory': 'IT Equipment',
                    'totalSpend': 500000.00,
                    'currency': 'USD',
                    'period': '2022-Q1',
                    'supplierCount': 15,
                    'transactionCount': 245,
                    'topSuppliers': [
                        {
                            'supplierName': 'Global Tech Partners',
                            'spend': 150000.00
                        },
                        {
                            'supplierName': 'Tech Solutions Inc',
                            'spend': 100000.00
                        }
                    ]
                }
            ]
        }

    @pytest.fixture
    def mock_raas_report(self):
        """Mock RAAS report response."""
        return {
            'Report_Entry': [
                {
                    'WID': 'WID-001',
                    'Supplier_Name': 'Global Tech Partners',
                    'Total_Spend': 1500000.00,
                    'Currency': 'USD',
                    'Period': '2022',
                    'Purchase_Order_Count': 45,
                    'Invoice_Count': 52,
                    'Average_Payment_Days': 28
                },
                {
                    'WID': 'WID-002',
                    'Supplier_Name': 'Office Essentials Inc',
                    'Total_Spend': 250000.00,
                    'Currency': 'USD',
                    'Period': '2022',
                    'Purchase_Order_Count': 120,
                    'Invoice_Count': 115,
                    'Average_Payment_Days': 35
                }
            ]
        }

    @patch('httpx.Client')
    def test_connector_initialization(self, mock_client, mock_credentials):
        """Test Workday connector initialization."""
        connector = WorkdayConnector(mock_credentials)

        assert connector.config.base_url == mock_credentials['base_url']
        assert connector.config.tenant == mock_credentials['tenant']
        assert connector.config.username == mock_credentials['username']
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

        connector = WorkdayConnector(mock_credentials)
        token = connector._authenticate()

        assert token == mock_oauth_response['access_token']
        assert connector.access_token == mock_oauth_response['access_token']
        assert connector.token_expires_at is not None
        assert connector.config.refresh_token == mock_oauth_response['refresh_token']

    @patch('httpx.Client')
    def test_refresh_token_flow(self, mock_client_class, mock_credentials, mock_oauth_response):
        """Test OAuth2 refresh token flow."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Set up connector with existing refresh token
        mock_credentials['refresh_token'] = 'existing_refresh_token'
        connector = WorkdayConnector(mock_credentials)

        # Mock refresh token response
        refresh_response = {
            'access_token': 'new_access_token_789',
            'token_type': 'Bearer',
            'expires_in': 3600,
            'refresh_token': 'new_refresh_token_789'
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = refresh_response
        mock_client.post.return_value = mock_response

        token = connector._refresh_access_token()

        assert token == 'new_access_token_789'
        assert connector.access_token == 'new_access_token_789'
        assert connector.config.refresh_token == 'new_refresh_token_789'

    @patch('httpx.Client')
    def test_connection_success(self, mock_client_class, mock_credentials, mock_oauth_response):
        """Test successful connection to Workday."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock OAuth response
        oauth_response = MagicMock()
        oauth_response.status_code = 200
        oauth_response.json.return_value = mock_oauth_response
        mock_client.post.return_value = oauth_response

        # Mock test query response
        test_response = MagicMock()
        test_response.status_code = 200
        test_response.json.return_value = {'data': []}
        mock_client.request.return_value = test_response

        connector = WorkdayConnector(mock_credentials)
        result = connector.connect()

        assert result is True
        assert connector.connected is True

    @patch('httpx.Client')
    def test_query_suppliers(self, mock_client_class, mock_credentials,
                           mock_oauth_response, mock_suppliers):
        """Test querying suppliers from Workday."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock OAuth response
        oauth_response = MagicMock()
        oauth_response.status_code = 200
        oauth_response.json.return_value = mock_oauth_response
        mock_client.post.return_value = oauth_response

        # Mock supplier query response
        supplier_response = MagicMock()
        supplier_response.status_code = 200
        supplier_response.json.return_value = mock_suppliers
        mock_client.request.return_value = supplier_response

        connector = WorkdayConnector(mock_credentials)
        connector.connected = True

        # Query suppliers
        suppliers = connector.get_suppliers(status='ACTIVE')

        assert len(suppliers) == 2
        assert suppliers[0]['supplier_id'] == 'SUP001'
        assert suppliers[0]['supplier_name'] == 'Global Tech Partners'
        assert suppliers[0]['tax_id'] == '98-7654321'
        assert suppliers[0]['status'] == 'ACTIVE'
        assert suppliers[1]['supplier_id'] == 'SUP002'

    @patch('httpx.Client')
    def test_query_purchase_orders(self, mock_client_class, mock_credentials,
                                  mock_oauth_response, mock_purchase_orders):
        """Test querying purchase orders from Workday."""
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

        connector = WorkdayConnector(mock_credentials)
        connector.connected = True

        # Query purchase orders
        pos = connector.get_purchase_orders('2022-01-01', '2022-01-31', company='Acme Corporation')

        assert len(pos) == 1
        assert pos[0]['po_number'] == 'PO-2022-0001'
        assert pos[0]['supplier_name'] == 'Global Tech Partners'
        assert pos[0]['total_amount'] == 75000.00
        assert pos[0]['currency'] == 'USD'
        assert pos[0]['status'] == 'APPROVED'
        assert len(pos[0]['lines']) == 2

    @patch('httpx.Client')
    def test_query_spend_analytics(self, mock_client_class, mock_credentials,
                                  mock_oauth_response, mock_spend_analytics):
        """Test querying spend analytics from Workday."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock OAuth response
        oauth_response = MagicMock()
        oauth_response.status_code = 200
        oauth_response.json.return_value = mock_oauth_response
        mock_client.post.return_value = oauth_response

        # Mock spend analytics response
        analytics_response = MagicMock()
        analytics_response.status_code = 200
        analytics_response.json.return_value = mock_spend_analytics
        mock_client.request.return_value = analytics_response

        connector = WorkdayConnector(mock_credentials)
        connector.connected = True

        # Query spend analytics
        analytics = connector.get_spend_analytics('2022-01-01', '2022-03-31', spend_category='IT Equipment')

        assert len(analytics) == 1
        # Raw data is returned without transformation for spend_analytics
        assert analytics[0]['spendCategory'] == 'IT Equipment'
        assert analytics[0]['totalSpend'] == 500000.00

    @patch('httpx.Client')
    def test_raas_report_query(self, mock_client_class, mock_credentials,
                              mock_oauth_response, mock_raas_report):
        """Test querying RAAS custom reports."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock OAuth response
        oauth_response = MagicMock()
        oauth_response.status_code = 200
        oauth_response.json.return_value = mock_oauth_response
        mock_client.post.return_value = oauth_response

        # Mock RAAS report response
        raas_response = MagicMock()
        raas_response.status_code = 200
        raas_response.json.return_value = mock_raas_report
        mock_client.request.return_value = raas_response

        connector = WorkdayConnector(mock_credentials)
        connector.connected = True

        # Query RAAS report (unknown entity type triggers RAAS)
        query = {
            'entity_type': 'supplier_spend_report',
            'start_date': '2022-01-01',
            'end_date': '2022-12-31',
            'filters': {'Spend_Threshold': 100000}
        }

        results = connector.query(query)

        assert len(results) == 2
        assert results[0]['type'] == 'raas_report'
        assert results[0]['report_name'] == 'supplier_spend_report'
        assert results[0]['data']['Supplier_Name'] == 'Global Tech Partners'
        assert results[0]['data']['Total_Spend'] == 1500000.00

    @patch('httpx.Client')
    def test_pagination_handling(self, mock_client_class, mock_credentials, mock_oauth_response):
        """Test pagination for large result sets."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock OAuth response
        oauth_response = MagicMock()
        oauth_response.status_code = 200
        oauth_response.json.return_value = mock_oauth_response
        mock_client.post.return_value = oauth_response

        # Mock paginated responses
        page1 = {
            'data': [{'id': f'SUP-{i}', 'supplierID': f'S{i:03d}'} for i in range(100)],
            'total': 250
        }

        page2 = {
            'data': [{'id': f'SUP-{i}', 'supplierID': f'S{i:03d}'} for i in range(100, 200)],
            'total': 250
        }

        page3 = {
            'data': [{'id': f'SUP-{i}', 'supplierID': f'S{i:03d}'} for i in range(200, 250)],
            'total': 250
        }

        responses = [
            MagicMock(status_code=200, json=MagicMock(return_value=page1)),
            MagicMock(status_code=200, json=MagicMock(return_value=page2)),
            MagicMock(status_code=200, json=MagicMock(return_value=page3))
        ]

        mock_client.request.side_effect = responses

        connector = WorkdayConnector(mock_credentials)
        connector.connected = True

        query = {'entity_type': 'suppliers'}
        results = connector.query(query)

        assert len(results) == 250
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
        api_response.json.return_value = {'data': []}
        api_response.raise_for_status = MagicMock()
        mock_client.request.return_value = api_response

        connector = WorkdayConnector(mock_credentials)
        connector.connected = True
        connector.config.rate_limit_requests_per_minute = 2  # Low limit for testing

        # Make multiple requests to trigger rate limiting
        for _ in range(3):
            connector._make_request('GET', 'https://api.example.com/test')

        # Verify rate limiting was enforced
        mock_sleep.assert_called()

    @patch('httpx.Client')
    def test_transform_supplier(self, mock_client_class, mock_credentials):
        """Test transformation of Workday supplier to standard format."""
        connector = WorkdayConnector(mock_credentials)

        workday_supplier = {
            'id': 'SUPPLIER-001',
            'supplierID': 'SUP001',
            'supplierName': 'Global Tech Partners',
            'supplierCategory': 'Technology',
            'taxID': '98-7654321',
            'dunsNumber': '987654321',
            'supplierStatus': 'ACTIVE',
            'vendorType': 'SUPPLIER',
            'paymentTerms': {'descriptor': 'Net 30 Days'},
            'defaultCurrency': 'USD',
            'supplierGroup': {'descriptor': 'Technology Suppliers'},
            'createdOn': '2021-01-15T10:00:00Z',
            'lastUpdated': '2022-01-15T14:30:00Z'
        }

        transformed = connector._transform_supplier(workday_supplier)

        assert transformed['id'] == 'SUPPLIER-001'
        assert transformed['type'] == 'supplier'
        assert transformed['source_system'] == 'Workday'
        assert transformed['supplier_id'] == 'SUP001'
        assert transformed['supplier_name'] == 'Global Tech Partners'
        assert transformed['tax_id'] == '98-7654321'
        assert transformed['status'] == 'ACTIVE'
        assert transformed['payment_terms'] == 'Net 30 Days'
        assert 'metadata' in transformed
        assert 'extracted_at' in transformed

    @patch('httpx.Client')
    def test_disconnect(self, mock_client_class, mock_credentials):
        """Test disconnection from Workday."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        connector = WorkdayConnector(mock_credentials)
        connector.connected = True

        connector.disconnect()

        assert connector.connected is False
        mock_client.close.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])