# -*- coding: utf-8 -*-
"""
Integration tests for Oracle ERP Cloud REST API Connector.

Tests authentication, data extraction, pagination, and error handling.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import httpx
import base64

from ..oracle_connector import OracleConnector, OracleConfig


class TestOracleConnector:
    """Test suite for Oracle ERP Cloud connector."""

    @pytest.fixture
    def mock_credentials(self):
        """Mock Oracle credentials."""
        return {
            'base_url': 'https://example.fa.us2.oraclecloud.com',
            'username': 'test_user',
            'password': 'test_password',
            'tenant_name': 'test_tenant'
        }

    @pytest.fixture
    def mock_purchase_orders(self):
        """Mock purchase order data from Oracle."""
        return {
            'items': [
                {
                    'POHeaderId': 300000123456789,
                    'OrderNumber': 'PO-2022-001',
                    'VendorId': 300000987654321,
                    'Supplier': 'Tech Supplies Inc.',
                    'SupplierSite': 'MAIN',
                    'CreationDate': '2022-01-15T10:30:00Z',
                    'ApprovedDate': '2022-01-15T14:00:00Z',
                    'Total': 45000.00,
                    'CurrencyCode': 'USD',
                    'StatusCode': 'APPROVED',
                    'Buyer': 'John Smith',
                    'ProcurementBU': 'US_BU',
                    'RequisitioningBU': 'US_BU',
                    'BillToLocation': 'New York Office',
                    'ShipToLocation': 'New York Warehouse',
                    'PaymentTerms': 'Net 30',
                    'FreightTerms': 'FOB Destination',
                    'Carrier': 'FedEx',
                    'lines': [
                        {
                            'LineNum': 1,
                            'ItemDescription': 'Laptop Computer',
                            'Quantity': 10,
                            'UnitPrice': 1500.00,
                            'Amount': 15000.00
                        },
                        {
                            'LineNum': 2,
                            'ItemDescription': 'Monitor',
                            'Quantity': 10,
                            'UnitPrice': 500.00,
                            'Amount': 5000.00
                        }
                    ]
                },
                {
                    'POHeaderId': 300000123456790,
                    'OrderNumber': 'PO-2022-002',
                    'VendorId': 300000987654322,
                    'Supplier': 'Office Supplies Co.',
                    'SupplierSite': 'HQ',
                    'CreationDate': '2022-01-20T09:00:00Z',
                    'ApprovedDate': '2022-01-20T16:30:00Z',
                    'Total': 12500.00,
                    'CurrencyCode': 'USD',
                    'StatusCode': 'OPEN',
                    'Buyer': 'Jane Doe',
                    'ProcurementBU': 'US_BU',
                    'RequisitioningBU': 'US_BU',
                    'BillToLocation': 'San Francisco Office',
                    'ShipToLocation': 'San Francisco Warehouse',
                    'PaymentTerms': 'Net 60',
                    'FreightTerms': 'FOB Origin',
                    'Carrier': 'UPS'
                }
            ],
            'hasMore': False,
            'totalResults': 2
        }

    @pytest.fixture
    def mock_suppliers(self):
        """Mock supplier data from Oracle."""
        return {
            'items': [
                {
                    'SupplierId': 300000987654321,
                    'SupplierNumber': 'SUP-001',
                    'Supplier': 'Tech Supplies Inc.',
                    'AlternateName': 'TSI',
                    'TaxOrganizationType': 'CORPORATION',
                    'TaxpayerIdentificationNumber': '12-3456789',
                    'DUNSNumber': '123456789',
                    'Status': 'ACTIVE',
                    'SupplierType': 'SUPPLIER',
                    'BusinessRelationship': 'PROSPECTIVE',
                    'CreationDate': '2020-01-01T00:00:00Z',
                    'addresses': [
                        {
                            'AddressName': 'HEADQUARTERS',
                            'Address1': '123 Tech Street',
                            'City': 'San Jose',
                            'State': 'CA',
                            'PostalCode': '95110',
                            'Country': 'US'
                        }
                    ],
                    'sites': [
                        {
                            'SupplierSite': 'MAIN',
                            'PurchasingSite': 'Y',
                            'PaySite': 'Y'
                        }
                    ]
                }
            ],
            'hasMore': False
        }

    @pytest.fixture
    def mock_inventory_transactions(self):
        """Mock inventory transaction data from Oracle."""
        return {
            'items': [
                {
                    'TransactionId': 400000111222333,
                    'TransactionNumber': 'INV-2022-00001',
                    'TransactionDate': '2022-01-15T14:30:00Z',
                    'TransactionType': 'Purchase order receipt',
                    'ItemNumber': 'ITEM-001',
                    'ItemDescription': 'Raw Material A',
                    'TransactionQuantity': 1000,
                    'TransactionUOM': 'EA',
                    'OrganizationCode': 'ORG1',
                    'Subinventory': 'MAIN',
                    'Locator': 'A.1.1',
                    'LotNumber': 'LOT123',
                    'TransactionSourceType': 'Purchase order',
                    'SourceReference': 'PO-2022-001',
                    'PONumber': 'PO-2022-001',
                    'ReceiptNumber': 'REC-001'
                }
            ],
            'hasMore': False
        }

    @patch('httpx.Client')
    def test_connector_initialization(self, mock_client, mock_credentials):
        """Test Oracle connector initialization."""
        connector = OracleConnector(mock_credentials)

        assert connector.config.base_url == mock_credentials['base_url']
        assert connector.config.username == mock_credentials['username']
        assert connector.config.tenant_name == mock_credentials['tenant_name']
        assert connector.connected is False
        assert connector.basic_auth is not None

    @patch('httpx.Client')
    def test_basic_auth_initialization(self, mock_client_class, mock_credentials):
        """Test basic authentication header creation."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        connector = OracleConnector(mock_credentials)

        # Verify basic auth header is correctly formatted
        expected_creds = f"{mock_credentials['username']}:{mock_credentials['password']}"
        expected_auth = f"Basic {base64.b64encode(expected_creds.encode()).decode()}"
        assert connector.basic_auth == expected_auth

    @patch('httpx.Client')
    def test_connection_success(self, mock_client_class, mock_credentials):
        """Test successful connection to Oracle."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock test query response
        test_response = MagicMock()
        test_response.status_code = 200
        test_response.json.return_value = {'items': []}
        mock_client.request.return_value = test_response

        connector = OracleConnector(mock_credentials)
        result = connector.connect()

        assert result is True
        assert connector.connected is True

    @patch('httpx.Client')
    def test_connection_failure(self, mock_client_class, mock_credentials):
        """Test failed connection to Oracle."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock failed response
        fail_response = MagicMock()
        fail_response.status_code = 401
        fail_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="401 Unauthorized",
            request=MagicMock(),
            response=fail_response
        )
        mock_client.request.return_value = fail_response

        connector = OracleConnector(mock_credentials)
        result = connector.connect()

        assert result is False
        assert connector.connected is False

    @patch('httpx.Client')
    def test_query_purchase_orders(self, mock_client_class, mock_credentials, mock_purchase_orders):
        """Test querying purchase orders from Oracle."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock PO query response
        po_response = MagicMock()
        po_response.status_code = 200
        po_response.json.return_value = mock_purchase_orders
        mock_client.request.return_value = po_response

        connector = OracleConnector(mock_credentials)
        connector.connected = True

        # Query purchase orders
        pos = connector.get_purchase_orders('2022-01-01', '2022-01-31', business_unit='US_BU')

        assert len(pos) == 2
        assert pos[0]['document_number'] == 'PO-2022-001'
        assert pos[0]['supplier_name'] == 'Tech Supplies Inc.'
        assert pos[0]['total_amount'] == 45000.00
        assert pos[0]['currency'] == 'USD'
        assert pos[0]['business_unit'] == 'US_BU'
        assert pos[1]['document_number'] == 'PO-2022-002'

    @patch('httpx.Client')
    def test_query_with_pagination(self, mock_client_class, mock_credentials):
        """Test pagination handling for large datasets."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock paginated responses
        page1 = {
            'items': [
                {'POHeaderId': i, 'OrderNumber': f'PO-{i}'} for i in range(500)
            ],
            'hasMore': True
        }

        page2 = {
            'items': [
                {'POHeaderId': i, 'OrderNumber': f'PO-{i}'} for i in range(500, 1000)
            ],
            'hasMore': True
        }

        page3 = {
            'items': [
                {'POHeaderId': i, 'OrderNumber': f'PO-{i}'} for i in range(1000, 1200)
            ],
            'hasMore': False
        }

        responses = [
            MagicMock(status_code=200, json=MagicMock(return_value=page1)),
            MagicMock(status_code=200, json=MagicMock(return_value=page2)),
            MagicMock(status_code=200, json=MagicMock(return_value=page3))
        ]

        mock_client.request.side_effect = responses

        connector = OracleConnector(mock_credentials)
        connector.connected = True

        query = {
            'entity_type': 'purchase_orders',
            'start_date': '2022-01-01',
            'end_date': '2022-01-31'
        }

        results = connector.query(query)

        assert len(results) == 1200
        assert mock_client.request.call_count == 3

    @patch('httpx.Client')
    @patch('time.sleep')
    def test_rate_limiting_429_response(self, mock_sleep, mock_client_class, mock_credentials):
        """Test handling of 429 rate limit response."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock 429 response followed by success
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {'Retry-After': '5'}

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {'items': []}
        success_response.raise_for_status = MagicMock()

        mock_client.request.side_effect = [rate_limit_response, success_response]

        connector = OracleConnector(mock_credentials)
        connector.connected = True

        response = connector._make_request('GET', 'https://api.example.com/test')

        assert response.status_code == 200
        mock_sleep.assert_called_with(5)

    @patch('httpx.Client')
    def test_query_suppliers(self, mock_client_class, mock_credentials, mock_suppliers):
        """Test querying suppliers from Oracle."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock supplier query response
        supplier_response = MagicMock()
        supplier_response.status_code = 200
        supplier_response.json.return_value = mock_suppliers
        mock_client.request.return_value = supplier_response

        connector = OracleConnector(mock_credentials)
        connector.connected = True

        # Query suppliers
        suppliers = connector.get_suppliers(status='ACTIVE')

        assert len(suppliers) == 1
        assert suppliers[0]['supplier_number'] == 'SUP-001'
        assert suppliers[0]['supplier_name'] == 'Tech Supplies Inc.'
        assert suppliers[0]['duns_number'] == '123456789'
        assert suppliers[0]['status'] == 'ACTIVE'

    @patch('httpx.Client')
    def test_query_inventory_transactions(self, mock_client_class, mock_credentials,
                                         mock_inventory_transactions):
        """Test querying inventory transactions from Oracle."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock inventory query response
        inv_response = MagicMock()
        inv_response.status_code = 200
        inv_response.json.return_value = mock_inventory_transactions
        mock_client.request.return_value = inv_response

        connector = OracleConnector(mock_credentials)
        connector.connected = True

        # Query inventory transactions
        transactions = connector.get_inventory_transactions('2022-01-01', '2022-01-31', 'ORG1')

        assert len(transactions) == 1
        assert transactions[0]['transaction_number'] == 'INV-2022-00001'
        assert transactions[0]['item_number'] == 'ITEM-001'
        assert transactions[0]['quantity'] == 1000
        assert transactions[0]['organization'] == 'ORG1'

    @patch('httpx.Client')
    def test_transform_purchase_order(self, mock_client_class, mock_credentials):
        """Test transformation of Oracle PO to standard format."""
        connector = OracleConnector(mock_credentials)

        oracle_po = {
            'POHeaderId': 300000123456789,
            'OrderNumber': 'PO-2022-001',
            'VendorId': 300000987654321,
            'Supplier': 'Tech Supplies Inc.',
            'SupplierSite': 'MAIN',
            'CreationDate': '2022-01-15T10:30:00Z',
            'ApprovedDate': '2022-01-15T14:00:00Z',
            'Total': 45000.00,
            'CurrencyCode': 'USD',
            'StatusCode': 'APPROVED',
            'Buyer': 'John Smith',
            'ProcurementBU': 'US_BU',
            'PaymentTerms': 'Net 30',
            'FreightTerms': 'FOB Destination'
        }

        transformed = connector._transform_purchase_order(oracle_po)

        assert transformed['id'] == 300000123456789
        assert transformed['type'] == 'purchase_order'
        assert transformed['source_system'] == 'Oracle'
        assert transformed['document_number'] == 'PO-2022-001'
        assert transformed['supplier_name'] == 'Tech Supplies Inc.'
        assert transformed['total_amount'] == 45000.00
        assert transformed['currency'] == 'USD'
        assert transformed['business_unit'] == 'US_BU'
        assert 'metadata' in transformed
        assert 'extracted_at' in transformed

    @patch('httpx.Client')
    def test_build_query_params_with_filters(self, mock_client_class, mock_credentials):
        """Test building Oracle query parameters with filters."""
        connector = OracleConnector(mock_credentials)

        query = {
            'start_date': '2022-01-01',
            'end_date': '2022-01-31',
            'filters': {
                'StatusCode': 'APPROVED',
                'ProcurementBU': ['US_BU', 'EU_BU']
            },
            'fields': ['OrderNumber', 'Total', 'Supplier'],
            'order_by': 'CreationDate:desc'
        }

        params = connector._build_query_params(query)

        assert 'CreationDate >= \'2022-01-01\'' in params['q']
        assert 'CreationDate <= \'2022-01-31\'' in params['q']
        assert 'StatusCode = \'APPROVED\'' in params['q']
        assert 'ProcurementBU in (\'US_BU\',\'EU_BU\')' in params['q']
        assert params['fields'] == 'OrderNumber,Total,Supplier'
        assert params['orderBy'] == 'CreationDate:desc'

    @patch('httpx.Client')
    def test_disconnect(self, mock_client_class, mock_credentials):
        """Test disconnection from Oracle."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        connector = OracleConnector(mock_credentials)
        connector.connected = True

        connector.disconnect()

        assert connector.connected is False
        mock_client.close.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])