# -*- coding: utf-8 -*-
"""
Integration tests for procurement system connector.

Tests the ProcurementSystemConnector integration:
- ERP system connectivity (SAP, Oracle)
- Purchase order management
- Inventory synchronization
- Supplier data retrieval
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from decimal import Decimal

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integrations.procurement_system_connector import (
    ProcurementSystemConnector,
    ProcurementConfig,
    PurchaseOrder,
    SupplierInfo,
    InventoryRecord
)


class TestProcurementSystemConnector:
    """Test suite for ProcurementSystemConnector integration."""

    @pytest.fixture
    def sap_config(self):
        """SAP ERP configuration."""
        return ProcurementConfig(
            erp_type='sap',
            host='sap.example.com',
            port=443,
            client='100',
            username='integration_user',
            password='secure_password',
            timeout_seconds=60
        )

    @pytest.fixture
    def connector(self, sap_config):
        """Create connector instance."""
        return ProcurementSystemConnector(sap_config)

    def test_connector_initialization(self, connector):
        """Test connector initializes correctly."""
        assert connector is not None
        assert connector.erp_type == 'sap'
        assert connector.is_connected is False

    @patch('integrations.procurement_system_connector.SAPClient')
    def test_sap_connection(self, mock_sap, connector):
        """Test SAP connection establishment."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_sap.return_value = mock_client

        result = connector.connect()

        assert result is True
        mock_client.connect.assert_called_once()

    @patch('integrations.procurement_system_connector.SAPClient')
    def test_get_open_purchase_orders(self, mock_sap, connector):
        """Test retrieving open purchase orders."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client.execute_bapi.return_value = {
            'PO_ITEMS': [
                {
                    'PO_NUMBER': '4500000001',
                    'MATERIAL': 'COAL_001',
                    'QUANTITY': 1000.0,
                    'UNIT': 'MT',
                    'PRICE': 35.00,
                    'CURRENCY': 'USD',
                    'DELIVERY_DATE': '2024-02-15'
                },
                {
                    'PO_NUMBER': '4500000002',
                    'MATERIAL': 'NATURAL_GAS',
                    'QUANTITY': 500.0,
                    'UNIT': 'MT',
                    'PRICE': 45.00,
                    'CURRENCY': 'USD',
                    'DELIVERY_DATE': '2024-02-20'
                }
            ]
        }
        mock_sap.return_value = mock_client

        connector.connect()
        orders = connector.get_open_purchase_orders()

        assert len(orders) == 2
        assert orders[0].po_number == '4500000001'
        assert orders[0].material == 'COAL_001'
        assert orders[0].quantity == 1000.0

    @patch('integrations.procurement_system_connector.SAPClient')
    def test_get_supplier_info(self, mock_sap, connector):
        """Test retrieving supplier information."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client.execute_bapi.return_value = {
            'SUPPLIER': {
                'VENDOR_NO': 'V001',
                'NAME': 'Coal Suppliers Inc.',
                'COUNTRY': 'US',
                'PAYMENT_TERMS': 'NET30',
                'CREDIT_LIMIT': 1000000.00,
                'CURRENT_BALANCE': 250000.00
            }
        }
        mock_sap.return_value = mock_client

        connector.connect()
        supplier = connector.get_supplier_info('V001')

        assert supplier.vendor_no == 'V001'
        assert supplier.name == 'Coal Suppliers Inc.'
        assert supplier.credit_limit == 1000000.00

    @patch('integrations.procurement_system_connector.SAPClient')
    def test_create_purchase_order(self, mock_sap, connector):
        """Test creating a new purchase order."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client.execute_bapi.return_value = {
            'PO_NUMBER': '4500000099',
            'STATUS': 'CREATED'
        }
        mock_sap.return_value = mock_client

        connector.connect()
        po = PurchaseOrder(
            material='BIOMASS_001',
            quantity=200.0,
            unit='MT',
            price=80.00,
            currency='USD',
            vendor_no='V002',
            delivery_date=datetime(2024, 3, 1)
        )
        result = connector.create_purchase_order(po)

        assert result.po_number == '4500000099'
        assert result.status == 'CREATED'


class TestOracleIntegration:
    """Test suite for Oracle ERP integration."""

    @pytest.fixture
    def oracle_config(self):
        """Oracle ERP configuration."""
        return ProcurementConfig(
            erp_type='oracle',
            host='oracle.example.com',
            port=1521,
            service_name='ERPDB',
            username='erp_user',
            password='secure_password'
        )

    @pytest.fixture
    def connector(self, oracle_config):
        """Create Oracle connector."""
        return ProcurementSystemConnector(oracle_config)

    @patch('integrations.procurement_system_connector.OracleClient')
    def test_oracle_connection(self, mock_oracle, connector):
        """Test Oracle connection."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_oracle.return_value = mock_client

        result = connector.connect()

        assert result is True

    @patch('integrations.procurement_system_connector.OracleClient')
    def test_oracle_query_inventory(self, mock_oracle, connector):
        """Test querying inventory from Oracle."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ('COAL_001', 'Coal Storage A', 5000.0, 'MT', '2024-01-15'),
            ('NG_001', 'Gas Tank 1', 2000.0, 'MT', '2024-01-15')
        ]
        mock_client.cursor.return_value = mock_cursor
        mock_oracle.return_value = mock_client

        connector.connect()
        inventory = connector.get_inventory_levels()

        assert len(inventory) == 2
        assert inventory[0].material_code == 'COAL_001'
        assert inventory[0].quantity == 5000.0


class TestInventorySynchronization:
    """Test suite for inventory synchronization."""

    @pytest.fixture
    def connector(self):
        """Create connector with mock."""
        config = ProcurementConfig(
            erp_type='sap',
            host='sap.example.com',
            port=443,
            client='100',
            username='user',
            password='pass'
        )
        return ProcurementSystemConnector(config)

    @patch('integrations.procurement_system_connector.SAPClient')
    def test_sync_inventory_to_erp(self, mock_sap, connector):
        """Test synchronizing inventory levels to ERP."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client.execute_bapi.return_value = {'STATUS': 'SUCCESS'}
        mock_sap.return_value = mock_client

        connector.connect()

        inventory_updates = [
            InventoryRecord(
                material_code='COAL_001',
                storage_location='STORAGE_A',
                quantity=4500.0,
                unit='MT',
                timestamp=datetime.now()
            )
        ]

        result = connector.sync_inventory(inventory_updates)

        assert result.success is True
        mock_client.execute_bapi.assert_called()

    @patch('integrations.procurement_system_connector.SAPClient')
    def test_conflict_resolution(self, mock_sap, connector):
        """Test conflict resolution during sync."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        # First call returns conflict, second resolves
        mock_client.execute_bapi.side_effect = [
            {'STATUS': 'CONFLICT', 'ERP_VALUE': 4800.0},
            {'STATUS': 'SUCCESS'}
        ]
        mock_sap.return_value = mock_client

        connector.connect()

        inventory = InventoryRecord(
            material_code='COAL_001',
            storage_location='STORAGE_A',
            quantity=4500.0,
            unit='MT',
            timestamp=datetime.now()
        )

        # Should resolve conflict and succeed
        result = connector.sync_inventory_with_conflict_resolution([inventory])

        assert result.success is True
        assert result.conflicts_resolved == 1


class TestProcurementOptimization:
    """Test suite for procurement optimization integration."""

    @pytest.fixture
    def connector(self):
        """Create connector."""
        config = ProcurementConfig(
            erp_type='sap',
            host='sap.example.com',
            port=443,
            client='100',
            username='user',
            password='pass'
        )
        return ProcurementSystemConnector(config)

    @patch('integrations.procurement_system_connector.SAPClient')
    def test_get_historical_prices(self, mock_sap, connector):
        """Test retrieving historical price data."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client.execute_bapi.return_value = {
            'PRICE_HISTORY': [
                {'DATE': '2024-01-01', 'PRICE': 34.50, 'MATERIAL': 'COAL_001'},
                {'DATE': '2024-01-08', 'PRICE': 35.00, 'MATERIAL': 'COAL_001'},
                {'DATE': '2024-01-15', 'PRICE': 35.50, 'MATERIAL': 'COAL_001'}
            ]
        }
        mock_sap.return_value = mock_client

        connector.connect()
        prices = connector.get_price_history(
            material='COAL_001',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )

        assert len(prices) == 3
        assert prices[0].price == 34.50
        assert prices[2].price == 35.50

    @patch('integrations.procurement_system_connector.SAPClient')
    def test_get_supplier_performance(self, mock_sap, connector):
        """Test retrieving supplier performance metrics."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client.execute_bapi.return_value = {
            'PERFORMANCE': {
                'VENDOR_NO': 'V001',
                'ON_TIME_DELIVERY_PERCENT': 95.5,
                'QUALITY_SCORE': 4.2,
                'TOTAL_ORDERS': 50,
                'DEFECT_RATE_PERCENT': 0.5
            }
        }
        mock_sap.return_value = mock_client

        connector.connect()
        performance = connector.get_supplier_performance('V001')

        assert performance.on_time_delivery_percent == 95.5
        assert performance.quality_score == 4.2

    @patch('integrations.procurement_system_connector.SAPClient')
    def test_calculate_eoq_from_erp_data(self, mock_sap, connector):
        """Test EOQ calculation using ERP consumption data."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client.execute_bapi.return_value = {
            'CONSUMPTION': {
                'MATERIAL': 'COAL_001',
                'ANNUAL_DEMAND': 12000.0,  # MT/year
                'ORDERING_COST': 500.0,  # USD/order
                'HOLDING_COST_PERCENT': 0.15  # 15% of value
            },
            'CURRENT_PRICE': 35.00
        }
        mock_sap.return_value = mock_client

        connector.connect()
        eoq = connector.calculate_eoq('COAL_001')

        # EOQ = sqrt(2 * D * S / H)
        # D = 12000, S = 500, H = 35 * 0.15 = 5.25
        # EOQ = sqrt(2 * 12000 * 500 / 5.25) = ~1512 MT
        assert 1400 < eoq.quantity < 1600
        assert eoq.reorder_point > 0


class TestErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def connector(self):
        """Create connector."""
        config = ProcurementConfig(
            erp_type='sap',
            host='sap.example.com',
            port=443,
            client='100',
            username='user',
            password='pass',
            retry_attempts=3
        )
        return ProcurementSystemConnector(config)

    @patch('integrations.procurement_system_connector.SAPClient')
    def test_connection_failure_handling(self, mock_sap, connector):
        """Test connection failure is handled."""
        from integrations.procurement_system_connector import ERPConnectionError

        mock_client = Mock()
        mock_client.connect.side_effect = Exception("Connection refused")
        mock_sap.return_value = mock_client

        with pytest.raises(ERPConnectionError):
            connector.connect()

    @patch('integrations.procurement_system_connector.SAPClient')
    def test_authentication_failure(self, mock_sap, connector):
        """Test authentication failure handling."""
        from integrations.procurement_system_connector import ERPAuthenticationError

        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client.execute_bapi.side_effect = Exception("Authentication failed")
        mock_sap.return_value = mock_client

        connector.connect()

        with pytest.raises(ERPAuthenticationError):
            connector.get_open_purchase_orders()

    @patch('integrations.procurement_system_connector.SAPClient')
    def test_transaction_rollback_on_error(self, mock_sap, connector):
        """Test transaction is rolled back on error."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client.execute_bapi.side_effect = Exception("Transaction failed")
        mock_sap.return_value = mock_client

        connector.connect()

        po = PurchaseOrder(
            material='COAL_001',
            quantity=100.0,
            unit='MT',
            price=35.00,
            currency='USD',
            vendor_no='V001',
            delivery_date=datetime(2024, 3, 1)
        )

        with pytest.raises(Exception):
            connector.create_purchase_order(po)

        # Rollback should have been called
        mock_client.rollback.assert_called()


class TestProvenanceTracking:
    """Test suite for provenance tracking in procurement data."""

    @pytest.fixture
    def connector(self):
        """Create connector."""
        config = ProcurementConfig(
            erp_type='sap',
            host='sap.example.com',
            port=443,
            client='100',
            username='user',
            password='pass'
        )
        return ProcurementSystemConnector(config)

    def test_purchase_order_provenance(self, connector):
        """Test PO includes provenance hash."""
        po = PurchaseOrder(
            po_number='4500000001',
            material='COAL_001',
            quantity=1000.0,
            unit='MT',
            price=35.00,
            currency='USD',
            vendor_no='V001',
            delivery_date=datetime(2024, 2, 15)
        )

        assert po.provenance_hash is not None
        assert len(po.provenance_hash) == 64

    def test_provenance_changes_with_data(self, connector):
        """Test provenance hash changes when data changes."""
        po1 = PurchaseOrder(
            po_number='4500000001',
            material='COAL_001',
            quantity=1000.0,
            unit='MT',
            price=35.00,
            currency='USD',
            vendor_no='V001',
            delivery_date=datetime(2024, 2, 15)
        )

        po2 = PurchaseOrder(
            po_number='4500000001',
            material='COAL_001',
            quantity=1001.0,  # Different quantity
            unit='MT',
            price=35.00,
            currency='USD',
            vendor_no='V001',
            delivery_date=datetime(2024, 2, 15)
        )

        assert po1.provenance_hash != po2.provenance_hash
