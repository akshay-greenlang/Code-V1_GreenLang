#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example Usage of ERP Connectors for GreenLang

Demonstrates how to use the SAP, Oracle, and Workday connectors
with transaction management and error handling.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Import the connectors
from sap_connector import SAPConnector
from oracle_connector import OracleConnector
from workday_connector import WorkdayConnector
from transaction_manager import TransactionManager, TransactionStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ERPDataExtractor:
    """
    Main class for extracting data from multiple ERP systems.

    Orchestrates data extraction from SAP, Oracle, and Workday
    with transaction management and error handling.
    """

    def __init__(self):
        """Initialize ERP data extractor with all connectors."""
        # Initialize connectors (credentials from environment variables)
        self.sap = SAPConnector()
        self.oracle = OracleConnector()
        self.workday = WorkdayConnector()

        # Initialize transaction manager
        self.transaction_manager = TransactionManager()

        # Connector map for transaction processing
        self.connectors = {
            'SAP': self.sap,
            'Oracle': self.oracle,
            'Workday': self.workday
        }

    def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all ERP systems.

        Returns:
            Dictionary with connection status for each system
        """
        connection_status = {}

        # Connect to SAP
        try:
            connection_status['SAP'] = self.sap.connect()
            logger.info(f"SAP connection: {connection_status['SAP']}")
        except Exception as e:
            logger.error(f"SAP connection failed: {e}")
            connection_status['SAP'] = False

        # Connect to Oracle
        try:
            connection_status['Oracle'] = self.oracle.connect()
            logger.info(f"Oracle connection: {connection_status['Oracle']}")
        except Exception as e:
            logger.error(f"Oracle connection failed: {e}")
            connection_status['Oracle'] = False

        # Connect to Workday
        try:
            connection_status['Workday'] = self.workday.connect()
            logger.info(f"Workday connection: {connection_status['Workday']}")
        except Exception as e:
            logger.error(f"Workday connection failed: {e}")
            connection_status['Workday'] = False

        return connection_status

    def extract_purchase_orders(self, start_date: str, end_date: str) -> Dict[str, List[Dict]]:
        """
        Extract purchase orders from all ERP systems.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary with purchase orders from each system
        """
        all_purchase_orders = {}

        # Extract from SAP
        logger.info(f"Extracting SAP purchase orders from {start_date} to {end_date}")
        try:
            sap_orders = self.sap.get_purchase_orders(start_date, end_date)
            all_purchase_orders['SAP'] = sap_orders
            logger.info(f"Retrieved {len(sap_orders)} purchase orders from SAP")
        except Exception as e:
            logger.error(f"Failed to extract SAP purchase orders: {e}")
            all_purchase_orders['SAP'] = []

        # Extract from Oracle
        logger.info(f"Extracting Oracle purchase orders from {start_date} to {end_date}")
        try:
            oracle_orders = self.oracle.get_purchase_orders(start_date, end_date)
            all_purchase_orders['Oracle'] = oracle_orders
            logger.info(f"Retrieved {len(oracle_orders)} purchase orders from Oracle")
        except Exception as e:
            logger.error(f"Failed to extract Oracle purchase orders: {e}")
            all_purchase_orders['Oracle'] = []

        # Extract from Workday
        logger.info(f"Extracting Workday purchase orders from {start_date} to {end_date}")
        try:
            workday_orders = self.workday.get_purchase_orders(start_date, end_date)
            all_purchase_orders['Workday'] = workday_orders
            logger.info(f"Retrieved {len(workday_orders)} purchase orders from Workday")
        except Exception as e:
            logger.error(f"Failed to extract Workday purchase orders: {e}")
            all_purchase_orders['Workday'] = []

        return all_purchase_orders

    def extract_suppliers(self) -> Dict[str, List[Dict]]:
        """
        Extract supplier data from all ERP systems.

        Returns:
            Dictionary with supplier data from each system
        """
        all_suppliers = {}

        # Extract from SAP
        logger.info("Extracting SAP suppliers")
        try:
            sap_suppliers = self.sap.get_suppliers()
            all_suppliers['SAP'] = sap_suppliers
            logger.info(f"Retrieved {len(sap_suppliers)} suppliers from SAP")
        except Exception as e:
            logger.error(f"Failed to extract SAP suppliers: {e}")
            all_suppliers['SAP'] = []

        # Extract from Oracle
        logger.info("Extracting Oracle suppliers")
        try:
            oracle_suppliers = self.oracle.get_suppliers()
            all_suppliers['Oracle'] = oracle_suppliers
            logger.info(f"Retrieved {len(oracle_suppliers)} suppliers from Oracle")
        except Exception as e:
            logger.error(f"Failed to extract Oracle suppliers: {e}")
            all_suppliers['Oracle'] = []

        # Extract from Workday
        logger.info("Extracting Workday suppliers")
        try:
            workday_suppliers = self.workday.get_suppliers()
            all_suppliers['Workday'] = workday_suppliers
            logger.info(f"Retrieved {len(workday_suppliers)} suppliers from Workday")
        except Exception as e:
            logger.error(f"Failed to extract Workday suppliers: {e}")
            all_suppliers['Workday'] = []

        return all_suppliers

    async def extract_with_transaction_management(self, start_date: str, end_date: str):
        """
        Extract data with transaction management and retry logic.

        Args:
            start_date: Start date for extraction
            end_date: End date for extraction
        """
        # Create transactions for each system and entity type
        transactions = []

        # SAP transactions
        sap_po_transaction = self.transaction_manager.create_transaction(
            source_system='SAP',
            entity_type='purchase_orders',
            query={
                'entity_type': 'purchase_orders',
                'start_date': start_date,
                'end_date': end_date
            }
        )
        transactions.append(sap_po_transaction)

        sap_supplier_transaction = self.transaction_manager.create_transaction(
            source_system='SAP',
            entity_type='suppliers',
            query={'entity_type': 'suppliers'}
        )
        transactions.append(sap_supplier_transaction)

        # Oracle transactions
        oracle_po_transaction = self.transaction_manager.create_transaction(
            source_system='Oracle',
            entity_type='purchase_orders',
            query={
                'entity_type': 'purchase_orders',
                'start_date': start_date,
                'end_date': end_date
            }
        )
        transactions.append(oracle_po_transaction)

        # Workday transactions
        workday_spend_transaction = self.transaction_manager.create_transaction(
            source_system='Workday',
            entity_type='spend_analytics',
            query={
                'entity_type': 'spend_analytics',
                'start_date': start_date,
                'end_date': end_date
            }
        )
        transactions.append(workday_spend_transaction)

        # Process all transactions
        processor_map = {
            'SAP': self._process_sap_query,
            'Oracle': self._process_oracle_query,
            'Workday': self._process_workday_query
        }

        # Process pending transactions
        await self.transaction_manager.process_pending_transactions(processor_map)

        # Process retry queue
        await self.transaction_manager.process_retry_queue(processor_map)

        # Get statistics
        stats = self.transaction_manager.get_stats()
        logger.info(f"Transaction processing complete: {stats}")

    async def _process_sap_query(self, query: Dict[str, Any]) -> List[Dict]:
        """Process SAP query."""
        return self.sap.query(query)

    async def _process_oracle_query(self, query: Dict[str, Any]) -> List[Dict]:
        """Process Oracle query."""
        return self.oracle.query(query)

    async def _process_workday_query(self, query: Dict[str, Any]) -> List[Dict]:
        """Process Workday query."""
        return self.workday.query(query)

    def disconnect_all(self):
        """Disconnect from all ERP systems."""
        self.sap.disconnect()
        self.oracle.disconnect()
        self.workday.disconnect()
        logger.info("Disconnected from all ERP systems")


def example_basic_usage():
    """
    Basic example of using ERP connectors.
    """
    print("\n" + "="*60)
    print("BASIC ERP CONNECTOR USAGE EXAMPLE")
    print("="*60)

    # Initialize extractor
    extractor = ERPDataExtractor()

    # Connect to all systems
    print("\n1. Connecting to ERP systems...")
    connection_status = extractor.connect_all()
    print(f"Connection status: {connection_status}")

    # Set date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    # Extract purchase orders
    print(f"\n2. Extracting purchase orders ({start_date} to {end_date})...")
    purchase_orders = extractor.extract_purchase_orders(start_date, end_date)

    for system, orders in purchase_orders.items():
        print(f"   {system}: {len(orders)} purchase orders")
        if orders and len(orders) > 0:
            # Show sample order
            sample = orders[0]
            print(f"      Sample: {sample.get('document_number')} - {sample.get('supplier_name')}")

    # Extract suppliers
    print("\n3. Extracting supplier data...")
    suppliers = extractor.extract_suppliers()

    for system, supplier_list in suppliers.items():
        print(f"   {system}: {len(supplier_list)} suppliers")
        if supplier_list and len(supplier_list) > 0:
            # Show sample supplier
            sample = supplier_list[0]
            print(f"      Sample: {sample.get('supplier_name')} ({sample.get('status')})")

    # Disconnect
    print("\n4. Disconnecting from ERP systems...")
    extractor.disconnect_all()

    print("\nBasic example complete!")


def example_advanced_usage():
    """
    Advanced example with custom queries and error handling.
    """
    print("\n" + "="*60)
    print("ADVANCED ERP CONNECTOR USAGE EXAMPLE")
    print("="*60)

    # Direct connector usage with custom queries
    sap = SAPConnector()

    try:
        # Connect
        print("\n1. Connecting to SAP...")
        if sap.connect():
            print("   Connected successfully!")

            # Custom query with filters
            print("\n2. Running custom query with filters...")
            query = {
                'entity_type': 'purchase_orders',
                'start_date': '2024-01-01',
                'end_date': '2024-01-31',
                'filters': {
                    'Plant': ['1000', '2000'],  # Multiple plants
                    'CompanyCode': '1000'
                },
                'fields': [  # Select specific fields
                    'PurchaseOrder',
                    'Supplier',
                    'TotalNetAmount',
                    'DocumentCurrency'
                ],
                'expand': ['to_PurchaseOrderItem'],  # Include line items
                'limit': 100  # Limit results
            }

            results = sap.query(query)
            print(f"   Retrieved {len(results)} records")

            # Show sample result
            if results:
                sample = results[0]
                print(f"\n   Sample Purchase Order:")
                print(f"      ID: {sample.get('id')}")
                print(f"      Supplier: {sample.get('supplier_name')}")
                print(f"      Amount: {sample.get('total_amount')} {sample.get('currency')}")
                print(f"      Items: {len(sample.get('items', []))}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        sap.disconnect()
        print("\nAdvanced example complete!")


async def example_transaction_management():
    """
    Example using transaction management with retry and DLQ.
    """
    print("\n" + "="*60)
    print("TRANSACTION MANAGEMENT EXAMPLE")
    print("="*60)

    extractor = ERPDataExtractor()

    # Connect to systems
    print("\n1. Connecting to ERP systems...")
    extractor.connect_all()

    # Run extraction with transaction management
    print("\n2. Running extraction with transaction management...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    await extractor.extract_with_transaction_management(start_date, end_date)

    # Check dead letter queue
    print("\n3. Checking dead letter queue...")
    dlq_stats = extractor.transaction_manager.dlq.get_stats()
    print(f"   DLQ Statistics: {dlq_stats}")

    # List failed transactions
    failed_transactions = extractor.transaction_manager.dlq.list_all(limit=5)
    if failed_transactions:
        print(f"\n   Failed Transactions:")
        for tx in failed_transactions:
            print(f"      - {tx.id}: {tx.source_system}/{tx.entity_type}")
            print(f"        Error: {tx.error_message}")

    # Disconnect
    extractor.disconnect_all()
    print("\nTransaction management example complete!")


def example_data_transformation():
    """
    Example showing data transformation and standardization.
    """
    print("\n" + "="*60)
    print("DATA TRANSFORMATION EXAMPLE")
    print("="*60)

    # Connect to Oracle
    oracle = OracleConnector()
    oracle.connect()

    print("\n1. Extracting and transforming Oracle data...")

    # Get inventory transactions
    query = {
        'entity_type': 'inventory_transactions',
        'start_date': '2024-01-01',
        'end_date': '2024-01-31',
        'filters': {
            'OrganizationCode': 'ORG1'
        }
    }

    transactions = oracle.query(query)

    if transactions:
        print(f"   Retrieved {len(transactions)} inventory transactions")

        # Show standardized format
        sample = transactions[0]
        print("\n   Standardized Transaction Format:")
        print(f"      ID: {sample.get('id')}")
        print(f"      Type: {sample.get('type')}")
        print(f"      Source System: {sample.get('source_system')}")
        print(f"      Item: {sample.get('item_number')}")
        print(f"      Quantity: {sample.get('quantity')} {sample.get('unit_of_measure')}")
        print(f"      Organization: {sample.get('organization')}")
        print(f"      Extracted At: {sample.get('extracted_at')}")

    oracle.disconnect()
    print("\nData transformation example complete!")


if __name__ == "__main__":
    """Run examples."""
    print("\n" + "="*60)
    print("GreenLang ERP Connector Examples")
    print("="*60)

    # Check for required environment variables
    required_vars = [
        'SAP_BASE_URL', 'SAP_CLIENT_ID', 'SAP_CLIENT_SECRET',
        'ORACLE_BASE_URL', 'ORACLE_USERNAME', 'ORACLE_PASSWORD',
        'WORKDAY_BASE_URL', 'WORKDAY_TENANT', 'WORKDAY_CLIENT_ID'
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("\n⚠️  Warning: Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nSet these variables for full functionality.")
        print("Running with mock data for demonstration...")

    # Run examples
    try:
        # Basic usage
        example_basic_usage()

        # Advanced usage
        example_advanced_usage()

        # Data transformation
        example_data_transformation()

        # Transaction management (async)
        print("\nRunning async transaction management example...")
        asyncio.run(example_transaction_management())

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError running examples: {e}")

    print("\n" + "="*60)
    print("All examples complete!")
    print("="*60)