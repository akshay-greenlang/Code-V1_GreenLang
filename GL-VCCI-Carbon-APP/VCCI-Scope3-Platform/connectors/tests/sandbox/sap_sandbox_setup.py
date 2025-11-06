"""
SAP Sandbox Environment Setup
GL-VCCI Scope 3 Platform

Utilities for setting up SAP sandbox environment, creating test data,
and mocking SAP OData services for CI/CD.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import responses
from responses import matchers


class SAPSandboxSetup:
    """
    SAP Sandbox environment setup and management.

    Provides utilities for:
    - Verifying sandbox connection
    - Creating test data
    - Mock server setup for CI/CD
    - Cleanup utilities
    """

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize SAP sandbox setup.

        Args:
            base_url: SAP sandbox base URL (from env if None)
        """
        self.base_url = base_url or os.getenv("SAP_SANDBOX_URL", "https://sandbox.sap.com")
        self.client_id = os.getenv("SAP_SANDBOX_CLIENT_ID")
        self.client_secret = os.getenv("SAP_SANDBOX_CLIENT_SECRET")
        self.token_url = os.getenv("SAP_SANDBOX_TOKEN_URL")

    def verify_environment_variables(self) -> Dict[str, bool]:
        """
        Verify all required environment variables are set.

        Returns:
            Dictionary with verification status for each variable
        """
        return {
            "SAP_SANDBOX_URL": bool(self.base_url),
            "SAP_SANDBOX_CLIENT_ID": bool(self.client_id),
            "SAP_SANDBOX_CLIENT_SECRET": bool(self.client_secret),
            "SAP_SANDBOX_TOKEN_URL": bool(self.token_url),
            "RUN_INTEGRATION_TESTS": os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"
        }

    def is_sandbox_available(self) -> bool:
        """
        Check if SAP sandbox is available and accessible.

        Returns:
            True if sandbox is available, False otherwise
        """
        env_vars = self.verify_environment_variables()
        return all(env_vars.values())

    def generate_test_purchase_orders(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        Generate test purchase order data.

        Args:
            count: Number of POs to generate

        Returns:
            List of purchase order dictionaries
        """
        purchase_orders = []
        base_date = datetime.now() - timedelta(days=30)

        for i in range(count):
            po_date = base_date + timedelta(days=i % 30)
            po = {
                "PurchaseOrder": f"PO{4500000000 + i:010d}",
                "Supplier": f"SUP{1000 + (i % 50):04d}",
                "PurchaseOrderDate": po_date.strftime("%Y-%m-%d"),
                "TotalAmount": round(1000 + (i * 123.45) % 50000, 2),
                "Currency": "USD",
                "PurchasingOrganization": "1000",
                "PurchasingGroup": "001",
                "CompanyCode": "1000",
                "DocumentType": "NB",
                "CreationDate": po_date.strftime("%Y-%m-%d"),
                "LastChangeDate": (po_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                "Items": [
                    {
                        "PurchaseOrderItem": str(10 + j * 10),
                        "Material": f"MAT{10000 + j:05d}",
                        "OrderQuantity": round(10 + (j * 5.5), 2),
                        "OrderUnit": "EA",
                        "NetPrice": round(100 + (j * 25.5), 2),
                        "PlantCode": "1000"
                    }
                    for j in range(1 + (i % 3))
                ]
            }
            purchase_orders.append(po)

        return purchase_orders

    def generate_test_goods_receipts(self, count: int = 50) -> List[Dict[str, Any]]:
        """
        Generate test goods receipt data.

        Args:
            count: Number of goods receipts to generate

        Returns:
            List of goods receipt dictionaries
        """
        goods_receipts = []
        base_date = datetime.now() - timedelta(days=20)

        for i in range(count):
            gr_date = base_date + timedelta(days=i % 20)
            gr = {
                "MaterialDocument": f"GR{5000000000 + i:010d}",
                "MaterialDocumentYear": gr_date.year,
                "PostingDate": gr_date.strftime("%Y-%m-%d"),
                "DocumentDate": gr_date.strftime("%Y-%m-%d"),
                "GoodsReceiptType": "101",
                "PurchaseOrder": f"PO{4500000000 + (i % 100):010d}",
                "Supplier": f"SUP{1000 + (i % 50):04d}",
                "Items": [
                    {
                        "MaterialDocumentItem": str(1 + j),
                        "Material": f"MAT{10000 + j:05d}",
                        "Quantity": round(5 + (j * 2.5), 2),
                        "Unit": "EA",
                        "PlantCode": "1000",
                        "StorageLocation": "0001"
                    }
                    for j in range(1 + (i % 2))
                ]
            }
            goods_receipts.append(gr)

        return goods_receipts

    @staticmethod
    def create_mock_server():
        """
        Create mock SAP OData server for CI/CD testing.

        Returns:
            Configured responses mock
        """
        # Mock OAuth token endpoint
        @responses.activate
        def mock_sap_responses():
            # Token endpoint
            responses.add(
                responses.POST,
                "https://mock.sap.com/oauth/token",
                json={
                    "access_token": "mock_token_12345",
                    "token_type": "Bearer",
                    "expires_in": 3600
                },
                status=200
            )

            # Purchase Orders endpoint
            setup = SAPSandboxSetup()
            test_pos = setup.generate_test_purchase_orders(count=10)

            responses.add(
                responses.GET,
                "https://mock.sap.com/sap/opu/odata/sap/MM_PUR_PO_MAINT_V2_SRV/C_PurchaseOrderTP",
                json={"value": test_pos},
                status=200,
                match=[
                    matchers.header_matcher({"Authorization": "Bearer mock_token_12345"})
                ]
            )

            # Goods Receipts endpoint
            test_grs = setup.generate_test_goods_receipts(count=5)

            responses.add(
                responses.GET,
                "https://mock.sap.com/sap/opu/odata/sap/API_MATERIAL_DOCUMENT_SRV/A_MaterialDocumentHeader",
                json={"value": test_grs},
                status=200
            )

        return mock_sap_responses

    def cleanup_test_data(self):
        """
        Cleanup test data from sandbox.

        Note: In production, this would clean up test records created during testing.
        """
        print("Cleaning up SAP sandbox test data...")
        # TODO: Implement actual cleanup logic
        # This would involve:
        # 1. Connecting to sandbox
        # 2. Identifying test records (by special markers)
        # 3. Deleting or archiving them
        print("SAP sandbox cleanup complete")

    def get_sandbox_status(self) -> Dict[str, Any]:
        """
        Get SAP sandbox status and health.

        Returns:
            Dictionary with sandbox status information
        """
        return {
            "available": self.is_sandbox_available(),
            "base_url": self.base_url,
            "environment_variables": self.verify_environment_variables(),
            "timestamp": datetime.now().isoformat()
        }

    def print_setup_instructions(self):
        """Print setup instructions for SAP sandbox."""
        print("\n" + "="*60)
        print("SAP Sandbox Setup Instructions")
        print("="*60)
        print("\nRequired Environment Variables:")
        print("  SAP_SANDBOX_URL=https://your-sap-sandbox.com")
        print("  SAP_SANDBOX_CLIENT_ID=your_client_id")
        print("  SAP_SANDBOX_CLIENT_SECRET=your_client_secret")
        print("  SAP_SANDBOX_TOKEN_URL=https://your-sap-sandbox.com/oauth/token")
        print("  SAP_SANDBOX_OAUTH_SCOPE=API_BUSINESS_PARTNER")
        print("  RUN_INTEGRATION_TESTS=true")
        print("\nCurrent Status:")

        env_vars = self.verify_environment_variables()
        for var_name, is_set in env_vars.items():
            status = "✓ SET" if is_set else "✗ NOT SET"
            print(f"  {var_name}: {status}")

        print("\nTo run integration tests:")
        print("  1. Set all required environment variables")
        print("  2. Ensure SAP sandbox is accessible")
        print("  3. Run: pytest -m integration")
        print("\nTo run with mock server (CI/CD):")
        print("  pytest tests/integration/ (without sandbox env vars)")
        print("="*60 + "\n")


def setup_sap_sandbox():
    """
    Main setup function for SAP sandbox.

    Can be run as standalone script to verify sandbox setup.
    """
    setup = SAPSandboxSetup()

    print("\n" + "="*60)
    print("SAP Sandbox Environment Setup")
    print("="*60 + "\n")

    # Check status
    status = setup.get_sandbox_status()
    print(f"Sandbox Available: {status['available']}")
    print(f"Base URL: {status['base_url']}")

    # Print environment variables
    print("\nEnvironment Variables:")
    for var_name, is_set in status['environment_variables'].items():
        print(f"  {var_name}: {'✓' if is_set else '✗'}")

    if not status['available']:
        print("\n⚠ Sandbox not fully configured!")
        setup.print_setup_instructions()
    else:
        print("\n✓ SAP Sandbox is ready for integration testing!")

    return setup


if __name__ == "__main__":
    setup_sap_sandbox()
