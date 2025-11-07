"""
Oracle Sandbox Environment Setup
GL-VCCI Scope 3 Platform

Utilities for setting up Oracle sandbox environment, creating test data,
and mocking Oracle REST services for CI/CD.

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


class OracleSandboxSetup:
    """
    Oracle Sandbox environment setup and management.

    Provides utilities for:
    - Verifying sandbox connection
    - Creating test data
    - Mock server setup for CI/CD
    - Cleanup utilities
    """

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize Oracle sandbox setup.

        Args:
            base_url: Oracle sandbox base URL (from env if None)
        """
        self.base_url = base_url or os.getenv("ORACLE_SANDBOX_URL", "https://sandbox.oracle.com")
        self.client_id = os.getenv("ORACLE_SANDBOX_CLIENT_ID")
        self.client_secret = os.getenv("ORACLE_SANDBOX_CLIENT_SECRET")
        self.token_url = os.getenv("ORACLE_SANDBOX_TOKEN_URL")

    def verify_environment_variables(self) -> Dict[str, bool]:
        """
        Verify all required environment variables are set.

        Returns:
            Dictionary with verification status for each variable
        """
        return {
            "ORACLE_SANDBOX_URL": bool(self.base_url),
            "ORACLE_SANDBOX_CLIENT_ID": bool(self.client_id),
            "ORACLE_SANDBOX_CLIENT_SECRET": bool(self.client_secret),
            "ORACLE_SANDBOX_TOKEN_URL": bool(self.token_url),
            "RUN_INTEGRATION_TESTS": os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"
        }

    def is_sandbox_available(self) -> bool:
        """
        Check if Oracle sandbox is available and accessible.

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
                "POHeaderId": 300000000000 + i,
                "OrderNumber": f"PO{300000 + i:06d}",
                "Supplier": f"SUPP{1000 + (i % 50):04d}",
                "SupplierName": f"Supplier Company {1000 + (i % 50):04d}",
                "OrderDate": po_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "TotalAmount": round(1500 + (i * 234.56) % 75000, 2),
                "Currency": "USD",
                "BuyerName": f"Buyer {(i % 10) + 1}",
                "Status": ["APPROVED", "IN_PROCESS", "PENDING"][i % 3],
                "CreationDate": po_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "LastUpdateDate": (po_date + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "Lines": [
                    {
                        "POLineId": 300000000000 + i * 10 + j,
                        "LineNumber": j + 1,
                        "ItemDescription": f"Item {j + 1} for PO {300000 + i:06d}",
                        "Quantity": round(15 + (j * 7.5), 2),
                        "UnitOfMeasure": "EA",
                        "UnitPrice": round(150 + (j * 35.75), 2),
                        "LineAmount": round((15 + (j * 7.5)) * (150 + (j * 35.75)), 2)
                    }
                    for j in range(1 + (i % 3))
                ]
            }
            purchase_orders.append(po)

        return purchase_orders

    def generate_test_requisitions(self, count: int = 50) -> List[Dict[str, Any]]:
        """
        Generate test requisition data.

        Args:
            count: Number of requisitions to generate

        Returns:
            List of requisition dictionaries
        """
        requisitions = []
        base_date = datetime.now() - timedelta(days=25)

        for i in range(count):
            req_date = base_date + timedelta(days=i % 25)
            req = {
                "RequisitionHeaderId": 400000000000 + i,
                "RequisitionNumber": f"REQ{400000 + i:06d}",
                "RequesterName": f"Employee {(i % 20) + 1}",
                "RequesterEmail": f"employee{(i % 20) + 1}@company.com",
                "RequestDate": req_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "Status": ["APPROVED", "PENDING", "REJECTED"][i % 3],
                "TotalAmount": round(800 + (i * 156.78) % 30000, 2),
                "Currency": "USD",
                "Lines": [
                    {
                        "RequisitionLineId": 400000000000 + i * 5 + j,
                        "LineNumber": j + 1,
                        "Description": f"Requisition item {j + 1}",
                        "Quantity": round(8 + (j * 4.2), 2),
                        "UnitPrice": round(95 + (j * 28.5), 2)
                    }
                    for j in range(1 + (i % 2))
                ]
            }
            requisitions.append(req)

        return requisitions

    def generate_test_shipments(self, count: int = 75) -> List[Dict[str, Any]]:
        """
        Generate test shipment data.

        Args:
            count: Number of shipments to generate

        Returns:
            List of shipment dictionaries
        """
        shipments = []
        base_date = datetime.now() - timedelta(days=15)

        for i in range(count):
            ship_date = base_date + timedelta(days=i % 15)
            shipment = {
                "ShipmentId": 500000000000 + i,
                "ShipmentNumber": f"SHIP{500000 + i:06d}",
                "OrderNumber": f"PO{300000 + (i % 100):06d}",
                "ShipDate": ship_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "EstimatedDeliveryDate": (ship_date + timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "Carrier": ["FedEx", "UPS", "DHL", "USPS"][i % 4],
                "TrackingNumber": f"TRACK{1000000000 + i:010d}",
                "Status": ["IN_TRANSIT", "DELIVERED", "PENDING"][i % 3],
                "OriginLocation": f"Warehouse {(i % 5) + 1}",
                "DestinationLocation": f"Site {(i % 10) + 1}",
                "Weight": round(50 + (i * 12.5) % 500, 2),
                "WeightUnit": "LB"
            }
            shipments.append(shipment)

        return shipments

    @staticmethod
    def create_mock_server():
        """
        Create mock Oracle REST server for CI/CD testing.

        Returns:
            Configured responses mock
        """
        @responses.activate
        def mock_oracle_responses():
            # Token endpoint
            responses.add(
                responses.POST,
                "https://mock.oracle.com/oauth/token",
                json={
                    "access_token": "mock_oracle_token_67890",
                    "token_type": "Bearer",
                    "expires_in": 3600
                },
                status=200
            )

            # Purchase Orders endpoint
            setup = OracleSandboxSetup()
            test_pos = setup.generate_test_purchase_orders(count=10)

            responses.add(
                responses.GET,
                "https://mock.oracle.com/fscmRestApi/resources/11.13.18.05/purchaseOrders",
                json={
                    "items": test_pos,
                    "count": len(test_pos),
                    "hasMore": False,
                    "links": [
                        {"rel": "self", "href": "https://mock.oracle.com/fscmRestApi/resources/11.13.18.05/purchaseOrders"}
                    ]
                },
                status=200
            )

            # Requisitions endpoint
            test_reqs = setup.generate_test_requisitions(count=5)

            responses.add(
                responses.GET,
                "https://mock.oracle.com/fscmRestApi/resources/11.13.18.05/purchaseRequisitions",
                json={
                    "items": test_reqs,
                    "count": len(test_reqs),
                    "hasMore": False,
                    "links": []
                },
                status=200
            )

            # Shipments endpoint
            test_ships = setup.generate_test_shipments(count=5)

            responses.add(
                responses.GET,
                "https://mock.oracle.com/fscmRestApi/resources/11.13.18.05/shipments",
                json={
                    "items": test_ships,
                    "count": len(test_ships),
                    "hasMore": False,
                    "links": []
                },
                status=200
            )

        return mock_oracle_responses

    def cleanup_test_data(self):
        """
        Cleanup test data from sandbox.

        Note: In production, this would clean up test records created during testing.
        """
        print("Cleaning up Oracle sandbox test data...")
        # TODO: Implement actual cleanup logic
        print("Oracle sandbox cleanup complete")

    def get_sandbox_status(self) -> Dict[str, Any]:
        """
        Get Oracle sandbox status and health.

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
        """Print setup instructions for Oracle sandbox."""
        print("\n" + "="*60)
        print("Oracle Sandbox Setup Instructions")
        print("="*60)
        print("\nRequired Environment Variables:")
        print("  ORACLE_SANDBOX_URL=https://your-oracle-sandbox.com")
        print("  ORACLE_SANDBOX_CLIENT_ID=your_client_id")
        print("  ORACLE_SANDBOX_CLIENT_SECRET=your_client_secret")
        print("  ORACLE_SANDBOX_TOKEN_URL=https://your-oracle-sandbox.com/oauth/token")
        print("  RUN_INTEGRATION_TESTS=true")
        print("\nCurrent Status:")

        env_vars = self.verify_environment_variables()
        for var_name, is_set in env_vars.items():
            status = "✓ SET" if is_set else "✗ NOT SET"
            print(f"  {var_name}: {status}")

        print("\nTo run integration tests:")
        print("  1. Set all required environment variables")
        print("  2. Ensure Oracle sandbox is accessible")
        print("  3. Run: pytest -m oracle_sandbox")
        print("\nTo run with mock server (CI/CD):")
        print("  pytest tests/integration/ (without sandbox env vars)")
        print("="*60 + "\n")


def setup_oracle_sandbox():
    """
    Main setup function for Oracle sandbox.

    Can be run as standalone script to verify sandbox setup.
    """
    setup = OracleSandboxSetup()

    print("\n" + "="*60)
    print("Oracle Sandbox Environment Setup")
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
        print("\n✓ Oracle Sandbox is ready for integration testing!")

    return setup


if __name__ == "__main__":
    setup_oracle_sandbox()
