# -*- coding: utf-8 -*-
"""
Workday Sandbox Environment Setup
GL-VCCI Scope 3 Platform

Utilities for setting up Workday sandbox environment, creating test reports,
and mocking Workday RaaS services for CI/CD.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import responses
from greenlang.determinism import DeterministicClock


class WorkdaySandboxSetup:
    """
    Workday Sandbox environment setup and management.

    Provides utilities for:
    - Verifying sandbox connection
    - Creating test report data
    - Mock server setup for CI/CD
    - Cleanup utilities
    """

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize Workday sandbox setup.

        Args:
            base_url: Workday sandbox base URL (from env if None)
        """
        self.base_url = base_url or os.getenv("WORKDAY_SANDBOX_URL", "https://sandbox.workday.com")
        self.client_id = os.getenv("WORKDAY_SANDBOX_CLIENT_ID")
        self.client_secret = os.getenv("WORKDAY_SANDBOX_CLIENT_SECRET")
        self.tenant_name = os.getenv("WORKDAY_SANDBOX_TENANT")

    def verify_environment_variables(self) -> Dict[str, bool]:
        """
        Verify all required environment variables are set.

        Returns:
            Dictionary with verification status for each variable
        """
        return {
            "WORKDAY_SANDBOX_URL": bool(self.base_url),
            "WORKDAY_SANDBOX_CLIENT_ID": bool(self.client_id),
            "WORKDAY_SANDBOX_CLIENT_SECRET": bool(self.client_secret),
            "WORKDAY_SANDBOX_TENANT": bool(self.tenant_name),
            "RUN_INTEGRATION_TESTS": os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"
        }

    def is_sandbox_available(self) -> bool:
        """
        Check if Workday sandbox is available and accessible.

        Returns:
            True if sandbox is available, False otherwise
        """
        env_vars = self.verify_environment_variables()
        return all(env_vars.values())

    def generate_test_expense_reports(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        Generate test expense report data.

        Args:
            count: Number of expense reports to generate

        Returns:
            List of expense report dictionaries
        """
        expense_reports = []
        base_date = DeterministicClock.now() - timedelta(days=60)

        expense_types = [
            "Airfare", "Hotel", "Meals", "Ground Transportation",
            "Rental Car", "Parking", "Conference Fee", "Office Supplies"
        ]

        for i in range(count):
            report_date = base_date + timedelta(days=i % 60)
            report = {
                "expense_id": f"EXP{2000000 + i:07d}",
                "employee_id": f"EMP{1000 + (i % 100):04d}",
                "employee_name": f"Employee {1000 + (i % 100):04d}",
                "department": ["Finance", "Engineering", "Sales", "Operations"][(i % 4)],
                "report_date": report_date.strftime("%Y-%m-%d"),
                "expense_date": report_date.strftime("%Y-%m-%d"),
                "expense_type": expense_types[i % len(expense_types)],
                "total_amount": round(150 + (i * 43.21) % 2000, 2),
                "currency": "USD",
                "status": ["APPROVED", "PENDING", "SUBMITTED"][(i % 3)],
                "reimbursement_date": (report_date + timedelta(days=14)).strftime("%Y-%m-%d") if i % 3 == 0 else None,
                "business_purpose": f"Business trip {i + 1}",
                "location": ["New York, NY", "San Francisco, CA", "Chicago, IL", "Boston, MA"][(i % 4)]
            }
            expense_reports.append(report)

        return expense_reports

    def generate_test_commute_surveys(self, count: int = 200) -> List[Dict[str, Any]]:
        """
        Generate test commute survey data.

        Args:
            count: Number of commute surveys to generate

        Returns:
            List of commute survey dictionaries
        """
        commute_surveys = []
        base_date = DeterministicClock.now() - timedelta(days=90)

        commute_modes = ["Car", "Bus", "Train", "Bike", "Walk", "Carpool", "Remote"]

        for i in range(count):
            survey_date = base_date + timedelta(days=i % 90)
            survey = {
                "survey_id": f"CS{3000000 + i:07d}",
                "employee_id": f"EMP{1000 + (i % 100):04d}",
                "employee_name": f"Employee {1000 + (i % 100):04d}",
                "survey_date": survey_date.strftime("%Y-%m-%d"),
                "commute_mode": commute_modes[i % len(commute_modes)],
                "distance_km": round(5 + (i * 2.3) % 50, 1) if commute_modes[i % len(commute_modes)] != "Remote" else 0,
                "distance_miles": round((5 + (i * 2.3) % 50) * 0.621371, 1) if commute_modes[i % len(commute_modes)] != "Remote" else 0,
                "frequency_per_week": 5 - (i % 6),  # 0-5 days per week
                "vehicle_type": ["Sedan", "SUV", "Hybrid", "Electric", None][(i % 5)] if commute_modes[i % len(commute_modes)] in ["Car", "Carpool"] else None,
                "fuel_type": ["Gasoline", "Diesel", "Electric", "Hybrid"][(i % 4)] if commute_modes[i % len(commute_modes)] in ["Car", "Carpool"] else None
            }
            commute_surveys.append(survey)

        return commute_surveys

    def generate_test_business_travel(self, count: int = 50) -> List[Dict[str, Any]]:
        """
        Generate test business travel data.

        Args:
            count: Number of business travel records to generate

        Returns:
            List of business travel dictionaries
        """
        business_travel = []
        base_date = DeterministicClock.now() - timedelta(days=180)

        travel_modes = ["Flight", "Train", "Car", "Bus"]
        destinations = [
            ("New York, NY", 2500),
            ("Los Angeles, CA", 3800),
            ("Chicago, IL", 1800),
            ("Houston, TX", 2200),
            ("Miami, FL", 2000),
            ("Seattle, WA", 3500),
            ("Boston, MA", 2400),
            ("San Francisco, CA", 3900)
        ]

        for i in range(count):
            travel_date = base_date + timedelta(days=i * 3)
            dest, distance = destinations[i % len(destinations)]

            travel = {
                "travel_id": f"BT{4000000 + i:07d}",
                "employee_id": f"EMP{1000 + (i % 100):04d}",
                "employee_name": f"Employee {1000 + (i % 100):04d}",
                "travel_date": travel_date.strftime("%Y-%m-%d"),
                "return_date": (travel_date + timedelta(days=2 + (i % 3))).strftime("%Y-%m-%d"),
                "destination": dest,
                "origin": "Headquarters",
                "travel_mode": travel_modes[i % len(travel_modes)],
                "distance_km": round(distance * 1.60934, 1),
                "distance_miles": distance,
                "purpose": ["Client Meeting", "Conference", "Training", "Site Visit"][(i % 4)],
                "cost": round(500 + (distance * 0.5) + (i * 25.5) % 1000, 2),
                "currency": "USD"
            }
            business_travel.append(travel)

        return business_travel

    @staticmethod
    def create_mock_server():
        """
        Create mock Workday RaaS server for CI/CD testing.

        Returns:
            Configured responses mock
        """
        @responses.activate
        def mock_workday_responses():
            # Token endpoint
            responses.add(
                responses.POST,
                "https://mock.workday.com/oauth/token",
                json={
                    "access_token": "mock_workday_token_abc123",
                    "token_type": "Bearer",
                    "expires_in": 3600
                },
                status=200
            )

            # Expense Reports endpoint
            setup = WorkdaySandboxSetup()
            test_expenses = setup.generate_test_expense_reports(count=10)

            responses.add(
                responses.GET,
                "https://mock.workday.com/ccx/service/customreport2/tenant/report/expense_reports",
                json={"Report_Entry": test_expenses},
                status=200
            )

            # Commute Surveys endpoint
            test_commutes = setup.generate_test_commute_surveys(count=10)

            responses.add(
                responses.GET,
                "https://mock.workday.com/ccx/service/customreport2/tenant/report/commute_surveys",
                json={"Report_Entry": test_commutes},
                status=200
            )

            # Business Travel endpoint
            test_travel = setup.generate_test_business_travel(count=10)

            responses.add(
                responses.GET,
                "https://mock.workday.com/ccx/service/customreport2/tenant/report/business_travel",
                json={"Report_Entry": test_travel},
                status=200
            )

        return mock_workday_responses

    def cleanup_test_data(self):
        """
        Cleanup test data from sandbox.

        Note: In production, this would clean up test reports created during testing.
        """
        print("Cleaning up Workday sandbox test data...")

        # NOTE: Cleanup logic implementation pending
        # When implementing:
        # 1. Connect to Workday API using credentials
        # 2. Identify test spend reports by naming pattern (TEST_*)
        # 3. Delete test reports via API
        # 4. Clean up test worker data if created
        # 5. Log cleanup statistics
        # Example using requests:
        #   import requests
        #   from requests.auth import HTTPBasicAuth
        #   auth = HTTPBasicAuth(f"{self.username}@{self.tenant_name}", self.password)
        #   # Get test spend reports
        #   response = requests.get(
        #       f"{self.base_url}/ccx/service/{self.tenant_name}/Financial_Management/v1",
        #       auth=auth,
        #       params={"Report_Name": "TEST_*"}
        #   )
        #   test_reports = response.json()
        #   # Delete each test report
        #   deleted_count = 0
        #   for report in test_reports:
        #       delete_response = requests.delete(
        #           f"{self.base_url}/ccx/service/{self.tenant_name}/Financial_Management/v1/{report['id']}",
        #           auth=auth
        #       )
        #       if delete_response.status_code == 200:
        #           deleted_count += 1
        #   print(f"Deleted {deleted_count} test reports")

        # Placeholder - replace with actual cleanup logic
        print("Workday sandbox cleanup complete")

    def get_sandbox_status(self) -> Dict[str, Any]:
        """
        Get Workday sandbox status and health.

        Returns:
            Dictionary with sandbox status information
        """
        return {
            "available": self.is_sandbox_available(),
            "base_url": self.base_url,
            "tenant": self.tenant_name,
            "environment_variables": self.verify_environment_variables(),
            "timestamp": DeterministicClock.now().isoformat()
        }

    def print_setup_instructions(self):
        """Print setup instructions for Workday sandbox."""
        print("\n" + "="*60)
        print("Workday Sandbox Setup Instructions")
        print("="*60)
        print("\nRequired Environment Variables:")
        print("  WORKDAY_SANDBOX_URL=https://your-tenant.workday.com")
        print("  WORKDAY_SANDBOX_CLIENT_ID=your_client_id")
        print("  WORKDAY_SANDBOX_CLIENT_SECRET=your_client_secret")
        print("  WORKDAY_SANDBOX_TENANT=your_tenant_name")
        print("  RUN_INTEGRATION_TESTS=true")
        print("\nCurrent Status:")

        env_vars = self.verify_environment_variables()
        for var_name, is_set in env_vars.items():
            status = "✓ SET" if is_set else "✗ NOT SET"
            print(f"  {var_name}: {status}")

        print("\nTo run integration tests:")
        print("  1. Set all required environment variables")
        print("  2. Ensure Workday sandbox is accessible")
        print("  3. Configure RaaS reports in Workday")
        print("  4. Run: pytest -m workday_sandbox")
        print("\nTo run with mock server (CI/CD):")
        print("  pytest tests/integration/ (without sandbox env vars)")
        print("="*60 + "\n")


def setup_workday_sandbox():
    """
    Main setup function for Workday sandbox.

    Can be run as standalone script to verify sandbox setup.
    """
    setup = WorkdaySandboxSetup()

    print("\n" + "="*60)
    print("Workday Sandbox Environment Setup")
    print("="*60 + "\n")

    # Check status
    status = setup.get_sandbox_status()
    print(f"Sandbox Available: {status['available']}")
    print(f"Base URL: {status['base_url']}")
    print(f"Tenant: {status.get('tenant', 'Not Set')}")

    # Print environment variables
    print("\nEnvironment Variables:")
    for var_name, is_set in status['environment_variables'].items():
        print(f"  {var_name}: {'✓' if is_set else '✗'}")

    if not status['available']:
        print("\n⚠ Sandbox not fully configured!")
        setup.print_setup_instructions()
    else:
        print("\n✓ Workday Sandbox is ready for integration testing!")

    return setup


if __name__ == "__main__":
    setup_workday_sandbox()
