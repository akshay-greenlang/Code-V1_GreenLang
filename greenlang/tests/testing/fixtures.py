"""
Test Fixtures
=============

Test fixtures and mock objects for GreenLang testing.

Author: Testing Team
Created: 2025-11-21
"""

import unittest
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path


class AgentTestCase(unittest.TestCase):
    """Base test case for testing agents."""

    def setUp(self):
        """Set up test case."""
        self.agent = None
        self.test_data = {}

    def tearDown(self):
        """Tear down test case."""
        pass

    def run_agent(self, agent_class, input_data):
        """Run an agent with test data."""
        agent = agent_class()
        return agent.process(input_data)


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "MockAgent"):
        self.name = name
        self.process_count = 0

    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process mock data."""
        self.process_count += 1
        return {
            "status": "SUCCESS",
            "data": input_data,
            "agent": self.name,
            "timestamp": datetime.now().isoformat()
        }


class MockChatSession:
    """Mock LLM chat session."""

    def __init__(self):
        self.messages = []
        self.responses = []

    def send_message(self, message: str) -> str:
        """Send message and get mock response."""
        self.messages.append(message)
        response = f"Mock response to: {message[:50]}..."
        self.responses.append(response)
        return response

    def clear(self):
        """Clear session."""
        self.messages.clear()
        self.responses.clear()


class MockDatabase:
    """Mock database for testing."""

    def __init__(self):
        self.data = {}
        self.query_count = 0

    def query(self, table: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Mock query."""
        self.query_count += 1
        return self.data.get(table, [])

    def insert(self, table: str, record: Dict) -> int:
        """Mock insert."""
        if table not in self.data:
            self.data[table] = []
        self.data[table].append(record)
        return len(self.data[table])

    def clear(self):
        """Clear database."""
        self.data.clear()
        self.query_count = 0


class MockLLMClient:
    """Mock LLM client."""

    def __init__(self):
        self.request_count = 0
        self.total_tokens = 0

    def generate(self, prompt: str) -> str:
        """Generate mock response."""
        self.request_count += 1
        self.total_tokens += len(prompt.split()) * 2
        return f"Mock LLM response for: {prompt[:30]}..."

    def classify(self, text: str, categories: List[str]) -> str:
        """Mock classification."""
        return categories[0] if categories else "unknown"


def create_test_data(data_type: str = "emissions") -> Dict[str, Any]:
    """
    Create test data for various scenarios.

    Args:
        data_type: Type of test data to create

    Returns:
        Test data dictionary
    """
    test_data = {
        "emissions": {
            "activity_data": {
                "fuel_consumption": 1000,
                "electricity_usage": 5000,
                "waste_generated": 200
            },
            "emission_factors": {
                "fuel": 2.5,
                "electricity": 0.5,
                "waste": 0.3
            },
            "expected_total": 3560.0
        },
        "supplier": {
            "supplier_id": "SUP001",
            "name": "Test Supplier",
            "country": "USA",
            "emissions_data": {
                "scope1": 1000,
                "scope2": 500,
                "scope3": 2000
            }
        },
        "shipment": {
            "shipment_id": "SHIP001",
            "origin": "China",
            "destination": "USA",
            "weight": 1000,
            "product_type": "steel",
            "transport_mode": "sea"
        }
    }

    return test_data.get(data_type, {})


def load_test_fixture(fixture_name: str) -> Any:
    """
    Load test fixture from file.

    Args:
        fixture_name: Name of fixture to load

    Returns:
        Loaded fixture data
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixture_path = fixtures_dir / f"{fixture_name}.json"

    if fixture_path.exists():
        with open(fixture_path, 'r') as f:
            return json.load(f)

    # Return default fixture if file doesn't exist
    return create_test_data(fixture_name)
