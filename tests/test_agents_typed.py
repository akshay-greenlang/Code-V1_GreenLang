"""Fully typed test suite for GreenLang agents."""

import unittest
from typing import Dict, Any, List, Optional
from greenlang.agents import (
    FuelAgent, CarbonAgent, InputValidatorAgent,
    ReportAgent, BenchmarkAgent
)
from greenlang.agents.types import (
    FuelInput, FuelOutput,
    CarbonInput, CarbonOutput,
    ValidatorInput, ValidatorOutput,
    ReportInput, ReportOutput,
    BenchmarkInput, BenchmarkOutput
)
from greenlang.types import AgentResult


class TestFuelAgent(unittest.TestCase):
    """Test suite for FuelAgent with type hints."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.agent: FuelAgent = FuelAgent()
    
    def test_electricity_calculation(self) -> None:
        """Test electricity emissions calculation."""
        input_data: FuelInput = {
            "fuel_type": "electricity",
            "consumption": {"value": 1000, "unit": "kWh"},
            "country": "US"
        }
        result: AgentResult[FuelOutput] = self.agent.run(input_data)
        
        self.assertTrue(result["success"])
        self.assertIsNotNone(result.get("data"))
        if result["success"] and result.get("data"):
            self.assertAlmostEqual(result["data"]["co2e_emissions_kg"], 385.0, places=1)
    
    def test_natural_gas_calculation(self) -> None:
        """Test natural gas emissions calculation."""
        input_data: FuelInput = {
            "fuel_type": "natural_gas",
            "consumption": {"value": 100, "unit": "therms"},
            "country": "US"
        }
        result: AgentResult[FuelOutput] = self.agent.run(input_data)
        
        self.assertTrue(result["success"])
        if result["success"] and result.get("data"):
            self.assertAlmostEqual(result["data"]["co2e_emissions_kg"], 530.0, places=1)
    
    def test_invalid_fuel_type(self) -> None:
        """Test handling of invalid fuel type."""
        input_data: Dict[str, Any] = {
            "fuel_type": "invalid_fuel",
            "consumption": {"value": 100, "unit": "units"},
            "country": "US"
        }
        result: AgentResult[FuelOutput] = self.agent.run(input_data)
        
        self.assertFalse(result["success"])
        self.assertIsNotNone(result.get("error"))


class TestCarbonAgent(unittest.TestCase):
    """Test suite for CarbonAgent with type hints."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.agent: CarbonAgent = CarbonAgent()
    
    def test_aggregation(self) -> None:
        """Test emissions aggregation."""
        input_data: CarbonInput = {
            "emissions": [
                {"fuel": "electricity", "co2e_emissions_kg": 385.0},
                {"fuel": "natural_gas", "co2e_emissions_kg": 530.0}
            ]
        }
        result: AgentResult[CarbonOutput] = self.agent.run(input_data)
        
        self.assertTrue(result["success"])
        if result["success"] and result.get("data"):
            self.assertAlmostEqual(result["data"]["total_co2e_kg"], 915.0, places=1)
            self.assertEqual(len(result["data"]["by_fuel"]), 2)
    
    def test_empty_emissions(self) -> None:
        """Test handling of empty emissions list."""
        input_data: CarbonInput = {"emissions": []}
        result: AgentResult[CarbonOutput] = self.agent.run(input_data)
        
        self.assertTrue(result["success"])
        if result["success"] and result.get("data"):
            self.assertEqual(result["data"]["total_co2e_kg"], 0)


class TestInputValidatorAgent(unittest.TestCase):
    """Test suite for InputValidatorAgent with type hints."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.agent: InputValidatorAgent = InputValidatorAgent()
    
    def test_valid_input(self) -> None:
        """Test validation of valid input."""
        input_data: ValidatorInput = {
            "fuels": [
                {"type": "electricity", "consumption": 1000, "unit": "kWh"},
                {"type": "natural_gas", "consumption": 500, "unit": "therms"}
            ]
        }
        result: AgentResult[ValidatorOutput] = self.agent.run(input_data)
        
        self.assertTrue(result["success"])
        if result["success"] and result.get("data"):
            self.assertTrue(result["data"]["is_valid"])
            self.assertEqual(len(result["data"]["normalized_data"]["fuels"]), 2)
    
    def test_invalid_fuel_type(self) -> None:
        """Test validation of invalid fuel type."""
        input_data: Dict[str, Any] = {
            "fuels": [
                {"type": "invalid", "consumption": 1000, "unit": "kWh"}
            ]
        }
        result: AgentResult[ValidatorOutput] = self.agent.run(input_data)
        
        self.assertFalse(result["success"])
    
    def test_negative_consumption(self) -> None:
        """Test validation of negative consumption."""
        input_data: Dict[str, Any] = {
            "fuels": [
                {"type": "electricity", "consumption": -100, "unit": "kWh"}
            ]
        }
        result: AgentResult[ValidatorOutput] = self.agent.run(input_data)
        
        self.assertFalse(result["success"])


class TestReportAgent(unittest.TestCase):
    """Test suite for ReportAgent with type hints."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.agent: ReportAgent = ReportAgent()
    
    def test_text_report(self) -> None:
        """Test text format report generation."""
        input_data: ReportInput = {
            "emissions": {
                "total_co2e_tons": 10.5,
                "total_co2e_kg": 10500,
                "by_fuel": {
                    "electricity": 5500,
                    "natural_gas": 5000
                }
            },
            "format": "text"
        }
        result: AgentResult[ReportOutput] = self.agent.run(input_data)
        
        self.assertTrue(result["success"])
        if result["success"] and result.get("data"):
            self.assertIn("report_content", result["data"])
            self.assertIn("10.5", str(result["data"]["report_content"]))
    
    def test_json_report(self) -> None:
        """Test JSON format report generation."""
        input_data: ReportInput = {
            "emissions": {
                "total_co2e_tons": 10.5,
                "total_co2e_kg": 10500,
                "by_fuel": {}
            },
            "format": "json"
        }
        result: AgentResult[ReportOutput] = self.agent.run(input_data)
        
        self.assertTrue(result["success"])
        if result["success"] and result.get("data"):
            self.assertIn("report_content", result["data"])


class TestBenchmarkAgent(unittest.TestCase):
    """Test suite for BenchmarkAgent with type hints."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.agent: BenchmarkAgent = BenchmarkAgent()
    
    def test_benchmark_rating(self) -> None:
        """Test benchmark rating calculation."""
        input_data: BenchmarkInput = {
            "co2e_per_sqft": 12.0,
            "building_type": "commercial_office"
        }
        result: AgentResult[BenchmarkOutput] = self.agent.run(input_data)
        
        self.assertTrue(result["success"])
        if result["success"] and result.get("data"):
            self.assertIn("rating", result["data"])
            self.assertIn("score", result["data"])
            self.assertIn("recommendations", result["data"])
    
    def test_excellent_rating(self) -> None:
        """Test excellent rating calculation."""
        input_data: BenchmarkInput = {
            "co2e_per_sqft": 1.0,
            "building_type": "commercial_office"
        }
        result: AgentResult[BenchmarkOutput] = self.agent.run(input_data)
        
        self.assertTrue(result["success"])
        if result["success"] and result.get("data"):
            self.assertEqual(result["data"]["rating"], "Excellent")
            self.assertGreaterEqual(result["data"]["score"], 90)


if __name__ == "__main__":
    unittest.main()