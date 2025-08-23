import unittest
from greenlang.agents import (
    FuelAgent, CarbonAgent, InputValidatorAgent,
    ReportAgent, BenchmarkAgent
)


class TestFuelAgent(unittest.TestCase):
    def setUp(self):
        self.agent = FuelAgent()
    
    def test_electricity_calculation(self):
        input_data = {
            "fuel_type": "electricity",
            "consumption": 1000,
            "unit": "kWh",
            "region": "US"
        }
        result = self.agent.run(input_data)
        
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.data["co2e_emissions_kg"], 385.0, places=1)
    
    def test_natural_gas_calculation(self):
        input_data = {
            "fuel_type": "natural_gas",
            "consumption": 100,
            "unit": "therms",
            "region": "US"
        }
        result = self.agent.run(input_data)
        
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.data["co2e_emissions_kg"], 530.0, places=1)
    
    def test_invalid_fuel_type(self):
        input_data = {
            "fuel_type": "invalid_fuel",
            "consumption": 100,
            "unit": "units"
        }
        result = self.agent.run(input_data)
        
        self.assertFalse(result.success)


class TestCarbonAgent(unittest.TestCase):
    def setUp(self):
        self.agent = CarbonAgent()
    
    def test_aggregation(self):
        input_data = {
            "emissions": [
                {"fuel_type": "electricity", "co2e_emissions_kg": 385.0},
                {"fuel_type": "natural_gas", "co2e_emissions_kg": 530.0}
            ]
        }
        result = self.agent.run(input_data)
        
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.data["total_co2e_kg"], 915.0, places=1)
        self.assertEqual(len(result.data["emissions_breakdown"]), 2)
    
    def test_empty_emissions(self):
        input_data = {"emissions": []}
        result = self.agent.run(input_data)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["total_co2e_kg"], 0)


class TestInputValidatorAgent(unittest.TestCase):
    def setUp(self):
        self.agent = InputValidatorAgent()
    
    def test_valid_input(self):
        input_data = {
            "fuels": [
                {"type": "electricity", "consumption": 1000, "unit": "kWh"},
                {"type": "natural_gas", "consumption": 500, "unit": "therms"}
            ]
        }
        result = self.agent.run(input_data)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["validated_data"]["fuels"]), 2)
    
    def test_invalid_fuel_type(self):
        input_data = {
            "fuels": [
                {"type": "invalid", "consumption": 1000, "unit": "kWh"}
            ]
        }
        result = self.agent.run(input_data)
        
        self.assertFalse(result.success)
        self.assertIn("errors", result.data)
    
    def test_negative_consumption(self):
        input_data = {
            "fuels": [
                {"type": "electricity", "consumption": -100, "unit": "kWh"}
            ]
        }
        result = self.agent.run(input_data)
        
        self.assertFalse(result.success)


class TestReportAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ReportAgent()
    
    def test_text_report(self):
        input_data = {
            "carbon_data": {
                "total_co2e_tons": 10.5,
                "total_co2e_kg": 10500,
                "emissions_breakdown": [
                    {"source": "electricity", "co2e_tons": 5.5, "percentage": 52.4},
                    {"source": "natural_gas", "co2e_tons": 5.0, "percentage": 47.6}
                ]
            },
            "format": "text"
        }
        result = self.agent.run(input_data)
        
        self.assertTrue(result.success)
        self.assertIn("CARBON FOOTPRINT REPORT", result.data["report"])
        self.assertIn("10.500", result.data["report"])
    
    def test_json_report(self):
        input_data = {
            "carbon_data": {"total_co2e_tons": 10.5},
            "format": "json"
        }
        result = self.agent.run(input_data)
        
        self.assertTrue(result.success)
        self.assertIsInstance(result.data["report"], dict)
        self.assertEqual(result.data["report"]["emissions"]["total"]["value"], 10.5)


class TestBenchmarkAgent(unittest.TestCase):
    def setUp(self):
        self.agent = BenchmarkAgent()
    
    def test_benchmark_rating(self):
        input_data = {
            "total_emissions_kg": 10000,
            "building_area": 10000,
            "building_type": "commercial_office",
            "period_months": 1
        }
        result = self.agent.run(input_data)
        
        self.assertTrue(result.success)
        self.assertIn("rating", result.data)
        self.assertIn("percentile", result.data)
        self.assertIn("recommendations", result.data)
    
    def test_excellent_rating(self):
        input_data = {
            "total_emissions_kg": 1000,
            "building_area": 10000,
            "building_type": "commercial_office",
            "period_months": 12
        }
        result = self.agent.run(input_data)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["rating"], "Excellent")
        self.assertEqual(result.data["percentile"], 90)


if __name__ == "__main__":
    unittest.main()