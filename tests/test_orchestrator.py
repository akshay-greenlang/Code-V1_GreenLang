# -*- coding: utf-8 -*-
import unittest
from greenlang.core.orchestrator import Orchestrator
from greenlang.core.workflow import Workflow, WorkflowStep
from greenlang.agents import FuelAgent, CarbonAgent, ReportAgent


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orchestrator = Orchestrator()
        self.orchestrator.register_agent("fuel", FuelAgent())
        self.orchestrator.register_agent("carbon", CarbonAgent())
        self.orchestrator.register_agent("report", ReportAgent())
    
    def test_simple_workflow(self):
        workflow = Workflow(
            name="test_workflow",
            description="Test workflow",
            steps=[
                WorkflowStep(name="calc", agent_id="fuel"),
                WorkflowStep(name="agg", agent_id="carbon")
            ]
        )
        
        self.orchestrator.register_workflow("test", workflow)
        
        input_data = {
            "fuel_type": "electricity",
            "consumption": 1000,
            "unit": "kWh",
            "emissions": []
        }
        
        result = self.orchestrator.execute_workflow("test", input_data)
        
        self.assertTrue(result["success"])
        self.assertIn("results", result)
    
    def test_workflow_with_input_mapping(self):
        workflow = Workflow(
            name="mapped_workflow",
            description="Workflow with input mapping",
            steps=[
                WorkflowStep(
                    name="calc",
                    agent_id="fuel",
                    input_mapping={
                        "fuel_type": "input.fuel.type",
                        "consumption": "input.fuel.amount",
                        "unit": "input.fuel.unit"
                    }
                )
            ]
        )
        
        self.orchestrator.register_workflow("mapped", workflow)
        
        input_data = {
            "fuel": {
                "type": "electricity",
                "amount": 500,
                "unit": "kWh"
            }
        }
        
        result = self.orchestrator.execute_workflow("mapped", input_data)
        
        self.assertTrue(result["success"])
    
    def test_workflow_with_condition(self):
        workflow = Workflow(
            name="conditional_workflow",
            description="Workflow with conditions",
            steps=[
                WorkflowStep(name="step1", agent_id="fuel"),
                WorkflowStep(
                    name="step2",
                    agent_id="carbon",
                    condition="context['results']['step1']['success']"
                )
            ]
        )
        
        self.orchestrator.register_workflow("conditional", workflow)
        
        input_data = {
            "fuel_type": "electricity",
            "consumption": 1000,
            "unit": "kWh",
            "emissions": []
        }
        
        result = self.orchestrator.execute_workflow("conditional", input_data)

        self.assertTrue(result["success"])

    def test_malicious_condition_rejected(self):
        workflow = Workflow(
            name="malicious_workflow",
            description="Workflow with malicious condition",
            steps=[
                WorkflowStep(name="step1", agent_id="fuel"),
                WorkflowStep(
                    name="step2",
                    agent_id="carbon",
                    condition="__import__('os').system('echo hacked')"
                )
            ]
        )

        self.orchestrator.register_workflow("malicious", workflow)

        input_data = {
            "fuel_type": "electricity",
            "consumption": 1000,
            "unit": "kWh",
            "emissions": []
        }

        result = self.orchestrator.execute_workflow("malicious", input_data)

        self.assertTrue(result["success"])
        self.assertNotIn("step2", result["results"])
    
    def test_workflow_error_handling(self):
        workflow = Workflow(
            name="error_workflow",
            description="Workflow with error handling",
            steps=[
                WorkflowStep(
                    name="fail_step",
                    agent_id="fuel",
                    on_failure="skip"
                ),
                WorkflowStep(
                    name="continue_step",
                    agent_id="carbon"
                )
            ]
        )
        
        self.orchestrator.register_workflow("error", workflow)
        
        input_data = {"emissions": []}
        
        result = self.orchestrator.execute_workflow("error", input_data)
        
        self.assertIn("errors", result)
        self.assertEqual(len(result["errors"]), 1)
    
    def test_list_agents(self):
        agents = self.orchestrator.list_agents()
        
        self.assertIn("fuel", agents)
        self.assertIn("carbon", agents)
        self.assertIn("report", agents)
    
    def test_get_agent_info(self):
        info = self.orchestrator.get_agent_info("fuel")
        
        self.assertIsNotNone(info)
        self.assertEqual(info["id"], "fuel")
        self.assertEqual(info["name"], "FuelAgent")


if __name__ == "__main__":
    unittest.main()
