#!/usr/bin/env python
"""Test script to verify orchestrator functionality"""

from greenlang.core.orchestrator import Orchestrator
from greenlang.core.workflow import Workflow, WorkflowStep
from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult

class TestAgent(BaseAgent):
    """Simple test agent"""
    
    def __init__(self, agent_id: str):
        config = AgentConfig(
            name=agent_id,
            description=f"Test agent {agent_id}",
            version="1.0.0"
        )
        super().__init__(config)
    
    def execute(self, input_data: dict) -> dict:
        """Execute the test agent"""
        return {
            "success": True,
            "data": {
                "message": f"Processed by {self.config.name}",
                "input": input_data
            }
        }

def test_basic_workflow():
    """Test basic workflow execution"""
    print("Testing basic workflow execution...")
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Register test agents
    agent1 = TestAgent("agent1")
    agent2 = TestAgent("agent2")
    orchestrator.register_agent("agent1", agent1)
    orchestrator.register_agent("agent2", agent2)
    
    # Create workflow
    workflow = Workflow(
        name="test_workflow",
        description="Test workflow",
        steps=[
            WorkflowStep(
                name="step1",
                agent_id="agent1",
                description="First step"
            ),
            WorkflowStep(
                name="step2",
                agent_id="agent2",
                description="Second step"
            )
        ]
    )
    
    # Register workflow
    orchestrator.register_workflow("test_workflow", workflow)
    
    # Execute workflow
    result = orchestrator.execute_workflow("test_workflow", {"test": "data"})
    
    print(f"Success: {result['success']}")
    print(f"Errors: {result['errors']}")
    print(f"Results: {result.get('results', {})}")
    
    assert result['success'] == True, "Workflow should succeed"
    assert len(result['errors']) == 0, "Should have no errors"
    print("Basic workflow test PASSED!")

def test_retry_logic():
    """Test retry logic"""
    print("\nTesting retry logic...")
    
    class FailingAgent(BaseAgent):
        """Agent that fails initially then succeeds"""
        
        def __init__(self):
            config = AgentConfig(
                name="failing_agent",
                description="Agent that fails then succeeds",
                version="1.0.0"
            )
            super().__init__(config)
            self.attempt = 0
        
        def execute(self, input_data: dict) -> dict:
            self.attempt += 1
            if self.attempt < 2:
                return {
                    "success": False,
                    "error": f"Failed on attempt {self.attempt}"
                }
            return {
                "success": True,
                "data": {"message": f"Succeeded on attempt {self.attempt}"}
            }
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Register failing agent
    failing_agent = FailingAgent()
    orchestrator.register_agent("failing", failing_agent)
    
    # Create workflow with retry
    workflow = Workflow(
        name="retry_workflow",
        description="Test retry workflow",
        steps=[
            WorkflowStep(
                name="retry_step",
                agent_id="failing",
                description="Step with retry",
                retry_count=2  # Allow 2 retries
            )
        ]
    )
    
    # Register and execute
    orchestrator.register_workflow("retry_workflow", workflow)
    result = orchestrator.execute_workflow("retry_workflow", {})
    
    print(f"Success: {result['success']}")
    print(f"Errors: {result['errors']}")
    
    assert result['success'] == True, "Workflow should succeed after retry"
    assert len(result['errors']) == 0, "Should have no errors after retry"
    print("Retry logic test PASSED!")

def test_conditional_execution():
    """Test conditional step execution"""
    print("\nTesting conditional execution...")
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Register test agents
    agent1 = TestAgent("checker")
    agent2 = TestAgent("conditional")
    orchestrator.register_agent("checker", agent1)
    orchestrator.register_agent("conditional", agent2)
    
    # Create workflow with conditional step
    workflow = Workflow(
        name="conditional_workflow",
        description="Test conditional workflow",
        steps=[
            WorkflowStep(
                name="check",
                agent_id="checker",
                description="Check step"
            ),
            WorkflowStep(
                name="conditional",
                agent_id="conditional",
                description="Conditional step",
                condition="results['check']['success'] == True"
            )
        ]
    )
    
    # Register and execute
    orchestrator.register_workflow("conditional_workflow", workflow)
    result = orchestrator.execute_workflow("conditional_workflow", {})
    
    print(f"Success: {result['success']}")
    print(f"Steps executed: {list(result.get('results', {}).keys())}")
    
    assert result['success'] == True, "Workflow should succeed"
    assert 'conditional' in result.get('results', {}), "Conditional step should execute"
    print("Conditional execution test PASSED!")

def test_safe_eval():
    """Test safe expression evaluation"""
    print("\nTesting safe expression evaluation...")
    
    orchestrator = Orchestrator()
    
    # Test various safe expressions
    context = {
        "input": {"value": 10},
        "results": {"step1": {"success": True, "data": {"count": 5}}}
    }
    
    test_cases = [
        ("input['value'] > 5", True),
        ("input['value'] == 10", True),
        ("input['value'] < 5", False),
        ("results['step1']['success'] == True", True),
        ("results['step1']['data']['count'] >= 5", True),
    ]
    
    for expr, expected in test_cases:
        try:
            result = orchestrator._evaluate_condition(expr, context)
            assert result == expected, f"Expression '{expr}' should evaluate to {expected}"
            print(f"  [OK] '{expr}' = {result}")
        except Exception as e:
            print(f"  [FAIL] '{expr}' raised: {e}")
            raise
    
    # Test that dangerous expressions are blocked
    dangerous_exprs = [
        "__import__('os').system('echo test')",
        "exec('print(1)')",
        "eval('1+1')",
    ]
    
    for expr in dangerous_exprs:
        try:
            result = orchestrator._evaluate_condition(expr, context)
            print(f"  [FAIL] Dangerous expression '{expr}' was not blocked!")
            assert False, f"Should have blocked: {expr}"
        except (ValueError, SyntaxError):
            print(f"  [OK] Blocked dangerous expression: '{expr}'")
    
    print("Safe eval test PASSED!")

def main():
    """Run all tests"""
    print("="*60)
    print("ORCHESTRATOR FUNCTIONALITY TESTS")
    print("="*60)
    
    try:
        test_basic_workflow()
        test_retry_logic()
        test_conditional_execution()
        test_safe_eval()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        return 0
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())