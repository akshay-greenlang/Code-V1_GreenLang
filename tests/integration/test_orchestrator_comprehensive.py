#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprehensive test to identify any issues in orchestrator.py"""

import sys
import traceback
from greenlang.core.orchestrator import Orchestrator
from greenlang.core.workflow import Workflow, WorkflowStep
from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult

def test_imports():
    """Test all imports work"""
    print("Testing imports...")
    try:
        from greenlang.core.orchestrator import Orchestrator
        from greenlang.agents.base import BaseAgent, AgentResult
        from greenlang.core.workflow import Workflow
        import logging
        import json
        import ast
        import operator
        print("  [OK] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False

def test_class_instantiation():
    """Test class can be instantiated"""
    print("\nTesting class instantiation...")
    try:
        o = Orchestrator()
        assert hasattr(o, 'agents')
        assert hasattr(o, 'workflows')
        assert hasattr(o, 'execution_history')
        assert hasattr(o, 'logger')
        print("  [OK] Orchestrator instantiated with all attributes")
        return True
    except Exception as e:
        print(f"  [FAIL] Instantiation error: {e}")
        traceback.print_exc()
        return False

def test_agent_registration():
    """Test agent registration"""
    print("\nTesting agent registration...")
    try:
        o = Orchestrator()
        
        class TestAgent(BaseAgent):
            def __init__(self):
                config = AgentConfig(name="test", description="Test", version="1.0")
                super().__init__(config)
            
            def execute(self, input_data: dict) -> dict:
                return {"success": True, "data": {}}
        
        agent = TestAgent()
        o.register_agent("test", agent)
        
        assert "test" in o.agents
        assert o.agents["test"] == agent
        print("  [OK] Agent registration works")
        return True
    except Exception as e:
        print(f"  [FAIL] Agent registration error: {e}")
        traceback.print_exc()
        return False

def test_workflow_registration():
    """Test workflow registration"""
    print("\nTesting workflow registration...")
    try:
        o = Orchestrator()
        workflow = Workflow(
            name="test",
            description="Test workflow",
            steps=[]
        )
        o.register_workflow("test", workflow)
        
        assert "test" in o.workflows
        assert o.workflows["test"] == workflow
        print("  [OK] Workflow registration works")
        return True
    except Exception as e:
        print(f"  [FAIL] Workflow registration error: {e}")
        traceback.print_exc()
        return False

def test_single_agent_execution():
    """Test single agent execution"""
    print("\nTesting single agent execution...")
    try:
        o = Orchestrator()
        
        class TestAgent(BaseAgent):
            def __init__(self):
                config = AgentConfig(name="test", description="Test", version="1.0")
                super().__init__(config)
            
            def execute(self, input_data: dict) -> dict:
                return {"success": True, "data": {"echo": input_data}}
        
        agent = TestAgent()
        o.register_agent("test", agent)
        
        result = o.execute_single_agent("test", {"message": "hello"})
        
        assert isinstance(result, dict)
        assert "success" in result or "data" in result
        print(f"  [OK] Single agent execution works: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] Single agent execution error: {e}")
        traceback.print_exc()
        return False

def test_workflow_execution():
    """Test workflow execution"""
    print("\nTesting workflow execution...")
    try:
        o = Orchestrator()
        
        class TestAgent(BaseAgent):
            def __init__(self, name):
                config = AgentConfig(name=name, description="Test", version="1.0")
                super().__init__(config)
            
            def execute(self, input_data: dict) -> dict:
                return {
                    "success": True, 
                    "data": {"processed_by": self.config.name, "input": input_data}
                }
        
        agent1 = TestAgent("agent1")
        agent2 = TestAgent("agent2")
        o.register_agent("agent1", agent1)
        o.register_agent("agent2", agent2)
        
        workflow = Workflow(
            name="test",
            description="Test workflow",
            steps=[
                WorkflowStep(name="step1", agent_id="agent1", description="Step 1"),
                WorkflowStep(name="step2", agent_id="agent2", description="Step 2")
            ]
        )
        o.register_workflow("test", workflow)
        
        result = o.execute_workflow("test", {"initial": "data"})
        
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] == True
        print(f"  [OK] Workflow execution works")
        print(f"      Success: {result['success']}")
        print(f"      Errors: {result.get('errors', [])}")
        return True
    except Exception as e:
        print(f"  [FAIL] Workflow execution error: {e}")
        traceback.print_exc()
        return False

def test_retry_mechanism():
    """Test retry mechanism"""
    print("\nTesting retry mechanism...")
    try:
        o = Orchestrator()
        
        class FailingAgent(BaseAgent):
            def __init__(self):
                config = AgentConfig(name="failing", description="Fails first", version="1.0")
                super().__init__(config)
                self.attempts = 0
            
            def execute(self, input_data: dict) -> dict:
                self.attempts += 1
                if self.attempts < 2:
                    return {"success": False, "error": f"Attempt {self.attempts} failed"}
                return {"success": True, "data": {"attempts": self.attempts}}
        
        agent = FailingAgent()
        o.register_agent("failing", agent)
        
        workflow = Workflow(
            name="retry_test",
            description="Test retry",
            steps=[
                WorkflowStep(
                    name="retry_step",
                    agent_id="failing",
                    description="Step with retry",
                    retry_count=2
                )
            ]
        )
        o.register_workflow("retry_test", workflow)
        
        result = o.execute_workflow("retry_test", {})
        
        assert result["success"] == True
        assert agent.attempts == 2
        print(f"  [OK] Retry mechanism works (took {agent.attempts} attempts)")
        return True
    except Exception as e:
        print(f"  [FAIL] Retry mechanism error: {e}")
        traceback.print_exc()
        return False

def test_condition_evaluation():
    """Test condition evaluation"""
    print("\nTesting condition evaluation...")
    try:
        o = Orchestrator()
        context = {
            "input": {"value": 10, "flag": True},
            "results": {
                "step1": {"success": True, "data": {"count": 5}}
            }
        }
        
        test_cases = [
            ("input['value'] > 5", True),
            ("input['value'] == 10", True),
            ("input['flag'] == True", True),
            ("results['step1']['success'] == True", True),
            ("results['step1']['data']['count'] < 10", True),
            ("input['value'] < 5", False),
            ("input['flag'] == False", False),
        ]
        
        all_passed = True
        for expr, expected in test_cases:
            try:
                result = o._evaluate_condition(expr, context)
                if result == expected:
                    print(f"  [OK] '{expr}' = {result}")
                else:
                    print(f"  [FAIL] '{expr}' expected {expected}, got {result}")
                    all_passed = False
            except Exception as e:
                print(f"  [FAIL] '{expr}' raised: {e}")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"  [FAIL] Condition evaluation error: {e}")
        traceback.print_exc()
        return False

def test_dangerous_expressions_blocked():
    """Test that dangerous expressions are blocked"""
    print("\nTesting dangerous expression blocking...")
    try:
        o = Orchestrator()
        context = {"input": {}, "results": {}}
        
        dangerous = [
            "__import__('os').system('ls')",
            "exec('print(1)')",
            "eval('1+1')",
            "compile('1+1', 'string', 'eval')",
        ]
        
        all_blocked = True
        for expr in dangerous:
            try:
                result = o._evaluate_condition(expr, context)
                print(f"  [FAIL] Dangerous expression not blocked: '{expr}'")
                all_blocked = False
            except (ValueError, SyntaxError, AttributeError) as e:
                print(f"  [OK] Blocked: '{expr}' ({type(e).__name__})")
            except Exception as e:
                print(f"  [WARN] Unexpected error for '{expr}': {e}")
        
        return all_blocked
    except Exception as e:
        print(f"  [FAIL] Security test error: {e}")
        traceback.print_exc()
        return False

def test_agent_info():
    """Test get_agent_info method"""
    print("\nTesting get_agent_info...")
    try:
        o = Orchestrator()
        
        class TestAgent(BaseAgent):
            def __init__(self):
                config = AgentConfig(
                    name="test_agent",
                    description="A test agent",
                    version="1.0.0",
                    enabled=True
                )
                super().__init__(config)
            
            def execute(self, input_data: dict) -> dict:
                return {"success": True}
        
        agent = TestAgent()
        o.register_agent("test", agent)
        
        info = o.get_agent_info("test")
        assert info is not None
        assert info["name"] == "test_agent"
        assert info["description"] == "A test agent"
        assert info["version"] == "1.0.0"
        assert info["enabled"] == True
        
        print(f"  [OK] get_agent_info works: {info}")
        return True
    except Exception as e:
        print(f"  [FAIL] get_agent_info error: {e}")
        traceback.print_exc()
        return False

def test_workflow_info():
    """Test get_workflow_info method"""
    print("\nTesting get_workflow_info...")
    try:
        o = Orchestrator()
        
        workflow = Workflow(
            name="test_workflow",
            description="A test workflow",
            steps=[
                WorkflowStep(name="s1", agent_id="a1", description="Step 1"),
                WorkflowStep(name="s2", agent_id="a2", description="Step 2")
            ]
        )
        o.register_workflow("test", workflow)
        
        info = o.get_workflow_info("test")
        assert info is not None
        assert info["name"] == "test_workflow"
        assert info["description"] == "A test workflow"
        assert len(info["steps"]) == 2
        assert info["steps"][0]["name"] == "s1"
        
        print(f"  [OK] get_workflow_info works")
        print(f"      Name: {info['name']}")
        print(f"      Steps: {len(info['steps'])}")
        return True
    except Exception as e:
        print(f"  [FAIL] get_workflow_info error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*70)
    print("COMPREHENSIVE ORCHESTRATOR TESTING")
    print("="*70)
    
    tests = [
        test_imports,
        test_class_instantiation,
        test_agent_registration,
        test_workflow_registration,
        test_single_agent_execution,
        test_workflow_execution,
        test_retry_mechanism,
        test_condition_evaluation,
        test_dangerous_expressions_blocked,
        test_agent_info,
        test_workflow_info,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\nUnexpected error in {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("ALL TESTS PASSED!")
        print("="*70)
        return 0
    else:
        print("SOME TESTS FAILED - Please review the output above")
        print("="*70)
        return 1

if __name__ == "__main__":
    sys.exit(main())