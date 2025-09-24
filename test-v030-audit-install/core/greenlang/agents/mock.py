"""
Mock Agent for Testing
=======================

A simple mock agent that returns predictable outputs for testing.
"""

from typing import Dict, Any


class MockAgent:
    """Mock agent for testing pipelines"""

    def __init__(self):
        self.name = "mock"
        self.version = "1.0.0"
        self.call_count = 0

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute mock operation"""
        self.call_count += 1

        # Return input data with some transformation
        result = {
            "status": "success",
            "call_count": self.call_count,
            "received_inputs": kwargs,
            "mock_output": "test_output_value",
        }

        # If data is provided, echo it back
        if "data" in kwargs:
            result["processed_data"] = f"processed_{kwargs['data']}"

        # If expression is provided (for calculator mock)
        if "expression" in kwargs:
            try:
                import ast
                # Only evaluate literal expressions safely
                result["result"] = ast.literal_eval(kwargs["expression"])
            except (SyntaxError, ValueError, NameError, ZeroDivisionError) as e:
                result["result"] = "error"

        return result

    def validate_inputs(self, **kwargs) -> bool:
        """Validate inputs (always returns True for mock)"""
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata"""
        return {
            "name": self.name,
            "version": self.version,
            "type": "mock",
            "description": "Mock agent for testing",
        }


# Module-level function for direct execution
def execute(**kwargs) -> Dict[str, Any]:
    """Direct execution function for agent loading"""
    agent = MockAgent()
    return agent.execute(**kwargs)


# Compatibility with different loading patterns
def run(**kwargs) -> Dict[str, Any]:
    """Alternative entry point"""
    return execute(**kwargs)


# For class-based loading
Agent = MockAgent
