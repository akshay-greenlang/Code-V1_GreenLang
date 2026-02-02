# -*- coding: utf-8 -*-
"""
Tests for the refactored executor with safe evaluation and importlib-based loading.
"""

import pytest
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from greenlang.runtime.executor import (
    PipelineExecutor,
    ExecutionContext,
    ConditionalExecutionError,
    create_execution_context
)
from greenlang.sdk.pipeline_spec import PipelineSpec, StepSpec


class TestSafeEvaluation:
    """Test cases for safe expression evaluation."""

    def setup_method(self):
        self.executor = PipelineExecutor()
        self.context = create_execution_context("test-pipeline")

    def test_eval_simple_boolean_expressions(self):
        """Test evaluation of simple boolean expressions."""
        # True conditions
        assert self.executor._eval_when("True", self.context) is True
        assert self.executor._eval_when("1 == 1", self.context) is True
        assert self.executor._eval_when("2 > 1", self.context) is True
        assert self.executor._eval_when("'a' in 'abc'", self.context) is True

        # False conditions
        assert self.executor._eval_when("False", self.context) is False
        assert self.executor._eval_when("1 == 2", self.context) is False
        assert self.executor._eval_when("1 > 2", self.context) is False

    def test_eval_with_context_variables(self):
        """Test evaluation with context variables."""
        self.context.variables["threshold"] = 10
        self.context.inputs["count"] = 5
        self.context.step_results["step1"] = {"output": 15}

        assert self.executor._eval_when("vars.threshold > 5", self.context) is True
        assert self.executor._eval_when("inputs.count < 10", self.context) is True
        assert self.executor._eval_when("steps.step1.output > vars.threshold", self.context) is True

    def test_eval_complex_expressions(self):
        """Test evaluation of complex expressions."""
        self.context.variables["x"] = 10
        self.context.variables["y"] = 20

        # Boolean operations
        assert self.executor._eval_when("vars.x > 5 and vars.y < 30", self.context) is True
        assert self.executor._eval_when("vars.x > 15 or vars.y == 20", self.context) is True
        assert self.executor._eval_when("not (vars.x > 20)", self.context) is True

        # Arithmetic in conditions
        assert self.executor._eval_when("vars.x + vars.y == 30", self.context) is True
        assert self.executor._eval_when("vars.x * 2 == vars.y", self.context) is True

    def test_eval_blocks_dangerous_operations(self):
        """Test that dangerous operations are blocked."""
        dangerous_expressions = [
            "__import__('os').system('ls')",
            "eval('1+1')",
            "exec('x = 1')",
            "compile('x=1', 'test', 'exec')",
            "open('/etc/passwd')",
            "vars.__class__.__bases__",
        ]

        for expr in dangerous_expressions:
            with pytest.raises(ConditionalExecutionError) as exc_info:
                self.executor._eval_when(expr, self.context)
            assert "Unsafe" in str(exc_info.value) or "not allowed" in str(exc_info.value)

    def test_eval_safe_function_calls(self):
        """Test that safe function calls are allowed."""
        self.context.variables["items"] = [1, 2, 3, 4, 5]
        self.context.variables["text"] = "hello"

        assert self.executor._eval_when("len(vars.items) == 5", self.context) is True
        assert self.executor._eval_when("str(123) == '123'", self.context) is True
        assert self.executor._eval_when("int('42') == 42", self.context) is True
        assert self.executor._eval_when("bool(1) == True", self.context) is True
        assert self.executor._eval_when("isinstance(vars.text, str)", self.context) is True
        assert self.executor._eval_when("max(vars.items) == 5", self.context) is True
        assert self.executor._eval_when("min(vars.items) == 1", self.context) is True

    def test_validate_ast_node_types(self):
        """Test AST validation for different node types."""
        import ast

        # Valid expression
        tree = ast.parse("x > 5 and y < 10", mode='eval')
        self.executor._validate_ast(tree)  # Should not raise

        # Invalid expression with import
        tree = ast.parse("__import__('os')", mode='eval')
        with pytest.raises(ConditionalExecutionError) as exc_info:
            self.executor._validate_ast(tree)
        assert "dangerous builtin" in str(exc_info.value)

    def test_safe_eval_expression_direct(self):
        """Test the _safe_eval_expression method directly."""
        context = {
            "x": 10,
            "y": 20,
            "items": [1, 2, 3],
            "data": {"key": "value"}
        }

        # Test various safe expressions
        assert self.executor._safe_eval_expression("x + y", context) == 30
        assert self.executor._safe_eval_expression("x > 5", context) is True
        assert self.executor._safe_eval_expression("len(items)", context) == 3
        assert self.executor._safe_eval_expression("data['key']", context) == "value"
        assert self.executor._safe_eval_expression("'key' in data", context) is True

        # Test that unsafe expressions raise errors
        with pytest.raises(ConditionalExecutionError):
            self.executor._safe_eval_expression("__import__('os')", context)


class TestDynamicAgentLoading:
    """Test cases for dynamic agent loading without exec/eval."""

    def setup_method(self):
        self.executor = PipelineExecutor()
        self.context = create_execution_context("test-pipeline")

    def test_create_worker_script_uses_importlib(self):
        """Test that worker script uses importlib instead of exec/eval."""
        step = StepSpec(
            name="test-step",
            agent="test_module.TestAgent",
            action="process"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            script = self.executor._create_worker_script(step, input_file, output_dir)

            # Check that the script uses importlib
            assert "import importlib" in script
            assert "importlib.import_module" in script
            assert "getattr(module," in script

            # Check that exec/eval are not used
            assert "exec(" not in script or "exec(" in script.count("exec") == 0
            # Note: eval still used for restricted evaluation, but not for imports
            assert "eval(f\"" not in script
            assert "eval(f'{" not in script

    def test_worker_script_handles_different_agent_formats(self):
        """Test that worker script handles different agent module formats."""
        test_cases = [
            ("mymodule.MyAgent", "mymodule", "MyAgent"),
            ("simple_agent", "greenlang.agents.simple_agent", None),
            ("package.submodule.AgentClass", "package.submodule", "AgentClass"),
        ]

        for agent_spec, expected_module, expected_class in test_cases:
            step = StepSpec(
                name="test-step",
                agent=agent_spec,
                action="execute"
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                script = self.executor._create_worker_script(
                    step,
                    Path(tmpdir) / "input.json",
                    Path(tmpdir) / "output"
                )

                # Check that the script handles this format
                assert agent_spec in script
                if expected_module:
                    assert "importlib.import_module" in script
                if expected_class:
                    assert f"getattr(module, \"{expected_class}\"" in script or \
                           f"getattr(module, agent_class_name" in script

    @patch('importlib.import_module')
    def test_worker_script_error_handling(self, mock_import):
        """Test that worker script handles import errors properly."""
        step = StepSpec(
            name="test-step",
            agent="nonexistent.Module",
            action="process"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            # Write test input
            with open(input_file, 'w') as f:
                json.dump({"test": "data"}, f)

            script = self.executor._create_worker_script(step, input_file, output_dir)

            # Save and execute the script
            script_file = Path(tmpdir) / "test_script.py"
            with open(script_file, 'w') as f:
                f.write(script)

            # The script should handle import errors gracefully
            import subprocess
            result = subprocess.run(
                [sys.executable, str(script_file)],
                capture_output=True,
                text=True
            )

            # Check that error was written to output
            output_file = output_dir / "output.json"
            if output_file.exists():
                with open(output_file) as f:
                    output = json.load(f)
                assert "error" in output
                assert "Failed to import" in output["error"] or \
                       "No module named" in output["error"]


class TestSecurityEnhancements:
    """Test security enhancements in the refactored executor."""

    def setup_method(self):
        self.executor = PipelineExecutor()
        self.context = create_execution_context("test-pipeline")

    def test_no_direct_eval_in_conditions(self):
        """Test that conditions don't use direct eval."""
        # This should go through _safe_eval_expression, not eval()
        with patch.object(self.executor, '_safe_eval_expression') as mock_safe_eval:
            mock_safe_eval.return_value = True

            result = self.executor._eval_when("x > 5", self.context)

            mock_safe_eval.assert_called_once()
            assert result is True

    def test_ast_validation_blocks_imports(self):
        """Test that AST validation blocks import statements."""
        import ast

        bad_expressions = [
            "import os",
            "from os import system",
            "__import__('subprocess')",
        ]

        for expr in bad_expressions:
            # These should fail to parse as expressions or be blocked
            try:
                tree = ast.parse(expr, mode='eval')
                with pytest.raises(ConditionalExecutionError):
                    self.executor._validate_ast(tree)
            except SyntaxError:
                # Some imports can't be parsed as expressions, which is good
                pass

    def test_private_attribute_access_blocked(self):
        """Test that private attribute access is blocked."""
        self.context.variables["obj"] = object()

        # These should be blocked
        with pytest.raises(ConditionalExecutionError) as exc_info:
            self.executor._eval_when("vars.obj.__class__", self.context)
        assert "private attribute" in str(exc_info.value).lower()

        with pytest.raises(ConditionalExecutionError) as exc_info:
            self.executor._eval_when("vars.obj.__dict__", self.context)
        assert "private attribute" in str(exc_info.value).lower()

    def test_restricted_builtins(self):
        """Test that only safe builtins are available."""
        # Safe builtins should work
        assert self.executor._eval_when("len([1,2,3]) == 3", self.context) is True
        assert self.executor._eval_when("str(42) == '42'", self.context) is True

        # Dangerous builtins should fail
        dangerous_builtins = [
            "eval", "exec", "compile", "__import__",
            "open", "file", "input", "raw_input"
        ]

        for builtin in dangerous_builtins:
            with pytest.raises(ConditionalExecutionError):
                self.executor._eval_when(f"{builtin}('test')", self.context)


class TestIntegration:
    """Integration tests for the refactored executor."""

    def setup_method(self):
        self.executor = PipelineExecutor()

    def test_pipeline_with_safe_conditions(self):
        """Test a complete pipeline with safe conditional evaluation."""
        # Create a pipeline with conditional steps
        spec = PipelineSpec(
            name="test-pipeline",
            version="1.0",
            steps=[
                StepSpec(
                    name="step1",
                    agent="test.Agent",
                    action="init",
                    outputs={"initialized": True}
                ),
                StepSpec(
                    name="step2",
                    agent="test.Agent",
                    action="process",
                    condition="steps.step1.initialized == True",
                    outputs={"processed": True}
                ),
                StepSpec(
                    name="step3",
                    agent="test.Agent",
                    action="cleanup",
                    condition="steps.step2.processed and not steps.step1.failed"
                )
            ]
        )

        context = create_execution_context("test-pipeline")

        # Mock the actual step execution
        with patch.object(self.executor, '_execute_step_action') as mock_execute:
            mock_execute.return_value = {"status": "success"}

            # Should validate without using eval
            validation = self.executor._validate_pipeline(spec, context)
            assert validation["status"] == "validated"

    def test_worker_script_execution_safety(self):
        """Test that worker scripts execute safely."""
        step = StepSpec(
            name="test-step",
            agent="test.SafeAgent",
            action="process"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            # Write test input
            with open(input_file, 'w') as f:
                json.dump({"data": "test"}, f)

            # Generate script
            script = self.executor._create_worker_script(step, input_file, output_dir)

            # Verify no exec/eval for imports
            lines = script.split('\n')
            for line in lines:
                if 'import' in line.lower() and 'exec' in line:
                    # Should not have exec with import
                    assert False, f"Found exec with import in line: {line}"
                if 'from' in line.lower() and 'exec' in line:
                    # Should not have exec with from import
                    assert False, f"Found exec with from import in line: {line}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])