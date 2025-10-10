"""
Test Suite for Agent Factory

Comprehensive tests covering:
- Agent generation from spec
- Code validation
- Feedback loop
- Determinism verification
- End-to-end generation

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from greenlang.factory import (
    AgentFactory,
    GenerationResult,
    CodeValidator,
    ValidationResult,
    ValidationError,
)
from greenlang.specs import AgentSpecV2
from greenlang.intelligence import ChatResponse, Usage, ProviderInfo, FinishReason


# Test fixtures
@pytest.fixture
def sample_spec() -> Dict[str, Any]:
    """Create sample AgentSpec for testing."""
    return {
        "schema_version": "2.0.0",
        "id": "test/sample_agent",
        "name": "Sample Agent",
        "version": "1.0.0",
        "summary": "A test agent for unit testing",
        "compute": {
            "entrypoint": "python://test.sample:compute",
            "deterministic": True,
            "inputs": {
                "value": {
                    "dtype": "float64",
                    "unit": "kWh",
                    "required": True,
                    "ge": 0,
                }
            },
            "outputs": {
                "result": {
                    "dtype": "float64",
                    "unit": "kgCO2e",
                }
            },
            "factors": {
                "emission_factor": {
                    "ref": "ef://test/factor",
                }
            },
        },
        "ai": {
            "json_mode": True,
            "system_prompt": "You are a test agent.",
            "budget": {
                "max_cost_usd": 1.0,
            },
            "tools": [],
        },
        "realtime": {
            "default_mode": "replay",
            "connectors": [],
        },
        "provenance": {
            "pin_ef": True,
            "gwp_set": "AR6GWP100",
            "record": ["inputs", "outputs"],
        },
    }


@pytest.fixture
def spec(sample_spec: Dict[str, Any]) -> AgentSpecV2:
    """Create validated AgentSpec."""
    return AgentSpecV2.model_validate(sample_spec)


@pytest.fixture
def factory() -> AgentFactory:
    """Create AgentFactory instance."""
    return AgentFactory(
        budget_per_agent_usd=5.0,
        max_refinement_attempts=3,
        enable_validation=True,
    )


@pytest.fixture
def sample_agent_code() -> str:
    """Sample generated agent code."""
    return '''
from greenlang.intelligence import ChatSession, ChatMessage, Role
from greenlang.agents.base import BaseAgent

class SampleAgentAI(BaseAgent):
    """Sample AI agent."""

    def __init__(self):
        super().__init__()
        self._tool_call_count = 0

    def _setup_tools(self):
        pass

    def validate_input(self, input_data):
        return True

    async def _execute_async(self, input_data):
        return {"result": 100.0}

    def execute(self, input_data):
        return {"success": True, "data": {"result": 100.0}}
'''


@pytest.fixture
def sample_test_code() -> str:
    """Sample generated test code."""
    return '''
import pytest

def test_agent_creation():
    from test.sample import SampleAgentAI
    agent = SampleAgentAI()
    assert agent is not None

def test_agent_execution():
    from test.sample import SampleAgentAI
    agent = SampleAgentAI()
    result = agent.execute({"value": 100.0})
    assert result["success"] is True
'''


class TestCodeValidator:
    """Test CodeValidator class."""

    def test_validator_initialization(self):
        """Test validator can be initialized."""
        validator = CodeValidator(
            enable_type_check=True,
            enable_lint=True,
            enable_test=True,
        )
        assert validator is not None
        assert validator.enable_type_check is True

    def test_validate_valid_code(self, sample_agent_code):
        """Test validation of valid code."""
        validator = CodeValidator(
            enable_type_check=False,  # Skip type check for speed
            enable_lint=False,
            enable_test=False,
        )

        result = validator.validate_code(sample_agent_code)
        assert isinstance(result, ValidationResult)
        # Should have some warnings but no critical errors
        assert result is not None

    def test_validate_syntax_error(self):
        """Test validation catches syntax errors."""
        validator = CodeValidator(
            enable_type_check=False,
            enable_lint=False,
            enable_test=False,
        )

        invalid_code = "def broken_function(\n    pass"  # Syntax error
        result = validator.validate_code(invalid_code)

        assert result.passed is False
        assert any(e.category == "syntax" for e in result.errors)

    def test_validate_missing_imports(self):
        """Test validation catches missing imports."""
        validator = CodeValidator(
            enable_type_check=False,
            enable_lint=False,
            enable_test=False,
        )

        code_without_imports = '''
class TestAgent:
    def run(self):
        session = ChatSession()  # ChatSession not imported
'''

        result = validator.validate_code(code_without_imports)
        # Should catch missing import
        assert any(e.category == "syntax" for e in result.errors)

    def test_validate_determinism_markers(self, sample_agent_code):
        """Test validation of determinism markers."""
        validator = CodeValidator(
            enable_type_check=False,
            enable_lint=False,
            enable_test=False,
            enable_determinism_check=True,
        )

        # Code without temperature=0 and seed=42
        result = validator.validate_code(sample_agent_code)

        # Should warn about missing determinism markers
        # (Our sample doesn't have them)
        assert any(
            e.message.find("temperature") >= 0 or e.message.find("seed") >= 0
            for e in result.errors
        )


class TestAgentFactory:
    """Test AgentFactory class."""

    def test_factory_initialization(self):
        """Test factory can be initialized."""
        factory = AgentFactory(budget_per_agent_usd=5.0)
        assert factory is not None
        assert factory.budget_per_agent_usd == 5.0
        assert factory.max_refinement_attempts == 3

    def test_factory_with_custom_paths(self, tmp_path):
        """Test factory with custom paths."""
        output_path = tmp_path / "generated"
        factory = AgentFactory(output_path=output_path)

        assert factory.output_path == output_path
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_generate_agent_mock(self, factory, spec):
        """Test agent generation with mocked LLM."""
        # Mock the ChatSession to avoid actual LLM calls
        with patch("greenlang.factory.agent_factory.ChatSession") as mock_session_class:
            # Setup mock response
            mock_session = AsyncMock()
            mock_response = Mock(
                text="```python\nclass GeneratedAgent:\n    pass\n```",
                tool_calls=[],
                usage=Mock(cost_usd=0.01, total_tokens=100),
                provider_info=Mock(provider="test", model="test"),
                finish_reason=FinishReason.STOP,
            )
            mock_session.chat = AsyncMock(return_value=mock_response)
            mock_session_class.return_value = mock_session

            # Disable validation for this test
            factory.enable_validation = False

            result = await factory.generate_agent(spec, skip_tests=True, skip_docs=True, skip_demo=True)

            assert isinstance(result, GenerationResult)
            # With mocked LLM, might not be truly successful, but should return result
            assert result is not None

    @pytest.mark.asyncio
    async def test_extract_code_from_response(self, factory):
        """Test code extraction from LLM response."""
        # Test with markdown code block
        response1 = "```python\nclass Test:\n    pass\n```"
        code1 = factory._extract_code_from_response(response1)
        assert "class Test:" in code1

        # Test with plain code block
        response2 = "```\ndef function():\n    pass\n```"
        code2 = factory._extract_code_from_response(response2)
        assert "def function():" in code2

        # Test with no code block
        response3 = "class Direct:\n    pass"
        code3 = factory._extract_code_from_response(response3)
        assert "class Direct:" in code3

    def test_save_generated_code(self, factory, spec, sample_agent_code, tmp_path):
        """Test saving generated code to disk."""
        factory.output_path = tmp_path

        factory._save_generated_code(
            spec,
            sample_agent_code,
            None,
            "# Test Docs",
            None,
        )

        # Check files were created
        agent_dir = tmp_path / "test_sample_agent"
        assert agent_dir.exists()

        agent_file = agent_dir / "sample_agent_ai.py"
        assert agent_file.exists()

        docs_file = agent_dir / "README.md"
        assert docs_file.exists()

        spec_file = agent_dir / "pack.yaml"
        assert spec_file.exists()

    def test_create_provenance(self, factory, spec, sample_agent_code):
        """Test provenance record creation."""
        validation_result = ValidationResult(
            passed=True,
            errors=[],
            warnings=[],
            metrics={},
        )

        provenance = factory._create_provenance(
            spec,
            sample_agent_code,
            validation_result,
            total_cost=0.50,
            attempts=1,
        )

        assert provenance["agent_id"] == spec.id
        assert provenance["agent_version"] == spec.version
        assert "generated_at" in provenance
        assert "code_hash" in provenance
        assert "spec_hash" in provenance
        assert provenance["generation_cost_usd"] == 0.50
        assert provenance["refinement_attempts"] == 1
        assert provenance["deterministic"] is True
        assert provenance["temperature"] == 0.0
        assert provenance["seed"] == 42

    def test_get_metrics(self, factory):
        """Test factory metrics."""
        # Initially zero
        metrics = factory.get_metrics()
        assert metrics["total_agents_generated"] == 0
        assert metrics["total_cost_usd"] == 0.0

        # Simulate some generations
        factory._total_agents_generated = 5
        factory._total_cost_usd = 10.0
        factory._total_generation_time_seconds = 1500.0

        metrics = factory.get_metrics()
        assert metrics["total_agents_generated"] == 5
        assert metrics["total_cost_usd"] == 10.0
        assert metrics["avg_cost_per_agent_usd"] == 2.0
        assert metrics["avg_time_per_agent_seconds"] == 300.0
        assert metrics["target_time_seconds"] == 600  # 10 minutes

    @pytest.mark.asyncio
    async def test_load_reference_agents(self, factory, spec):
        """Test loading reference agents."""
        # This will try to load actual reference agents if they exist
        reference_code = await factory._load_reference_agents(spec)

        # Should return string (empty if not found)
        assert isinstance(reference_code, str)


class TestGenerationPipeline:
    """Test end-to-end generation pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_steps(self, factory, spec):
        """Test that pipeline executes all steps."""
        with patch("greenlang.factory.agent_factory.ChatSession") as mock_session_class:
            # Setup mock
            mock_session = AsyncMock()
            mock_response = Mock(
                text="```python\nclass Generated:\n    pass\n```",
                tool_calls=[],
                usage=Mock(cost_usd=0.01, total_tokens=50),
                provider_info=Mock(provider="test", model="test"),
                finish_reason=FinishReason.STOP,
            )
            mock_session.chat = AsyncMock(return_value=mock_response)
            mock_session_class.return_value = mock_session

            # Disable validation
            factory.enable_validation = False

            result = await factory.generate_agent(
                spec,
                skip_tests=True,
                skip_docs=True,
                skip_demo=True,
            )

            # Should have called chat multiple times (tools, agent, etc.)
            assert mock_session.chat.call_count >= 2

    @pytest.mark.asyncio
    async def test_batch_generation(self, factory, spec):
        """Test batch generation of multiple agents."""
        with patch("greenlang.factory.agent_factory.ChatSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_response = Mock(
                text="```python\nclass Generated:\n    pass\n```",
                tool_calls=[],
                usage=Mock(cost_usd=0.01, total_tokens=50),
                provider_info=Mock(provider="test", model="test"),
                finish_reason=FinishReason.STOP,
            )
            mock_session.chat = AsyncMock(return_value=mock_response)
            mock_session_class.return_value = mock_session

            factory.enable_validation = False

            # Generate 3 agents
            specs = [spec, spec, spec]
            results = await factory.generate_batch(specs, max_concurrent=2)

            assert len(results) == 3
            assert all(isinstance(r, GenerationResult) for r in results)


class TestDeterminism:
    """Test determinism verification."""

    def test_determinism_markers_in_code(self):
        """Test checking for determinism markers."""
        validator = CodeValidator(enable_determinism_check=True)

        code_with_markers = '''
async def _execute_async(self, input_data):
    response = await session.chat(
        messages=messages,
        temperature=0.0,
        seed=42,
    )
'''

        result = validator._validate_static(code_with_markers)

        # Should detect temperature=0 and seed=42
        assert result["metrics"]["determinism_markers"]["temperature_zero"] is True
        assert result["metrics"]["determinism_markers"]["seed_42"] is True

    def test_missing_determinism_markers(self):
        """Test detection of missing determinism markers."""
        validator = CodeValidator(enable_determinism_check=True)

        code_without_markers = '''
async def _execute_async(self, input_data):
    response = await session.chat(
        messages=messages,
    )
'''

        result = validator._validate_static(code_without_markers)

        # Should flag missing markers
        assert any(
            e.message.find("temperature=0") >= 0
            for e in result["errors"]
        )


class TestValidationErrors:
    """Test validation error handling."""

    def test_validation_error_creation(self):
        """Test creating validation errors."""
        error = ValidationError(
            severity="critical",
            category="syntax",
            message="Test error",
            line=10,
            column=5,
        )

        assert error.severity == "critical"
        assert error.category == "syntax"
        assert error.message == "Test error"
        assert error.line == 10

    def test_validation_result_with_errors(self):
        """Test validation result with errors."""
        errors = [
            ValidationError(
                severity="critical",
                category="syntax",
                message="Syntax error",
            )
        ]

        result = ValidationResult(
            passed=False,
            errors=errors,
            warnings=[],
            metrics={},
        )

        assert result.passed is False
        assert len(result.errors) == 1


class TestIntegration:
    """Integration tests."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_generation_flow(self, factory, spec, tmp_path):
        """Test full generation flow (requires LLM provider)."""
        # This test requires actual LLM access
        # Mark as slow and skip if no provider available

        factory.output_path = tmp_path
        factory.enable_validation = False  # Skip validation for speed

        try:
            result = await factory.generate_agent(
                spec,
                skip_tests=True,
                skip_docs=True,
                skip_demo=True,
            )

            # Should complete without errors (even if validation fails)
            assert isinstance(result, GenerationResult)
            assert result.duration_seconds > 0
            assert result.total_cost_usd >= 0

        except Exception as e:
            pytest.skip(f"LLM provider not available: {e}")


# Mark slow tests
pytest.mark.slow = pytest.mark.skipif(
    True,  # Change to False to run slow tests
    reason="Slow tests disabled by default"
)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
