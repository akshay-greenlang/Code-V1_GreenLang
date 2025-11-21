# -*- coding: utf-8 -*-
"""
Agent Factory - Code Generation Templates

This module provides code generation templates for GreenLang agents following
the tool-first architecture pattern.

Key Features:
- Tool-first architecture templates
- Agent implementation scaffolding
- Test suite templates
- Documentation templates
- Demo script templates

Author: GreenLang Framework Team
Date: October 2025
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from greenlang.determinism import DeterministicClock


@dataclass
class AgentTemplate:
    """Template data for agent generation."""
    agent_id: str
    agent_name: str
    agent_class: str
    base_agent_class: str
    module_path: str
    summary: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    factors: Dict[str, Any]
    tools: List[Dict[str, Any]]
    system_prompt: str
    budget_usd: float
    enable_explanations: bool
    enable_recommendations: bool


class CodeTemplates:
    """Code generation templates for GreenLang agents."""

    @staticmethod
    def agent_module_header(template: AgentTemplate) -> str:
        """Generate module docstring header."""
        return f'''"""AI-powered {template.agent_name} with ChatSession Integration.

This module provides an AI-enhanced version of the {template.base_agent_class} that uses
ChatSession for orchestration while preserving all deterministic calculations
as tool implementations.

Key Differences from Original {template.base_agent_class}:
    1. AI Orchestration: Uses ChatSession for natural language interaction
    2. Tool-First Numerics: All calculations wrapped as tools (zero hallucinated numbers)
    3. Natural Explanations: AI generates human-readable explanations
    4. Deterministic Results: temperature=0, seed=42 for reproducibility
    5. Enhanced Provenance: Full audit trail of AI decisions and tool calls
    6. Backward Compatible: Same API as original {template.base_agent_class}

Architecture:
    {template.agent_class} (orchestration) -> ChatSession (AI) -> Tools (exact calculations)

Example:
    >>> agent = {template.agent_class}()
    >>> result = agent.run({{
    ...     # Input parameters here
    ... }})
    >>> print(result["data"]["explanation"])
    "AI-generated explanation here..."
    >>> print(result["data"]["primary_output"])
    # Exact calculation from tool

Author: GreenLang Agent Factory
Date: October 2025
Generated from AgentSpec: {template.agent_id}
"""
'''

    @staticmethod
    def imports_section(template: AgentTemplate) -> str:
        """Generate imports section."""
        return '''
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio
import logging

from greenlang.types import Agent, AgentResult, ErrorInfo
from greenlang.agents.base import BaseAgent, AgentConfig
from greenlang.intelligence import (
    ChatSession,
    ChatMessage,
    Role,
    Budget,
    BudgetExceeded,
    create_provider,
)
from greenlang.intelligence.schemas.tools import ToolDef

logger = logging.getLogger(__name__)
'''

    @staticmethod
    def agent_class_definition(template: AgentTemplate) -> str:
        """Generate agent class definition and docstring."""
        return f'''

class {template.agent_class}(BaseAgent):
    """AI-powered {template.agent_name} using ChatSession.

    This agent enhances the original {template.base_agent_class} with AI orchestration while
    maintaining exact deterministic calculations through tool implementations.

    Features:
    - AI orchestration via ChatSession for intelligent analysis
    - Tool-first numerics (all calculations use tools, zero hallucinated numbers)
    - Natural language explanations
    - Deterministic results (temperature=0, seed=42)
    - Full provenance tracking (AI decisions + tool calls)
    - All original {template.base_agent_class} features preserved
    - Backward compatible API

    Determinism Guarantees:
    - Same input always produces same output
    - All numeric values come from tools (no LLM math)
    - Reproducible AI responses via seed=42
    - Auditable decision trail

    Performance:
    - Async support for concurrent processing
    - Budget enforcement (max ${template.budget_usd} per calculation by default)
    - Performance metrics tracking

    Example:
        >>> agent = {template.agent_class}()
        >>> result = agent.execute({{
        ...     # Input parameters
        ... }})
        >>> print(result.data["explanation"])
        >>> print(result.success)
        True
    """
'''

    @staticmethod
    def init_method(template: AgentTemplate) -> str:
        """Generate __init__ method."""
        return f'''
    def __init__(
        self,
        config: AgentConfig = None,
        *,
        budget_usd: float = {template.budget_usd},
        enable_explanations: bool = {str(template.enable_explanations)},
        enable_recommendations: bool = {str(template.enable_recommendations)},
    ):
        """Initialize the AI-powered {template.agent_name}.

        Args:
            config: Agent configuration (optional)
            budget_usd: Maximum USD to spend per calculation (default: ${template.budget_usd})
            enable_explanations: Enable AI-generated explanations (default: {template.enable_explanations})
            enable_recommendations: Enable AI recommendations (default: {template.enable_recommendations})
        """
        if config is None:
            config = AgentConfig(
                name="{template.agent_class}",
                description="{template.summary}",
                version="0.1.0",
            )
        super().__init__(config)

        # Initialize base agent for tool implementations
        self.base_agent = {template.base_agent_class}()

        # Configuration
        self.budget_usd = budget_usd
        self.enable_explanations = enable_explanations
        self.enable_recommendations = enable_recommendations

        # Initialize LLM provider (auto-detects available provider)
        self.provider = create_provider()

        # Performance tracking
        self._ai_call_count = 0
        self._tool_call_count = 0
        self._total_cost_usd = 0.0

        # Setup tools for ChatSession
        self._setup_tools()
'''

    @staticmethod
    def setup_tools_method(tools: List[Dict[str, Any]]) -> str:
        """Generate _setup_tools method."""
        code = '''
    def _setup_tools(self) -> None:
        """Setup tool definitions for ChatSession."""

'''
        for i, tool in enumerate(tools, 1):
            tool_name = tool["name"]
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})

            code += f'''
        # Tool {i}: {tool_name}
        self.{tool_name}_tool = ToolDef(
            name="{tool_name}",
            description="{description}",
            parameters={parameters},
        )
'''
        return code

    @staticmethod
    def validate_input_method() -> str:
        """Generate validate_input method."""
        return '''
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data.

        Args:
            input_data: Input dictionary

        Returns:
            bool: True if valid
        """
        return self.base_agent.validate_input(input_data)
'''

    @staticmethod
    def execute_method() -> str:
        """Generate execute method."""
        return '''
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute agent with AI orchestration.

        This method uses ChatSession to orchestrate the workflow
        while ensuring all numeric calculations use deterministic tools.

        Process:
        1. Validate input
        2. Build AI prompt with requirements
        3. AI uses tools for exact calculations
        4. AI generates natural language explanation
        5. Return results with provenance

        Args:
            input_data: Input data dictionary

        Returns:
            AgentResult with data and metadata
        """
        start_time = DeterministicClock.now()

        # Validate input
        if not self.validate_input(input_data):
            return AgentResult(
                success=False,
                error="Invalid input: validation failed",
            )

        try:
            # Run async calculation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._execute_async(input_data))
            finally:
                loop.close()

            # Calculate duration
            duration = (DeterministicClock.now() - start_time).total_seconds()

            # Add performance metadata
            if result.success:
                result.metadata["calculation_time_ms"] = duration * 1000
                result.metadata["ai_calls"] = self._ai_call_count
                result.metadata["tool_calls"] = self._tool_call_count
                result.metadata["total_cost_usd"] = self._total_cost_usd

            return result

        except Exception as e:
            self.logger.error(f"Error in AI agent execution: {e}")
            return AgentResult(
                success=False,
                error=f"Failed to execute agent: {str(e)}",
            )
'''

    @staticmethod
    def execute_async_method() -> str:
        """Generate _execute_async method."""
        return '''
    async def _execute_async(self, input_data: Dict[str, Any]) -> AgentResult:
        """Async execution with ChatSession.

        Args:
            input_data: Input data

        Returns:
            AgentResult with data and AI insights
        """
        # Create ChatSession
        session = ChatSession(self.provider)

        # Build AI prompt
        prompt = self._build_prompt(input_data)

        # Prepare messages
        messages = [
            ChatMessage(
                role=Role.system,
                content=(
                    "You are an AI assistant for GreenLang climate calculations. "
                    "You help perform accurate calculations using authoritative tools. "
                    "IMPORTANT: You must use the provided tools for ALL numeric calculations. "
                    "Never estimate or guess numbers. Always explain your calculations clearly."
                ),
            ),
            ChatMessage(role=Role.user, content=prompt),
        ]

        # Create budget
        budget = Budget(max_usd=self.budget_usd)

        try:
            # Call AI with tools
            self._ai_call_count += 1

            response = await session.chat(
                messages=messages,
                tools=self._get_tool_list(),
                budget=budget,
                temperature=0.0,  # Deterministic
                seed=42,          # Reproducible
                tool_choice="auto",
            )

            # Track cost
            self._total_cost_usd += response.usage.cost_usd

            # Extract tool results
            tool_results = self._extract_tool_results(response)

            # Build output from tool results
            output = self._build_output(
                input_data,
                tool_results,
                response.text if self.enable_explanations else None,
            )

            return AgentResult(
                success=True,
                data=output,
                metadata={
                    "agent": self.config.name,
                    "provider": response.provider_info.provider,
                    "model": response.provider_info.model,
                    "tokens": response.usage.total_tokens,
                    "cost_usd": response.usage.cost_usd,
                    "tool_calls": len(response.tool_calls),
                    "deterministic": True,
                },
            )

        except BudgetExceeded as e:
            self.logger.error(f"Budget exceeded: {e}")
            return AgentResult(
                success=False,
                error=f"AI budget exceeded: {str(e)}",
            )
'''

    @staticmethod
    def get_tool_list_method(tools: List[Dict[str, Any]]) -> str:
        """Generate _get_tool_list method."""
        code = '''
    def _get_tool_list(self) -> List[ToolDef]:
        """Get list of tool definitions for ChatSession.

        Returns:
            List of ToolDef objects
        """
        return [
'''
        for tool in tools:
            tool_name = tool["name"]
            code += f'            self.{tool_name}_tool,\n'

        code += '''        ]
'''
        return code

    @staticmethod
    def build_prompt_method(template: AgentTemplate) -> str:
        """Generate _build_prompt method."""
        return f'''
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build AI prompt for execution.

        Args:
            input_data: Input data

        Returns:
            str: Formatted prompt
        """
        prompt = f"""Execute {template.agent_name} calculation:

INPUT DATA:
{{input_data}}

TASKS:
1. Use the provided tools to perform exact calculations
2. Generate a clear explanation of the calculation process
"""

        if self.enable_recommendations:
            prompt += "3. Generate actionable recommendations based on results\\n"

        prompt += """
IMPORTANT:
- Use tools for ALL numeric calculations
- Do not estimate or guess any numbers
- Explain your calculations step-by-step
- Format numbers clearly (e.g., "1,234.56" not "1234.56")
"""

        return prompt
'''

    @staticmethod
    def extract_tool_results_method(tools: List[Dict[str, Any]]) -> str:
        """Generate _extract_tool_results method."""
        code = '''
    def _extract_tool_results(self, response) -> Dict[str, Any]:
        """Extract results from AI tool calls.

        Args:
            response: ChatResponse from session

        Returns:
            Dict with tool results
        """
        results = {}

        for tool_call in response.tool_calls:
            name = tool_call.get("name", "")
            args = tool_call.get("arguments", {})

'''
        for tool in tools:
            tool_name = tool["name"]
            impl_name = tool_name.replace("_tool", "_impl")
            code += f'''            if name == "{tool_name}":
                results["{tool_name}"] = self._{impl_name}(**args)
'''

        code += '''
        return results
'''
        return code

    @staticmethod
    def build_output_method(template: AgentTemplate) -> str:
        """Generate _build_output method."""
        return '''
    def _build_output(
        self,
        input_data: Dict[str, Any],
        tool_results: Dict[str, Any],
        ai_explanation: Optional[str],
    ) -> Dict[str, Any]:
        """Build output from tool results.

        Args:
            input_data: Original input
            tool_results: Results from tool calls
            ai_explanation: AI-generated explanation

        Returns:
            Dict with all output data
        """
        output = {}

        # Extract results from tools
        for key, value in tool_results.items():
            if isinstance(value, dict):
                output.update(value)
            else:
                output[key] = value

        # Add AI explanation if enabled
        if ai_explanation and self.enable_explanations:
            output["explanation"] = ai_explanation

        return output
'''

    @staticmethod
    def get_performance_summary_method() -> str:
        """Generate get_performance_summary method."""
        return '''
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary.

        Returns:
            Dict with AI and tool metrics
        """
        return {
            "agent": self.config.name,
            "ai_metrics": {
                "ai_call_count": self._ai_call_count,
                "tool_call_count": self._tool_call_count,
                "total_cost_usd": self._total_cost_usd,
                "avg_cost_per_call": (
                    self._total_cost_usd / max(self._ai_call_count, 1)
                ),
            },
            "base_agent_metrics": {
                "agent": self.base_agent.__class__.__name__,
            },
        }
'''

    @staticmethod
    def generate_complete_agent(template: AgentTemplate) -> str:
        """Generate complete agent module."""
        code = CodeTemplates.agent_module_header(template)
        code += CodeTemplates.imports_section(template)
        code += CodeTemplates.agent_class_definition(template)
        code += CodeTemplates.init_method(template)
        code += CodeTemplates.setup_tools_method(template.tools)

        # Add tool implementations (to be generated separately)
        code += "\n    # Tool implementations\n"

        code += CodeTemplates.validate_input_method()
        code += CodeTemplates.execute_method()
        code += CodeTemplates.execute_async_method()
        code += CodeTemplates.get_tool_list_method(template.tools)
        code += CodeTemplates.build_prompt_method(template)
        code += CodeTemplates.extract_tool_results_method(template.tools)
        code += CodeTemplates.build_output_method(template)
        code += CodeTemplates.get_performance_summary_method()

        return code


class TestTemplates:
    """Test suite templates for GreenLang agents."""

    @staticmethod
    def test_module_header(agent_name: str) -> str:
        """Generate test module header."""
        return f'''"""
Test Suite for {agent_name}

Comprehensive tests covering:
- Tool implementations
- Input validation
- Agent execution
- AI orchestration
- Determinism
- Integration tests

Author: GreenLang Agent Factory
Date: October 2025
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from greenlang.intelligence import ChatResponse, Usage, ProviderInfo, FinishReason


class TestToolImplementations:
    """Test tool implementations."""

    def test_tool_call_increments_counter(self, agent):
        """Test that tool calls increment counter."""
        initial_count = agent._tool_call_count
        # Call tool implementation
        assert agent._tool_call_count > initial_count

    def test_tool_returns_dict(self, agent):
        """Test that tools return dictionaries."""
        # Test each tool returns Dict[str, Any]
        pass


class TestAgentValidation:
    """Test input validation."""

    def test_valid_input_passes(self, agent, valid_input):
        """Test that valid input passes validation."""
        assert agent.validate_input(valid_input) is True

    def test_missing_required_field_fails(self, agent):
        """Test that missing required field fails validation."""
        invalid_input = {{}}  # Missing required fields
        assert agent.validate_input(invalid_input) is False


class TestAgentExecution:
    """Test agent execution."""

    def test_run_success_with_valid_input(self, agent, valid_input):
        """Test successful execution with valid input."""
        result = agent.execute(valid_input)
        assert result.success is True
        assert result.data is not None

    def test_run_returns_agent_result(self, agent, valid_input):
        """Test that run returns AgentResult."""
        result = agent.execute(valid_input)
        assert hasattr(result, "success")
        assert hasattr(result, "data")
        assert hasattr(result, "metadata")


class TestAIOrchestration:
    """Test AI orchestration."""

    @patch("greenlang.intelligence.ChatSession")
    async def test_creates_chat_session(self, mock_session, agent, valid_input):
        """Test that ChatSession is created."""
        # Mock ChatSession
        mock_session.return_value.chat = AsyncMock(return_value=Mock(
            text="Explanation",
            tool_calls=[],
            usage=Usage(total_tokens=100, cost_usd=0.01),
            provider_info=ProviderInfo(provider="test", model="test"),
            finish_reason=FinishReason.STOP,
        ))

        result = await agent._execute_async(valid_input)
        assert result.success is True

    def test_uses_deterministic_settings(self, agent):
        """Test that deterministic settings are used."""
        # Verify temperature=0, seed=42 in chat call
        pass


class TestDeterminism:
    """Test deterministic behavior."""

    def test_same_input_same_output(self, agent, valid_input):
        """Test that same input produces same output."""
        result1 = agent.execute(valid_input)
        result2 = agent.execute(valid_input)
        # Compare results (accounting for timestamps)
        assert result1.data == result2.data


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_calculation(self, agent, valid_input):
        """Test end-to-end calculation flow."""
        result = agent.execute(valid_input)
        assert result.success is True
        assert "explanation" in result.data or not agent.enable_explanations
'''

    @staticmethod
    def test_fixtures(template: AgentTemplate) -> str:
        """Generate test fixtures."""
        return f'''

@pytest.fixture
def agent():
    """Create agent instance for testing."""
    from {template.module_path} import {template.agent_class}
    return {template.agent_class}(
        budget_usd=1.00,
        enable_explanations=True,
        enable_recommendations=True,
    )


@pytest.fixture
def valid_input():
    """Create valid input data for testing."""
    return {{
        # Add sample input fields
    }}


@pytest.fixture
def invalid_input():
    """Create invalid input data for testing."""
    return {{
        # Add invalid input
    }}
'''


class DocumentationTemplates:
    """Documentation templates for GreenLang agents."""

    @staticmethod
    def readme_template(template: AgentTemplate) -> str:
        """Generate README.md template."""
        return f'''# {template.agent_name}

AI-powered {template.agent_name} with tool-first architecture and deterministic execution.

## Overview

{template.summary}

## Features

- **Tool-First Architecture**: All calculations use deterministic tools (zero hallucinated numbers)
- **AI Orchestration**: Natural language explanations via ChatSession
- **Deterministic**: Same input always produces same output (temperature=0, seed=42)
- **Provenance**: Full audit trail of AI decisions and tool calls
- **Budget Enforcement**: Cost controls and token limits
- **Backward Compatible**: Same API as original agent

## Architecture

```mermaid
graph LR
    A[User Input] --> B[{template.agent_class}]
    B --> C[ChatSession]
    C --> D[Tool 1]
    C --> E[Tool 2]
    C --> F[Tool 3]
    D --> G[Base Agent]
    E --> G
    F --> G
    G --> H[Exact Calculation]
    H --> I[Results]
    C --> J[AI Explanation]
    I --> K[Output]
    J --> K
```

## Installation

```bash
pip install greenlang
```

## Quick Start

```python
from {template.module_path} import {template.agent_class}

# Create agent
agent = {template.agent_class}()

# Run calculation
result = agent.execute({{
    # Input parameters
}})

# Print results
print(result.data["explanation"])
print(result.data)
```

## Configuration

```python
agent = {template.agent_class}(
    budget_usd={template.budget_usd},  # Max cost per calculation
    enable_explanations={template.enable_explanations},
    enable_recommendations={template.enable_recommendations},
)
```

## API Reference

### Methods

#### `execute(input_data: Dict[str, Any]) -> AgentResult`

Execute agent calculation.

**Parameters:**
- `input_data`: Input dictionary with required parameters

**Returns:**
- `AgentResult` with success, data, metadata

#### `validate_input(input_data: Dict[str, Any]) -> bool`

Validate input data.

#### `get_performance_summary() -> Dict[str, Any]`

Get performance metrics.

## Testing

```bash
pytest tests/factory/test_{template.agent_id.replace("/", "_")}.py -v
```

## Performance

- **Execution Time**: ~2-5 seconds
- **Token Usage**: ~500-2000 tokens
- **Cost**: ~$0.01-0.05 per calculation

## License

Apache 2.0
'''


class DemoScriptTemplates:
    """Demo script templates for GreenLang agents."""

    @staticmethod
    def demo_script_template(template: AgentTemplate) -> str:
        """Generate demo.py template."""
        return f'''"""
Demo Script for {template.agent_name}

Interactive demonstration of agent capabilities.

Author: GreenLang Agent Factory
Date: October 2025
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from {template.module_path} import {template.agent_class}
import time


console = Console()


def print_header():
    """Print demo header."""
    console.print(Panel.fit(
        "[bold cyan]{template.agent_name} Demo[/bold cyan]\\n"
        "[dim]AI-powered climate calculations with tool-first architecture[/dim]",
        border_style="cyan"
    ))


def scenario_1_basic():
    """Scenario 1: Basic calculation."""
    console.print("\\n[bold]Scenario 1: Basic Calculation[/bold]")

    agent = {template.agent_class}()

    input_data = {{
        # Add sample input
    }}

    console.print("Input:", input_data)

    start_time = time.time()
    result = agent.execute(input_data)
    duration = time.time() - start_time

    console.print(f"Duration: {{duration:.2f}}s")
    console.print("Success:", result.success)
    console.print("Explanation:", result.data.get("explanation", "N/A"))


def main():
    """Run demo scenarios."""
    print_header()

    scenario_1_basic()

    console.print("\\n[bold green]Demo complete![/bold green]")


if __name__ == "__main__":
    main()
'''
