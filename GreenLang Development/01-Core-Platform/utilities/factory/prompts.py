# -*- coding: utf-8 -*-
"""
Agent Factory - LLM Prompt Templates

This module provides LLM prompt templates for generating GreenLang agents.
All prompts follow the tool-first architecture pattern with deterministic execution.

Key Design Principles:
- Tool-first numerics (zero hallucinated numbers)
- Deterministic execution (temperature=0, seed=42)
- Pattern extraction from 5 reference agents
- Multi-step generation pipeline
- Self-refinement prompts for iterative improvement

Author: GreenLang Framework Team
Date: October 2025
"""

from typing import Dict, List, Any, Optional


class AgentFactoryPrompts:
    """
    LLM prompt templates for agent code generation.

    This class provides structured prompts for each stage of the
    agent generation pipeline:
    1. Tool generation (deterministic calculations)
    2. Agent implementation (AI orchestration)
    3. Test generation (unit + integration)
    4. Documentation generation
    5. Self-refinement (error correction)
    """

    # Reference agents for pattern extraction
    REFERENCE_AGENTS = [
        "fuel_agent_ai.py",
        "carbon_agent_ai.py",
        "grid_factor_agent_ai.py",
        "recommendation_agent_ai.py",
        "report_agent_ai.py",
    ]

    @staticmethod
    def system_prompt() -> str:
        """System prompt for agent code generation."""
        return """You are an expert GreenLang agent developer specializing in AI-powered climate calculation agents.

Your expertise includes:
- Tool-first architecture (all calculations use tools, zero hallucinated numbers)
- Deterministic execution (temperature=0, seed=42)
- ChatSession integration for AI orchestration
- Type-safe Python code with comprehensive validation
- Test-driven development with 100% coverage

CRITICAL RULES:
1. ALL numeric calculations MUST use tools (never calculate in LLM)
2. ALWAYS use temperature=0 and seed=42 for determinism
3. ALWAYS delegate to deterministic base agent for calculations
4. ALWAYS provide natural language explanations via AI
5. ALWAYS include comprehensive error handling
6. ALWAYS follow the exact pattern from reference agents

Quality Standards:
- Production-ready code only (no TODOs, no placeholders)
- Complete type annotations
- Comprehensive docstrings (Google style)
- Full error handling and validation
- Performance-conscious (async, budget enforcement)
"""

    @staticmethod
    def tool_generation_prompt(spec: Dict[str, Any], reference_code: str) -> str:
        """
        Generate prompt for creating tool implementations.

        Args:
            spec: AgentSpec v2 specification
            reference_code: Reference agent code for pattern matching

        Returns:
            Formatted prompt string
        """
        agent_id = spec.get("id", "unknown")
        name = spec.get("name", "Unknown Agent")
        summary = spec.get("summary", "")
        compute = spec.get("compute", {})
        inputs = compute.get("inputs", {})
        outputs = compute.get("outputs", {})
        factors = compute.get("factors", {})

        prompt = f"""Generate tool implementations for the {name} agent.

AGENT SPECIFICATION:
- ID: {agent_id}
- Name: {name}
- Summary: {summary}

INPUTS:
"""
        for input_name, input_spec in inputs.items():
            dtype = input_spec.get("dtype", "unknown")
            unit = input_spec.get("unit", "unknown")
            desc = input_spec.get("description", "")
            prompt += f"- {input_name}: {dtype} ({unit}) - {desc}\n"

        prompt += "\nOUTPUTS:\n"
        for output_name, output_spec in outputs.items():
            dtype = output_spec.get("dtype", "unknown")
            unit = output_spec.get("unit", "unknown")
            desc = output_spec.get("description", "")
            prompt += f"- {output_name}: {dtype} ({unit}) - {desc}\n"

        if factors:
            prompt += "\nEMISSION FACTORS:\n"
            for factor_name, factor_spec in factors.items():
                ref = factor_spec.get("ref", "unknown")
                desc = factor_spec.get("description", "")
                prompt += f"- {factor_name}: {ref} - {desc}\n"

        prompt += f"""
REFERENCE PATTERN:
Study this reference agent code to extract the tool pattern:

```python
{reference_code}
```

TASK:
Generate 3-5 tool implementations following this pattern:

1. Tool: calculate_[primary_output]
   - Delegates to base agent for exact calculation
   - Returns Dict with calculation results
   - Increments tool_call_count

2. Tool: lookup_[emission_factor]
   - Fetches emission factor from database
   - Returns factor with metadata
   - Caches results when possible

3. Tool: generate_recommendations (optional)
   - Creates actionable recommendations
   - Uses deterministic rules
   - Returns structured recommendation list

4. Tool: validate_inputs (optional)
   - Validates input parameters
   - Returns validation errors if any
   - Uses base agent validation logic

5. Tool: calculate_[secondary_metrics] (optional)
   - Calculates derived metrics
   - Uses exact formulas (no estimation)
   - Returns Dict with metric values

REQUIREMENTS:
- Each tool must be a method: _[tool_name]_impl(self, ...)
- Each tool must increment self._tool_call_count
- Each tool must use exact calculations (no approximations)
- Each tool must return Dict[str, Any] with results
- Each tool must handle errors with clear messages
- Each tool must include comprehensive docstring

Generate complete, production-ready tool implementations.
"""
        return prompt

    @staticmethod
    def agent_implementation_prompt(
        spec: Dict[str, Any],
        tools_code: str,
        reference_code: str
    ) -> str:
        """
        Generate prompt for creating main agent class.

        Args:
            spec: AgentSpec v2 specification
            tools_code: Generated tool implementations
            reference_code: Reference agent code for pattern matching

        Returns:
            Formatted prompt string
        """
        agent_id = spec.get("id", "unknown")
        name = spec.get("name", "Unknown Agent")
        summary = spec.get("summary", "")
        ai_config = spec.get("ai", {})
        system_prompt = ai_config.get("system_prompt", "")
        budget = ai_config.get("budget", {})

        prompt = f"""Generate complete AI-powered agent implementation for {name}.

AGENT SPECIFICATION:
- ID: {agent_id}
- Name: {name}
- Summary: {summary}
- System Prompt: {system_prompt}
- Budget: {budget}

TOOL IMPLEMENTATIONS:
```python
{tools_code}
```

REFERENCE PATTERN:
```python
{reference_code}
```

TASK:
Generate complete agent class following this structure:

1. CLASS DEFINITION:
   - Class name: {name.replace(' ', '').replace('-', '')}AI
   - Inherits from: Agent or BaseAgent
   - Includes comprehensive docstring

2. __init__ METHOD:
   - Initialize base agent for tool delegation
   - Configure budget, explanations, recommendations
   - Initialize LLM provider
   - Setup performance tracking
   - Call _setup_tools()

3. _setup_tools METHOD:
   - Define ToolDef for each tool
   - Include JSON schema for parameters
   - Set descriptions for AI understanding

4. validate METHOD:
   - Delegate to base agent validation
   - Return bool

5. run/execute METHOD:
   - Validate input
   - Run async calculation via _run_async/_execute_async
   - Track performance metrics
   - Return AgentResult with data and metadata

6. _run_async/_execute_async METHOD:
   - Create ChatSession
   - Build AI prompt via _build_prompt
   - Call session.chat with tools, budget, temperature=0, seed=42
   - Extract tool results via _extract_tool_results
   - Build output via _build_output
   - Return AgentResult

7. _build_prompt METHOD:
   - Format input data for AI
   - List tasks for AI to perform
   - Specify which tools to use
   - Include important constraints

8. _extract_tool_results METHOD:
   - Parse tool calls from response
   - Call corresponding _impl methods
   - Return Dict of results

9. _build_output METHOD:
   - Combine tool results into output format
   - Add AI explanation if enabled
   - Add recommendations if enabled
   - Return complete output Dict

10. get_performance_summary METHOD:
    - Return AI and tool metrics
    - Include base agent metrics

CRITICAL REQUIREMENTS:
- MUST use temperature=0, seed=42 for determinism
- MUST use tools for ALL numeric calculations
- MUST delegate to base agent for exact calculations
- MUST include natural language explanations
- MUST track AI calls, tool calls, and costs
- MUST handle BudgetExceeded exception
- MUST return structured AgentResult
- MUST include comprehensive error handling

Generate complete, production-ready agent implementation.
"""
        return prompt

    @staticmethod
    def test_generation_prompt(
        spec: Dict[str, Any],
        agent_code: str,
        reference_tests: str
    ) -> str:
        """
        Generate prompt for creating test suite.

        Args:
            spec: AgentSpec v2 specification
            agent_code: Generated agent code
            reference_tests: Reference test code for pattern matching

        Returns:
            Formatted prompt string
        """
        agent_id = spec.get("id", "unknown")
        name = spec.get("name", "Unknown Agent")
        compute = spec.get("compute", {})
        inputs = compute.get("inputs", {})

        prompt = f"""Generate comprehensive test suite for {name} agent.

AGENT CODE:
```python
{agent_code}
```

REFERENCE TEST PATTERN:
```python
{reference_tests}
```

TASK:
Generate complete test suite with these test classes:

1. TestToolImplementations:
   - test_tool_call_increments_counter
   - test_tool_returns_dict
   - test_tool_handles_invalid_input
   - test_tool_delegates_to_base_agent
   - test_each_tool_individually (one test per tool)

2. TestAgentValidation:
   - test_valid_input_passes
   - test_missing_required_field_fails
   - test_invalid_type_fails
   - test_invalid_unit_fails
   - test_out_of_bounds_fails

3. TestAgentExecution:
   - test_run_success_with_valid_input
   - test_run_returns_agent_result
   - test_run_includes_metadata
   - test_run_tracks_performance
   - test_run_handles_errors

4. TestAIOrchestration:
   - test_creates_chat_session
   - test_uses_deterministic_settings (temperature=0, seed=42)
   - test_calls_tools_correctly
   - test_generates_explanation
   - test_generates_recommendations (if applicable)
   - test_budget_enforcement
   - test_budget_exceeded_handling

5. TestDeterminism:
   - test_same_input_same_output
   - test_reproducible_with_seed
   - test_no_hallucinated_numbers
   - test_all_numbers_from_tools

6. TestIntegration:
   - test_end_to_end_calculation
   - test_with_multiple_scenarios
   - test_edge_cases
   - test_boundary_conditions

SAMPLE TEST INPUTS:
"""
        # Generate sample inputs from spec
        sample_input = "{\n"
        for input_name, input_spec in inputs.items():
            dtype = input_spec.get("dtype", "float64")
            unit = input_spec.get("unit", "1")
            if dtype in ["float32", "float64"]:
                sample_input += f'    "{input_name}": 1000.0,  # {unit}\n'
            elif dtype in ["int32", "int64"]:
                sample_input += f'    "{input_name}": 100,  # {unit}\n'
            elif dtype == "string":
                sample_input += f'    "{input_name}": "test_value",\n'
            elif dtype == "bool":
                sample_input += f'    "{input_name}": true,\n'
        sample_input += "}"

        prompt += sample_input

        prompt += """

REQUIREMENTS:
- Use pytest framework
- Include fixtures for common test data
- Mock external dependencies (LLM provider, etc.)
- Test both success and failure paths
- Include edge cases and boundary conditions
- Aim for 100% code coverage
- Include docstrings for each test
- Group related tests in classes
- Use descriptive test names
- Include assertions for all critical outputs

Generate complete, production-ready test suite.
"""
        return prompt

    @staticmethod
    def documentation_generation_prompt(
        spec: Dict[str, Any],
        agent_code: str,
        test_code: str
    ) -> str:
        """
        Generate prompt for creating documentation.

        Args:
            spec: AgentSpec v2 specification
            agent_code: Generated agent code
            test_code: Generated test code

        Returns:
            Formatted prompt string
        """
        agent_id = spec.get("id", "unknown")
        name = spec.get("name", "Unknown Agent")
        summary = spec.get("summary", "")

        prompt = f"""Generate comprehensive documentation for {name} agent.

AGENT SPECIFICATION:
- ID: {agent_id}
- Name: {name}
- Summary: {summary}

TASK:
Generate complete README.md with these sections:

1. HEADER
   - Agent name
   - Short description
   - Status badges (tests, coverage, type-check)

2. OVERVIEW
   - What the agent does
   - Key features
   - Use cases

3. ARCHITECTURE
   - Tool-first design
   - AI orchestration via ChatSession
   - Deterministic execution guarantees
   - Component diagram (mermaid)

4. INSTALLATION
   - Requirements
   - Installation steps
   - Configuration

5. USAGE
   - Quick start example
   - Basic usage
   - Advanced usage
   - API reference

6. EXAMPLES
   - Example 1: Basic calculation
   - Example 2: With recommendations
   - Example 3: Edge cases

7. CONFIGURATION
   - Budget settings
   - AI configuration
   - Determinism settings

8. TESTING
   - How to run tests
   - Test coverage
   - Adding new tests

9. PERFORMANCE
   - Typical execution time
   - Token usage
   - Cost estimates
   - Performance tips

10. TROUBLESHOOTING
    - Common issues
    - Error messages
    - Debug tips

11. API REFERENCE
    - Class documentation
    - Method signatures
    - Parameter descriptions
    - Return values

12. PROVENANCE
    - Determinism guarantees
    - Audit trail
    - Reproducibility

13. CONTRIBUTING
    - How to contribute
    - Code style
    - PR process

14. LICENSE
    - License info
    - Attribution

REQUIREMENTS:
- Use clear, professional language
- Include code examples with syntax highlighting
- Include mermaid diagrams where helpful
- Include tables for parameter descriptions
- Include links to related documentation
- Format with proper markdown
- Include TOC for easy navigation

Generate complete, production-ready documentation.
"""
        return prompt

    @staticmethod
    def self_refinement_prompt(
        code: str,
        errors: List[str],
        attempt: int,
        max_attempts: int = 3
    ) -> str:
        """
        Generate prompt for self-refinement based on errors.

        Args:
            code: Generated code with errors
            errors: List of error messages
            attempt: Current attempt number
            max_attempts: Maximum refinement attempts

        Returns:
            Formatted prompt string
        """
        prompt = f"""SELF-REFINEMENT - Attempt {attempt}/{max_attempts}

The generated code has errors that need to be fixed.

CURRENT CODE:
```python
{code}
```

ERRORS DETECTED:
"""
        for i, error in enumerate(errors, 1):
            prompt += f"{i}. {error}\n"

        prompt += """
TASK:
Fix ALL errors in the code. For each error:
1. Identify the root cause
2. Implement the correct fix
3. Ensure no new errors are introduced
4. Preserve all existing functionality

COMMON ERROR PATTERNS TO CHECK:
- Missing imports
- Incorrect type annotations
- Invalid syntax
- Undefined variables
- Missing method implementations
- Incorrect indentation
- Missing docstrings
- Invalid tool definitions
- Incorrect ChatSession usage
- Missing error handling

REQUIREMENTS:
- Fix ALL errors completely
- Do not introduce new errors
- Maintain code quality
- Keep all docstrings
- Preserve type annotations
- Maintain test coverage
- Follow PEP 8 style guide

Return the complete, corrected code.
"""
        return prompt

    @staticmethod
    def demo_script_generation_prompt(
        spec: Dict[str, Any],
        agent_code: str
    ) -> str:
        """
        Generate prompt for creating demo script.

        Args:
            spec: AgentSpec v2 specification
            agent_code: Generated agent code

        Returns:
            Formatted prompt string
        """
        agent_id = spec.get("id", "unknown")
        name = spec.get("name", "Unknown Agent")

        prompt = f"""Generate interactive demo script for {name} agent.

TASK:
Create demo.py script that showcases the agent's capabilities.

STRUCTURE:
1. Imports
2. Configuration
3. Demo scenarios (3-5 examples)
4. Performance summary
5. Main function

DEMO SCENARIOS:
- Scenario 1: Basic usage (simple input)
- Scenario 2: Advanced usage (complex input)
- Scenario 3: Edge case (boundary conditions)
- Scenario 4: Error handling (invalid input)
- Scenario 5: Performance test (multiple runs)

Each scenario should:
- Print scenario name
- Show input data
- Run agent
- Print results
- Print AI explanation
- Print performance metrics

REQUIREMENTS:
- Use rich library for beautiful output
- Include color-coded output
- Show progress bars for long operations
- Print tables for structured data
- Include ASCII art header
- Make it interactive (optional user input)
- Include timing information
- Show cost information

Generate complete, production-ready demo script.
"""
        return prompt

    @staticmethod
    def code_review_prompt(
        code: str,
        spec: Dict[str, Any]
    ) -> str:
        """
        Generate prompt for code review and quality assessment.

        Args:
            code: Generated code to review
            spec: AgentSpec v2 specification

        Returns:
            Formatted prompt string
        """
        prompt = f"""Perform comprehensive code review of the generated agent.

CODE TO REVIEW:
```python
{code}
```

REVIEW CHECKLIST:

1. ARCHITECTURE
   - Follows tool-first pattern?
   - Uses ChatSession correctly?
   - Delegates to base agent for calculations?
   - Deterministic (temperature=0, seed=42)?

2. CODE QUALITY
   - Type annotations complete?
   - Docstrings comprehensive?
   - Error handling robust?
   - Follows PEP 8?

3. DETERMINISM
   - All numbers from tools?
   - No LLM math?
   - Reproducible with seed?
   - Provenance tracking?

4. PERFORMANCE
   - Async properly used?
   - Budget enforced?
   - Metrics tracked?
   - Efficient tool calls?

5. TESTING
   - 100% coverage achievable?
   - Edge cases handled?
   - Error paths tested?
   - Integration tests present?

6. DOCUMENTATION
   - Complete docstrings?
   - Usage examples?
   - API reference?
   - Clear explanations?

7. SECURITY
   - Input validation?
   - Safe tool execution?
   - No code injection risks?
   - Budget limits enforced?

For each issue found, provide:
- Severity (critical/major/minor)
- Location (file:line)
- Description
- Suggested fix

Return structured review report.
"""
        return prompt


# Prompt templates for specific tool types
TOOL_TEMPLATES = {
    "calculation": """
def _calculate_{name}_impl(
    self,
    {params}
) -> Dict[str, Any]:
    '''Tool implementation: {description}

    Args:
        {param_docs}

    Returns:
        Dict with calculation results
    '''
    self._tool_call_count += 1

    # Delegate to base agent for exact calculation
    result = self.base_agent.run({{
        {input_mapping}
    }})

    if not result["success"]:
        raise ValueError(f"Calculation failed: {{result['error']['message']}}")

    return {{
        {output_mapping}
    }}
""",

    "lookup": """
def _lookup_{name}_impl(
    self,
    {params}
) -> Dict[str, Any]:
    '''Tool implementation: {description}

    Args:
        {param_docs}

    Returns:
        Dict with lookup results
    '''
    self._tool_call_count += 1

    # Lookup from database/cache
    result = self.base_agent._get_cached_{name}({args})

    if result is None:
        raise ValueError(f"{name} not found")

    return {{
        "{name}": result,
        "source": "GreenLang Database",
        "cached": True,
    }}
""",

    "recommendation": """
def _generate_recommendations_impl(
    self,
    {params}
) -> Dict[str, Any]:
    '''Tool implementation: Generate recommendations

    Args:
        {param_docs}

    Returns:
        Dict with recommendations list
    '''
    self._tool_call_count += 1

    # Use base agent's recommendation logic
    recommendations = self.base_agent._generate_recommendations({args})

    return {{
        "recommendations": recommendations,
        "count": len(recommendations),
    }}
""",
}
