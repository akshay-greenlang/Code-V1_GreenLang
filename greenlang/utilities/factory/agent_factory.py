# -*- coding: utf-8 -*-
"""
Agent Factory - Main LLM-Powered Code Generation System

This module provides the core AgentFactory class for generating GreenLang agents
from AgentSpec specifications using LLM-powered code generation.
from greenlang.utilities.determinism import DeterministicClock

Key Features:
- Multi-step generation pipeline (tools → agent → tests → docs)
- LLM-powered code generation via ChatSession
- Feedback loop for iterative refinement
- Comprehensive validation (syntax, type, lint, test)
- Determinism verification
- Provenance tracking for generated code

Performance Target: 10 minutes per agent (vs 2 weeks manual)

Author: GreenLang Framework Team
Date: October 2025
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json
import yaml

from greenlang.intelligence import (
    ChatSession,
    ChatMessage,
    Role,
    Budget,
    BudgetExceeded,
    create_provider,
)
from greenlang.specs import AgentSpecV2, agent_from_yaml, agent_from_json

from .prompts import AgentFactoryPrompts
from .templates import AgentTemplate, CodeTemplates, TestTemplates, DocumentationTemplates, DemoScriptTemplates
from .validators import CodeValidator, ValidationResult, DeterminismVerifier

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of agent generation."""
    success: bool
    agent_code: Optional[str] = None
    test_code: Optional[str] = None
    docs: Optional[str] = None
    demo_script: Optional[str] = None
    validation_result: Optional[ValidationResult] = None
    provenance: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    total_cost_usd: float = 0.0
    attempts: int = 0


class AgentFactory:
    """
    LLM-powered code generation system for GreenLang agents.

    This factory generates complete agent packages from AgentSpec specifications:
    1. Tool implementations (deterministic calculations)
    2. Agent class (AI orchestration)
    3. Test suite (unit + integration)
    4. Documentation (README, API reference)
    5. Demo script (interactive examples)

    The generation process uses:
    - ChatSession for LLM-powered generation
    - Pattern extraction from 5 reference agents
    - Multi-step pipeline with validation
    - Feedback loop for iterative refinement
    - Comprehensive quality gates

    Performance:
    - Target: 10 minutes per agent
    - Validation: syntax, type, lint, test
    - Determinism: temperature=0, seed=42
    - Max refinement attempts: 3

    Example:
        >>> factory = AgentFactory()
        >>> spec = from_yaml("specs/my_agent.yaml")
        >>> result = await factory.generate_agent(spec)
        >>> if result.success:
        ...     print(f"Agent generated in {result.duration_seconds}s")
        ...     print(f"Cost: ${result.total_cost_usd}")
    """

    def __init__(
        self,
        *,
        budget_per_agent_usd: float = 5.00,
        max_refinement_attempts: int = 3,
        enable_validation: bool = True,
        enable_determinism_check: bool = True,
        reference_agents_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ):
        """
        Initialize Agent Factory.

        Args:
            budget_per_agent_usd: Max USD to spend per agent (default: $5.00)
            max_refinement_attempts: Max refinement attempts (default: 3)
            enable_validation: Enable code validation (default: True)
            enable_determinism_check: Enable determinism verification (default: True)
            reference_agents_path: Path to reference agents (default: greenlang/agents/)
            output_path: Path for generated code (default: ./generated/)
        """
        self.budget_per_agent_usd = budget_per_agent_usd
        self.max_refinement_attempts = max_refinement_attempts
        self.enable_validation = enable_validation
        self.enable_determinism_check = enable_determinism_check

        # Paths
        if reference_agents_path is None:
            self.reference_agents_path = Path(__file__).parent.parent / "agents"
        else:
            self.reference_agents_path = Path(reference_agents_path)

        if output_path is None:
            self.output_path = Path.cwd() / "generated"
        else:
            self.output_path = Path(output_path)

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize LLM provider
        self.provider = create_provider()

        # Initialize validator
        self.validator = CodeValidator(
            enable_type_check=True,
            enable_lint=True,
            enable_test=True,
            enable_determinism_check=enable_determinism_check,
        )

        # Performance tracking
        self._total_agents_generated = 0
        self._total_cost_usd = 0.0
        self._total_generation_time_seconds = 0.0

        logger.info(f"Agent Factory initialized (budget: ${budget_per_agent_usd}/agent)")

    async def generate_agent(
        self,
        spec: AgentSpecV2,
        *,
        skip_tests: bool = False,
        skip_docs: bool = False,
        skip_demo: bool = False,
    ) -> GenerationResult:
        """
        Generate complete agent package from AgentSpec.

        This is the main entry point for agent generation. It orchestrates
        the multi-step generation pipeline:

        1. Load reference agents for pattern extraction
        2. Generate tool implementations
        3. Generate agent class
        4. Generate test suite
        5. Generate documentation
        6. Generate demo script
        7. Validate all code
        8. Refine if errors found (up to max_attempts)
        9. Save generated code
        10. Create provenance record

        Args:
            spec: AgentSpec v2 specification
            skip_tests: Skip test generation (default: False)
            skip_docs: Skip documentation generation (default: False)
            skip_demo: Skip demo script generation (default: False)

        Returns:
            GenerationResult with generated code and validation results
        """
        start_time = DeterministicClock.now()
        total_cost = 0.0
        attempts = 0

        logger.info(f"Starting agent generation: {spec.id}")

        try:
            # Create budget for this agent
            budget = Budget(max_usd=self.budget_per_agent_usd)

            # Load reference agents
            logger.info("Loading reference agents for pattern extraction...")
            reference_code = await self._load_reference_agents(spec)

            # Step 1: Generate tools
            logger.info("Step 1/5: Generating tool implementations...")
            tools_code, cost1 = await self._generate_tools(spec, reference_code, budget)
            total_cost += cost1

            # Step 2: Generate agent
            logger.info("Step 2/5: Generating agent implementation...")
            agent_code, cost2 = await self._generate_agent_class(
                spec, tools_code, reference_code, budget
            )
            total_cost += cost2

            # Combine tools and agent
            complete_agent_code = tools_code + "\n\n" + agent_code

            # Step 3: Generate tests
            test_code = None
            if not skip_tests:
                logger.info("Step 3/5: Generating test suite...")
                reference_tests = await self._load_reference_tests(spec)
                test_code, cost3 = await self._generate_tests(
                    spec, complete_agent_code, reference_tests, budget
                )
                total_cost += cost3

            # Step 4: Validate and refine
            logger.info("Step 4/5: Validating generated code...")
            validation_result = None
            if self.enable_validation:
                for attempt in range(1, self.max_refinement_attempts + 1):
                    attempts = attempt
                    logger.info(f"Validation attempt {attempt}/{self.max_refinement_attempts}")

                    validation_result = self.validator.validate_code(
                        complete_agent_code,
                        test_code,
                        spec.model_dump(),
                    )

                    if validation_result.passed:
                        logger.info("Validation passed!")
                        break

                    if attempt < self.max_refinement_attempts:
                        logger.warning(f"Validation failed, refining code (attempt {attempt})...")
                        complete_agent_code, refinement_cost = await self._refine_code(
                            complete_agent_code,
                            validation_result.errors,
                            attempt,
                            budget,
                        )
                        total_cost += refinement_cost

                        # Re-validate tests if agent was refined
                        if test_code and not skip_tests:
                            test_code, test_cost = await self._generate_tests(
                                spec, complete_agent_code, reference_tests, budget
                            )
                            total_cost += test_cost
                    else:
                        logger.error("Max refinement attempts reached, validation failed")

            # Step 5: Generate documentation and demo
            docs = None
            demo_script = None

            if not skip_docs:
                logger.info("Step 5/5: Generating documentation...")
                docs, cost4 = await self._generate_documentation(
                    spec, complete_agent_code, test_code or "", budget
                )
                total_cost += cost4

            if not skip_demo:
                logger.info("Generating demo script...")
                demo_script, cost5 = await self._generate_demo_script(
                    spec, complete_agent_code, budget
                )
                total_cost += cost5

            # Save generated code
            logger.info("Saving generated code...")
            self._save_generated_code(
                spec,
                complete_agent_code,
                test_code,
                docs,
                demo_script,
            )

            # Create provenance record
            provenance = self._create_provenance(
                spec,
                complete_agent_code,
                validation_result,
                total_cost,
                attempts,
            )

            # Calculate duration
            duration = (DeterministicClock.now() - start_time).total_seconds()

            # Update global metrics
            self._total_agents_generated += 1
            self._total_cost_usd += total_cost
            self._total_generation_time_seconds += duration

            logger.info(
                f"Agent generation complete: {spec.id} "
                f"(duration: {duration:.1f}s, cost: ${total_cost:.4f}, attempts: {attempts})"
            )

            return GenerationResult(
                success=True,
                agent_code=complete_agent_code,
                test_code=test_code,
                docs=docs,
                demo_script=demo_script,
                validation_result=validation_result,
                provenance=provenance,
                duration_seconds=duration,
                total_cost_usd=total_cost,
                attempts=attempts,
            )

        except BudgetExceeded as e:
            logger.error(f"Budget exceeded: {e}")
            return GenerationResult(
                success=False,
                error=f"Budget exceeded: {str(e)}",
                total_cost_usd=total_cost,
                attempts=attempts,
            )

        except Exception as e:
            logger.error(f"Agent generation failed: {e}", exc_info=True)
            return GenerationResult(
                success=False,
                error=f"Generation failed: {str(e)}",
                total_cost_usd=total_cost,
                attempts=attempts,
            )

    async def _load_reference_agents(self, spec: AgentSpecV2) -> str:
        """
        Load reference agent code for pattern extraction.

        Selects the most similar reference agent based on spec characteristics.

        Args:
            spec: AgentSpec to match against

        Returns:
            Reference agent source code
        """
        # For now, use fuel_agent_ai as default reference
        # In future, could use LLM to select best match
        reference_file = self.reference_agents_path / "fuel_agent_ai.py"

        if reference_file.exists():
            return reference_file.read_text()
        else:
            logger.warning(f"Reference agent not found: {reference_file}")
            return ""

    async def _load_reference_tests(self, spec: AgentSpecV2) -> str:
        """
        Load reference test code for pattern extraction.

        Args:
            spec: AgentSpec to match against

        Returns:
            Reference test source code
        """
        # Load reference tests (if they exist)
        test_file = self.reference_agents_path.parent / "tests" / "agents" / "test_fuel_agent_ai.py"

        if test_file.exists():
            return test_file.read_text()
        else:
            logger.warning(f"Reference tests not found: {test_file}")
            return ""

    async def _generate_tools(
        self,
        spec: AgentSpecV2,
        reference_code: str,
        budget: Budget,
    ) -> Tuple[str, float]:
        """
        Generate tool implementations using LLM.

        Args:
            spec: AgentSpec v2
            reference_code: Reference agent code
            budget: Budget for LLM calls

        Returns:
            Tuple of (tools_code, cost_usd)
        """
        session = ChatSession(self.provider)

        # Build prompt
        prompt = AgentFactoryPrompts.tool_generation_prompt(
            spec.model_dump(),
            reference_code,
        )

        # Call LLM
        response = await session.chat(
            messages=[
                ChatMessage(role=Role.system, content=AgentFactoryPrompts.system_prompt()),
                ChatMessage(role=Role.user, content=prompt),
            ],
            budget=budget,
            temperature=0.0,  # Deterministic
            seed=42,
        )

        # Extract code from response
        tools_code = self._extract_code_from_response(response.text)

        return tools_code, response.usage.cost_usd

    async def _generate_agent_class(
        self,
        spec: AgentSpecV2,
        tools_code: str,
        reference_code: str,
        budget: Budget,
    ) -> Tuple[str, float]:
        """
        Generate agent class implementation using LLM.

        Args:
            spec: AgentSpec v2
            tools_code: Generated tool implementations
            reference_code: Reference agent code
            budget: Budget for LLM calls

        Returns:
            Tuple of (agent_code, cost_usd)
        """
        session = ChatSession(self.provider)

        # Build prompt
        prompt = AgentFactoryPrompts.agent_implementation_prompt(
            spec.model_dump(),
            tools_code,
            reference_code,
        )

        # Call LLM
        response = await session.chat(
            messages=[
                ChatMessage(role=Role.system, content=AgentFactoryPrompts.system_prompt()),
                ChatMessage(role=Role.user, content=prompt),
            ],
            budget=budget,
            temperature=0.0,
            seed=42,
        )

        # Extract code from response
        agent_code = self._extract_code_from_response(response.text)

        return agent_code, response.usage.cost_usd

    async def _generate_tests(
        self,
        spec: AgentSpecV2,
        agent_code: str,
        reference_tests: str,
        budget: Budget,
    ) -> Tuple[str, float]:
        """
        Generate test suite using LLM.

        Args:
            spec: AgentSpec v2
            agent_code: Generated agent code
            reference_tests: Reference test code
            budget: Budget for LLM calls

        Returns:
            Tuple of (test_code, cost_usd)
        """
        session = ChatSession(self.provider)

        # Build prompt
        prompt = AgentFactoryPrompts.test_generation_prompt(
            spec.model_dump(),
            agent_code,
            reference_tests,
        )

        # Call LLM
        response = await session.chat(
            messages=[
                ChatMessage(role=Role.system, content=AgentFactoryPrompts.system_prompt()),
                ChatMessage(role=Role.user, content=prompt),
            ],
            budget=budget,
            temperature=0.0,
            seed=42,
        )

        # Extract code from response
        test_code = self._extract_code_from_response(response.text)

        return test_code, response.usage.cost_usd

    async def _generate_documentation(
        self,
        spec: AgentSpecV2,
        agent_code: str,
        test_code: str,
        budget: Budget,
    ) -> Tuple[str, float]:
        """
        Generate documentation using LLM.

        Args:
            spec: AgentSpec v2
            agent_code: Generated agent code
            test_code: Generated test code
            budget: Budget for LLM calls

        Returns:
            Tuple of (docs, cost_usd)
        """
        session = ChatSession(self.provider)

        # Build prompt
        prompt = AgentFactoryPrompts.documentation_generation_prompt(
            spec.model_dump(),
            agent_code,
            test_code,
        )

        # Call LLM
        response = await session.chat(
            messages=[
                ChatMessage(role=Role.system, content=AgentFactoryPrompts.system_prompt()),
                ChatMessage(role=Role.user, content=prompt),
            ],
            budget=budget,
            temperature=0.0,
            seed=42,
        )

        return response.text, response.usage.cost_usd

    async def _generate_demo_script(
        self,
        spec: AgentSpecV2,
        agent_code: str,
        budget: Budget,
    ) -> Tuple[str, float]:
        """
        Generate demo script using LLM.

        Args:
            spec: AgentSpec v2
            agent_code: Generated agent code
            budget: Budget for LLM calls

        Returns:
            Tuple of (demo_script, cost_usd)
        """
        session = ChatSession(self.provider)

        # Build prompt
        prompt = AgentFactoryPrompts.demo_script_generation_prompt(
            spec.model_dump(),
            agent_code,
        )

        # Call LLM
        response = await session.chat(
            messages=[
                ChatMessage(role=Role.system, content=AgentFactoryPrompts.system_prompt()),
                ChatMessage(role=Role.user, content=prompt),
            ],
            budget=budget,
            temperature=0.0,
            seed=42,
        )

        # Extract code from response
        demo_code = self._extract_code_from_response(response.text)

        return demo_code, response.usage.cost_usd

    async def _refine_code(
        self,
        code: str,
        errors: List[Any],
        attempt: int,
        budget: Budget,
    ) -> Tuple[str, float]:
        """
        Refine code based on validation errors.

        Args:
            code: Current code with errors
            errors: List of validation errors
            attempt: Current refinement attempt
            budget: Budget for LLM calls

        Returns:
            Tuple of (refined_code, cost_usd)
        """
        session = ChatSession(self.provider)

        # Build refinement prompt
        error_messages = [f"{e.category}: {e.message}" for e in errors]
        prompt = AgentFactoryPrompts.self_refinement_prompt(
            code,
            error_messages,
            attempt,
            self.max_refinement_attempts,
        )

        # Call LLM
        response = await session.chat(
            messages=[
                ChatMessage(role=Role.system, content=AgentFactoryPrompts.system_prompt()),
                ChatMessage(role=Role.user, content=prompt),
            ],
            budget=budget,
            temperature=0.0,
            seed=42,
        )

        # Extract refined code
        refined_code = self._extract_code_from_response(response.text)

        return refined_code, response.usage.cost_usd

    def _extract_code_from_response(self, response_text: str) -> str:
        """
        Extract code from LLM response (handles markdown code blocks).

        Args:
            response_text: LLM response text

        Returns:
            Extracted code
        """
        # Look for ```python ... ``` blocks
        if "```python" in response_text:
            parts = response_text.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                return code_part.strip()

        # Look for ``` ... ``` blocks
        if "```" in response_text:
            parts = response_text.split("```")
            if len(parts) > 1:
                return parts[1].strip()

        # Return as-is if no code blocks found
        return response_text.strip()

    def _save_generated_code(
        self,
        spec: AgentSpecV2,
        agent_code: str,
        test_code: Optional[str],
        docs: Optional[str],
        demo_script: Optional[str],
    ):
        """
        Save generated code to output directory.

        Args:
            spec: AgentSpec v2
            agent_code: Generated agent code
            test_code: Generated test code (optional)
            docs: Generated documentation (optional)
            demo_script: Generated demo script (optional)
        """
        # Create agent directory
        agent_dir = self.output_path / spec.id.replace("/", "_")
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Save agent code
        agent_file = agent_dir / f"{spec.id.split('/')[-1]}_ai.py"
        agent_file.write_text(agent_code)
        logger.info(f"Saved agent code: {agent_file}")

        # Save test code
        if test_code:
            test_file = agent_dir / f"test_{spec.id.split('/')[-1]}_ai.py"
            test_file.write_text(test_code)
            logger.info(f"Saved test code: {test_file}")

        # Save documentation
        if docs:
            docs_file = agent_dir / "README.md"
            docs_file.write_text(docs)
            logger.info(f"Saved documentation: {docs_file}")

        # Save demo script
        if demo_script:
            demo_file = agent_dir / "demo.py"
            demo_file.write_text(demo_script)
            logger.info(f"Saved demo script: {demo_file}")

        # Save spec
        spec_file = agent_dir / "pack.yaml"
        spec_file.write_text(yaml.dump(spec.model_dump(), sort_keys=False))
        logger.info(f"Saved spec: {spec_file}")

    def _create_provenance(
        self,
        spec: AgentSpecV2,
        agent_code: str,
        validation_result: Optional[ValidationResult],
        total_cost: float,
        attempts: int,
    ) -> Dict[str, Any]:
        """
        Create provenance record for generated agent.

        Args:
            spec: AgentSpec v2
            agent_code: Generated agent code
            validation_result: Validation results
            total_cost: Total generation cost
            attempts: Number of refinement attempts

        Returns:
            Provenance dictionary
        """
        from .validators import calculate_code_hash

        return {
            "agent_id": spec.id,
            "agent_version": spec.version,
            "generated_at": DeterministicClock.now().isoformat(),
            "generator": "AgentFactory",
            "generator_version": "0.1.0",
            "code_hash": calculate_code_hash(agent_code),
            "spec_hash": calculate_code_hash(json.dumps(spec.model_dump(), sort_keys=True)),
            "generation_cost_usd": total_cost,
            "refinement_attempts": attempts,
            "validation_passed": validation_result.passed if validation_result else None,
            "deterministic": True,
            "llm_provider": self.provider.__class__.__name__,
            "temperature": 0.0,
            "seed": 42,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get factory performance metrics.

        Returns:
            Dict with metrics
        """
        avg_time = (
            self._total_generation_time_seconds / max(self._total_agents_generated, 1)
        )
        avg_cost = (
            self._total_cost_usd / max(self._total_agents_generated, 1)
        )

        return {
            "total_agents_generated": self._total_agents_generated,
            "total_cost_usd": self._total_cost_usd,
            "total_generation_time_seconds": self._total_generation_time_seconds,
            "avg_time_per_agent_seconds": avg_time,
            "avg_cost_per_agent_usd": avg_cost,
            "target_time_seconds": 600,  # 10 minutes
            "performance_ratio": avg_time / 600,  # <1 is good
        }

    async def generate_batch(
        self,
        specs: List[AgentSpecV2],
        *,
        max_concurrent: int = 3,
    ) -> List[GenerationResult]:
        """
        Generate multiple agents concurrently.

        Args:
            specs: List of AgentSpecs to generate
            max_concurrent: Maximum concurrent generations (default: 3)

        Returns:
            List of GenerationResults
        """
        logger.info(f"Starting batch generation: {len(specs)} agents")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_semaphore(spec):
            async with semaphore:
                return await self.generate_agent(spec)

        # Generate all agents concurrently
        results = await asyncio.gather(*[
            generate_with_semaphore(spec) for spec in specs
        ])

        # Log summary
        successful = sum(1 for r in results if r.success)
        logger.info(
            f"Batch generation complete: {successful}/{len(specs)} successful "
            f"(total cost: ${sum(r.total_cost_usd for r in results):.2f})"
        )

        return results
