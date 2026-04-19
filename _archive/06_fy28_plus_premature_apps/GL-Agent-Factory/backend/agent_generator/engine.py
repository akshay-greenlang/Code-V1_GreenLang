"""
Agent Generator Engine for GreenLang Agent Factory

This module provides the main orchestration engine for agent code generation.
It coordinates the complete pipeline from YAML spec to generated Python code.

The engine handles:
- Loading and parsing pack.yaml specifications
- Validating specs against schema and business rules
- Orchestrating code generation via multiple generators
- Managing output directories and file creation
- Progress callbacks and error handling with rollback
- Async support for parallel generation

Example:
    >>> engine = AgentGeneratorEngine()
    >>> result = await engine.generate_from_yaml("path/to/pack.yaml")
    >>> print(f"Generated {len(result.files_created)} files")
"""

import asyncio
import hashlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from backend.agent_generator.parser.yaml_parser import (
    YAMLParser,
    AgentSpec,
    AgentDefinition,
    AgentDef,
    InputDef,
    OutputDef,
)
from backend.agent_generator.parser.spec_validator import (
    SpecValidator,
    ValidationResult,
)
from backend.agent_generator.generators.agent_gen import AgentGenerator
from backend.agent_generator.generators.model_gen import ModelGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# Types and Enums
# =============================================================================

class GenerationStatus(str, Enum):
    """Status of the generation process."""

    PENDING = "pending"
    PARSING = "parsing"
    VALIDATING = "validating"
    GENERATING = "generating"
    WRITING = "writing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class GenerationStep(str, Enum):
    """Individual generation steps."""

    PARSE_YAML = "parse_yaml"
    VALIDATE_SPEC = "validate_spec"
    GENERATE_MODELS = "generate_models"
    GENERATE_AGENTS = "generate_agents"
    GENERATE_TESTS = "generate_tests"
    GENERATE_INIT = "generate_init"
    WRITE_FILES = "write_files"
    FINALIZE = "finalize"


@dataclass
class GeneratedFile:
    """Represents a generated file."""

    path: str
    content: str
    file_type: str  # 'python', 'yaml', 'json', etc.
    checksum: str = ""

    def __post_init__(self):
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()


@dataclass
class GenerationProgress:
    """Progress update for callbacks."""

    step: GenerationStep
    status: GenerationStatus
    progress_percent: float
    message: str
    current_item: Optional[str] = None
    items_completed: int = 0
    items_total: int = 0


@dataclass
class GenerationResult:
    """
    Result of agent generation.

    Contains all generated files, statistics, and metadata.
    """

    success: bool
    status: GenerationStatus
    spec: Optional[AgentSpec] = None
    validation_result: Optional[ValidationResult] = None
    files_created: List[str] = field(default_factory=list)
    generated_files: List[GeneratedFile] = field(default_factory=list)
    output_directory: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    generation_time_ms: float = 0.0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    @property
    def file_count(self) -> int:
        """Get total number of files created."""
        return len(self.files_created)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "status": self.status.value,
            "file_count": self.file_count,
            "files_created": self.files_created,
            "output_directory": self.output_directory,
            "errors": self.errors,
            "warnings": self.warnings,
            "generation_time_ms": self.generation_time_ms,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# =============================================================================
# Generator Configuration
# =============================================================================

@dataclass
class GeneratorConfig:
    """
    Configuration for the agent generator engine.

    Attributes:
        output_base_path: Base path for generated output
        overwrite_existing: Whether to overwrite existing files
        generate_tests: Whether to generate test files
        generate_docs: Whether to generate documentation
        dry_run: If True, don't write files (just validate)
        enable_rollback: Whether to rollback on failure
        parallel_generation: Enable parallel generation
        template_path: Custom Jinja2 template directory
    """

    output_base_path: str = "./generated"
    overwrite_existing: bool = False
    generate_tests: bool = True
    generate_docs: bool = True
    dry_run: bool = False
    enable_rollback: bool = True
    parallel_generation: bool = True
    template_path: Optional[str] = None

    # Code style options
    line_length: int = 100
    use_black_formatting: bool = True
    add_type_hints: bool = True
    docstring_style: str = "google"  # 'google', 'numpy', 'sphinx'

    # Generation options
    include_provenance: bool = True
    include_logging: bool = True
    include_error_handling: bool = True


# =============================================================================
# Agent Generator Engine
# =============================================================================

class AgentGeneratorEngine:
    """
    Main orchestration engine for agent code generation.

    This engine coordinates the complete pipeline:
    1. Parse YAML spec
    2. Validate spec
    3. Generate models
    4. Generate agent code
    5. Generate tests
    6. Write files

    Supports async operation and progress callbacks.

    Example:
        >>> config = GeneratorConfig(output_base_path="./agents")
        >>> engine = AgentGeneratorEngine(config)
        >>> result = await engine.generate_from_yaml("pack.yaml")
        >>> if result.success:
        ...     print(f"Created {result.file_count} files")
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        progress_callback: Optional[Callable[[GenerationProgress], None]] = None,
    ):
        """
        Initialize the generator engine.

        Args:
            config: Generator configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config or GeneratorConfig()
        self.progress_callback = progress_callback

        # Initialize components
        self.parser = YAMLParser()
        self.validator = SpecValidator()
        self.agent_generator = AgentGenerator(self.config)
        self.model_generator = ModelGenerator(self.config)

        # State tracking
        self._current_status = GenerationStatus.PENDING
        self._created_files: List[str] = []
        self._rollback_files: List[str] = []

        logger.info("AgentGeneratorEngine initialized")

    async def generate_from_yaml(
        self,
        yaml_path: Union[str, Path],
        output_path: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate agent code from a pack.yaml file.

        This is the main entry point for async generation.

        Args:
            yaml_path: Path to the pack.yaml file
            output_path: Optional override for output directory

        Returns:
            GenerationResult with all generated files and metadata
        """
        start_time = datetime.utcnow()
        result = GenerationResult(
            success=False,
            status=GenerationStatus.PENDING,
            started_at=start_time,
        )

        try:
            # Step 1: Parse YAML
            self._update_progress(
                GenerationStep.PARSE_YAML,
                GenerationStatus.PARSING,
                10,
                "Parsing YAML specification...",
            )

            spec = self.parser.parse_file(yaml_path)
            result.spec = spec

            # Step 2: Validate spec
            self._update_progress(
                GenerationStep.VALIDATE_SPEC,
                GenerationStatus.VALIDATING,
                20,
                "Validating specification...",
            )

            validation_result = self.validator.validate(spec)
            result.validation_result = validation_result

            if not validation_result.is_valid:
                result.errors = [str(e) for e in validation_result.errors]
                result.warnings = [str(w) for w in validation_result.warnings]
                result.status = GenerationStatus.FAILED
                logger.error(f"Validation failed with {len(result.errors)} errors")
                return result

            result.warnings = [str(w) for w in validation_result.warnings]

            # Determine output directory
            output_dir = output_path or self.config.output_base_path
            output_dir = Path(output_dir)
            agent_dir = output_dir / f"gl_{spec.pack.id.replace('-', '_')}"
            result.output_directory = str(agent_dir)

            # Step 3: Generate code
            self._current_status = GenerationStatus.GENERATING

            if self.config.parallel_generation:
                generated_files = await self._generate_parallel(spec)
            else:
                generated_files = await self._generate_sequential(spec)

            result.generated_files = generated_files

            # Step 4: Write files
            if not self.config.dry_run:
                self._update_progress(
                    GenerationStep.WRITE_FILES,
                    GenerationStatus.WRITING,
                    90,
                    "Writing generated files...",
                )

                written_files = await self._write_files(generated_files, agent_dir)
                result.files_created = written_files

            # Finalize
            self._update_progress(
                GenerationStep.FINALIZE,
                GenerationStatus.COMPLETED,
                100,
                "Generation complete!",
            )

            result.success = True
            result.status = GenerationStatus.COMPLETED

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            result.errors.append(str(e))
            result.status = GenerationStatus.FAILED

            # Rollback if enabled
            if self.config.enable_rollback and self._created_files:
                await self._rollback()
                result.status = GenerationStatus.ROLLED_BACK

        finally:
            result.completed_at = datetime.utcnow()
            result.generation_time_ms = (
                result.completed_at - start_time
            ).total_seconds() * 1000

            logger.info(
                f"Generation {'succeeded' if result.success else 'failed'}: "
                f"{result.file_count} files in {result.generation_time_ms:.2f}ms"
            )

        return result

    def generate_from_yaml_sync(
        self,
        yaml_path: Union[str, Path],
        output_path: Optional[str] = None,
    ) -> GenerationResult:
        """
        Synchronous wrapper for generate_from_yaml.

        Use this when not in an async context.

        Args:
            yaml_path: Path to the pack.yaml file
            output_path: Optional override for output directory

        Returns:
            GenerationResult with all generated files
        """
        return asyncio.run(self.generate_from_yaml(yaml_path, output_path))

    async def generate_from_spec(
        self,
        spec: AgentSpec,
        output_path: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate agent code from an AgentSpec object.

        Use this when you already have a parsed spec.

        Args:
            spec: The AgentSpec to generate from
            output_path: Optional override for output directory

        Returns:
            GenerationResult with all generated files
        """
        start_time = datetime.utcnow()
        result = GenerationResult(
            success=False,
            status=GenerationStatus.PENDING,
            started_at=start_time,
            spec=spec,
        )

        try:
            # Validate
            validation_result = self.validator.validate(spec)
            result.validation_result = validation_result

            if not validation_result.is_valid:
                result.errors = [str(e) for e in validation_result.errors]
                result.status = GenerationStatus.FAILED
                return result

            result.warnings = [str(w) for w in validation_result.warnings]

            # Determine output directory
            output_dir = output_path or self.config.output_base_path
            output_dir = Path(output_dir)
            agent_dir = output_dir / f"gl_{spec.pack.id.replace('-', '_')}"
            result.output_directory = str(agent_dir)

            # Generate
            generated_files = await self._generate_parallel(spec)
            result.generated_files = generated_files

            # Write
            if not self.config.dry_run:
                written_files = await self._write_files(generated_files, agent_dir)
                result.files_created = written_files

            result.success = True
            result.status = GenerationStatus.COMPLETED

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            result.errors.append(str(e))
            result.status = GenerationStatus.FAILED

        finally:
            result.completed_at = datetime.utcnow()
            result.generation_time_ms = (
                result.completed_at - start_time
            ).total_seconds() * 1000

        return result

    async def _generate_parallel(self, spec: AgentSpec) -> List[GeneratedFile]:
        """Generate all files in parallel."""
        files: List[GeneratedFile] = []

        # Generate models
        self._update_progress(
            GenerationStep.GENERATE_MODELS,
            GenerationStatus.GENERATING,
            30,
            "Generating Pydantic models...",
        )
        model_files = await self._generate_models(spec)
        files.extend(model_files)

        # Generate agents in parallel
        self._update_progress(
            GenerationStep.GENERATE_AGENTS,
            GenerationStatus.GENERATING,
            50,
            "Generating agent code...",
            items_total=len(spec.agents),
        )

        agent_tasks = [
            self._generate_agent(spec, agent)
            for agent in spec.agents
        ]
        agent_results = await asyncio.gather(*agent_tasks)

        for i, agent_files in enumerate(agent_results):
            files.extend(agent_files)
            self._update_progress(
                GenerationStep.GENERATE_AGENTS,
                GenerationStatus.GENERATING,
                50 + (30 * (i + 1) / len(spec.agents)),
                f"Generated {spec.agents[i].name}",
                current_item=spec.agents[i].id,
                items_completed=i + 1,
                items_total=len(spec.agents),
            )

        # Generate tests
        if self.config.generate_tests:
            self._update_progress(
                GenerationStep.GENERATE_TESTS,
                GenerationStatus.GENERATING,
                85,
                "Generating test files...",
            )
            test_files = await self._generate_tests(spec)
            files.extend(test_files)

        # Generate __init__.py
        self._update_progress(
            GenerationStep.GENERATE_INIT,
            GenerationStatus.GENERATING,
            88,
            "Generating package files...",
        )
        init_files = self._generate_init_files(spec)
        files.extend(init_files)

        return files

    async def _generate_sequential(self, spec: AgentSpec) -> List[GeneratedFile]:
        """Generate all files sequentially."""
        files: List[GeneratedFile] = []

        # Generate models
        model_files = await self._generate_models(spec)
        files.extend(model_files)

        # Generate agents
        for agent in spec.agents:
            agent_files = await self._generate_agent(spec, agent)
            files.extend(agent_files)

        # Generate tests
        if self.config.generate_tests:
            test_files = await self._generate_tests(spec)
            files.extend(test_files)

        # Generate __init__.py
        init_files = self._generate_init_files(spec)
        files.extend(init_files)

        return files

    async def _generate_models(self, spec: AgentSpec) -> List[GeneratedFile]:
        """Generate Pydantic model files."""
        files = []

        # Generate shared models
        models_content = self.model_generator.generate_models(spec)
        files.append(GeneratedFile(
            path="models.py",
            content=models_content,
            file_type="python",
        ))

        return files

    async def _generate_agent(
        self,
        spec: AgentSpec,
        agent: AgentDefinition,
    ) -> List[GeneratedFile]:
        """Generate files for a single agent."""
        files = []

        # Generate main agent file
        agent_content = self.agent_generator.generate_agent(spec, agent)
        files.append(GeneratedFile(
            path=f"agents/{agent.get_module_name()}.py",
            content=agent_content,
            file_type="python",
        ))

        return files

    async def _generate_tests(self, spec: AgentSpec) -> List[GeneratedFile]:
        """Generate test files."""
        files = []

        # Generate test file for each agent
        for agent in spec.agents:
            test_content = self.agent_generator.generate_tests(spec, agent)
            files.append(GeneratedFile(
                path=f"tests/test_{agent.get_module_name()}.py",
                content=test_content,
                file_type="python",
            ))

        return files

    def _generate_init_files(self, spec: AgentSpec) -> List[GeneratedFile]:
        """Generate __init__.py files."""
        files = []

        # Root __init__.py
        root_init = self._generate_root_init(spec)
        files.append(GeneratedFile(
            path="__init__.py",
            content=root_init,
            file_type="python",
        ))

        # agents/__init__.py
        agents_init = self._generate_agents_init(spec)
        files.append(GeneratedFile(
            path="agents/__init__.py",
            content=agents_init,
            file_type="python",
        ))

        # tests/__init__.py
        if self.config.generate_tests:
            files.append(GeneratedFile(
                path="tests/__init__.py",
                content='"""Test package."""\n',
                file_type="python",
            ))

        return files

    def _generate_root_init(self, spec: AgentSpec) -> str:
        """Generate root __init__.py content."""
        lines = [
            '"""',
            f'{spec.pack.name}',
            '',
            f'{spec.pack.description}' if spec.pack.description else 'Generated by GreenLang Agent Factory.',
            '',
            f'Version: {spec.pack.version}',
            f'Pack ID: {spec.pack.id}',
            '"""',
            '',
            'from .models import *',
        ]

        # Import agents
        for agent in spec.agents:
            class_name = agent.get_class_name()
            module_name = agent.get_module_name()
            lines.append(f'from .agents.{module_name} import {class_name}')

        lines.append('')
        lines.append(f'__version__ = "{spec.pack.version}"')
        lines.append(f'__pack_id__ = "{spec.pack.id}"')
        lines.append('')

        # __all__ list
        all_exports = ['__version__', '__pack_id__']
        for agent in spec.agents:
            all_exports.append(agent.get_class_name())

        lines.append('__all__ = [')
        for export in all_exports:
            lines.append(f'    "{export}",')
        lines.append(']')
        lines.append('')

        return '\n'.join(lines)

    def _generate_agents_init(self, spec: AgentSpec) -> str:
        """Generate agents/__init__.py content."""
        lines = [
            '"""Agent implementations."""',
            '',
        ]

        for agent in spec.agents:
            class_name = agent.get_class_name()
            module_name = agent.get_module_name()
            lines.append(f'from .{module_name} import {class_name}')

        lines.append('')
        lines.append('__all__ = [')
        for agent in spec.agents:
            lines.append(f'    "{agent.get_class_name()}",')
        lines.append(']')
        lines.append('')

        return '\n'.join(lines)

    async def _write_files(
        self,
        files: List[GeneratedFile],
        output_dir: Path,
    ) -> List[str]:
        """Write generated files to disk."""
        written_files: List[str] = []

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        for gen_file in files:
            file_path = output_dir / gen_file.path

            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists
            if file_path.exists() and not self.config.overwrite_existing:
                logger.warning(f"Skipping existing file: {file_path}")
                continue

            # Track for rollback
            self._rollback_files.append(str(file_path))

            # Write file
            file_path.write_text(gen_file.content, encoding="utf-8")
            written_files.append(str(file_path))
            self._created_files.append(str(file_path))

            logger.debug(f"Created file: {file_path}")

        return written_files

    async def _rollback(self) -> None:
        """Rollback created files on failure."""
        logger.warning("Rolling back created files...")

        for file_path in self._created_files:
            try:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    logger.debug(f"Removed: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")

        self._created_files = []

    def _update_progress(
        self,
        step: GenerationStep,
        status: GenerationStatus,
        progress: float,
        message: str,
        current_item: Optional[str] = None,
        items_completed: int = 0,
        items_total: int = 0,
    ) -> None:
        """Update progress and call callback if registered."""
        self._current_status = status

        if self.progress_callback:
            progress_update = GenerationProgress(
                step=step,
                status=status,
                progress_percent=progress,
                message=message,
                current_item=current_item,
                items_completed=items_completed,
                items_total=items_total,
            )
            self.progress_callback(progress_update)

        logger.info(f"[{progress:.0f}%] {message}")


# =============================================================================
# Factory Function
# =============================================================================

def create_generator(
    output_path: str = "./generated",
    overwrite: bool = False,
    dry_run: bool = False,
    progress_callback: Optional[Callable[[GenerationProgress], None]] = None,
) -> AgentGeneratorEngine:
    """
    Factory function to create a configured generator engine.

    Args:
        output_path: Base path for generated output
        overwrite: Whether to overwrite existing files
        dry_run: If True, don't write files
        progress_callback: Optional progress callback

    Returns:
        Configured AgentGeneratorEngine
    """
    config = GeneratorConfig(
        output_base_path=output_path,
        overwrite_existing=overwrite,
        dry_run=dry_run,
    )
    return AgentGeneratorEngine(config, progress_callback)


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """CLI entry point for agent generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GreenLang Agent Generator"
    )
    parser.add_argument(
        "yaml_path",
        help="Path to pack.yaml file",
    )
    parser.add_argument(
        "-o", "--output",
        default="./generated",
        help="Output directory (default: ./generated)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and generate but don't write files",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create engine and generate
    engine = create_generator(
        output_path=args.output,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    result = await engine.generate_from_yaml(args.yaml_path)

    # Print result
    if result.success:
        print(f"\nGeneration successful!")
        print(f"Output directory: {result.output_directory}")
        print(f"Files created: {result.file_count}")
        print(f"Generation time: {result.generation_time_ms:.2f}ms")
    else:
        print(f"\nGeneration failed!")
        for error in result.errors:
            print(f"  Error: {error}")

    # Print warnings
    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  - {warning}")

    return 0 if result.success else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
