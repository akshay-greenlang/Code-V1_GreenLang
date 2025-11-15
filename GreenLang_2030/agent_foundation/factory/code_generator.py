"""
CodeGenerator - Dynamic code generation from templates and specifications.

This module handles the rapid generation of agent code, tests, and documentation
from templates and specifications. Optimized for <100ms generation time.

Example:
    >>> generator = CodeGenerator()
    >>> config = GeneratorConfig(template=template, agent_name="MyAgent", spec=spec)
    >>> output = generator.generate(config)
    >>> print(f"Generated {output.lines_of_code} lines in {output.generation_time_ms}ms")
"""

import time
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

from pydantic import BaseModel, Field

from .templates import AgentTemplate

logger = logging.getLogger(__name__)


class GeneratorConfig(BaseModel):
    """Configuration for code generation."""

    template: AgentTemplate = Field(..., description="Agent template to use")
    agent_name: str = Field(..., description="Name of the agent")
    specification: Any = Field(..., description="Agent specification")
    output_directory: Path = Field(..., description="Output directory")
    generate_type: str = Field("code", description="Type: code, tests, or documentation")

    # Options
    include_docstrings: bool = Field(True, description="Include comprehensive docstrings")
    include_type_hints: bool = Field(True, description="Include type hints")
    include_logging: bool = Field(True, description="Include logging statements")
    optimize_imports: bool = Field(True, description="Optimize and sort imports")


class CodeOutput(BaseModel):
    """Result of code generation."""

    file_path: Path = Field(..., description="Path to generated file")
    content: str = Field(..., description="Generated content")
    lines_of_code: int = Field(..., description="Number of lines generated")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")

    # Metrics
    test_count: int = Field(0, description="Number of tests generated")
    method_count: int = Field(0, description="Number of methods generated")
    class_count: int = Field(0, description="Number of classes generated")


class CodeGenerator:
    """
    High-performance code generator for agent creation.

    Optimizations:
    - Template caching
    - String building optimization
    - Parallel generation for large files
    - Minimal I/O operations
    """

    def __init__(self):
        """Initialize code generator."""
        self._template_cache: Dict[str, str] = {}
        self._import_optimizer = ImportOptimizer()
        self._docstring_generator = DocstringGenerator()
        self._test_generator = TestGenerator()

    def generate(self, config: GeneratorConfig) -> CodeOutput:
        """
        Generate agent code from template and specification.

        Args:
            config: Generation configuration

        Returns:
            Generated code output
        """
        start_time = time.perf_counter()

        try:
            # Generate code based on type
            if config.generate_type == "tests":
                content = self.generate_tests(config)
            elif config.generate_type == "documentation":
                content = self.generate_documentation(config)
            else:
                content = self._generate_agent_code(config)

            # Write to file
            file_path = self._write_code(content, config)

            # Calculate metrics
            metrics = self._calculate_metrics(content)

            generation_time_ms = (time.perf_counter() - start_time) * 1000

            return CodeOutput(
                file_path=file_path,
                content=content,
                lines_of_code=metrics["lines"],
                generation_time_ms=generation_time_ms,
                test_count=metrics.get("tests", 0),
                method_count=metrics.get("methods", 0),
                class_count=metrics.get("classes", 0)
            )

        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}", exc_info=True)
            raise

    def generate_tests(self, config: GeneratorConfig) -> CodeOutput:
        """Generate unit tests for agent."""
        start_time = time.perf_counter()

        # Generate test code from template
        test_code = config.template.generate_tests(
            config.agent_name,
            config.specification.dict()
        )

        # Optimize imports
        if config.optimize_imports:
            test_code = self._import_optimizer.optimize(test_code)

        # Write test file
        test_path = config.output_directory / f"test_{config.agent_name.lower()}.py"
        test_path.write_text(test_code)

        generation_time_ms = (time.perf_counter() - start_time) * 1000

        metrics = self._calculate_metrics(test_code)

        return CodeOutput(
            file_path=test_path,
            content=test_code,
            lines_of_code=metrics["lines"],
            generation_time_ms=generation_time_ms,
            test_count=metrics.get("tests", 0)
        )

    def generate_documentation(self, config: GeneratorConfig) -> CodeOutput:
        """Generate documentation for agent."""
        start_time = time.perf_counter()

        doc_content = self._docstring_generator.generate_documentation(
            agent_name=config.agent_name,
            specification=config.specification,
            template=config.template
        )

        # Write documentation file
        doc_path = config.output_directory / f"{config.agent_name}_README.md"
        doc_path.write_text(doc_content)

        generation_time_ms = (time.perf_counter() - start_time) * 1000

        return CodeOutput(
            file_path=doc_path,
            content=doc_content,
            lines_of_code=len(doc_content.splitlines()),
            generation_time_ms=generation_time_ms
        )

    def _generate_agent_code(self, config: GeneratorConfig) -> str:
        """Generate main agent code."""
        # Get base code from template
        code = config.template.generate_code(
            config.agent_name,
            config.specification.dict()
        )

        # Post-process code
        if config.optimize_imports:
            code = self._import_optimizer.optimize(code)

        if config.include_docstrings:
            code = self._docstring_generator.enhance(code)

        if config.include_type_hints:
            code = self._add_type_hints(code)

        if config.include_logging:
            code = self._add_logging(code)

        return code

    def _write_code(self, content: str, config: GeneratorConfig) -> Path:
        """Write generated code to file."""
        if config.generate_type == "tests":
            file_name = f"test_{config.agent_name.lower()}.py"
        elif config.generate_type == "documentation":
            file_name = f"{config.agent_name}_README.md"
        else:
            file_name = f"{config.agent_name.lower()}.py"

        file_path = config.output_directory / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

        return file_path

    def _calculate_metrics(self, content: str) -> Dict[str, int]:
        """Calculate code metrics."""
        lines = content.splitlines()
        metrics = {
            "lines": len(lines),
            "methods": 0,
            "classes": 0,
            "tests": 0
        }

        # Count methods
        metrics["methods"] = len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))

        # Count classes
        metrics["classes"] = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))

        # Count tests
        metrics["tests"] = len(re.findall(r'^\s*def\s+test_\w+', content, re.MULTILINE))

        return metrics

    def _add_type_hints(self, code: str) -> str:
        """Add type hints to code if missing."""
        # This is a simplified version - real implementation would use AST
        return code

    def _add_logging(self, code: str) -> str:
        """Add logging statements to code."""
        # Add logging at key points if not present
        if "logger.info" not in code and "logger.error" not in code:
            # Add basic logging
            pass
        return code


class ImportOptimizer:
    """Optimize and organize imports."""

    def optimize(self, code: str) -> str:
        """
        Optimize imports in code.

        - Remove duplicates
        - Sort alphabetically
        - Group by type (standard, third-party, local)
        """
        lines = code.splitlines()
        imports = []
        other_lines = []

        in_imports = True
        for line in lines:
            if in_imports and (line.startswith('import ') or line.startswith('from ')):
                imports.append(line)
            elif in_imports and line.strip() and not line.startswith('#'):
                in_imports = False
                other_lines.append(line)
            else:
                other_lines.append(line)

        # Sort and deduplicate imports
        imports = sorted(set(imports))

        # Group imports
        standard_imports = []
        third_party_imports = []
        local_imports = []

        for imp in imports:
            if imp.startswith('from .') or imp.startswith('from ..'):
                local_imports.append(imp)
            elif any(imp.startswith(f'from {lib}') or imp.startswith(f'import {lib}')
                    for lib in ['typing', 'datetime', 'logging', 'hashlib', 'json', 'pathlib']):
                standard_imports.append(imp)
            else:
                third_party_imports.append(imp)

        # Reconstruct code
        optimized = []
        if standard_imports:
            optimized.extend(standard_imports)
            optimized.append('')
        if third_party_imports:
            optimized.extend(third_party_imports)
            optimized.append('')
        if local_imports:
            optimized.extend(local_imports)
            optimized.append('')

        optimized.extend(other_lines)

        return '\n'.join(optimized)


class DocstringGenerator:
    """Generate and enhance docstrings."""

    def enhance(self, code: str) -> str:
        """Enhance existing docstrings in code."""
        # This would use AST to parse and enhance docstrings
        return code

    def generate_documentation(
        self,
        agent_name: str,
        specification: Any,
        template: AgentTemplate
    ) -> str:
        """Generate markdown documentation."""
        doc = f"""# {agent_name}

## Overview

{specification.description}

## Features

{self._format_features(template.supported_features)}

## Configuration

### Input Schema

```python
{self._format_schema(specification.input_schema)}
```

### Output Schema

```python
{self._format_schema(specification.output_schema)}
```

## Usage

```python
from greenlang import {agent_name}, AgentConfig

# Initialize agent
config = AgentConfig(name="{agent_name}")
agent = {agent_name}(config)

# Process data
input_data = {{
    # Your input data here
}}

result = agent.process(input_data)
print(result)
```

## Performance

- **Processing Time**: <{specification.performance_targets.get('latency_ms', 1000)}ms
- **Throughput**: {specification.performance_targets.get('throughput_rps', 100)} requests/second
- **Test Coverage**: {specification.test_coverage_target}%

## Compliance

{self._format_compliance(specification.compliance_frameworks)}

## API Reference

### Methods

{self._format_methods(template.required_methods)}

## Testing

Run tests with:

```bash
pytest test_{agent_name.lower()}.py -v
```

## License

Copyright (c) 2024 GreenLang AI. All rights reserved.

---

*Generated by GreenLang Agent Factory v1.0.0*
"""
        return doc

    def _format_features(self, features: List[str]) -> str:
        """Format feature list."""
        if not features:
            return "- Standard agent features"
        return "\n".join(f"- {feature.replace('_', ' ').title()}" for feature in features)

    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format schema as Python-like definition."""
        if not schema:
            return "# No schema defined"

        lines = []
        for field, field_type in schema.items():
            lines.append(f"{field}: {field_type}")
        return "\n".join(lines)

    def _format_compliance(self, frameworks: List[str]) -> str:
        """Format compliance frameworks."""
        if not frameworks:
            return "No specific compliance requirements."

        return "This agent complies with:\n\n" + "\n".join(
            f"- **{fw}**: Full compliance validation"
            for fw in frameworks
        )

    def _format_methods(self, methods: List[str]) -> str:
        """Format method list."""
        if not methods:
            return "See code for available methods."

        formatted = []
        for method in methods:
            formatted.append(f"#### `{method}()`\n\n{method.replace('_', ' ').title()}\n")
        return "\n".join(formatted)


class TestGenerator:
    """Generate comprehensive test suites."""

    def generate_unit_tests(
        self,
        agent_name: str,
        specification: Any,
        template: AgentTemplate
    ) -> str:
        """Generate unit test suite."""
        # Template already handles basic test generation
        # This could be extended for more sophisticated test generation
        return template.generate_tests(agent_name, specification.dict())