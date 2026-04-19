"""
Dimension 10: Documentation Verification

This dimension verifies that agents have proper documentation
including docstrings, API documentation, and usage examples.

Checks:
    - Docstrings present
    - API documented
    - Examples provided
    - README exists

Example:
    >>> dimension = DocumentationDimension()
    >>> result = dimension.evaluate(agent_path, agent, sample_input)
    >>> assert result.status == DimensionStatus.PASS
"""

import ast
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseDimension, CheckResult, DimensionResult, DimensionStatus

logger = logging.getLogger(__name__)


class DocumentationDimension(BaseDimension):
    """
    Documentation Dimension Evaluator (D10).

    Verifies that agents have comprehensive documentation.

    Configuration:
        require_readme: Require README.md (default: True)
        require_examples: Require usage examples (default: True)
        min_docstring_coverage: Minimum docstring coverage % (default: 80)
    """

    DIMENSION_ID = "D10"
    DIMENSION_NAME = "Documentation"
    DESCRIPTION = "Verifies docstrings present, API documented, examples provided"
    WEIGHT = 0.8
    REQUIRED_FOR_CERTIFICATION = False  # Important but not blocking

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize documentation dimension evaluator."""
        super().__init__(config)

        self.require_readme = self.config.get("require_readme", True)
        self.require_examples = self.config.get("require_examples", True)
        self.min_docstring_coverage = self.config.get("min_docstring_coverage", 80)

    def evaluate(
        self,
        agent_path: Path,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
    ) -> DimensionResult:
        """
        Evaluate documentation for the given agent.

        Args:
            agent_path: Path to agent directory
            agent: Optional agent instance
            sample_input: Optional sample input

        Returns:
            DimensionResult with documentation evaluation
        """
        start_time = datetime.utcnow()
        self._reset_checks()

        logger.info("Starting documentation evaluation")

        # Load agent source code
        agent_file = agent_path / "agent.py"
        if not agent_file.exists():
            self._add_check(
                name="agent_file_exists",
                passed=False,
                message="agent.py not found",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        try:
            source_code = agent_file.read_text(encoding="utf-8")
        except Exception as e:
            self._add_check(
                name="source_readable",
                passed=False,
                message=f"Cannot read agent source: {str(e)}",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        # Check 1: Module docstring
        module_doc = self._check_module_docstring(source_code)
        self._add_check(
            name="module_docstring",
            passed=module_doc["has_docstring"],
            message="Module docstring present"
            if module_doc["has_docstring"]
            else "Missing module docstring",
            severity="error",
            details=module_doc,
        )

        # Check 2: Class docstrings
        class_docs = self._check_class_docstrings(source_code)
        coverage = class_docs["coverage"]
        self._add_check(
            name="class_docstrings",
            passed=coverage >= self.min_docstring_coverage,
            message=f"Class docstring coverage: {coverage:.0f}%"
            if class_docs["total_classes"] > 0
            else "No classes found",
            severity="warning" if coverage < self.min_docstring_coverage else "info",
            details=class_docs,
        )

        # Check 3: Method docstrings
        method_docs = self._check_method_docstrings(source_code)
        method_coverage = method_docs["coverage"]
        self._add_check(
            name="method_docstrings",
            passed=method_coverage >= self.min_docstring_coverage,
            message=f"Method docstring coverage: {method_coverage:.0f}%"
            if method_docs["total_methods"] > 0
            else "No methods found",
            severity="warning" if method_coverage < self.min_docstring_coverage else "info",
            details=method_docs,
        )

        # Check 4: README.md exists
        readme_check = self._check_readme(agent_path)
        self._add_check(
            name="readme_exists",
            passed=readme_check["exists"],
            message="README.md exists"
            if readme_check["exists"]
            else "Missing README.md",
            severity="warning" if self.require_readme else "info",
            details=readme_check,
        )

        # Check 5: Usage examples
        examples_check = self._check_examples(source_code, agent_path)
        self._add_check(
            name="usage_examples",
            passed=examples_check["has_examples"],
            message=f"Found {examples_check['example_count']} usage example(s)"
            if examples_check["has_examples"]
            else "No usage examples found",
            severity="warning" if self.require_examples else "info",
            details=examples_check,
        )

        # Check 6: API documentation
        api_docs = self._check_api_documentation(source_code)
        self._add_check(
            name="api_documentation",
            passed=api_docs["has_api_docs"],
            message="API documentation present"
            if api_docs["has_api_docs"]
            else "Missing API documentation",
            severity="warning",
            details=api_docs,
        )

        # Check 7: Type hints
        type_hints = self._check_type_hints(source_code)
        type_coverage = type_hints["coverage"]
        self._add_check(
            name="type_hints",
            passed=type_coverage >= 80,
            message=f"Type hint coverage: {type_coverage:.0f}%",
            severity="warning" if type_coverage < 80 else "info",
            details=type_hints,
        )

        # Check 8: Changelog/version history
        changelog_check = self._check_changelog(agent_path)
        self._add_check(
            name="changelog",
            passed=True,  # Optional
            message="Changelog present"
            if changelog_check["exists"]
            else "No changelog (optional)",
            severity="info",
            details=changelog_check,
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return self._create_result(
            execution_time,
            details={
                "module_docstring": module_doc["has_docstring"],
                "class_coverage": class_docs["coverage"],
                "method_coverage": method_docs["coverage"],
                "type_hint_coverage": type_hints["coverage"],
            },
        )

    def _check_module_docstring(self, source_code: str) -> Dict[str, Any]:
        """
        Check for module-level docstring.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with module docstring check results
        """
        result = {
            "has_docstring": False,
            "docstring_length": 0,
            "has_description": False,
            "has_example": False,
        }

        try:
            tree = ast.parse(source_code)
            docstring = ast.get_docstring(tree)

            if docstring:
                result["has_docstring"] = True
                result["docstring_length"] = len(docstring)
                result["has_description"] = len(docstring) > 50
                result["has_example"] = ">>>" in docstring or "Example:" in docstring

        except Exception as e:
            logger.error(f"Failed to parse module docstring: {str(e)}")

        return result

    def _check_class_docstrings(self, source_code: str) -> Dict[str, Any]:
        """
        Check class docstring coverage.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with class docstring coverage results
        """
        result = {
            "total_classes": 0,
            "documented_classes": 0,
            "coverage": 0.0,
            "undocumented": [],
        }

        try:
            tree = ast.parse(source_code)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    result["total_classes"] += 1
                    docstring = ast.get_docstring(node)

                    if docstring and len(docstring) > 10:
                        result["documented_classes"] += 1
                    else:
                        result["undocumented"].append(node.name)

            if result["total_classes"] > 0:
                result["coverage"] = (
                    result["documented_classes"] / result["total_classes"] * 100
                )

        except Exception as e:
            logger.error(f"Failed to parse class docstrings: {str(e)}")

        return result

    def _check_method_docstrings(self, source_code: str) -> Dict[str, Any]:
        """
        Check method docstring coverage.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with method docstring coverage results
        """
        result = {
            "total_methods": 0,
            "documented_methods": 0,
            "coverage": 0.0,
            "undocumented": [],
        }

        try:
            tree = ast.parse(source_code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip private methods (starting with _)
                    if node.name.startswith("__") and node.name.endswith("__"):
                        continue  # Skip dunder methods

                    result["total_methods"] += 1
                    docstring = ast.get_docstring(node)

                    if docstring and len(docstring) > 10:
                        result["documented_methods"] += 1
                    else:
                        result["undocumented"].append(node.name)

            if result["total_methods"] > 0:
                result["coverage"] = (
                    result["documented_methods"] / result["total_methods"] * 100
                )

        except Exception as e:
            logger.error(f"Failed to parse method docstrings: {str(e)}")

        return result

    def _check_readme(self, agent_path: Path) -> Dict[str, Any]:
        """
        Check for README.md file.

        Args:
            agent_path: Path to agent directory

        Returns:
            Dictionary with README check results
        """
        result = {
            "exists": False,
            "has_description": False,
            "has_installation": False,
            "has_usage": False,
            "word_count": 0,
        }

        readme_file = agent_path / "README.md"
        if readme_file.exists():
            result["exists"] = True
            try:
                content = readme_file.read_text(encoding="utf-8")
                result["word_count"] = len(content.split())

                # Check for common sections
                content_lower = content.lower()
                result["has_description"] = (
                    "description" in content_lower or len(content) > 100
                )
                result["has_installation"] = (
                    "install" in content_lower or "pip" in content_lower
                )
                result["has_usage"] = (
                    "usage" in content_lower or "example" in content_lower
                )

            except Exception:
                pass

        return result

    def _check_examples(
        self,
        source_code: str,
        agent_path: Path,
    ) -> Dict[str, Any]:
        """
        Check for usage examples.

        Args:
            source_code: Agent source code
            agent_path: Path to agent directory

        Returns:
            Dictionary with examples check results
        """
        result = {
            "has_examples": False,
            "example_count": 0,
            "example_locations": [],
        }

        # Check in docstrings
        doctest_pattern = re.compile(r">>>\s+.*\n")
        matches = doctest_pattern.findall(source_code)
        if matches:
            result["has_examples"] = True
            result["example_count"] += len(matches)
            result["example_locations"].append("docstrings")

        # Check for Example: sections
        example_pattern = re.compile(r"Example[s]?:\s*\n", re.IGNORECASE)
        if example_pattern.search(source_code):
            result["has_examples"] = True
            result["example_locations"].append("Example sections")

        # Check for examples directory
        examples_dir = agent_path / "examples"
        if examples_dir.exists():
            example_files = list(examples_dir.glob("*.py"))
            if example_files:
                result["has_examples"] = True
                result["example_count"] += len(example_files)
                result["example_locations"].append(f"examples/ ({len(example_files)} files)")

        # Check README for examples
        readme_file = agent_path / "README.md"
        if readme_file.exists():
            try:
                content = readme_file.read_text(encoding="utf-8")
                if "```python" in content or "```py" in content:
                    result["has_examples"] = True
                    result["example_locations"].append("README.md")
            except Exception:
                pass

        return result

    def _check_api_documentation(self, source_code: str) -> Dict[str, Any]:
        """
        Check for API documentation.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with API documentation check results
        """
        result = {
            "has_api_docs": False,
            "documented_endpoints": [],
        }

        # Check for Args/Returns in docstrings
        args_pattern = re.compile(r"Args:\s*\n", re.IGNORECASE)
        returns_pattern = re.compile(r"Returns:\s*\n", re.IGNORECASE)

        if args_pattern.search(source_code) and returns_pattern.search(source_code):
            result["has_api_docs"] = True
            result["documented_endpoints"].append("Function signatures")

        # Check for Pydantic model descriptions
        field_pattern = re.compile(r"Field\s*\([^)]*description\s*=", re.IGNORECASE)
        if field_pattern.search(source_code):
            result["has_api_docs"] = True
            result["documented_endpoints"].append("Pydantic field descriptions")

        return result

    def _check_type_hints(self, source_code: str) -> Dict[str, Any]:
        """
        Check type hint coverage.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with type hint check results
        """
        result = {
            "total_functions": 0,
            "typed_functions": 0,
            "coverage": 0.0,
            "untyped": [],
        }

        try:
            tree = ast.parse(source_code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    result["total_functions"] += 1

                    # Check for return annotation
                    has_return_type = node.returns is not None

                    # Check for parameter annotations
                    has_param_types = all(
                        arg.annotation is not None
                        for arg in node.args.args
                        if arg.arg != "self"
                    )

                    if has_return_type and (not node.args.args or has_param_types):
                        result["typed_functions"] += 1
                    else:
                        result["untyped"].append(node.name)

            if result["total_functions"] > 0:
                result["coverage"] = (
                    result["typed_functions"] / result["total_functions"] * 100
                )

        except Exception as e:
            logger.error(f"Failed to parse type hints: {str(e)}")

        return result

    def _check_changelog(self, agent_path: Path) -> Dict[str, Any]:
        """
        Check for changelog.

        Args:
            agent_path: Path to agent directory

        Returns:
            Dictionary with changelog check results
        """
        result = {
            "exists": False,
            "filename": None,
        }

        changelog_names = ["CHANGELOG.md", "HISTORY.md", "CHANGES.md", "changelog.md"]
        for name in changelog_names:
            changelog_file = agent_path / name
            if changelog_file.exists():
                result["exists"] = True
                result["filename"] = name
                break

        return result

    def _get_check_remediation(self, check: CheckResult) -> Optional[str]:
        """Get remediation for failed checks."""
        remediation_map = {
            "agent_file_exists": (
                "Create agent.py in the agent directory."
            ),
            "source_readable": (
                "Ensure agent.py is readable and uses UTF-8 encoding."
            ),
            "module_docstring": (
                'Add module docstring at the top of agent.py:\n'
                '  """\n'
                '  Agent Name - Brief description\n'
                '\n'
                '  Detailed description of what the agent does.\n'
                '\n'
                '  Example:\n'
                '      >>> agent = MyAgent()\n'
                '      >>> result = agent.run(input)\n'
                '  """'
            ),
            "class_docstrings": (
                "Add docstrings to all classes:\n"
                "  class MyClass:\n"
                '      """Brief description.\n'
                "\n"
                "      Detailed description.\n"
                "\n"
                "      Attributes:\n"
                "          attr1: Description\n"
                '      """'
            ),
            "method_docstrings": (
                "Add docstrings to all public methods:\n"
                "  def my_method(self, arg1: str) -> str:\n"
                '      """Brief description.\n'
                "\n"
                "      Args:\n"
                "          arg1: Description of arg1\n"
                "\n"
                "      Returns:\n"
                "          Description of return value\n"
                '      """'
            ),
            "readme_exists": (
                "Create README.md with:\n"
                "  - Description of the agent\n"
                "  - Installation instructions\n"
                "  - Usage examples\n"
                "  - Configuration options"
            ),
            "usage_examples": (
                "Add usage examples:\n"
                "  - Doctest examples in docstrings\n"
                "  - Code blocks in README.md\n"
                "  - Example scripts in examples/ directory"
            ),
            "api_documentation": (
                "Add API documentation:\n"
                "  - Args/Returns sections in docstrings\n"
                "  - Field descriptions in Pydantic models\n"
                "  - Type hints on all functions"
            ),
            "type_hints": (
                "Add type hints:\n"
                "  def method(self, arg: str) -> Result:\n"
                "      ...\n"
                "\n"
                "  Use typing module for complex types:\n"
                "  from typing import Dict, List, Optional"
            ),
        }

        return remediation_map.get(check.name)
