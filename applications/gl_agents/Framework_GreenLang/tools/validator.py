"""
GreenLang Framework - Agent Validator

Validates agents against Framework standards.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import ast
import re


class AgentValidator:
    """
    Validates GreenLang agents against Framework standards.

    Checks:
    - Directory structure
    - Required files
    - Code quality patterns
    - Provenance implementation
    - Test coverage
    """

    REQUIRED_DIRECTORIES = [
        "core",
        "models",
        "tests",
    ]

    REQUIRED_FILES = [
        "__init__.py",
        "README.md",
    ]

    RECOMMENDED_FILES = [
        "pyproject.toml",
        "Dockerfile",
    ]

    def __init__(self, agent_path: str):
        """Initialize validator with agent path."""
        self.agent_path = Path(agent_path)

    def validate(self) -> Dict[str, Any]:
        """
        Run full validation.

        Returns:
            Validation report with findings
        """
        report = {
            "agent_path": str(self.agent_path),
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "info": [],
            "checks": {},
        }

        # Check directory structure
        dir_check = self._check_directories()
        report["checks"]["directories"] = dir_check
        report["errors"].extend(dir_check.get("errors", []))
        report["warnings"].extend(dir_check.get("warnings", []))

        # Check required files
        file_check = self._check_files()
        report["checks"]["files"] = file_check
        report["errors"].extend(file_check.get("errors", []))
        report["warnings"].extend(file_check.get("warnings", []))

        # Check provenance implementation
        prov_check = self._check_provenance()
        report["checks"]["provenance"] = prov_check
        if not prov_check.get("has_provenance"):
            report["warnings"].append("Provenance tracking not detected")

        # Check data models
        model_check = self._check_models()
        report["checks"]["models"] = model_check
        report["warnings"].extend(model_check.get("warnings", []))

        # Check tests
        test_check = self._check_tests()
        report["checks"]["tests"] = test_check
        if test_check.get("test_count", 0) == 0:
            report["errors"].append("No tests found")

        # Determine overall validity
        report["is_valid"] = len(report["errors"]) == 0

        return report

    def _check_directories(self) -> Dict[str, Any]:
        """Check required directories exist."""
        result = {
            "found": [],
            "missing": [],
            "errors": [],
            "warnings": [],
        }

        for dir_name in self.REQUIRED_DIRECTORIES:
            dir_path = self.agent_path / dir_name
            if dir_path.is_dir():
                result["found"].append(dir_name)
            else:
                result["missing"].append(dir_name)
                result["errors"].append(f"Required directory missing: {dir_name}")

        # Check for recommended directories
        for dir_name in ["api", "deployment", "explainability"]:
            dir_path = self.agent_path / dir_name
            if not dir_path.is_dir():
                result["warnings"].append(f"Recommended directory missing: {dir_name}")

        return result

    def _check_files(self) -> Dict[str, Any]:
        """Check required files exist."""
        result = {
            "found": [],
            "missing": [],
            "errors": [],
            "warnings": [],
        }

        for file_name in self.REQUIRED_FILES:
            file_path = self.agent_path / file_name
            if file_path.is_file():
                result["found"].append(file_name)
            else:
                result["missing"].append(file_name)
                result["errors"].append(f"Required file missing: {file_name}")

        for file_name in self.RECOMMENDED_FILES:
            file_path = self.agent_path / file_name
            if not file_path.is_file():
                result["warnings"].append(f"Recommended file missing: {file_name}")

        return result

    def _check_provenance(self) -> Dict[str, Any]:
        """Check for provenance tracking implementation."""
        result = {
            "has_provenance": False,
            "hash_functions": [],
            "tracking_methods": [],
        }

        # Search for hash computation
        hash_patterns = [
            r"hashlib\.sha256",
            r"compute_hash",
            r"computation_hash",
            r"inputs_hash",
        ]

        for py_file in self.agent_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern in hash_patterns:
                    if re.search(pattern, content):
                        result["has_provenance"] = True
                        result["hash_functions"].append(str(py_file.relative_to(self.agent_path)))
                        break
            except Exception:
                pass

        return result

    def _check_models(self) -> Dict[str, Any]:
        """Check data model implementation."""
        result = {
            "has_pydantic": False,
            "model_files": [],
            "model_count": 0,
            "warnings": [],
        }

        models_dir = self.agent_path / "models"
        if not models_dir.is_dir():
            return result

        for py_file in models_dir.glob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                if "BaseModel" in content or "pydantic" in content:
                    result["has_pydantic"] = True
                    result["model_files"].append(py_file.name)

                    # Count model classes
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            for base in node.bases:
                                if isinstance(base, ast.Name) and "Model" in base.id:
                                    result["model_count"] += 1
            except Exception:
                pass

        if not result["has_pydantic"]:
            result["warnings"].append("Pydantic models not detected")

        return result

    def _check_tests(self) -> Dict[str, Any]:
        """Check test implementation."""
        result = {
            "test_count": 0,
            "test_files": [],
            "has_unit_tests": False,
            "has_integration_tests": False,
            "has_golden_tests": False,
        }

        tests_dir = self.agent_path / "tests"
        if not tests_dir.is_dir():
            return result

        for test_file in tests_dir.rglob("test_*.py"):
            result["test_files"].append(str(test_file.relative_to(self.agent_path)))

            try:
                content = test_file.read_text(encoding='utf-8')
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                        result["test_count"] += 1
            except Exception:
                pass

            # Categorize tests
            rel_path = str(test_file.relative_to(tests_dir))
            if "unit" in rel_path.lower():
                result["has_unit_tests"] = True
            if "integration" in rel_path.lower():
                result["has_integration_tests"] = True
            if "golden" in rel_path.lower():
                result["has_golden_tests"] = True

        return result

    def get_summary(self) -> str:
        """Get human-readable validation summary."""
        report = self.validate()

        lines = [
            f"Agent Validation: {self.agent_path.name}",
            "=" * 50,
            f"Status: {'VALID' if report['is_valid'] else 'INVALID'}",
            "",
        ]

        if report["errors"]:
            lines.append("ERRORS:")
            for error in report["errors"]:
                lines.append(f"  - {error}")
            lines.append("")

        if report["warnings"]:
            lines.append("WARNINGS:")
            for warning in report["warnings"]:
                lines.append(f"  - {warning}")
            lines.append("")

        # Add summary stats
        lines.append("Summary:")
        lines.append(f"  - Directories: {len(report['checks']['directories']['found'])} found")
        lines.append(f"  - Files: {len(report['checks']['files']['found'])} found")
        lines.append(f"  - Provenance: {'Yes' if report['checks']['provenance']['has_provenance'] else 'No'}")
        lines.append(f"  - Tests: {report['checks']['tests']['test_count']} test functions")

        return "\n".join(lines)
