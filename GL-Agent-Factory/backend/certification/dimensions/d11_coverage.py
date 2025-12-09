"""
Dimension 11: Test Coverage Verification

This dimension verifies that agents have adequate test coverage
including unit tests, integration tests, and error path coverage.

Checks:
    - Code coverage > 90%
    - All paths tested
    - Error paths covered
    - Golden tests present

Example:
    >>> dimension = CoverageDimension()
    >>> result = dimension.evaluate(agent_path, agent, sample_input)
    >>> assert result.status == DimensionStatus.PASS
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseDimension, CheckResult, DimensionResult, DimensionStatus

logger = logging.getLogger(__name__)


class CoverageDimension(BaseDimension):
    """
    Test Coverage Dimension Evaluator (D11).

    Verifies that agents have comprehensive test coverage.

    Configuration:
        min_coverage: Minimum code coverage % (default: 90)
        require_golden_tests: Require golden tests (default: True)
        require_error_tests: Require error path tests (default: True)
    """

    DIMENSION_ID = "D11"
    DIMENSION_NAME = "Test Coverage"
    DESCRIPTION = "Verifies code coverage > 90%, all paths tested, error paths covered"
    WEIGHT = 1.0
    REQUIRED_FOR_CERTIFICATION = True

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize test coverage dimension evaluator."""
        super().__init__(config)

        self.min_coverage = self.config.get("min_coverage", 90)
        self.require_golden_tests = self.config.get("require_golden_tests", True)
        self.require_error_tests = self.config.get("require_error_tests", True)

    def evaluate(
        self,
        agent_path: Path,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
    ) -> DimensionResult:
        """
        Evaluate test coverage for the given agent.

        Args:
            agent_path: Path to agent directory
            agent: Optional agent instance
            sample_input: Optional sample input

        Returns:
            DimensionResult with test coverage evaluation
        """
        start_time = datetime.utcnow()
        self._reset_checks()

        logger.info("Starting test coverage evaluation")

        # Check 1: Test files exist
        test_files = self._find_test_files(agent_path)
        self._add_check(
            name="test_files_exist",
            passed=len(test_files) > 0,
            message=f"Found {len(test_files)} test file(s)"
            if test_files
            else "No test files found",
            severity="error",
            details={"test_files": [str(f.name) for f in test_files]},
        )

        # Check 2: Test count
        test_analysis = self._analyze_tests(test_files)
        self._add_check(
            name="test_count",
            passed=test_analysis["total_tests"] >= 5,
            message=f"Found {test_analysis['total_tests']} test(s)"
            if test_analysis["total_tests"] > 0
            else "No tests found",
            severity="warning" if test_analysis["total_tests"] < 5 else "info",
            details=test_analysis,
        )

        # Check 3: Golden tests present
        golden_check = self._check_golden_tests(agent_path)
        self._add_check(
            name="golden_tests",
            passed=golden_check["has_golden_tests"],
            message=f"Found {golden_check['golden_test_count']} golden test(s)"
            if golden_check["has_golden_tests"]
            else "No golden tests found",
            severity="error" if self.require_golden_tests else "warning",
            details=golden_check,
        )

        # Check 4: Error path tests
        error_tests = self._check_error_tests(test_files)
        self._add_check(
            name="error_path_tests",
            passed=error_tests["has_error_tests"],
            message=f"Found {error_tests['error_test_count']} error path test(s)"
            if error_tests["has_error_tests"]
            else "No error path tests found",
            severity="warning" if self.require_error_tests else "info",
            details=error_tests,
        )

        # Check 5: Edge case tests
        edge_tests = self._check_edge_case_tests(test_files)
        self._add_check(
            name="edge_case_tests",
            passed=edge_tests["has_edge_tests"],
            message=f"Found {edge_tests['edge_test_count']} edge case test(s)"
            if edge_tests["has_edge_tests"]
            else "No edge case tests found",
            severity="warning",
            details=edge_tests,
        )

        # Check 6: Coverage configuration
        coverage_config = self._check_coverage_config(agent_path)
        self._add_check(
            name="coverage_configured",
            passed=coverage_config["has_config"],
            message="Coverage configuration present"
            if coverage_config["has_config"]
            else "No coverage configuration found",
            severity="warning",
            details=coverage_config,
        )

        # Check 7: Test structure
        structure_check = self._check_test_structure(test_files)
        self._add_check(
            name="test_structure",
            passed=structure_check["well_structured"],
            message="Tests are well structured"
            if structure_check["well_structured"]
            else "Test structure could be improved",
            severity="warning",
            details=structure_check,
        )

        # Check 8: Fixture usage
        fixture_check = self._check_fixtures(test_files)
        self._add_check(
            name="fixtures",
            passed=True,  # Optional but good
            message=f"Uses {fixture_check['fixture_count']} fixture(s)"
            if fixture_check["has_fixtures"]
            else "No fixtures (consider using pytest fixtures)",
            severity="info",
            details=fixture_check,
        )

        # Check 9: Estimated coverage
        estimated_coverage = self._estimate_coverage(agent_path, test_analysis)
        self._add_check(
            name="estimated_coverage",
            passed=estimated_coverage["estimated_pct"] >= self.min_coverage,
            message=f"Estimated coverage: {estimated_coverage['estimated_pct']:.0f}% (target: {self.min_coverage}%)",
            severity="warning" if estimated_coverage["estimated_pct"] < self.min_coverage else "info",
            details=estimated_coverage,
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return self._create_result(
            execution_time,
            details={
                "test_files_count": len(test_files),
                "total_tests": test_analysis["total_tests"],
                "golden_test_count": golden_check.get("golden_test_count", 0),
                "estimated_coverage": estimated_coverage["estimated_pct"],
            },
        )

    def _find_test_files(self, agent_path: Path) -> List[Path]:
        """
        Find all test files in agent directory.

        Args:
            agent_path: Path to agent directory

        Returns:
            List of test file paths
        """
        test_files = []

        # Look for test files in tests/ directory
        tests_dir = agent_path / "tests"
        if tests_dir.exists():
            test_files.extend(tests_dir.glob("test_*.py"))
            test_files.extend(tests_dir.glob("*_test.py"))

        # Look for test files in agent directory
        test_files.extend(agent_path.glob("test_*.py"))
        test_files.extend(agent_path.glob("*_test.py"))

        return list(set(test_files))

    def _analyze_tests(self, test_files: List[Path]) -> Dict[str, Any]:
        """
        Analyze test files for test counts and types.

        Args:
            test_files: List of test file paths

        Returns:
            Dictionary with test analysis results
        """
        result = {
            "total_tests": 0,
            "test_classes": 0,
            "test_functions": 0,
            "async_tests": 0,
        }

        for test_file in test_files:
            try:
                content = test_file.read_text(encoding="utf-8")

                # Count test functions
                test_func_pattern = re.compile(r"def\s+(test_\w+)\s*\(")
                test_funcs = test_func_pattern.findall(content)
                result["test_functions"] += len(test_funcs)
                result["total_tests"] += len(test_funcs)

                # Count test classes
                test_class_pattern = re.compile(r"class\s+(Test\w+)\s*[\(:]")
                test_classes = test_class_pattern.findall(content)
                result["test_classes"] += len(test_classes)

                # Count async tests
                async_pattern = re.compile(r"async\s+def\s+test_\w+")
                async_tests = async_pattern.findall(content)
                result["async_tests"] += len(async_tests)

            except Exception as e:
                logger.warning(f"Failed to analyze {test_file}: {str(e)}")

        return result

    def _check_golden_tests(self, agent_path: Path) -> Dict[str, Any]:
        """
        Check for golden tests.

        Args:
            agent_path: Path to agent directory

        Returns:
            Dictionary with golden test check results
        """
        result = {
            "has_golden_tests": False,
            "golden_test_count": 0,
            "locations": [],
        }

        # Check pack.yaml
        try:
            import yaml

            pack_file = agent_path / "pack.yaml"
            if pack_file.exists():
                with open(pack_file, "r", encoding="utf-8") as f:
                    pack_spec = yaml.safe_load(f)

                golden_tests = pack_spec.get("tests", {}).get("golden", [])
                if golden_tests:
                    result["has_golden_tests"] = True
                    result["golden_test_count"] += len(golden_tests)
                    result["locations"].append("pack.yaml")

        except Exception:
            pass

        # Check golden_tests.yaml
        golden_file = agent_path / "golden_tests.yaml"
        if golden_file.exists():
            try:
                import yaml

                with open(golden_file, "r", encoding="utf-8") as f:
                    golden_spec = yaml.safe_load(f)

                tests = golden_spec.get("test_cases", [])
                if tests:
                    result["has_golden_tests"] = True
                    result["golden_test_count"] += len(tests)
                    result["locations"].append("golden_tests.yaml")

            except Exception:
                pass

        # Check for golden tests in test files
        tests_dir = agent_path / "tests"
        if tests_dir.exists():
            for test_file in tests_dir.glob("*golden*"):
                result["has_golden_tests"] = True
                result["locations"].append(str(test_file.name))

        return result

    def _check_error_tests(self, test_files: List[Path]) -> Dict[str, Any]:
        """
        Check for error path tests.

        Args:
            test_files: List of test file paths

        Returns:
            Dictionary with error test check results
        """
        result = {
            "has_error_tests": False,
            "error_test_count": 0,
            "error_patterns": [],
        }

        error_patterns = [
            (r"pytest\.raises", "pytest.raises"),
            (r"assertRaises", "assertRaises"),
            (r"test_\w*error", "error test function"),
            (r"test_\w*invalid", "invalid input test"),
            (r"test_\w*fail", "failure test"),
            (r"with\s+pytest\.raises\s*\(", "exception testing"),
        ]

        for test_file in test_files:
            try:
                content = test_file.read_text(encoding="utf-8")

                for pattern, description in error_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        result["has_error_tests"] = True
                        if description not in result["error_patterns"]:
                            result["error_patterns"].append(description)
                        result["error_test_count"] += len(
                            re.findall(pattern, content, re.IGNORECASE)
                        )

            except Exception:
                pass

        return result

    def _check_edge_case_tests(self, test_files: List[Path]) -> Dict[str, Any]:
        """
        Check for edge case tests.

        Args:
            test_files: List of test file paths

        Returns:
            Dictionary with edge case test check results
        """
        result = {
            "has_edge_tests": False,
            "edge_test_count": 0,
            "edge_patterns": [],
        }

        edge_patterns = [
            (r"test_\w*zero", "zero value test"),
            (r"test_\w*empty", "empty input test"),
            (r"test_\w*null|test_\w*none", "null/None test"),
            (r"test_\w*boundary", "boundary test"),
            (r"test_\w*large|test_\w*max", "large value test"),
            (r"test_\w*negative", "negative value test"),
            (r"test_\w*edge", "edge case test"),
        ]

        for test_file in test_files:
            try:
                content = test_file.read_text(encoding="utf-8")

                for pattern, description in edge_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        result["has_edge_tests"] = True
                        if description not in result["edge_patterns"]:
                            result["edge_patterns"].append(description)
                        result["edge_test_count"] += len(
                            re.findall(pattern, content, re.IGNORECASE)
                        )

            except Exception:
                pass

        return result

    def _check_coverage_config(self, agent_path: Path) -> Dict[str, Any]:
        """
        Check for coverage configuration.

        Args:
            agent_path: Path to agent directory

        Returns:
            Dictionary with coverage config check results
        """
        result = {
            "has_config": False,
            "config_file": None,
            "min_coverage_set": False,
        }

        config_files = [
            ("pyproject.toml", r"\[tool\.coverage"),
            (".coveragerc", r"\[run\]"),
            ("setup.cfg", r"\[coverage:"),
            ("pytest.ini", r"--cov"),
        ]

        for filename, pattern in config_files:
            config_path = agent_path / filename
            if config_path.exists():
                try:
                    content = config_path.read_text(encoding="utf-8")
                    if re.search(pattern, content):
                        result["has_config"] = True
                        result["config_file"] = filename

                        # Check for minimum coverage
                        if re.search(r"fail_under\s*=\s*\d+", content):
                            result["min_coverage_set"] = True

                        break

                except Exception:
                    pass

        return result

    def _check_test_structure(self, test_files: List[Path]) -> Dict[str, Any]:
        """
        Check test file structure.

        Args:
            test_files: List of test file paths

        Returns:
            Dictionary with test structure check results
        """
        result = {
            "well_structured": False,
            "has_setup": False,
            "has_teardown": False,
            "uses_classes": False,
            "issues": [],
        }

        for test_file in test_files:
            try:
                content = test_file.read_text(encoding="utf-8")

                # Check for setup/teardown
                if re.search(r"def\s+setup|@pytest\.fixture", content):
                    result["has_setup"] = True

                if re.search(r"def\s+teardown|yield", content):
                    result["has_teardown"] = True

                # Check for test classes
                if re.search(r"class\s+Test\w+", content):
                    result["uses_classes"] = True

            except Exception:
                pass

        # Well structured if has setup and reasonable organization
        result["well_structured"] = result["has_setup"] or result["uses_classes"]

        return result

    def _check_fixtures(self, test_files: List[Path]) -> Dict[str, Any]:
        """
        Check for pytest fixtures.

        Args:
            test_files: List of test file paths

        Returns:
            Dictionary with fixture check results
        """
        result = {
            "has_fixtures": False,
            "fixture_count": 0,
            "fixture_names": [],
        }

        fixture_pattern = re.compile(r"@pytest\.fixture\s*(?:\([^)]*\))?\s*\ndef\s+(\w+)")

        for test_file in test_files:
            try:
                content = test_file.read_text(encoding="utf-8")

                fixtures = fixture_pattern.findall(content)
                if fixtures:
                    result["has_fixtures"] = True
                    result["fixture_count"] += len(fixtures)
                    result["fixture_names"].extend(fixtures)

            except Exception:
                pass

        # Check for conftest.py
        conftest_files = list(Path(test_files[0].parent if test_files else ".").glob("**/conftest.py"))
        for conftest in conftest_files:
            try:
                content = conftest.read_text(encoding="utf-8")
                fixtures = fixture_pattern.findall(content)
                if fixtures:
                    result["has_fixtures"] = True
                    result["fixture_count"] += len(fixtures)
                    result["fixture_names"].extend(fixtures)
            except Exception:
                pass

        return result

    def _estimate_coverage(
        self,
        agent_path: Path,
        test_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Estimate test coverage based on test analysis.

        Args:
            agent_path: Path to agent directory
            test_analysis: Results from test analysis

        Returns:
            Dictionary with estimated coverage
        """
        result = {
            "estimated_pct": 0,
            "method": "heuristic",
            "factors": [],
        }

        # Base estimation on test count vs code complexity
        agent_file = agent_path / "agent.py"
        if not agent_file.exists():
            return result

        try:
            content = agent_file.read_text(encoding="utf-8")

            # Count functions in agent
            func_pattern = re.compile(r"def\s+(\w+)\s*\(")
            functions = func_pattern.findall(content)
            func_count = len(functions)

            # Count lines of code (rough)
            lines = len([l for l in content.split("\n") if l.strip() and not l.strip().startswith("#")])

            test_count = test_analysis["total_tests"]

            # Heuristic: coverage ~ tests_per_function * 30
            if func_count > 0:
                tests_per_func = test_count / func_count
                estimated = min(100, tests_per_func * 30)
                result["estimated_pct"] = estimated
                result["factors"].append(f"{tests_per_func:.1f} tests per function")

            # Bonus for having many tests
            if test_count >= 10:
                result["estimated_pct"] = min(100, result["estimated_pct"] + 10)
                result["factors"].append("Good test count bonus")

        except Exception as e:
            logger.error(f"Coverage estimation failed: {str(e)}")

        return result

    def _get_check_remediation(self, check: CheckResult) -> Optional[str]:
        """Get remediation for failed checks."""
        remediation_map = {
            "test_files_exist": (
                "Create test files in tests/ directory:\n"
                "  tests/\n"
                "    test_agent.py\n"
                "    test_calculations.py\n"
                "    conftest.py"
            ),
            "test_count": (
                "Add more tests:\n"
                "  - At least one test per public method\n"
                "  - Tests for different input scenarios\n"
                "  - Edge case tests"
            ),
            "golden_tests": (
                "Add golden tests in pack.yaml:\n"
                "  tests:\n"
                "    golden:\n"
                "      - name: 'test_basic'\n"
                "        input: {...}\n"
                "        expected: {...}"
            ),
            "error_path_tests": (
                "Add error path tests:\n"
                "  def test_invalid_input():\n"
                "      with pytest.raises(ValueError):\n"
                "          agent.run(invalid_input)"
            ),
            "edge_case_tests": (
                "Add edge case tests:\n"
                "  - test_zero_quantity()\n"
                "  - test_empty_input()\n"
                "  - test_max_value()\n"
                "  - test_negative_value()"
            ),
            "coverage_configured": (
                "Add coverage configuration to pyproject.toml:\n"
                "  [tool.coverage.run]\n"
                "  source = ['agent']\n"
                "  \n"
                "  [tool.coverage.report]\n"
                "  fail_under = 90"
            ),
            "test_structure": (
                "Improve test structure:\n"
                "  - Use pytest fixtures for setup\n"
                "  - Group related tests in classes\n"
                "  - Add conftest.py for shared fixtures"
            ),
            "estimated_coverage": (
                f"Increase test coverage (target: {self.min_coverage}%):\n"
                "  - Run: pytest --cov=agent --cov-report=term-missing\n"
                "  - Add tests for uncovered lines\n"
                "  - Focus on business logic paths"
            ),
        }

        return remediation_map.get(check.name)
