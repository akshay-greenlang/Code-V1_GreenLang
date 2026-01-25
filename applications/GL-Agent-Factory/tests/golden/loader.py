"""
GoldenTestLoader - YAML Test Case Loader

Loads golden test cases from YAML files and parses expected outputs
for validation against agent calculations.

YAML Test Format:
    test_suite:
      name: "Natural Gas US Tests"
      agent_id: "emissions/carbon_calculator_v1"
      version: "1.0.0"

    tests:
      - test_id: "carbon_001"
        test_name: "Natural Gas 1000 m3 US"
        description: "Basic natural gas calculation"
        input:
          fuel_type: "natural_gas"
          quantity: 1000
          unit: "m3"
          region: "US"
        expected:
          emissions_kgco2e: 1930.0
          emission_factor_used: 1.93
          scope: 1

Example:
    loader = GoldenTestLoader()
    tests = loader.load_from_file("tests/golden/carbon_emissions/test_natural_gas_us.yaml")
    for test in tests:
        print(f"Test: {test.test_name}, Expected: {test.expected_output}")
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExpectedOutput:
    """Expected output values for a golden test.

    Attributes:
        emissions_value: Primary emissions value (varies by agent)
        emission_factor: Emission factor used
        additional_fields: Other fields to validate
        provenance_hash: Expected hash (optional, for static tests)
    """

    emissions_value: Optional[float] = None
    emission_factor: Optional[float] = None
    additional_fields: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpectedOutput":
        """Create from dictionary."""
        return cls(
            emissions_value=data.get("emissions_kgco2e") or data.get("total_emissions_kgco2e"),
            emission_factor=data.get("emission_factor_used"),
            additional_fields=data,
            provenance_hash=data.get("provenance_hash"),
        )


@dataclass
class GoldenTestCase:
    """A single golden test case.

    Attributes:
        test_id: Unique identifier for the test
        test_name: Human-readable test name
        description: Detailed test description
        input_data: Input data for the agent
        expected_output: Expected output values
        tolerance_override: Optional per-test tolerance
        tags: Test categorization tags
        metadata: Additional test metadata
    """

    test_id: str
    test_name: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    description: str = ""
    tolerance_override: Optional[Dict[str, float]] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenTestCase":
        """Create from YAML dictionary."""
        return cls(
            test_id=data.get("test_id", "unknown"),
            test_name=data.get("test_name", data.get("name", "Unnamed Test")),
            description=data.get("description", ""),
            input_data=data.get("input", {}),
            expected_output=data.get("expected", {}),
            tolerance_override=data.get("tolerance"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GoldenTestFile:
    """Parsed golden test file.

    Attributes:
        suite_name: Name of the test suite
        agent_id: Agent being tested
        version: Agent version
        tests: List of test cases
        file_path: Source file path
        description: Suite description
    """

    suite_name: str
    agent_id: str
    version: str
    tests: List[GoldenTestCase]
    file_path: str = ""
    description: str = ""

    @property
    def test_count(self) -> int:
        """Number of tests in the file."""
        return len(self.tests)


class GoldenTestLoader:
    """Loads golden tests from YAML files.

    Supports loading individual files, directories, and filtering by tags.
    """

    def __init__(self, base_path: Optional[str] = None):
        """Initialize the loader.

        Args:
            base_path: Base path for test files (defaults to tests/golden/)
        """
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path(__file__).parent

    def load_from_file(self, file_path: Union[str, Path]) -> List[GoldenTestCase]:
        """Load test cases from a YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            List of GoldenTestCase objects
        """
        path = Path(file_path)

        if not path.is_absolute():
            path = self.base_path / path

        if not path.exists():
            raise FileNotFoundError(f"Test file not found: {path}")

        logger.info(f"Loading golden tests from: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return self._parse_yaml_data(data, str(path))

    def load_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "test_*.yaml",
    ) -> List[GoldenTestCase]:
        """Load all test files from a directory.

        Args:
            directory: Directory path
            pattern: Glob pattern for test files

        Returns:
            Combined list of test cases from all files
        """
        path = Path(directory)

        if not path.is_absolute():
            path = self.base_path / path

        if not path.exists():
            raise FileNotFoundError(f"Test directory not found: {path}")

        tests: List[GoldenTestCase] = []

        for test_file in sorted(path.glob(pattern)):
            try:
                file_tests = self.load_from_file(test_file)
                tests.extend(file_tests)
                logger.info(f"Loaded {len(file_tests)} tests from {test_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {test_file}: {e}")

        logger.info(f"Total tests loaded from {path}: {len(tests)}")
        return tests

    def load_by_agent(self, agent_id: str) -> List[GoldenTestCase]:
        """Load all tests for a specific agent.

        Args:
            agent_id: Agent identifier (e.g., "gl_001_carbon_emissions")

        Returns:
            List of test cases for the agent
        """
        # Map agent IDs to directories
        agent_dirs = {
            "gl_001": "carbon_emissions",
            "gl_002": "cbam",
            "gl_004": "eudr",
            "gl_006": "scope3",
            "emissions/carbon_calculator_v1": "carbon_emissions",
            "regulatory/cbam_compliance_v1": "cbam",
            "regulatory/eudr_compliance_v1": "eudr",
            "emissions/scope3_v1": "scope3",
        }

        directory = None
        for prefix, dir_name in agent_dirs.items():
            if agent_id.startswith(prefix) or agent_id == prefix:
                directory = dir_name
                break

        if not directory:
            raise ValueError(f"Unknown agent ID: {agent_id}")

        return self.load_from_directory(directory)

    def load_by_tags(
        self,
        directory: Union[str, Path],
        tags: List[str],
    ) -> List[GoldenTestCase]:
        """Load tests filtered by tags.

        Args:
            directory: Directory to search
            tags: Tags to filter by (OR logic)

        Returns:
            Tests matching any of the specified tags
        """
        all_tests = self.load_from_directory(directory)

        filtered = [
            test for test in all_tests
            if any(tag in test.tags for tag in tags)
        ]

        logger.info(f"Filtered to {len(filtered)} tests with tags: {tags}")
        return filtered

    def _parse_yaml_data(
        self,
        data: Dict[str, Any],
        file_path: str,
    ) -> List[GoldenTestCase]:
        """Parse YAML data into test cases.

        Supports two formats:
        1. Simple list of tests
        2. Full suite format with metadata
        """
        tests: List[GoldenTestCase] = []

        # Extract suite metadata
        suite_info = data.get("test_suite", {})

        # Get tests list
        test_list = data.get("tests", [])

        if not test_list:
            logger.warning(f"No tests found in {file_path}")
            return tests

        for i, test_data in enumerate(test_list):
            try:
                test = GoldenTestCase.from_dict(test_data)

                # Add suite metadata to each test
                test.metadata["suite_name"] = suite_info.get("name", "")
                test.metadata["agent_id"] = suite_info.get("agent_id", "")
                test.metadata["file_path"] = file_path

                tests.append(test)

            except Exception as e:
                logger.error(f"Failed to parse test {i} in {file_path}: {e}")

        return tests

    def get_test_summary(self, tests: List[GoldenTestCase]) -> Dict[str, Any]:
        """Generate summary of loaded tests.

        Args:
            tests: List of loaded test cases

        Returns:
            Summary dictionary with counts and categories
        """
        tags_count: Dict[str, int] = {}
        for test in tests:
            for tag in test.tags:
                tags_count[tag] = tags_count.get(tag, 0) + 1

        return {
            "total_tests": len(tests),
            "unique_test_ids": len(set(t.test_id for t in tests)),
            "tags": tags_count,
            "tests_with_tolerance_override": sum(
                1 for t in tests if t.tolerance_override
            ),
        }


class GoldenTestGenerator:
    """Generates golden test YAML files from agent specifications.

    Used to create initial test files based on agent emission factor databases.
    """

    def __init__(self, agent_id: str, agent_version: str):
        """Initialize generator.

        Args:
            agent_id: Agent identifier
            agent_version: Agent version
        """
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.tests: List[Dict[str, Any]] = []

    def add_test(
        self,
        test_id: str,
        test_name: str,
        input_data: Dict[str, Any],
        expected_output: Dict[str, Any],
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> "GoldenTestGenerator":
        """Add a test case.

        Args:
            test_id: Unique test ID
            test_name: Human-readable name
            input_data: Input data dictionary
            expected_output: Expected output dictionary
            description: Test description
            tags: Categorization tags

        Returns:
            Self for chaining
        """
        self.tests.append({
            "test_id": test_id,
            "test_name": test_name,
            "description": description,
            "tags": tags or [],
            "input": input_data,
            "expected": expected_output,
        })
        return self

    def generate_yaml(self, suite_name: str, description: str = "") -> str:
        """Generate YAML content for the test suite.

        Args:
            suite_name: Name of the test suite
            description: Suite description

        Returns:
            YAML string content
        """
        data = {
            "test_suite": {
                "name": suite_name,
                "agent_id": self.agent_id,
                "version": self.agent_version,
                "description": description,
            },
            "tests": self.tests,
        }

        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def save_to_file(
        self,
        file_path: Union[str, Path],
        suite_name: str,
        description: str = "",
    ) -> None:
        """Save tests to a YAML file.

        Args:
            file_path: Output file path
            suite_name: Name of the test suite
            description: Suite description
        """
        yaml_content = self.generate_yaml(suite_name, description)

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        logger.info(f"Saved {len(self.tests)} tests to {path}")


def create_carbon_emissions_tests() -> GoldenTestGenerator:
    """Create golden tests for Carbon Emissions Agent.

    Returns reference test cases based on EPA/DEFRA emission factors.
    """
    gen = GoldenTestGenerator("emissions/carbon_calculator_v1", "1.0.0")

    # Natural gas tests
    gen.add_test(
        "carbon_ng_us_1000",
        "Natural Gas 1000 m3 US",
        {"fuel_type": "natural_gas", "quantity": 1000, "unit": "m3", "region": "US", "scope": 1},
        {"emissions_kgco2e": 1930.0, "emission_factor_used": 1.93, "scope": 1},
        "Standard US natural gas calculation",
        ["natural_gas", "us", "scope1"],
    )

    return gen


def create_cbam_tests() -> GoldenTestGenerator:
    """Create golden tests for CBAM Compliance Agent."""
    gen = GoldenTestGenerator("regulatory/cbam_compliance_v1", "1.0.0")

    # Steel import tests
    gen.add_test(
        "cbam_steel_cn_1000",
        "Steel Import 1000t from China",
        {
            "cn_code": "72081000",
            "quantity_tonnes": 1000,
            "country_of_origin": "CN",
            "reporting_period": "Q1 2026",
        },
        {
            "product_category": "iron_steel",
            "direct_emissions_tco2e": 2100.0,
            "indirect_emissions_tco2e": 450.0,
            "total_embedded_emissions_tco2e": 2550.0,
            "cbam_liability_eur": 216750.0,
        },
        "China steel import with country-specific factors",
        ["steel", "china", "cbam"],
    )

    return gen
