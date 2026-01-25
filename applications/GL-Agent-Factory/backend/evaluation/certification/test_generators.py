"""
Test Generators for Agent Certification

Auto-generates test cases for comprehensive agent validation including:
- Golden test cases from formulas
- Boundary condition tests
- Fuzz testing
- Adversarial inputs

Example:
    >>> generator = TestGenerator()
    >>> tests = generator.generate_all(pack_spec)
    >>> for test in tests.golden_tests:
    ...     print(f"Test: {test.name}")

"""

import hashlib
import json
import logging
import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


@dataclass
class GeneratedTest:
    """A generated test case."""
    test_id: str
    test_type: str  # golden, boundary, fuzz, adversarial
    name: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    expected_behavior: str = "success"  # success, validation_error, error
    tolerance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedTestSuite:
    """Collection of generated tests."""
    golden_tests: List[GeneratedTest] = field(default_factory=list)
    boundary_tests: List[GeneratedTest] = field(default_factory=list)
    fuzz_tests: List[GeneratedTest] = field(default_factory=list)
    adversarial_tests: List[GeneratedTest] = field(default_factory=list)
    generation_timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tests(self) -> int:
        """Total number of generated tests."""
        return (
            len(self.golden_tests) +
            len(self.boundary_tests) +
            len(self.fuzz_tests) +
            len(self.adversarial_tests)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "golden_tests": [self._test_to_dict(t) for t in self.golden_tests],
            "boundary_tests": [self._test_to_dict(t) for t in self.boundary_tests],
            "fuzz_tests": [self._test_to_dict(t) for t in self.fuzz_tests],
            "adversarial_tests": [self._test_to_dict(t) for t in self.adversarial_tests],
            "total_tests": self.total_tests,
            "generation_timestamp": self.generation_timestamp.isoformat(),
        }

    def _test_to_dict(self, test: GeneratedTest) -> Dict[str, Any]:
        """Convert test to dict."""
        return {
            "test_id": test.test_id,
            "test_type": test.test_type,
            "name": test.name,
            "description": test.description,
            "input": test.input_data,
            "expect": test.expected_output,
            "expected_behavior": test.expected_behavior,
            "tolerance": test.tolerance,
        }


class BaseTestGenerator(ABC):
    """Base class for test generators."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed."""
        self.seed = seed or 42
        random.seed(self.seed)

    @abstractmethod
    def generate(
        self,
        pack_spec: Dict[str, Any],
        sample_inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[GeneratedTest]:
        """Generate test cases."""
        pass

    def _generate_test_id(self, prefix: str) -> str:
        """Generate unique test ID."""
        timestamp = datetime.utcnow().strftime("%H%M%S%f")
        random_suffix = "".join(random.choices(string.ascii_lowercase, k=4))
        return f"{prefix}_{timestamp}_{random_suffix}"


class GoldenTestGenerator(BaseTestGenerator):
    """
    Generates golden test cases from pack specification.

    Creates deterministic test cases with known expected outputs
    based on formula definitions and example calculations.
    """

    def generate(
        self,
        pack_spec: Dict[str, Any],
        sample_inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[GeneratedTest]:
        """
        Generate golden test cases.

        Args:
            pack_spec: Agent pack specification
            sample_inputs: Optional existing sample inputs

        Returns:
            List of golden test cases
        """
        tests = []

        # Generate from formula if available
        formula_tests = self._generate_from_formula(pack_spec)
        tests.extend(formula_tests)

        # Generate from sample inputs
        if sample_inputs:
            input_tests = self._generate_from_samples(pack_spec, sample_inputs)
            tests.extend(input_tests)

        # Generate from input schema
        schema_tests = self._generate_from_schema(pack_spec)
        tests.extend(schema_tests)

        logger.info(f"Generated {len(tests)} golden tests")
        return tests

    def _generate_from_formula(
        self, pack_spec: Dict[str, Any]
    ) -> List[GeneratedTest]:
        """Generate tests from formula definition."""
        tests = []
        calculation = pack_spec.get("calculation", {})
        formula = calculation.get("formula")

        if not formula:
            return tests

        # Parse formula to identify variables
        formula_str = str(formula)

        # Generate test cases for common emission calculations
        if "emission" in formula_str.lower() or "activity" in formula_str.lower():
            # Standard emission calculation test cases
            test_values = [
                {"activity_data": 100, "emission_factor": 2.5},
                {"activity_data": 1000, "emission_factor": 0.5},
                {"activity_data": 0, "emission_factor": 2.5},
            ]

            for i, values in enumerate(test_values):
                expected = values.get("activity_data", 0) * values.get("emission_factor", 1)
                tests.append(GeneratedTest(
                    test_id=self._generate_test_id("golden_formula"),
                    test_type="golden",
                    name=f"formula_test_{i+1}",
                    description=f"Formula validation with activity={values.get('activity_data')}",
                    input_data=values,
                    expected_output={"emissions_kg_co2e": expected},
                    tolerance={"emissions_kg_co2e": 1e-9},
                ))

        return tests

    def _generate_from_samples(
        self,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> List[GeneratedTest]:
        """Generate tests from sample inputs."""
        tests = []

        for i, sample in enumerate(sample_inputs[:5]):
            tests.append(GeneratedTest(
                test_id=self._generate_test_id("golden_sample"),
                test_type="golden",
                name=f"sample_test_{i+1}",
                description=f"Test with sample input {i+1}",
                input_data=sample.copy(),
                expected_output=None,  # Will be populated by actual run
                expected_behavior="success",
            ))

        return tests

    def _generate_from_schema(
        self, pack_spec: Dict[str, Any]
    ) -> List[GeneratedTest]:
        """Generate tests from input schema."""
        tests = []
        input_schema = pack_spec.get("input", {})

        if not input_schema:
            return tests

        # Generate a typical valid input
        valid_input = {}
        for field_name, field_spec in input_schema.items():
            if isinstance(field_spec, dict):
                field_type = field_spec.get("type", "string")
                default = field_spec.get("default")

                if default is not None:
                    valid_input[field_name] = default
                elif field_type in ["number", "float", "double"]:
                    valid_input[field_name] = 100.0
                elif field_type in ["integer", "int"]:
                    valid_input[field_name] = 100
                elif field_type == "string":
                    valid_input[field_name] = "test_value"
                elif field_type == "boolean":
                    valid_input[field_name] = True
            else:
                valid_input[field_name] = field_spec

        if valid_input:
            tests.append(GeneratedTest(
                test_id=self._generate_test_id("golden_schema"),
                test_type="golden",
                name="schema_valid_input",
                description="Test with schema-derived valid input",
                input_data=valid_input,
                expected_behavior="success",
            ))

        return tests


class BoundaryTestGenerator(BaseTestGenerator):
    """
    Generates boundary condition test cases.

    Tests edge cases including:
    - Minimum/maximum values
    - Zero values
    - Negative values
    - Boundary crossings
    """

    # Common boundary values
    BOUNDARY_VALUES = {
        "numeric": [0, -1, 1, -0.001, 0.001, 1e-15, 1e15, float("inf"), float("-inf")],
        "integer": [0, -1, 1, -2147483648, 2147483647],
        "string": ["", " ", "a", "a" * 1000],
    }

    def generate(
        self,
        pack_spec: Dict[str, Any],
        sample_inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[GeneratedTest]:
        """
        Generate boundary test cases.

        Args:
            pack_spec: Agent pack specification
            sample_inputs: Optional existing sample inputs

        Returns:
            List of boundary test cases
        """
        tests = []
        input_schema = pack_spec.get("input", {})

        # Generate boundary tests for each input field
        for field_name, field_spec in input_schema.items():
            field_tests = self._generate_field_boundary_tests(
                field_name, field_spec, sample_inputs
            )
            tests.extend(field_tests)

        # Generate combination boundary tests
        if sample_inputs:
            combo_tests = self._generate_combination_tests(
                pack_spec, sample_inputs
            )
            tests.extend(combo_tests)

        logger.info(f"Generated {len(tests)} boundary tests")
        return tests

    def _generate_field_boundary_tests(
        self,
        field_name: str,
        field_spec: Any,
        sample_inputs: Optional[List[Dict[str, Any]]],
    ) -> List[GeneratedTest]:
        """Generate boundary tests for a field."""
        tests = []

        # Determine field type
        if isinstance(field_spec, dict):
            field_type = field_spec.get("type", "string")
            min_val = field_spec.get("minimum")
            max_val = field_spec.get("maximum")
        else:
            field_type = "string"
            min_val = max_val = None

        # Get base input
        base_input = sample_inputs[0].copy() if sample_inputs else {}

        # Generate boundary tests based on type
        if field_type in ["number", "float", "double", "integer", "int"]:
            boundary_values = self.BOUNDARY_VALUES["numeric"]

            # Add min/max boundaries if specified
            if min_val is not None:
                boundary_values = [min_val, min_val - 1, min_val + 1] + boundary_values
            if max_val is not None:
                boundary_values = [max_val, max_val - 1, max_val + 1] + boundary_values

            for i, value in enumerate(boundary_values[:5]):  # Limit to 5 tests per field
                test_input = base_input.copy()
                test_input[field_name] = value

                tests.append(GeneratedTest(
                    test_id=self._generate_test_id("boundary"),
                    test_type="boundary",
                    name=f"boundary_{field_name}_{i+1}",
                    description=f"Boundary test: {field_name}={value}",
                    input_data=test_input,
                    expected_behavior="success" if value >= 0 else "validation_error",
                    metadata={"boundary_value": value, "field": field_name},
                ))

        return tests

    def _generate_combination_tests(
        self,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> List[GeneratedTest]:
        """Generate combination boundary tests."""
        tests = []

        if not sample_inputs:
            return tests

        base_input = sample_inputs[0].copy()
        numeric_fields = [
            k for k, v in base_input.items()
            if isinstance(v, (int, float))
        ]

        # Test all zeros
        if numeric_fields:
            zero_input = base_input.copy()
            for field in numeric_fields:
                zero_input[field] = 0

            tests.append(GeneratedTest(
                test_id=self._generate_test_id("boundary_combo"),
                test_type="boundary",
                name="boundary_all_zeros",
                description="All numeric fields set to zero",
                input_data=zero_input,
                expected_behavior="success",
            ))

        return tests


class FuzzTestGenerator(BaseTestGenerator):
    """
    Generates fuzz test cases.

    Creates random and malformed inputs to test robustness:
    - Random values
    - Type confusion
    - Malformed data
    - Unicode strings
    """

    def generate(
        self,
        pack_spec: Dict[str, Any],
        sample_inputs: Optional[List[Dict[str, Any]]] = None,
        num_tests: int = 10,
    ) -> List[GeneratedTest]:
        """
        Generate fuzz test cases.

        Args:
            pack_spec: Agent pack specification
            sample_inputs: Optional existing sample inputs
            num_tests: Number of fuzz tests to generate

        Returns:
            List of fuzz test cases
        """
        tests = []
        input_schema = pack_spec.get("input", {})

        for i in range(num_tests):
            fuzz_input = self._generate_fuzz_input(input_schema, sample_inputs)

            tests.append(GeneratedTest(
                test_id=self._generate_test_id("fuzz"),
                test_type="fuzz",
                name=f"fuzz_test_{i+1}",
                description=f"Fuzz test with random/malformed input",
                input_data=fuzz_input,
                expected_behavior="any",  # May succeed or fail gracefully
                metadata={"fuzz_type": "random"},
            ))

        logger.info(f"Generated {len(tests)} fuzz tests")
        return tests

    def _generate_fuzz_input(
        self,
        input_schema: Dict[str, Any],
        sample_inputs: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Generate a single fuzz input."""
        fuzz_input = {}

        # Use schema if available
        if input_schema:
            for field_name, field_spec in input_schema.items():
                fuzz_input[field_name] = self._generate_fuzz_value(field_spec)
        elif sample_inputs:
            # Mutate sample input
            base = sample_inputs[0].copy()
            for key in base:
                if random.random() < 0.3:  # 30% chance to mutate
                    base[key] = self._generate_fuzz_value(type(base[key]))
            fuzz_input = base
        else:
            # Generate random fields
            for i in range(random.randint(1, 5)):
                field_name = f"field_{i}"
                fuzz_input[field_name] = self._generate_fuzz_value(None)

        return fuzz_input

    def _generate_fuzz_value(self, field_spec: Any) -> Any:
        """Generate a fuzz value for a field."""
        fuzz_options = [
            # Numbers
            random.random() * 1000000,
            random.randint(-1000000, 1000000),
            0,
            -1,
            float("inf"),
            float("nan"),
            1e308,
            # Strings
            "",
            " " * 100,
            "".join(random.choices(string.printable, k=random.randint(1, 100))),
            "\x00" * 10,  # Null bytes
            # Special values
            None,
            [],
            {},
            True,
            False,
        ]

        return random.choice(fuzz_options)


class AdversarialTestGenerator(BaseTestGenerator):
    """
    Generates adversarial test cases.

    Creates inputs designed to expose vulnerabilities:
    - Injection attacks
    - Overflow attempts
    - Format string attacks
    - Path traversal
    """

    # Adversarial patterns
    INJECTION_PATTERNS = [
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "{{7*7}}",
        "${7*7}",
        "../../etc/passwd",
        "%s%s%s%s%s",
        "\\x00",
        "\\n\\r\\n",
        "UNION SELECT * FROM passwords",
    ]

    OVERFLOW_VALUES = [
        "A" * 10000,
        "1" * 1000,
        str(2**64),
        str(-(2**64)),
        "1e1000",
    ]

    def generate(
        self,
        pack_spec: Dict[str, Any],
        sample_inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[GeneratedTest]:
        """
        Generate adversarial test cases.

        Args:
            pack_spec: Agent pack specification
            sample_inputs: Optional existing sample inputs

        Returns:
            List of adversarial test cases
        """
        tests = []
        input_schema = pack_spec.get("input", {})

        # Generate injection tests
        injection_tests = self._generate_injection_tests(
            input_schema, sample_inputs
        )
        tests.extend(injection_tests)

        # Generate overflow tests
        overflow_tests = self._generate_overflow_tests(
            input_schema, sample_inputs
        )
        tests.extend(overflow_tests)

        # Generate type confusion tests
        type_tests = self._generate_type_confusion_tests(
            input_schema, sample_inputs
        )
        tests.extend(type_tests)

        logger.info(f"Generated {len(tests)} adversarial tests")
        return tests

    def _generate_injection_tests(
        self,
        input_schema: Dict[str, Any],
        sample_inputs: Optional[List[Dict[str, Any]]],
    ) -> List[GeneratedTest]:
        """Generate injection attack tests."""
        tests = []
        base_input = sample_inputs[0] if sample_inputs else {}

        # Find string fields
        string_fields = []
        for field_name, field_spec in input_schema.items():
            if isinstance(field_spec, dict):
                if field_spec.get("type") == "string":
                    string_fields.append(field_name)
            elif isinstance(base_input.get(field_name), str):
                string_fields.append(field_name)

        # Generate injection tests
        for pattern in self.INJECTION_PATTERNS[:5]:
            for field_name in string_fields[:2]:
                test_input = base_input.copy()
                test_input[field_name] = pattern

                tests.append(GeneratedTest(
                    test_id=self._generate_test_id("adversarial_injection"),
                    test_type="adversarial",
                    name=f"injection_{field_name}",
                    description=f"Injection test: {pattern[:20]}... in {field_name}",
                    input_data=test_input,
                    expected_behavior="validation_error",
                    metadata={"attack_type": "injection", "pattern": pattern[:50]},
                ))

        return tests

    def _generate_overflow_tests(
        self,
        input_schema: Dict[str, Any],
        sample_inputs: Optional[List[Dict[str, Any]]],
    ) -> List[GeneratedTest]:
        """Generate overflow attack tests."""
        tests = []
        base_input = sample_inputs[0] if sample_inputs else {}

        for overflow_value in self.OVERFLOW_VALUES[:3]:
            test_input = base_input.copy()

            # Apply to first field
            if test_input:
                first_field = list(test_input.keys())[0]
                test_input[first_field] = overflow_value

                tests.append(GeneratedTest(
                    test_id=self._generate_test_id("adversarial_overflow"),
                    test_type="adversarial",
                    name=f"overflow_{first_field}",
                    description=f"Overflow test with value length {len(str(overflow_value))}",
                    input_data=test_input,
                    expected_behavior="validation_error",
                    metadata={"attack_type": "overflow"},
                ))

        return tests

    def _generate_type_confusion_tests(
        self,
        input_schema: Dict[str, Any],
        sample_inputs: Optional[List[Dict[str, Any]]],
    ) -> List[GeneratedTest]:
        """Generate type confusion tests."""
        tests = []
        base_input = sample_inputs[0] if sample_inputs else {}

        type_confusions = [
            ("string_to_list", "value", ["value"]),
            ("string_to_dict", "value", {"key": "value"}),
            ("number_to_string", 100, "100"),
            ("number_to_list", 100, [100]),
            ("bool_to_string", True, "true"),
        ]

        for confusion_name, original, confused in type_confusions[:3]:
            test_input = base_input.copy()

            # Apply to matching fields
            for field_name, field_value in test_input.items():
                if type(field_value) == type(original):
                    test_input[field_name] = confused
                    break

            tests.append(GeneratedTest(
                test_id=self._generate_test_id("adversarial_type"),
                test_type="adversarial",
                name=f"type_confusion_{confusion_name}",
                description=f"Type confusion: {confusion_name}",
                input_data=test_input,
                expected_behavior="validation_error",
                metadata={"attack_type": "type_confusion", "confusion": confusion_name},
            ))

        return tests


class TestGenerator:
    """
    Main test generator that orchestrates all test generation.

    Combines golden, boundary, fuzz, and adversarial test generation.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize test generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed or 42
        self.golden_generator = GoldenTestGenerator(seed)
        self.boundary_generator = BoundaryTestGenerator(seed)
        self.fuzz_generator = FuzzTestGenerator(seed)
        self.adversarial_generator = AdversarialTestGenerator(seed)

        logger.info("TestGenerator initialized")

    def generate_all(
        self,
        pack_spec: Dict[str, Any],
        sample_inputs: Optional[List[Dict[str, Any]]] = None,
        include_fuzz: bool = True,
        include_adversarial: bool = True,
        num_fuzz_tests: int = 10,
    ) -> GeneratedTestSuite:
        """
        Generate all test types.

        Args:
            pack_spec: Agent pack specification
            sample_inputs: Optional sample inputs
            include_fuzz: Include fuzz tests
            include_adversarial: Include adversarial tests
            num_fuzz_tests: Number of fuzz tests

        Returns:
            GeneratedTestSuite with all tests
        """
        suite = GeneratedTestSuite()

        # Generate golden tests
        suite.golden_tests = self.golden_generator.generate(
            pack_spec, sample_inputs
        )

        # Generate boundary tests
        suite.boundary_tests = self.boundary_generator.generate(
            pack_spec, sample_inputs
        )

        # Generate fuzz tests
        if include_fuzz:
            suite.fuzz_tests = self.fuzz_generator.generate(
                pack_spec, sample_inputs, num_fuzz_tests
            )

        # Generate adversarial tests
        if include_adversarial:
            suite.adversarial_tests = self.adversarial_generator.generate(
                pack_spec, sample_inputs
            )

        suite.metadata = {
            "seed": self.seed,
            "include_fuzz": include_fuzz,
            "include_adversarial": include_adversarial,
        }

        logger.info(f"Generated {suite.total_tests} total tests")
        return suite

    def export_to_yaml(
        self,
        suite: GeneratedTestSuite,
        output_path: str,
    ) -> None:
        """
        Export test suite to YAML format.

        Args:
            suite: Generated test suite
            output_path: Output file path
        """
        import yaml

        data = suite.to_dict()

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Exported {suite.total_tests} tests to {output_path}")

    def export_to_json(
        self,
        suite: GeneratedTestSuite,
        output_path: str,
    ) -> None:
        """
        Export test suite to JSON format.

        Args:
            suite: Generated test suite
            output_path: Output file path
        """
        data = suite.to_dict()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {suite.total_tests} tests to {output_path}")
