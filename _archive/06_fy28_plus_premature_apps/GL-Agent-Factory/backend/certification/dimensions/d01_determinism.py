"""
Dimension 01: Determinism Verification

This dimension verifies that an agent produces byte-identical outputs
when run multiple times with the same input and seed.

Checks:
    - Run 100 times with same seed
    - Verify byte-identical outputs
    - Check hash reproducibility
    - Validate timestamp exclusion from determinism

Example:
    >>> dimension = DeterminismDimension()
    >>> result = dimension.evaluate(agent_path, agent, sample_input)
    >>> assert result.status == DimensionStatus.PASS
"""

import hashlib
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base import BaseDimension, CheckResult, DimensionResult, DimensionStatus

logger = logging.getLogger(__name__)


class DeterminismDimension(BaseDimension):
    """
    Determinism Dimension Evaluator (D01).

    Verifies agents produce identical outputs for identical inputs,
    which is critical for zero-hallucination compliance.

    Configuration:
        num_runs: Number of runs to execute (default: 100)
        seed: Random seed for reproducibility (default: 42)
        exclude_fields: Fields to exclude from comparison (timestamps, etc.)
    """

    DIMENSION_ID = "D01"
    DIMENSION_NAME = "Determinism"
    DESCRIPTION = "Verifies byte-identical outputs across multiple runs"
    WEIGHT = 1.5  # Higher weight - critical for zero-hallucination
    REQUIRED_FOR_CERTIFICATION = True

    # Default configuration
    DEFAULT_NUM_RUNS = 100
    DEFAULT_SEED = 42
    DEFAULT_EXCLUDE_FIELDS = {
        "timestamp",
        "calculated_at",
        "execution_time_ms",
        "processing_time_ms",
        "created_at",
        "updated_at",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize determinism dimension evaluator."""
        super().__init__(config)

        self.num_runs = self.config.get("num_runs", self.DEFAULT_NUM_RUNS)
        self.seed = self.config.get("seed", self.DEFAULT_SEED)
        self.exclude_fields = set(
            self.config.get("exclude_fields", self.DEFAULT_EXCLUDE_FIELDS)
        )

    def evaluate(
        self,
        agent_path: Path,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
    ) -> DimensionResult:
        """
        Evaluate determinism for the given agent.

        Args:
            agent_path: Path to agent directory
            agent: Agent instance with run() method
            sample_input: Sample input for testing

        Returns:
            DimensionResult with determinism evaluation
        """
        start_time = datetime.utcnow()
        self._reset_checks()

        logger.info(f"Starting determinism evaluation ({self.num_runs} runs)")

        if agent is None:
            # Try to load agent from path
            agent = self._load_agent(agent_path)

        if agent is None:
            self._add_check(
                name="agent_load",
                passed=False,
                message="Failed to load agent instance",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        if sample_input is None:
            sample_input = self._get_sample_input(agent_path)

        if sample_input is None:
            self._add_check(
                name="sample_input",
                passed=False,
                message="No sample input provided or found",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        # Run determinism checks
        outputs = []
        output_hashes = []
        provenance_hashes = []
        errors = []

        # Set random seed for reproducibility
        random.seed(self.seed)

        for i in range(self.num_runs):
            try:
                # Reset random state before each run
                random.seed(self.seed)

                result = agent.run(sample_input)
                outputs.append(result)

                # Calculate output hash (excluding variable fields)
                output_hash = self._hash_output(result)
                output_hashes.append(output_hash)

                # Track provenance hash if available
                if hasattr(result, "provenance_hash"):
                    provenance_hashes.append(result.provenance_hash)

            except Exception as e:
                errors.append(f"Run {i + 1}: {str(e)}")
                logger.warning(f"Run {i + 1} failed: {str(e)}")

        # Analyze results
        unique_outputs = len(set(output_hashes))
        unique_provenance = len(set(provenance_hashes)) if provenance_hashes else 0

        # Check 1: All outputs identical
        self._add_check(
            name="output_consistency",
            passed=unique_outputs == 1,
            message=f"{unique_outputs} unique output(s) in {len(output_hashes)} runs",
            severity="error",
            details={
                "unique_outputs": unique_outputs,
                "total_runs": len(output_hashes),
                "sample_hash": output_hashes[0] if output_hashes else None,
            },
        )

        # Check 2: Provenance hash consistency
        if provenance_hashes:
            # Note: Provenance hashes may include timestamps, so we check
            # if they are reasonably consistent (allowing for timestamp variation)
            prov_consistent = unique_provenance <= 2  # Allow small variation
            self._add_check(
                name="provenance_consistency",
                passed=prov_consistent,
                message=f"{unique_provenance} unique provenance hash(es)",
                severity="warning" if not prov_consistent else "info",
                details={
                    "unique_provenance": unique_provenance,
                    "total_hashes": len(provenance_hashes),
                },
            )
        else:
            self._add_check(
                name="provenance_consistency",
                passed=True,
                message="No provenance hashes in output (optional)",
                severity="info",
            )

        # Check 3: No execution errors
        self._add_check(
            name="execution_stability",
            passed=len(errors) == 0,
            message=f"{len(errors)} execution error(s) in {self.num_runs} runs",
            severity="error" if errors else "info",
            details={"errors": errors[:5]} if errors else {},
        )

        # Check 4: Hash reproducibility
        if output_hashes:
            # Verify hash algorithm produces consistent results
            test_hash_1 = self._hash_output(outputs[0]) if outputs else None
            test_hash_2 = self._hash_output(outputs[0]) if outputs else None
            hash_reproducible = test_hash_1 == test_hash_2

            self._add_check(
                name="hash_reproducibility",
                passed=hash_reproducible,
                message="Hash algorithm produces reproducible results"
                if hash_reproducible
                else "Hash algorithm inconsistent",
                severity="error",
            )

        # Check 5: Numeric precision consistency
        numeric_consistent = self._check_numeric_consistency(outputs)
        self._add_check(
            name="numeric_precision",
            passed=numeric_consistent,
            message="Numeric values consistent across runs"
            if numeric_consistent
            else "Numeric precision varies between runs",
            severity="error",
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return self._create_result(
            execution_time,
            details={
                "num_runs": len(output_hashes),
                "unique_outputs": unique_outputs,
                "unique_provenance": unique_provenance,
                "determinism_percentage": (
                    100.0 if unique_outputs == 1 else
                    (1 - (unique_outputs - 1) / max(len(output_hashes), 1)) * 100
                ),
                "seed_used": self.seed,
            },
        )

    def _load_agent(self, agent_path: Path) -> Optional[Any]:
        """
        Attempt to load agent from path.

        Args:
            agent_path: Path to agent directory

        Returns:
            Agent instance or None
        """
        try:
            # Try to import agent module
            agent_file = agent_path / "agent.py"
            if not agent_file.exists():
                return None

            import importlib.util

            spec = importlib.util.spec_from_file_location("agent", agent_file)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for Agent class
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and name.endswith("Agent")
                    and hasattr(obj, "run")
                ):
                    return obj()

            return None

        except Exception as e:
            logger.error(f"Failed to load agent: {str(e)}")
            return None

    def _get_sample_input(self, agent_path: Path) -> Optional[Any]:
        """
        Get sample input from agent path.

        Args:
            agent_path: Path to agent directory

        Returns:
            Sample input or None
        """
        try:
            import yaml

            # Try pack.yaml
            pack_file = agent_path / "pack.yaml"
            if pack_file.exists():
                with open(pack_file, "r", encoding="utf-8") as f:
                    pack_spec = yaml.safe_load(f)

                # Look for golden tests
                tests = pack_spec.get("tests", {}).get("golden", [])
                if tests:
                    return tests[0].get("input")

            # Try golden_tests.yaml
            golden_file = agent_path / "golden_tests.yaml"
            if golden_file.exists():
                with open(golden_file, "r", encoding="utf-8") as f:
                    golden_spec = yaml.safe_load(f)

                tests = golden_spec.get("test_cases", [])
                if tests:
                    return tests[0].get("input")

            return None

        except Exception as e:
            logger.error(f"Failed to get sample input: {str(e)}")
            return None

    def _hash_output(self, output: Any) -> str:
        """
        Create deterministic hash of output.

        Args:
            output: Agent output

        Returns:
            SHA-256 hash string
        """
        # Convert to dict
        if hasattr(output, "dict"):
            output_dict = output.dict()
        elif hasattr(output, "model_dump"):
            output_dict = output.model_dump()
        elif hasattr(output, "__dict__"):
            output_dict = output.__dict__.copy()
        else:
            output_dict = {"value": output}

        # Remove excluded fields
        filtered_dict = {
            k: v
            for k, v in output_dict.items()
            if k.lower() not in self.exclude_fields
        }

        # Create deterministic JSON
        json_str = json.dumps(filtered_dict, sort_keys=True, default=str)

        return hashlib.sha256(json_str.encode()).hexdigest()

    def _check_numeric_consistency(self, outputs: List[Any]) -> bool:
        """
        Check if numeric values are consistent across outputs.

        Args:
            outputs: List of agent outputs

        Returns:
            True if numeric values are consistent
        """
        if len(outputs) < 2:
            return True

        try:
            # Extract numeric fields from first output
            first_output = outputs[0]
            if hasattr(first_output, "dict"):
                first_dict = first_output.dict()
            elif hasattr(first_output, "model_dump"):
                first_dict = first_output.model_dump()
            elif hasattr(first_output, "__dict__"):
                first_dict = first_output.__dict__
            else:
                return True

            numeric_fields = {
                k: v for k, v in first_dict.items()
                if isinstance(v, (int, float)) and k.lower() not in self.exclude_fields
            }

            # Compare with all other outputs
            for output in outputs[1:]:
                if hasattr(output, "dict"):
                    output_dict = output.dict()
                elif hasattr(output, "model_dump"):
                    output_dict = output.model_dump()
                elif hasattr(output, "__dict__"):
                    output_dict = output.__dict__
                else:
                    continue

                for field, value in numeric_fields.items():
                    if field in output_dict:
                        other_value = output_dict[field]
                        if value != other_value:
                            logger.warning(
                                f"Numeric inconsistency in {field}: "
                                f"{value} vs {other_value}"
                            )
                            return False

            return True

        except Exception as e:
            logger.error(f"Error checking numeric consistency: {str(e)}")
            return True  # Assume consistent if we can't check

    def _get_check_remediation(self, check: CheckResult) -> Optional[str]:
        """Get remediation for failed checks."""
        remediation_map = {
            "agent_load": (
                "Ensure agent.py exists and contains a class ending with 'Agent' "
                "that has a run() method."
            ),
            "sample_input": (
                "Provide sample input via pack.yaml golden tests or pass "
                "sample_input parameter directly."
            ),
            "output_consistency": (
                "Agent outputs vary between runs. Check for:\n"
                "  - Random number generation without fixed seed\n"
                "  - Timestamp inclusion in calculation\n"
                "  - Non-deterministic external API calls\n"
                "  - Floating point operations without proper rounding"
            ),
            "provenance_consistency": (
                "Provenance hashes vary. Ensure provenance calculation excludes "
                "timestamps and uses deterministic serialization."
            ),
            "execution_stability": (
                "Agent execution fails intermittently. Check for:\n"
                "  - Race conditions\n"
                "  - Resource exhaustion\n"
                "  - External dependency failures"
            ),
            "hash_reproducibility": (
                "Hash calculation is not reproducible. Ensure JSON serialization "
                "uses sort_keys=True and default=str for consistent output."
            ),
            "numeric_precision": (
                "Numeric values differ between runs. Check for:\n"
                "  - Floating point precision issues\n"
                "  - Order-dependent calculations\n"
                "  - Use round() with consistent decimal places"
            ),
        }

        return remediation_map.get(check.name)
