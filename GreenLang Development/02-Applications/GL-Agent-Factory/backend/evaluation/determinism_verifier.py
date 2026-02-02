"""
Determinism Verifier for GreenLang Agents

Verifies that agents produce bit-perfect deterministic outputs:
- Run agent N times with same input
- Verify all outputs are identical
- Check provenance hashes match
- Report any non-deterministic behavior

This is critical for zero-hallucination compliance.

Example:
    >>> verifier = DeterminismVerifier()
    >>> result = verifier.verify(agent, test_input, num_runs=100)
    >>> assert result.is_deterministic

"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DeterminismResult:
    """Result of determinism verification."""
    is_deterministic: bool
    num_runs: int
    unique_outputs: int
    unique_provenance_hashes: int
    execution_time_ms: float
    non_deterministic_fields: List[str] = field(default_factory=list)
    sample_outputs: List[Any] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def determinism_percentage(self) -> float:
        """Calculate determinism percentage."""
        if self.unique_outputs == 1:
            return 100.0
        return (1 - (self.unique_outputs - 1) / self.num_runs) * 100


class DeterminismVerifier:
    """
    Verify agent determinism through repeated execution.

    This verifier ensures agents produce identical outputs given
    identical inputs - a requirement for zero-hallucination compliance.
    """

    def __init__(self, tolerance: float = 0.0):
        """
        Initialize determinism verifier.

        Args:
            tolerance: Numeric tolerance for float comparisons (default 0.0 for exact match)
        """
        self.tolerance = tolerance
        logger.info(f"DeterminismVerifier initialized (tolerance={tolerance})")

    def verify(
        self,
        agent: Any,
        test_input: Any,
        num_runs: int = 100,
        check_provenance: bool = True,
        verbose: bool = True,
    ) -> DeterminismResult:
        """
        Verify agent determinism.

        Args:
            agent: Agent instance with run() method
            test_input: Input data for agent
            num_runs: Number of runs to execute
            check_provenance: Verify provenance hashes match
            verbose: Print progress

        Returns:
            DeterminismResult with verification details
        """
        if verbose:
            logger.info(f"Starting determinism verification ({num_runs} runs)...")

        start_time = datetime.utcnow()

        # Execute agent multiple times
        outputs = []
        provenance_hashes = []

        for i in range(num_runs):
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{num_runs} runs completed")

            try:
                result = agent.run(test_input)
                outputs.append(result)

                # Extract provenance hash if available
                if check_provenance and hasattr(result, 'provenance_hash'):
                    provenance_hashes.append(result.provenance_hash)

            except Exception as e:
                logger.error(f"Run {i + 1} failed: {str(e)}")
                raise

        execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Analyze results
        unique_outputs = self._count_unique_outputs(outputs)
        unique_hashes = len(set(provenance_hashes)) if provenance_hashes else 0

        is_deterministic = (unique_outputs == 1)
        if check_provenance and provenance_hashes:
            is_deterministic = is_deterministic and (unique_hashes == 1)

        # Identify non-deterministic fields
        non_deterministic_fields = []
        if not is_deterministic:
            non_deterministic_fields = self._find_non_deterministic_fields(outputs)

        result = DeterminismResult(
            is_deterministic=is_deterministic,
            num_runs=num_runs,
            unique_outputs=unique_outputs,
            unique_provenance_hashes=unique_hashes,
            execution_time_ms=execution_time_ms,
            non_deterministic_fields=non_deterministic_fields,
            sample_outputs=outputs[:5],  # Store first 5 outputs for inspection
        )

        if verbose:
            self._print_summary(result)

        return result

    def verify_provenance_only(
        self,
        agent: Any,
        test_input: Any,
        num_runs: int = 100,
        verbose: bool = True,
    ) -> bool:
        """
        Verify only provenance hash determinism.

        Faster check that focuses only on provenance hashes.

        Args:
            agent: Agent instance
            test_input: Input data
            num_runs: Number of runs
            verbose: Print output

        Returns:
            True if all provenance hashes match
        """
        if verbose:
            logger.info(f"Verifying provenance determinism ({num_runs} runs)...")

        hashes = set()

        for i in range(num_runs):
            result = agent.run(test_input)
            if hasattr(result, 'provenance_hash'):
                hashes.add(result.provenance_hash)

        is_deterministic = len(hashes) == 1

        if verbose:
            if is_deterministic:
                logger.info(f"SUCCESS: All {num_runs} runs produced identical provenance hash")
            else:
                logger.error(f"FAIL: Found {len(hashes)} different provenance hashes")

        return is_deterministic

    def _count_unique_outputs(self, outputs: List[Any]) -> int:
        """
        Count unique outputs.

        Args:
            outputs: List of agent outputs

        Returns:
            Number of unique outputs
        """
        # Convert outputs to hashable representations
        output_hashes = set()

        for output in outputs:
            output_hash = self._hash_output(output)
            output_hashes.add(output_hash)

        return len(output_hashes)

    def _hash_output(self, output: Any) -> str:
        """
        Create hash of output for comparison.

        Args:
            output: Agent output

        Returns:
            SHA-256 hash of output
        """
        # Convert to dict if Pydantic model
        if hasattr(output, 'dict'):
            output_dict = output.dict()
        elif hasattr(output, '__dict__'):
            output_dict = output.__dict__
        else:
            output_dict = {'value': output}

        # Remove known variable fields
        output_dict = {
            k: v for k, v in output_dict.items()
            if k not in ['timestamp', 'calculated_at', 'execution_time_ms']
        }

        # Create deterministic JSON representation
        json_str = json.dumps(output_dict, sort_keys=True, default=str)

        # Hash it
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _find_non_deterministic_fields(self, outputs: List[Any]) -> List[str]:
        """
        Identify which fields are non-deterministic.

        Args:
            outputs: List of agent outputs

        Returns:
            List of field names that vary across runs
        """
        if not outputs:
            return []

        non_deterministic = []

        # Convert first output to dict for field inspection
        if hasattr(outputs[0], 'dict'):
            first_dict = outputs[0].dict()
        elif hasattr(outputs[0], '__dict__'):
            first_dict = outputs[0].__dict__
        else:
            return []

        # Check each field
        for field_name in first_dict.keys():
            # Skip known variable fields
            if field_name in ['timestamp', 'calculated_at', 'execution_time_ms']:
                continue

            # Collect values for this field across all runs
            values = []
            for output in outputs:
                if hasattr(output, 'dict'):
                    output_dict = output.dict()
                elif hasattr(output, '__dict__'):
                    output_dict = output.__dict__
                else:
                    continue

                if field_name in output_dict:
                    values.append(output_dict[field_name])

            # Check if values vary
            if len(set(map(str, values))) > 1:
                non_deterministic.append(field_name)

        return non_deterministic

    def _print_summary(self, result: DeterminismResult) -> None:
        """Print verification summary."""
        print("\n" + "=" * 80)
        print("DETERMINISM VERIFICATION RESULTS")
        print("=" * 80)
        print(f"Total runs:              {result.num_runs}")
        print(f"Unique outputs:          {result.unique_outputs}")
        print(f"Unique prov hashes:      {result.unique_provenance_hashes}")
        print(f"Determinism:             {result.determinism_percentage:.2f}%")
        print(f"Is deterministic:        {'YES' if result.is_deterministic else 'NO'}")
        print(f"Execution time:          {result.execution_time_ms:.2f}ms")

        if result.non_deterministic_fields:
            print(f"\nNon-deterministic fields:")
            for field in result.non_deterministic_fields:
                print(f"  - {field}")

        print("=" * 80)

        if result.is_deterministic:
            print("SUCCESS: Agent is 100% deterministic")
        else:
            print("FAIL: Agent produces non-deterministic outputs")

        print()
