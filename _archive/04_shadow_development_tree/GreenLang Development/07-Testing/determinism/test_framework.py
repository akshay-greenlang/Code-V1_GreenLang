# -*- coding: utf-8 -*-
"""Core Determinism Testing Framework.

This module provides the DeterminismTester class for hash-based reproducibility
verification of agent outputs. It supports both sync and async agents, handles
cross-platform differences, and provides detailed comparison utilities.

Key Features:
- Hash-based output verification (SHA256)
- Byte-level and field-level comparison
- Async/sync agent support
- Cross-platform normalization
- Detailed diff reporting

Author: GreenLang Framework Team
Phase: Phase 3 - Production Hardening
Date: November 2024
"""

import asyncio
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import inspect


@dataclass
class DeterminismResult:
    """Result of a determinism test.

    Attributes:
        is_deterministic: Whether outputs were identical
        run_count: Number of runs executed
        hashes: List of output hashes from each run
        outputs: List of actual outputs (optional, for debugging)
        differences: List of differences found between runs
        normalized_output: Normalized version of output (if available)
    """
    is_deterministic: bool
    run_count: int
    hashes: List[str]
    outputs: Optional[List[Any]] = None
    differences: List[Dict[str, Any]] = field(default_factory=list)
    normalized_output: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """Human-readable representation."""
        status = "PASS" if self.is_deterministic else "FAIL"
        return (
            f"DeterminismResult({status}):\n"
            f"  Runs: {self.run_count}\n"
            f"  Unique hashes: {len(set(self.hashes))}\n"
            f"  Differences: {len(self.differences)}"
        )


class DeterminismTester:
    """Framework for testing agent determinism.

    This class provides utilities for running agents multiple times with identical
    inputs and verifying that outputs are deterministic (hash-identical).

    Usage:
        >>> tester = DeterminismTester()
        >>> result = await tester.test_agent_async(agent, payload, runs=5)
        >>> assert result.is_deterministic

    Features:
        - Supports both sync and async agents
        - Hash-based comparison (SHA256)
        - Field-level difference detection
        - Cross-platform normalization
        - Configurable comparison modes
    """

    def __init__(
        self,
        normalize_platform: bool = True,
        normalize_timestamps: bool = True,
        normalize_floats: bool = True,
        float_precision: int = 6,
        sort_keys: bool = True,
    ):
        """Initialize DeterminismTester.

        Args:
            normalize_platform: Normalize platform-specific values
            normalize_timestamps: Remove/normalize timestamps
            normalize_floats: Round floats to specified precision
            float_precision: Decimal places for float normalization
            sort_keys: Sort JSON keys for consistent serialization
        """
        self.normalize_platform = normalize_platform
        self.normalize_timestamps = normalize_timestamps
        self.normalize_floats = normalize_floats
        self.float_precision = float_precision
        self.sort_keys = sort_keys

    def compute_hash(self, data: Any) -> str:
        """Compute SHA256 hash of data.

        Args:
            data: Data to hash (will be JSON-serialized)

        Returns:
            Hexadecimal hash string
        """
        # Normalize data before hashing
        normalized = self._normalize_data(data)

        # Serialize to JSON with sorted keys for consistency
        json_str = json.dumps(normalized, sort_keys=self.sort_keys, ensure_ascii=True)

        # Compute hash
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _normalize_data(self, data: Any) -> Any:
        """Normalize data for deterministic comparison.

        Args:
            data: Data to normalize

        Returns:
            Normalized data
        """
        if data is None:
            return None

        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # Skip certain non-deterministic fields
                if self.normalize_timestamps and key in (
                    'timestamp', 'created_at', 'updated_at', 'run_id',
                    'execution_time', 'duration_ms', 'wall_time',
                ):
                    continue

                # Skip platform-specific fields
                if self.normalize_platform and key in (
                    'platform', 'hostname', 'username', 'pid',
                    'thread_id', 'process_id',
                ):
                    continue

                result[key] = self._normalize_data(value)
            return result

        if isinstance(data, (list, tuple)):
            return [self._normalize_data(item) for item in data]

        if isinstance(data, float) and self.normalize_floats:
            # Round floats to specified precision
            return round(data, self.float_precision)

        if isinstance(data, str):
            # Optionally normalize paths
            if self.normalize_platform:
                # Remove platform-specific path separators
                return data.replace('\\', '/')
            return data

        # Return as-is for other types (int, bool, etc.)
        return data

    def compare_outputs(
        self,
        output1: Any,
        output2: Any,
        path: str = "root"
    ) -> List[Dict[str, Any]]:
        """Compare two outputs and return differences.

        Args:
            output1: First output
            output2: Second output
            path: Current path in nested structure

        Returns:
            List of difference dictionaries
        """
        differences = []

        # Normalize both outputs
        norm1 = self._normalize_data(output1)
        norm2 = self._normalize_data(output2)

        # Type mismatch
        if type(norm1) != type(norm2):
            differences.append({
                "path": path,
                "type": "type_mismatch",
                "value1_type": str(type(norm1).__name__),
                "value2_type": str(type(norm2).__name__),
                "value1": str(norm1)[:100],
                "value2": str(norm2)[:100],
            })
            return differences

        # Dict comparison
        if isinstance(norm1, dict):
            all_keys = set(norm1.keys()) | set(norm2.keys())
            for key in all_keys:
                if key not in norm1:
                    differences.append({
                        "path": f"{path}.{key}",
                        "type": "missing_in_first",
                        "value": norm2[key],
                    })
                elif key not in norm2:
                    differences.append({
                        "path": f"{path}.{key}",
                        "type": "missing_in_second",
                        "value": norm1[key],
                    })
                else:
                    # Recursive comparison
                    differences.extend(
                        self.compare_outputs(norm1[key], norm2[key], f"{path}.{key}")
                    )

        # List comparison
        elif isinstance(norm1, list):
            if len(norm1) != len(norm2):
                differences.append({
                    "path": path,
                    "type": "length_mismatch",
                    "length1": len(norm1),
                    "length2": len(norm2),
                })
            else:
                for i, (item1, item2) in enumerate(zip(norm1, norm2)):
                    differences.extend(
                        self.compare_outputs(item1, item2, f"{path}[{i}]")
                    )

        # Value comparison
        elif norm1 != norm2:
            differences.append({
                "path": path,
                "type": "value_mismatch",
                "value1": norm1,
                "value2": norm2,
            })

        return differences

    async def test_agent_async(
        self,
        agent: Any,
        payload: Dict[str, Any],
        runs: int = 5,
        store_outputs: bool = False,
    ) -> DeterminismResult:
        """Test async agent for determinism.

        Args:
            agent: Async agent instance with run() method
            payload: Input payload for agent
            runs: Number of runs to execute
            store_outputs: Whether to store actual outputs (uses memory)

        Returns:
            DeterminismResult with comparison details
        """
        hashes = []
        outputs = [] if store_outputs else None

        for i in range(runs):
            # Run agent
            result = await agent.run(payload)

            # Extract data (handle both dict and object responses)
            if isinstance(result, dict):
                output = result
            elif hasattr(result, 'data'):
                output = result.data
            else:
                output = str(result)

            # Compute hash
            hash_val = self.compute_hash(output)
            hashes.append(hash_val)

            if store_outputs:
                outputs.append(output)

        # Check if all hashes are identical
        unique_hashes = set(hashes)
        is_deterministic = len(unique_hashes) == 1

        # Find differences if not deterministic
        differences = []
        if not is_deterministic and store_outputs:
            # Compare first output with all others
            for i in range(1, len(outputs)):
                diffs = self.compare_outputs(outputs[0], outputs[i], f"run_0_vs_run_{i}")
                differences.extend(diffs)

        return DeterminismResult(
            is_deterministic=is_deterministic,
            run_count=runs,
            hashes=hashes,
            outputs=outputs,
            differences=differences,
            normalized_output=self._normalize_data(outputs[0]) if outputs else None,
        )

    def test_agent_sync(
        self,
        agent: Any,
        payload: Dict[str, Any],
        runs: int = 5,
        store_outputs: bool = False,
    ) -> DeterminismResult:
        """Test sync agent for determinism.

        Args:
            agent: Sync agent instance with run() method
            payload: Input payload for agent
            runs: Number of runs to execute
            store_outputs: Whether to store actual outputs (uses memory)

        Returns:
            DeterminismResult with comparison details
        """
        hashes = []
        outputs = [] if store_outputs else None

        for i in range(runs):
            # Run agent
            result = agent.run(payload)

            # Extract data (handle both dict and object responses)
            if isinstance(result, dict):
                output = result
            elif hasattr(result, 'data'):
                output = result.data
            else:
                output = str(result)

            # Compute hash
            hash_val = self.compute_hash(output)
            hashes.append(hash_val)

            if store_outputs:
                outputs.append(output)

        # Check if all hashes are identical
        unique_hashes = set(hashes)
        is_deterministic = len(unique_hashes) == 1

        # Find differences if not deterministic
        differences = []
        if not is_deterministic and store_outputs:
            # Compare first output with all others
            for i in range(1, len(outputs)):
                diffs = self.compare_outputs(outputs[0], outputs[i], f"run_0_vs_run_{i}")
                differences.extend(diffs)

        return DeterminismResult(
            is_deterministic=is_deterministic,
            run_count=runs,
            hashes=hashes,
            outputs=outputs,
            differences=differences,
            normalized_output=self._normalize_data(outputs[0]) if outputs else None,
        )

    def test_function(
        self,
        func: Callable,
        *args,
        runs: int = 5,
        store_outputs: bool = False,
        **kwargs
    ) -> DeterminismResult:
        """Test any function for determinism.

        Args:
            func: Function to test (can be sync or async)
            *args: Positional arguments for function
            runs: Number of runs to execute
            store_outputs: Whether to store actual outputs
            **kwargs: Keyword arguments for function

        Returns:
            DeterminismResult with comparison details
        """
        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            # Run async function
            async def _async_test():
                hashes = []
                outputs = [] if store_outputs else None

                for i in range(runs):
                    result = await func(*args, **kwargs)
                    hash_val = self.compute_hash(result)
                    hashes.append(hash_val)

                    if store_outputs:
                        outputs.append(result)

                return hashes, outputs

            # Execute in event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create new task if loop already running
                    import nest_asyncio
                    nest_asyncio.apply()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            hashes, outputs = loop.run_until_complete(_async_test())
        else:
            # Run sync function
            hashes = []
            outputs = [] if store_outputs else None

            for i in range(runs):
                result = func(*args, **kwargs)
                hash_val = self.compute_hash(result)
                hashes.append(hash_val)

                if store_outputs:
                    outputs.append(result)

        # Check determinism
        unique_hashes = set(hashes)
        is_deterministic = len(unique_hashes) == 1

        # Find differences
        differences = []
        if not is_deterministic and store_outputs:
            for i in range(1, len(outputs)):
                diffs = self.compare_outputs(outputs[0], outputs[i], f"run_0_vs_run_{i}")
                differences.extend(diffs)

        return DeterminismResult(
            is_deterministic=is_deterministic,
            run_count=runs,
            hashes=hashes,
            outputs=outputs,
            differences=differences,
            normalized_output=self._normalize_data(outputs[0]) if outputs else None,
        )


def assert_deterministic(
    result: DeterminismResult,
    message: Optional[str] = None
) -> None:
    """Assert that a determinism result is passing.

    Args:
        result: DeterminismResult to check
        message: Optional custom error message

    Raises:
        AssertionError: If result is not deterministic
    """
    if not result.is_deterministic:
        error_msg = message or (
            f"Agent outputs are not deterministic!\n"
            f"{result}\n"
            f"Unique hashes: {len(set(result.hashes))}\n"
            f"Hashes: {result.hashes[:3]}...\n"
        )

        if result.differences:
            error_msg += f"\nFirst 5 differences:\n"
            for diff in result.differences[:5]:
                error_msg += f"  - {diff['path']}: {diff['type']}\n"

        raise AssertionError(error_msg)


# Convenience functions for common patterns
def quick_test_agent_determinism(
    agent: Any,
    payload: Dict[str, Any],
    runs: int = 5,
    **tester_kwargs
) -> DeterminismResult:
    """Quick test for agent determinism (auto-detects sync/async).

    Args:
        agent: Agent instance
        payload: Input payload
        runs: Number of runs
        **tester_kwargs: Additional DeterminismTester arguments

    Returns:
        DeterminismResult
    """
    tester = DeterminismTester(**tester_kwargs)

    # Check if agent.run is async
    if hasattr(agent, 'run'):
        run_method = agent.run
        if inspect.iscoroutinefunction(run_method):
            # Async agent
            return asyncio.run(tester.test_agent_async(agent, payload, runs=runs))
        else:
            # Sync agent
            return tester.test_agent_sync(agent, payload, runs=runs)
    else:
        raise ValueError(f"Agent {agent} does not have a run() method")
