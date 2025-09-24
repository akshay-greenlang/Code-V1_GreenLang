"""
Golden Tests for Deterministic Execution
=========================================

Creates and validates golden tests to ensure reproducible pipeline execution.
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

logger = logging.getLogger(__name__)


def create_golden_test(
    pipeline: Dict[str, Any],
    inputs: Dict[str, Any],
    expected_outputs: Dict[str, Any],
    tolerance: float = 1e-6,
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Create a golden test for deterministic execution

    Args:
        pipeline: Pipeline specification
        inputs: Input parameters
        expected_outputs: Expected outputs
        tolerance: Numerical tolerance for floating point comparisons
        metadata: Optional metadata

    Returns:
        Golden test specification
    """
    # Calculate stable hash for test ID
    test_data = json.dumps(
        {"pipeline": pipeline, "inputs": inputs}, sort_keys=True, separators=(",", ":")
    )
    test_id = hashlib.sha256(test_data.encode()).hexdigest()[:16]

    return {
        "version": "1.0",
        "test_id": test_id,
        "name": pipeline.get("name", "unnamed") + "_golden",
        "pipeline": pipeline,
        "inputs": inputs,
        "expected_outputs": expected_outputs,
        "tolerance": tolerance,
        "metadata": metadata or {},
        "created_at": datetime.utcnow().isoformat(),
    }


def run_golden_test(
    test_spec: Dict[str, Any], executor: Any
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run a golden test

    Args:
        test_spec: Golden test specification
        executor: Executor instance

    Returns:
        Tuple of (passed, comparison_result)
    """

    # Ensure deterministic mode
    if not executor.deterministic:
        logger.warning("Executor not in deterministic mode, enabling it")
        executor.deterministic = True

    # Execute pipeline
    result = executor.execute(test_spec["pipeline"], test_spec["inputs"])

    if not result.success:
        return False, {
            "error": result.error,
            "test_id": test_spec.get("test_id"),
            "passed": False,
        }

    # Compare outputs
    comparison = compare_outputs(
        result.data, test_spec["expected_outputs"], test_spec.get("tolerance", 1e-6)
    )

    comparison["test_id"] = test_spec.get("test_id")
    comparison["passed"] = comparison["match"]

    return comparison["match"], comparison


def compare_outputs(
    actual: Any, expected: Any, tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Compare actual outputs with expected outputs

    Args:
        actual: Actual outputs
        expected: Expected outputs
        tolerance: Numerical tolerance

    Returns:
        Comparison result
    """

    def compare_values(a, b, path=""):
        """Recursively compare values"""
        if type(a) != type(b):
            return False, f"Type mismatch at {path}: {type(a)} vs {type(b)}"

        if isinstance(a, (int, float)):
            if abs(a - b) >= tolerance:
                return False, f"Value mismatch at {path}: {a} vs {b} (diff: {abs(a-b)})"
            return True, None

        elif HAS_NUMPY and isinstance(a, np.ndarray):
            if not np.allclose(a, b, atol=tolerance):
                max_diff = np.max(np.abs(a - b))
                return False, f"Array mismatch at {path}: max diff {max_diff}"
            return True, None

        elif isinstance(a, dict):
            if set(a.keys()) != set(b.keys()):
                return (
                    False,
                    f"Keys mismatch at {path}: {set(a.keys())} vs {set(b.keys())}",
                )

            for key in a.keys():
                match, error = compare_values(a[key], b[key], f"{path}.{key}")
                if not match:
                    return False, error
            return True, None

        elif isinstance(a, list):
            if len(a) != len(b):
                return False, f"Length mismatch at {path}: {len(a)} vs {len(b)}"

            for i in range(len(a)):
                match, error = compare_values(a[i], b[i], f"{path}[{i}]")
                if not match:
                    return False, error
            return True, None

        else:
            if a != b:
                return False, f"Value mismatch at {path}: {a} vs {b}"
            return True, None

    match, error = compare_values(actual, expected)

    return {
        "match": match,
        "error": error,
        "actual": actual,
        "expected": expected,
        "tolerance": tolerance,
    }


def save_golden_test(test_spec: Dict[str, Any], path: Path):
    """
    Save golden test to file

    Args:
        test_spec: Golden test specification
        path: Path to save file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(test_spec, f, indent=2, sort_keys=True)

    logger.info(f"Golden test saved to {path}")


def load_golden_test(path: Path) -> Dict[str, Any]:
    """
    Load golden test from file

    Args:
        path: Path to test file

    Returns:
        Golden test specification
    """
    with open(path) as f:
        return json.load(f)


def run_golden_test_suite(
    test_dir: Path, executor: Any, pattern: str = "*.golden.json"
) -> Dict[str, Any]:
    """
    Run a suite of golden tests

    Args:
        test_dir: Directory containing golden tests
        executor: Executor instance
        pattern: File pattern for test files

    Returns:
        Test suite results
    """
    test_dir = Path(test_dir)
    results = {"total": 0, "passed": 0, "failed": 0, "tests": []}

    for test_file in test_dir.glob(pattern):
        try:
            test_spec = load_golden_test(test_file)
            passed, comparison = run_golden_test(test_spec, executor)

            results["total"] += 1
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1

            results["tests"].append(
                {
                    "file": str(test_file),
                    "name": test_spec.get("name"),
                    "passed": passed,
                    "details": comparison,
                }
            )

        except Exception as e:
            logger.error(f"Failed to run test {test_file}: {e}")
            results["total"] += 1
            results["failed"] += 1
            results["tests"].append(
                {"file": str(test_file), "passed": False, "error": str(e)}
            )

    return results


def generate_golden_output(
    pipeline: Dict[str, Any], inputs: Dict[str, Any], executor: Any, runs: int = 3
) -> Tuple[Dict[str, Any], bool]:
    """
    Generate golden output by running pipeline multiple times
    and checking for determinism

    Args:
        pipeline: Pipeline specification
        inputs: Input parameters
        executor: Executor instance (must be deterministic)
        runs: Number of runs to verify determinism

    Returns:
        Tuple of (golden_output, is_deterministic)
    """
    if not executor.deterministic:
        raise ValueError("Executor must be in deterministic mode")

    outputs = []

    for i in range(runs):
        result = executor.execute(pipeline, inputs)

        if not result.success:
            raise RuntimeError(
                f"Pipeline execution failed on run {i+1}: {result.error}"
            )

        outputs.append(result.data)

    # Check if all outputs are identical
    is_deterministic = True
    for i in range(1, len(outputs)):
        comparison = compare_outputs(outputs[0], outputs[i])
        if not comparison["match"]:
            is_deterministic = False
            logger.warning(f"Non-deterministic output detected: {comparison['error']}")
            break

    return outputs[0], is_deterministic


def create_golden_test_from_run(
    run_file: Path, output_file: Optional[Path] = None, tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Create a golden test from a saved run

    Args:
        run_file: Path to run.json file
        output_file: Optional output path for golden test
        tolerance: Numerical tolerance

    Returns:
        Golden test specification
    """
    with open(run_file) as f:
        run_data = json.load(f)

    # Extract pipeline and inputs from run
    pipeline = run_data.get("pipeline", {})
    inputs = run_data.get("input", {})
    outputs = run_data.get("output", {})

    # Create golden test
    golden_test = create_golden_test(
        pipeline=pipeline,
        inputs=inputs,
        expected_outputs=outputs,
        tolerance=tolerance,
        metadata={"source_run": str(run_file), "run_id": run_data.get("run_id")},
    )

    # Save if output file specified
    if output_file:
        save_golden_test(golden_test, output_file)

    return golden_test


def validate_determinism(
    pipeline: Dict[str, Any],
    test_inputs: List[Dict[str, Any]],
    executor: Any,
    runs_per_input: int = 5,
) -> Dict[str, Any]:
    """
    Validate determinism of a pipeline across multiple inputs and runs

    Args:
        pipeline: Pipeline specification
        test_inputs: List of test inputs
        executor: Executor instance (must be deterministic)
        runs_per_input: Number of runs per input

    Returns:
        Validation report
    """
    if not executor.deterministic:
        raise ValueError("Executor must be in deterministic mode")

    report = {
        "pipeline": pipeline.get("name", "unnamed"),
        "total_inputs": len(test_inputs),
        "runs_per_input": runs_per_input,
        "deterministic_inputs": 0,
        "non_deterministic_inputs": 0,
        "details": [],
    }

    for idx, inputs in enumerate(test_inputs):
        logger.info(f"Testing input set {idx+1}/{len(test_inputs)}")

        try:
            golden_output, is_deterministic = generate_golden_output(
                pipeline, inputs, executor, runs_per_input
            )

            if is_deterministic:
                report["deterministic_inputs"] += 1
            else:
                report["non_deterministic_inputs"] += 1

            report["details"].append(
                {
                    "input_index": idx,
                    "deterministic": is_deterministic,
                    "output_hash": hashlib.sha256(
                        json.dumps(golden_output, sort_keys=True).encode()
                    ).hexdigest()[:16],
                }
            )

        except Exception as e:
            report["non_deterministic_inputs"] += 1
            report["details"].append(
                {"input_index": idx, "deterministic": False, "error": str(e)}
            )

    report["fully_deterministic"] = report["non_deterministic_inputs"] == 0

    return report
