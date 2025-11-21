#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GL-002 Determinism Audit - Direct Execution Script

This script runs a comprehensive determinism audit without requiring pytest.
It verifies byte-identical reproducibility across 10 runs.

Author: GL-Determinism Auditor
"""

import hashlib
import json
import sys
import os
import platform
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from boiler_efficiency_orchestrator import BoilerEfficiencyOptimizer
    from config import create_default_config
    from tools import BoilerEfficiencyTools
    logger.info("Successfully imported GL-002 modules")
except ImportError as e:
    logger.error(f"Failed to import GL-002 modules: {e}")
    sys.exit(1)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RunArtifact:
    """Single run execution artifact."""
    run_id: str
    timestamp: str
    input_hash: str
    output_hash: str
    efficiency_result: float
    combustion_hash: str
    steam_hash: str
    emissions_hash: str
    dashboard_hash: str
    provenance_hash: str
    execution_time_ms: float
    success: bool
    error_message: str = ""


@dataclass
class AuditResult:
    """Complete audit result."""
    pass_fail: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    all_hashes_match: bool
    numerical_stability_percent: float
    mismatches: List[Dict[str, Any]]
    duration_seconds: float
    recommendations: List[str]


# ============================================================================
# DETERMINISM TESTING
# ============================================================================

def calculate_hash(data: Any, algorithm: str = 'sha256') -> str:
    """Calculate deterministic hash of data."""
    try:
        json_str = json.dumps(data, sort_keys=True, default=str)
        if algorithm == 'sha256':
            return hashlib.sha256(json_str.encode()).hexdigest()
        elif algorithm == 'md5':
            return hashlib.md5(json_str.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Hash calculation failed: {e}")
        return ""
    return ""


def get_test_input() -> Dict[str, Any]:
    """Get standard test input for determinism testing."""
    return {
        'boiler_data': {
            'boiler_id': 'BOILER-001',
            'fuel_type': 'natural_gas',
            'fuel_properties': {
                'carbon_percent': 75.0,
                'hydrogen_percent': 25.0,
                'sulfur_percent': 0.0,
                'nitrogen_percent': 0.0,
                'oxygen_percent': 0.0,
                'moisture_percent': 0.0,
                'heating_value_mj_kg': 50.0
            }
        },
        'sensor_feeds': {
            'fuel_flow_kg_hr': 1000.0,
            'steam_flow_kg_hr': 10000.0,
            'stack_temperature_c': 180.0,
            'ambient_temperature_c': 25.0,
            'o2_percent': 3.0,
            'co_ppm': 50,
            'steam_pressure_bar': 10.0,
            'steam_temperature_c': 180.0,
            'feedwater_temperature_c': 80.0,
            'blowdown_rate_percent': 3.0,
            'tds_ppm': 2000.0,
            'load_percent': 75.0
        },
        'constraints': {
            'min_excess_air_percent': 5.0,
            'max_excess_air_percent': 25.0,
            'max_tds_ppm': 3500,
            'min_steam_quality': 0.95,
            'nox_limit_ppm': 30,
            'co_limit_ppm': 100
        },
        'fuel_data': {
            'cost_usd_per_kg': 0.05,
            'properties': {
                'carbon_percent': 75.0,
                'hydrogen_percent': 25.0,
                'sulfur_percent': 0.0,
                'heating_value_mj_kg': 50.0
            }
        },
        'steam_demand': {
            'required_flow_kg_hr': 10000.0,
            'required_pressure_bar': 10.0,
            'required_temperature_c': 180.0
        },
        'emission_limits': {
            'nox_limit_ppm': 30,
            'co_limit_ppm': 100
        }
    }


def run_single_execution(run_id: int, config: Any, test_input: Dict[str, Any]) -> RunArtifact:
    """Execute single run and capture artifact."""
    import time
    run_start = time.time()
    timestamp = datetime.now(timezone.utc).isoformat()

    artifact = RunArtifact(
        run_id=f"RUN-{run_id:03d}",
        timestamp=timestamp,
        input_hash="",
        output_hash="",
        efficiency_result=0.0,
        combustion_hash="",
        steam_hash="",
        emissions_hash="",
        dashboard_hash="",
        provenance_hash="",
        execution_time_ms=0.0,
        success=False
    )

    try:
        # Calculate input hash
        artifact.input_hash = calculate_hash(test_input)
        logger.info(f"Run {run_id+1}: Input hash = {artifact.input_hash[:16]}...")

        # Create orchestrator
        orchestrator = BoilerEfficiencyOptimizer(config)

        # Execute (synchronous mode)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(orchestrator.execute(test_input))
        loop.close()

        # Calculate hashes
        artifact.output_hash = calculate_hash(result)
        artifact.efficiency_result = result.get('operational_state', {}).get(
            'efficiency_percent', 0.0
        )

        combustion = result.get('combustion_optimization', {})
        artifact.combustion_hash = calculate_hash(combustion)

        steam = result.get('steam_generation', {})
        artifact.steam_hash = calculate_hash(steam)

        emissions = result.get('emissions_optimization', {})
        artifact.emissions_hash = calculate_hash(emissions)

        dashboard = result.get('kpi_dashboard', {})
        artifact.dashboard_hash = calculate_hash(dashboard)

        artifact.provenance_hash = result.get('provenance_hash', '')

        execution_time = (time.time() - run_start) * 1000
        artifact.execution_time_ms = execution_time

        artifact.success = True

        logger.info(
            f"Run {run_id+1}: "
            f"Output={artifact.output_hash[:16]}..., "
            f"Efficiency={artifact.efficiency_result:.4f}, "
            f"Time={execution_time:.2f}ms"
        )

    except Exception as e:
        artifact.success = False
        artifact.error_message = str(e)
        logger.error(f"Run {run_id+1} failed: {e}")
        logger.error(traceback.format_exc())

    return artifact


def analyze_results(artifacts: List[RunArtifact]) -> AuditResult:
    """Analyze all run artifacts."""
    successful = [a for a in artifacts if a.success]
    failed = [a for a in artifacts if not a.success]

    result = AuditResult(
        pass_fail="PASS",
        total_runs=len(artifacts),
        successful_runs=len(successful),
        failed_runs=len(failed),
        all_hashes_match=True,
        numerical_stability_percent=100.0,
        mismatches=[],
        duration_seconds=0.0,
        recommendations=[]
    )

    if not successful:
        result.pass_fail = "FAIL"
        result.all_hashes_match = False
        result.recommendations.append("No successful runs")
        return result

    # Compare hashes
    first = successful[0]
    hash_fields = [
        'input_hash', 'output_hash', 'combustion_hash',
        'steam_hash', 'emissions_hash', 'dashboard_hash'
    ]

    for field in hash_fields:
        first_val = getattr(first, field)
        for idx, artifact in enumerate(successful[1:], 1):
            current_val = getattr(artifact, field)
            if current_val != first_val:
                result.all_hashes_match = False
                result.mismatches.append({
                    'field': field,
                    'run_1': first_val[:32] if first_val else "NONE",
                    'run_n': current_val[:32] if current_val else "NONE",
                    'run_number': idx,
                    'artifact_id': artifact.run_id
                })

    # Check numerical stability
    efficiencies = [a.efficiency_result for a in successful]
    if efficiencies:
        matches = sum(1 for e in efficiencies if e == efficiencies[0])
        result.numerical_stability_percent = (matches / len(efficiencies)) * 100

    # Determine pass/fail
    if result.all_hashes_match and result.numerical_stability_percent == 100.0:
        result.pass_fail = "PASS"
    else:
        result.pass_fail = "FAIL"
        if not result.all_hashes_match:
            result.recommendations.append(
                f"Hash mismatches detected ({len(result.mismatches)} mismatches)"
            )
        if result.numerical_stability_percent < 100.0:
            result.recommendations.append(
                f"Numerical stability {result.numerical_stability_percent:.1f}% - "
                "non-deterministic calculations detected"
            )

    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute determinism audit."""
    import time

    print("=" * 80)
    print("GL-002 BOILER EFFICIENCY OPTIMIZER - DETERMINISM AUDIT")
    print("=" * 80)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print("=" * 80)

    audit_start = time.time()

    # Create configuration
    try:
        config = create_default_config()
        logger.info("Configuration created successfully")
    except Exception as e:
        logger.error(f"Failed to create config: {e}")
        sys.exit(1)

    # Get test input
    test_input = get_test_input()
    logger.info(f"Test input prepared, hash={calculate_hash(test_input)[:16]}...")

    # Run 10 executions
    artifacts = []
    for run_id in range(10):
        logger.info(f"\n--- Execution {run_id + 1}/10 ---")
        artifact = run_single_execution(run_id, config, test_input)
        artifacts.append(artifact)

    # Analyze results
    audit_result = analyze_results(artifacts)
    audit_duration = time.time() - audit_start
    audit_result.duration_seconds = audit_duration

    # Print results
    print("\n" + "=" * 80)
    print("AUDIT RESULTS")
    print("=" * 80)
    print(f"Status: {audit_result.pass_fail}")
    print(f"Total Runs: {audit_result.total_runs}")
    print(f"Successful: {audit_result.successful_runs}")
    print(f"Failed: {audit_result.failed_runs}")
    print(f"All Hashes Match: {audit_result.all_hashes_match}")
    print(f"Numerical Stability: {audit_result.numerical_stability_percent:.1f}%")
    print(f"Duration: {audit_result.duration_seconds:.2f}s")

    if audit_result.mismatches:
        print(f"\nHash Mismatches: {len(audit_result.mismatches)}")
        for mismatch in audit_result.mismatches:
            print(f"  Field: {mismatch['field']}")
            print(f"    Run 1: {mismatch['run_1']}")
            print(f"    Run {mismatch['run_number']+1}: {mismatch['run_n']}")

    if audit_result.recommendations:
        print("\nRecommendations:")
        for rec in audit_result.recommendations:
            print(f"  - {rec}")

    print("=" * 80)

    # Exit with appropriate code
    sys.exit(0 if audit_result.pass_fail == "PASS" else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Audit failed with exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
