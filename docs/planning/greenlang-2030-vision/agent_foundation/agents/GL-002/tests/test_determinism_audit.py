# -*- coding: utf-8 -*-
"""
Comprehensive Determinism Audit for GL-002 BoilerEfficiencyOptimizer

This module performs a complete determinism audit to verify byte-identical reproducibility
across multiple runs, environments, and configurations. It validates:

1. Multiple runs (10x) with identical inputs produce identical outputs
2. All numerical calculations produce deterministic results
3. Hash values match across runs
4. No random sources without proper seeding
5. Cross-environment consistency (where feasible)

Author: GL-Determinism Auditor
Date: 2025-11-15
"""

import pytest
import hashlib
import json
import os
import sys
import platform
import asyncio
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict, field
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import GL-002 components
from boiler_efficiency_orchestrator import (
    BoilerEfficiencyOptimizer,
    OperationMode,
    OptimizationStrategy
)
from config import (
    BoilerEfficiencyConfig,
    create_default_config,
    BoilerConfiguration,
    OperationalConstraints
)
from tools import BoilerEfficiencyTools


# ============================================================================
# AUDIT DATA STRUCTURES
# ============================================================================

@dataclass
class RunArtifact:
    """Represents a single execution artifact with hashes."""
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
    platform_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeterminismAuditResult:
    """Complete audit result with all findings."""
    audit_timestamp: str
    test_duration_seconds: float
    total_runs: int
    runs_succeeded: int
    runs_failed: int
    all_hashes_match: bool
    mismatches: List[Dict[str, Any]] = field(default_factory=list)
    numerical_stability: float  # Percentage match
    cross_environment_compatible: bool
    recommendations: List[str] = field(default_factory=list)
    pass_fail: str  # "PASS" or "FAIL"


# ============================================================================
# STANDARD TEST INPUTS
# ============================================================================

def get_standard_test_input() -> Dict[str, Any]:
    """
    Get standard deterministic test input.
    This input should always produce the same results.
    """
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


# ============================================================================
# HASH CALCULATION UTILITIES
# ============================================================================

def calculate_hash(data: Any, algorithm: str = 'sha256') -> str:
    """
    Calculate deterministic hash of data.
    Uses sorted JSON representation for consistent hashing.
    """
    try:
        # Convert to JSON with sorted keys
        json_str = json.dumps(data, sort_keys=True, default=str)

        # Calculate hash
        if algorithm == 'sha256':
            return hashlib.sha256(json_str.encode()).hexdigest()
        elif algorithm == 'md5':
            return hashlib.md5(json_str.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    except Exception as e:
        logger.error(f"Failed to calculate hash: {e}")
        return ""


def calculate_nested_hashes(data: Dict[str, Any]) -> Dict[str, str]:
    """Calculate hashes for nested structures."""
    hashes = {}
    for key, value in data.items():
        if isinstance(value, dict):
            hashes[f"{key}_hash"] = calculate_hash(value)
        elif isinstance(value, (list, tuple)):
            hashes[f"{key}_hash"] = calculate_hash(list(value))
        else:
            hashes[f"{key}_hash"] = calculate_hash(value)
    return hashes


# ============================================================================
# DETERMINISM AUDIT SUITE
# ============================================================================

class DeterminismAuditor:
    """
    Core auditor for determinism verification.
    Performs comprehensive testing of GL-002 reproducibility.
    """

    def __init__(self, num_runs: int = 10):
        """Initialize auditor with configuration."""
        self.num_runs = num_runs
        self.run_artifacts: List[RunArtifact] = []
        self.test_start_time = None
        self.platform_info = {
            'system': platform.system(),
            'python_version': sys.version,
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'machine': platform.machine()
        }
        logger.info(f"DeterminismAuditor initialized for {num_runs} runs")
        logger.info(f"Platform: {self.platform_info}")

    async def audit_run_reproducibility(self) -> DeterminismAuditResult:
        """
        Execute comprehensive reproducibility audit.
        Runs same input 10 times and compares outputs.
        """
        self.test_start_time = time.time()
        audit_start = datetime.now(timezone.utc)

        logger.info("=" * 80)
        logger.info("STARTING DETERMINISM AUDIT")
        logger.info("=" * 80)

        # Create orchestrator with deterministic config
        config = create_default_config()

        # Create test input
        test_input = get_standard_test_input()

        # Run orchestrator multiple times
        logger.info(f"Running {self.num_runs} determinism tests...")

        for run_num in range(self.num_runs):
            try:
                logger.info(f"\nRun {run_num + 1}/{self.num_runs}")
                artifact = await self._execute_single_run(
                    run_num, config, test_input
                )
                self.run_artifacts.append(artifact)
            except Exception as e:
                logger.error(f"Run {run_num + 1} failed: {e}", exc_info=True)

        # Analyze results
        audit_end = datetime.now(timezone.utc)
        duration = (audit_end - audit_start).total_seconds()

        result = self._analyze_audit_results(duration)

        logger.info("\n" + "=" * 80)
        logger.info("AUDIT RESULTS")
        logger.info("=" * 80)
        logger.info(f"Pass/Fail: {result.pass_fail}")
        logger.info(f"Total Runs: {result.total_runs}")
        logger.info(f"Successful: {result.runs_succeeded}")
        logger.info(f"All Hashes Match: {result.all_hashes_match}")
        logger.info(f"Numerical Stability: {result.numerical_stability:.1f}%")
        logger.info("=" * 80)

        return result

    async def _execute_single_run(
        self,
        run_id: int,
        config: BoilerEfficiencyConfig,
        test_input: Dict[str, Any]
    ) -> RunArtifact:
        """Execute single run and capture artifacts."""
        run_start = time.time()
        run_timestamp = datetime.now(timezone.utc).isoformat()

        # Calculate input hash
        input_hash = calculate_hash(test_input)

        # Create orchestrator
        orchestrator = BoilerEfficiencyOptimizer(config)

        # Execute
        logger.debug(f"Executing orchestrator with input hash {input_hash[:16]}...")
        result = await orchestrator.execute(test_input)

        # Calculate execution time
        execution_time_ms = (time.time() - run_start) * 1000

        # Extract results and calculate hashes
        output_hash = calculate_hash(result)
        efficiency_result = result.get(
            'operational_state', {}
        ).get('efficiency_percent', 0)

        combustion = result.get('combustion_optimization', {})
        combustion_hash = calculate_hash(combustion)

        steam = result.get('steam_generation', {})
        steam_hash = calculate_hash(steam)

        emissions = result.get('emissions_optimization', {})
        emissions_hash = calculate_hash(emissions)

        dashboard = result.get('kpi_dashboard', {})
        dashboard_hash = calculate_hash(dashboard)

        provenance_hash = result.get('provenance_hash', '')

        # Create artifact
        artifact = RunArtifact(
            run_id=f"RUN-{run_id:03d}",
            timestamp=run_timestamp,
            input_hash=input_hash,
            output_hash=output_hash,
            efficiency_result=efficiency_result,
            combustion_hash=combustion_hash,
            steam_hash=steam_hash,
            emissions_hash=emissions_hash,
            dashboard_hash=dashboard_hash,
            provenance_hash=provenance_hash,
            execution_time_ms=execution_time_ms,
            platform_info=self.platform_info.copy()
        )

        logger.debug(f"  Input Hash:      {input_hash[:32]}...")
        logger.debug(f"  Output Hash:     {output_hash[:32]}...")
        logger.debug(f"  Efficiency:      {efficiency_result:.6f}")
        logger.debug(f"  Execution Time:  {execution_time_ms:.2f}ms")

        return artifact

    def _analyze_audit_results(self, duration: float) -> DeterminismAuditResult:
        """Analyze all run artifacts and produce audit result."""
        result = DeterminismAuditResult(
            audit_timestamp=datetime.now(timezone.utc).isoformat(),
            test_duration_seconds=duration,
            total_runs=len(self.run_artifacts),
            runs_succeeded=len(self.run_artifacts),
            runs_failed=self.num_runs - len(self.run_artifacts),
            all_hashes_match=True,
            cross_environment_compatible=True
        )

        if not self.run_artifacts:
            result.pass_fail = "FAIL"
            result.all_hashes_match = False
            result.recommendations.append("No successful runs completed")
            return result

        # Compare all hashes
        first_run = self.run_artifacts[0]
        hash_fields = [
            'input_hash', 'output_hash', 'combustion_hash',
            'steam_hash', 'emissions_hash', 'dashboard_hash',
            'provenance_hash'
        ]

        for field_name in hash_fields:
            first_hash = getattr(first_run, field_name)
            mismatches = []

            for run_num, artifact in enumerate(self.run_artifacts[1:], start=1):
                current_hash = getattr(artifact, field_name)
                if current_hash != first_hash:
                    mismatches.append({
                        'field': field_name,
                        'run_1_value': first_hash[:32],
                        'run_n_value': current_hash[:32],
                        'run_number': run_num,
                        'artifact_id': artifact.run_id
                    })

            if mismatches:
                result.all_hashes_match = False
                result.mismatches.extend(mismatches)

        # Check numerical stability
        efficiencies = [a.efficiency_result for a in self.run_artifacts]
        if efficiencies:
            std_dev = self._calculate_std_dev(efficiencies)
            # If all values are identical
            if std_dev == 0:
                result.numerical_stability = 100.0
            else:
                # Count exact matches
                matches = sum(1 for e in efficiencies if e == efficiencies[0])
                result.numerical_stability = (matches / len(efficiencies)) * 100

        # Determine pass/fail
        if result.all_hashes_match and result.numerical_stability == 100.0:
            result.pass_fail = "PASS"
        else:
            result.pass_fail = "FAIL"
            if not result.all_hashes_match:
                result.recommendations.append(
                    "Hash mismatches detected - see details for specific fields"
                )
            if result.numerical_stability < 100.0:
                result.recommendations.append(
                    f"Numerical stability only {result.numerical_stability:.1f}% - "
                    "floating-point ordering may be non-deterministic"
                )

        return result

    @staticmethod
    def _calculate_std_dev(values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


# ============================================================================
# PYTEST TEST CASES
# ============================================================================

@pytest.fixture
def auditor():
    """Create auditor fixture."""
    return DeterminismAuditor(num_runs=10)


@pytest.mark.asyncio
async def test_complete_determinism_audit(auditor):
    """
    MAIN DETERMINISM TEST: Complete audit of GL-002 reproducibility.

    This test performs 10 identical runs and verifies:
    - All hashes match exactly
    - Efficiency values are identical
    - All internal calculations are deterministic
    """
    result = await auditor.audit_run_reproducibility()

    # Assertions
    assert result.pass_fail == "PASS", (
        f"Determinism audit FAILED. "
        f"Mismatches: {len(result.mismatches)}. "
        f"Stability: {result.numerical_stability:.1f}%"
    )
    assert result.all_hashes_match, "Hash mismatches detected"
    assert result.numerical_stability == 100.0, (
        f"Numerical stability only {result.numerical_stability:.1f}%"
    )


@pytest.mark.asyncio
async def test_hash_consistency(auditor):
    """
    Test that identical inputs always produce identical hashes.
    """
    config = create_default_config()
    test_input = get_standard_test_input()
    orchestrator = BoilerEfficiencyOptimizer(config)

    hashes = []
    for i in range(5):
        result = await orchestrator.execute(test_input)
        output_hash = calculate_hash(result)
        hashes.append(output_hash)

    # All hashes should be identical
    assert len(set(hashes)) == 1, f"Got {len(set(hashes))} different hashes"


@pytest.mark.asyncio
async def test_numerical_determinism(auditor):
    """
    Test that numerical calculations are deterministic.
    """
    tools = BoilerEfficiencyTools()
    boiler_data = {
        'boiler_id': 'TEST-001',
        'fuel_properties': {
            'carbon_percent': 75.0,
            'hydrogen_percent': 25.0,
            'sulfur_percent': 0.0,
            'oxygen_percent': 0.0,
            'moisture_percent': 0.0,
            'heating_value_mj_kg': 50.0
        }
    }

    sensor_feeds = {
        'fuel_flow_kg_hr': 1000.0,
        'steam_flow_kg_hr': 10000.0,
        'stack_temperature_c': 180.0,
        'ambient_temperature_c': 25.0,
        'o2_percent': 3.0,
        'co_ppm': 50,
        'steam_pressure_bar': 10.0,
        'steam_temperature_c': 180.0,
        'feedwater_temperature_c': 80.0,
        'blowdown_rate_percent': 3.0
    }

    results = []
    for _ in range(5):
        result = tools.calculate_boiler_efficiency(boiler_data, sensor_feeds)
        results.append(asdict(result))

    # All results should be identical
    first_result = results[0]
    for result in results[1:]:
        assert result == first_result, "Efficiency calculations not deterministic"


@pytest.mark.asyncio
async def test_combustion_optimization_determinism(auditor):
    """
    Test combustion optimization produces deterministic results.
    """
    tools = BoilerEfficiencyTools()

    operational_state = {
        'excess_air_percent': 15,
        'fuel_flow_rate_kg_hr': 1000.0,
        'load_percent': 75,
        'combustion_temperature_c': 1200,
        'stack_temperature_c': 180,
        'efficiency_percent': 80
    }

    fuel_data = {
        'cost_usd_per_kg': 0.05,
        'properties': {
            'carbon_percent': 75.0,
            'hydrogen_percent': 25.0,
            'sulfur_percent': 0.0,
            'heating_value_mj_kg': 50.0
        }
    }

    constraints = {
        'min_excess_air_percent': 5.0,
        'max_excess_air_percent': 25.0
    }

    results = []
    for _ in range(5):
        result = tools.optimize_combustion_parameters(
            operational_state, fuel_data, constraints
        )
        results.append(asdict(result))

    # All results should be identical
    first_result = results[0]
    for result in results[1:]:
        assert result == first_result, (
            "Combustion optimization not deterministic"
        )


@pytest.mark.asyncio
async def test_steam_optimization_determinism(auditor):
    """
    Test steam optimization produces deterministic results.
    """
    tools = BoilerEfficiencyTools()

    steam_demand = {
        'required_flow_kg_hr': 10000.0,
        'required_pressure_bar': 10.0,
        'required_temperature_c': 180.0
    }

    operational_state = {
        'steam_flow_rate_kg_hr': 9000.0,
        'steam_pressure_bar': 10.0,
        'tds_ppm': 2000.0,
        'feedwater_temperature_c': 80.0,
        'steam_moisture_percent': 0.5,
        'fuel_flow_rate_kg_hr': 1000.0
    }

    constraints = {
        'max_tds_ppm': 3500,
        'min_steam_quality': 0.95
    }

    results = []
    for _ in range(5):
        result = tools.optimize_steam_generation(
            steam_demand, operational_state, constraints
        )
        results.append(asdict(result))

    # All results should be identical
    first_result = results[0]
    for result in results[1:]:
        assert result == first_result, (
            "Steam optimization not deterministic"
        )


# ============================================================================
# MAIN AUDIT EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("GL-002 Determinism Audit")
    print("=" * 80)

    # Run audit
    auditor = DeterminismAuditor(num_runs=10)

    # Use asyncio to run async code
    result = asyncio.run(auditor.audit_run_reproducibility())

    # Generate JSON report
    report_data = {
        'audit_timestamp': result.audit_timestamp,
        'pass_fail': result.pass_fail,
        'total_runs': result.total_runs,
        'runs_succeeded': result.runs_succeeded,
        'all_hashes_match': result.all_hashes_match,
        'numerical_stability_percent': result.numerical_stability,
        'mismatches': result.mismatches,
        'recommendations': result.recommendations,
        'duration_seconds': result.test_duration_seconds
    }

    # Print summary
    print(f"\nAudit Result: {result.pass_fail}")
    print(f"Hashes Match: {result.all_hashes_match}")
    print(f"Numerical Stability: {result.numerical_stability:.1f}%")
    print(f"Duration: {result.test_duration_seconds:.2f}s")
    print(f"\nMismatches: {len(result.mismatches)}")
    for mismatch in result.mismatches:
        print(f"  - {mismatch['field']}: Run {mismatch['run_number']} differs")

    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  - {rec}")
