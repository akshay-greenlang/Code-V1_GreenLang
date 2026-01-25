"""
ESDSimulator - ESD Simulation Mode for Testing

This module implements simulation capabilities for Emergency Shutdown
Systems per IEC 61511 requirements for testing and validation.
Simulation mode allows testing ESD logic without affecting actual
process equipment.

Key features:
- Full logic simulation without output activation
- Response time measurement
- Sequence verification
- Training mode support

Reference: IEC 61511-1 Clause 16, ISA TR84.00.04

Example:
    >>> from greenlang.safety.esd.esd_simulator import ESDSimulator
    >>> simulator = ESDSimulator(config)
    >>> result = simulator.simulate_shutdown(level=1)
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime
import time
import uuid

logger = logging.getLogger(__name__)


class SimulationMode(str, Enum):
    """Simulation mode types."""

    FULL_SIMULATION = "full_simulation"  # All components simulated
    PARTIAL_SIMULATION = "partial_simulation"  # Some real, some simulated
    LOGIC_ONLY = "logic_only"  # Logic simulation only
    RESPONSE_TEST = "response_test"  # Response time testing
    TRAINING = "training"  # Training mode


class SimulationConfig(BaseModel):
    """Configuration for ESD simulation."""

    simulation_id: str = Field(
        default_factory=lambda: f"SIM-{uuid.uuid4().hex[:8].upper()}",
        description="Simulation identifier"
    )
    mode: SimulationMode = Field(
        default=SimulationMode.FULL_SIMULATION,
        description="Simulation mode"
    )
    esd_system_id: str = Field(
        ...,
        description="ESD system to simulate"
    )
    shutdown_level: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Shutdown level to simulate"
    )
    simulate_sensors: bool = Field(
        default=True,
        description="Simulate sensor inputs"
    )
    simulate_logic: bool = Field(
        default=True,
        description="Simulate logic execution"
    )
    simulate_actuators: bool = Field(
        default=True,
        description="Simulate actuator outputs (no real output)"
    )
    inject_faults: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Faults to inject during simulation"
    )
    expected_response_ms: float = Field(
        default=1000.0,
        gt=0,
        description="Expected response time (ms)"
    )
    timeout_ms: float = Field(
        default=5000.0,
        gt=0,
        description="Simulation timeout (ms)"
    )
    record_timestamps: bool = Field(
        default=True,
        description="Record detailed timestamps"
    )


class SimulationStep(BaseModel):
    """Individual step in simulation."""

    step_number: int = Field(
        ...,
        description="Step sequence number"
    )
    step_name: str = Field(
        ...,
        description="Step name"
    )
    component_type: str = Field(
        ...,
        description="Component type (sensor, logic, actuator)"
    )
    component_id: str = Field(
        ...,
        description="Component identifier"
    )
    action: str = Field(
        ...,
        description="Simulated action"
    )
    expected_state: str = Field(
        ...,
        description="Expected state after action"
    )
    actual_state: Optional[str] = Field(
        None,
        description="Actual state achieved"
    )
    start_time_ms: float = Field(
        ...,
        description="Step start time (ms from simulation start)"
    )
    end_time_ms: Optional[float] = Field(
        None,
        description="Step end time (ms)"
    )
    duration_ms: Optional[float] = Field(
        None,
        description="Step duration (ms)"
    )
    passed: bool = Field(
        default=False,
        description="Did step pass"
    )
    fault_injected: bool = Field(
        default=False,
        description="Was fault injected in this step"
    )
    notes: str = Field(
        default="",
        description="Step notes"
    )


class SimulationResult(BaseModel):
    """Result of ESD simulation."""

    simulation_id: str = Field(
        ...,
        description="Simulation identifier"
    )
    mode: SimulationMode = Field(
        ...,
        description="Simulation mode"
    )
    start_time: datetime = Field(
        ...,
        description="Simulation start time"
    )
    end_time: datetime = Field(
        ...,
        description="Simulation end time"
    )
    total_duration_ms: float = Field(
        ...,
        description="Total duration (ms)"
    )
    shutdown_level: int = Field(
        ...,
        description="Simulated shutdown level"
    )
    overall_passed: bool = Field(
        ...,
        description="Did simulation pass overall"
    )
    steps_total: int = Field(
        ...,
        description="Total steps"
    )
    steps_passed: int = Field(
        ...,
        description="Steps passed"
    )
    steps_failed: int = Field(
        ...,
        description="Steps failed"
    )
    steps: List[SimulationStep] = Field(
        default_factory=list,
        description="Simulation steps"
    )
    response_time_ms: float = Field(
        ...,
        description="Measured response time (ms)"
    )
    response_time_met: bool = Field(
        ...,
        description="Did response time meet requirement"
    )
    faults_injected: int = Field(
        default=0,
        description="Number of faults injected"
    )
    faults_detected: int = Field(
        default=0,
        description="Number of faults detected by logic"
    )
    safe_state_achieved: bool = Field(
        default=False,
        description="Was safe state achieved"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Error messages"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    conducted_by: str = Field(
        default="",
        description="Person conducting simulation"
    )
    witnessed_by: Optional[str] = Field(
        None,
        description="Witness (if required)"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ESDSimulator:
    """
    ESD System Simulator.

    Provides simulation capabilities for ESD systems without
    affecting actual process equipment. Used for:
    - Pre-commissioning testing
    - Proof test simulation
    - Training
    - Response time validation

    The simulator follows IEC 61511 principles:
    - No actual outputs during simulation
    - Complete logging
    - Response time measurement

    Attributes:
        config: SimulationConfig settings
        simulation_history: Historical results

    Example:
        >>> config = SimulationConfig(esd_system_id="ESD-001")
        >>> simulator = ESDSimulator(config)
        >>> result = simulator.run_simulation()
    """

    # Default simulation steps for each shutdown level
    DEFAULT_STEPS: Dict[int, List[Dict[str, Any]]] = {
        0: [  # Total facility shutdown
            {"name": "All fuel isolation", "component": "XV-001", "type": "actuator"},
            {"name": "Main burner trip", "component": "BMS-001", "type": "logic"},
            {"name": "Emergency vent", "component": "BDV-001", "type": "actuator"},
        ],
        1: [  # Process area shutdown
            {"name": "Area isolation", "component": "XV-101", "type": "actuator"},
            {"name": "Depressurization", "component": "BDV-101", "type": "actuator"},
        ],
        2: [  # Unit shutdown
            {"name": "Unit isolation", "component": "XV-201", "type": "actuator"},
            {"name": "Pump trip", "component": "PS-201", "type": "actuator"},
        ],
        3: [  # Equipment shutdown
            {"name": "Equipment isolation", "component": "XV-301", "type": "actuator"},
        ],
    }

    def __init__(
        self,
        config: SimulationConfig,
        step_executor: Optional[Callable] = None
    ):
        """
        Initialize ESDSimulator.

        Args:
            config: SimulationConfig settings
            step_executor: Optional callback for step execution
        """
        self.config = config
        self.step_executor = step_executor or self._default_executor
        self.simulation_history: List[SimulationResult] = []

        logger.info(
            f"ESDSimulator initialized: {config.simulation_id}, "
            f"mode={config.mode.value}"
        )

    def run_simulation(
        self,
        conducted_by: str,
        witnessed_by: Optional[str] = None,
        custom_steps: Optional[List[Dict[str, Any]]] = None
    ) -> SimulationResult:
        """
        Run ESD simulation.

        Args:
            conducted_by: Person running simulation
            witnessed_by: Witness (if required)
            custom_steps: Custom simulation steps (optional)

        Returns:
            SimulationResult with detailed results
        """
        start_time = datetime.utcnow()
        start_ms = time.time() * 1000

        logger.info(
            f"Starting ESD simulation {self.config.simulation_id} "
            f"by {conducted_by}"
        )

        # Get steps for shutdown level
        steps_config = custom_steps or self.DEFAULT_STEPS.get(
            self.config.shutdown_level, []
        )

        # Add sensor steps if simulating
        all_steps = []
        step_num = 1

        if self.config.simulate_sensors:
            all_steps.append({
                "name": "Sensor activation",
                "component": "SENSOR-SIM",
                "type": "sensor",
                "step_num": step_num
            })
            step_num += 1

        if self.config.simulate_logic:
            all_steps.append({
                "name": "Logic evaluation",
                "component": "LOGIC-SIM",
                "type": "logic",
                "step_num": step_num
            })
            step_num += 1

        for step in steps_config:
            step["step_num"] = step_num
            all_steps.append(step)
            step_num += 1

        # Execute simulation steps
        simulation_steps = []
        errors = []
        warnings = []
        faults_detected = 0
        response_time_ms = 0.0

        for step_config in all_steps:
            step_start_ms = (time.time() * 1000) - start_ms

            # Check for fault injection
            fault_injected = False
            for fault in self.config.inject_faults:
                if fault.get("component") == step_config.get("component"):
                    fault_injected = True
                    break

            # Execute step
            step = SimulationStep(
                step_number=step_config["step_num"],
                step_name=step_config["name"],
                component_type=step_config["type"],
                component_id=step_config["component"],
                action="simulate",
                expected_state="SAFE",
                start_time_ms=step_start_ms,
                fault_injected=fault_injected,
            )

            try:
                success, actual_state = self.step_executor(step_config, fault_injected)

                step.end_time_ms = (time.time() * 1000) - start_ms
                step.duration_ms = step.end_time_ms - step.start_time_ms
                step.actual_state = actual_state
                step.passed = success

                if fault_injected and success:
                    faults_detected += 1

                if not success:
                    errors.append(f"Step {step.step_number} failed: {step.step_name}")

            except Exception as e:
                step.passed = False
                step.notes = str(e)
                errors.append(f"Step {step.step_number} error: {e}")

            simulation_steps.append(step)

            # Track response time to first output
            if step_config["type"] == "actuator" and response_time_ms == 0:
                response_time_ms = step.end_time_ms or 0

        end_time = datetime.utcnow()
        total_duration_ms = (time.time() * 1000) - start_ms

        # If no actuator steps, use total duration as response time
        if response_time_ms == 0:
            response_time_ms = total_duration_ms

        # Calculate results
        steps_passed = sum(1 for s in simulation_steps if s.passed)
        steps_failed = len(simulation_steps) - steps_passed
        overall_passed = steps_failed == 0
        response_time_met = response_time_ms <= self.config.expected_response_ms

        if not response_time_met:
            warnings.append(
                f"Response time {response_time_ms:.0f}ms exceeds "
                f"requirement {self.config.expected_response_ms:.0f}ms"
            )

        # Build result
        result = SimulationResult(
            simulation_id=self.config.simulation_id,
            mode=self.config.mode,
            start_time=start_time,
            end_time=end_time,
            total_duration_ms=total_duration_ms,
            shutdown_level=self.config.shutdown_level,
            overall_passed=overall_passed,
            steps_total=len(simulation_steps),
            steps_passed=steps_passed,
            steps_failed=steps_failed,
            steps=simulation_steps,
            response_time_ms=response_time_ms,
            response_time_met=response_time_met,
            faults_injected=len(self.config.inject_faults),
            faults_detected=faults_detected,
            safe_state_achieved=overall_passed,
            errors=errors,
            warnings=warnings,
            conducted_by=conducted_by,
            witnessed_by=witnessed_by,
        )

        # Calculate provenance
        result.provenance_hash = self._calculate_provenance(result)

        # Store in history
        self.simulation_history.append(result)

        logger.info(
            f"Simulation {self.config.simulation_id} complete: "
            f"{'PASS' if overall_passed else 'FAIL'}, "
            f"response={response_time_ms:.0f}ms"
        )

        return result

    def simulate_fault_scenario(
        self,
        fault_component: str,
        fault_type: str,
        conducted_by: str
    ) -> SimulationResult:
        """
        Simulate a specific fault scenario.

        Args:
            fault_component: Component to inject fault
            fault_type: Type of fault
            conducted_by: Person running simulation

        Returns:
            SimulationResult
        """
        # Add fault to configuration
        self.config.inject_faults.append({
            "component": fault_component,
            "type": fault_type,
            "injected_at": datetime.utcnow().isoformat(),
        })

        return self.run_simulation(conducted_by)

    def validate_response_time(
        self,
        iterations: int = 5,
        conducted_by: str = ""
    ) -> Dict[str, Any]:
        """
        Run multiple simulations to validate response time.

        Args:
            iterations: Number of test iterations
            conducted_by: Person running tests

        Returns:
            Response time validation results
        """
        logger.info(f"Running {iterations} response time validation iterations")

        response_times = []
        all_passed = True

        for i in range(iterations):
            result = self.run_simulation(conducted_by)
            response_times.append(result.response_time_ms)

            if not result.response_time_met:
                all_passed = False

        # Calculate statistics
        import statistics
        avg_response = statistics.mean(response_times)
        max_response = max(response_times)
        min_response = min(response_times)
        stdev_response = statistics.stdev(response_times) if len(response_times) > 1 else 0

        return {
            "iterations": iterations,
            "all_passed": all_passed,
            "response_times_ms": response_times,
            "average_ms": round(avg_response, 1),
            "max_ms": round(max_response, 1),
            "min_ms": round(min_response, 1),
            "stdev_ms": round(stdev_response, 1),
            "requirement_ms": self.config.expected_response_ms,
            "margin_ms": round(self.config.expected_response_ms - max_response, 1),
            "recommendation": (
                "Response time acceptable"
                if all_passed
                else "Response time improvement required"
            ),
        }

    def get_simulation_report(self) -> Dict[str, Any]:
        """Generate simulation summary report."""
        if not self.simulation_history:
            return {"error": "No simulations performed"}

        total = len(self.simulation_history)
        passed = sum(1 for r in self.simulation_history if r.overall_passed)
        avg_response = sum(
            r.response_time_ms for r in self.simulation_history
        ) / total

        return {
            "report_timestamp": datetime.utcnow().isoformat(),
            "esd_system_id": self.config.esd_system_id,
            "total_simulations": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate_percent": round((passed / total) * 100, 1),
            "average_response_ms": round(avg_response, 1),
            "requirement_ms": self.config.expected_response_ms,
            "latest_result": {
                "simulation_id": self.simulation_history[-1].simulation_id,
                "passed": self.simulation_history[-1].overall_passed,
                "response_ms": self.simulation_history[-1].response_time_ms,
                "timestamp": self.simulation_history[-1].start_time.isoformat(),
            },
            "provenance_hash": hashlib.sha256(
                f"{datetime.utcnow().isoformat()}|{total}|{passed}".encode()
            ).hexdigest()
        }

    def _default_executor(
        self,
        step_config: Dict[str, Any],
        fault_injected: bool
    ) -> tuple:
        """
        Default step executor (simulation).

        Args:
            step_config: Step configuration
            fault_injected: Was fault injected

        Returns:
            Tuple of (success, actual_state)
        """
        # Simulate processing time
        time.sleep(0.01)  # 10ms per step

        # If fault injected, 50% chance of detection (simplified)
        if fault_injected:
            import random
            detected = random.random() > 0.5
            return detected, "FAULT_DETECTED" if detected else "FAULT_MISSED"

        return True, "SAFE"

    def _calculate_provenance(self, result: SimulationResult) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{result.simulation_id}|"
            f"{result.overall_passed}|"
            f"{result.response_time_ms}|"
            f"{result.end_time.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
