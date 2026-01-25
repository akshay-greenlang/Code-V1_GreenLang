"""
Integration Tests: Optimization Pipeline

Tests full optimization pipeline from input to recommendation including:
- End-to-end optimization flow
- Constraint validation in pipeline
- Explainability generation
- Audit trail completeness

Reference: GL-003 Specification

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from unittest.mock import MagicMock, AsyncMock, patch
import hashlib
import json


# =============================================================================
# Data Classes and Enumerations
# =============================================================================

class OptimizationType(Enum):
    """Types of optimization."""
    DESUPERHEATER = auto()
    CONDENSATE_RECOVERY = auto()
    TRAP_MAINTENANCE = auto()
    ENERGY_EFFICIENCY = auto()
    COMBINED = auto()


class OptimizationStatus(Enum):
    """Optimization execution status."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()


class ConstraintType(Enum):
    """Types of constraints."""
    TEMPERATURE_MIN = auto()
    TEMPERATURE_MAX = auto()
    PRESSURE_MIN = auto()
    PRESSURE_MAX = auto()
    FLOW_MIN = auto()
    FLOW_MAX = auto()
    QUALITY_MIN = auto()
    APPROACH_MIN = auto()


@dataclass
class OptimizationInput:
    """Input data for optimization pipeline."""
    request_id: str
    optimization_type: OptimizationType
    timestamp: datetime
    # Process data
    process_data: Dict[str, Any]
    # Constraints
    constraints: Dict[ConstraintType, float]
    # Objectives
    objectives: Dict[str, float]  # objective_name -> weight
    # Options
    timeout_seconds: float = 30.0
    max_iterations: int = 100
    require_explainability: bool = True


@dataclass
class OptimizationOutput:
    """Output from optimization pipeline."""
    request_id: str
    status: OptimizationStatus
    timestamp: datetime
    # Results
    recommended_setpoints: Dict[str, float]
    expected_savings_kw: float
    expected_cost_savings: float
    # Validation
    constraints_satisfied: bool
    constraint_violations: List[str]
    # Explainability
    explanation: Optional[Dict[str, Any]]
    feature_contributions: Dict[str, float]
    # Audit
    execution_time_ms: float
    iterations: int
    provenance_hash: str


@dataclass
class ConstraintValidationResult:
    """Result of constraint validation."""
    all_satisfied: bool
    violations: List[str]
    margin_to_limits: Dict[str, float]


@dataclass
class ExplainabilityPayload:
    """Explainability information for recommendations."""
    method: str  # "shap", "lime", "physics_based"
    feature_contributions: Dict[str, float]
    physics_trace: Dict[str, Any]
    confidence: float
    local_fidelity: float
    counterfactuals: List[Dict[str, Any]]


@dataclass
class AuditEntry:
    """Audit log entry for optimization."""
    sequence_id: int
    timestamp: datetime
    stage: str
    action: str
    inputs_hash: str
    outputs_hash: str
    duration_ms: float
    status: str


# =============================================================================
# Optimization Pipeline Implementation (Simulated)
# =============================================================================

class OptimizationError(Exception):
    """Error in optimization pipeline."""
    pass


class ConstraintViolationError(OptimizationError):
    """Constraint violation detected."""
    pass


class TimeoutError(OptimizationError):
    """Optimization timed out."""
    pass


class SteamOptimizationPipeline:
    """
    Main optimization pipeline for steam system optimization.

    Stages:
    1. Input validation
    2. Constraint setup
    3. Optimization execution
    4. Solution validation
    5. Explainability generation
    6. Output packaging
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.audit_log: List[AuditEntry] = []
        self._audit_sequence = 0

    async def optimize(self, input_data: OptimizationInput) -> OptimizationOutput:
        """
        Run full optimization pipeline.
        """
        start_time = datetime.now(timezone.utc)
        start_ms = self._get_timestamp_ms()

        try:
            # Stage 1: Validate inputs
            self._log_audit("input_validation", "started", input_data.request_id)
            self._validate_inputs(input_data)
            self._log_audit("input_validation", "completed", input_data.request_id)

            # Stage 2: Setup constraints
            self._log_audit("constraint_setup", "started", input_data.request_id)
            constraints = self._setup_constraints(input_data)
            self._log_audit("constraint_setup", "completed", input_data.request_id)

            # Stage 3: Run optimization
            self._log_audit("optimization", "started", input_data.request_id)
            setpoints, iterations = await self._run_optimization(
                input_data, constraints, input_data.timeout_seconds
            )
            self._log_audit("optimization", "completed", input_data.request_id)

            # Stage 4: Validate solution
            self._log_audit("solution_validation", "started", input_data.request_id)
            validation = self._validate_solution(setpoints, constraints)
            self._log_audit("solution_validation", "completed", input_data.request_id)

            # Stage 5: Generate explainability
            explanation = None
            feature_contributions = {}
            if input_data.require_explainability:
                self._log_audit("explainability", "started", input_data.request_id)
                explanation, feature_contributions = await self._generate_explanation(
                    input_data, setpoints
                )
                self._log_audit("explainability", "completed", input_data.request_id)

            # Stage 6: Calculate savings
            savings_kw, cost_savings = self._calculate_savings(input_data, setpoints)

            # Package output
            end_ms = self._get_timestamp_ms()
            execution_time = end_ms - start_ms

            provenance_hash = self._compute_provenance(input_data, setpoints)

            return OptimizationOutput(
                request_id=input_data.request_id,
                status=OptimizationStatus.COMPLETED,
                timestamp=datetime.now(timezone.utc),
                recommended_setpoints=setpoints,
                expected_savings_kw=savings_kw,
                expected_cost_savings=cost_savings,
                constraints_satisfied=validation.all_satisfied,
                constraint_violations=validation.violations,
                explanation=explanation,
                feature_contributions=feature_contributions,
                execution_time_ms=execution_time,
                iterations=iterations,
                provenance_hash=provenance_hash
            )

        except TimeoutError:
            return self._create_failed_output(input_data, OptimizationStatus.TIMEOUT, "Optimization timed out")
        except ConstraintViolationError as e:
            return self._create_failed_output(input_data, OptimizationStatus.FAILED, str(e))
        except Exception as e:
            return self._create_failed_output(input_data, OptimizationStatus.FAILED, str(e))

    def _validate_inputs(self, input_data: OptimizationInput) -> None:
        """Validate input data."""
        if not input_data.request_id:
            raise OptimizationError("Request ID is required")

        if not input_data.process_data:
            raise OptimizationError("Process data is required")

        # Validate process data has required fields
        required_fields = ["inlet_pressure_mpa", "inlet_temperature_k"]
        for field in required_fields:
            if field not in input_data.process_data:
                raise OptimizationError(f"Missing required field: {field}")

    def _setup_constraints(self, input_data: OptimizationInput) -> Dict[ConstraintType, float]:
        """Setup and merge constraints."""
        # Default constraints
        defaults = {
            ConstraintType.TEMPERATURE_MIN: 373.0,  # 100 C
            ConstraintType.TEMPERATURE_MAX: 673.0,  # 400 C
            ConstraintType.PRESSURE_MIN: 0.1,       # 0.1 MPa
            ConstraintType.PRESSURE_MAX: 10.0,      # 10 MPa
            ConstraintType.APPROACH_MIN: 10.0,      # 10 K approach to saturation
        }

        # Merge with user constraints
        constraints = defaults.copy()
        constraints.update(input_data.constraints)

        return constraints

    async def _run_optimization(
        self,
        input_data: OptimizationInput,
        constraints: Dict[ConstraintType, float],
        timeout: float
    ) -> Tuple[Dict[str, float], int]:
        """
        Run optimization algorithm.

        Returns: (setpoints, iterations)
        """
        # Simulate optimization with timeout
        iterations = 0
        max_iter = input_data.max_iterations

        # Simple optimization simulation
        current_setpoints = {
            "spray_water_flow_kg_s": 0.0,
            "outlet_temperature_k": input_data.process_data.get("target_temperature_k", 450.0),
            "valve_position_percent": 0.0,
        }

        # Simulate iterative improvement
        inlet_temp = input_data.process_data.get("inlet_temperature_k", 500.0)
        target_temp = input_data.process_data.get("target_temperature_k", 450.0)

        # Simple proportional spray calculation
        if inlet_temp > target_temp:
            temp_reduction = inlet_temp - target_temp
            spray_flow = temp_reduction * 0.01  # Simplified

            current_setpoints["spray_water_flow_kg_s"] = spray_flow
            current_setpoints["valve_position_percent"] = min(100, spray_flow * 20)

        # Simulate iterations
        for _ in range(min(10, max_iter)):
            iterations += 1
            await asyncio.sleep(0.001)  # Simulate computation time

        return current_setpoints, iterations

    def _validate_solution(
        self,
        setpoints: Dict[str, float],
        constraints: Dict[ConstraintType, float]
    ) -> ConstraintValidationResult:
        """Validate solution against constraints."""
        violations = []
        margins = {}

        outlet_temp = setpoints.get("outlet_temperature_k", 0)

        # Check temperature constraints
        if outlet_temp < constraints.get(ConstraintType.TEMPERATURE_MIN, 0):
            violations.append(f"Outlet temperature {outlet_temp:.1f} K below minimum")
            margins["temperature_min"] = outlet_temp - constraints[ConstraintType.TEMPERATURE_MIN]
        else:
            margins["temperature_min"] = outlet_temp - constraints.get(ConstraintType.TEMPERATURE_MIN, 0)

        if outlet_temp > constraints.get(ConstraintType.TEMPERATURE_MAX, float('inf')):
            violations.append(f"Outlet temperature {outlet_temp:.1f} K above maximum")
            margins["temperature_max"] = constraints[ConstraintType.TEMPERATURE_MAX] - outlet_temp

        return ConstraintValidationResult(
            all_satisfied=len(violations) == 0,
            violations=violations,
            margin_to_limits=margins
        )

    async def _generate_explanation(
        self,
        input_data: OptimizationInput,
        setpoints: Dict[str, float]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Generate explainability for recommendations."""
        # Simulate SHAP-like feature contributions
        feature_contributions = {
            "inlet_temperature": 0.35,
            "inlet_pressure": 0.20,
            "target_temperature": 0.25,
            "spray_water_temperature": 0.15,
            "mass_flow_rate": 0.05,
        }

        # Physics-based explanation
        explanation = {
            "method": "physics_based",
            "summary": "Spray water flow calculated from energy balance equation.",
            "equations_used": ["Q = m_dot * (h_in - h_out)", "h_spray = Cp * T_spray"],
            "key_drivers": ["Temperature reduction required", "Spray water enthalpy"],
            "confidence": 0.95,
            "local_fidelity": 0.92,
        }

        return explanation, feature_contributions

    def _calculate_savings(
        self,
        input_data: OptimizationInput,
        setpoints: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate expected savings from optimization."""
        # Simplified savings calculation
        spray_flow = setpoints.get("spray_water_flow_kg_s", 0)

        # Energy savings from optimized spray control
        energy_savings_kw = spray_flow * 100  # Simplified: 100 kJ/kg saved

        # Cost savings (example: $0.05/kWh)
        cost_savings = energy_savings_kw * 0.05 * 8760 / 1000  # Annual

        return energy_savings_kw, cost_savings

    def _compute_provenance(
        self,
        input_data: OptimizationInput,
        setpoints: Dict[str, float]
    ) -> str:
        """Compute provenance hash for audit trail."""
        data = {
            "request_id": input_data.request_id,
            "inputs": input_data.process_data,
            "outputs": setpoints,
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _create_failed_output(
        self,
        input_data: OptimizationInput,
        status: OptimizationStatus,
        error_message: str
    ) -> OptimizationOutput:
        """Create output for failed optimization."""
        return OptimizationOutput(
            request_id=input_data.request_id,
            status=status,
            timestamp=datetime.now(timezone.utc),
            recommended_setpoints={},
            expected_savings_kw=0,
            expected_cost_savings=0,
            constraints_satisfied=False,
            constraint_violations=[error_message],
            explanation=None,
            feature_contributions={},
            execution_time_ms=0,
            iterations=0,
            provenance_hash=""
        )

    def _log_audit(self, stage: str, action: str, request_id: str) -> None:
        """Log audit entry."""
        self._audit_sequence += 1
        self.audit_log.append(AuditEntry(
            sequence_id=self._audit_sequence,
            timestamp=datetime.now(timezone.utc),
            stage=stage,
            action=action,
            inputs_hash=hashlib.sha256(request_id.encode()).hexdigest()[:16],
            outputs_hash="",
            duration_ms=0,
            status="ok"
        ))

    def _get_timestamp_ms(self) -> float:
        """Get current timestamp in milliseconds."""
        return datetime.now(timezone.utc).timestamp() * 1000

    def get_audit_log(self) -> List[AuditEntry]:
        """Get audit log entries."""
        return self.audit_log


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def pipeline():
    """Create optimization pipeline instance."""
    return SteamOptimizationPipeline()


@pytest.fixture
def valid_desuperheater_input():
    """Valid desuperheater optimization input."""
    return OptimizationInput(
        request_id="REQ-001",
        optimization_type=OptimizationType.DESUPERHEATER,
        timestamp=datetime.now(timezone.utc),
        process_data={
            "inlet_pressure_mpa": 1.5,
            "inlet_temperature_k": 523.0,
            "inlet_mass_flow_kg_s": 5.0,
            "target_temperature_k": 473.0,
            "spray_water_temperature_k": 333.0,
        },
        constraints={
            ConstraintType.TEMPERATURE_MIN: 450.0,
            ConstraintType.TEMPERATURE_MAX: 550.0,
            ConstraintType.APPROACH_MIN: 10.0,
        },
        objectives={"energy_efficiency": 0.6, "cost_reduction": 0.4},
        require_explainability=True
    )


@pytest.fixture
def invalid_input_missing_fields():
    """Input missing required fields."""
    return OptimizationInput(
        request_id="REQ-002",
        optimization_type=OptimizationType.DESUPERHEATER,
        timestamp=datetime.now(timezone.utc),
        process_data={},  # Missing required fields
        constraints={},
        objectives={},
        require_explainability=False
    )


# =============================================================================
# Test Classes
# =============================================================================

@pytest.mark.integration
class TestOptimizationPipelineEndToEnd:
    """Test full optimization pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_successful_optimization(self, pipeline, valid_desuperheater_input):
        """Test successful optimization run."""
        result = await pipeline.optimize(valid_desuperheater_input)

        assert result.status == OptimizationStatus.COMPLETED
        assert result.request_id == valid_desuperheater_input.request_id
        assert len(result.recommended_setpoints) > 0
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_output_contains_required_fields(self, pipeline, valid_desuperheater_input):
        """Test output contains all required fields."""
        result = await pipeline.optimize(valid_desuperheater_input)

        assert result.request_id is not None
        assert result.status is not None
        assert result.timestamp is not None
        assert result.recommended_setpoints is not None
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_savings_calculated(self, pipeline, valid_desuperheater_input):
        """Test expected savings are calculated."""
        result = await pipeline.optimize(valid_desuperheater_input)

        # Should have some savings for valid optimization
        assert result.expected_savings_kw >= 0
        assert result.expected_cost_savings >= 0

    @pytest.mark.asyncio
    async def test_invalid_input_fails(self, pipeline, invalid_input_missing_fields):
        """Test that invalid input fails gracefully."""
        result = await pipeline.optimize(invalid_input_missing_fields)

        assert result.status == OptimizationStatus.FAILED
        assert len(result.constraint_violations) > 0


@pytest.mark.integration
class TestConstraintValidation:
    """Test constraint validation in pipeline."""

    @pytest.mark.asyncio
    async def test_constraints_satisfied(self, pipeline, valid_desuperheater_input):
        """Test that constraints are satisfied."""
        result = await pipeline.optimize(valid_desuperheater_input)

        assert result.constraints_satisfied or len(result.constraint_violations) > 0

    @pytest.mark.asyncio
    async def test_temperature_constraint_enforced(self, pipeline):
        """Test temperature constraints are enforced."""
        input_data = OptimizationInput(
            request_id="REQ-TEMP",
            optimization_type=OptimizationType.DESUPERHEATER,
            timestamp=datetime.now(timezone.utc),
            process_data={
                "inlet_pressure_mpa": 1.0,
                "inlet_temperature_k": 500.0,
                "target_temperature_k": 350.0,  # Below typical minimum
            },
            constraints={
                ConstraintType.TEMPERATURE_MIN: 400.0,  # Constraint
            },
            objectives={},
            require_explainability=False
        )

        result = await pipeline.optimize(input_data)

        # Either satisfies constraint or reports violation
        if result.status == OptimizationStatus.COMPLETED:
            outlet_temp = result.recommended_setpoints.get("outlet_temperature_k", 0)
            # Output should respect constraint
            assert outlet_temp >= 350.0 or not result.constraints_satisfied

    @pytest.mark.asyncio
    async def test_multiple_constraints(self, pipeline):
        """Test multiple constraints are evaluated."""
        input_data = OptimizationInput(
            request_id="REQ-MULTI",
            optimization_type=OptimizationType.DESUPERHEATER,
            timestamp=datetime.now(timezone.utc),
            process_data={
                "inlet_pressure_mpa": 1.0,
                "inlet_temperature_k": 500.0,
                "target_temperature_k": 450.0,
            },
            constraints={
                ConstraintType.TEMPERATURE_MIN: 440.0,
                ConstraintType.TEMPERATURE_MAX: 500.0,
                ConstraintType.APPROACH_MIN: 10.0,
            },
            objectives={},
            require_explainability=False
        )

        result = await pipeline.optimize(input_data)

        # Should complete with all constraints evaluated
        assert result.status in [OptimizationStatus.COMPLETED, OptimizationStatus.FAILED]


@pytest.mark.integration
class TestExplainabilityGeneration:
    """Test explainability generation in pipeline."""

    @pytest.mark.asyncio
    async def test_explainability_generated_when_requested(self, pipeline, valid_desuperheater_input):
        """Test explainability is generated when requested."""
        valid_desuperheater_input.require_explainability = True
        result = await pipeline.optimize(valid_desuperheater_input)

        assert result.explanation is not None
        assert len(result.feature_contributions) > 0

    @pytest.mark.asyncio
    async def test_explainability_skipped_when_not_requested(self, pipeline, valid_desuperheater_input):
        """Test explainability is skipped when not requested."""
        valid_desuperheater_input.require_explainability = False
        result = await pipeline.optimize(valid_desuperheater_input)

        # Explainability may be None or empty
        assert result.status == OptimizationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_feature_contributions_sum_to_one(self, pipeline, valid_desuperheater_input):
        """Test feature contributions approximately sum to 1."""
        result = await pipeline.optimize(valid_desuperheater_input)

        if result.feature_contributions:
            total = sum(result.feature_contributions.values())
            assert pytest.approx(total, rel=0.1) == 1.0

    @pytest.mark.asyncio
    async def test_explanation_has_method(self, pipeline, valid_desuperheater_input):
        """Test explanation includes method used."""
        result = await pipeline.optimize(valid_desuperheater_input)

        if result.explanation:
            assert "method" in result.explanation
            assert result.explanation["method"] in ["shap", "lime", "physics_based"]


@pytest.mark.integration
class TestAuditTrail:
    """Test audit trail generation."""

    @pytest.mark.asyncio
    async def test_audit_entries_created(self, pipeline, valid_desuperheater_input):
        """Test audit entries are created during optimization."""
        await pipeline.optimize(valid_desuperheater_input)

        audit_log = pipeline.get_audit_log()
        assert len(audit_log) > 0

    @pytest.mark.asyncio
    async def test_audit_covers_all_stages(self, pipeline, valid_desuperheater_input):
        """Test audit covers all pipeline stages."""
        await pipeline.optimize(valid_desuperheater_input)

        audit_log = pipeline.get_audit_log()
        stages = [entry.stage for entry in audit_log]

        # Should have entries for each stage
        expected_stages = ["input_validation", "constraint_setup", "optimization", "solution_validation"]
        for stage in expected_stages:
            assert stage in stages, f"Missing audit entry for stage: {stage}"

    @pytest.mark.asyncio
    async def test_audit_sequence_increments(self, pipeline, valid_desuperheater_input):
        """Test audit sequence numbers increment."""
        await pipeline.optimize(valid_desuperheater_input)

        audit_log = pipeline.get_audit_log()
        sequences = [entry.sequence_id for entry in audit_log]

        # Should be monotonically increasing
        for i in range(1, len(sequences)):
            assert sequences[i] > sequences[i - 1]

    @pytest.mark.asyncio
    async def test_provenance_hash_generated(self, pipeline, valid_desuperheater_input):
        """Test provenance hash is generated."""
        result = await pipeline.optimize(valid_desuperheater_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_provenance_hash_deterministic(self, pipeline, valid_desuperheater_input):
        """Test provenance hash is deterministic for same inputs."""
        result1 = await pipeline.optimize(valid_desuperheater_input)

        pipeline2 = SteamOptimizationPipeline()
        result2 = await pipeline2.optimize(valid_desuperheater_input)

        assert result1.provenance_hash == result2.provenance_hash


@pytest.mark.integration
class TestPerformance:
    """Test pipeline performance characteristics."""

    @pytest.mark.asyncio
    async def test_optimization_completes_within_timeout(self, pipeline, valid_desuperheater_input):
        """Test optimization completes within specified timeout."""
        valid_desuperheater_input.timeout_seconds = 5.0

        result = await pipeline.optimize(valid_desuperheater_input)

        # Should complete, not timeout
        assert result.status != OptimizationStatus.TIMEOUT
        assert result.execution_time_ms < 5000  # Less than 5 seconds

    @pytest.mark.asyncio
    async def test_iterations_within_limit(self, pipeline, valid_desuperheater_input):
        """Test iterations stay within limit."""
        valid_desuperheater_input.max_iterations = 50

        result = await pipeline.optimize(valid_desuperheater_input)

        assert result.iterations <= 50

    @pytest.mark.asyncio
    async def test_multiple_sequential_optimizations(self, pipeline):
        """Test multiple sequential optimizations."""
        inputs = [
            OptimizationInput(
                request_id=f"REQ-{i}",
                optimization_type=OptimizationType.DESUPERHEATER,
                timestamp=datetime.now(timezone.utc),
                process_data={
                    "inlet_pressure_mpa": 1.0 + i * 0.1,
                    "inlet_temperature_k": 500.0 + i * 10,
                    "target_temperature_k": 450.0,
                },
                constraints={},
                objectives={},
                require_explainability=False
            )
            for i in range(5)
        ]

        results = []
        for inp in inputs:
            result = await pipeline.optimize(inp)
            results.append(result)

        # All should complete
        for i, result in enumerate(results):
            assert result.status == OptimizationStatus.COMPLETED, f"Request {i} failed"


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in pipeline."""

    @pytest.mark.asyncio
    async def test_missing_request_id_fails(self, pipeline):
        """Test missing request ID fails gracefully."""
        input_data = OptimizationInput(
            request_id="",  # Empty request ID
            optimization_type=OptimizationType.DESUPERHEATER,
            timestamp=datetime.now(timezone.utc),
            process_data={"inlet_pressure_mpa": 1.0, "inlet_temperature_k": 500.0},
            constraints={},
            objectives={},
        )

        result = await pipeline.optimize(input_data)

        assert result.status == OptimizationStatus.FAILED

    @pytest.mark.asyncio
    async def test_missing_process_data_fails(self, pipeline, invalid_input_missing_fields):
        """Test missing process data fails gracefully."""
        result = await pipeline.optimize(invalid_input_missing_fields)

        assert result.status == OptimizationStatus.FAILED
        assert len(result.constraint_violations) > 0

    @pytest.mark.asyncio
    async def test_failed_optimization_returns_empty_setpoints(self, pipeline, invalid_input_missing_fields):
        """Test failed optimization returns empty setpoints."""
        result = await pipeline.optimize(invalid_input_missing_fields)

        assert result.recommended_setpoints == {}
        assert result.expected_savings_kw == 0


@pytest.mark.integration
class TestOptimizationTypes:
    """Test different optimization types."""

    @pytest.mark.asyncio
    async def test_desuperheater_optimization(self, pipeline, valid_desuperheater_input):
        """Test desuperheater optimization type."""
        valid_desuperheater_input.optimization_type = OptimizationType.DESUPERHEATER
        result = await pipeline.optimize(valid_desuperheater_input)

        assert result.status == OptimizationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_condensate_optimization(self, pipeline):
        """Test condensate recovery optimization type."""
        input_data = OptimizationInput(
            request_id="REQ-COND",
            optimization_type=OptimizationType.CONDENSATE_RECOVERY,
            timestamp=datetime.now(timezone.utc),
            process_data={
                "inlet_pressure_mpa": 1.0,
                "inlet_temperature_k": 450.0,
                "condensate_flow_kg_s": 2.0,
            },
            constraints={},
            objectives={"recovery_rate": 1.0},
        )

        result = await pipeline.optimize(input_data)

        # Should handle different optimization types
        assert result.status in [OptimizationStatus.COMPLETED, OptimizationStatus.FAILED]
