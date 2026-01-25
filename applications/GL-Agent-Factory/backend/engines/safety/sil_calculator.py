"""
SIL Calculator Module - Safety Integrity Level Determination

This module implements Safety Integrity Level (SIL) calculations per IEC 61508
and IEC 61511 standards. All calculations are deterministic with full
provenance tracking for regulatory audit compliance.

Standards:
    - IEC 61508: Functional Safety of Electrical/Electronic/PE Systems
    - IEC 61511: Safety Instrumented Systems for Process Industries

SIL Levels (Probability of Failure on Demand - PFD):
    - SIL 1: PFD 10^-1 to 10^-2 (0.1 to 0.01), RRF 10-100
    - SIL 2: PFD 10^-2 to 10^-3 (0.01 to 0.001), RRF 100-1000
    - SIL 3: PFD 10^-3 to 10^-4 (0.001 to 0.0001), RRF 1000-10000
    - SIL 4: PFD 10^-4 to 10^-5 (0.0001 to 0.00001), RRF 10000-100000

Example:
    >>> calc = SILCalculator()
    >>> result = calc.calculate_sil_level(
    ...     consequence_severity=3,
    ...     likelihood_without_protection=0.1,
    ...     target_risk_level=1e-4
    ... )
    >>> print(f"Required SIL: {result.sil_level}")
    >>> print(f"Required PFD: {result.required_pfd}")

CRITICAL: All calculations are DETERMINISTIC - NO LLM calls permitted.
"""

import hashlib
import json
import logging
import math
import time
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class SILLevel(IntEnum):
    """
    Safety Integrity Level per IEC 61508.

    Higher SIL levels require higher safety integrity (lower PFD).
    SIL 4 is the highest level and is rarely used in process industries.
    """
    NONE = 0  # No safety function required
    SIL_1 = 1  # PFD 0.1 to 0.01
    SIL_2 = 2  # PFD 0.01 to 0.001
    SIL_3 = 3  # PFD 0.001 to 0.0001
    SIL_4 = 4  # PFD 0.0001 to 0.00001


class VotingArchitecture(str, Enum):
    """
    Voting architectures for redundant safety systems.

    The naming convention is MooN where:
    - M = minimum number that must function
    - N = total number of channels
    """
    ONE_OO_ONE = "1oo1"      # Single channel (no redundancy)
    ONE_OO_TWO = "1oo2"      # Dual redundant (either can trip)
    TWO_OO_TWO = "2oo2"      # Dual (both must agree - high availability)
    ONE_OO_THREE = "1oo3"    # Triple redundant (any can trip)
    TWO_OO_THREE = "2oo3"    # Triple (2 of 3 must agree - TMR)
    TWO_OO_FOUR = "2oo4"     # Quad (2 of 4 must agree)


class ComponentType(str, Enum):
    """Types of safety components in a SIF."""
    SENSOR = "sensor"
    LOGIC_SOLVER = "logic_solver"
    FINAL_ELEMENT = "final_element"
    TRANSMITTER = "transmitter"
    VALVE = "valve"
    RELAY = "relay"
    PLC = "plc"
    SOLENOID = "solenoid"


class SafetyComponent(BaseModel):
    """
    Safety component with failure rate data.

    Attributes:
        component_id: Unique identifier for the component
        component_type: Type of component (sensor, valve, etc.)
        lambda_du: Dangerous undetected failure rate (per hour)
        lambda_dd: Dangerous detected failure rate (per hour)
        beta: Common cause failure factor (0-1)
        mttr: Mean time to repair (hours)
        proof_test_interval: Time between proof tests (hours)
        diagnostic_coverage: Fraction of dangerous failures detected (0-1)
    """
    component_id: str = Field(..., description="Unique component identifier")
    component_type: ComponentType = Field(..., description="Type of safety component")
    lambda_du: float = Field(..., ge=0, description="Dangerous undetected failure rate (per hour)")
    lambda_dd: float = Field(0.0, ge=0, description="Dangerous detected failure rate (per hour)")
    beta: float = Field(0.1, ge=0, le=1, description="Common cause failure factor")
    mttr: float = Field(8.0, gt=0, description="Mean time to repair (hours)")
    proof_test_interval: float = Field(8760.0, gt=0, description="Proof test interval (hours)")
    diagnostic_coverage: float = Field(0.0, ge=0, le=1, description="Diagnostic coverage (0-1)")
    manufacturer: Optional[str] = Field(None, description="Component manufacturer")
    model: Optional[str] = Field(None, description="Component model number")
    sff: Optional[float] = Field(None, ge=0, le=1, description="Safe Failure Fraction")

    @validator('lambda_du')
    def validate_lambda_du(cls, v):
        """Validate dangerous undetected failure rate is reasonable."""
        if v > 1e-3:
            logger.warning(f"High lambda_du value: {v}. Verify this is per-hour rate.")
        return v

    @validator('proof_test_interval')
    def validate_proof_test_interval(cls, v):
        """Validate proof test interval is reasonable."""
        if v > 87600:  # 10 years in hours
            raise ValueError("Proof test interval exceeds 10 years - verify value")
        return v


class SILCalculationStep(BaseModel):
    """Individual calculation step with provenance."""
    step_number: int = Field(..., description="Step sequence number")
    description: str = Field(..., description="Step description")
    formula: str = Field(..., description="Formula applied")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input values")
    output_name: str = Field(..., description="Output variable name")
    output_value: float = Field(..., description="Calculated value")
    source_reference: Optional[str] = Field(None, description="Standard reference")
    step_hash: str = Field("", description="SHA-256 hash of this step")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.step_hash:
            self.step_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of this calculation step."""
        hash_data = {
            "step_number": self.step_number,
            "formula": self.formula,
            "inputs": {k: str(v) for k, v in self.inputs.items()},
            "output_name": self.output_name,
            "output_value": str(self.output_value),
        }
        return hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()


class SILCalculationResult(BaseModel):
    """
    Complete SIL calculation result with provenance.

    Contains the determined SIL level, calculated PFD values,
    and full audit trail for regulatory compliance.
    """
    # Result values
    sil_level: SILLevel = Field(..., description="Determined SIL level")
    required_pfd: float = Field(..., description="Required PFD to meet target")
    achieved_pfd: Optional[float] = Field(None, description="Achieved PFD of design")
    required_rrf: float = Field(..., description="Required Risk Reduction Factor")

    # Calculation details
    target_risk_level: float = Field(..., description="Target tolerable risk level")
    unmitigated_risk: float = Field(..., description="Risk without safety function")
    risk_reduction_required: float = Field(..., description="Risk reduction required")

    # Architecture details
    architecture: Optional[VotingArchitecture] = Field(None, description="Voting architecture")
    component_count: int = Field(0, description="Number of components analyzed")

    # Provenance
    calculation_steps: List[SILCalculationStep] = Field(
        default_factory=list, description="Calculation audit trail"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash of complete calculation")
    input_hash: str = Field(..., description="SHA-256 hash of inputs")

    # Metadata
    calculation_time_ms: float = Field(..., description="Calculation time in ms")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    standard_reference: str = Field("IEC 61508/61511", description="Applicable standard")

    # Compliance flags
    sil_achievable: bool = Field(True, description="Whether SIL is achievable with design")
    warnings: List[str] = Field(default_factory=list, description="Calculation warnings")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            SILLevel: lambda v: v.name,
            VotingArchitecture: lambda v: v.value,
        }

    def verify_provenance(self) -> bool:
        """Verify the provenance hash matches calculation steps."""
        step_data = [
            {
                "step_number": s.step_number,
                "formula": s.formula,
                "inputs": {k: str(v) for k, v in s.inputs.items()},
                "output_value": str(s.output_value),
            }
            for s in self.calculation_steps
        ]
        recalculated = hashlib.sha256(
            json.dumps(step_data, sort_keys=True).encode()
        ).hexdigest()
        return recalculated == self.provenance_hash

    def to_audit_dict(self) -> Dict[str, Any]:
        """Export as audit-ready dictionary."""
        return {
            "result": {
                "sil_level": self.sil_level.name,
                "required_pfd": self.required_pfd,
                "achieved_pfd": self.achieved_pfd,
                "required_rrf": self.required_rrf,
            },
            "provenance": {
                "provenance_hash": self.provenance_hash,
                "input_hash": self.input_hash,
                "step_count": len(self.calculation_steps),
                "steps": [s.dict() for s in self.calculation_steps],
            },
            "metadata": {
                "calculated_at": self.calculated_at.isoformat(),
                "calculation_time_ms": self.calculation_time_ms,
                "standard_reference": self.standard_reference,
            },
            "compliance": {
                "sil_achievable": self.sil_achievable,
                "warnings": self.warnings,
            },
        }


class SILCalculator:
    """
    Safety Integrity Level Calculator per IEC 61508/61511.

    This class provides deterministic SIL level calculations with
    complete provenance tracking. All calculations follow IEC 61508
    formulas exactly.

    Key Methods:
        calculate_sil_level: Determine required SIL from risk parameters
        calculate_pfd: Calculate PFD for a component architecture
        verify_sil_capability: Verify if design can achieve target SIL

    Example:
        >>> calc = SILCalculator()
        >>> result = calc.calculate_sil_level(
        ...     consequence_severity=3,
        ...     likelihood_without_protection=0.1,
        ...     target_risk_level=1e-4
        ... )
        >>> print(f"SIL Level: {result.sil_level}")
        >>> print(f"Required PFD: {result.required_pfd}")

    CRITICAL: All calculations are DETERMINISTIC. NO LLM calls permitted.
    """

    VERSION = "1.0.0"

    # SIL Level PFD boundaries per IEC 61508 (low demand mode)
    SIL_PFD_BOUNDS: Dict[SILLevel, Tuple[float, float]] = {
        SILLevel.SIL_1: (1e-2, 1e-1),    # 0.01 to 0.1
        SILLevel.SIL_2: (1e-3, 1e-2),    # 0.001 to 0.01
        SILLevel.SIL_3: (1e-4, 1e-3),    # 0.0001 to 0.001
        SILLevel.SIL_4: (1e-5, 1e-4),    # 0.00001 to 0.0001
    }

    # Risk Reduction Factor ranges
    SIL_RRF_RANGES: Dict[SILLevel, Tuple[int, int]] = {
        SILLevel.SIL_1: (10, 100),
        SILLevel.SIL_2: (100, 1000),
        SILLevel.SIL_3: (1000, 10000),
        SILLevel.SIL_4: (10000, 100000),
    }

    # Beta factors for common cause failure by voting architecture
    BETA_FACTORS: Dict[VotingArchitecture, float] = {
        VotingArchitecture.ONE_OO_ONE: 0.0,
        VotingArchitecture.ONE_OO_TWO: 0.1,
        VotingArchitecture.TWO_OO_TWO: 0.1,
        VotingArchitecture.ONE_OO_THREE: 0.05,
        VotingArchitecture.TWO_OO_THREE: 0.05,
        VotingArchitecture.TWO_OO_FOUR: 0.02,
    }

    def __init__(self):
        """Initialize SIL Calculator."""
        self._steps: List[SILCalculationStep] = []
        self._step_counter = 0
        self._start_time: Optional[float] = None
        self._warnings: List[str] = []

    def _start_calculation(self) -> None:
        """Reset calculation state for new calculation."""
        self._steps = []
        self._step_counter = 0
        self._start_time = time.perf_counter()
        self._warnings = []

    def _record_step(
        self,
        description: str,
        formula: str,
        inputs: Dict[str, Any],
        output_name: str,
        output_value: float,
        source_reference: Optional[str] = None,
    ) -> SILCalculationStep:
        """Record a calculation step with provenance."""
        self._step_counter += 1
        step = SILCalculationStep(
            step_number=self._step_counter,
            description=description,
            formula=formula,
            inputs=inputs,
            output_name=output_name,
            output_value=output_value,
            source_reference=source_reference,
        )
        self._steps.append(step)
        logger.debug(f"Step {self._step_counter}: {description} = {output_value}")
        return step

    def _compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash of all calculation steps."""
        step_data = [
            {
                "step_number": s.step_number,
                "formula": s.formula,
                "inputs": {k: str(v) for k, v in s.inputs.items()},
                "output_value": str(s.output_value),
            }
            for s in self._steps
        ]
        return hashlib.sha256(
            json.dumps(step_data, sort_keys=True).encode()
        ).hexdigest()

    def _compute_input_hash(self, inputs: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of inputs."""
        serializable = {k: str(v) for k, v in inputs.items()}
        return hashlib.sha256(
            json.dumps(serializable, sort_keys=True).encode()
        ).hexdigest()

    def _get_calculation_time_ms(self) -> float:
        """Get calculation time in milliseconds."""
        if self._start_time is None:
            return 0.0
        return (time.perf_counter() - self._start_time) * 1000

    def calculate_sil_level(
        self,
        consequence_severity: int,
        likelihood_without_protection: float,
        target_risk_level: float,
    ) -> SILCalculationResult:
        """
        Calculate required SIL level from risk parameters.

        Uses the risk-based approach per IEC 61511 to determine the
        required Safety Integrity Level based on consequence severity,
        unmitigated likelihood, and target tolerable risk.

        Args:
            consequence_severity: Severity factor (1-4, where 4 is catastrophic)
            likelihood_without_protection: Annual frequency without protection (0-1)
            target_risk_level: Target tolerable risk level (e.g., 1e-4)

        Returns:
            SILCalculationResult with determined SIL and audit trail

        Raises:
            ValueError: If input parameters are invalid

        Example:
            >>> calc = SILCalculator()
            >>> result = calc.calculate_sil_level(
            ...     consequence_severity=3,
            ...     likelihood_without_protection=0.1,
            ...     target_risk_level=1e-4
            ... )
        """
        self._start_calculation()
        logger.info("Starting SIL level calculation")

        # Validate inputs
        if consequence_severity < 1 or consequence_severity > 4:
            raise ValueError("consequence_severity must be between 1 and 4")
        if likelihood_without_protection <= 0 or likelihood_without_protection > 1:
            raise ValueError("likelihood_without_protection must be > 0 and <= 1")
        if target_risk_level <= 0 or target_risk_level > 1:
            raise ValueError("target_risk_level must be > 0 and <= 1")

        inputs_summary = {
            "consequence_severity": consequence_severity,
            "likelihood_without_protection": likelihood_without_protection,
            "target_risk_level": target_risk_level,
        }

        # Step 1: Calculate unmitigated risk
        unmitigated_risk = consequence_severity * likelihood_without_protection
        self._record_step(
            description="Calculate unmitigated risk",
            formula="unmitigated_risk = consequence_severity * likelihood_without_protection",
            inputs={
                "consequence_severity": consequence_severity,
                "likelihood_without_protection": likelihood_without_protection,
            },
            output_name="unmitigated_risk",
            output_value=unmitigated_risk,
            source_reference="IEC 61511 Clause 9.4",
        )

        # Step 2: Calculate required risk reduction factor (RRF)
        required_rrf = unmitigated_risk / target_risk_level
        self._record_step(
            description="Calculate required Risk Reduction Factor",
            formula="RRF = unmitigated_risk / target_risk_level",
            inputs={
                "unmitigated_risk": unmitigated_risk,
                "target_risk_level": target_risk_level,
            },
            output_name="required_rrf",
            output_value=required_rrf,
            source_reference="IEC 61511 Clause 9.4",
        )

        # Step 3: Calculate required PFD (PFD = 1/RRF)
        required_pfd = 1.0 / required_rrf
        self._record_step(
            description="Calculate required Probability of Failure on Demand",
            formula="PFD = 1 / RRF",
            inputs={"required_rrf": required_rrf},
            output_name="required_pfd",
            output_value=required_pfd,
            source_reference="IEC 61508-1 Table 3",
        )

        # Step 4: Determine SIL level from PFD
        sil_level = self._pfd_to_sil(required_pfd)
        self._record_step(
            description="Determine SIL level from PFD",
            formula="SIL = lookup_table(PFD)",
            inputs={"required_pfd": required_pfd},
            output_name="sil_level",
            output_value=sil_level.value,
            source_reference="IEC 61508-1 Table 2",
        )

        # Step 5: Calculate risk reduction required
        risk_reduction_required = unmitigated_risk - target_risk_level
        self._record_step(
            description="Calculate risk reduction required",
            formula="risk_reduction = unmitigated_risk - target_risk_level",
            inputs={
                "unmitigated_risk": unmitigated_risk,
                "target_risk_level": target_risk_level,
            },
            output_name="risk_reduction_required",
            output_value=risk_reduction_required,
            source_reference="IEC 61511 Clause 9.4",
        )

        # Add warnings if needed
        if sil_level == SILLevel.SIL_4:
            self._warnings.append(
                "SIL 4 is rarely achievable in process industries. "
                "Consider additional non-SIS protection layers."
            )
        if required_rrf > 100000:
            self._warnings.append(
                f"Required RRF of {required_rrf:.0f} exceeds typical SIS capability. "
                "Multiple protection layers recommended."
            )

        # Build result
        result = SILCalculationResult(
            sil_level=sil_level,
            required_pfd=required_pfd,
            achieved_pfd=None,
            required_rrf=required_rrf,
            target_risk_level=target_risk_level,
            unmitigated_risk=unmitigated_risk,
            risk_reduction_required=risk_reduction_required,
            architecture=None,
            component_count=0,
            calculation_steps=self._steps.copy(),
            provenance_hash=self._compute_provenance_hash(),
            input_hash=self._compute_input_hash(inputs_summary),
            calculation_time_ms=self._get_calculation_time_ms(),
            sil_achievable=sil_level <= SILLevel.SIL_3,
            warnings=self._warnings.copy(),
        )

        logger.info(f"SIL calculation complete: SIL {sil_level.value}, PFD {required_pfd:.2e}")
        return result

    def calculate_pfd(
        self,
        components: List[SafetyComponent],
        architecture: VotingArchitecture,
    ) -> SILCalculationResult:
        """
        Calculate PFD for a safety function with given components.

        Calculates the Probability of Failure on Demand for a Safety
        Instrumented Function (SIF) based on component failure rates
        and voting architecture.

        Args:
            components: List of safety components with failure data
            architecture: Voting architecture (1oo1, 1oo2, 2oo3, etc.)

        Returns:
            SILCalculationResult with achieved PFD and SIL level

        Raises:
            ValueError: If components list is empty or architecture invalid

        Example:
            >>> calc = SILCalculator()
            >>> components = [
            ...     SafetyComponent(
            ...         component_id="PT-001",
            ...         component_type=ComponentType.SENSOR,
            ...         lambda_du=1e-6,
            ...         proof_test_interval=8760,
            ...     )
            ... ]
            >>> result = calc.calculate_pfd(components, VotingArchitecture.ONE_OO_TWO)
        """
        self._start_calculation()
        logger.info(f"Calculating PFD for {len(components)} components, {architecture.value}")

        if not components:
            raise ValueError("At least one component required for PFD calculation")

        inputs_summary = {
            "component_count": len(components),
            "architecture": architecture.value,
            "component_ids": [c.component_id for c in components],
        }

        # Step 1: Calculate individual component PFDs
        component_pfds: List[float] = []
        for i, comp in enumerate(components):
            pfd_i = self._calculate_component_pfd(comp, step_offset=i)
            component_pfds.append(pfd_i)

        # Step 2: Calculate system PFD based on architecture
        system_pfd = self._calculate_architecture_pfd(
            component_pfds,
            architecture,
            components[0].beta if components else 0.1,
        )

        # Step 3: Determine achieved SIL level
        achieved_sil = self._pfd_to_sil(system_pfd)
        self._record_step(
            description="Determine achieved SIL from system PFD",
            formula="SIL = lookup_table(system_PFD)",
            inputs={"system_pfd": system_pfd},
            output_name="achieved_sil",
            output_value=achieved_sil.value,
            source_reference="IEC 61508-1 Table 2",
        )

        # Calculate equivalent RRF
        rrf = 1.0 / system_pfd if system_pfd > 0 else float('inf')

        # Build result
        result = SILCalculationResult(
            sil_level=achieved_sil,
            required_pfd=0.0,  # Not applicable for achieved calculation
            achieved_pfd=system_pfd,
            required_rrf=rrf,
            target_risk_level=0.0,
            unmitigated_risk=0.0,
            risk_reduction_required=0.0,
            architecture=architecture,
            component_count=len(components),
            calculation_steps=self._steps.copy(),
            provenance_hash=self._compute_provenance_hash(),
            input_hash=self._compute_input_hash(inputs_summary),
            calculation_time_ms=self._get_calculation_time_ms(),
            sil_achievable=True,
            warnings=self._warnings.copy(),
        )

        logger.info(f"PFD calculation complete: {system_pfd:.2e}, SIL {achieved_sil.value}")
        return result

    def _calculate_component_pfd(
        self,
        component: SafetyComponent,
        step_offset: int = 0,
    ) -> float:
        """
        Calculate PFD for a single component per IEC 61508.

        Formula: PFDavg = lambda_DU * TI / 2 + lambda_DD * MTTR

        Where:
            lambda_DU = Dangerous undetected failure rate
            lambda_DD = Dangerous detected failure rate
            TI = Proof test interval
            MTTR = Mean time to repair
        """
        # PFD formula per IEC 61508-6
        # PFDavg = (lambda_DU * TI / 2) + (lambda_DD * MTTR)
        pfd_undetected = component.lambda_du * component.proof_test_interval / 2
        pfd_detected = component.lambda_dd * component.mttr
        pfd_total = pfd_undetected + pfd_detected

        self._record_step(
            description=f"Calculate PFD for component {component.component_id}",
            formula="PFD = (lambda_DU * TI / 2) + (lambda_DD * MTTR)",
            inputs={
                "component_id": component.component_id,
                "lambda_du": component.lambda_du,
                "lambda_dd": component.lambda_dd,
                "proof_test_interval": component.proof_test_interval,
                "mttr": component.mttr,
            },
            output_name=f"pfd_{component.component_id}",
            output_value=pfd_total,
            source_reference="IEC 61508-6 Annex B",
        )

        return pfd_total

    def _calculate_architecture_pfd(
        self,
        component_pfds: List[float],
        architecture: VotingArchitecture,
        beta: float,
    ) -> float:
        """
        Calculate system PFD based on voting architecture.

        Formulas per IEC 61508-6:
        - 1oo1: PFD_sys = PFD_1
        - 1oo2: PFD_sys = PFD_1 * PFD_2 + beta * (PFD_1 + PFD_2) / 2
        - 2oo3: PFD_sys = PFD_1 * PFD_2 + PFD_1 * PFD_3 + PFD_2 * PFD_3 - 2*PFD_1*PFD_2*PFD_3
        """
        n = len(component_pfds)
        pfd_sys: float = 0.0

        if architecture == VotingArchitecture.ONE_OO_ONE:
            # Single channel: PFD = PFD_1
            pfd_sys = component_pfds[0] if component_pfds else 0.0
            self._record_step(
                description="Calculate 1oo1 system PFD",
                formula="PFD_sys = PFD_1",
                inputs={"pfd_1": component_pfds[0] if component_pfds else 0},
                output_name="pfd_sys",
                output_value=pfd_sys,
                source_reference="IEC 61508-6 B.3.2",
            )

        elif architecture == VotingArchitecture.ONE_OO_TWO:
            # Dual redundant: Both must fail (parallel)
            # PFD = PFD_1 * PFD_2 + beta_D * (lambda_D * TI / 2)
            if n >= 2:
                pfd_1, pfd_2 = component_pfds[0], component_pfds[1]
                # Include common cause contribution
                pfd_independent = pfd_1 * pfd_2
                pfd_common = beta * (pfd_1 + pfd_2) / 2
                pfd_sys = pfd_independent + pfd_common
            else:
                pfd_sys = component_pfds[0] if component_pfds else 0.0

            self._record_step(
                description="Calculate 1oo2 system PFD with common cause",
                formula="PFD_sys = PFD_1 * PFD_2 + beta * (PFD_1 + PFD_2) / 2",
                inputs={
                    "pfd_1": component_pfds[0] if n >= 1 else 0,
                    "pfd_2": component_pfds[1] if n >= 2 else 0,
                    "beta": beta,
                },
                output_name="pfd_sys",
                output_value=pfd_sys,
                source_reference="IEC 61508-6 B.3.3",
            )

        elif architecture == VotingArchitecture.TWO_OO_TWO:
            # High availability: Either failure causes spurious trip
            # PFD for danger = 2 * PFD_1 * (1 - PFD_1) + PFD_1^2
            if n >= 2:
                pfd_avg = sum(component_pfds[:2]) / 2
                pfd_sys = 2 * pfd_avg - pfd_avg ** 2
            else:
                pfd_sys = component_pfds[0] if component_pfds else 0.0

            self._record_step(
                description="Calculate 2oo2 system PFD",
                formula="PFD_sys = 2 * PFD_avg - PFD_avg^2",
                inputs={"pfd_avg": pfd_avg if n >= 2 else 0},
                output_name="pfd_sys",
                output_value=pfd_sys,
                source_reference="IEC 61508-6 B.3.4",
            )

        elif architecture == VotingArchitecture.TWO_OO_THREE:
            # Triple modular redundancy: 2 of 3 must fail
            # PFD = 3 * PFD^2 - 2 * PFD^3 (for identical channels)
            if n >= 3:
                pfd_1, pfd_2, pfd_3 = component_pfds[0], component_pfds[1], component_pfds[2]
                # General formula for non-identical channels
                pfd_independent = (
                    pfd_1 * pfd_2 + pfd_1 * pfd_3 + pfd_2 * pfd_3
                    - 2 * pfd_1 * pfd_2 * pfd_3
                )
                pfd_common = beta * (pfd_1 + pfd_2 + pfd_3) / 3
                pfd_sys = pfd_independent + pfd_common
            else:
                pfd_sys = component_pfds[0] if component_pfds else 0.0

            self._record_step(
                description="Calculate 2oo3 system PFD (TMR)",
                formula="PFD_sys = PFD_12 + PFD_13 + PFD_23 - 2*PFD_123 + beta*avg(PFD)",
                inputs={
                    "pfd_1": component_pfds[0] if n >= 1 else 0,
                    "pfd_2": component_pfds[1] if n >= 2 else 0,
                    "pfd_3": component_pfds[2] if n >= 3 else 0,
                    "beta": beta,
                },
                output_name="pfd_sys",
                output_value=pfd_sys,
                source_reference="IEC 61508-6 B.3.3.4",
            )

        elif architecture == VotingArchitecture.ONE_OO_THREE:
            # Any single failure causes trip
            if n >= 3:
                pfd_1, pfd_2, pfd_3 = component_pfds[0], component_pfds[1], component_pfds[2]
                pfd_sys = pfd_1 * pfd_2 * pfd_3 + beta * (pfd_1 + pfd_2 + pfd_3) / 3
            else:
                pfd_sys = component_pfds[0] if component_pfds else 0.0

            self._record_step(
                description="Calculate 1oo3 system PFD",
                formula="PFD_sys = PFD_1 * PFD_2 * PFD_3 + beta * avg(PFD)",
                inputs={
                    "pfd_1": component_pfds[0] if n >= 1 else 0,
                    "pfd_2": component_pfds[1] if n >= 2 else 0,
                    "pfd_3": component_pfds[2] if n >= 3 else 0,
                    "beta": beta,
                },
                output_name="pfd_sys",
                output_value=pfd_sys,
                source_reference="IEC 61508-6 B.3.3.2",
            )

        elif architecture == VotingArchitecture.TWO_OO_FOUR:
            # 2 of 4 must fail
            if n >= 4:
                pfd_avg = sum(component_pfds[:4]) / 4
                # Simplified formula for identical channels
                pfd_sys = 6 * (pfd_avg ** 2) - 8 * (pfd_avg ** 3) + 3 * (pfd_avg ** 4)
                pfd_sys += beta * pfd_avg  # Common cause
            else:
                pfd_sys = component_pfds[0] if component_pfds else 0.0

            self._record_step(
                description="Calculate 2oo4 system PFD",
                formula="PFD_sys = 6*PFD^2 - 8*PFD^3 + 3*PFD^4 + beta*PFD",
                inputs={"pfd_avg": pfd_avg if n >= 4 else 0, "beta": beta},
                output_name="pfd_sys",
                output_value=pfd_sys,
                source_reference="IEC 61508-6 B.3.3",
            )

        return pfd_sys

    def _pfd_to_sil(self, pfd: float) -> SILLevel:
        """
        Convert PFD to SIL level per IEC 61508 Table 2.

        Args:
            pfd: Probability of Failure on Demand

        Returns:
            Corresponding SIL level
        """
        if pfd >= 1e-1:
            return SILLevel.NONE
        elif pfd >= 1e-2:
            return SILLevel.SIL_1
        elif pfd >= 1e-3:
            return SILLevel.SIL_2
        elif pfd >= 1e-4:
            return SILLevel.SIL_3
        else:
            return SILLevel.SIL_4

    def verify_sil_capability(
        self,
        target_sil: SILLevel,
        components: List[SafetyComponent],
        architecture: VotingArchitecture,
    ) -> Tuple[bool, SILCalculationResult]:
        """
        Verify if a design can achieve target SIL level.

        Args:
            target_sil: Required SIL level
            components: Safety components in the design
            architecture: Voting architecture

        Returns:
            Tuple of (can_achieve, calculation_result)
        """
        result = self.calculate_pfd(components, architecture)

        can_achieve = result.sil_level >= target_sil

        if not can_achieve:
            self._warnings.append(
                f"Design achieves SIL {result.sil_level.value}, "
                f"target is SIL {target_sil.value}. "
                f"Consider improving component reliability or architecture."
            )
            result.warnings = self._warnings.copy()
            result.sil_achievable = False

        return can_achieve, result

    def get_sil_pfd_range(self, sil_level: SILLevel) -> Tuple[float, float]:
        """
        Get the PFD range for a SIL level.

        Args:
            sil_level: The SIL level

        Returns:
            Tuple of (lower_bound, upper_bound) PFD values
        """
        if sil_level == SILLevel.NONE:
            return (1e-1, 1.0)
        return self.SIL_PFD_BOUNDS.get(sil_level, (1.0, 1.0))

    def get_sil_rrf_range(self, sil_level: SILLevel) -> Tuple[int, int]:
        """
        Get the RRF range for a SIL level.

        Args:
            sil_level: The SIL level

        Returns:
            Tuple of (lower_bound, upper_bound) RRF values
        """
        if sil_level == SILLevel.NONE:
            return (1, 10)
        return self.SIL_RRF_RANGES.get(sil_level, (1, 10))
