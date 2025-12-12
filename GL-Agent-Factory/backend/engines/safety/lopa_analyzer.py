"""
LOPA Analyzer Module - Layer of Protection Analysis

This module implements Layer of Protection Analysis (LOPA) per IEC 61511
and CCPS guidelines. LOPA is a semi-quantitative method for evaluating
the adequacy of protection layers against specific accident scenarios.

Standards:
    - IEC 61511: Safety Instrumented Systems for Process Industries
    - CCPS LOPA Guidelines: Guidelines for Enabling Conditions and
      Conditional Modifiers in Layer of Protection Analysis

Key Concepts:
    - Initiating Event (IE): Event that can lead to hazardous scenario
    - Independent Protection Layer (IPL): System that can prevent
      scenario progression independent of initiating event
    - Conditional Modifier: Factors that affect frequency
    - Target Mitigated Event Likelihood (TMEL): Tolerable frequency

Example:
    >>> analyzer = LOPAAnalyzer()
    >>> result = analyzer.analyze_scenario(
    ...     initiating_event=InitiatingEvent(name="Valve failure", frequency=0.1),
    ...     independent_protection_layers=[
    ...         IPL(name="BPCS", pfd=0.1),
    ...         IPL(name="Alarm", pfd=0.1),
    ...         IPL(name="SIF", pfd=0.01),
    ...     ],
    ...     target_mitigated_frequency=1e-5
    ... )
    >>> print(f"Gap: {result.gap}")

CRITICAL: All calculations are DETERMINISTIC - NO LLM calls permitted.
"""

import hashlib
import json
import logging
import math
import time
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class IPLType(str, Enum):
    """Types of Independent Protection Layers."""
    BPCS = "bpcs"                        # Basic Process Control System
    ALARM = "alarm"                       # Operator response to alarm
    SIF = "sif"                          # Safety Instrumented Function
    RELIEF_DEVICE = "relief_device"       # Pressure relief valve/disk
    MECHANICAL = "mechanical"             # Mechanical interlock
    DIKE = "dike"                        # Containment dike
    FIRE_PROTECTION = "fire_protection"   # Fire suppression system
    DELUGE = "deluge"                    # Water deluge system
    OPERATOR = "operator"                 # Routine operator action
    ADMINISTRATIVE = "administrative"     # Administrative control
    INHERENT = "inherent"                # Inherent safety feature


class ConditionalModifierType(str, Enum):
    """Types of conditional modifiers in LOPA."""
    PROBABILITY_OF_IGNITION = "p_ignition"
    PROBABILITY_OF_PERSONNEL = "p_personnel"
    PROBABILITY_OF_FATAL_INJURY = "p_fatal"
    PROBABILITY_OF_WEATHER = "p_weather"
    DEMAND_RATE = "demand_rate"
    ENABLING_CONDITION = "enabling_condition"


class InitiatingEvent(BaseModel):
    """
    Initiating Event for LOPA analysis.

    An initiating event is the first event in a sequence that could
    lead to a hazardous scenario if not interrupted by protection layers.

    Attributes:
        event_id: Unique identifier
        name: Descriptive name of the event
        frequency: Annual frequency of occurrence (per year)
        description: Detailed description
        source_reference: Source of frequency data
    """
    event_id: str = Field(default="", description="Unique event identifier")
    name: str = Field(..., description="Event name")
    frequency: float = Field(..., gt=0, description="Annual frequency (per year)")
    description: Optional[str] = Field(None, description="Event description")
    source_reference: Optional[str] = Field(None, description="Data source")
    category: Optional[str] = Field(None, description="Event category")

    @validator('frequency')
    def validate_frequency(cls, v):
        """Validate frequency is reasonable."""
        if v > 10:
            logger.warning(f"High initiating event frequency: {v}/year")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        if not self.event_id:
            self.event_id = f"IE-{hash(self.name) % 10000:04d}"


class IPL(BaseModel):
    """
    Independent Protection Layer.

    An IPL must be:
    - Independent: Functions independently of the initiating event
    - Effective: Prevents the consequence when functioning
    - Auditable: Has verifiable PFD
    - Specific: Designed for the specific scenario

    Typical PFD values:
    - BPCS: 0.1 (10^-1)
    - Operator response: 0.1 (10^-1)
    - SIF SIL 1: 0.1 to 0.01
    - SIF SIL 2: 0.01 to 0.001
    - SIF SIL 3: 0.001 to 0.0001
    - Relief device: 0.01 to 0.001
    """
    ipl_id: str = Field(default="", description="Unique IPL identifier")
    name: str = Field(..., description="IPL name")
    ipl_type: IPLType = Field(..., description="Type of IPL")
    pfd: float = Field(..., gt=0, le=1, description="Probability of Failure on Demand")
    description: Optional[str] = Field(None, description="IPL description")
    is_independent: bool = Field(True, description="Independence verified")
    is_auditable: bool = Field(True, description="PFD is verifiable")
    proof_test_interval: Optional[float] = Field(None, description="Test interval (hours)")
    source_reference: Optional[str] = Field(None, description="PFD data source")

    @validator('pfd')
    def validate_pfd(cls, v, values):
        """Validate PFD is appropriate for IPL type."""
        ipl_type = values.get('ipl_type')
        if ipl_type == IPLType.OPERATOR and v < 0.01:
            logger.warning(
                f"Operator response PFD of {v} is optimistic. "
                "CCPS recommends >= 0.1 without specific justification."
            )
        return v

    def __init__(self, **data):
        super().__init__(**data)
        if not self.ipl_id:
            self.ipl_id = f"IPL-{hash(self.name) % 10000:04d}"


class ConditionalModifier(BaseModel):
    """
    Conditional modifier for LOPA frequency adjustment.

    Conditional modifiers adjust the frequency based on probability
    of specific conditions being present (e.g., personnel in area,
    ignition source present).
    """
    modifier_id: str = Field(default="", description="Modifier identifier")
    name: str = Field(..., description="Modifier name")
    modifier_type: ConditionalModifierType = Field(..., description="Type of modifier")
    probability: float = Field(..., gt=0, le=1, description="Probability (0-1)")
    description: Optional[str] = Field(None, description="Modifier description")
    source_reference: Optional[str] = Field(None, description="Data source")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.modifier_id:
            self.modifier_id = f"CM-{hash(self.name) % 10000:04d}"


class LOPAScenario(BaseModel):
    """
    Complete LOPA scenario definition.

    A LOPA scenario includes the initiating event, all protection
    layers, conditional modifiers, and target frequency.
    """
    scenario_id: str = Field(..., description="Unique scenario identifier")
    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    initiating_event: InitiatingEvent = Field(..., description="Initiating event")
    ipls: List[IPL] = Field(default_factory=list, description="Protection layers")
    conditional_modifiers: List[ConditionalModifier] = Field(
        default_factory=list, description="Conditional modifiers"
    )
    consequence_description: str = Field(..., description="Potential consequence")
    consequence_severity: int = Field(..., ge=1, le=5, description="Severity (1-5)")
    target_mitigated_frequency: float = Field(
        ..., gt=0, description="Target frequency (per year)"
    )


class LOPACalculationStep(BaseModel):
    """Individual LOPA calculation step with provenance."""
    step_number: int = Field(..., description="Step sequence number")
    description: str = Field(..., description="Step description")
    formula: str = Field(..., description="Formula applied")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input values")
    output_name: str = Field(..., description="Output variable name")
    output_value: float = Field(..., description="Calculated value")
    source_reference: Optional[str] = Field(None, description="Standard reference")
    step_hash: str = Field("", description="SHA-256 hash")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.step_hash:
            self.step_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of this step."""
        hash_data = {
            "step_number": self.step_number,
            "formula": self.formula,
            "inputs": {k: str(v) for k, v in self.inputs.items()},
            "output_value": str(self.output_value),
        }
        return hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()


class LOPAResult(BaseModel):
    """
    Complete LOPA analysis result with provenance.

    Contains the mitigated frequency, gap analysis, and full
    audit trail for regulatory compliance.
    """
    # Scenario identification
    scenario_id: str = Field(..., description="Scenario identifier")
    scenario_name: str = Field(..., description="Scenario name")

    # Frequency values
    initiating_event_frequency: float = Field(
        ..., description="Initial event frequency (per year)"
    )
    conditional_modifier_product: float = Field(
        1.0, description="Product of conditional modifiers"
    )
    ipl_pfd_product: float = Field(..., description="Product of IPL PFDs")
    mitigated_frequency: float = Field(
        ..., description="Final mitigated frequency (per year)"
    )
    target_frequency: float = Field(..., description="Target frequency (per year)")

    # Gap analysis
    gap: Optional[float] = Field(None, description="Gap to target (negative = OK)")
    gap_in_orders_of_magnitude: Optional[float] = Field(
        None, description="Gap in orders of magnitude"
    )
    additional_protection_required: bool = Field(
        ..., description="Whether additional IPLs needed"
    )
    suggested_sil: Optional[int] = Field(
        None, description="Suggested SIL for additional SIF"
    )

    # IPL summary
    ipl_count: int = Field(..., description="Number of IPLs")
    total_risk_reduction: float = Field(..., description="Total risk reduction factor")

    # Provenance
    calculation_steps: List[LOPACalculationStep] = Field(
        default_factory=list, description="Calculation audit trail"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash")
    input_hash: str = Field(..., description="Input hash")

    # Metadata
    calculation_time_ms: float = Field(..., description="Calculation time")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    standard_reference: str = Field("IEC 61511 / CCPS LOPA", description="Standard")

    # Compliance
    meets_target: bool = Field(..., description="Whether scenario meets target")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def verify_provenance(self) -> bool:
        """Verify provenance hash matches calculation steps."""
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
            "scenario": {
                "id": self.scenario_id,
                "name": self.scenario_name,
            },
            "frequencies": {
                "initiating_event": self.initiating_event_frequency,
                "conditional_modifier_product": self.conditional_modifier_product,
                "ipl_pfd_product": self.ipl_pfd_product,
                "mitigated": self.mitigated_frequency,
                "target": self.target_frequency,
            },
            "gap_analysis": {
                "gap": self.gap,
                "gap_orders_of_magnitude": self.gap_in_orders_of_magnitude,
                "additional_protection_required": self.additional_protection_required,
                "suggested_sil": self.suggested_sil,
                "meets_target": self.meets_target,
            },
            "provenance": {
                "hash": self.provenance_hash,
                "input_hash": self.input_hash,
                "step_count": len(self.calculation_steps),
            },
            "metadata": {
                "calculated_at": self.calculated_at.isoformat(),
                "calculation_time_ms": self.calculation_time_ms,
                "standard": self.standard_reference,
            },
        }


class LOPAAnalyzer:
    """
    Layer of Protection Analysis Analyzer per IEC 61511.

    This class performs LOPA calculations to determine if existing
    protection layers are sufficient to meet target risk levels,
    and if not, what additional protection is needed.

    Key Methods:
        analyze_scenario: Perform complete LOPA for a scenario
        calculate_gap: Determine protection gap
        suggest_additional_ipl: Recommend additional protection

    Example:
        >>> analyzer = LOPAAnalyzer()
        >>> result = analyzer.analyze_scenario(
        ...     initiating_event=InitiatingEvent(name="Loss of cooling", frequency=0.1),
        ...     independent_protection_layers=[
        ...         IPL(name="BPCS", ipl_type=IPLType.BPCS, pfd=0.1),
        ...         IPL(name="High temp alarm", ipl_type=IPLType.ALARM, pfd=0.1),
        ...     ],
        ...     target_mitigated_frequency=1e-4
        ... )
        >>> if result.additional_protection_required:
        ...     print(f"Need SIF with SIL {result.suggested_sil}")

    CRITICAL: All calculations are DETERMINISTIC. NO LLM calls permitted.
    """

    VERSION = "1.0.0"

    # Default PFD values per CCPS guidelines
    DEFAULT_PFD: Dict[IPLType, float] = {
        IPLType.BPCS: 0.1,
        IPLType.ALARM: 0.1,
        IPLType.SIF: 0.01,  # SIL 2 default
        IPLType.RELIEF_DEVICE: 0.01,
        IPLType.MECHANICAL: 0.01,
        IPLType.DIKE: 0.01,
        IPLType.FIRE_PROTECTION: 0.1,
        IPLType.DELUGE: 0.05,
        IPLType.OPERATOR: 0.1,
        IPLType.ADMINISTRATIVE: 1.0,  # Not credited as IPL
        IPLType.INHERENT: 0.0,  # Infinite risk reduction
    }

    # Target frequencies by consequence severity (per CCPS)
    DEFAULT_TARGETS: Dict[int, float] = {
        1: 1e-2,   # Minor injury
        2: 1e-3,   # Serious injury
        3: 1e-4,   # Single fatality
        4: 1e-5,   # Multiple fatalities
        5: 1e-6,   # Catastrophic
    }

    def __init__(self):
        """Initialize LOPA Analyzer."""
        self._steps: List[LOPACalculationStep] = []
        self._step_counter = 0
        self._start_time: Optional[float] = None
        self._warnings: List[str] = []
        self._recommendations: List[str] = []

    def _start_calculation(self) -> None:
        """Reset calculation state."""
        self._steps = []
        self._step_counter = 0
        self._start_time = time.perf_counter()
        self._warnings = []
        self._recommendations = []

    def _record_step(
        self,
        description: str,
        formula: str,
        inputs: Dict[str, Any],
        output_name: str,
        output_value: float,
        source_reference: Optional[str] = None,
    ) -> LOPACalculationStep:
        """Record a calculation step with provenance."""
        self._step_counter += 1
        step = LOPACalculationStep(
            step_number=self._step_counter,
            description=description,
            formula=formula,
            inputs=inputs,
            output_name=output_name,
            output_value=output_value,
            source_reference=source_reference,
        )
        self._steps.append(step)
        logger.debug(f"LOPA Step {self._step_counter}: {description} = {output_value}")
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
        serializable = {}
        for k, v in inputs.items():
            if isinstance(v, (list, dict)):
                serializable[k] = json.dumps(v, default=str)
            else:
                serializable[k] = str(v)
        return hashlib.sha256(
            json.dumps(serializable, sort_keys=True).encode()
        ).hexdigest()

    def _get_calculation_time_ms(self) -> float:
        """Get calculation time in milliseconds."""
        if self._start_time is None:
            return 0.0
        return (time.perf_counter() - self._start_time) * 1000

    def analyze_scenario(
        self,
        initiating_event: InitiatingEvent,
        independent_protection_layers: List[IPL],
        target_mitigated_frequency: float,
        conditional_modifiers: Optional[List[ConditionalModifier]] = None,
        scenario_id: Optional[str] = None,
        scenario_name: Optional[str] = None,
    ) -> LOPAResult:
        """
        Perform complete LOPA analysis for a scenario.

        Calculates the mitigated event frequency and determines if
        existing protection layers are sufficient to meet the target.

        Formula:
            f_mitigated = f_IE x Product(CM) x Product(PFD_IPL)

        Where:
            f_IE = Initiating event frequency
            CM = Conditional modifiers
            PFD_IPL = Probability of failure on demand for each IPL

        Args:
            initiating_event: The initiating event with frequency
            independent_protection_layers: List of IPLs credited
            target_mitigated_frequency: Target frequency (per year)
            conditional_modifiers: Optional conditional modifiers
            scenario_id: Optional scenario identifier
            scenario_name: Optional scenario name

        Returns:
            LOPAResult with complete analysis and provenance

        Raises:
            ValueError: If inputs are invalid
        """
        self._start_calculation()
        logger.info(f"Starting LOPA analysis for: {initiating_event.name}")

        # Validate inputs
        if target_mitigated_frequency <= 0:
            raise ValueError("target_mitigated_frequency must be positive")

        conditional_modifiers = conditional_modifiers or []
        scenario_id = scenario_id or f"LOPA-{hash(initiating_event.name) % 10000:04d}"
        scenario_name = scenario_name or f"LOPA: {initiating_event.name}"

        inputs_summary = {
            "initiating_event": initiating_event.name,
            "ie_frequency": initiating_event.frequency,
            "ipl_count": len(independent_protection_layers),
            "ipl_names": [ipl.name for ipl in independent_protection_layers],
            "modifier_count": len(conditional_modifiers),
            "target_frequency": target_mitigated_frequency,
        }

        # Step 1: Record initiating event frequency
        ie_freq = initiating_event.frequency
        self._record_step(
            description=f"Initiating event frequency: {initiating_event.name}",
            formula="f_IE = given",
            inputs={"event": initiating_event.name},
            output_name="f_ie",
            output_value=ie_freq,
            source_reference=initiating_event.source_reference or "Site data",
        )

        # Step 2: Calculate conditional modifier product
        cm_product = 1.0
        for cm in conditional_modifiers:
            cm_product *= cm.probability
            self._record_step(
                description=f"Apply conditional modifier: {cm.name}",
                formula="cm_product *= probability",
                inputs={
                    "modifier": cm.name,
                    "probability": cm.probability,
                    "running_product": cm_product / cm.probability,
                },
                output_name="cm_product",
                output_value=cm_product,
                source_reference=cm.source_reference or "CCPS LOPA Guidelines",
            )

        if conditional_modifiers:
            self._record_step(
                description="Total conditional modifier product",
                formula="CM_total = Product(CM_i)",
                inputs={"modifier_count": len(conditional_modifiers)},
                output_name="cm_total",
                output_value=cm_product,
                source_reference="CCPS LOPA Guidelines",
            )

        # Step 3: Validate IPL independence
        self._validate_ipl_independence(independent_protection_layers)

        # Step 4: Calculate IPL PFD product
        ipl_product = 1.0
        for ipl in independent_protection_layers:
            ipl_product *= ipl.pfd
            self._record_step(
                description=f"Apply IPL: {ipl.name} ({ipl.ipl_type.value})",
                formula="ipl_product *= PFD",
                inputs={
                    "ipl": ipl.name,
                    "type": ipl.ipl_type.value,
                    "pfd": ipl.pfd,
                    "running_product": ipl_product / ipl.pfd,
                },
                output_name="ipl_product",
                output_value=ipl_product,
                source_reference=ipl.source_reference or "IEC 61511",
            )

        self._record_step(
            description="Total IPL PFD product",
            formula="IPL_total = Product(PFD_i)",
            inputs={"ipl_count": len(independent_protection_layers)},
            output_name="ipl_total",
            output_value=ipl_product,
            source_reference="IEC 61511 Clause 9",
        )

        # Step 5: Calculate mitigated frequency
        mitigated_freq = ie_freq * cm_product * ipl_product
        self._record_step(
            description="Calculate mitigated event frequency",
            formula="f_mitigated = f_IE x CM_total x IPL_total",
            inputs={
                "f_ie": ie_freq,
                "cm_total": cm_product,
                "ipl_total": ipl_product,
            },
            output_name="f_mitigated",
            output_value=mitigated_freq,
            source_reference="IEC 61511 Clause 9.4",
        )

        # Step 6: Calculate gap
        gap = mitigated_freq - target_mitigated_frequency
        gap_orders = math.log10(mitigated_freq / target_mitigated_frequency) if mitigated_freq > 0 else 0

        self._record_step(
            description="Calculate gap to target",
            formula="gap = f_mitigated - f_target",
            inputs={
                "f_mitigated": mitigated_freq,
                "f_target": target_mitigated_frequency,
            },
            output_name="gap",
            output_value=gap,
            source_reference="IEC 61511",
        )

        # Step 7: Determine if additional protection required
        additional_required = mitigated_freq > target_mitigated_frequency
        meets_target = not additional_required

        # Step 8: Calculate suggested SIL if gap exists
        suggested_sil: Optional[int] = None
        if additional_required:
            required_rrf = mitigated_freq / target_mitigated_frequency
            suggested_sil = self._rrf_to_sil(required_rrf)

            self._record_step(
                description="Calculate required additional RRF",
                formula="required_RRF = f_mitigated / f_target",
                inputs={
                    "f_mitigated": mitigated_freq,
                    "f_target": target_mitigated_frequency,
                },
                output_name="required_rrf",
                output_value=required_rrf,
                source_reference="IEC 61511",
            )

            self._recommendations.append(
                f"Add SIF with SIL {suggested_sil} (RRF >= {required_rrf:.0f}) "
                f"to close gap of {gap_orders:.1f} orders of magnitude"
            )

        # Calculate total risk reduction
        total_rrf = 1.0 / ipl_product if ipl_product > 0 else float('inf')

        # Build result
        result = LOPAResult(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            initiating_event_frequency=ie_freq,
            conditional_modifier_product=cm_product,
            ipl_pfd_product=ipl_product,
            mitigated_frequency=mitigated_freq,
            target_frequency=target_mitigated_frequency,
            gap=gap,
            gap_in_orders_of_magnitude=gap_orders,
            additional_protection_required=additional_required,
            suggested_sil=suggested_sil,
            ipl_count=len(independent_protection_layers),
            total_risk_reduction=total_rrf,
            calculation_steps=self._steps.copy(),
            provenance_hash=self._compute_provenance_hash(),
            input_hash=self._compute_input_hash(inputs_summary),
            calculation_time_ms=self._get_calculation_time_ms(),
            meets_target=meets_target,
            warnings=self._warnings.copy(),
            recommendations=self._recommendations.copy(),
        )

        logger.info(
            f"LOPA complete: mitigated={mitigated_freq:.2e}, "
            f"target={target_mitigated_frequency:.2e}, meets_target={meets_target}"
        )
        return result

    def calculate_gap(
        self,
        current_mitigated_frequency: float,
        target_frequency: float,
    ) -> Optional[float]:
        """
        Calculate the gap between current and target frequency.

        Args:
            current_mitigated_frequency: Current mitigated frequency
            target_frequency: Target tolerable frequency

        Returns:
            Gap value (positive means additional protection needed)
            None if inputs invalid
        """
        if current_mitigated_frequency <= 0 or target_frequency <= 0:
            return None

        gap = current_mitigated_frequency - target_frequency
        return gap

    def calculate_required_sif_pfd(
        self,
        current_mitigated_frequency: float,
        target_frequency: float,
    ) -> Tuple[float, int]:
        """
        Calculate required SIF PFD to close gap.

        Args:
            current_mitigated_frequency: Current frequency
            target_frequency: Target frequency

        Returns:
            Tuple of (required_pfd, suggested_sil)
        """
        if current_mitigated_frequency <= target_frequency:
            return (1.0, 0)  # No SIF needed

        required_rrf = current_mitigated_frequency / target_frequency
        required_pfd = 1.0 / required_rrf
        suggested_sil = self._rrf_to_sil(required_rrf)

        return (required_pfd, suggested_sil)

    def _validate_ipl_independence(self, ipls: List[IPL]) -> None:
        """
        Validate IPL independence requirements.

        Per IEC 61511, IPLs must be independent of:
        - The initiating event
        - Other credited IPLs
        """
        # Check for multiple BPCS credits (common mistake)
        bpcs_count = sum(1 for ipl in ipls if ipl.ipl_type == IPLType.BPCS)
        if bpcs_count > 1:
            self._warnings.append(
                f"Multiple BPCS IPLs credited ({bpcs_count}). "
                "Verify independence per IEC 61511 Clause 9.3."
            )

        # Check for operator IPLs
        operator_count = sum(
            1 for ipl in ipls
            if ipl.ipl_type in (IPLType.ALARM, IPLType.OPERATOR)
        )
        if operator_count > 2:
            self._warnings.append(
                f"Multiple operator-dependent IPLs credited ({operator_count}). "
                "Consider operator reliability limitations."
            )

        # Check for administrative controls
        admin_ipls = [ipl for ipl in ipls if ipl.ipl_type == IPLType.ADMINISTRATIVE]
        if admin_ipls:
            self._warnings.append(
                "Administrative controls credited as IPLs. "
                "Per CCPS, administrative controls generally should not be "
                "credited as IPLs without specific justification."
            )

        # Check for independence flags
        non_independent = [ipl for ipl in ipls if not ipl.is_independent]
        if non_independent:
            names = [ipl.name for ipl in non_independent]
            self._warnings.append(
                f"IPLs marked as not independent: {names}. "
                "Verify these should be credited."
            )

    def _rrf_to_sil(self, rrf: float) -> int:
        """Convert Risk Reduction Factor to SIL level."""
        if rrf < 10:
            return 0
        elif rrf < 100:
            return 1
        elif rrf < 1000:
            return 2
        elif rrf < 10000:
            return 3
        else:
            return 4

    def analyze_full_scenario(self, scenario: LOPAScenario) -> LOPAResult:
        """
        Analyze a complete LOPA scenario object.

        Convenience method that unpacks LOPAScenario into analysis.

        Args:
            scenario: Complete scenario definition

        Returns:
            LOPAResult with analysis
        """
        return self.analyze_scenario(
            initiating_event=scenario.initiating_event,
            independent_protection_layers=scenario.ipls,
            target_mitigated_frequency=scenario.target_mitigated_frequency,
            conditional_modifiers=scenario.conditional_modifiers,
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
        )

    def get_default_target(self, consequence_severity: int) -> float:
        """
        Get default target frequency for consequence severity.

        Args:
            consequence_severity: Severity level (1-5)

        Returns:
            Default target frequency per year
        """
        return self.DEFAULT_TARGETS.get(
            min(max(consequence_severity, 1), 5),
            1e-4
        )

    def get_default_pfd(self, ipl_type: IPLType) -> float:
        """
        Get default PFD for IPL type.

        Args:
            ipl_type: Type of IPL

        Returns:
            Default PFD value
        """
        return self.DEFAULT_PFD.get(ipl_type, 0.1)
