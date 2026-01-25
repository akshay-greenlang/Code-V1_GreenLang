"""
DiagnosticCoverageCalculator - Diagnostic Coverage Calculator

This module implements Diagnostic Coverage (DC) calculations per IEC 61511
and IEC 61508. DC quantifies the effectiveness of online diagnostics
in detecting dangerous failures.

Key concepts:
- DC = lambda_DD / (lambda_DD + lambda_DU)
- Higher DC reduces effective dangerous failure rate
- Required DC depends on SIL and architecture

Reference: IEC 61508-2 Clause 7.4.3.2, IEC 61511-1 Tables 6 and 7

Example:
    >>> from greenlang.safety.srs.diagnostic_coverage import DiagnosticCoverageCalculator
    >>> calc = DiagnosticCoverageCalculator()
    >>> result = calc.calculate_dc(lambda_dd=1e-6, lambda_du=1e-6)
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DCCategory(str, Enum):
    """Diagnostic Coverage categories per IEC 61508."""

    NONE = "none"  # DC < 60%
    LOW = "low"  # 60% <= DC < 90%
    MEDIUM = "medium"  # 90% <= DC < 99%
    HIGH = "high"  # DC >= 99%


class DiagnosticType(str, Enum):
    """Types of diagnostic techniques."""

    COMPARISON = "comparison"  # Compare redundant signals
    WATCHDOG = "watchdog"  # Watchdog timer
    RANGE_CHECK = "range_check"  # Signal range checking
    PLAUSIBILITY = "plausibility"  # Plausibility check
    SELF_TEST = "self_test"  # Component self-test
    PARTIAL_STROKE = "partial_stroke"  # Partial stroke test
    REDUNDANCY = "redundancy"  # Redundancy check
    CROSS_MONITORING = "cross_monitoring"  # Cross-monitoring
    RAM_CHECK = "ram_check"  # RAM memory check
    ROM_CHECK = "rom_check"  # ROM memory check
    CPU_CHECK = "cpu_check"  # CPU self-test
    IO_CHECK = "io_check"  # I/O module check


class DiagnosticTechnique(BaseModel):
    """Individual diagnostic technique specification."""

    technique_id: str = Field(
        ...,
        description="Technique identifier"
    )
    technique_type: DiagnosticType = Field(
        ...,
        description="Type of diagnostic"
    )
    description: str = Field(
        default="",
        description="Description of technique"
    )
    coverage_claim: float = Field(
        ...,
        ge=0,
        le=1,
        description="Claimed coverage (0-1)"
    )
    test_interval_ms: float = Field(
        default=1000.0,
        gt=0,
        description="Test interval (milliseconds)"
    )
    detection_time_ms: float = Field(
        default=100.0,
        gt=0,
        description="Time to detect fault (milliseconds)"
    )
    failure_modes_covered: List[str] = Field(
        default_factory=list,
        description="List of failure modes covered"
    )
    reference_standard: str = Field(
        default="IEC 61508-2",
        description="Reference standard for coverage claim"
    )


class DCInput(BaseModel):
    """Input parameters for DC calculation."""

    component_id: str = Field(
        ...,
        description="Component identifier"
    )
    component_type: str = Field(
        default="",
        description="Component type"
    )
    lambda_dd: float = Field(
        ...,
        ge=0,
        description="Dangerous detected failure rate (per hour)"
    )
    lambda_du: float = Field(
        ...,
        ge=0,
        description="Dangerous undetected failure rate (per hour)"
    )
    lambda_sd: float = Field(
        default=0,
        ge=0,
        description="Safe detected failure rate (per hour)"
    )
    lambda_su: float = Field(
        default=0,
        ge=0,
        description="Safe undetected failure rate (per hour)"
    )
    diagnostic_techniques: List[DiagnosticTechnique] = Field(
        default_factory=list,
        description="Applied diagnostic techniques"
    )
    target_dc: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Target DC to achieve"
    )


class DCResult(BaseModel):
    """Result of DC calculation."""

    component_id: str = Field(
        ...,
        description="Component identifier"
    )
    dc_calculated: float = Field(
        ...,
        ge=0,
        le=1,
        description="Calculated diagnostic coverage"
    )
    dc_category: DCCategory = Field(
        ...,
        description="DC category per IEC 61508"
    )
    lambda_dd: float = Field(
        ...,
        description="Dangerous detected failure rate"
    )
    lambda_du: float = Field(
        ...,
        description="Dangerous undetected failure rate"
    )
    lambda_d_total: float = Field(
        ...,
        description="Total dangerous failure rate"
    )
    sff_calculated: Optional[float] = Field(
        None,
        description="Safe Failure Fraction if safe rates provided"
    )
    meets_target: bool = Field(
        default=True,
        description="Does DC meet target?"
    )
    target_dc: Optional[float] = Field(
        None,
        description="Target DC if specified"
    )
    gap_to_target: Optional[float] = Field(
        None,
        description="Gap between calculated and target DC"
    )
    techniques_applied: List[str] = Field(
        default_factory=list,
        description="Diagnostic techniques applied"
    )
    improvement_potential: float = Field(
        default=0,
        description="Potential DC improvement with additional diagnostics"
    )
    calculation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of calculation"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )
    formula_used: str = Field(
        default="DC = lambda_DD / (lambda_DD + lambda_DU)",
        description="Formula used"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DiagnosticCoverageCalculator:
    """
    Diagnostic Coverage Calculator.

    Calculates DC per IEC 61508/61511 and provides:
    - DC from failure rate data
    - DC categorization
    - SFF calculation
    - Improvement recommendations

    The calculator follows zero-hallucination principles:
    - All calculations are deterministic
    - No LLM involvement in numeric computations
    - Complete audit trail

    Attributes:
        dc_category_limits: Dict of DC category boundaries

    Example:
        >>> calc = DiagnosticCoverageCalculator()
        >>> result = calc.calculate_dc(DCInput(...))
    """

    # DC category boundaries per IEC 61508
    DC_CATEGORY_LIMITS: Dict[DCCategory, Tuple[float, float]] = {
        DCCategory.NONE: (0.0, 0.6),
        DCCategory.LOW: (0.6, 0.9),
        DCCategory.MEDIUM: (0.9, 0.99),
        DCCategory.HIGH: (0.99, 1.0),
    }

    # Typical DC values for common diagnostic techniques per IEC 61508-2 Table A.2-A.14
    TECHNIQUE_DC_VALUES: Dict[DiagnosticType, Tuple[float, float]] = {
        DiagnosticType.COMPARISON: (0.9, 0.99),
        DiagnosticType.WATCHDOG: (0.6, 0.9),
        DiagnosticType.RANGE_CHECK: (0.6, 0.9),
        DiagnosticType.PLAUSIBILITY: (0.6, 0.9),
        DiagnosticType.SELF_TEST: (0.9, 0.99),
        DiagnosticType.PARTIAL_STROKE: (0.6, 0.9),
        DiagnosticType.REDUNDANCY: (0.9, 0.99),
        DiagnosticType.CROSS_MONITORING: (0.9, 0.99),
        DiagnosticType.RAM_CHECK: (0.9, 0.99),
        DiagnosticType.ROM_CHECK: (0.99, 1.0),
        DiagnosticType.CPU_CHECK: (0.9, 0.99),
        DiagnosticType.IO_CHECK: (0.6, 0.9),
    }

    def __init__(self):
        """Initialize DiagnosticCoverageCalculator."""
        logger.info("DiagnosticCoverageCalculator initialized")

    def calculate_dc(self, input_data: DCInput) -> DCResult:
        """
        Calculate Diagnostic Coverage.

        DC = lambda_DD / (lambda_DD + lambda_DU)

        Args:
            input_data: DCInput with failure rate data

        Returns:
            DCResult with calculated DC

        Raises:
            ValueError: If input parameters are invalid
        """
        logger.info(f"Calculating DC for {input_data.component_id}")

        try:
            # Calculate total dangerous failure rate
            lambda_d_total = input_data.lambda_dd + input_data.lambda_du

            # Calculate DC
            if lambda_d_total == 0:
                dc = 0.0
                logger.warning(
                    f"Total dangerous failure rate is 0 for {input_data.component_id}"
                )
            else:
                dc = input_data.lambda_dd / lambda_d_total

            # Determine DC category
            dc_category = self._categorize_dc(dc)

            # Calculate SFF if safe failure rates provided
            sff = None
            if input_data.lambda_sd > 0 or input_data.lambda_su > 0:
                sff = self._calculate_sff(
                    input_data.lambda_sd,
                    input_data.lambda_su,
                    input_data.lambda_dd,
                    input_data.lambda_du
                )

            # Check against target
            meets_target = True
            gap_to_target = None
            if input_data.target_dc is not None:
                meets_target = dc >= input_data.target_dc
                gap_to_target = input_data.target_dc - dc

            # Calculate improvement potential
            improvement_potential = 1.0 - dc  # Maximum possible improvement

            # Generate recommendations
            recommendations = self._generate_recommendations(
                dc, dc_category, input_data.target_dc, input_data.diagnostic_techniques
            )

            # Build result
            result = DCResult(
                component_id=input_data.component_id,
                dc_calculated=dc,
                dc_category=dc_category,
                lambda_dd=input_data.lambda_dd,
                lambda_du=input_data.lambda_du,
                lambda_d_total=lambda_d_total,
                sff_calculated=sff,
                meets_target=meets_target,
                target_dc=input_data.target_dc,
                gap_to_target=gap_to_target,
                techniques_applied=[t.technique_type.value for t in input_data.diagnostic_techniques],
                improvement_potential=improvement_potential,
                recommendations=recommendations,
            )

            # Calculate provenance hash
            result.provenance_hash = self._calculate_provenance(input_data, dc)

            logger.info(
                f"DC calculated for {input_data.component_id}: "
                f"{dc:.2%} ({dc_category.value})"
            )

            return result

        except Exception as e:
            logger.error(f"DC calculation failed: {str(e)}", exc_info=True)
            raise

    def calculate_combined_dc(
        self,
        techniques: List[DiagnosticTechnique]
    ) -> Dict[str, Any]:
        """
        Calculate combined DC from multiple diagnostic techniques.

        Combined DC uses parallel coverage model:
        DC_combined = 1 - Product(1 - DC_i)

        Args:
            techniques: List of diagnostic techniques

        Returns:
            Dict with combined DC calculation
        """
        if not techniques:
            return {
                "combined_dc": 0.0,
                "technique_count": 0,
                "formula": "No techniques applied"
            }

        # Calculate combined DC using parallel model
        uncovered_fraction = 1.0
        for technique in techniques:
            uncovered_fraction *= (1.0 - technique.coverage_claim)

        combined_dc = 1.0 - uncovered_fraction

        return {
            "combined_dc": combined_dc,
            "combined_dc_category": self._categorize_dc(combined_dc).value,
            "technique_count": len(techniques),
            "individual_coverages": {
                t.technique_id: t.coverage_claim for t in techniques
            },
            "formula": "DC_combined = 1 - Product(1 - DC_i)"
        }

    def estimate_dc_improvement(
        self,
        current_dc: float,
        additional_technique: DiagnosticType
    ) -> Dict[str, float]:
        """
        Estimate DC improvement from adding a diagnostic technique.

        Args:
            current_dc: Current diagnostic coverage
            additional_technique: Type of technique to add

        Returns:
            Dict with estimated improvement
        """
        # Get typical DC range for technique
        dc_range = self.TECHNIQUE_DC_VALUES.get(
            additional_technique,
            (0.5, 0.9)
        )
        technique_dc_low, technique_dc_high = dc_range

        # Calculate new DC using parallel model
        # New DC = 1 - (1 - current) * (1 - technique)
        new_dc_low = 1.0 - (1.0 - current_dc) * (1.0 - technique_dc_low)
        new_dc_high = 1.0 - (1.0 - current_dc) * (1.0 - technique_dc_high)

        improvement_low = new_dc_low - current_dc
        improvement_high = new_dc_high - current_dc

        return {
            "current_dc": current_dc,
            "technique": additional_technique.value,
            "technique_dc_range": (technique_dc_low, technique_dc_high),
            "new_dc_range": (new_dc_low, new_dc_high),
            "improvement_range": (improvement_low, improvement_high),
            "new_category_low": self._categorize_dc(new_dc_low).value,
            "new_category_high": self._categorize_dc(new_dc_high).value,
        }

    def _categorize_dc(self, dc: float) -> DCCategory:
        """
        Categorize DC per IEC 61508.

        Args:
            dc: Diagnostic coverage value

        Returns:
            DCCategory
        """
        for category, (lower, upper) in self.DC_CATEGORY_LIMITS.items():
            if lower <= dc < upper:
                return category

        if dc >= 0.99:
            return DCCategory.HIGH

        return DCCategory.NONE

    def _calculate_sff(
        self,
        lambda_sd: float,
        lambda_su: float,
        lambda_dd: float,
        lambda_du: float
    ) -> float:
        """
        Calculate Safe Failure Fraction.

        SFF = (lambda_SD + lambda_SU + lambda_DD) /
              (lambda_SD + lambda_SU + lambda_DD + lambda_DU)

        Args:
            lambda_sd: Safe detected failure rate
            lambda_su: Safe undetected failure rate
            lambda_dd: Dangerous detected failure rate
            lambda_du: Dangerous undetected failure rate

        Returns:
            SFF value
        """
        total = lambda_sd + lambda_su + lambda_dd + lambda_du

        if total == 0:
            return 1.0

        sff = (lambda_sd + lambda_su + lambda_dd) / total
        return sff

    def _generate_recommendations(
        self,
        dc: float,
        dc_category: DCCategory,
        target_dc: Optional[float],
        techniques: List[DiagnosticTechnique]
    ) -> List[str]:
        """Generate recommendations for DC improvement."""
        recommendations = []

        if dc_category == DCCategory.NONE:
            recommendations.append(
                "DC is below 60%. Add diagnostic techniques such as "
                "comparison, self-test, or watchdog."
            )

        if target_dc and dc < target_dc:
            gap = target_dc - dc
            recommendations.append(
                f"DC is {gap:.1%} below target. Consider additional diagnostics."
            )

        # Suggest techniques not yet applied
        applied_types = {t.technique_type for t in techniques}
        high_value_techniques = [
            DiagnosticType.COMPARISON,
            DiagnosticType.SELF_TEST,
            DiagnosticType.REDUNDANCY,
        ]

        for technique in high_value_techniques:
            if technique not in applied_types:
                dc_range = self.TECHNIQUE_DC_VALUES[technique]
                recommendations.append(
                    f"Consider adding {technique.value} diagnostic "
                    f"(typical DC: {dc_range[0]:.0%}-{dc_range[1]:.0%})"
                )

        if dc >= 0.99:
            recommendations.append(
                "High DC achieved. Maintain diagnostic test intervals."
            )

        return recommendations

    def get_minimum_dc_for_sil(
        self,
        sil: int,
        hft: int = 0,
        component_type: str = "type_b"
    ) -> float:
        """
        Get minimum DC required for SIL per IEC 61508.

        Args:
            sil: Target SIL level (1-4)
            hft: Hardware Fault Tolerance
            component_type: "type_a" or "type_b"

        Returns:
            Minimum required DC
        """
        # Per IEC 61508-2 Table 2 and 3
        # Simplified lookup - actual requirements depend on architecture

        if component_type == "type_a":
            # Type A: well-understood
            if hft >= sil:
                return 0.0  # No DC required with sufficient HFT
            elif hft == sil - 1:
                return 0.6  # Low DC required
            elif hft == sil - 2:
                return 0.9  # Medium DC required
            else:
                return 0.99  # High DC required
        else:
            # Type B: complex
            if hft >= sil + 1:
                return 0.0
            elif hft == sil:
                return 0.6
            elif hft == sil - 1:
                return 0.9
            else:
                return 0.99

    def validate_dc_for_architecture(
        self,
        dc: float,
        sil: int,
        architecture: str,
        component_type: str = "type_b"
    ) -> Dict[str, Any]:
        """
        Validate DC for specific SIL and architecture.

        Args:
            dc: Achieved diagnostic coverage
            sil: Target SIL level
            architecture: Voting architecture (1oo1, 1oo2, 2oo3)
            component_type: Component type

        Returns:
            Validation result dictionary
        """
        # Determine HFT from architecture
        hft_map = {
            "1oo1": 0,
            "1oo2": 1,
            "2oo2": 1,
            "2oo3": 2,
        }
        hft = hft_map.get(architecture, 0)

        # Get minimum required DC
        min_dc = self.get_minimum_dc_for_sil(sil, hft, component_type)

        is_valid = dc >= min_dc

        return {
            "is_valid": is_valid,
            "achieved_dc": dc,
            "achieved_dc_category": self._categorize_dc(dc).value,
            "required_dc": min_dc,
            "required_dc_category": self._categorize_dc(min_dc).value,
            "target_sil": sil,
            "architecture": architecture,
            "hft": hft,
            "component_type": component_type,
            "margin": dc - min_dc,
            "recommendation": (
                "DC adequate for SIL requirement" if is_valid
                else f"Increase DC by {min_dc - dc:.1%} or increase HFT"
            )
        }

    def _calculate_provenance(
        self,
        input_data: DCInput,
        dc: float
    ) -> str:
        """Calculate SHA-256 provenance hash for DC calculation."""
        provenance_str = (
            f"{input_data.component_id}|"
            f"{input_data.lambda_dd}|"
            f"{input_data.lambda_du}|"
            f"{dc}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
