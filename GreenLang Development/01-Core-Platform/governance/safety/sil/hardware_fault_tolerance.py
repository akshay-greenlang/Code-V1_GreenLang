"""
HardwareFaultTolerance - Hardware Fault Tolerance Requirements

This module implements Hardware Fault Tolerance (HFT) assessment and
requirements per IEC 61511 for Safety Instrumented Systems.

HFT specifies the minimum redundancy requirements to achieve a target SIL:
- HFT = 0: Single channel (1oo1)
- HFT = 1: Dual redundant (1oo2, 2oo2)
- HFT = 2: Triple redundant (2oo3)
- HFT = 3: Quadruple redundant (2oo4)

Reference: IEC 61511-1 Tables 6 and 7, IEC 61508-2 Clause 7.4.3

Example:
    >>> from greenlang.safety.sil.hardware_fault_tolerance import HardwareFaultTolerance
    >>> hft = HardwareFaultTolerance()
    >>> requirement = hft.get_requirement(target_sil=3, component_type="type_b")
    >>> print(f"Minimum HFT: {requirement.min_hft}")
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ComponentType(str, Enum):
    """Component type classification per IEC 61508."""

    TYPE_A = "type_a"  # Simple, well-understood failure modes
    TYPE_B = "type_b"  # Complex, less well-understood failure modes


class ArchitectureType(str, Enum):
    """Architecture types for HFT implementation."""

    ONE_OO_ONE = "1oo1"
    ONE_OO_TWO = "1oo2"
    TWO_OO_TWO = "2oo2"
    TWO_OO_THREE = "2oo3"
    ONE_OO_THREE = "1oo3"
    TWO_OO_FOUR = "2oo4"


class SafeFailureFraction(str, Enum):
    """Safe Failure Fraction categories per IEC 61508."""

    LOW = "low"  # SFF < 60%
    MEDIUM = "medium"  # 60% <= SFF < 90%
    HIGH = "high"  # 90% <= SFF < 99%
    VERY_HIGH = "very_high"  # SFF >= 99%


class HFTRequirement(BaseModel):
    """Hardware Fault Tolerance requirement specification."""

    target_sil: int = Field(
        ...,
        ge=1,
        le=4,
        description="Target SIL level"
    )
    component_type: ComponentType = Field(
        ...,
        description="Component type (A or B)"
    )
    min_hft: int = Field(
        ...,
        ge=0,
        description="Minimum HFT required"
    )
    safe_failure_fraction: Optional[SafeFailureFraction] = Field(
        None,
        description="SFF category if applicable"
    )
    recommended_architectures: List[ArchitectureType] = Field(
        default_factory=list,
        description="Recommended voting architectures"
    )
    prior_use_credit: bool = Field(
        default=False,
        description="Can prior use reduce HFT requirement?"
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Requirement notes"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class HFTAssessment(BaseModel):
    """Assessment of achieved HFT against requirements."""

    assessment_id: str = Field(
        default="",
        description="Assessment identifier"
    )
    equipment_id: str = Field(
        ...,
        description="Equipment identifier"
    )
    target_sil: int = Field(
        ...,
        ge=1,
        le=4,
        description="Target SIL"
    )
    component_type: ComponentType = Field(
        ...,
        description="Component type"
    )
    architecture: ArchitectureType = Field(
        ...,
        description="Implemented architecture"
    )
    achieved_hft: int = Field(
        ...,
        ge=0,
        description="Achieved HFT"
    )
    required_hft: int = Field(
        ...,
        ge=0,
        description="Required HFT for target SIL"
    )
    safe_failure_fraction: float = Field(
        ...,
        ge=0,
        le=1,
        description="Achieved SFF"
    )
    diagnostic_coverage: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Achieved diagnostic coverage"
    )
    meets_requirement: bool = Field(
        ...,
        description="Does achieved HFT meet requirement?"
    )
    sil_achievable: int = Field(
        ...,
        ge=0,
        le=4,
        description="Maximum achievable SIL"
    )
    assessment_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Assessment date"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Improvement recommendations"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HardwareFaultTolerance:
    """
    Hardware Fault Tolerance Assessor.

    Implements HFT requirements and assessment per IEC 61511/61508.
    Determines minimum HFT based on:
    - Target SIL level
    - Component type (A or B)
    - Safe Failure Fraction
    - Prior use qualification

    The assessor uses deterministic table lookups only (zero hallucination).

    Attributes:
        hft_table_type_a: HFT requirements for Type A components
        hft_table_type_b: HFT requirements for Type B components

    Example:
        >>> assessor = HardwareFaultTolerance()
        >>> req = assessor.get_requirement(target_sil=2, component_type="type_a")
        >>> print(f"Min HFT: {req.min_hft}")
    """

    # HFT requirements for Type A components per IEC 61511-1 Table 6
    # Key: (SIL, SFF category) -> minimum HFT
    HFT_TABLE_TYPE_A: Dict[Tuple[int, str], int] = {
        # SIL 1
        (1, "low"): 0,
        (1, "medium"): 0,
        (1, "high"): 0,
        (1, "very_high"): 0,
        # SIL 2
        (2, "low"): 1,
        (2, "medium"): 0,
        (2, "high"): 0,
        (2, "very_high"): 0,
        # SIL 3
        (3, "low"): 2,
        (3, "medium"): 1,
        (3, "high"): 0,
        (3, "very_high"): 0,
        # SIL 4 (Note: rarely used in process industry)
        (4, "low"): 3,
        (4, "medium"): 2,
        (4, "high"): 1,
        (4, "very_high"): 0,
    }

    # HFT requirements for Type B components per IEC 61511-1 Table 7
    HFT_TABLE_TYPE_B: Dict[Tuple[int, str], int] = {
        # SIL 1
        (1, "low"): 1,
        (1, "medium"): 0,
        (1, "high"): 0,
        (1, "very_high"): 0,
        # SIL 2
        (2, "low"): 2,
        (2, "medium"): 1,
        (2, "high"): 0,
        (2, "very_high"): 0,
        # SIL 3
        (3, "low"): 3,
        (3, "medium"): 2,
        (3, "high"): 1,
        (3, "very_high"): 0,
        # SIL 4
        (4, "low"): None,  # Not achievable
        (4, "medium"): 3,
        (4, "high"): 2,
        (4, "very_high"): 1,
    }

    # Architecture to HFT mapping
    ARCHITECTURE_HFT: Dict[ArchitectureType, int] = {
        ArchitectureType.ONE_OO_ONE: 0,
        ArchitectureType.ONE_OO_TWO: 1,
        ArchitectureType.TWO_OO_TWO: 1,
        ArchitectureType.TWO_OO_THREE: 2,
        ArchitectureType.ONE_OO_THREE: 2,
        ArchitectureType.TWO_OO_FOUR: 3,
    }

    def __init__(self):
        """Initialize HardwareFaultTolerance assessor."""
        logger.info("HardwareFaultTolerance assessor initialized")

    def get_requirement(
        self,
        target_sil: int,
        component_type: str,
        safe_failure_fraction: Optional[float] = None
    ) -> HFTRequirement:
        """
        Get HFT requirement for target SIL and component type.

        Args:
            target_sil: Target SIL level (1-4)
            component_type: Component type ("type_a" or "type_b")
            safe_failure_fraction: SFF value (0-1) if known

        Returns:
            HFTRequirement with minimum HFT and recommendations

        Raises:
            ValueError: If parameters are invalid
        """
        logger.info(
            f"Getting HFT requirement for SIL {target_sil}, {component_type}"
        )

        if target_sil < 1 or target_sil > 4:
            raise ValueError(f"Invalid SIL: {target_sil}. Must be 1-4.")

        comp_type = ComponentType(component_type)

        # Determine SFF category
        sff_category = self._get_sff_category(safe_failure_fraction)

        # Look up HFT requirement
        if comp_type == ComponentType.TYPE_A:
            table = self.HFT_TABLE_TYPE_A
        else:
            table = self.HFT_TABLE_TYPE_B

        key = (target_sil, sff_category.value if sff_category else "low")
        min_hft = table.get(key)

        if min_hft is None:
            min_hft = 99  # Not achievable marker
            notes = [
                f"SIL {target_sil} not achievable with {component_type} components "
                f"and SFF {sff_category.value if sff_category else 'low'}. "
                "Consider Type A components or improved SFF."
            ]
        else:
            notes = []

        # Get recommended architectures
        recommended_arch = self._get_recommended_architectures(min_hft)

        # Create requirement
        requirement = HFTRequirement(
            target_sil=target_sil,
            component_type=comp_type,
            min_hft=min_hft if min_hft != 99 else 4,
            safe_failure_fraction=sff_category,
            recommended_architectures=recommended_arch,
            prior_use_credit=(comp_type == ComponentType.TYPE_A),
            notes=notes,
        )

        # Calculate provenance hash
        requirement.provenance_hash = self._calculate_provenance(
            target_sil, component_type, min_hft
        )

        return requirement

    def assess_architecture(
        self,
        equipment_id: str,
        target_sil: int,
        component_type: str,
        architecture: str,
        safe_failure_fraction: float,
        diagnostic_coverage: float = 0.0
    ) -> HFTAssessment:
        """
        Assess if architecture meets HFT requirements.

        Args:
            equipment_id: Equipment identifier
            target_sil: Target SIL level
            component_type: Component type
            architecture: Implemented architecture
            safe_failure_fraction: Achieved SFF (0-1)
            diagnostic_coverage: Achieved DC (0-1)

        Returns:
            HFTAssessment with compliance determination

        Raises:
            ValueError: If parameters are invalid
        """
        logger.info(f"Assessing HFT for {equipment_id}")

        # Get architecture HFT
        arch = ArchitectureType(architecture)
        achieved_hft = self.ARCHITECTURE_HFT[arch]

        # Get requirement
        requirement = self.get_requirement(
            target_sil=target_sil,
            component_type=component_type,
            safe_failure_fraction=safe_failure_fraction
        )
        required_hft = requirement.min_hft

        # Determine if requirement is met
        meets_requirement = achieved_hft >= required_hft

        # Calculate maximum achievable SIL
        sil_achievable = self._calculate_achievable_sil(
            achieved_hft,
            ComponentType(component_type),
            safe_failure_fraction
        )

        # Generate recommendations
        recommendations = []
        if not meets_requirement:
            recommendations.append(
                f"Increase HFT from {achieved_hft} to at least {required_hft}. "
                f"Consider {self._get_recommended_architectures(required_hft)}."
            )

            if safe_failure_fraction < 0.9:
                recommendations.append(
                    "Consider improving SFF through better diagnostics "
                    "to reduce HFT requirement."
                )

        # Create assessment
        assessment = HFTAssessment(
            assessment_id=f"HFT-{equipment_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            equipment_id=equipment_id,
            target_sil=target_sil,
            component_type=ComponentType(component_type),
            architecture=arch,
            achieved_hft=achieved_hft,
            required_hft=required_hft,
            safe_failure_fraction=safe_failure_fraction,
            diagnostic_coverage=diagnostic_coverage,
            meets_requirement=meets_requirement,
            sil_achievable=sil_achievable,
            recommendations=recommendations,
        )

        # Calculate provenance hash
        assessment.provenance_hash = self._calculate_assessment_provenance(
            assessment
        )

        logger.info(
            f"HFT assessment for {equipment_id}: "
            f"achieved={achieved_hft}, required={required_hft}, "
            f"meets={meets_requirement}"
        )

        return assessment

    def _get_sff_category(
        self,
        sff: Optional[float]
    ) -> Optional[SafeFailureFraction]:
        """
        Categorize Safe Failure Fraction.

        Args:
            sff: SFF value (0-1)

        Returns:
            SafeFailureFraction category
        """
        if sff is None:
            return SafeFailureFraction.LOW

        if sff < 0.6:
            return SafeFailureFraction.LOW
        elif sff < 0.9:
            return SafeFailureFraction.MEDIUM
        elif sff < 0.99:
            return SafeFailureFraction.HIGH
        else:
            return SafeFailureFraction.VERY_HIGH

    def _get_recommended_architectures(
        self,
        min_hft: int
    ) -> List[ArchitectureType]:
        """
        Get recommended architectures for given HFT.

        Args:
            min_hft: Minimum required HFT

        Returns:
            List of recommended architectures
        """
        recommendations = []

        for arch, hft in self.ARCHITECTURE_HFT.items():
            if hft >= min_hft:
                recommendations.append(arch)

        return recommendations

    def _calculate_achievable_sil(
        self,
        hft: int,
        component_type: ComponentType,
        sff: float
    ) -> int:
        """
        Calculate maximum achievable SIL given HFT and SFF.

        Args:
            hft: Hardware Fault Tolerance
            component_type: Component type
            sff: Safe Failure Fraction

        Returns:
            Maximum achievable SIL (0-4)
        """
        sff_category = self._get_sff_category(sff)

        if component_type == ComponentType.TYPE_A:
            table = self.HFT_TABLE_TYPE_A
        else:
            table = self.HFT_TABLE_TYPE_B

        achievable_sil = 0

        for sil in range(4, 0, -1):
            key = (sil, sff_category.value)
            required_hft = table.get(key)

            if required_hft is not None and hft >= required_hft:
                achievable_sil = sil
                break

        return achievable_sil

    def calculate_sff(
        self,
        lambda_sd: float,
        lambda_su: float,
        lambda_dd: float,
        lambda_du: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Safe Failure Fraction from failure rates.

        SFF = (lambda_SD + lambda_SU + lambda_DD) /
              (lambda_SD + lambda_SU + lambda_DD + lambda_DU)

        Args:
            lambda_sd: Safe detected failure rate
            lambda_su: Safe undetected failure rate
            lambda_dd: Dangerous detected failure rate
            lambda_du: Dangerous undetected failure rate

        Returns:
            Tuple of (SFF value, breakdown dict)
        """
        total_safe = lambda_sd + lambda_su
        total_dangerous = lambda_dd + lambda_du
        total = total_safe + total_dangerous

        if total == 0:
            sff = 1.0
        else:
            sff = (total_safe + lambda_dd) / total

        breakdown = {
            "lambda_sd": lambda_sd,
            "lambda_su": lambda_su,
            "lambda_dd": lambda_dd,
            "lambda_du": lambda_du,
            "total_safe": total_safe,
            "total_dangerous": total_dangerous,
            "sff": sff,
            "sff_category": self._get_sff_category(sff).value,
        }

        logger.debug(f"Calculated SFF: {sff:.2%}")

        return sff, breakdown

    def calculate_dc(
        self,
        lambda_dd: float,
        lambda_du: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Diagnostic Coverage from failure rates.

        DC = lambda_DD / (lambda_DD + lambda_DU)

        Args:
            lambda_dd: Dangerous detected failure rate
            lambda_du: Dangerous undetected failure rate

        Returns:
            Tuple of (DC value, breakdown dict)
        """
        total_dangerous = lambda_dd + lambda_du

        if total_dangerous == 0:
            dc = 0.0
        else:
            dc = lambda_dd / total_dangerous

        breakdown = {
            "lambda_dd": lambda_dd,
            "lambda_du": lambda_du,
            "total_dangerous": total_dangerous,
            "dc": dc,
        }

        logger.debug(f"Calculated DC: {dc:.2%}")

        return dc, breakdown

    def get_architecture_comparison(
        self,
        target_sil: int,
        component_type: str,
        safe_failure_fraction: float
    ) -> List[Dict[str, Any]]:
        """
        Compare architectures for given requirements.

        Args:
            target_sil: Target SIL level
            component_type: Component type
            safe_failure_fraction: SFF value

        Returns:
            List of architecture comparisons
        """
        requirement = self.get_requirement(
            target_sil=target_sil,
            component_type=component_type,
            safe_failure_fraction=safe_failure_fraction
        )

        comparisons = []

        for arch, hft in self.ARCHITECTURE_HFT.items():
            comparison = {
                "architecture": arch.value,
                "achieved_hft": hft,
                "required_hft": requirement.min_hft,
                "meets_requirement": hft >= requirement.min_hft,
                "margin": hft - requirement.min_hft,
                "recommended": arch in requirement.recommended_architectures,
            }
            comparisons.append(comparison)

        return sorted(comparisons, key=lambda x: x["achieved_hft"])

    def _calculate_provenance(
        self,
        sil: int,
        component_type: str,
        hft: int
    ) -> str:
        """Calculate SHA-256 provenance hash for requirement."""
        provenance_str = (
            f"{sil}|{component_type}|{hft}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _calculate_assessment_provenance(
        self,
        assessment: HFTAssessment
    ) -> str:
        """Calculate SHA-256 provenance hash for assessment."""
        provenance_str = (
            f"{assessment.equipment_id}|"
            f"{assessment.achieved_hft}|"
            f"{assessment.required_hft}|"
            f"{assessment.meets_requirement}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
