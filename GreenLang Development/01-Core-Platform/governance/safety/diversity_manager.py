"""
DiversityManager - Diversity Requirements Management Module

This module implements diversity tracking and validation for Safety Instrumented
Systems (SIS) per IEC 61511 and IEC 61508. Diversity is a key CCF mitigation
strategy that ensures redundant channels use different designs, technologies,
or implementations.

Key implementations:
- Hardware diversity tracking and validation
- Software diversity validation
- Measurement diversity requirements
- Diverse shutdown path verification
- Diversity score calculation
- Diversity gap analysis

Reference: IEC 61508-2 Clause 7.4.2.3, IEC 61511-1 Clause 11.4

Example:
    >>> from greenlang.safety.diversity_manager import DiversityManager
    >>> manager = DiversityManager()
    >>> result = manager.evaluate_hardware_diversity(channel_a, channel_b)
    >>> print(f"Diversity Score: {result.diversity_score}")

Author: GreenLang Safety Engineering Team
Version: 1.0
Date: 2025-12-07
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from dataclasses import dataclass, field
import hashlib
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class DiversityType(str, Enum):
    """Types of diversity for safety systems."""
    HARDWARE = "hardware"
    SOFTWARE = "software"
    MEASUREMENT = "measurement"
    FUNCTIONAL = "functional"
    SIGNAL = "signal"
    EQUIPMENT = "equipment"


class DiversityLevel(str, Enum):
    """Diversity achievement levels."""
    NONE = "none"  # Identical, no diversity
    LOW = "low"  # Same type, different batches
    MEDIUM = "medium"  # Different manufacturers/models
    HIGH = "high"  # Different technology/principle


class ComponentCategory(str, Enum):
    """Component categories for diversity tracking."""
    SENSOR = "sensor"
    TRANSMITTER = "transmitter"
    LOGIC_SOLVER = "logic_solver"
    FINAL_ELEMENT = "final_element"
    POWER_SUPPLY = "power_supply"
    COMMUNICATION = "communication"
    SOFTWARE = "software"


class DiversityStatus(str, Enum):
    """Diversity requirement status."""
    MET = "met"
    PARTIALLY_MET = "partially_met"
    NOT_MET = "not_met"
    NOT_APPLICABLE = "not_applicable"


# =============================================================================
# Data Models
# =============================================================================

class ComponentInfo(BaseModel):
    """Component information for diversity tracking."""

    component_id: str = Field(
        default_factory=lambda: f"COMP-{uuid.uuid4().hex[:8].upper()}",
        description="Component identifier"
    )
    category: ComponentCategory = Field(
        ...,
        description="Component category"
    )
    manufacturer: str = Field(
        ...,
        description="Component manufacturer"
    )
    model: str = Field(
        ...,
        description="Component model/part number"
    )
    technology: str = Field(
        default="",
        description="Technology or sensing principle"
    )
    software_version: Optional[str] = Field(
        None,
        description="Software/firmware version"
    )
    hardware_revision: Optional[str] = Field(
        None,
        description="Hardware revision"
    )
    batch_lot: Optional[str] = Field(
        None,
        description="Manufacturing batch/lot number"
    )
    channel: str = Field(
        default="A",
        description="Channel assignment (A, B, C, etc.)"
    )
    description: str = Field(
        default="",
        description="Component description"
    )


class DiversityRequirement(BaseModel):
    """Diversity requirement specification."""

    requirement_id: str = Field(
        default_factory=lambda: f"DIV-REQ-{uuid.uuid4().hex[:6].upper()}",
        description="Requirement identifier"
    )
    diversity_type: DiversityType = Field(
        ...,
        description="Type of diversity required"
    )
    component_category: ComponentCategory = Field(
        ...,
        description="Component category affected"
    )
    required_level: DiversityLevel = Field(
        ...,
        description="Required diversity level"
    )
    description: str = Field(
        ...,
        description="Requirement description"
    )
    standard_reference: str = Field(
        default="IEC 61511-1 Clause 11.4",
        description="Standard reference"
    )
    target_sil: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Target SIL level"
    )
    is_mandatory: bool = Field(
        default=True,
        description="Is this requirement mandatory"
    )


class DiversityEvaluation(BaseModel):
    """Result of diversity evaluation between components."""

    evaluation_id: str = Field(
        default_factory=lambda: f"DIV-EVAL-{uuid.uuid4().hex[:8].upper()}",
        description="Evaluation identifier"
    )
    component_a_id: str = Field(..., description="First component ID")
    component_b_id: str = Field(..., description="Second component ID")
    diversity_type: DiversityType = Field(..., description="Type evaluated")
    achieved_level: DiversityLevel = Field(..., description="Achieved level")
    diversity_score: int = Field(
        ...,
        ge=0,
        le=3,
        description="Diversity score (0-3)"
    )
    is_manufacturer_diverse: bool = Field(
        default=False,
        description="Different manufacturers"
    )
    is_model_diverse: bool = Field(
        default=False,
        description="Different models"
    )
    is_technology_diverse: bool = Field(
        default=False,
        description="Different technologies"
    )
    is_software_diverse: bool = Field(
        default=False,
        description="Different software versions"
    )
    is_batch_diverse: bool = Field(
        default=False,
        description="Different manufacturing batches"
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Evaluation findings"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Improvement recommendations"
    )
    evaluation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Evaluation timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )


class ShutdownPathInfo(BaseModel):
    """Shutdown path information for diverse path verification."""

    path_id: str = Field(
        default_factory=lambda: f"PATH-{uuid.uuid4().hex[:6].upper()}",
        description="Path identifier"
    )
    path_name: str = Field(..., description="Path name")
    components: List[str] = Field(
        default_factory=list,
        description="Component IDs in this path"
    )
    is_primary: bool = Field(default=True, description="Is primary path")
    technology: str = Field(default="", description="Path technology")
    response_time_ms: float = Field(
        default=0,
        gt=0,
        description="Path response time in ms"
    )


class DiversityReport(BaseModel):
    """Comprehensive diversity assessment report."""

    report_id: str = Field(
        default_factory=lambda: f"DIV-RPT-{uuid.uuid4().hex[:8].upper()}",
        description="Report identifier"
    )
    system_id: str = Field(..., description="System being assessed")
    target_sil: int = Field(..., description="Target SIL level")
    overall_score: float = Field(
        default=0,
        ge=0,
        le=100,
        description="Overall diversity score (0-100)"
    )
    overall_status: DiversityStatus = Field(
        default=DiversityStatus.NOT_MET,
        description="Overall status"
    )
    hardware_diversity_score: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Hardware diversity score"
    )
    software_diversity_score: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Software diversity score"
    )
    measurement_diversity_score: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Measurement diversity score"
    )
    path_diversity_verified: bool = Field(
        default=False,
        description="Diverse shutdown paths verified"
    )
    requirements_met: int = Field(
        default=0,
        description="Number of requirements met"
    )
    requirements_total: int = Field(
        default=0,
        description="Total number of requirements"
    )
    evaluations: List[DiversityEvaluation] = Field(
        default_factory=list,
        description="Individual evaluations"
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Identified gaps"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Overall recommendations"
    )
    report_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Report timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# Diversity Manager
# =============================================================================

class DiversityManager:
    """
    Diversity Requirements Manager.

    Tracks and validates diversity requirements for Safety Instrumented
    Systems per IEC 61511 and IEC 61508. Provides comprehensive
    assessment of hardware, software, and measurement diversity.

    The manager follows zero-hallucination principles:
    - All evaluations are deterministic
    - No LLM involvement in assessments
    - Complete audit trail with provenance hashing

    Attributes:
        components: Dict of tracked components
        requirements: List of diversity requirements
        evaluations: List of completed evaluations

    Example:
        >>> manager = DiversityManager()
        >>> manager.add_component(sensor_a)
        >>> manager.add_component(sensor_b)
        >>> result = manager.evaluate_diversity("A", "B")
    """

    # Diversity score mapping
    LEVEL_SCORES: Dict[DiversityLevel, int] = {
        DiversityLevel.NONE: 0,
        DiversityLevel.LOW: 1,
        DiversityLevel.MEDIUM: 2,
        DiversityLevel.HIGH: 3,
    }

    # Required diversity by SIL level
    SIL_DIVERSITY_REQUIREMENTS: Dict[int, Dict[str, DiversityLevel]] = {
        1: {
            "hardware": DiversityLevel.LOW,
            "software": DiversityLevel.NONE,
            "measurement": DiversityLevel.NONE,
        },
        2: {
            "hardware": DiversityLevel.MEDIUM,
            "software": DiversityLevel.LOW,
            "measurement": DiversityLevel.LOW,
        },
        3: {
            "hardware": DiversityLevel.HIGH,
            "software": DiversityLevel.MEDIUM,
            "measurement": DiversityLevel.MEDIUM,
        },
        4: {
            "hardware": DiversityLevel.HIGH,
            "software": DiversityLevel.HIGH,
            "measurement": DiversityLevel.HIGH,
        },
    }

    def __init__(self):
        """Initialize DiversityManager."""
        self.components: Dict[str, ComponentInfo] = {}
        self.requirements: List[DiversityRequirement] = []
        self.evaluations: List[DiversityEvaluation] = []
        self.shutdown_paths: List[ShutdownPathInfo] = []
        logger.info("DiversityManager initialized")

    def add_component(self, component: ComponentInfo) -> None:
        """
        Add a component to the tracking system.

        Args:
            component: ComponentInfo to add
        """
        self.components[component.component_id] = component
        logger.info(f"Added component {component.component_id} ({component.category.value})")

    def add_requirement(self, requirement: DiversityRequirement) -> None:
        """
        Add a diversity requirement.

        Args:
            requirement: DiversityRequirement to add
        """
        self.requirements.append(requirement)
        logger.info(f"Added requirement {requirement.requirement_id}")

    def add_shutdown_path(self, path: ShutdownPathInfo) -> None:
        """
        Add a shutdown path for diverse path verification.

        Args:
            path: ShutdownPathInfo to add
        """
        self.shutdown_paths.append(path)
        logger.info(f"Added shutdown path {path.path_id}")

    def evaluate_hardware_diversity(
        self,
        component_a: ComponentInfo,
        component_b: ComponentInfo
    ) -> DiversityEvaluation:
        """
        Evaluate hardware diversity between two components.

        Args:
            component_a: First component
            component_b: Second component

        Returns:
            DiversityEvaluation with results
        """
        logger.info(
            f"Evaluating hardware diversity: {component_a.component_id} vs {component_b.component_id}"
        )

        findings = []
        recommendations = []

        # Check manufacturer diversity
        is_manufacturer_diverse = (
            component_a.manufacturer.lower() != component_b.manufacturer.lower()
        )
        if is_manufacturer_diverse:
            findings.append(
                f"Different manufacturers: {component_a.manufacturer} vs {component_b.manufacturer}"
            )
        else:
            findings.append(f"Same manufacturer: {component_a.manufacturer}")
            recommendations.append("Consider using different manufacturers for redundant components")

        # Check model diversity
        is_model_diverse = component_a.model != component_b.model
        if is_model_diverse:
            findings.append(f"Different models: {component_a.model} vs {component_b.model}")
        else:
            findings.append(f"Same model: {component_a.model}")

        # Check technology diversity
        is_technology_diverse = (
            component_a.technology != component_b.technology and
            component_a.technology != "" and
            component_b.technology != ""
        )
        if is_technology_diverse:
            findings.append(
                f"Different technologies: {component_a.technology} vs {component_b.technology}"
            )
        elif component_a.technology:
            findings.append(f"Same technology: {component_a.technology}")
            recommendations.append("Consider using different sensing principles for higher diversity")

        # Check batch diversity
        is_batch_diverse = (
            component_a.batch_lot != component_b.batch_lot and
            component_a.batch_lot is not None and
            component_b.batch_lot is not None
        )
        if is_batch_diverse:
            findings.append(f"Different batches: {component_a.batch_lot} vs {component_b.batch_lot}")

        # Determine diversity level
        if is_technology_diverse:
            achieved_level = DiversityLevel.HIGH
        elif is_manufacturer_diverse:
            achieved_level = DiversityLevel.MEDIUM
        elif is_model_diverse or is_batch_diverse:
            achieved_level = DiversityLevel.LOW
        else:
            achieved_level = DiversityLevel.NONE

        diversity_score = self.LEVEL_SCORES[achieved_level]

        evaluation = DiversityEvaluation(
            component_a_id=component_a.component_id,
            component_b_id=component_b.component_id,
            diversity_type=DiversityType.HARDWARE,
            achieved_level=achieved_level,
            diversity_score=diversity_score,
            is_manufacturer_diverse=is_manufacturer_diverse,
            is_model_diverse=is_model_diverse,
            is_technology_diverse=is_technology_diverse,
            is_batch_diverse=is_batch_diverse,
            findings=findings,
            recommendations=recommendations,
        )

        evaluation.provenance_hash = self._calculate_provenance(evaluation)
        self.evaluations.append(evaluation)

        logger.info(f"Hardware diversity: {achieved_level.value} (score: {diversity_score})")

        return evaluation

    def evaluate_software_diversity(
        self,
        component_a: ComponentInfo,
        component_b: ComponentInfo
    ) -> DiversityEvaluation:
        """
        Evaluate software diversity between two components.

        Args:
            component_a: First component
            component_b: Second component

        Returns:
            DiversityEvaluation with results
        """
        logger.info(
            f"Evaluating software diversity: {component_a.component_id} vs {component_b.component_id}"
        )

        findings = []
        recommendations = []

        # Check software version diversity
        is_software_diverse = (
            component_a.software_version != component_b.software_version and
            component_a.software_version is not None and
            component_b.software_version is not None
        )

        if is_software_diverse:
            findings.append(
                f"Different software versions: {component_a.software_version} vs {component_b.software_version}"
            )
        elif component_a.software_version:
            findings.append(f"Same software version: {component_a.software_version}")
            recommendations.append("Consider using different software versions or implementations")

        # Combined with manufacturer diversity for overall assessment
        is_manufacturer_diverse = (
            component_a.manufacturer.lower() != component_b.manufacturer.lower()
        )

        # Determine software diversity level
        if is_manufacturer_diverse and is_software_diverse:
            achieved_level = DiversityLevel.HIGH
        elif is_manufacturer_diverse or is_software_diverse:
            achieved_level = DiversityLevel.MEDIUM
        else:
            achieved_level = DiversityLevel.LOW

        diversity_score = self.LEVEL_SCORES[achieved_level]

        evaluation = DiversityEvaluation(
            component_a_id=component_a.component_id,
            component_b_id=component_b.component_id,
            diversity_type=DiversityType.SOFTWARE,
            achieved_level=achieved_level,
            diversity_score=diversity_score,
            is_manufacturer_diverse=is_manufacturer_diverse,
            is_software_diverse=is_software_diverse,
            findings=findings,
            recommendations=recommendations,
        )

        evaluation.provenance_hash = self._calculate_provenance(evaluation)
        self.evaluations.append(evaluation)

        return evaluation

    def evaluate_measurement_diversity(
        self,
        components: List[ComponentInfo]
    ) -> DiversityEvaluation:
        """
        Evaluate measurement diversity across multiple sensors.

        Args:
            components: List of sensor components

        Returns:
            DiversityEvaluation with results
        """
        if len(components) < 2:
            raise ValueError("At least 2 components required for diversity evaluation")

        logger.info(f"Evaluating measurement diversity for {len(components)} components")

        findings = []
        recommendations = []

        # Collect unique attributes
        manufacturers = set(c.manufacturer for c in components)
        technologies = set(c.technology for c in components if c.technology)
        models = set(c.model for c in components)

        is_manufacturer_diverse = len(manufacturers) > 1
        is_technology_diverse = len(technologies) > 1
        is_model_diverse = len(models) > 1

        if is_manufacturer_diverse:
            findings.append(f"Multiple manufacturers: {', '.join(manufacturers)}")
        else:
            findings.append(f"Single manufacturer: {list(manufacturers)[0]}")

        if is_technology_diverse:
            findings.append(f"Multiple technologies: {', '.join(technologies)}")
        else:
            recommendations.append("Consider using diverse measurement principles")

        # Determine level
        if is_technology_diverse:
            achieved_level = DiversityLevel.HIGH
        elif is_manufacturer_diverse:
            achieved_level = DiversityLevel.MEDIUM
        elif is_model_diverse:
            achieved_level = DiversityLevel.LOW
        else:
            achieved_level = DiversityLevel.NONE

        diversity_score = self.LEVEL_SCORES[achieved_level]

        evaluation = DiversityEvaluation(
            component_a_id=components[0].component_id,
            component_b_id=components[1].component_id if len(components) > 1 else "",
            diversity_type=DiversityType.MEASUREMENT,
            achieved_level=achieved_level,
            diversity_score=diversity_score,
            is_manufacturer_diverse=is_manufacturer_diverse,
            is_technology_diverse=is_technology_diverse,
            is_model_diverse=is_model_diverse,
            findings=findings,
            recommendations=recommendations,
        )

        evaluation.provenance_hash = self._calculate_provenance(evaluation)
        self.evaluations.append(evaluation)

        return evaluation

    def verify_diverse_shutdown_paths(self) -> Dict[str, Any]:
        """
        Verify that diverse shutdown paths exist.

        Returns:
            Verification result dictionary
        """
        logger.info("Verifying diverse shutdown paths")

        if len(self.shutdown_paths) < 2:
            return {
                "verified": False,
                "paths_count": len(self.shutdown_paths),
                "message": "At least 2 shutdown paths required for diversity",
                "recommendations": ["Add additional shutdown path"]
            }

        # Check technology diversity between paths
        technologies = set(p.technology for p in self.shutdown_paths if p.technology)
        is_technology_diverse = len(technologies) > 1

        # Check component diversity
        all_components: Set[str] = set()
        for path in self.shutdown_paths:
            all_components.update(path.components)

        # Get unique components per path
        path_components = [set(p.components) for p in self.shutdown_paths]
        is_component_diverse = len(path_components) > 1 and not any(
            p1 == p2 for i, p1 in enumerate(path_components)
            for j, p2 in enumerate(path_components) if i < j
        )

        verified = is_technology_diverse or is_component_diverse

        result = {
            "verified": verified,
            "paths_count": len(self.shutdown_paths),
            "technologies": list(technologies),
            "is_technology_diverse": is_technology_diverse,
            "is_component_diverse": is_component_diverse,
            "paths": [
                {
                    "path_id": p.path_id,
                    "name": p.path_name,
                    "technology": p.technology,
                    "is_primary": p.is_primary,
                    "response_time_ms": p.response_time_ms,
                }
                for p in self.shutdown_paths
            ],
            "recommendations": []
        }

        if not verified:
            result["recommendations"].append(
                "Implement diverse shutdown paths using different technologies"
            )

        return result

    def calculate_diversity_score(
        self,
        system_id: str,
        target_sil: int
    ) -> DiversityReport:
        """
        Calculate overall diversity score for a system.

        Args:
            system_id: System identifier
            target_sil: Target SIL level

        Returns:
            DiversityReport with comprehensive assessment
        """
        logger.info(f"Calculating diversity score for {system_id} (SIL {target_sil})")

        # Get requirements for SIL
        sil_requirements = self.SIL_DIVERSITY_REQUIREMENTS.get(target_sil, {})

        # Evaluate by category
        hardware_evals = [
            e for e in self.evaluations
            if e.diversity_type == DiversityType.HARDWARE
        ]
        software_evals = [
            e for e in self.evaluations
            if e.diversity_type == DiversityType.SOFTWARE
        ]
        measurement_evals = [
            e for e in self.evaluations
            if e.diversity_type == DiversityType.MEASUREMENT
        ]

        # Calculate average scores
        hw_score = (
            sum(e.diversity_score for e in hardware_evals) // len(hardware_evals)
            if hardware_evals else 0
        )
        sw_score = (
            sum(e.diversity_score for e in software_evals) // len(software_evals)
            if software_evals else 0
        )
        meas_score = (
            sum(e.diversity_score for e in measurement_evals) // len(measurement_evals)
            if measurement_evals else 0
        )

        # Verify shutdown paths
        path_result = self.verify_diverse_shutdown_paths()
        path_verified = path_result.get("verified", False)

        # Check requirements
        gaps = []
        recommendations = []
        requirements_met = 0

        # Hardware requirement
        required_hw = sil_requirements.get("hardware", DiversityLevel.LOW)
        hw_level = self._score_to_level(hw_score)
        if self.LEVEL_SCORES[hw_level] >= self.LEVEL_SCORES[required_hw]:
            requirements_met += 1
        else:
            gaps.append(
                f"Hardware diversity: {hw_level.value} < {required_hw.value} required"
            )
            recommendations.append(
                f"Improve hardware diversity to {required_hw.value} level"
            )

        # Software requirement
        required_sw = sil_requirements.get("software", DiversityLevel.NONE)
        sw_level = self._score_to_level(sw_score)
        if self.LEVEL_SCORES[sw_level] >= self.LEVEL_SCORES[required_sw]:
            requirements_met += 1
        else:
            gaps.append(
                f"Software diversity: {sw_level.value} < {required_sw.value} required"
            )
            recommendations.append(
                f"Improve software diversity to {required_sw.value} level"
            )

        # Measurement requirement
        required_meas = sil_requirements.get("measurement", DiversityLevel.NONE)
        meas_level = self._score_to_level(meas_score)
        if self.LEVEL_SCORES[meas_level] >= self.LEVEL_SCORES[required_meas]:
            requirements_met += 1
        else:
            gaps.append(
                f"Measurement diversity: {meas_level.value} < {required_meas.value} required"
            )
            recommendations.append(
                f"Improve measurement diversity to {required_meas.value} level"
            )

        requirements_total = 3  # hardware, software, measurement

        # Calculate overall score
        max_possible = 9  # 3 categories x 3 max score
        actual = hw_score + sw_score + meas_score
        overall_score = (actual / max_possible) * 100 if max_possible > 0 else 0

        # Determine overall status
        if requirements_met == requirements_total and path_verified:
            overall_status = DiversityStatus.MET
        elif requirements_met >= requirements_total // 2:
            overall_status = DiversityStatus.PARTIALLY_MET
        else:
            overall_status = DiversityStatus.NOT_MET

        report = DiversityReport(
            system_id=system_id,
            target_sil=target_sil,
            overall_score=overall_score,
            overall_status=overall_status,
            hardware_diversity_score=hw_score,
            software_diversity_score=sw_score,
            measurement_diversity_score=meas_score,
            path_diversity_verified=path_verified,
            requirements_met=requirements_met,
            requirements_total=requirements_total,
            evaluations=self.evaluations,
            gaps=gaps,
            recommendations=recommendations + path_result.get("recommendations", []),
        )

        report.provenance_hash = self._calculate_report_provenance(report)

        logger.info(
            f"Diversity assessment complete: {overall_status.value} "
            f"({requirements_met}/{requirements_total} requirements met)"
        )

        return report

    def get_default_requirements(self, target_sil: int) -> List[DiversityRequirement]:
        """
        Get default diversity requirements for target SIL.

        Args:
            target_sil: Target SIL level

        Returns:
            List of default requirements
        """
        requirements = []

        sil_reqs = self.SIL_DIVERSITY_REQUIREMENTS.get(target_sil, {})

        for category, level in [
            (ComponentCategory.SENSOR, sil_reqs.get("measurement", DiversityLevel.LOW)),
            (ComponentCategory.TRANSMITTER, sil_reqs.get("hardware", DiversityLevel.LOW)),
            (ComponentCategory.LOGIC_SOLVER, sil_reqs.get("hardware", DiversityLevel.LOW)),
            (ComponentCategory.FINAL_ELEMENT, sil_reqs.get("hardware", DiversityLevel.LOW)),
            (ComponentCategory.SOFTWARE, sil_reqs.get("software", DiversityLevel.NONE)),
        ]:
            if level != DiversityLevel.NONE:
                requirements.append(DiversityRequirement(
                    diversity_type=DiversityType.HARDWARE if category != ComponentCategory.SOFTWARE else DiversityType.SOFTWARE,
                    component_category=category,
                    required_level=level,
                    description=f"{category.value} diversity requirement for SIL {target_sil}",
                    target_sil=target_sil,
                ))

        return requirements

    def _score_to_level(self, score: int) -> DiversityLevel:
        """Convert score to diversity level."""
        if score >= 3:
            return DiversityLevel.HIGH
        elif score >= 2:
            return DiversityLevel.MEDIUM
        elif score >= 1:
            return DiversityLevel.LOW
        else:
            return DiversityLevel.NONE

    def _calculate_provenance(self, evaluation: DiversityEvaluation) -> str:
        """Calculate SHA-256 provenance hash for evaluation."""
        provenance_str = (
            f"{evaluation.evaluation_id}|"
            f"{evaluation.component_a_id}|"
            f"{evaluation.component_b_id}|"
            f"{evaluation.diversity_score}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _calculate_report_provenance(self, report: DiversityReport) -> str:
        """Calculate SHA-256 provenance hash for report."""
        provenance_str = (
            f"{report.report_id}|"
            f"{report.system_id}|"
            f"{report.overall_score}|"
            f"{report.requirements_met}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def export_report(self, report: DiversityReport) -> Dict[str, Any]:
        """
        Export diversity report as dictionary.

        Args:
            report: DiversityReport to export

        Returns:
            Dictionary representation
        """
        return {
            "report_id": report.report_id,
            "system_id": report.system_id,
            "target_sil": report.target_sil,
            "overall_score": report.overall_score,
            "overall_status": report.overall_status.value,
            "scores": {
                "hardware": report.hardware_diversity_score,
                "software": report.software_diversity_score,
                "measurement": report.measurement_diversity_score,
            },
            "path_diversity_verified": report.path_diversity_verified,
            "compliance": {
                "requirements_met": report.requirements_met,
                "requirements_total": report.requirements_total,
                "percentage": (
                    report.requirements_met / report.requirements_total * 100
                    if report.requirements_total > 0 else 0
                ),
            },
            "gaps": report.gaps,
            "recommendations": report.recommendations,
            "timestamp": report.report_timestamp.isoformat(),
            "provenance_hash": report.provenance_hash,
        }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "DiversityManager",
    "ComponentInfo",
    "DiversityRequirement",
    "DiversityEvaluation",
    "ShutdownPathInfo",
    "DiversityReport",
    "DiversityType",
    "DiversityLevel",
    "DiversityStatus",
    "ComponentCategory",
]
