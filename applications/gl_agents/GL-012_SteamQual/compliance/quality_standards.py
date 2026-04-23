"""
GL-012 SteamQual - Steam Quality Standards Compliance

Comprehensive implementation of steam quality standards compliance including
ASME PTC 19.11, industry best practices, and site-specific requirements.

Regulatory References:
- ASME PTC 19.11: Steam and Water Sampling, Conditioning, and Analysis
- ASME Consensus on Operating Practices for Control of Feedwater Quality
- EPRI Guidelines for Steam Purity
- DOE Steam System Best Practices

This module provides:
1. ASME PTC 19.11 steam quality compliance validation
2. Industry best practice thresholds for steam purity
3. Site-specific quality requirement configuration
4. Quality deviation detection and alerting
5. Full provenance tracking for audit trails

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# STEAM QUALITY STANDARD ENUMERATIONS
# =============================================================================

class SteamQualityStandard(Enum):
    """Steam quality regulatory and industry standards."""

    # ASME Standards
    ASME_PTC_19_11 = "ASME PTC 19.11"
    ASME_CRTD_34 = "ASME CRTD-34"  # Consensus Guidelines

    # Industry Standards
    EPRI_GUIDELINES = "EPRI Steam Purity Guidelines"
    VGB_R450L = "VGB-R 450 L"  # European Standard
    JIS_B8223 = "JIS B 8223"  # Japanese Industrial Standard

    # DOE Guidelines
    DOE_STEAM_BEST = "DOE Steam Best Practices"

    # Site-Specific
    SITE_SPECIFIC = "Site-Specific Requirements"


class SteamPressureClass(Enum):
    """Steam pressure classification per ASME guidelines."""

    LOW_PRESSURE = "low_pressure"        # < 150 psig (1.0 MPa)
    MEDIUM_PRESSURE = "medium_pressure"  # 150-600 psig (1.0-4.1 MPa)
    HIGH_PRESSURE = "high_pressure"      # 600-1500 psig (4.1-10.3 MPa)
    VERY_HIGH_PRESSURE = "very_high"     # > 1500 psig (> 10.3 MPa)
    SUPERCRITICAL = "supercritical"      # > 3208 psia (22.1 MPa)


class SteamApplication(Enum):
    """Steam application categories affecting quality requirements."""

    PROCESS_HEATING = "process_heating"
    POWER_GENERATION = "power_generation"
    DIRECT_INJECTION = "direct_injection"  # Food/pharma
    TURBINE_DRIVE = "turbine_drive"
    HEAT_EXCHANGE = "heat_exchange"
    HUMIDIFICATION = "humidification"
    STERILIZATION = "sterilization"


class QualityParameter(Enum):
    """Steam quality measurement parameters."""

    DRYNESS_FRACTION = "dryness_fraction"          # Steam quality (x)
    TOTAL_DISSOLVED_SOLIDS = "total_dissolved_solids"  # TDS (ppm)
    SILICA = "silica"                              # SiO2 (ppb)
    SODIUM = "sodium"                              # Na (ppb)
    SPECIFIC_CONDUCTANCE = "specific_conductance"  # umhos/cm
    CATION_CONDUCTIVITY = "cation_conductivity"    # umhos/cm
    PH = "ph"                                      # pH units
    IRON = "iron"                                  # Fe (ppb)
    COPPER = "copper"                              # Cu (ppb)
    DISSOLVED_OXYGEN = "dissolved_oxygen"          # DO (ppb)
    AMMONIA = "ammonia"                            # NH3 (ppm)
    SUPERHEAT = "superheat"                        # Degrees above saturation
    MOISTURE_CONTENT = "moisture_content"          # % moisture


class ComplianceStatus(Enum):
    """Compliance status classification."""

    COMPLIANT = "compliant"
    WARNING = "warning"           # Within 10% of limit
    NON_COMPLIANT = "non_compliant"
    CRITICAL = "critical"         # > 150% of limit
    NOT_MEASURED = "not_measured"


# =============================================================================
# ASME PTC 19.11 QUALITY LIMITS
# =============================================================================

@dataclass
class QualityLimit:
    """
    Steam quality limit specification with regulatory reference.

    Provides traceable limits for steam quality parameters
    with full regulatory citation.
    """

    parameter: QualityParameter
    limit_value: Decimal
    unit: str
    limit_type: str  # "max", "min", "range"
    pressure_class: SteamPressureClass
    application: Optional[SteamApplication]
    standard: SteamQualityStandard
    reference_section: str
    notes: Optional[str] = None
    warning_threshold_percent: Decimal = Decimal("90")  # Warn at 90% of limit

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "parameter": self.parameter.value,
            "limit_value": str(self.limit_value),
            "unit": self.unit,
            "limit_type": self.limit_type,
            "pressure_class": self.pressure_class.value,
            "application": self.application.value if self.application else None,
            "standard": self.standard.value,
            "reference_section": self.reference_section,
            "notes": self.notes,
            "warning_threshold_percent": str(self.warning_threshold_percent),
        }


# ASME PTC 19.11 Steam Purity Limits by Pressure Class
ASME_QUALITY_LIMITS: List[QualityLimit] = [
    # Low Pressure Steam (< 150 psig)
    QualityLimit(
        parameter=QualityParameter.DRYNESS_FRACTION,
        limit_value=Decimal("0.95"),
        unit="fraction",
        limit_type="min",
        pressure_class=SteamPressureClass.LOW_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.2",
        notes="Minimum dryness fraction for process steam",
    ),
    QualityLimit(
        parameter=QualityParameter.TOTAL_DISSOLVED_SOLIDS,
        limit_value=Decimal("2.0"),
        unit="ppm",
        limit_type="max",
        pressure_class=SteamPressureClass.LOW_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.3",
    ),
    QualityLimit(
        parameter=QualityParameter.SILICA,
        limit_value=Decimal("200"),
        unit="ppb",
        limit_type="max",
        pressure_class=SteamPressureClass.LOW_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.4",
    ),

    # Medium Pressure Steam (150-600 psig)
    QualityLimit(
        parameter=QualityParameter.DRYNESS_FRACTION,
        limit_value=Decimal("0.97"),
        unit="fraction",
        limit_type="min",
        pressure_class=SteamPressureClass.MEDIUM_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.2",
    ),
    QualityLimit(
        parameter=QualityParameter.TOTAL_DISSOLVED_SOLIDS,
        limit_value=Decimal("0.5"),
        unit="ppm",
        limit_type="max",
        pressure_class=SteamPressureClass.MEDIUM_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.3",
    ),
    QualityLimit(
        parameter=QualityParameter.SILICA,
        limit_value=Decimal("50"),
        unit="ppb",
        limit_type="max",
        pressure_class=SteamPressureClass.MEDIUM_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.4",
    ),
    QualityLimit(
        parameter=QualityParameter.SPECIFIC_CONDUCTANCE,
        limit_value=Decimal("1.5"),
        unit="umhos/cm",
        limit_type="max",
        pressure_class=SteamPressureClass.MEDIUM_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.5",
    ),

    # High Pressure Steam (600-1500 psig)
    QualityLimit(
        parameter=QualityParameter.DRYNESS_FRACTION,
        limit_value=Decimal("0.995"),
        unit="fraction",
        limit_type="min",
        pressure_class=SteamPressureClass.HIGH_PRESSURE,
        application=SteamApplication.TURBINE_DRIVE,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.2",
        notes="Critical for turbine protection",
    ),
    QualityLimit(
        parameter=QualityParameter.TOTAL_DISSOLVED_SOLIDS,
        limit_value=Decimal("0.1"),
        unit="ppm",
        limit_type="max",
        pressure_class=SteamPressureClass.HIGH_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.3",
    ),
    QualityLimit(
        parameter=QualityParameter.SILICA,
        limit_value=Decimal("20"),
        unit="ppb",
        limit_type="max",
        pressure_class=SteamPressureClass.HIGH_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.4",
        notes="Critical for turbine blade deposits",
    ),
    QualityLimit(
        parameter=QualityParameter.SODIUM,
        limit_value=Decimal("10"),
        unit="ppb",
        limit_type="max",
        pressure_class=SteamPressureClass.HIGH_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.6",
    ),
    QualityLimit(
        parameter=QualityParameter.CATION_CONDUCTIVITY,
        limit_value=Decimal("0.3"),
        unit="umhos/cm",
        limit_type="max",
        pressure_class=SteamPressureClass.HIGH_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.5",
    ),

    # Very High Pressure Steam (> 1500 psig)
    QualityLimit(
        parameter=QualityParameter.DRYNESS_FRACTION,
        limit_value=Decimal("0.998"),
        unit="fraction",
        limit_type="min",
        pressure_class=SteamPressureClass.VERY_HIGH_PRESSURE,
        application=SteamApplication.TURBINE_DRIVE,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.2",
    ),
    QualityLimit(
        parameter=QualityParameter.TOTAL_DISSOLVED_SOLIDS,
        limit_value=Decimal("0.05"),
        unit="ppm",
        limit_type="max",
        pressure_class=SteamPressureClass.VERY_HIGH_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.3",
    ),
    QualityLimit(
        parameter=QualityParameter.SILICA,
        limit_value=Decimal("10"),
        unit="ppb",
        limit_type="max",
        pressure_class=SteamPressureClass.VERY_HIGH_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.4",
    ),
    QualityLimit(
        parameter=QualityParameter.SODIUM,
        limit_value=Decimal("5"),
        unit="ppb",
        limit_type="max",
        pressure_class=SteamPressureClass.VERY_HIGH_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.6",
    ),
    QualityLimit(
        parameter=QualityParameter.CATION_CONDUCTIVITY,
        limit_value=Decimal("0.15"),
        unit="umhos/cm",
        limit_type="max",
        pressure_class=SteamPressureClass.VERY_HIGH_PRESSURE,
        application=None,
        standard=SteamQualityStandard.ASME_PTC_19_11,
        reference_section="Section 4.5",
    ),
]


# =============================================================================
# INDUSTRY BEST PRACTICE THRESHOLDS
# =============================================================================

@dataclass
class BestPracticeThreshold:
    """
    Industry best practice threshold for steam quality.

    Represents achievable targets beyond minimum compliance,
    used for optimization and benchmarking.
    """

    parameter: QualityParameter
    target_value: Decimal
    achievable_value: Decimal  # Top quartile performance
    unit: str
    pressure_class: SteamPressureClass
    source: str
    potential_benefit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "parameter": self.parameter.value,
            "target_value": str(self.target_value),
            "achievable_value": str(self.achievable_value),
            "unit": self.unit,
            "pressure_class": self.pressure_class.value,
            "source": self.source,
            "potential_benefit": self.potential_benefit,
        }


INDUSTRY_BEST_PRACTICES: List[BestPracticeThreshold] = [
    BestPracticeThreshold(
        parameter=QualityParameter.DRYNESS_FRACTION,
        target_value=Decimal("0.98"),
        achievable_value=Decimal("0.995"),
        unit="fraction",
        pressure_class=SteamPressureClass.MEDIUM_PRESSURE,
        source="DOE Steam Best Practices",
        potential_benefit="1% dryness improvement = 0.5-1% energy savings",
    ),
    BestPracticeThreshold(
        parameter=QualityParameter.MOISTURE_CONTENT,
        target_value=Decimal("2.0"),
        achievable_value=Decimal("0.5"),
        unit="%",
        pressure_class=SteamPressureClass.HIGH_PRESSURE,
        source="EPRI Guidelines",
        potential_benefit="Reduced erosion, improved heat transfer",
    ),
    BestPracticeThreshold(
        parameter=QualityParameter.SUPERHEAT,
        target_value=Decimal("20"),
        achievable_value=Decimal("50"),
        unit="deg_F",
        pressure_class=SteamPressureClass.HIGH_PRESSURE,
        source="Industry Practice",
        potential_benefit="Ensures dry steam at delivery point",
    ),
]


# =============================================================================
# SITE-SPECIFIC QUALITY REQUIREMENTS
# =============================================================================

@dataclass
class SiteQualityRequirement:
    """
    Site-specific steam quality requirement.

    Allows configuration of quality limits specific to
    equipment, processes, or regulatory requirements at a site.
    """

    requirement_id: str
    site_id: str
    equipment_id: Optional[str]
    parameter: QualityParameter
    limit_value: Decimal
    unit: str
    limit_type: str  # "max", "min", "range"
    rationale: str
    effective_date: datetime
    expiration_date: Optional[datetime] = None
    approved_by: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "requirement_id": self.requirement_id,
            "site_id": self.site_id,
            "equipment_id": self.equipment_id,
            "parameter": self.parameter.value,
            "limit_value": str(self.limit_value),
            "unit": self.unit,
            "limit_type": self.limit_type,
            "rationale": self.rationale,
            "effective_date": self.effective_date.isoformat(),
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "approved_by": self.approved_by,
            "created_at": self.created_at.isoformat(),
        }

    def is_active(self) -> bool:
        """Check if requirement is currently active."""
        now = datetime.now(timezone.utc)
        if now < self.effective_date:
            return False
        if self.expiration_date and now > self.expiration_date:
            return False
        return True


# =============================================================================
# QUALITY VALIDATION RESULTS
# =============================================================================

@dataclass
class QualityValidationResult:
    """
    Result of steam quality validation against standards.

    Provides complete audit trail for quality measurement
    validation including provenance hash.
    """

    validation_id: str
    timestamp: datetime
    parameter: QualityParameter
    measured_value: Decimal
    limit_value: Decimal
    unit: str
    limit_type: str
    status: ComplianceStatus
    deviation_percent: Decimal
    standard: SteamQualityStandard
    reference_section: str
    pressure_class: SteamPressureClass
    provenance_hash: str
    sensor_id: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "validation_id": self.validation_id,
            "timestamp": self.timestamp.isoformat(),
            "parameter": self.parameter.value,
            "measured_value": str(self.measured_value),
            "limit_value": str(self.limit_value),
            "unit": self.unit,
            "limit_type": self.limit_type,
            "status": self.status.value,
            "deviation_percent": str(self.deviation_percent),
            "standard": self.standard.value,
            "reference_section": self.reference_section,
            "pressure_class": self.pressure_class.value,
            "provenance_hash": self.provenance_hash,
            "sensor_id": self.sensor_id,
            "notes": self.notes,
        }


# =============================================================================
# STEAM QUALITY STANDARDS VALIDATOR
# =============================================================================

class SteamQualityStandardsValidator:
    """
    Validates steam quality measurements against applicable standards.

    Provides:
    - ASME PTC 19.11 compliance validation
    - Industry best practice comparison
    - Site-specific requirement checking
    - Complete audit trail with provenance

    This validator follows zero-hallucination principles by using
    only deterministic comparisons against regulatory limits.

    Example:
        >>> validator = SteamQualityStandardsValidator()
        >>> result = validator.validate_parameter(
        ...     QualityParameter.DRYNESS_FRACTION,
        ...     Decimal("0.96"),
        ...     SteamPressureClass.MEDIUM_PRESSURE
        ... )
        >>> print(f"Status: {result.status.value}")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        site_requirements: Optional[List[SiteQualityRequirement]] = None,
    ) -> None:
        """
        Initialize steam quality standards validator.

        Args:
            site_requirements: Optional site-specific requirements
        """
        self._asme_limits = ASME_QUALITY_LIMITS
        self._best_practices = INDUSTRY_BEST_PRACTICES
        self._site_requirements = site_requirements or []

        # Build lookup indices for efficient access
        self._limit_index = self._build_limit_index()

        logger.info(
            f"SteamQualityStandardsValidator initialized with "
            f"{len(self._asme_limits)} ASME limits, "
            f"{len(self._site_requirements)} site requirements"
        )

    def _build_limit_index(self) -> Dict[Tuple[str, str], List[QualityLimit]]:
        """Build index for efficient limit lookup."""
        index: Dict[Tuple[str, str], List[QualityLimit]] = {}

        for limit in self._asme_limits:
            key = (limit.parameter.value, limit.pressure_class.value)
            if key not in index:
                index[key] = []
            index[key].append(limit)

        return index

    def get_applicable_limits(
        self,
        parameter: QualityParameter,
        pressure_class: SteamPressureClass,
        application: Optional[SteamApplication] = None,
    ) -> List[QualityLimit]:
        """
        Get applicable quality limits for parameter and pressure class.

        Args:
            parameter: Quality parameter to check
            pressure_class: Steam pressure classification
            application: Optional application filter

        Returns:
            List of applicable QualityLimit objects
        """
        key = (parameter.value, pressure_class.value)
        limits = self._limit_index.get(key, [])

        if application:
            # Filter by application if specified
            limits = [
                lim for lim in limits
                if lim.application is None or lim.application == application
            ]

        return limits

    def validate_parameter(
        self,
        parameter: QualityParameter,
        measured_value: Union[Decimal, float],
        pressure_class: SteamPressureClass,
        application: Optional[SteamApplication] = None,
        sensor_id: Optional[str] = None,
    ) -> QualityValidationResult:
        """
        Validate measured value against applicable standards.

        Uses zero-hallucination deterministic comparison against
        regulatory limits from ASME PTC 19.11.

        Args:
            parameter: Quality parameter being measured
            measured_value: Measured value
            pressure_class: Steam pressure classification
            application: Optional application context
            sensor_id: Optional sensor identifier

        Returns:
            QualityValidationResult with compliance status

        Raises:
            ValueError: If no applicable limit found
        """
        timestamp = datetime.now(timezone.utc)
        measured_dec = Decimal(str(measured_value))

        # Find applicable limit
        limits = self.get_applicable_limits(parameter, pressure_class, application)

        if not limits:
            # No ASME limit - check site-specific
            site_limit = self._get_site_limit(parameter, pressure_class)
            if site_limit:
                return self._validate_against_site_limit(
                    site_limit, measured_dec, sensor_id
                )
            raise ValueError(
                f"No applicable limit for {parameter.value} at {pressure_class.value}"
            )

        # Use first applicable limit (most specific)
        limit = limits[0]

        # Determine compliance status
        status, deviation = self._calculate_compliance(
            measured_dec, limit.limit_value, limit.limit_type, limit.warning_threshold_percent
        )

        # Calculate provenance hash
        provenance_data = {
            "parameter": parameter.value,
            "measured_value": str(measured_dec),
            "limit_value": str(limit.limit_value),
            "pressure_class": pressure_class.value,
            "timestamp": timestamp.isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return QualityValidationResult(
            validation_id=f"QV-{timestamp.strftime('%Y%m%d%H%M%S')}-{parameter.value[:4].upper()}",
            timestamp=timestamp,
            parameter=parameter,
            measured_value=measured_dec,
            limit_value=limit.limit_value,
            unit=limit.unit,
            limit_type=limit.limit_type,
            status=status,
            deviation_percent=deviation,
            standard=limit.standard,
            reference_section=limit.reference_section,
            pressure_class=pressure_class,
            provenance_hash=provenance_hash,
            sensor_id=sensor_id,
            notes=limit.notes,
        )

    def _calculate_compliance(
        self,
        measured: Decimal,
        limit: Decimal,
        limit_type: str,
        warning_threshold: Decimal,
    ) -> Tuple[ComplianceStatus, Decimal]:
        """
        Calculate compliance status and deviation.

        Zero-hallucination: Pure arithmetic comparison.

        Args:
            measured: Measured value
            limit: Limit value
            limit_type: "max" or "min"
            warning_threshold: Warning threshold as percentage

        Returns:
            Tuple of (ComplianceStatus, deviation_percent)
        """
        if limit == Decimal("0"):
            # Avoid division by zero
            if measured == Decimal("0"):
                return ComplianceStatus.COMPLIANT, Decimal("0")
            return ComplianceStatus.NON_COMPLIANT, Decimal("100")

        if limit_type == "max":
            # For max limits: value should be below limit
            deviation = ((measured - limit) / limit) * Decimal("100")

            if measured <= limit:
                # Within limit - check if in warning zone
                usage_percent = (measured / limit) * Decimal("100")
                if usage_percent >= warning_threshold:
                    return ComplianceStatus.WARNING, deviation.quantize(Decimal("0.01"))
                return ComplianceStatus.COMPLIANT, deviation.quantize(Decimal("0.01"))
            elif measured <= limit * Decimal("1.5"):
                return ComplianceStatus.NON_COMPLIANT, deviation.quantize(Decimal("0.01"))
            else:
                return ComplianceStatus.CRITICAL, deviation.quantize(Decimal("0.01"))

        elif limit_type == "min":
            # For min limits: value should be above limit
            deviation = ((limit - measured) / limit) * Decimal("100")

            if measured >= limit:
                return ComplianceStatus.COMPLIANT, Decimal("0")
            elif measured >= limit * Decimal("0.9"):
                return ComplianceStatus.WARNING, deviation.quantize(Decimal("0.01"))
            elif measured >= limit * Decimal("0.5"):
                return ComplianceStatus.NON_COMPLIANT, deviation.quantize(Decimal("0.01"))
            else:
                return ComplianceStatus.CRITICAL, deviation.quantize(Decimal("0.01"))

        return ComplianceStatus.COMPLIANT, Decimal("0")

    def _get_site_limit(
        self,
        parameter: QualityParameter,
        pressure_class: SteamPressureClass,
    ) -> Optional[SiteQualityRequirement]:
        """Get active site-specific limit for parameter."""
        for req in self._site_requirements:
            if req.parameter == parameter and req.is_active():
                return req
        return None

    def _validate_against_site_limit(
        self,
        site_req: SiteQualityRequirement,
        measured: Decimal,
        sensor_id: Optional[str],
    ) -> QualityValidationResult:
        """Validate against site-specific requirement."""
        timestamp = datetime.now(timezone.utc)

        status, deviation = self._calculate_compliance(
            measured, site_req.limit_value, site_req.limit_type, Decimal("90")
        )

        provenance_data = {
            "requirement_id": site_req.requirement_id,
            "measured_value": str(measured),
            "limit_value": str(site_req.limit_value),
            "timestamp": timestamp.isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return QualityValidationResult(
            validation_id=f"QV-SITE-{timestamp.strftime('%Y%m%d%H%M%S')}",
            timestamp=timestamp,
            parameter=site_req.parameter,
            measured_value=measured,
            limit_value=site_req.limit_value,
            unit=site_req.unit,
            limit_type=site_req.limit_type,
            status=status,
            deviation_percent=deviation,
            standard=SteamQualityStandard.SITE_SPECIFIC,
            reference_section=site_req.requirement_id,
            pressure_class=SteamPressureClass.MEDIUM_PRESSURE,  # Default
            provenance_hash=provenance_hash,
            sensor_id=sensor_id,
            notes=site_req.rationale,
        )

    def validate_quality_profile(
        self,
        measurements: Dict[QualityParameter, Decimal],
        pressure_class: SteamPressureClass,
        application: Optional[SteamApplication] = None,
    ) -> List[QualityValidationResult]:
        """
        Validate complete quality profile against all applicable standards.

        Args:
            measurements: Dictionary of parameter to measured value
            pressure_class: Steam pressure classification
            application: Optional application context

        Returns:
            List of QualityValidationResult for each parameter
        """
        results = []

        for parameter, value in measurements.items():
            try:
                result = self.validate_parameter(
                    parameter, value, pressure_class, application
                )
                results.append(result)
            except ValueError as e:
                logger.warning(f"Could not validate {parameter.value}: {e}")

        return results

    def compare_to_best_practice(
        self,
        parameter: QualityParameter,
        measured_value: Union[Decimal, float],
        pressure_class: SteamPressureClass,
    ) -> Optional[Dict[str, Any]]:
        """
        Compare measured value to industry best practice.

        Args:
            parameter: Quality parameter
            measured_value: Measured value
            pressure_class: Steam pressure classification

        Returns:
            Comparison dictionary or None if no best practice defined
        """
        measured_dec = Decimal(str(measured_value))

        for bp in self._best_practices:
            if bp.parameter == parameter and bp.pressure_class == pressure_class:
                target_gap = measured_dec - bp.target_value
                achievable_gap = measured_dec - bp.achievable_value

                return {
                    "parameter": parameter.value,
                    "measured_value": str(measured_dec),
                    "target_value": str(bp.target_value),
                    "achievable_value": str(bp.achievable_value),
                    "gap_to_target": str(target_gap),
                    "gap_to_achievable": str(achievable_gap),
                    "unit": bp.unit,
                    "source": bp.source,
                    "potential_benefit": bp.potential_benefit,
                }

        return None

    def add_site_requirement(self, requirement: SiteQualityRequirement) -> None:
        """
        Add site-specific quality requirement.

        Args:
            requirement: SiteQualityRequirement to add
        """
        self._site_requirements.append(requirement)
        logger.info(f"Added site requirement: {requirement.requirement_id}")

    def generate_compliance_report(
        self,
        validations: List[QualityValidationResult],
        site_id: str,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality compliance report.

        Args:
            validations: List of validation results
            site_id: Site identifier

        Returns:
            Complete compliance report with provenance
        """
        timestamp = datetime.now(timezone.utc)

        # Summarize by status
        status_counts = {status.value: 0 for status in ComplianceStatus}
        for v in validations:
            status_counts[v.status.value] += 1

        overall_compliant = all(
            v.status in [ComplianceStatus.COMPLIANT, ComplianceStatus.WARNING]
            for v in validations
        )

        # Group by parameter
        by_parameter: Dict[str, List[Dict]] = {}
        for v in validations:
            param = v.parameter.value
            if param not in by_parameter:
                by_parameter[param] = []
            by_parameter[param].append(v.to_dict())

        # Calculate report hash
        report_data = {
            "site_id": site_id,
            "timestamp": timestamp.isoformat(),
            "validations": [v.to_dict() for v in validations],
        }
        report_hash = hashlib.sha256(
            json.dumps(report_data, sort_keys=True).encode()
        ).hexdigest()

        return {
            "report_metadata": {
                "report_id": f"QCR-{timestamp.strftime('%Y%m%d%H%M%S')}",
                "site_id": site_id,
                "generated_at": timestamp.isoformat(),
                "report_hash": report_hash,
                "generator_version": self.VERSION,
            },
            "summary": {
                "total_validations": len(validations),
                "compliant": status_counts[ComplianceStatus.COMPLIANT.value],
                "warning": status_counts[ComplianceStatus.WARNING.value],
                "non_compliant": status_counts[ComplianceStatus.NON_COMPLIANT.value],
                "critical": status_counts[ComplianceStatus.CRITICAL.value],
                "overall_compliant": overall_compliant,
            },
            "validations_by_parameter": by_parameter,
            "applicable_standards": [
                SteamQualityStandard.ASME_PTC_19_11.value,
                SteamQualityStandard.DOE_STEAM_BEST.value,
            ],
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_quality_validator(
    site_requirements: Optional[List[SiteQualityRequirement]] = None,
) -> SteamQualityStandardsValidator:
    """
    Factory function to create quality standards validator.

    Args:
        site_requirements: Optional site-specific requirements

    Returns:
        Configured SteamQualityStandardsValidator
    """
    return SteamQualityStandardsValidator(site_requirements)


def get_asme_limits_for_pressure_class(
    pressure_class: SteamPressureClass,
) -> List[QualityLimit]:
    """
    Get all ASME quality limits for a pressure class.

    Args:
        pressure_class: Steam pressure classification

    Returns:
        List of applicable QualityLimit objects
    """
    return [
        limit for limit in ASME_QUALITY_LIMITS
        if limit.pressure_class == pressure_class
    ]


def get_quality_limits_for_parameter(
    parameter: QualityParameter,
) -> List[QualityLimit]:
    """
    Get all quality limits for a specific parameter across pressure classes.

    Args:
        parameter: Quality parameter

    Returns:
        List of QualityLimit objects for the parameter
    """
    return [
        limit for limit in ASME_QUALITY_LIMITS
        if limit.parameter == parameter
    ]
