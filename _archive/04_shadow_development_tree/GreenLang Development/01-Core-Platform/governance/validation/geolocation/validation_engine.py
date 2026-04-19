# -*- coding: utf-8 -*-
"""
GreenLang EUDR Geolocation Validation Engine

Zero-hallucination orchestration of all geolocation validation checks
for EUDR compliance. Combines GeoJSON parsing, coordinate validation,
protected area checks, deforestation baseline, and country risk assessment.

This module provides:
- Orchestration of all validation checks
- Comprehensive validation report generation
- Overall compliance score calculation
- Complete audit trail with SHA-256 hashes
- Deterministic, reproducible results

Author: GreenLang Calculator Engine
License: Proprietary
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Set
from pydantic import BaseModel, Field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from datetime import datetime, date
import hashlib
import json
import time
import logging

from .geojson_parser import (
    GeoJSONParser,
    GeoJSONParseResult,
    Coordinate,
    ParsedPolygon,
    ParsedMultiPolygon,
    ParsedPoint,
    GeoJSONType
)
from .coordinate_validator import (
    CoordinateValidator,
    CoordinateValidationResult,
    PrecisionLevel,
    DistanceResult
)
from .protected_area_checker import (
    ProtectedAreaChecker,
    ProtectedAreaCheckResult,
    ProtectionStatus,
    ProtectionLevel
)
from .deforestation_baseline import (
    DeforestationBaselineChecker,
    BaselineCheckResult,
    ForestStatus,
    DeforestationRisk,
    EUDR_CUTOFF_DATE
)
from .country_risk_db import (
    CountryRiskDatabase,
    CountryRiskProfile,
    EUDRRiskCategory,
    EUDRCommodity,
    CommodityRisk
)

logger = logging.getLogger(__name__)


class ValidationCheckType(str, Enum):
    """Types of validation checks performed."""
    GEOJSON_PARSE = "geojson_parse"
    COORDINATE_VALIDATION = "coordinate_validation"
    PROTECTED_AREA = "protected_area"
    DEFORESTATION_BASELINE = "deforestation_baseline"
    COUNTRY_RISK = "country_risk"
    COMMODITY_RISK = "commodity_risk"


class ValidationSeverity(str, Enum):
    """Severity levels for validation findings."""
    ERROR = "error"  # Compliance violation
    WARNING = "warning"  # Potential issue
    INFO = "info"  # Informational


class ComplianceStatus(str, Enum):
    """Overall EUDR compliance status."""
    COMPLIANT = "compliant"  # Passes all checks
    NON_COMPLIANT = "non_compliant"  # Fails one or more critical checks
    NEEDS_REVIEW = "needs_review"  # Warnings require manual review
    UNKNOWN = "unknown"  # Unable to determine


class ValidationFinding(BaseModel):
    """A single validation finding/issue."""
    check_type: ValidationCheckType
    severity: ValidationSeverity
    code: str = Field(..., description="Finding code for categorization")
    message: str = Field(..., description="Human-readable message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    recommendation: Optional[str] = Field(None, description="Recommended action")


class CheckResult(BaseModel):
    """Result of a single validation check."""
    check_type: ValidationCheckType
    passed: bool
    score: Decimal = Field(Decimal('0'), ge=0, le=100)
    findings: List[ValidationFinding] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    data: Dict[str, Any] = Field(default_factory=dict, description="Check-specific data")


class GeoLocationValidationReport(BaseModel):
    """
    Complete EUDR geolocation validation report.

    Contains results of all validation checks with complete
    audit trail and provenance tracking.
    """
    # Report identification
    report_id: str = Field(..., description="Unique report identifier")
    report_version: str = Field("1.0", description="Report format version")
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Input summary
    input_type: GeoJSONType
    input_hash: str = Field(..., description="SHA-256 hash of input GeoJSON")
    country_iso3: Optional[str] = None
    commodity: Optional[EUDRCommodity] = None

    # Geometry details
    area_hectares: Optional[Decimal] = None
    centroid: Optional[Coordinate] = None
    perimeter_meters: Optional[Decimal] = None

    # Overall compliance
    compliance_status: ComplianceStatus
    compliance_score: Decimal = Field(..., ge=0, le=100, description="Overall compliance score (0-100)")
    is_eudr_compliant: bool

    # Individual check results
    check_results: Dict[ValidationCheckType, CheckResult] = Field(default_factory=dict)

    # All findings (aggregated)
    findings: List[ValidationFinding] = Field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    # Execution metrics
    total_execution_time_ms: float = 0.0

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash of complete report")
    data_sources: List[str] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidationEngineConfig(BaseModel):
    """Configuration for the validation engine."""
    # Check enablement
    enable_geojson_parse: bool = True
    enable_coordinate_validation: bool = True
    enable_protected_area_check: bool = True
    enable_deforestation_check: bool = True
    enable_country_risk_check: bool = True
    enable_commodity_risk_check: bool = True

    # Precision requirements
    required_precision: PrecisionLevel = PrecisionLevel.EUDR_COMPLIANT

    # Protected area settings
    protected_area_buffer_meters: Optional[int] = None

    # Risk thresholds
    max_acceptable_risk_score: Decimal = Decimal('50')
    max_acceptable_deforestation_risk: Decimal = Decimal('50')

    # Performance settings
    timeout_per_check_ms: int = 5000
    enable_caching: bool = True


class EUDRGeolocationValidator:
    """
    Zero-Hallucination EUDR Geolocation Validation Engine.

    Orchestrates all validation checks for EUDR compliance:
    1. GeoJSON parsing and validation
    2. Coordinate validation (WGS84, precision)
    3. Protected area intersection check
    4. Deforestation baseline check (Dec 31, 2020)
    5. Country risk assessment
    6. Commodity-specific risk assessment

    Guarantees:
    - Deterministic results (same input -> same output)
    - Complete audit trail with SHA-256 hashes
    - Bit-perfect reproducibility
    - NO LLM in validation path

    Example:
        validator = EUDRGeolocationValidator()

        geojson = {
            "type": "Polygon",
            "coordinates": [[[...], [...]]]
        }

        report = validator.validate(
            geojson=geojson,
            country_iso3="BRA",
            commodity=EUDRCommodity.SOYA
        )

        print(f"Compliance: {report.compliance_status}")
        print(f"Score: {report.compliance_score}")
        print(f"Provenance: {report.provenance_hash}")
    """

    def __init__(self, config: Optional[ValidationEngineConfig] = None):
        """
        Initialize validation engine.

        Args:
            config: Engine configuration (uses defaults if not provided)
        """
        self.config = config or ValidationEngineConfig()

        # Initialize sub-validators
        self.geojson_parser = GeoJSONParser(
            require_eudr_precision=self.config.required_precision == PrecisionLevel.EUDR_COMPLIANT
        )
        self.coordinate_validator = CoordinateValidator(
            default_precision=self.config.required_precision
        )
        self.protected_area_checker = ProtectedAreaChecker(
            cache_enabled=self.config.enable_caching
        )
        self.deforestation_checker = DeforestationBaselineChecker(
            cache_enabled=self.config.enable_caching
        )
        self.country_risk_db = CountryRiskDatabase()

    def validate(
        self,
        geojson: Dict[str, Any],
        country_iso3: Optional[str] = None,
        commodity: Optional[EUDRCommodity] = None,
        observation_date: Optional[date] = None
    ) -> GeoLocationValidationReport:
        """
        Perform complete EUDR geolocation validation.

        DETERMINISTIC VALIDATION.

        Args:
            geojson: GeoJSON geometry or Feature
            country_iso3: ISO3 country code (required for full validation)
            commodity: EUDR commodity type (optional)
            observation_date: Date for baseline comparison (default: today)

        Returns:
            GeoLocationValidationReport with complete results
        """
        start_time = time.perf_counter()

        if observation_date is None:
            observation_date = date.today()

        # Initialize report
        report = GeoLocationValidationReport(
            report_id=self._generate_report_id(),
            input_type=GeoJSONType.POINT,  # Will be updated
            input_hash=self._calculate_hash(geojson),
            country_iso3=country_iso3,
            commodity=commodity,
            compliance_status=ComplianceStatus.UNKNOWN,
            compliance_score=Decimal('0'),
            is_eudr_compliant=True,  # Assume compliant until proven otherwise
            provenance_hash=""
        )

        check_results = {}
        all_findings = []

        # Check 1: GeoJSON Parsing
        if self.config.enable_geojson_parse:
            parse_result = self._run_geojson_parse(geojson)
            check_results[ValidationCheckType.GEOJSON_PARSE] = parse_result
            all_findings.extend(parse_result.findings)

            if not parse_result.passed:
                report.compliance_status = ComplianceStatus.NON_COMPLIANT
                report.is_eudr_compliant = False
                report.check_results = check_results
                report.findings = all_findings
                self._finalize_report(report, start_time)
                return report

            # Extract geometry data
            parsed = parse_result.data.get("parsed_result")
            if parsed:
                report.input_type = parsed.geometry_type

                if parsed.polygon:
                    report.area_hectares = parsed.polygon.area_hectares
                    report.centroid = parsed.polygon.centroid
                    report.perimeter_meters = parsed.polygon.perimeter_meters
                elif parsed.multi_polygon:
                    report.area_hectares = parsed.multi_polygon.total_area_hectares
                    report.centroid = parsed.multi_polygon.combined_centroid
                elif parsed.point:
                    report.centroid = parsed.point.coordinate

        # Check 2: Coordinate Validation
        if self.config.enable_coordinate_validation and report.centroid:
            coord_result = self._run_coordinate_validation(report.centroid)
            check_results[ValidationCheckType.COORDINATE_VALIDATION] = coord_result
            all_findings.extend(coord_result.findings)

        # Check 3: Protected Area Check
        if self.config.enable_protected_area_check and report.centroid:
            protected_result = self._run_protected_area_check(
                report.centroid, country_iso3
            )
            check_results[ValidationCheckType.PROTECTED_AREA] = protected_result
            all_findings.extend(protected_result.findings)

            if not protected_result.passed:
                report.is_eudr_compliant = False

        # Check 4: Deforestation Baseline Check
        if self.config.enable_deforestation_check and report.centroid and country_iso3:
            deforestation_result = self._run_deforestation_check(
                report.centroid, country_iso3, observation_date
            )
            check_results[ValidationCheckType.DEFORESTATION_BASELINE] = deforestation_result
            all_findings.extend(deforestation_result.findings)
            report.data_sources.extend(
                deforestation_result.data.get("data_sources", [])
            )

            if not deforestation_result.passed:
                report.is_eudr_compliant = False

        # Check 5: Country Risk Check
        if self.config.enable_country_risk_check and country_iso3:
            country_result = self._run_country_risk_check(country_iso3)
            check_results[ValidationCheckType.COUNTRY_RISK] = country_result
            all_findings.extend(country_result.findings)

        # Check 6: Commodity Risk Check
        if self.config.enable_commodity_risk_check and country_iso3 and commodity:
            commodity_result = self._run_commodity_risk_check(country_iso3, commodity)
            check_results[ValidationCheckType.COMMODITY_RISK] = commodity_result
            all_findings.extend(commodity_result.findings)

        # Aggregate results
        report.check_results = check_results
        report.findings = all_findings

        # Count findings by severity
        report.error_count = sum(
            1 for f in all_findings if f.severity == ValidationSeverity.ERROR
        )
        report.warning_count = sum(
            1 for f in all_findings if f.severity == ValidationSeverity.WARNING
        )
        report.info_count = sum(
            1 for f in all_findings if f.severity == ValidationSeverity.INFO
        )

        # Calculate overall compliance score
        report.compliance_score = self._calculate_compliance_score(check_results)

        # Determine final compliance status
        report.compliance_status = self._determine_compliance_status(
            report.is_eudr_compliant,
            report.error_count,
            report.warning_count
        )

        # Finalize report
        self._finalize_report(report, start_time)

        return report

    def _run_geojson_parse(self, geojson: Dict[str, Any]) -> CheckResult:
        """Run GeoJSON parsing check."""
        start_time = time.perf_counter()
        findings = []

        try:
            result = self.geojson_parser.parse(geojson)

            if not result.success:
                for error in result.errors:
                    findings.append(ValidationFinding(
                        check_type=ValidationCheckType.GEOJSON_PARSE,
                        severity=ValidationSeverity.ERROR,
                        code="GEOJSON_PARSE_ERROR",
                        message=error,
                        recommendation="Fix the GeoJSON structure and re-submit"
                    ))

            for warning in result.warnings:
                findings.append(ValidationFinding(
                    check_type=ValidationCheckType.GEOJSON_PARSE,
                    severity=ValidationSeverity.WARNING,
                    code="GEOJSON_PARSE_WARNING",
                    message=warning
                ))

            # Check area constraints if polygon
            if result.polygon and result.polygon.area_hectares:
                area = result.polygon.area_hectares
                if area < Decimal('0.0001'):
                    findings.append(ValidationFinding(
                        check_type=ValidationCheckType.GEOJSON_PARSE,
                        severity=ValidationSeverity.WARNING,
                        code="AREA_TOO_SMALL",
                        message=f"Plot area ({area} ha) is very small",
                        details={"area_hectares": float(area)}
                    ))
                elif area > Decimal('10000'):
                    findings.append(ValidationFinding(
                        check_type=ValidationCheckType.GEOJSON_PARSE,
                        severity=ValidationSeverity.INFO,
                        code="AREA_LARGE",
                        message=f"Large plot area ({area} ha) - verify accuracy",
                        details={"area_hectares": float(area)}
                    ))

            # Check for self-intersection
            if result.polygon and result.polygon.has_self_intersection:
                findings.append(ValidationFinding(
                    check_type=ValidationCheckType.GEOJSON_PARSE,
                    severity=ValidationSeverity.ERROR,
                    code="SELF_INTERSECTION",
                    message="Polygon has self-intersection - invalid geometry",
                    recommendation="Fix polygon boundaries to remove self-intersection"
                ))

            score = Decimal('100') if result.success else Decimal('0')
            if result.polygon and result.polygon.has_self_intersection:
                score = Decimal('0')

            return CheckResult(
                check_type=ValidationCheckType.GEOJSON_PARSE,
                passed=result.success and not (
                    result.polygon and result.polygon.has_self_intersection
                ),
                score=score,
                findings=findings,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                data={
                    "parsed_result": result,
                    "geometry_type": result.geometry_type.value if result.geometry_type else None
                }
            )

        except Exception as e:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.GEOJSON_PARSE,
                severity=ValidationSeverity.ERROR,
                code="PARSE_EXCEPTION",
                message=f"Exception during parsing: {str(e)}"
            ))

            return CheckResult(
                check_type=ValidationCheckType.GEOJSON_PARSE,
                passed=False,
                score=Decimal('0'),
                findings=findings,
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )

    def _run_coordinate_validation(self, coordinate: Coordinate) -> CheckResult:
        """Run coordinate validation check."""
        start_time = time.perf_counter()
        findings = []

        result = self.coordinate_validator.validate_coordinate(
            longitude=coordinate.longitude,
            latitude=coordinate.latitude,
            altitude=coordinate.altitude,
            required_precision=self.config.required_precision
        )

        if not result.is_valid:
            for error in result.errors:
                findings.append(ValidationFinding(
                    check_type=ValidationCheckType.COORDINATE_VALIDATION,
                    severity=ValidationSeverity.ERROR,
                    code="COORDINATE_INVALID",
                    message=error
                ))

        for warning in result.warnings:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.COORDINATE_VALIDATION,
                severity=ValidationSeverity.WARNING,
                code="COORDINATE_WARNING",
                message=warning
            ))

        if not result.meets_eudr_requirement:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.COORDINATE_VALIDATION,
                severity=ValidationSeverity.WARNING,
                code="PRECISION_BELOW_EUDR",
                message="Coordinate precision does not meet EUDR requirement (6 decimals)",
                details={"precision_level": result.precision_level.value if result.precision_level else None},
                recommendation="Provide coordinates with at least 6 decimal places"
            ))

        score = Decimal('100') if result.is_valid else Decimal('0')
        if not result.meets_eudr_requirement:
            score = score * Decimal('0.8')  # 20% penalty for precision

        return CheckResult(
            check_type=ValidationCheckType.COORDINATE_VALIDATION,
            passed=result.is_valid,
            score=score,
            findings=findings,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
            data={
                "precision_level": result.precision_level.value if result.precision_level else None,
                "meets_eudr": result.meets_eudr_requirement
            }
        )

    def _run_protected_area_check(
        self,
        coordinate: Coordinate,
        country_filter: Optional[str]
    ) -> CheckResult:
        """Run protected area intersection check."""
        start_time = time.perf_counter()
        findings = []

        result = self.protected_area_checker.check_coordinate(
            coordinate=coordinate,
            include_buffer=True,
            buffer_distance_meters=self.config.protected_area_buffer_meters,
            country_filter=country_filter
        )

        if result.status == ProtectionStatus.PROTECTED:
            for intersection in result.intersections:
                findings.append(ValidationFinding(
                    check_type=ValidationCheckType.PROTECTED_AREA,
                    severity=ValidationSeverity.ERROR,
                    code="PROTECTED_AREA_INTERSECTION",
                    message=f"Location intersects protected area: {intersection.protected_area.name}",
                    details={
                        "wdpa_id": intersection.protected_area.wdpa_id,
                        "designation": intersection.protected_area.designation,
                        "iucn_category": intersection.protected_area.iucn_category.value,
                        "overlap_percent": float(intersection.overlap_percentage)
                    },
                    recommendation="Products from protected areas are not EUDR compliant"
                ))

        elif result.status == ProtectionStatus.BUFFER_ZONE:
            for violation in result.buffer_violations:
                findings.append(ValidationFinding(
                    check_type=ValidationCheckType.PROTECTED_AREA,
                    severity=ValidationSeverity.WARNING,
                    code="BUFFER_ZONE_VIOLATION",
                    message=violation,
                    details={
                        "distance_meters": float(result.distance_to_nearest_meters) if result.distance_to_nearest_meters else None
                    },
                    recommendation="Verify production does not impact protected area"
                ))

        elif result.status == ProtectionStatus.ADJACENT:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.PROTECTED_AREA,
                severity=ValidationSeverity.INFO,
                code="ADJACENT_TO_PROTECTED",
                message=f"Location is adjacent to protected area: {result.nearest_protected_area.name if result.nearest_protected_area else 'Unknown'}",
                details={
                    "distance_meters": float(result.distance_to_nearest_meters) if result.distance_to_nearest_meters else None
                }
            ))

        # Calculate score
        if result.status == ProtectionStatus.PROTECTED:
            score = Decimal('0')
        elif result.status == ProtectionStatus.BUFFER_ZONE:
            score = Decimal('50')
        elif result.status == ProtectionStatus.ADJACENT:
            score = Decimal('80')
        else:
            score = Decimal('100')

        return CheckResult(
            check_type=ValidationCheckType.PROTECTED_AREA,
            passed=result.is_eudr_compliant,
            score=score,
            findings=findings,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
            data={
                "protection_status": result.status.value,
                "protection_level": result.protection_level.value,
                "intersections_count": len(result.intersections)
            }
        )

    def _run_deforestation_check(
        self,
        coordinate: Coordinate,
        country_iso3: str,
        observation_date: date
    ) -> CheckResult:
        """Run deforestation baseline check."""
        start_time = time.perf_counter()
        findings = []

        result = self.deforestation_checker.check_baseline(
            coordinate=coordinate,
            country_iso3=country_iso3,
            observation_date=observation_date
        )

        if result.forest_status == ForestStatus.DEFORESTED_POST_CUTOFF:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.DEFORESTATION_BASELINE,
                severity=ValidationSeverity.ERROR,
                code="POST_CUTOFF_DEFORESTATION",
                message=f"Deforestation detected after EUDR cutoff date ({EUDR_CUTOFF_DATE})",
                details={
                    "baseline_date": str(EUDR_CUTOFF_DATE),
                    "forest_cover_change_percent": float(result.forest_cover_change_percent) if result.forest_cover_change_percent else None,
                    "deforestation_events_count": len(result.deforestation_events)
                },
                recommendation="Products from deforested land after Dec 31, 2020 are not EUDR compliant"
            ))

        elif result.forest_status == ForestStatus.DEGRADED:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.DEFORESTATION_BASELINE,
                severity=ValidationSeverity.WARNING,
                code="FOREST_DEGRADATION",
                message="Forest degradation detected - verify no conversion",
                details={
                    "forest_cover_change_percent": float(result.forest_cover_change_percent) if result.forest_cover_change_percent else None
                }
            ))

        # Risk level findings
        if result.risk_level in [DeforestationRisk.HIGH, DeforestationRisk.CRITICAL]:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.DEFORESTATION_BASELINE,
                severity=ValidationSeverity.WARNING,
                code="HIGH_DEFORESTATION_RISK",
                message=f"High deforestation risk area (score: {result.risk_score})",
                details={
                    "risk_score": float(result.risk_score),
                    "risk_level": result.risk_level.value
                },
                recommendation="Enhanced due diligence recommended for high-risk areas"
            ))

        # Add warnings
        for warning in result.warnings:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.DEFORESTATION_BASELINE,
                severity=ValidationSeverity.WARNING,
                code="BASELINE_CHECK_WARNING",
                message=warning
            ))

        # Calculate score
        if not result.is_eudr_compliant:
            score = Decimal('0')
        else:
            score = Decimal('100') - result.risk_score

        return CheckResult(
            check_type=ValidationCheckType.DEFORESTATION_BASELINE,
            passed=result.is_eudr_compliant,
            score=max(score, Decimal('0')),
            findings=findings,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
            data={
                "forest_status": result.forest_status.value,
                "risk_level": result.risk_level.value,
                "risk_score": float(result.risk_score),
                "baseline_was_forest": result.baseline_was_forest,
                "current_is_forest": result.current_is_forest,
                "data_sources": result.data_sources
            }
        )

    def _run_country_risk_check(self, country_iso3: str) -> CheckResult:
        """Run country risk assessment check."""
        start_time = time.perf_counter()
        findings = []

        result = self.country_risk_db.get_country_risk(country_iso3)

        if not result.found:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.COUNTRY_RISK,
                severity=ValidationSeverity.WARNING,
                code="COUNTRY_NOT_FOUND",
                message=f"Country {country_iso3} not found in risk database",
                recommendation="Verify country code or update risk database"
            ))
            return CheckResult(
                check_type=ValidationCheckType.COUNTRY_RISK,
                passed=True,  # Not blocking
                score=Decimal('50'),  # Unknown risk
                findings=findings,
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )

        profile = result.risk_profile

        if profile.eudr_risk_category == EUDRRiskCategory.HIGH:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.COUNTRY_RISK,
                severity=ValidationSeverity.WARNING,
                code="HIGH_RISK_COUNTRY",
                message=f"{profile.country.name} is classified as HIGH RISK for EUDR",
                details={
                    "risk_category": profile.eudr_risk_category.value,
                    "risk_score": float(profile.eudr_risk_score),
                    "primary_drivers": [d.value for d in profile.primary_drivers]
                },
                recommendation="Enhanced due diligence required for high-risk countries"
            ))

        elif profile.eudr_risk_category == EUDRRiskCategory.STANDARD:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.COUNTRY_RISK,
                severity=ValidationSeverity.INFO,
                code="STANDARD_RISK_COUNTRY",
                message=f"{profile.country.name} is classified as STANDARD RISK for EUDR",
                details={
                    "risk_category": profile.eudr_risk_category.value,
                    "risk_score": float(profile.eudr_risk_score)
                }
            ))

        else:  # LOW
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.COUNTRY_RISK,
                severity=ValidationSeverity.INFO,
                code="LOW_RISK_COUNTRY",
                message=f"{profile.country.name} is classified as LOW RISK for EUDR",
                details={
                    "risk_category": profile.eudr_risk_category.value,
                    "risk_score": float(profile.eudr_risk_score)
                }
            ))

        # High-risk regions
        if profile.high_risk_regions:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.COUNTRY_RISK,
                severity=ValidationSeverity.INFO,
                code="HIGH_RISK_REGIONS",
                message=f"High-risk regions in {profile.country.name}: {', '.join(profile.high_risk_regions)}",
                details={"regions": profile.high_risk_regions}
            ))

        # Calculate score (inverse of risk)
        score = Decimal('100') - profile.eudr_risk_score

        return CheckResult(
            check_type=ValidationCheckType.COUNTRY_RISK,
            passed=True,  # Country risk is informational
            score=max(score, Decimal('0')),
            findings=findings,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
            data={
                "country_name": profile.country.name,
                "risk_category": profile.eudr_risk_category.value,
                "risk_score": float(profile.eudr_risk_score),
                "governance_score": float(profile.governance_scores.composite_score) if profile.governance_scores and profile.governance_scores.composite_score else None
            }
        )

    def _run_commodity_risk_check(
        self,
        country_iso3: str,
        commodity: EUDRCommodity
    ) -> CheckResult:
        """Run commodity-specific risk check."""
        start_time = time.perf_counter()
        findings = []

        commodity_risk = self.country_risk_db.get_commodity_risk(country_iso3, commodity)

        if commodity_risk is None:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.COMMODITY_RISK,
                severity=ValidationSeverity.INFO,
                code="COMMODITY_NOT_FOUND",
                message=f"No specific risk data for {commodity.value} in {country_iso3}",
            ))
            return CheckResult(
                check_type=ValidationCheckType.COMMODITY_RISK,
                passed=True,
                score=Decimal('50'),
                findings=findings,
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )

        # Deforestation risk
        if commodity_risk.deforestation_risk_score > self.config.max_acceptable_deforestation_risk:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.COMMODITY_RISK,
                severity=ValidationSeverity.WARNING,
                code="HIGH_COMMODITY_DEFORESTATION_RISK",
                message=f"{commodity.value} from {country_iso3} has high deforestation risk ({commodity_risk.deforestation_risk_score}%)",
                details={
                    "deforestation_risk_score": float(commodity_risk.deforestation_risk_score),
                    "primary_regions": commodity_risk.primary_regions
                },
                recommendation="Verify specific plot location and obtain certification"
            ))

        # Traceability score
        if commodity_risk.traceability_score < Decimal('50'):
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.COMMODITY_RISK,
                severity=ValidationSeverity.WARNING,
                code="LOW_TRACEABILITY",
                message=f"Low traceability capability ({commodity_risk.traceability_score}%) for {commodity.value}",
                details={"traceability_score": float(commodity_risk.traceability_score)},
                recommendation="Ensure robust traceability documentation"
            ))

        # Certification coverage
        if commodity_risk.certification_coverage_percent is not None:
            if commodity_risk.certification_coverage_percent > Decimal('30'):
                findings.append(ValidationFinding(
                    check_type=ValidationCheckType.COMMODITY_RISK,
                    severity=ValidationSeverity.INFO,
                    code="CERTIFICATION_AVAILABLE",
                    message=f"{commodity_risk.certification_coverage_percent}% of {commodity.value} production is certified",
                    details={"certification_coverage": float(commodity_risk.certification_coverage_percent)},
                    recommendation="Consider sourcing from certified producers"
                ))

        # Notes
        if commodity_risk.notes:
            findings.append(ValidationFinding(
                check_type=ValidationCheckType.COMMODITY_RISK,
                severity=ValidationSeverity.INFO,
                code="COMMODITY_NOTES",
                message=commodity_risk.notes
            ))

        # Calculate score
        score = Decimal('100') - commodity_risk.deforestation_risk_score

        return CheckResult(
            check_type=ValidationCheckType.COMMODITY_RISK,
            passed=True,
            score=max(score, Decimal('0')),
            findings=findings,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
            data={
                "commodity": commodity.value,
                "deforestation_risk": float(commodity_risk.deforestation_risk_score),
                "traceability_score": float(commodity_risk.traceability_score),
                "certification_coverage": float(commodity_risk.certification_coverage_percent) if commodity_risk.certification_coverage_percent else None
            }
        )

    def _calculate_compliance_score(
        self,
        check_results: Dict[ValidationCheckType, CheckResult]
    ) -> Decimal:
        """
        Calculate overall compliance score from individual check scores.

        Weighted average with critical checks having higher weight.
        """
        if not check_results:
            return Decimal('0')

        # Weight configuration
        weights = {
            ValidationCheckType.GEOJSON_PARSE: Decimal('15'),
            ValidationCheckType.COORDINATE_VALIDATION: Decimal('10'),
            ValidationCheckType.PROTECTED_AREA: Decimal('30'),
            ValidationCheckType.DEFORESTATION_BASELINE: Decimal('30'),
            ValidationCheckType.COUNTRY_RISK: Decimal('10'),
            ValidationCheckType.COMMODITY_RISK: Decimal('5'),
        }

        total_weight = Decimal('0')
        weighted_score = Decimal('0')

        for check_type, result in check_results.items():
            weight = weights.get(check_type, Decimal('10'))
            total_weight += weight
            weighted_score += result.score * weight

        if total_weight == 0:
            return Decimal('0')

        return (weighted_score / total_weight).quantize(
            Decimal('0.1'), rounding=ROUND_HALF_UP
        )

    def _determine_compliance_status(
        self,
        is_compliant: bool,
        error_count: int,
        warning_count: int
    ) -> ComplianceStatus:
        """Determine overall compliance status."""
        if not is_compliant or error_count > 0:
            return ComplianceStatus.NON_COMPLIANT
        elif warning_count > 0:
            return ComplianceStatus.NEEDS_REVIEW
        else:
            return ComplianceStatus.COMPLIANT

    def _finalize_report(
        self,
        report: GeoLocationValidationReport,
        start_time: float
    ) -> None:
        """Finalize report with timing and provenance."""
        report.total_execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Calculate provenance hash
        report.provenance_hash = self._calculate_hash({
            "report_id": report.report_id,
            "input_hash": report.input_hash,
            "compliance_status": report.compliance_status.value,
            "compliance_score": float(report.compliance_score),
            "is_eudr_compliant": report.is_eudr_compliant,
            "error_count": report.error_count,
            "warning_count": report.warning_count,
            "check_types": list(report.check_results.keys())
        })

    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        import uuid
        return f"EUDR-GEO-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8].upper()}"

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        def default_serializer(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, Decimal):
                return str(obj)
            elif isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, '__dict__'):
                return str(obj)
            return str(obj)

        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=default_serializer).encode()
        ).hexdigest()
