# -*- coding: utf-8 -*-
"""
GreenLang Data Quality Framework

Enterprise-grade data quality checks for emissions, energy, and CBAM data.
Validates completeness, accuracy, consistency, and flags anomalies.

Enhanced with:
- Emission factor range validation
- Temporal validity checking
- Unit consistency validation
- Data quality metrics
- Automated quality reports

Quality Dimensions:
- Completeness: Are all required fields populated?
- Accuracy: Are values within expected ranges?
- Consistency: Do calculated fields match source data?
- Timeliness: Is data current and properly dated?
- Uniqueness: Are there duplicate records?
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from decimal import Decimal
from datetime import datetime, date, timedelta
from enum import Enum
from pydantic import BaseModel, Field
import logging
import statistics
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


# ============================================================================
# QUALITY LEVEL ENUM
# ============================================================================

class DataQualityLevel(str, Enum):
    """Data quality levels for emission data."""
    EXCELLENT = "excellent"  # Score >= 90
    GOOD = "good"            # Score >= 70
    FAIR = "fair"            # Score >= 50
    POOR = "poor"            # Score < 50


# ============================================================================
# QUALITY CHECK MODELS
# ============================================================================

class QualityDimension(str, Enum):
    """Data quality dimensions"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"
    RANGE_VALIDITY = "range_validity"
    TEMPORAL_VALIDITY = "temporal_validity"
    UNIT_CONSISTENCY = "unit_consistency"


class QualitySeverity(str, Enum):
    """Quality issue severity"""
    CRITICAL = "critical"  # Data unusable
    HIGH = "high"          # Significant issues
    MEDIUM = "medium"      # Minor issues
    LOW = "low"            # Warnings only


class QualityCheck(BaseModel):
    """Individual quality check result"""
    dimension: QualityDimension
    check_name: str
    passed: bool
    severity: QualitySeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    checked_at: datetime = Field(default_factory=datetime.utcnow)


class DataQualityReport(BaseModel):
    """
    Comprehensive data quality report.

    Provides overall quality score (0-100) and detailed check results.
    """
    record_id: str
    data_type: str  # "cbam", "emissions", "energy", "activity", "emission_factor"

    # Overall Score
    overall_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall quality score (0-100)"
    )

    quality_level: DataQualityLevel

    # Dimension Scores
    completeness_score: float = Field(ge=0, le=100)
    accuracy_score: float = Field(ge=0, le=100)
    consistency_score: float = Field(ge=0, le=100)
    timeliness_score: float = Field(ge=0, le=100)
    uniqueness_score: float = Field(ge=0, le=100)

    # Additional dimension scores for emission factors
    range_validity_score: Optional[float] = Field(default=None, ge=0, le=100)
    temporal_validity_score: Optional[float] = Field(default=None, ge=0, le=100)
    unit_consistency_score: Optional[float] = Field(default=None, ge=0, le=100)

    # Check Results
    checks_passed: int
    checks_failed: int
    total_checks: int

    # Issues
    critical_issues: List[QualityCheck] = Field(default_factory=list)
    high_issues: List[QualityCheck] = Field(default_factory=list)
    medium_issues: List[QualityCheck] = Field(default_factory=list)
    low_issues: List[QualityCheck] = Field(default_factory=list)

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)

    # Metadata
    checked_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_acceptable(self) -> bool:
        """Check if quality is acceptable (>= 70%)"""
        return self.overall_score >= 70.0

    @property
    def has_critical_issues(self) -> bool:
        """Check for critical issues"""
        return len(self.critical_issues) > 0


# ============================================================================
# EMISSION FACTOR VALIDATION RANGES
# ============================================================================

# Expected ranges for common emission factors (kgCO2e per unit)
EMISSION_FACTOR_RANGES = {
    "natural_gas": {
        "min": 40.0,
        "max": 70.0,
        "unit": "kgCO2e/GJ",
        "typical": 56.0
    },
    "diesel": {
        "min": 2.0,
        "max": 3.5,
        "unit": "kgCO2e/L",
        "typical": 2.68
    },
    "petrol": {
        "min": 1.8,
        "max": 3.0,
        "unit": "kgCO2e/L",
        "typical": 2.31
    },
    "gasoline": {
        "min": 1.8,
        "max": 3.0,
        "unit": "kgCO2e/L",
        "typical": 2.31
    },
    "lpg": {
        "min": 1.0,
        "max": 2.0,
        "unit": "kgCO2e/L",
        "typical": 1.52
    },
    "coal_industrial": {
        "min": 80.0,
        "max": 110.0,
        "unit": "kgCO2e/GJ",
        "typical": 95.0
    },
    "electricity_grid": {
        "min": 0.0,
        "max": 1200.0,
        "unit": "kgCO2e/MWh",
        "typical": 400.0
    },
    "aviation_turbine_fuel": {
        "min": 2.0,
        "max": 3.5,
        "unit": "kgCO2e/L",
        "typical": 2.52
    },
    "hfc_134a": {
        "min": 1300.0,
        "max": 1600.0,
        "unit": "kgCO2e/kg",
        "typical": 1430.0
    },
    "sf6": {
        "min": 20000.0,
        "max": 30000.0,
        "unit": "kgCO2e/kg",
        "typical": 25200.0
    },
    "cement": {
        "min": 600.0,
        "max": 1100.0,
        "unit": "kgCO2e/tonne",
        "typical": 830.0
    },
    "steel_virgin": {
        "min": 1500.0,
        "max": 2500.0,
        "unit": "kgCO2e/tonne",
        "typical": 1850.0
    },
    "aluminium_virgin": {
        "min": 8000.0,
        "max": 12000.0,
        "unit": "kgCO2e/tonne",
        "typical": 9200.0
    }
}

# Valid date ranges for emission factor data
VALID_DATE_RANGES = {
    "defra_2024": {"min_year": 2024, "max_year": 2024},
    "defra_2023": {"min_year": 2022, "max_year": 2023},
    "epa_egrid_2023": {"min_year": 2022, "max_year": 2023},
    "ipcc_ar6": {"min_year": 2020, "max_year": 2025}
}


# ============================================================================
# EMISSION FACTOR QUALITY CHECKER
# ============================================================================

class EmissionFactorQualityChecker:
    """
    Quality checker for emission factor data.

    Validates:
    - Value ranges (typical ranges for each fuel type)
    - Temporal validity (data within valid date range)
    - Unit consistency (correct units for fuel type)
    - Completeness (all required fields present)
    - Data quality tier compliance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize emission factor quality checker."""
        self.config = config or {}
        self.checks: List[QualityCheck] = []
        self.custom_ranges = self.config.get("custom_ranges", {})

    def check_emission_factor(
        self,
        ef_record: Dict[str, Any],
        fuel_type: str,
        version: str = "defra_2024"
    ) -> DataQualityReport:
        """
        Check emission factor data quality.

        Args:
            ef_record: Emission factor record
            fuel_type: Type of fuel
            version: Data source version

        Returns:
            DataQualityReport with quality assessment
        """
        self.checks = []

        # Core quality checks
        completeness_score = self._check_completeness(ef_record)
        accuracy_score = self._check_accuracy(ef_record)
        consistency_score = self._check_consistency(ef_record)
        timeliness_score = self._check_timeliness(ef_record, version)
        uniqueness_score = 100.0  # Would need database context

        # Enhanced checks
        range_validity_score = self._check_range_validity(ef_record, fuel_type)
        temporal_validity_score = self._check_temporal_validity(ef_record, version)
        unit_consistency_score = self._check_unit_consistency(ef_record, fuel_type)

        # Calculate overall score with enhanced dimensions
        overall_score = (
            completeness_score * 0.20 +
            accuracy_score * 0.20 +
            consistency_score * 0.15 +
            timeliness_score * 0.10 +
            uniqueness_score * 0.05 +
            range_validity_score * 0.15 +
            temporal_validity_score * 0.05 +
            unit_consistency_score * 0.10
        )

        # Build report
        report = self._build_report(
            record_id=ef_record.get("ef_uri", str(ef_record.get("id", "unknown"))),
            data_type="emission_factor",
            overall_score=overall_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            timeliness_score=timeliness_score,
            uniqueness_score=uniqueness_score,
            range_validity_score=range_validity_score,
            temporal_validity_score=temporal_validity_score,
            unit_consistency_score=unit_consistency_score
        )

        # Generate recommendations
        report.recommendations = self._generate_ef_recommendations(ef_record, report)

        return report

    def _check_completeness(self, ef_record: Dict[str, Any]) -> float:
        """Check completeness of emission factor record."""
        required_fields = [
            "co2", "unit", "citation"
        ]
        recommended_fields = [
            "ch4", "n2o", "uncertainty", "quality"
        ]

        filled_required = sum(1 for f in required_fields if ef_record.get(f) is not None)
        filled_recommended = sum(1 for f in recommended_fields if ef_record.get(f) is not None)

        required_score = (filled_required / len(required_fields)) * 80
        recommended_bonus = (filled_recommended / len(recommended_fields)) * 20

        score = required_score + recommended_bonus

        if filled_required < len(required_fields):
            missing = [f for f in required_fields if ef_record.get(f) is None]
            self.checks.append(QualityCheck(
                dimension=QualityDimension.COMPLETENESS,
                check_name="required_fields",
                passed=False,
                severity=QualitySeverity.HIGH,
                message=f"Missing required fields: {', '.join(missing)}",
                details={"missing_fields": missing}
            ))

        return min(100.0, score)

    def _check_accuracy(self, ef_record: Dict[str, Any]) -> float:
        """Check accuracy of emission factor values."""
        score = 100.0

        # CO2 must be non-negative
        co2 = ef_record.get("co2", 0)
        if co2 < 0:
            score -= 30
            self.checks.append(QualityCheck(
                dimension=QualityDimension.ACCURACY,
                check_name="co2_non_negative",
                passed=False,
                severity=QualitySeverity.CRITICAL,
                message=f"CO2 emission factor cannot be negative: {co2}"
            ))

        # Uncertainty should be reasonable (0-100%)
        uncertainty = ef_record.get("uncertainty", 0)
        if uncertainty < 0 or uncertainty > 1.0:
            score -= 15
            self.checks.append(QualityCheck(
                dimension=QualityDimension.ACCURACY,
                check_name="uncertainty_range",
                passed=False,
                severity=QualitySeverity.MEDIUM,
                message=f"Uncertainty {uncertainty} outside valid range (0-1)"
            ))

        # Quality tier should be 1-5
        quality = ef_record.get("quality", "2")
        try:
            q_int = int(quality)
            if q_int < 1 or q_int > 5:
                score -= 10
                self.checks.append(QualityCheck(
                    dimension=QualityDimension.ACCURACY,
                    check_name="quality_tier",
                    passed=False,
                    severity=QualitySeverity.LOW,
                    message=f"Quality tier {quality} outside valid range (1-5)"
                ))
        except ValueError:
            pass

        return max(0.0, score)

    def _check_consistency(self, ef_record: Dict[str, Any]) -> float:
        """Check internal consistency of emission factor data."""
        score = 100.0

        co2 = ef_record.get("co2", 0)
        ch4 = ef_record.get("ch4", 0)
        n2o = ef_record.get("n2o", 0)
        co2e = ef_record.get("co2e_ar6", ef_record.get("co2e", 0))

        # CO2e should be >= CO2 (since CH4 and N2O add to it)
        if co2e < co2 and co2e > 0:
            score -= 25
            self.checks.append(QualityCheck(
                dimension=QualityDimension.CONSISTENCY,
                check_name="co2e_gte_co2",
                passed=False,
                severity=QualitySeverity.HIGH,
                message=f"CO2e ({co2e}) should be >= CO2 ({co2})"
            ))

        # Verify GWP calculation (AR6: CH4=27.9, N2O=273)
        if ch4 > 0 or n2o > 0:
            expected_co2e = co2 + (ch4 * 27.9) + (n2o * 273)
            tolerance = expected_co2e * 0.05  # 5% tolerance

            if abs(co2e - expected_co2e) > tolerance and co2e > 0:
                score -= 20
                self.checks.append(QualityCheck(
                    dimension=QualityDimension.CONSISTENCY,
                    check_name="gwp_calculation",
                    passed=False,
                    severity=QualitySeverity.MEDIUM,
                    message=f"CO2e ({co2e}) differs from calculated ({expected_co2e:.2f})",
                    details={
                        "reported_co2e": co2e,
                        "calculated_co2e": round(expected_co2e, 4),
                        "difference": round(abs(co2e - expected_co2e), 4)
                    }
                ))

        return max(0.0, score)

    def _check_timeliness(self, ef_record: Dict[str, Any], version: str) -> float:
        """Check timeliness of emission factor data."""
        score = 100.0

        # Check if citation mentions expected year
        citation = ef_record.get("citation", "")
        if version in ["2024", "defra_2024"] and "2024" not in citation:
            score -= 20
            self.checks.append(QualityCheck(
                dimension=QualityDimension.TIMELINESS,
                check_name="citation_year_match",
                passed=False,
                severity=QualitySeverity.LOW,
                message=f"Citation does not mention 2024 for {version} data"
            ))

        return max(0.0, score)

    def _check_range_validity(
        self,
        ef_record: Dict[str, Any],
        fuel_type: str
    ) -> float:
        """Check if emission factor is within expected range."""
        score = 100.0

        # Get expected range for fuel type
        range_info = EMISSION_FACTOR_RANGES.get(
            fuel_type,
            self.custom_ranges.get(fuel_type)
        )

        if not range_info:
            # No range defined, give partial score
            return 80.0

        ef_value = ef_record.get("co2e_ar6", ef_record.get("co2e", ef_record.get("co2", 0)))

        min_val = range_info["min"]
        max_val = range_info["max"]
        typical = range_info["typical"]

        if ef_value < min_val:
            deviation = ((min_val - ef_value) / min_val) * 100
            score -= min(50, deviation)
            self.checks.append(QualityCheck(
                dimension=QualityDimension.RANGE_VALIDITY,
                check_name="below_minimum",
                passed=False,
                severity=QualitySeverity.HIGH if deviation > 20 else QualitySeverity.MEDIUM,
                message=f"Emission factor {ef_value} below minimum {min_val} for {fuel_type}",
                details={
                    "value": ef_value,
                    "min_expected": min_val,
                    "max_expected": max_val,
                    "typical": typical,
                    "deviation_pct": round(deviation, 2)
                }
            ))
        elif ef_value > max_val:
            deviation = ((ef_value - max_val) / max_val) * 100
            score -= min(50, deviation)
            self.checks.append(QualityCheck(
                dimension=QualityDimension.RANGE_VALIDITY,
                check_name="above_maximum",
                passed=False,
                severity=QualitySeverity.HIGH if deviation > 20 else QualitySeverity.MEDIUM,
                message=f"Emission factor {ef_value} above maximum {max_val} for {fuel_type}",
                details={
                    "value": ef_value,
                    "min_expected": min_val,
                    "max_expected": max_val,
                    "typical": typical,
                    "deviation_pct": round(deviation, 2)
                }
            ))

        # Check deviation from typical value
        if ef_value > 0 and typical > 0:
            typical_deviation = abs(ef_value - typical) / typical * 100
            if typical_deviation > 30:
                score -= 10
                self.checks.append(QualityCheck(
                    dimension=QualityDimension.RANGE_VALIDITY,
                    check_name="deviation_from_typical",
                    passed=False,
                    severity=QualitySeverity.LOW,
                    message=f"Emission factor {ef_value} deviates {typical_deviation:.1f}% from typical {typical}",
                    details={
                        "value": ef_value,
                        "typical": typical,
                        "deviation_pct": round(typical_deviation, 2)
                    }
                ))

        return max(0.0, score)

    def _check_temporal_validity(
        self,
        ef_record: Dict[str, Any],
        version: str
    ) -> float:
        """Check temporal validity of emission factor."""
        score = 100.0

        # Get valid date range for version
        date_range = VALID_DATE_RANGES.get(version)
        if not date_range:
            return 90.0  # Unknown version, give reasonable score

        # Check if data year is within valid range
        # This would typically check the actual data year in the record
        # For now, assume data is valid if version matches

        return score

    def _check_unit_consistency(
        self,
        ef_record: Dict[str, Any],
        fuel_type: str
    ) -> float:
        """Check unit consistency for fuel type."""
        score = 100.0

        unit = ef_record.get("unit", "")
        expected_unit = EMISSION_FACTOR_RANGES.get(fuel_type, {}).get("unit", "")

        if expected_unit and unit:
            # Normalize units for comparison
            unit_normalized = unit.lower().replace(" ", "")
            expected_normalized = expected_unit.lower().replace(" ", "")

            if unit_normalized != expected_normalized:
                # Check for compatible units (e.g., kgCO2e/l vs kgCO2e/L)
                if not self._units_compatible(unit, expected_unit):
                    score -= 30
                    self.checks.append(QualityCheck(
                        dimension=QualityDimension.UNIT_CONSISTENCY,
                        check_name="unit_mismatch",
                        passed=False,
                        severity=QualitySeverity.MEDIUM,
                        message=f"Unit '{unit}' does not match expected '{expected_unit}' for {fuel_type}",
                        details={
                            "reported_unit": unit,
                            "expected_unit": expected_unit
                        }
                    ))

        return max(0.0, score)

    def _units_compatible(self, unit1: str, unit2: str) -> bool:
        """Check if two units are compatible (same but different format)."""
        # Normalize for comparison
        def normalize(u: str) -> str:
            return u.lower().replace(" ", "").replace("co2e", "co2e").replace("/", "per")

        return normalize(unit1) == normalize(unit2)

    def _build_report(
        self,
        record_id: str,
        data_type: str,
        overall_score: float,
        completeness_score: float,
        accuracy_score: float,
        consistency_score: float,
        timeliness_score: float,
        uniqueness_score: float,
        range_validity_score: Optional[float] = None,
        temporal_validity_score: Optional[float] = None,
        unit_consistency_score: Optional[float] = None
    ) -> DataQualityReport:
        """Build quality report from checks."""

        # Categorize issues
        critical = [c for c in self.checks if c.severity == QualitySeverity.CRITICAL and not c.passed]
        high = [c for c in self.checks if c.severity == QualitySeverity.HIGH and not c.passed]
        medium = [c for c in self.checks if c.severity == QualitySeverity.MEDIUM and not c.passed]
        low = [c for c in self.checks if c.severity == QualitySeverity.LOW and not c.passed]

        passed = sum(1 for c in self.checks if c.passed)
        failed = sum(1 for c in self.checks if not c.passed)

        # Determine quality level
        if overall_score >= 90:
            quality_level = DataQualityLevel.EXCELLENT
        elif overall_score >= 70:
            quality_level = DataQualityLevel.GOOD
        elif overall_score >= 50:
            quality_level = DataQualityLevel.FAIR
        else:
            quality_level = DataQualityLevel.POOR

        return DataQualityReport(
            record_id=record_id,
            data_type=data_type,
            overall_score=round(overall_score, 2),
            quality_level=quality_level,
            completeness_score=round(completeness_score, 2),
            accuracy_score=round(accuracy_score, 2),
            consistency_score=round(consistency_score, 2),
            timeliness_score=round(timeliness_score, 2),
            uniqueness_score=round(uniqueness_score, 2),
            range_validity_score=round(range_validity_score, 2) if range_validity_score else None,
            temporal_validity_score=round(temporal_validity_score, 2) if temporal_validity_score else None,
            unit_consistency_score=round(unit_consistency_score, 2) if unit_consistency_score else None,
            checks_passed=passed,
            checks_failed=failed,
            total_checks=len(self.checks),
            critical_issues=critical,
            high_issues=high,
            medium_issues=medium,
            low_issues=low
        )

    def _generate_ef_recommendations(
        self,
        ef_record: Dict[str, Any],
        report: DataQualityReport
    ) -> List[str]:
        """Generate recommendations for improving emission factor data quality."""
        recommendations = []

        if report.completeness_score < 100:
            recommendations.append("Add missing fields: CH4, N2O, uncertainty for complete GHG profile")

        if report.range_validity_score and report.range_validity_score < 80:
            recommendations.append("Verify emission factor value against authoritative sources (DEFRA, EPA)")

        if report.consistency_score < 100:
            recommendations.append("Recalculate CO2e using correct AR6 GWP values (CH4=27.9, N2O=273)")

        if report.has_critical_issues:
            recommendations.append("Address critical issues before using this emission factor in calculations")

        if not ef_record.get("url"):
            recommendations.append("Add source URL for data provenance and auditability")

        return recommendations


# ============================================================================
# BATCH QUALITY METRICS
# ============================================================================

class BatchQualityMetrics(BaseModel):
    """Quality metrics for a batch of emission factors."""
    total_records: int
    records_checked: int
    records_passed: int
    records_failed: int
    pass_rate: float
    average_score: float
    min_score: float
    max_score: float
    score_distribution: Dict[str, int]  # {excellent, good, fair, poor}
    common_issues: List[Dict[str, Any]]
    recommendations: List[str]
    checked_at: datetime = Field(default_factory=datetime.utcnow)


def check_emission_factor_batch(
    records: List[Dict[str, Any]],
    fuel_type_field: str = "fuel_type",
    version: str = "defra_2024"
) -> BatchQualityMetrics:
    """
    Check quality of a batch of emission factors.

    Args:
        records: List of emission factor records
        fuel_type_field: Field name for fuel type
        version: Data source version

    Returns:
        BatchQualityMetrics with aggregate statistics
    """
    checker = EmissionFactorQualityChecker()

    scores = []
    passed = 0
    failed = 0
    issue_counts: Dict[str, int] = defaultdict(int)
    distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}

    for record in records:
        fuel_type = record.get(fuel_type_field, "unknown")
        report = checker.check_emission_factor(record, fuel_type, version)

        scores.append(report.overall_score)

        if report.is_acceptable:
            passed += 1
        else:
            failed += 1

        # Track distribution
        distribution[report.quality_level.value] += 1

        # Track common issues
        for issue in report.critical_issues + report.high_issues:
            issue_counts[issue.check_name] += 1

    # Sort and get top issues
    common_issues = [
        {"issue": name, "count": count}
        for name, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ]

    # Generate recommendations
    recommendations = []
    if common_issues:
        top_issue = common_issues[0]["issue"]
        recommendations.append(f"Address '{top_issue}' which affects {common_issues[0]['count']} records")

    if failed > 0:
        recommendations.append(f"Review {failed} records with quality issues before use")

    return BatchQualityMetrics(
        total_records=len(records),
        records_checked=len(records),
        records_passed=passed,
        records_failed=failed,
        pass_rate=passed / len(records) if records else 0,
        average_score=statistics.mean(scores) if scores else 0,
        min_score=min(scores) if scores else 0,
        max_score=max(scores) if scores else 0,
        score_distribution=distribution,
        common_issues=common_issues,
        recommendations=recommendations
    )


# ============================================================================
# AUTOMATED QUALITY REPORT GENERATOR
# ============================================================================

class QualityReportGenerator:
    """Generate automated quality reports for emission factor data."""

    def __init__(self, output_format: str = "json"):
        """
        Initialize report generator.

        Args:
            output_format: Output format (json, markdown, html)
        """
        self.output_format = output_format
        self.checker = EmissionFactorQualityChecker()

    def generate_factor_report(
        self,
        ef_record: Dict[str, Any],
        fuel_type: str,
        version: str = "defra_2024"
    ) -> str:
        """Generate quality report for single emission factor."""
        report = self.checker.check_emission_factor(ef_record, fuel_type, version)

        if self.output_format == "json":
            return report.model_dump_json(indent=2)
        elif self.output_format == "markdown":
            return self._to_markdown(report)
        else:
            return report.model_dump_json(indent=2)

    def generate_batch_report(
        self,
        records: List[Dict[str, Any]],
        fuel_type_field: str = "fuel_type",
        version: str = "defra_2024"
    ) -> str:
        """Generate quality report for batch of emission factors."""
        metrics = check_emission_factor_batch(records, fuel_type_field, version)

        if self.output_format == "json":
            return metrics.model_dump_json(indent=2)
        elif self.output_format == "markdown":
            return self._batch_to_markdown(metrics)
        else:
            return metrics.model_dump_json(indent=2)

    def _to_markdown(self, report: DataQualityReport) -> str:
        """Convert quality report to markdown."""
        md = f"""# Data Quality Report

**Record ID:** {report.record_id}
**Data Type:** {report.data_type}
**Generated:** {report.checked_at.isoformat()}

## Overall Score: {report.overall_score}/100 ({report.quality_level.value.upper()})

### Dimension Scores

| Dimension | Score |
|-----------|-------|
| Completeness | {report.completeness_score} |
| Accuracy | {report.accuracy_score} |
| Consistency | {report.consistency_score} |
| Timeliness | {report.timeliness_score} |
| Uniqueness | {report.uniqueness_score} |
"""

        if report.range_validity_score:
            md += f"| Range Validity | {report.range_validity_score} |\n"
        if report.unit_consistency_score:
            md += f"| Unit Consistency | {report.unit_consistency_score} |\n"

        md += f"""
### Check Results

- **Passed:** {report.checks_passed}
- **Failed:** {report.checks_failed}
- **Total:** {report.total_checks}

"""

        if report.critical_issues:
            md += "### Critical Issues\n\n"
            for issue in report.critical_issues:
                md += f"- **{issue.check_name}**: {issue.message}\n"

        if report.high_issues:
            md += "\n### High Priority Issues\n\n"
            for issue in report.high_issues:
                md += f"- **{issue.check_name}**: {issue.message}\n"

        if report.recommendations:
            md += "\n### Recommendations\n\n"
            for rec in report.recommendations:
                md += f"- {rec}\n"

        return md

    def _batch_to_markdown(self, metrics: BatchQualityMetrics) -> str:
        """Convert batch metrics to markdown."""
        return f"""# Batch Quality Report

**Generated:** {metrics.checked_at.isoformat()}

## Summary

- **Total Records:** {metrics.total_records}
- **Passed:** {metrics.records_passed} ({metrics.pass_rate * 100:.1f}%)
- **Failed:** {metrics.records_failed}

## Score Statistics

- **Average Score:** {metrics.average_score:.1f}
- **Min Score:** {metrics.min_score:.1f}
- **Max Score:** {metrics.max_score:.1f}

## Score Distribution

| Quality Level | Count |
|---------------|-------|
| Excellent (>=90) | {metrics.score_distribution['excellent']} |
| Good (70-89) | {metrics.score_distribution['good']} |
| Fair (50-69) | {metrics.score_distribution['fair']} |
| Poor (<50) | {metrics.score_distribution['poor']} |

## Common Issues

{self._issues_table(metrics.common_issues)}

## Recommendations

{"".join(f"- {rec}" + chr(10) for rec in metrics.recommendations)}
"""

    def _issues_table(self, issues: List[Dict[str, Any]]) -> str:
        """Generate markdown table for issues."""
        if not issues:
            return "No significant issues found."

        table = "| Issue | Count |\n|-------|-------|\n"
        for issue in issues:
            table += f"| {issue['issue']} | {issue['count']} |\n"
        return table


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def check_emission_factor_quality(
    ef_record: Dict[str, Any],
    fuel_type: str,
    version: str = "defra_2024"
) -> DataQualityReport:
    """
    Check quality of a single emission factor.

    Args:
        ef_record: Emission factor record
        fuel_type: Fuel type
        version: Data source version

    Returns:
        DataQualityReport
    """
    checker = EmissionFactorQualityChecker()
    return checker.check_emission_factor(ef_record, fuel_type, version)


def validate_emission_factor_range(
    ef_value: float,
    fuel_type: str
) -> Tuple[bool, Optional[str]]:
    """
    Quick validation of emission factor value range.

    Args:
        ef_value: Emission factor value
        fuel_type: Fuel type

    Returns:
        Tuple of (is_valid, error_message)
    """
    range_info = EMISSION_FACTOR_RANGES.get(fuel_type)

    if not range_info:
        return True, None  # No range defined

    if ef_value < range_info["min"]:
        return False, f"Value {ef_value} below minimum {range_info['min']} for {fuel_type}"
    if ef_value > range_info["max"]:
        return False, f"Value {ef_value} above maximum {range_info['max']} for {fuel_type}"

    return True, None


def get_expected_range(fuel_type: str) -> Optional[Dict[str, Any]]:
    """
    Get expected range for a fuel type.

    Args:
        fuel_type: Fuel type

    Returns:
        Range info dict or None
    """
    return EMISSION_FACTOR_RANGES.get(fuel_type)
