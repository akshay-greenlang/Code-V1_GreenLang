# -*- coding: utf-8 -*-
"""
DataQualityBridge - PAI Data Quality Enforcement for SFDR Article 8
====================================================================

This module enforces data quality thresholds for PAI calculations,
portfolio holdings, and taxonomy alignment assessments. It validates
coverage, freshness, accuracy, completeness, and consistency of data
used in SFDR Article 8 disclosures.

Architecture:
    SFDR Pipeline Data --> DataQualityBridge --> Quality Score + Flags
                                |
                                v
    Coverage Report + Estimation Flags + Data Age Validation

Example:
    >>> config = DataQualityBridgeConfig()
    >>> bridge = DataQualityBridge(config)
    >>> quality = bridge.assess_quality(pai_data, holdings)
    >>> print(f"Quality score: {quality['overall_score']:.1f}")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Helpers
# =============================================================================


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# =============================================================================
# Enums
# =============================================================================


class QualityLevel(str, Enum):
    """Data quality level."""
    HIGH = "high"
    ACCEPTABLE = "acceptable"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class CheckCategory(str, Enum):
    """Data quality check category."""
    COVERAGE = "coverage"
    FRESHNESS = "freshness"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ESTIMATION = "estimation"
    PROVENANCE = "provenance"


class DataSourceType(str, Enum):
    """Type of data source."""
    REPORTED = "reported"
    ESTIMATED = "estimated"
    PROXY = "proxy"
    MODELED = "modeled"
    UNAVAILABLE = "unavailable"


# =============================================================================
# Data Models
# =============================================================================


class DataQualityBridgeConfig(BaseModel):
    """Configuration for the Data Quality Bridge."""
    min_coverage: float = Field(
        default=70.0, ge=0.0, le=100.0,
        description="Minimum data coverage percentage",
    )
    max_estimation_pct: float = Field(
        default=30.0, ge=0.0, le=100.0,
        description="Maximum percentage of estimated data",
    )
    max_data_age_days: int = Field(
        default=365, ge=1,
        description="Maximum acceptable data age in days",
    )
    min_quality_score: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="Minimum overall quality score",
    )
    require_provenance: bool = Field(
        default=True,
        description="Require provenance tracking for all data",
    )
    warn_on_proxy_data: bool = Field(
        default=True,
        description="Warn when proxy data is used",
    )
    completeness_threshold: float = Field(
        default=80.0, ge=0.0, le=100.0,
        description="Minimum field completeness percentage",
    )


class QualityCheckResult(BaseModel):
    """Result of a single quality check."""
    category: CheckCategory = Field(
        default=CheckCategory.COVERAGE, description="Check category"
    )
    check_name: str = Field(default="", description="Check name")
    status: QualityLevel = Field(
        default=QualityLevel.HIGH, description="Check result"
    )
    value: float = Field(default=0.0, description="Measured value")
    threshold: float = Field(default=0.0, description="Threshold value")
    passed: bool = Field(default=True, description="Whether check passed")
    detail: str = Field(default="", description="Check detail")


class CoverageReport(BaseModel):
    """Data coverage report for PAI indicators."""
    total_indicators: int = Field(
        default=0, description="Total PAI indicators"
    )
    covered_indicators: int = Field(
        default=0, description="Indicators with data"
    )
    coverage_pct: float = Field(
        default=0.0, description="Coverage percentage"
    )
    by_indicator: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Coverage by indicator"
    )
    by_data_source: Dict[str, int] = Field(
        default_factory=dict, description="Count by data source type"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class QualityAssessment(BaseModel):
    """Overall data quality assessment."""
    overall_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall quality score"
    )
    overall_level: QualityLevel = Field(
        default=QualityLevel.ACCEPTABLE, description="Overall quality level"
    )
    checks: List[QualityCheckResult] = Field(
        default_factory=list, description="Individual check results"
    )
    checks_passed: int = Field(default=0, description="Checks passed")
    checks_failed: int = Field(default=0, description="Checks failed")
    coverage_report: Optional[CoverageReport] = Field(
        default=None, description="Coverage report"
    )
    estimation_flags: List[Dict[str, Any]] = Field(
        default_factory=list, description="Estimation flags"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Quality improvement recommendations"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    assessed_at: str = Field(default="", description="Assessment timestamp")


# =============================================================================
# Quality Check Definitions
# =============================================================================


QUALITY_CHECKS: Dict[str, Dict[str, Any]] = {
    "pai_data_coverage": {
        "category": "coverage",
        "description": "PAI indicator data coverage",
        "weight": 15,
    },
    "holdings_isin_coverage": {
        "category": "coverage",
        "description": "Holdings with valid ISIN",
        "weight": 10,
    },
    "esg_rating_coverage": {
        "category": "coverage",
        "description": "Holdings with ESG ratings",
        "weight": 10,
    },
    "emissions_data_freshness": {
        "category": "freshness",
        "description": "Emissions data recency",
        "weight": 10,
    },
    "esg_data_freshness": {
        "category": "freshness",
        "description": "ESG data recency",
        "weight": 5,
    },
    "taxonomy_data_freshness": {
        "category": "freshness",
        "description": "Taxonomy data recency",
        "weight": 5,
    },
    "emissions_accuracy": {
        "category": "accuracy",
        "description": "Emissions data accuracy (reported vs estimated)",
        "weight": 10,
    },
    "taxonomy_accuracy": {
        "category": "accuracy",
        "description": "Taxonomy alignment data accuracy",
        "weight": 5,
    },
    "portfolio_completeness": {
        "category": "completeness",
        "description": "Portfolio field completeness",
        "weight": 10,
    },
    "sector_completeness": {
        "category": "completeness",
        "description": "Sector classification completeness",
        "weight": 5,
    },
    "country_completeness": {
        "category": "completeness",
        "description": "Country classification completeness",
        "weight": 5,
    },
    "weight_consistency": {
        "category": "consistency",
        "description": "Portfolio weight sum consistency",
        "weight": 5,
    },
    "cross_source_consistency": {
        "category": "consistency",
        "description": "Cross-source data consistency",
        "weight": 3,
    },
    "estimation_ratio": {
        "category": "estimation",
        "description": "Ratio of estimated vs reported data",
        "weight": 7,
    },
    "provenance_coverage": {
        "category": "provenance",
        "description": "Data provenance tracking coverage",
        "weight": 5,
    },
}


# =============================================================================
# Data Quality Bridge
# =============================================================================


class DataQualityBridge:
    """Data quality enforcement for SFDR Article 8 data.

    Validates data quality across coverage, freshness, accuracy,
    completeness, consistency, and estimation metrics. Produces a
    quality score and actionable recommendations.

    Attributes:
        config: Bridge configuration.

    Example:
        >>> bridge = DataQualityBridge(DataQualityBridgeConfig())
        >>> assessment = bridge.assess_quality(pai_data, holdings)
        >>> print(f"Score: {assessment.overall_score:.1f}")
    """

    def __init__(
        self, config: Optional[DataQualityBridgeConfig] = None
    ) -> None:
        """Initialize the Data Quality Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or DataQualityBridgeConfig()
        self.logger = logger

        self.logger.info(
            "DataQualityBridge initialized: coverage=%.0f%%, estimation=%.0f%%, "
            "age=%d days",
            self.config.min_coverage,
            self.config.max_estimation_pct,
            self.config.max_data_age_days,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def assess_quality(
        self,
        pai_data: Dict[str, Any],
        holdings: List[Dict[str, Any]],
        taxonomy_data: Optional[Dict[str, Any]] = None,
    ) -> QualityAssessment:
        """Run all quality checks and produce an overall assessment.

        Args:
            pai_data: PAI indicator data.
            holdings: Portfolio holdings.
            taxonomy_data: Optional taxonomy alignment data.

        Returns:
            QualityAssessment with score, checks, and recommendations.
        """
        checks: List[QualityCheckResult] = []
        passed = 0
        failed = 0

        # Coverage checks
        checks.append(self._check_pai_coverage(pai_data))
        checks.append(self._check_holdings_isin(holdings))
        checks.append(self._check_esg_coverage(holdings))

        # Freshness checks
        checks.append(self._check_data_freshness(holdings, "emissions"))
        checks.append(self._check_data_freshness(holdings, "esg"))
        checks.append(self._check_data_freshness(holdings, "taxonomy"))

        # Accuracy checks
        checks.append(self._check_emissions_accuracy(holdings))
        checks.append(self._check_taxonomy_accuracy(
            taxonomy_data or {}, holdings
        ))

        # Completeness checks
        checks.append(self._check_portfolio_completeness(holdings))
        checks.append(self._check_field_coverage(holdings, "sector", "sector"))
        checks.append(self._check_field_coverage(
            holdings, "country_code", "country"
        ))

        # Consistency checks
        checks.append(self._check_weight_consistency(holdings))
        checks.append(self._check_cross_source(holdings))

        # Estimation check
        estimation_check = self._check_estimation_ratio(holdings)
        checks.append(estimation_check)

        # Provenance check
        checks.append(self._check_provenance(holdings))

        for c in checks:
            if c.passed:
                passed += 1
            else:
                failed += 1

        # Weighted score
        total_weight = sum(
            QUALITY_CHECKS.get(c.check_name, {}).get("weight", 5)
            for c in checks
        )
        weighted_score = sum(
            QUALITY_CHECKS.get(c.check_name, {}).get("weight", 5) * (
                1.0 if c.passed else 0.0
            )
            for c in checks
        )
        overall_score = round(
            (weighted_score / max(total_weight, 1)) * 100, 1
        )

        # Determine level
        if overall_score >= 80:
            level = QualityLevel.HIGH
        elif overall_score >= 60:
            level = QualityLevel.ACCEPTABLE
        elif overall_score >= 40:
            level = QualityLevel.LOW
        else:
            level = QualityLevel.INSUFFICIENT

        # Coverage report
        coverage_report = self.get_coverage_report(pai_data)

        # Estimation flags
        estimation_flags = self.flag_estimations(holdings)

        # Recommendations
        recommendations = self._generate_recommendations(checks, overall_score)

        assessment = QualityAssessment(
            overall_score=overall_score,
            overall_level=level,
            checks=checks,
            checks_passed=passed,
            checks_failed=failed,
            coverage_report=coverage_report,
            estimation_flags=estimation_flags,
            recommendations=recommendations,
            assessed_at=_utcnow().isoformat(),
        )
        assessment.provenance_hash = _hash_data({
            "score": overall_score, "passed": passed, "failed": failed,
        })

        self.logger.info(
            "Quality assessment: score=%.1f (%s), %d/%d checks passed",
            overall_score, level.value, passed, len(checks),
        )
        return assessment

    def get_coverage_report(
        self,
        pai_data: Dict[str, Any],
    ) -> CoverageReport:
        """Generate a data coverage report for PAI indicators.

        Args:
            pai_data: PAI indicator data with coverage information.

        Returns:
            CoverageReport with per-indicator coverage.
        """
        indicators = pai_data.get("pai_indicators", {})
        if isinstance(indicators, list):
            indicator_map = {str(i): v for i, v in enumerate(indicators)}
        else:
            indicator_map = indicators

        total = len(indicator_map) or 18
        covered = 0
        by_indicator: Dict[str, Dict[str, Any]] = {}
        by_source: Dict[str, int] = {}

        for ind_key, ind_data in indicator_map.items():
            if isinstance(ind_data, dict):
                cov = float(ind_data.get("coverage_pct", 0.0))
                source = ind_data.get("data_source", "unavailable")
                name = ind_data.get("name", f"PAI {ind_key}")
            else:
                cov = 0.0
                source = "unavailable"
                name = f"PAI {ind_key}"

            if cov > 0:
                covered += 1

            by_indicator[ind_key] = {
                "name": name,
                "coverage_pct": cov,
                "data_source": source,
            }
            by_source[source] = by_source.get(source, 0) + 1

        coverage_pct = round((covered / max(total, 1)) * 100, 1)

        report = CoverageReport(
            total_indicators=total,
            covered_indicators=covered,
            coverage_pct=coverage_pct,
            by_indicator=by_indicator,
            by_data_source=by_source,
        )
        report.provenance_hash = _hash_data({
            "total": total, "covered": covered, "pct": coverage_pct,
        })

        return report

    def flag_estimations(
        self,
        holdings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Flag holdings that use estimated data.

        Args:
            holdings: Portfolio holdings.

        Returns:
            List of estimation flags with holding details.
        """
        flags: List[Dict[str, Any]] = []

        for h in holdings:
            data_source = h.get("data_source", "")
            emissions_source = h.get("emissions", {}).get("data_source", "")
            esg_source = h.get("esg_data_source", "")

            estimated_fields: List[str] = []

            if data_source in ("estimated", "proxy", "modeled"):
                estimated_fields.append(f"holding_data ({data_source})")
            if emissions_source in ("estimated", "proxy", "modeled"):
                estimated_fields.append(f"emissions ({emissions_source})")
            if esg_source in ("estimated", "proxy"):
                estimated_fields.append(f"esg ({esg_source})")

            if estimated_fields:
                flags.append({
                    "isin": h.get("isin", ""),
                    "name": h.get("name", ""),
                    "weight": h.get("weight", 0.0),
                    "estimated_fields": estimated_fields,
                })

        if self.config.warn_on_proxy_data and flags:
            self.logger.warning(
                "Data quality: %d/%d holdings use estimated/proxy data",
                len(flags), len(holdings),
            )

        return flags

    def validate_data_age(
        self,
        holdings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate data age for all holdings.

        Args:
            holdings: Portfolio holdings with data timestamps.

        Returns:
            Data age validation report.
        """
        now = _utcnow()
        stale_count = 0
        fresh_count = 0
        ages: List[int] = []

        for h in holdings:
            data_date_str = h.get("data_date", "")
            if not data_date_str:
                stale_count += 1
                continue

            try:
                data_date = datetime.fromisoformat(data_date_str).replace(
                    tzinfo=timezone.utc
                )
                age_days = (now - data_date).days
                ages.append(age_days)

                if age_days > self.config.max_data_age_days:
                    stale_count += 1
                else:
                    fresh_count += 1
            except (ValueError, TypeError):
                stale_count += 1

        avg_age = round(sum(ages) / max(len(ages), 1), 0) if ages else 0
        max_age = max(ages) if ages else 0

        return {
            "total_holdings": len(holdings),
            "fresh_count": fresh_count,
            "stale_count": stale_count,
            "max_data_age_days": self.config.max_data_age_days,
            "average_age_days": avg_age,
            "max_age_days": max_age,
            "freshness_pct": round(
                (fresh_count / max(len(holdings), 1)) * 100, 1
            ),
            "passed": stale_count == 0,
        }

    def get_quality_score(
        self,
        pai_data: Dict[str, Any],
        holdings: List[Dict[str, Any]],
    ) -> float:
        """Get the overall quality score (0-100).

        Args:
            pai_data: PAI indicator data.
            holdings: Portfolio holdings.

        Returns:
            Quality score as a float.
        """
        assessment = self.assess_quality(pai_data, holdings)
        return assessment.overall_score

    # -------------------------------------------------------------------------
    # Internal Check Methods
    # -------------------------------------------------------------------------

    def _check_pai_coverage(
        self, pai_data: Dict[str, Any]
    ) -> QualityCheckResult:
        """Check PAI indicator data coverage."""
        report = self.get_coverage_report(pai_data)
        passed = report.coverage_pct >= self.config.min_coverage
        return QualityCheckResult(
            category=CheckCategory.COVERAGE,
            check_name="pai_data_coverage",
            status=QualityLevel.HIGH if passed else QualityLevel.LOW,
            value=report.coverage_pct,
            threshold=self.config.min_coverage,
            passed=passed,
            detail=f"{report.covered_indicators}/{report.total_indicators} covered",
        )

    def _check_holdings_isin(
        self, holdings: List[Dict[str, Any]]
    ) -> QualityCheckResult:
        """Check holdings ISIN coverage."""
        total = len(holdings) or 1
        with_isin = sum(1 for h in holdings if h.get("isin"))
        pct = round((with_isin / total) * 100, 1)
        passed = pct >= 90.0
        return QualityCheckResult(
            category=CheckCategory.COVERAGE,
            check_name="holdings_isin_coverage",
            status=QualityLevel.HIGH if passed else QualityLevel.LOW,
            value=pct,
            threshold=90.0,
            passed=passed,
            detail=f"{with_isin}/{total} holdings have ISIN",
        )

    def _check_esg_coverage(
        self, holdings: List[Dict[str, Any]]
    ) -> QualityCheckResult:
        """Check ESG rating coverage."""
        total = len(holdings) or 1
        with_esg = sum(1 for h in holdings if h.get("esg_rating"))
        pct = round((with_esg / total) * 100, 1)
        passed = pct >= 50.0
        return QualityCheckResult(
            category=CheckCategory.COVERAGE,
            check_name="esg_rating_coverage",
            status=QualityLevel.HIGH if passed else QualityLevel.LOW,
            value=pct,
            threshold=50.0,
            passed=passed,
            detail=f"{with_esg}/{total} holdings have ESG rating",
        )

    def _check_data_freshness(
        self, holdings: List[Dict[str, Any]], data_type: str
    ) -> QualityCheckResult:
        """Check data freshness for a specific data type."""
        now = _utcnow()
        date_field = f"{data_type}_date" if data_type != "emissions" else "data_date"
        fresh = 0
        total = len(holdings) or 1

        for h in holdings:
            date_str = h.get(date_field, h.get("data_date", ""))
            if not date_str:
                continue
            try:
                dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                if (now - dt).days <= self.config.max_data_age_days:
                    fresh += 1
            except (ValueError, TypeError):
                pass

        pct = round((fresh / total) * 100, 1)
        passed = pct >= 50.0
        return QualityCheckResult(
            category=CheckCategory.FRESHNESS,
            check_name=f"{data_type}_data_freshness",
            status=QualityLevel.HIGH if passed else QualityLevel.LOW,
            value=pct,
            threshold=50.0,
            passed=passed,
            detail=f"{fresh}/{total} holdings have fresh {data_type} data",
        )

    def _check_emissions_accuracy(
        self, holdings: List[Dict[str, Any]]
    ) -> QualityCheckResult:
        """Check emissions data accuracy (reported vs estimated)."""
        total = len(holdings) or 1
        reported = sum(
            1 for h in holdings
            if h.get("emissions", {}).get("data_source") == "reported"
            or h.get("data_source") == "reported"
        )
        pct = round((reported / total) * 100, 1)
        passed = pct >= (100.0 - self.config.max_estimation_pct)
        return QualityCheckResult(
            category=CheckCategory.ACCURACY,
            check_name="emissions_accuracy",
            status=QualityLevel.HIGH if passed else QualityLevel.LOW,
            value=pct,
            threshold=100.0 - self.config.max_estimation_pct,
            passed=passed,
            detail=f"{reported}/{total} use reported emissions data",
        )

    def _check_taxonomy_accuracy(
        self,
        taxonomy_data: Dict[str, Any],
        holdings: List[Dict[str, Any]],
    ) -> QualityCheckResult:
        """Check taxonomy alignment data accuracy."""
        total = len(holdings) or 1
        with_data = sum(
            1 for h in holdings
            if h.get("taxonomy_eligible") is not None
        )
        pct = round((with_data / total) * 100, 1)
        passed = pct >= 30.0
        return QualityCheckResult(
            category=CheckCategory.ACCURACY,
            check_name="taxonomy_accuracy",
            status=QualityLevel.HIGH if passed else QualityLevel.ACCEPTABLE,
            value=pct,
            threshold=30.0,
            passed=passed,
            detail=f"{with_data}/{total} have taxonomy classification",
        )

    def _check_portfolio_completeness(
        self, holdings: List[Dict[str, Any]]
    ) -> QualityCheckResult:
        """Check portfolio field completeness."""
        required_fields = [
            "isin", "name", "weight", "sector", "country_code",
        ]
        total = len(holdings) or 1
        complete = 0

        for h in holdings:
            fields_present = sum(1 for f in required_fields if h.get(f))
            if fields_present == len(required_fields):
                complete += 1

        pct = round((complete / total) * 100, 1)
        passed = pct >= self.config.completeness_threshold
        return QualityCheckResult(
            category=CheckCategory.COMPLETENESS,
            check_name="portfolio_completeness",
            status=QualityLevel.HIGH if passed else QualityLevel.LOW,
            value=pct,
            threshold=self.config.completeness_threshold,
            passed=passed,
            detail=f"{complete}/{total} holdings fully complete",
        )

    def _check_field_coverage(
        self,
        holdings: List[Dict[str, Any]],
        field_name: str,
        label: str,
    ) -> QualityCheckResult:
        """Check coverage of a specific field."""
        total = len(holdings) or 1
        with_field = sum(1 for h in holdings if h.get(field_name))
        pct = round((with_field / total) * 100, 1)
        passed = pct >= 70.0
        return QualityCheckResult(
            category=CheckCategory.COMPLETENESS,
            check_name=f"{label}_completeness",
            status=QualityLevel.HIGH if passed else QualityLevel.LOW,
            value=pct,
            threshold=70.0,
            passed=passed,
            detail=f"{with_field}/{total} holdings have {label}",
        )

    def _check_weight_consistency(
        self, holdings: List[Dict[str, Any]]
    ) -> QualityCheckResult:
        """Check portfolio weight sum consistency."""
        total_weight = sum(float(h.get("weight", 0.0)) for h in holdings)
        deviation = abs(total_weight - 100.0)
        passed = deviation <= 5.0
        return QualityCheckResult(
            category=CheckCategory.CONSISTENCY,
            check_name="weight_consistency",
            status=QualityLevel.HIGH if passed else QualityLevel.LOW,
            value=round(total_weight, 2),
            threshold=100.0,
            passed=passed,
            detail=f"Weight sum: {total_weight:.2f}% (deviation: {deviation:.2f}%)",
        )

    def _check_cross_source(
        self, holdings: List[Dict[str, Any]]
    ) -> QualityCheckResult:
        """Check cross-source data consistency."""
        inconsistent = 0
        for h in holdings:
            esg_rating = h.get("esg_rating", "")
            esg_score = float(h.get("esg_score", 0.0))
            if esg_rating and esg_score > 0:
                # Simple consistency: high rating should map to high score
                if esg_rating in ("AAA", "AA", "A") and esg_score < 30:
                    inconsistent += 1

        total = len(holdings) or 1
        consistent_pct = round(
            ((total - inconsistent) / total) * 100, 1
        )
        passed = consistent_pct >= 90.0
        return QualityCheckResult(
            category=CheckCategory.CONSISTENCY,
            check_name="cross_source_consistency",
            status=QualityLevel.HIGH if passed else QualityLevel.LOW,
            value=consistent_pct,
            threshold=90.0,
            passed=passed,
            detail=f"{inconsistent} inconsistencies found",
        )

    def _check_estimation_ratio(
        self, holdings: List[Dict[str, Any]]
    ) -> QualityCheckResult:
        """Check the ratio of estimated vs reported data."""
        total = len(holdings) or 1
        estimated = sum(
            1 for h in holdings
            if h.get("data_source", "") in ("estimated", "proxy", "modeled")
            or h.get("emissions", {}).get("data_source") in ("estimated", "proxy")
        )
        pct = round((estimated / total) * 100, 1)
        passed = pct <= self.config.max_estimation_pct
        return QualityCheckResult(
            category=CheckCategory.ESTIMATION,
            check_name="estimation_ratio",
            status=QualityLevel.HIGH if passed else QualityLevel.LOW,
            value=pct,
            threshold=self.config.max_estimation_pct,
            passed=passed,
            detail=f"{estimated}/{total} holdings use estimated data ({pct:.1f}%)",
        )

    def _check_provenance(
        self, holdings: List[Dict[str, Any]]
    ) -> QualityCheckResult:
        """Check data provenance tracking coverage."""
        total = len(holdings) or 1
        with_provenance = sum(
            1 for h in holdings if h.get("provenance_hash") or h.get("data_source")
        )
        pct = round((with_provenance / total) * 100, 1)
        passed = pct >= 50.0 or not self.config.require_provenance
        return QualityCheckResult(
            category=CheckCategory.PROVENANCE,
            check_name="provenance_coverage",
            status=QualityLevel.HIGH if passed else QualityLevel.LOW,
            value=pct,
            threshold=50.0,
            passed=passed,
            detail=f"{with_provenance}/{total} holdings have provenance",
        )

    # -------------------------------------------------------------------------
    # Recommendations
    # -------------------------------------------------------------------------

    def _generate_recommendations(
        self,
        checks: List[QualityCheckResult],
        overall_score: float,
    ) -> List[str]:
        """Generate quality improvement recommendations.

        Args:
            checks: Completed quality checks.
            overall_score: Overall quality score.

        Returns:
            List of recommendations.
        """
        recommendations: List[str] = []

        for c in checks:
            if c.passed:
                continue

            if c.check_name == "pai_data_coverage":
                recommendations.append(
                    "Improve PAI data coverage by sourcing additional ESG data providers"
                )
            elif c.check_name == "esg_rating_coverage":
                recommendations.append(
                    "Increase ESG rating coverage by onboarding ESG rating provider"
                )
            elif c.check_name == "estimation_ratio":
                recommendations.append(
                    "Reduce estimation ratio by engaging investees for reported data"
                )
            elif "freshness" in c.check_name:
                recommendations.append(
                    f"Update stale {c.check_name.split('_')[0]} data "
                    f"(max age: {self.config.max_data_age_days} days)"
                )
            elif c.check_name == "weight_consistency":
                recommendations.append(
                    "Review portfolio weights; sum should be 100%"
                )
            elif "completeness" in c.check_name:
                recommendations.append(
                    f"Improve {c.check_name.replace('_', ' ')} to {c.threshold:.0f}%+"
                )

        if overall_score < 60:
            recommendations.append(
                "Overall data quality is below acceptable threshold; "
                "prioritize reported data sourcing"
            )

        return recommendations
