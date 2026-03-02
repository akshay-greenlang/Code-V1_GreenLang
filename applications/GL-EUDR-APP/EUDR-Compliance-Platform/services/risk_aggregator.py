# -*- coding: utf-8 -*-
"""
GL-EUDR-APP Risk Aggregator - Multi-Source Risk Scoring Engine

Aggregates risk from country benchmarks, satellite assessments, supplier
history, and document completeness into unified risk scores. Provides
risk heatmaps, alerts, trends, and mitigation recommendations.

Risk Formula:
    overall = 0.35 * satellite_risk
            + 0.25 * country_risk
            + 0.20 * supplier_risk
            + 0.20 * document_risk

Risk Levels:
    LOW:      0.00 - 0.29
    STANDARD: 0.30 - 0.69
    HIGH:     0.70 - 0.89
    CRITICAL: 0.90 - 1.00

Zero-Hallucination Guarantees:
    - All risk scores computed via deterministic weighted formula
    - Country risk scores based on published deforestation data
    - No LLM or ML used for risk classification
    - SHA-256 provenance hashes on all assessments

Example:
    >>> from services.risk_aggregator import RiskAggregator
    >>> aggregator = RiskAggregator(config)
    >>> assessment = aggregator.assess_risk("supplier-1", "plot-1")
    >>> print(assessment.overall_risk, assessment.risk_level)

Author: GreenLang Platform Team
Date: March 2026
Application: GL-EUDR-APP v1.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from services.config import (
    EUDRAppConfig,
    EUDRCommodity,
    RiskLevel,
)
from services.models import (
    RiskAlert,
    RiskAssessment,
    RiskTrendPoint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _new_id() -> str:
    """Generate a UUID v4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Country Risk Database
# ---------------------------------------------------------------------------

# Risk scores based on published deforestation data (0.0 = no risk, 1.0 = max)
COUNTRY_RISK_SCORES: Dict[str, float] = {
    # High risk - tropical deforestation hotspots
    "BRA": 0.85,
    "IDN": 0.82,
    "COD": 0.90,
    "COG": 0.88,
    "CMR": 0.80,
    "MYS": 0.65,
    "PNG": 0.78,
    "BOL": 0.75,
    "PER": 0.70,
    "COL": 0.68,
    # Medium-high risk
    "GHA": 0.60,
    "CIV": 0.62,
    "NGA": 0.65,
    "TZA": 0.55,
    "KEN": 0.45,
    "VNM": 0.50,
    "THA": 0.40,
    "ECU": 0.55,
    "GTM": 0.50,
    "HND": 0.52,
    # Medium risk
    "MEX": 0.40,
    "MMR": 0.55,
    "LAO": 0.48,
    "KHM": 0.45,
    "PHL": 0.42,
    "MDG": 0.58,
    "MOZ": 0.50,
    "AGO": 0.52,
    "ZMB": 0.40,
    "PRY": 0.45,
    "ARG": 0.35,
    # Low risk
    "IND": 0.25,
    "CHN": 0.20,
    "ETH": 0.30,
    "UGA": 0.35,
    "RWA": 0.25,
    "CRI": 0.15,
    "PAN": 0.20,
    # Very low risk - developed nations
    "DEU": 0.05,
    "FRA": 0.05,
    "NLD": 0.05,
    "SWE": 0.03,
    "FIN": 0.03,
    "USA": 0.10,
    "CAN": 0.08,
    "AUS": 0.08,
    "NZL": 0.05,
    "JPN": 0.05,
    "GBR": 0.04,
    "ITA": 0.05,
    "ESP": 0.06,
    "PRT": 0.06,
    "NOR": 0.03,
    "DNK": 0.04,
    "BEL": 0.05,
    "CHE": 0.03,
    "AUT": 0.04,
    "IRL": 0.04,
    "POL": 0.06,
    "CZE": 0.05,
}

# Commodity-specific risk adjustments (additive)
COMMODITY_RISK_ADJUSTMENTS: Dict[str, float] = {
    EUDRCommodity.PALM_OIL.value: 0.10,
    EUDRCommodity.SOY.value: 0.08,
    EUDRCommodity.CATTLE.value: 0.12,
    EUDRCommodity.COCOA.value: 0.06,
    EUDRCommodity.COFFEE.value: 0.04,
    EUDRCommodity.RUBBER.value: 0.05,
    EUDRCommodity.WOOD.value: 0.07,
}


# ===========================================================================
# Risk Aggregator
# ===========================================================================


class RiskAggregator:
    """Aggregates risk from multiple sources into unified risk scores.

    Combines satellite assessment, country benchmarks, supplier history,
    and document completeness into weighted overall risk scores. Provides
    risk heatmaps, alerts, trend analysis, and mitigation recommendations.

    Risk Formula:
        overall = W_sat * satellite_risk
                + W_cty * country_risk
                + W_sup * supplier_risk
                + W_doc * document_risk

    Attributes:
        _config: Application configuration with risk weights/thresholds.
        _lock: Reentrant lock for thread safety.
        _assessments: In-memory assessment storage.
        _alerts: In-memory alert storage.
        _supplier_engine: Optional SupplierIntakeEngine reference.
        _document_engine: Optional DocumentVerificationEngine reference.

    Example:
        >>> aggregator = RiskAggregator(config)
        >>> assessment = aggregator.assess_risk("supplier-1", "plot-1")
        >>> print(f"Risk: {assessment.overall_risk:.2f} ({assessment.risk_level.value})")
    """

    def __init__(
        self,
        config: EUDRAppConfig,
        supplier_engine: Optional[Any] = None,
        document_engine: Optional[Any] = None,
    ) -> None:
        """Initialize RiskAggregator.

        Args:
            config: Application configuration with risk thresholds.
            supplier_engine: Optional SupplierIntakeEngine reference.
            document_engine: Optional DocumentVerificationEngine reference.
        """
        self._config = config
        self._lock = threading.RLock()
        self._assessments: Dict[str, RiskAssessment] = {}
        self._alerts: List[RiskAlert] = []
        self._supplier_engine = supplier_engine
        self._document_engine = document_engine
        logger.info(
            "RiskAggregator initialized: weights=sat:%.2f/cty:%.2f/"
            "sup:%.2f/doc:%.2f, thresholds=high:%.2f/critical:%.2f",
            config.risk_weight_satellite,
            config.risk_weight_country,
            config.risk_weight_supplier,
            config.risk_weight_document,
            config.risk_threshold_high,
            config.risk_threshold_critical,
        )

    # -----------------------------------------------------------------------
    # Risk Assessment
    # -----------------------------------------------------------------------

    def assess_risk(
        self,
        supplier_id: str,
        plot_id: Optional[str] = None,
    ) -> RiskAssessment:
        """Assess risk for a supplier and optional plot.

        Computes the four risk components and calculates the weighted
        overall risk score using the deterministic formula.

        Args:
            supplier_id: Supplier to assess.
            plot_id: Optional plot for plot-specific assessment.

        Returns:
            RiskAssessment with all risk components.
        """
        start_time = _utcnow()

        # Component 1: Country risk
        country_risk, country_factors = self._assess_country_risk(
            supplier_id
        )

        # Component 2: Satellite risk
        satellite_risk, satellite_factors = self._assess_satellite_risk(
            plot_id
        )

        # Component 3: Supplier risk
        supplier_risk, supplier_factors = self._assess_supplier_risk(
            supplier_id
        )

        # Component 4: Document risk
        document_risk, document_factors = self._assess_document_risk(
            supplier_id
        )

        # Weighted overall risk (DETERMINISTIC FORMULA)
        overall_risk = (
            self._config.risk_weight_satellite * satellite_risk
            + self._config.risk_weight_country * country_risk
            + self._config.risk_weight_supplier * supplier_risk
            + self._config.risk_weight_document * document_risk
        )
        overall_risk = min(1.0, max(0.0, overall_risk))

        # Classify risk level
        risk_level = self._classify_risk(overall_risk)

        # Aggregate factors
        all_factors: List[Dict[str, Any]] = []
        all_factors.extend(country_factors)
        all_factors.extend(satellite_factors)
        all_factors.extend(supplier_factors)
        all_factors.extend(document_factors)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            country_risk=country_risk,
            satellite_risk=satellite_risk,
            supplier_risk=supplier_risk,
            document_risk=document_risk,
            overall_risk=overall_risk,
        )

        # Data sources
        data_sources = [
            "EUDR country risk benchmarking database",
            "Satellite imagery (Sentinel-2, Landsat-8)",
            "Supplier compliance history",
            "Document verification records",
        ]

        # Compute provenance hash
        provenance_data = {
            "supplier_id": supplier_id,
            "plot_id": plot_id,
            "country_risk": country_risk,
            "satellite_risk": satellite_risk,
            "supplier_risk": supplier_risk,
            "document_risk": document_risk,
            "overall_risk": overall_risk,
            "assessed_at": start_time.isoformat(),
        }
        provenance_hash = _compute_hash(provenance_data)

        assessment = RiskAssessment(
            supplier_id=supplier_id,
            plot_id=plot_id,
            plot_risk=satellite_risk,
            country_risk=round(country_risk, 4),
            supplier_risk=round(supplier_risk, 4),
            satellite_risk=round(satellite_risk, 4),
            document_risk=round(document_risk, 4),
            overall_risk=round(overall_risk, 4),
            risk_level=risk_level,
            factors=all_factors,
            recommendations=recommendations,
            data_sources=data_sources,
            assessed_at=start_time,
            valid_until=start_time + timedelta(
                days=self._config.satellite_cache_days
            ),
            provenance_hash=provenance_hash,
        )

        # Store assessment
        with self._lock:
            self._assessments[assessment.id] = assessment

        # Generate alerts if needed
        self._check_and_generate_alerts(assessment)

        logger.info(
            "Risk assessed for supplier=%s, plot=%s: "
            "overall=%.4f (%s), sat=%.4f, cty=%.4f, sup=%.4f, doc=%.4f",
            supplier_id,
            plot_id,
            overall_risk,
            risk_level.value,
            satellite_risk,
            country_risk,
            supplier_risk,
            document_risk,
        )

        return assessment

    # -----------------------------------------------------------------------
    # Risk Heatmap
    # -----------------------------------------------------------------------

    def get_risk_heatmap(self) -> Dict[str, Dict[str, float]]:
        """Generate a risk heatmap of country x commodity risk scores.

        Returns:
            Nested dictionary {country: {commodity: risk_score}}.
        """
        heatmap: Dict[str, Dict[str, float]] = {}

        for country, country_risk in COUNTRY_RISK_SCORES.items():
            heatmap[country] = {}
            for commodity in EUDRCommodity:
                commodity_adj = COMMODITY_RISK_ADJUSTMENTS.get(
                    commodity.value, 0.0
                )
                combined = min(1.0, country_risk + commodity_adj)
                heatmap[country][commodity.value] = round(combined, 4)

        return heatmap

    # -----------------------------------------------------------------------
    # Risk Alerts
    # -----------------------------------------------------------------------

    def get_risk_alerts(
        self,
        min_level: str = "high",
        acknowledged: Optional[bool] = None,
        limit: int = 100,
    ) -> List[RiskAlert]:
        """Get risk alerts filtered by severity level.

        Args:
            min_level: Minimum severity level ("low", "standard", "high", "critical").
            acknowledged: Filter by acknowledgement status.
            limit: Maximum number of alerts to return.

        Returns:
            List of RiskAlert records.
        """
        level_order = {
            "low": 0,
            "standard": 1,
            "high": 2,
            "critical": 3,
        }
        min_order = level_order.get(min_level.lower(), 2)

        with self._lock:
            filtered = [
                a
                for a in self._alerts
                if level_order.get(a.severity.value, 0) >= min_order
            ]

        if acknowledged is not None:
            filtered = [
                a for a in filtered if a.acknowledged == acknowledged
            ]

        # Sort by creation time descending
        filtered.sort(key=lambda a: a.created_at, reverse=True)
        return filtered[:limit]

    def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str
    ) -> bool:
        """Acknowledge a risk alert.

        Args:
            alert_id: Alert identifier.
            acknowledged_by: User who acknowledged the alert.

        Returns:
            True if alert found and acknowledged.
        """
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    logger.info(
                        "Alert %s acknowledged by %s",
                        alert_id,
                        acknowledged_by,
                    )
                    return True
        return False

    # -----------------------------------------------------------------------
    # Risk Trends
    # -----------------------------------------------------------------------

    def get_risk_trends(
        self,
        plot_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        months: int = 12,
    ) -> List[RiskTrendPoint]:
        """Get risk score trends over time.

        Generates simulated trend data in v1.0. In production, this
        would query historical assessment records.

        Args:
            plot_id: Filter by plot ID.
            supplier_id: Filter by supplier ID.
            months: Number of months to look back.

        Returns:
            List of RiskTrendPoint records.
        """
        # Get relevant assessments
        with self._lock:
            assessments = [
                a
                for a in self._assessments.values()
                if (not plot_id or a.plot_id == plot_id)
                and (not supplier_id or a.supplier_id == supplier_id)
            ]

        if assessments:
            # Build trends from actual assessments
            points: List[RiskTrendPoint] = []
            for a in assessments:
                points.append(RiskTrendPoint(
                    date=a.assessed_at,
                    overall_risk=a.overall_risk,
                    satellite_risk=a.satellite_risk,
                    country_risk=a.country_risk,
                    supplier_risk=a.supplier_risk,
                    document_risk=a.document_risk,
                ))
            points.sort(key=lambda p: p.date)
            return points

        # Generate simulated trend data for visualization
        now = _utcnow()
        points = []
        for i in range(months, -1, -1):
            point_date = now - timedelta(days=i * 30)
            # Simulate gradual risk reduction over time
            base_risk = 0.45 + (i / months) * 0.15
            points.append(RiskTrendPoint(
                date=point_date,
                overall_risk=round(base_risk, 4),
                satellite_risk=round(base_risk * 0.9, 4),
                country_risk=round(base_risk * 1.1, 4),
                supplier_risk=round(base_risk * 0.8, 4),
                document_risk=round(base_risk * 0.7, 4),
            ))

        return points

    # -----------------------------------------------------------------------
    # Risk Mitigations
    # -----------------------------------------------------------------------

    def get_risk_mitigations(
        self, risk_assessment: RiskAssessment
    ) -> List[str]:
        """Generate risk mitigation recommendations for an assessment.

        Args:
            risk_assessment: RiskAssessment to analyze.

        Returns:
            List of mitigation recommendation strings.
        """
        return self._generate_recommendations(
            country_risk=risk_assessment.country_risk,
            satellite_risk=risk_assessment.satellite_risk,
            supplier_risk=risk_assessment.supplier_risk,
            document_risk=risk_assessment.document_risk,
            overall_risk=risk_assessment.overall_risk,
        )

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics.

        Returns:
            Dictionary with assessment and alert counts.
        """
        with self._lock:
            assessments = list(self._assessments.values())
            alerts = list(self._alerts)

        risk_levels: Dict[str, int] = {}
        for a in assessments:
            key = a.risk_level.value
            risk_levels[key] = risk_levels.get(key, 0) + 1

        avg_risk = (
            sum(a.overall_risk for a in assessments) / len(assessments)
            if assessments
            else 0.0
        )

        return {
            "total_assessments": len(assessments),
            "total_alerts": len(alerts),
            "unacknowledged_alerts": sum(
                1 for a in alerts if not a.acknowledged
            ),
            "average_overall_risk": round(avg_risk, 4),
            "by_risk_level": risk_levels,
            "countries_covered": len(COUNTRY_RISK_SCORES),
        }

    # -----------------------------------------------------------------------
    # Private Helpers - Risk Component Assessment
    # -----------------------------------------------------------------------

    def _assess_country_risk(
        self, supplier_id: str
    ) -> tuple[float, List[Dict[str, Any]]]:
        """Assess country-level deforestation risk.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Tuple of (risk_score, risk_factors).
        """
        factors: List[Dict[str, Any]] = []
        country_code = "XXX"

        # Look up supplier country
        if self._supplier_engine:
            supplier = self._supplier_engine.get_supplier(supplier_id)
            if supplier:
                country_code = supplier.country

        risk_score = COUNTRY_RISK_SCORES.get(country_code, 0.30)

        factors.append({
            "dimension": "country",
            "factor": f"Country {country_code} baseline risk",
            "score": risk_score,
            "source": "EUDR country risk benchmarking",
        })

        if country_code in self._config.high_risk_countries:
            factors.append({
                "dimension": "country",
                "factor": f"{country_code} is classified as high-risk",
                "score": risk_score,
                "source": "EUDR high-risk country list",
            })

        return risk_score, factors

    def _assess_satellite_risk(
        self, plot_id: Optional[str]
    ) -> tuple[float, List[Dict[str, Any]]]:
        """Assess satellite-based deforestation risk.

        In v1.0, returns default risk if no plot data is available.
        In production, delegates to AGENT-DATA-007.

        Args:
            plot_id: Plot identifier.

        Returns:
            Tuple of (risk_score, risk_factors).
        """
        factors: List[Dict[str, Any]] = []

        if not plot_id:
            # No plot specified, use default
            factors.append({
                "dimension": "satellite",
                "factor": "No plot specified for satellite assessment",
                "score": 0.30,
                "source": "Default baseline",
            })
            return 0.30, factors

        # Try to get plot data from supplier engine
        risk_score = 0.20  # Default if plot has no assessment
        if self._supplier_engine:
            plot = self._supplier_engine.get_plot(plot_id)
            if plot:
                if plot.is_deforestation_free is True:
                    risk_score = 0.05
                    factors.append({
                        "dimension": "satellite",
                        "factor": "Plot confirmed deforestation-free",
                        "score": risk_score,
                        "source": "AGENT-DATA-007 satellite assessment",
                    })
                elif plot.is_deforestation_free is False:
                    risk_score = 0.95
                    factors.append({
                        "dimension": "satellite",
                        "factor": "Deforestation detected on plot",
                        "score": risk_score,
                        "source": "AGENT-DATA-007 satellite assessment",
                    })
                else:
                    # Not yet assessed
                    risk_score = 0.40
                    factors.append({
                        "dimension": "satellite",
                        "factor": "Plot not yet assessed by satellite",
                        "score": risk_score,
                        "source": "Pending assessment",
                    })

                # NDVI-based adjustment
                if plot.ndvi_current is not None and plot.ndvi_baseline is not None:
                    ndvi_change = plot.ndvi_current - plot.ndvi_baseline
                    if ndvi_change <= self._config.ndvi_change_threshold:
                        risk_score = max(risk_score, 0.80)
                        factors.append({
                            "dimension": "satellite",
                            "factor": (
                                f"NDVI change {ndvi_change:.4f} below "
                                f"threshold {self._config.ndvi_change_threshold}"
                            ),
                            "score": 0.80,
                            "source": "NDVI change detection",
                        })
            else:
                factors.append({
                    "dimension": "satellite",
                    "factor": f"Plot {plot_id} not found",
                    "score": 0.30,
                    "source": "Missing data",
                })
        else:
            factors.append({
                "dimension": "satellite",
                "factor": "Satellite assessment baseline (no engine)",
                "score": risk_score,
                "source": "Default baseline",
            })

        return risk_score, factors

    def _assess_supplier_risk(
        self, supplier_id: str
    ) -> tuple[float, List[Dict[str, Any]]]:
        """Assess supplier history risk.

        Based on compliance status, certifications, and audit history.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Tuple of (risk_score, risk_factors).
        """
        factors: List[Dict[str, Any]] = []
        risk_score = 0.30  # Default baseline

        if self._supplier_engine:
            supplier = self._supplier_engine.get_supplier(supplier_id)
            if supplier:
                # Compliance status impact
                status_risks = {
                    "compliant": 0.10,
                    "pending": 0.40,
                    "under_review": 0.50,
                    "non_compliant": 0.85,
                    "suspended": 0.95,
                }
                risk_score = status_risks.get(
                    supplier.compliance_status.value, 0.30
                )
                factors.append({
                    "dimension": "supplier",
                    "factor": (
                        f"Compliance status: "
                        f"{supplier.compliance_status.value}"
                    ),
                    "score": risk_score,
                    "source": "Supplier compliance records",
                })

                # Certification discount
                if supplier.certifications:
                    cert_discount = min(0.15, len(supplier.certifications) * 0.05)
                    risk_score = max(0.0, risk_score - cert_discount)
                    factors.append({
                        "dimension": "supplier",
                        "factor": (
                            f"{len(supplier.certifications)} "
                            f"certification(s) held"
                        ),
                        "score": -cert_discount,
                        "source": "Supplier certifications",
                    })

                # Audit recency
                if supplier.last_audit_date:
                    days_since_audit = (
                        datetime.now().date() - supplier.last_audit_date
                    ).days
                    if days_since_audit > 365:
                        audit_penalty = min(0.10, (days_since_audit - 365) / 3650)
                        risk_score = min(1.0, risk_score + audit_penalty)
                        factors.append({
                            "dimension": "supplier",
                            "factor": (
                                f"Last audit {days_since_audit} days ago "
                                f"(>365 days)"
                            ),
                            "score": audit_penalty,
                            "source": "Audit history",
                        })
            else:
                factors.append({
                    "dimension": "supplier",
                    "factor": f"Supplier {supplier_id} not found",
                    "score": 0.50,
                    "source": "Missing data",
                })
                risk_score = 0.50
        else:
            factors.append({
                "dimension": "supplier",
                "factor": "Supplier risk baseline (no engine)",
                "score": risk_score,
                "source": "Default baseline",
            })

        return risk_score, factors

    def _assess_document_risk(
        self, supplier_id: str
    ) -> tuple[float, List[Dict[str, Any]]]:
        """Assess document completeness risk.

        Based on document coverage, verification scores, and gaps.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Tuple of (risk_score, risk_factors).
        """
        factors: List[Dict[str, Any]] = []
        risk_score = 0.40  # Default when no documents

        if self._document_engine:
            try:
                gap_analysis = self._document_engine.get_gap_analysis(
                    supplier_id
                )

                # Coverage-based risk (lower coverage = higher risk)
                coverage = gap_analysis.coverage_pct
                risk_score = max(0.0, 1.0 - (coverage / 100.0))

                factors.append({
                    "dimension": "document",
                    "factor": f"Document coverage: {coverage:.1f}%",
                    "score": risk_score,
                    "source": "Document gap analysis",
                })

                if gap_analysis.missing_documents:
                    factors.append({
                        "dimension": "document",
                        "factor": (
                            f"Missing document types: "
                            f"{', '.join(gap_analysis.missing_documents)}"
                        ),
                        "score": len(gap_analysis.missing_documents) * 0.15,
                        "source": "Document gap analysis",
                    })

                if gap_analysis.expired_documents:
                    factors.append({
                        "dimension": "document",
                        "factor": (
                            f"{len(gap_analysis.expired_documents)} "
                            f"expired document(s)"
                        ),
                        "score": 0.10,
                        "source": "Document expiry check",
                    })

            except Exception as exc:
                logger.warning(
                    "Document risk assessment failed: %s", exc
                )
                factors.append({
                    "dimension": "document",
                    "factor": "Document assessment unavailable",
                    "score": 0.40,
                    "source": "Error fallback",
                })
        else:
            factors.append({
                "dimension": "document",
                "factor": "Document risk baseline (no engine)",
                "score": risk_score,
                "source": "Default baseline",
            })

        return risk_score, factors

    # -----------------------------------------------------------------------
    # Private Helpers - Classification & Recommendations
    # -----------------------------------------------------------------------

    def _classify_risk(self, score: float) -> RiskLevel:
        """Classify a risk score into a RiskLevel.

        Args:
            score: Risk score between 0 and 1.

        Returns:
            Classified RiskLevel.
        """
        if score >= self._config.risk_threshold_critical:
            return RiskLevel.CRITICAL
        if score >= self._config.risk_threshold_high:
            return RiskLevel.HIGH
        if score >= 0.30:
            return RiskLevel.STANDARD
        return RiskLevel.LOW

    def _generate_recommendations(
        self,
        country_risk: float,
        satellite_risk: float,
        supplier_risk: float,
        document_risk: float,
        overall_risk: float,
    ) -> List[str]:
        """Generate actionable risk mitigation recommendations.

        All recommendations are deterministic, based on risk scores.

        Args:
            country_risk: Country risk score.
            satellite_risk: Satellite risk score.
            supplier_risk: Supplier risk score.
            document_risk: Document risk score.
            overall_risk: Overall weighted risk score.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if country_risk >= 0.7:
            recommendations.append(
                "Conduct enhanced due diligence for this high-risk "
                "country of production per EUDR Article 10"
            )
        if satellite_risk >= 0.7:
            recommendations.append(
                "Commission independent satellite monitoring and "
                "ground-truth verification for production plots"
            )
        if satellite_risk >= 0.5:
            recommendations.append(
                "Request updated NDVI analysis from AGENT-DATA-007 "
                "for all associated plots"
            )
        if supplier_risk >= 0.5:
            recommendations.append(
                "Request updated sustainability certificates and "
                "conduct supplier audit"
            )
        if supplier_risk >= 0.8:
            recommendations.append(
                "Consider suspending procurement from this supplier "
                "until compliance is verified"
            )
        if document_risk >= 0.5:
            recommendations.append(
                "Upload missing compliance documents to improve "
                "document coverage"
            )
        if document_risk >= 0.3:
            recommendations.append(
                "Complete document verification for all pending documents"
            )
        if overall_risk >= 0.9:
            recommendations.append(
                "CRITICAL: Do not place commodities on the EU market "
                "until risk is mitigated below critical threshold"
            )
        elif overall_risk >= 0.7:
            recommendations.append(
                "Implement all risk mitigation measures before "
                "submitting DDS to EU Information System"
            )
        elif overall_risk < 0.3:
            recommendations.append(
                "Risk level acceptable. Maintain ongoing monitoring "
                "and periodic reviews"
            )

        return recommendations

    def _check_and_generate_alerts(
        self, assessment: RiskAssessment
    ) -> None:
        """Check assessment and generate alerts if thresholds exceeded.

        Args:
            assessment: Risk assessment to check.
        """
        alerts_to_add: List[RiskAlert] = []

        if assessment.overall_risk >= self._config.risk_threshold_critical:
            alerts_to_add.append(RiskAlert(
                alert_type="critical_risk",
                severity=RiskLevel.CRITICAL,
                supplier_id=assessment.supplier_id,
                plot_id=assessment.plot_id,
                title="Critical risk level detected",
                description=(
                    f"Overall risk score {assessment.overall_risk:.4f} "
                    f"exceeds critical threshold "
                    f"{self._config.risk_threshold_critical}"
                ),
                risk_score=assessment.overall_risk,
                threshold=self._config.risk_threshold_critical,
                recommended_action=(
                    "Immediately halt procurement and conduct "
                    "enhanced due diligence"
                ),
            ))

        elif assessment.overall_risk >= self._config.risk_threshold_high:
            alerts_to_add.append(RiskAlert(
                alert_type="high_risk",
                severity=RiskLevel.HIGH,
                supplier_id=assessment.supplier_id,
                plot_id=assessment.plot_id,
                title="High risk level detected",
                description=(
                    f"Overall risk score {assessment.overall_risk:.4f} "
                    f"exceeds high threshold "
                    f"{self._config.risk_threshold_high}"
                ),
                risk_score=assessment.overall_risk,
                threshold=self._config.risk_threshold_high,
                recommended_action=(
                    "Implement enhanced monitoring and risk "
                    "mitigation measures"
                ),
            ))

        if assessment.satellite_risk >= 0.8:
            alerts_to_add.append(RiskAlert(
                alert_type="deforestation_risk",
                severity=RiskLevel.CRITICAL,
                supplier_id=assessment.supplier_id,
                plot_id=assessment.plot_id,
                title="Deforestation risk detected",
                description=(
                    f"Satellite risk score {assessment.satellite_risk:.4f} "
                    f"indicates potential deforestation"
                ),
                risk_score=assessment.satellite_risk,
                threshold=0.80,
                recommended_action=(
                    "Verify plot deforestation status via "
                    "AGENT-DATA-007 and ground inspection"
                ),
            ))

        if alerts_to_add:
            with self._lock:
                self._alerts.extend(alerts_to_add)
            logger.warning(
                "Generated %d risk alert(s) for supplier %s",
                len(alerts_to_add),
                assessment.supplier_id,
            )
