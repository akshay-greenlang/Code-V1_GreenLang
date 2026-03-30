# -*- coding: utf-8 -*-
"""
ComplianceReporter - AGENT-EUDR-004: Forest Cover Analysis (Engine 4)

Generates EUDR compliance reports with forest cover evidence for due
diligence statement (DDS) submission. Compiles analysis results from
canopy height estimation, fragmentation analysis, and biomass estimation
into structured reports across four output formats (JSON, PDF-ready,
CSV, EUDR XML).

This engine is the final output stage of the Forest Cover Analysis Agent.
It takes all upstream analysis results and packages them into regulatory-
ready evidence bundles with complete provenance chains.

Report Types (5):
    1. Plot Assessment:         Single plot forest cover assessment
    2. Batch Report:            Multi-plot commodity verification
    3. Deforestation-Free:      Evidence package for certification
    4. Historical Report:       Forest cover reconstruction narrative
    5. Dashboard Data:          Summary data for DDS dashboard

Output Formats (4):
    1. JSON:     Structured data for API consumption
    2. PDF:      Structured dict for PDF renderer
    3. CSV:      Tabular data for spreadsheet analysis
    4. EUDR_XML: EU regulatory submission format

Report Contents:
    - Forest/non-forest determination with confidence
    - Before/after comparison (cutoff vs current)
    - Supporting metrics (canopy density, height, biomass, fragmentation)
    - Provenance chain (SHA-256 hashes)
    - Regulatory references (EUDR articles, FAO definitions)
    - Data quality assessment
    - Methodology description

Zero-Hallucination Guarantees:
    - All report data is sourced from upstream engine outputs.
    - Compliance verdicts use deterministic threshold comparisons.
    - Regulatory references are hardcoded from EUDR regulation text.
    - No LLM-generated content in any compliance determination.
    - SHA-256 provenance hash on every generated report.

Performance Targets:
    - Single plot report: <100ms
    - Batch report (100 plots): <5 seconds
    - Evidence package compilation: <50ms

Regulatory References:
    - EUDR Article 2(1): Forest and deforestation definitions
    - EUDR Article 2(3): Degradation definition
    - EUDR Article 2(6): Cutoff date December 31, 2020
    - EUDR Article 4(2): DDS submission requirements
    - EUDR Article 9: Geolocation and spatial evidence
    - EUDR Article 10: Risk assessment evidence
    - EUDR Article 11: Monitoring evidence obligations
    - EUDR Article 31: Record retention (5-year audit trail)
    - FAO FRA 2020: Forest definition (5m height, 10% canopy cover)

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 (Engine 4: Compliance Reporting)
Agent ID: GL-EUDR-FCA-004
Status: Production Ready
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Data to hash (dict or other JSON-serializable object).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id(prefix: str = "rpt") -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

# ---------------------------------------------------------------------------
# Constants: Compliance Verdicts
# ---------------------------------------------------------------------------

VERDICT_DEFORESTATION_FREE = "DEFORESTATION_FREE"
VERDICT_DEFORESTED = "DEFORESTED"
VERDICT_DEGRADED = "DEGRADED"
VERDICT_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
VERDICT_MANUAL_REVIEW = "MANUAL_REVIEW_REQUIRED"

# ---------------------------------------------------------------------------
# Constants: Output Format Identifiers
# ---------------------------------------------------------------------------

FORMAT_JSON = "JSON"
FORMAT_PDF = "PDF"
FORMAT_CSV = "CSV"
FORMAT_EUDR_XML = "EUDR_XML"

VALID_FORMATS = frozenset({FORMAT_JSON, FORMAT_PDF, FORMAT_CSV, FORMAT_EUDR_XML})

# ---------------------------------------------------------------------------
# Constants: Confidence Thresholds
# ---------------------------------------------------------------------------

#: Minimum confidence for automated compliance determination.
CONFIDENCE_THRESHOLD_HIGH: float = 0.7

#: Below this confidence, verdict is INSUFFICIENT_DATA.
CONFIDENCE_THRESHOLD_LOW: float = 0.5

# ---------------------------------------------------------------------------
# Constants: EUDR Regulatory References
# ---------------------------------------------------------------------------

EUDR_REFERENCES: Dict[str, str] = {
    "article_2_1": (
        "EUDR Article 2(1): 'Deforestation' means the conversion of "
        "forest to agricultural use, whether human-induced or not."
    ),
    "article_2_3": (
        "EUDR Article 2(3): 'Forest degradation' means structural "
        "changes to forest cover taking the form of the conversion of "
        "primary forests or naturally regenerating forests into "
        "plantation forests or into other wooded land."
    ),
    "article_2_5": (
        "EUDR Article 2(5): 'Forest' means an area spanning more than "
        "0.5 hectares with trees higher than 5 metres and a canopy "
        "cover of more than 10 percent, or trees able to reach those "
        "thresholds in situ."
    ),
    "article_2_6": (
        "EUDR Article 2(6): 'Cutoff date' means 31 December 2020."
    ),
    "article_4_2": (
        "EUDR Article 4(2): Operators shall exercise due diligence to "
        "ensure the relevant commodities and products are deforestation-"
        "free, have been produced in accordance with relevant "
        "legislation and are covered by a due diligence statement."
    ),
    "article_9": (
        "EUDR Article 9: Due diligence statements shall contain "
        "geolocation coordinates of all plots of land where the "
        "relevant commodities were produced."
    ),
    "article_10": (
        "EUDR Article 10: Risk assessment shall take into account "
        "satellite monitoring data and forest cover analysis."
    ),
    "article_11": (
        "EUDR Article 11: Operators shall take adequate and "
        "proportionate risk mitigation measures."
    ),
    "article_31": (
        "EUDR Article 31: Operators and traders shall keep the due "
        "diligence statements and supporting documentation for a "
        "period of 5 years."
    ),
    "fao_forest_def": (
        "FAO FRA 2020: Forest is land spanning more than 0.5 ha with "
        "trees higher than 5 m and a canopy cover of more than 10 "
        "percent, or trees able to reach these thresholds in situ."
    ),
}

# ---------------------------------------------------------------------------
# Constants: Data Quality Levels
# ---------------------------------------------------------------------------

DATA_QUALITY_LEVELS: Dict[str, Dict[str, Any]] = {
    "HIGH": {
        "min_confidence": 0.7,
        "min_sources": 3,
        "description": "Multiple corroborating high-quality sources",
    },
    "MEDIUM": {
        "min_confidence": 0.5,
        "min_sources": 2,
        "description": "At least two sources with moderate agreement",
    },
    "LOW": {
        "min_confidence": 0.3,
        "min_sources": 1,
        "description": "Limited source data, higher uncertainty",
    },
    "INSUFFICIENT": {
        "min_confidence": 0.0,
        "min_sources": 0,
        "description": "Insufficient data for reliable assessment",
    },
}

# ---------------------------------------------------------------------------
# Constants: Required Report Fields
# ---------------------------------------------------------------------------

REQUIRED_PLOT_FIELDS: List[str] = [
    "plot_id",
    "verdict",
    "confidence",
    "forest_cover_pct",
    "provenance_hash",
]

REQUIRED_EVIDENCE_FIELDS: List[str] = [
    "plot_id",
    "verdict",
    "confidence",
    "canopy_analysis",
    "provenance_chain",
    "regulatory_references",
    "data_quality",
]

# ---------------------------------------------------------------------------
# Constants: Methodology Descriptions
# ---------------------------------------------------------------------------

METHODOLOGY_DESCRIPTIONS: Dict[str, str] = {
    "canopy_height": (
        "Canopy height estimated via multi-source fusion of GEDI L2A/L2B "
        "lidar (25m, +/-3m), ICESat-2 ATL08 (100m, +/-5m), ETH Zurich "
        "global canopy height map (10m, 2020), Meta/WRI global map (1m, "
        "2023), and Sentinel-2 GLCM texture proxy. Fused using accuracy-"
        "weighted averaging with uncertainty propagation."
    ),
    "fragmentation": (
        "Forest fragmentation assessed using six landscape ecology "
        "metrics: patch count (8-connectivity), edge density, core area "
        "(100m buffer), nearest-neighbor connectivity, shape complexity "
        "(perimeter-area ratio), and effective mesh size. Classification "
        "per Jaeger (2000) effective mesh size framework."
    ),
    "biomass": (
        "Above-ground biomass (AGB) estimated via multi-source fusion of "
        "ESA CCI Biomass maps (100m), GEDI L4A predictions (25m), "
        "Sentinel-1 SAR C-band backscatter regression (10m, saturates at "
        "~150 Mg/ha), and NDVI allometric equations. Carbon stock = AGB "
        "* 0.47 (IPCC 2006 default)."
    ),
    "change_detection": (
        "Deforestation change detection via NDVI differencing between "
        "December 31, 2020 baseline and current satellite imagery, "
        "supplemented by spectral angle mapping and time-series break "
        "detection. Multi-source fusion of Sentinel-2, Landsat, and "
        "Global Forest Watch alert data."
    ),
}

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class PlotAssessment:
    """Single plot forest cover assessment report.

    Attributes:
        report_id: Unique report identifier.
        plot_id: Plot identifier.
        report_type: Report type ("PLOT_ASSESSMENT").
        generated_at: Report generation timestamp.
        verdict: Compliance verdict.
        confidence: Overall confidence (0-1).
        forest_cover_pct: Current forest cover percentage.
        canopy_height_m: Estimated canopy height in metres.
        meets_fao_height: Whether height meets FAO 5m threshold.
        biomass_agb_mg_per_ha: AGB estimate in Mg/ha.
        carbon_stock_tc_per_ha: Carbon stock in tC/ha.
        fragmentation_level: Fragmentation classification.
        deforestation_risk_score: Risk score (0-1).
        data_quality_level: Data quality classification.
        source_count: Number of data sources used.
        sources_used: List of source identifiers.
        regulatory_references: Applicable EUDR articles.
        methodology: Methodology descriptions.
        high_risk_flag: True if verdict requires manual review.
        provenance_hash: SHA-256 provenance hash.
    """

    report_id: str = ""
    plot_id: str = ""
    report_type: str = "PLOT_ASSESSMENT"
    generated_at: str = ""
    verdict: str = ""
    confidence: float = 0.0
    forest_cover_pct: float = 0.0
    canopy_height_m: float = 0.0
    meets_fao_height: bool = False
    biomass_agb_mg_per_ha: float = 0.0
    carbon_stock_tc_per_ha: float = 0.0
    fragmentation_level: str = ""
    deforestation_risk_score: float = 0.0
    data_quality_level: str = ""
    source_count: int = 0
    sources_used: List[str] = field(default_factory=list)
    regulatory_references: Dict[str, str] = field(default_factory=dict)
    methodology: Dict[str, str] = field(default_factory=dict)
    high_risk_flag: bool = False
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "report_id": self.report_id,
            "plot_id": self.plot_id,
            "report_type": self.report_type,
            "generated_at": self.generated_at,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "forest_cover_pct": round(self.forest_cover_pct, 2),
            "canopy_height_m": round(self.canopy_height_m, 2),
            "meets_fao_height": self.meets_fao_height,
            "biomass_agb_mg_per_ha": round(self.biomass_agb_mg_per_ha, 2),
            "carbon_stock_tc_per_ha": round(self.carbon_stock_tc_per_ha, 2),
            "fragmentation_level": self.fragmentation_level,
            "deforestation_risk_score": round(self.deforestation_risk_score, 3),
            "data_quality_level": self.data_quality_level,
            "source_count": self.source_count,
            "sources_used": self.sources_used,
            "high_risk_flag": self.high_risk_flag,
        }

@dataclass
class BatchReport:
    """Batch commodity verification report across multiple plots.

    Attributes:
        report_id: Unique report identifier.
        report_type: Report type ("BATCH_REPORT").
        generated_at: Timestamp.
        commodity: Commodity being verified.
        total_plots: Total plots assessed.
        compliant_count: Plots with deforestation-free verdict.
        non_compliant_count: Plots with deforestation/degradation.
        manual_review_count: Plots requiring manual review.
        insufficient_data_count: Plots with insufficient data.
        overall_compliance_pct: Percentage of compliant plots.
        plot_assessments: Individual plot assessments.
        summary_statistics: Aggregate statistics.
        high_risk_plots: Plot IDs flagged for review.
        provenance_hash: SHA-256 provenance hash.
    """

    report_id: str = ""
    report_type: str = "BATCH_REPORT"
    generated_at: str = ""
    commodity: str = ""
    total_plots: int = 0
    compliant_count: int = 0
    non_compliant_count: int = 0
    manual_review_count: int = 0
    insufficient_data_count: int = 0
    overall_compliance_pct: float = 0.0
    plot_assessments: List[PlotAssessment] = field(default_factory=list)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    high_risk_plots: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "generated_at": self.generated_at,
            "commodity": self.commodity,
            "total_plots": self.total_plots,
            "compliant_count": self.compliant_count,
            "non_compliant_count": self.non_compliant_count,
            "manual_review_count": self.manual_review_count,
            "insufficient_data_count": self.insufficient_data_count,
            "overall_compliance_pct": round(self.overall_compliance_pct, 2),
            "high_risk_plots": self.high_risk_plots,
        }

@dataclass
class EvidencePackage:
    """Deforestation-free evidence package for certification.

    Attributes:
        package_id: Unique package identifier.
        plot_id: Plot identifier.
        package_type: Package type ("EVIDENCE_PACKAGE").
        generated_at: Timestamp.
        verdict: Compliance verdict.
        confidence: Overall confidence.
        canopy_analysis: Canopy height analysis results.
        fragmentation_analysis: Fragmentation metrics.
        biomass_analysis: Biomass estimation results.
        change_detection: Change detection results.
        before_after: Before/after comparison data.
        data_quality: Quality assessment.
        methodology: Methodology descriptions.
        regulatory_references: EUDR article references.
        provenance_chain: Chain of SHA-256 hashes.
        export_formats: Available export formats.
        provenance_hash: SHA-256 of the entire package.
    """

    package_id: str = ""
    plot_id: str = ""
    package_type: str = "EVIDENCE_PACKAGE"
    generated_at: str = ""
    verdict: str = ""
    confidence: float = 0.0
    canopy_analysis: Dict[str, Any] = field(default_factory=dict)
    fragmentation_analysis: Dict[str, Any] = field(default_factory=dict)
    biomass_analysis: Dict[str, Any] = field(default_factory=dict)
    change_detection: Dict[str, Any] = field(default_factory=dict)
    before_after: Dict[str, Any] = field(default_factory=dict)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    methodology: Dict[str, str] = field(default_factory=dict)
    regulatory_references: Dict[str, str] = field(default_factory=dict)
    provenance_chain: List[str] = field(default_factory=list)
    export_formats: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "package_id": self.package_id,
            "plot_id": self.plot_id,
            "package_type": self.package_type,
            "generated_at": self.generated_at,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "canopy_analysis": self.canopy_analysis,
            "fragmentation_analysis": self.fragmentation_analysis,
            "biomass_analysis": self.biomass_analysis,
            "change_detection": self.change_detection,
            "before_after": self.before_after,
            "data_quality": self.data_quality,
            "provenance_chain": self.provenance_chain,
        }

@dataclass
class DashboardData:
    """Summary data for DDS dashboard integration.

    Attributes:
        dashboard_id: Unique identifier.
        generated_at: Timestamp.
        total_plots: Total monitored plots.
        compliance_summary: Verdict distribution.
        risk_distribution: Risk score distribution.
        forest_cover_stats: Forest cover statistics.
        biomass_stats: Biomass statistics.
        fragmentation_stats: Fragmentation level distribution.
        data_quality_summary: Quality level distribution.
        alerts: Active high-risk alerts.
        provenance_hash: SHA-256 provenance hash.
    """

    dashboard_id: str = ""
    generated_at: str = ""
    total_plots: int = 0
    compliance_summary: Dict[str, int] = field(default_factory=dict)
    risk_distribution: Dict[str, int] = field(default_factory=dict)
    forest_cover_stats: Dict[str, float] = field(default_factory=dict)
    biomass_stats: Dict[str, float] = field(default_factory=dict)
    fragmentation_stats: Dict[str, int] = field(default_factory=dict)
    data_quality_summary: Dict[str, int] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "dashboard_id": self.dashboard_id,
            "generated_at": self.generated_at,
            "total_plots": self.total_plots,
            "compliance_summary": self.compliance_summary,
            "risk_distribution": self.risk_distribution,
            "forest_cover_stats": self.forest_cover_stats,
            "biomass_stats": self.biomass_stats,
            "fragmentation_stats": self.fragmentation_stats,
            "data_quality_summary": self.data_quality_summary,
            "alerts_count": len(self.alerts),
        }

# ---------------------------------------------------------------------------
# ComplianceReporter
# ---------------------------------------------------------------------------

class ComplianceReporter:
    """EUDR compliance report generation engine.

    Compiles forest cover analysis results into structured compliance
    reports across multiple formats. All verdicts are deterministic
    and based on configurable thresholds. Every generated report
    receives a SHA-256 provenance hash for audit trail integrity.

    Example::

        reporter = ComplianceReporter()
        assessment = reporter.generate_plot_assessment(
            plot_id="PLOT-001",
            forest_cover_pct=85.0,
            canopy_height_m=22.5,
            meets_fao_height=True,
            biomass_agb=250.0,
            fragmentation_level="INTACT",
            deforestation_risk=0.05,
            confidence=0.85,
            sources_used=["gedi", "esa_cci", "sentinel2"],
        )
        assert assessment.verdict == "DEFORESTATION_FREE"
        assert assessment.provenance_hash != ""

    Attributes:
        confidence_threshold: Minimum confidence for automated verdict.
        deforestation_risk_threshold: Risk above this triggers non-compliance.
        degradation_risk_threshold: Risk above this triggers degradation.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD_HIGH,
        deforestation_risk_threshold: float = 0.7,
        degradation_risk_threshold: float = 0.5,
    ) -> None:
        """Initialize the ComplianceReporter.

        Args:
            confidence_threshold: Minimum confidence for automated verdicts.
            deforestation_risk_threshold: Risk threshold for deforestation.
            degradation_risk_threshold: Risk threshold for degradation.

        Raises:
            ValueError: If thresholds are outside valid ranges.
        """
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], "
                f"got {confidence_threshold}"
            )
        if not (0.0 <= deforestation_risk_threshold <= 1.0):
            raise ValueError(
                f"deforestation_risk_threshold must be in [0, 1], "
                f"got {deforestation_risk_threshold}"
            )

        self.confidence_threshold = confidence_threshold
        self.deforestation_risk_threshold = deforestation_risk_threshold
        self.degradation_risk_threshold = degradation_risk_threshold

        logger.info(
            "ComplianceReporter initialized: confidence_threshold=%.2f, "
            "deforestation_risk=%.2f, degradation_risk=%.2f",
            confidence_threshold, deforestation_risk_threshold,
            degradation_risk_threshold,
        )

    # ------------------------------------------------------------------
    # Public API: Report Type 1 - Plot Assessment
    # ------------------------------------------------------------------

    def generate_plot_assessment(
        self,
        plot_id: str,
        forest_cover_pct: float = 0.0,
        canopy_height_m: float = 0.0,
        meets_fao_height: bool = False,
        biomass_agb: float = 0.0,
        carbon_stock: float = 0.0,
        fragmentation_level: str = "",
        deforestation_risk: float = 0.0,
        confidence: float = 0.0,
        sources_used: Optional[List[str]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> PlotAssessment:
        """Generate a single plot forest cover assessment report.

        Determines compliance verdict based on forest cover metrics
        and risk assessment. Annotates with regulatory references
        and methodology descriptions.

        Args:
            plot_id: Unique plot identifier.
            forest_cover_pct: Current forest cover percentage.
            canopy_height_m: Estimated canopy height in metres.
            meets_fao_height: Whether FAO 5m height threshold is met.
            biomass_agb: AGB in Mg/ha.
            carbon_stock: Carbon stock in tC/ha.
            fragmentation_level: Fragmentation classification.
            deforestation_risk: Deforestation risk score (0-1).
            confidence: Overall confidence score (0-1).
            sources_used: List of source identifiers.
            additional_data: Extra data to include.

        Returns:
            PlotAssessment with verdict and supporting evidence.
        """
        start_time = time.monotonic()

        actual_sources = sources_used or []

        verdict = self._determine_verdict(
            deforestation_risk, confidence, forest_cover_pct,
            fragmentation_level,
        )

        data_quality = self._assess_data_quality(
            confidence, len(actual_sources)
        )

        high_risk = self.auto_flag_high_risk(verdict, deforestation_risk)

        references = self._select_regulatory_references(verdict)
        methodology = dict(METHODOLOGY_DESCRIPTIONS)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        assessment = PlotAssessment(
            report_id=_generate_id("rpt-plot"),
            plot_id=plot_id,
            generated_at=str(utcnow()),
            verdict=verdict,
            confidence=round(confidence, 3),
            forest_cover_pct=round(forest_cover_pct, 2),
            canopy_height_m=round(canopy_height_m, 2),
            meets_fao_height=meets_fao_height,
            biomass_agb_mg_per_ha=round(biomass_agb, 2),
            carbon_stock_tc_per_ha=round(carbon_stock, 2),
            fragmentation_level=fragmentation_level,
            deforestation_risk_score=round(deforestation_risk, 3),
            data_quality_level=data_quality,
            source_count=len(actual_sources),
            sources_used=actual_sources,
            regulatory_references=references,
            methodology=methodology,
            high_risk_flag=high_risk,
        )
        assessment.provenance_hash = _compute_hash(assessment.to_dict())

        logger.info(
            "Plot '%s' assessment: verdict=%s, confidence=%.3f, "
            "risk=%.3f, quality=%s, high_risk=%s, %.2fms",
            plot_id, verdict, confidence, deforestation_risk,
            data_quality, high_risk, elapsed_ms,
        )

        return assessment

    # ------------------------------------------------------------------
    # Public API: Report Type 2 - Batch Report
    # ------------------------------------------------------------------

    def generate_batch_report(
        self,
        commodity: str,
        plot_data_list: List[Dict[str, Any]],
    ) -> BatchReport:
        """Generate batch commodity verification report.

        Processes multiple plots and compiles aggregate compliance
        statistics for a given commodity.

        Args:
            commodity: EUDR commodity (palm_oil, soya, cocoa, etc.).
            plot_data_list: List of dicts, each containing parameters
                for generate_plot_assessment.

        Returns:
            BatchReport with individual assessments and summary.
        """
        start_time = time.monotonic()

        assessments: List[PlotAssessment] = []
        for plot_data in plot_data_list:
            assessment = self.generate_plot_assessment(**plot_data)
            assessments.append(assessment)

        compliant = sum(
            1 for a in assessments
            if a.verdict == VERDICT_DEFORESTATION_FREE
        )
        non_compliant = sum(
            1 for a in assessments
            if a.verdict in (VERDICT_DEFORESTED, VERDICT_DEGRADED)
        )
        manual_review = sum(
            1 for a in assessments
            if a.verdict == VERDICT_MANUAL_REVIEW
        )
        insufficient = sum(
            1 for a in assessments
            if a.verdict == VERDICT_INSUFFICIENT_DATA
        )

        total = len(assessments)
        compliance_pct = (
            (compliant / total * 100.0) if total > 0 else 0.0
        )

        high_risk = [
            a.plot_id for a in assessments if a.high_risk_flag
        ]

        summary = self._compute_batch_statistics(assessments)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        report = BatchReport(
            report_id=_generate_id("rpt-batch"),
            generated_at=str(utcnow()),
            commodity=commodity,
            total_plots=total,
            compliant_count=compliant,
            non_compliant_count=non_compliant,
            manual_review_count=manual_review,
            insufficient_data_count=insufficient,
            overall_compliance_pct=round(compliance_pct, 2),
            plot_assessments=assessments,
            summary_statistics=summary,
            high_risk_plots=high_risk,
        )
        report.provenance_hash = _compute_hash(report.to_dict())

        logger.info(
            "Batch report for '%s': %d plots, %.1f%% compliant, "
            "%d high-risk, %.2fms",
            commodity, total, compliance_pct, len(high_risk), elapsed_ms,
        )

        return report

    # ------------------------------------------------------------------
    # Public API: Report Type 3 - Deforestation-Free Evidence
    # ------------------------------------------------------------------

    def generate_deforestation_free_evidence(
        self,
        plot_id: str,
        canopy_analysis: Optional[Dict[str, Any]] = None,
        fragmentation_analysis: Optional[Dict[str, Any]] = None,
        biomass_analysis: Optional[Dict[str, Any]] = None,
        change_detection: Optional[Dict[str, Any]] = None,
        baseline_data: Optional[Dict[str, Any]] = None,
        current_data: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        source_hashes: Optional[List[str]] = None,
    ) -> EvidencePackage:
        """Generate deforestation-free evidence package.

        Compiles all analysis results into a comprehensive evidence
        bundle suitable for regulatory submission. Includes before/after
        comparison, full provenance chain, and applicable regulations.

        Args:
            plot_id: Plot identifier.
            canopy_analysis: Canopy height analysis results dict.
            fragmentation_analysis: Fragmentation metrics dict.
            biomass_analysis: Biomass estimation results dict.
            change_detection: Change detection results dict.
            baseline_data: Forest metrics at cutoff date.
            current_data: Current forest metrics.
            confidence: Overall confidence.
            source_hashes: SHA-256 hashes from upstream engines.

        Returns:
            EvidencePackage ready for export.
        """
        start_time = time.monotonic()

        evidence = self.compile_evidence(
            canopy_analysis=canopy_analysis or {},
            fragmentation_analysis=fragmentation_analysis or {},
            biomass_analysis=biomass_analysis or {},
            change_detection=change_detection or {},
        )

        before_after = self._build_before_after(
            baseline_data or {}, current_data or {}
        )

        verdict = self._determine_evidence_verdict(
            evidence, confidence
        )

        data_quality = self._assess_evidence_quality(evidence, confidence)

        provenance_chain = list(source_hashes or [])
        package_content_hash = _compute_hash({
            "evidence": evidence,
            "before_after": before_after,
            "verdict": verdict,
        })
        provenance_chain.append(package_content_hash)

        references = self.add_regulatory_context(verdict)
        methodology = dict(METHODOLOGY_DESCRIPTIONS)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        package = EvidencePackage(
            package_id=_generate_id("evd"),
            plot_id=plot_id,
            generated_at=str(utcnow()),
            verdict=verdict,
            confidence=round(confidence, 3),
            canopy_analysis=canopy_analysis or {},
            fragmentation_analysis=fragmentation_analysis or {},
            biomass_analysis=biomass_analysis or {},
            change_detection=change_detection or {},
            before_after=before_after,
            data_quality=data_quality,
            methodology=methodology,
            regulatory_references=references,
            provenance_chain=provenance_chain,
            export_formats=list(VALID_FORMATS),
        )
        package.provenance_hash = _compute_hash(package.to_dict())

        logger.info(
            "Evidence package for plot '%s': verdict=%s, "
            "confidence=%.3f, chain_length=%d, %.2fms",
            plot_id, verdict, confidence,
            len(provenance_chain), elapsed_ms,
        )

        return package

    # ------------------------------------------------------------------
    # Public API: Report Type 4 - Historical Report
    # ------------------------------------------------------------------

    def generate_historical_report(
        self,
        plot_id: str,
        time_series: List[Dict[str, Any]],
        cutoff_date: str = "2020-12-31",
    ) -> Dict[str, Any]:
        """Generate historical forest cover reconstruction report.

        Builds a timeline of forest cover metrics from the cutoff date
        to the present, highlighting any significant changes.

        Args:
            plot_id: Plot identifier.
            time_series: List of dicts with date and metrics for each
                observation period. Each dict should contain:
                - date (str): Observation date.
                - forest_cover_pct (float): Forest cover.
                - canopy_height_m (float): Canopy height.
                - biomass_agb (float): AGB estimate.
                - fragmentation_level (str): Classification.
            cutoff_date: EUDR cutoff date string.

        Returns:
            Historical report dictionary with timeline and changes.
        """
        start_time = time.monotonic()

        sorted_series = sorted(
            time_series, key=lambda x: x.get("date", "")
        )

        changes: List[Dict[str, Any]] = []
        for i in range(1, len(sorted_series)):
            prev = sorted_series[i - 1]
            curr = sorted_series[i]
            change = self._detect_period_change(prev, curr)
            if change is not None:
                changes.append(change)

        cutoff_metrics = None
        current_metrics = None
        for entry in sorted_series:
            entry_date = entry.get("date", "")
            if entry_date <= cutoff_date:
                cutoff_metrics = entry
            current_metrics = entry

        overall_change = {}
        if cutoff_metrics and current_metrics:
            overall_change = {
                "baseline_date": cutoff_metrics.get("date", cutoff_date),
                "current_date": current_metrics.get("date", ""),
                "forest_cover_change_pct": round(
                    current_metrics.get("forest_cover_pct", 0)
                    - cutoff_metrics.get("forest_cover_pct", 0), 2
                ),
                "biomass_change_mg_ha": round(
                    current_metrics.get("biomass_agb", 0)
                    - cutoff_metrics.get("biomass_agb", 0), 2
                ),
            }

        elapsed_ms = (time.monotonic() - start_time) * 1000

        report = {
            "report_id": _generate_id("rpt-hist"),
            "report_type": "HISTORICAL_REPORT",
            "plot_id": plot_id,
            "generated_at": str(utcnow()),
            "cutoff_date": cutoff_date,
            "observation_count": len(sorted_series),
            "time_series": sorted_series,
            "significant_changes": changes,
            "overall_change": overall_change,
            "cutoff_metrics": cutoff_metrics,
            "current_metrics": current_metrics,
            "processing_time_ms": round(elapsed_ms, 2),
        }
        report["provenance_hash"] = _compute_hash(report)

        logger.info(
            "Historical report for plot '%s': %d observations, "
            "%d changes, %.2fms",
            plot_id, len(sorted_series), len(changes), elapsed_ms,
        )

        return report

    # ------------------------------------------------------------------
    # Public API: Report Type 5 - Dashboard Data
    # ------------------------------------------------------------------

    def generate_dashboard_data(
        self,
        assessments: List[PlotAssessment],
    ) -> DashboardData:
        """Generate summary data for DDS dashboard integration.

        Aggregates all plot assessments into dashboard-ready statistics.

        Args:
            assessments: List of PlotAssessment results.

        Returns:
            DashboardData with aggregated statistics.
        """
        start_time = time.monotonic()

        compliance_summary: Dict[str, int] = {
            VERDICT_DEFORESTATION_FREE: 0,
            VERDICT_DEFORESTED: 0,
            VERDICT_DEGRADED: 0,
            VERDICT_MANUAL_REVIEW: 0,
            VERDICT_INSUFFICIENT_DATA: 0,
        }
        for a in assessments:
            if a.verdict in compliance_summary:
                compliance_summary[a.verdict] += 1

        risk_dist: Dict[str, int] = {
            "low": 0, "medium": 0, "high": 0, "critical": 0,
        }
        for a in assessments:
            risk = a.deforestation_risk_score
            if risk < 0.2:
                risk_dist["low"] += 1
            elif risk < 0.5:
                risk_dist["medium"] += 1
            elif risk < 0.8:
                risk_dist["high"] += 1
            else:
                risk_dist["critical"] += 1

        forest_stats = self._compute_stat_summary(
            [a.forest_cover_pct for a in assessments]
        )
        biomass_stats = self._compute_stat_summary(
            [a.biomass_agb_mg_per_ha for a in assessments]
        )

        frag_dist: Dict[str, int] = {}
        for a in assessments:
            level = a.fragmentation_level or "UNKNOWN"
            frag_dist[level] = frag_dist.get(level, 0) + 1

        quality_dist: Dict[str, int] = {}
        for a in assessments:
            level = a.data_quality_level or "UNKNOWN"
            quality_dist[level] = quality_dist.get(level, 0) + 1

        alerts = [
            {
                "plot_id": a.plot_id,
                "verdict": a.verdict,
                "risk_score": a.deforestation_risk_score,
            }
            for a in assessments if a.high_risk_flag
        ]

        elapsed_ms = (time.monotonic() - start_time) * 1000

        dashboard = DashboardData(
            dashboard_id=_generate_id("dash"),
            generated_at=str(utcnow()),
            total_plots=len(assessments),
            compliance_summary=compliance_summary,
            risk_distribution=risk_dist,
            forest_cover_stats=forest_stats,
            biomass_stats=biomass_stats,
            fragmentation_stats=frag_dist,
            data_quality_summary=quality_dist,
            alerts=alerts,
        )
        dashboard.provenance_hash = _compute_hash(dashboard.to_dict())

        logger.info(
            "Dashboard data: %d plots, %d alerts, %.2fms",
            len(assessments), len(alerts), elapsed_ms,
        )

        return dashboard

    # ------------------------------------------------------------------
    # Public API: Output Formatting
    # ------------------------------------------------------------------

    def format_as_json(self, report: Any) -> str:
        """Format a report as JSON string.

        Args:
            report: Any report dataclass with a to_dict method.

        Returns:
            Pretty-printed JSON string.
        """
        if hasattr(report, "to_dict"):
            data = report.to_dict()
        elif isinstance(report, dict):
            data = report
        else:
            data = {"error": "Unsupported report type"}

        return json.dumps(data, indent=2, sort_keys=False, default=str)

    def format_as_csv(
        self,
        assessments: List[PlotAssessment],
    ) -> str:
        """Format plot assessments as CSV string.

        Args:
            assessments: List of PlotAssessment objects.

        Returns:
            CSV string with header row and one row per assessment.
        """
        output = io.StringIO()
        fieldnames = [
            "report_id", "plot_id", "verdict", "confidence",
            "forest_cover_pct", "canopy_height_m", "meets_fao_height",
            "biomass_agb_mg_per_ha", "carbon_stock_tc_per_ha",
            "fragmentation_level", "deforestation_risk_score",
            "data_quality_level", "source_count", "high_risk_flag",
            "generated_at", "provenance_hash",
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for a in assessments:
            writer.writerow({
                "report_id": a.report_id,
                "plot_id": a.plot_id,
                "verdict": a.verdict,
                "confidence": round(a.confidence, 3),
                "forest_cover_pct": round(a.forest_cover_pct, 2),
                "canopy_height_m": round(a.canopy_height_m, 2),
                "meets_fao_height": a.meets_fao_height,
                "biomass_agb_mg_per_ha": round(a.biomass_agb_mg_per_ha, 2),
                "carbon_stock_tc_per_ha": round(a.carbon_stock_tc_per_ha, 2),
                "fragmentation_level": a.fragmentation_level,
                "deforestation_risk_score": round(
                    a.deforestation_risk_score, 3
                ),
                "data_quality_level": a.data_quality_level,
                "source_count": a.source_count,
                "high_risk_flag": a.high_risk_flag,
                "generated_at": a.generated_at,
                "provenance_hash": a.provenance_hash,
            })

        return output.getvalue()

    def format_as_pdf_structure(
        self,
        assessment: PlotAssessment,
    ) -> Dict[str, Any]:
        """Generate PDF-ready structured data for a report.

        Returns a dictionary that can be consumed by a PDF rendering
        engine. This engine does not generate PDF binary data; it
        provides the structured content for a downstream renderer.

        Args:
            assessment: PlotAssessment to format.

        Returns:
            Dictionary with sections for PDF rendering.
        """
        return {
            "title": f"EUDR Forest Cover Assessment - {assessment.plot_id}",
            "subtitle": f"Report ID: {assessment.report_id}",
            "generated": assessment.generated_at,
            "sections": [
                {
                    "heading": "Executive Summary",
                    "content": {
                        "verdict": assessment.verdict,
                        "confidence": f"{assessment.confidence:.1%}",
                        "data_quality": assessment.data_quality_level,
                        "high_risk": assessment.high_risk_flag,
                    },
                },
                {
                    "heading": "Forest Cover Metrics",
                    "content": {
                        "forest_cover": f"{assessment.forest_cover_pct:.1f}%",
                        "canopy_height": f"{assessment.canopy_height_m:.1f} m",
                        "meets_fao_height": assessment.meets_fao_height,
                        "biomass_agb": (
                            f"{assessment.biomass_agb_mg_per_ha:.1f} Mg/ha"
                        ),
                        "carbon_stock": (
                            f"{assessment.carbon_stock_tc_per_ha:.1f} tC/ha"
                        ),
                    },
                },
                {
                    "heading": "Fragmentation Analysis",
                    "content": {
                        "level": assessment.fragmentation_level,
                        "risk_score": (
                            f"{assessment.deforestation_risk_score:.3f}"
                        ),
                    },
                },
                {
                    "heading": "Data Sources",
                    "content": {
                        "source_count": assessment.source_count,
                        "sources": assessment.sources_used,
                    },
                },
                {
                    "heading": "Methodology",
                    "content": assessment.methodology,
                },
                {
                    "heading": "Regulatory References",
                    "content": assessment.regulatory_references,
                },
            ],
            "footer": {
                "provenance_hash": assessment.provenance_hash,
                "disclaimer": (
                    "This report is generated by the GreenLang EUDR "
                    "Forest Cover Analysis Agent (GL-EUDR-FCA-004) "
                    "using deterministic calculations. All data sources "
                    "and provenance are documented for audit purposes."
                ),
            },
        }

    def format_as_eudr_xml(
        self,
        assessment: PlotAssessment,
    ) -> str:
        """Generate simplified EUDR XML submission format.

        Produces an XML string following a simplified version of the
        EU DDS submission schema. This is a self-contained XML template
        without external file dependencies.

        Args:
            assessment: PlotAssessment to format.

        Returns:
            XML string for regulatory submission.
        """
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<EUDRDueDiligenceStatement '
            'xmlns="urn:eu:eudr:dds:1.0" '
            'version="1.0">',
            '  <Header>',
            f'    <ReportID>{_escape_xml(assessment.report_id)}</ReportID>',
            f'    <GeneratedAt>{_escape_xml(assessment.generated_at)}'
            f'</GeneratedAt>',
            f'    <AgentID>GL-EUDR-FCA-004</AgentID>',
            f'    <ModuleVersion>{_MODULE_VERSION}</ModuleVersion>',
            '  </Header>',
            '  <PlotAssessment>',
            f'    <PlotID>{_escape_xml(assessment.plot_id)}</PlotID>',
            f'    <Verdict>{_escape_xml(assessment.verdict)}</Verdict>',
            f'    <Confidence>{assessment.confidence:.3f}</Confidence>',
            f'    <HighRisk>{str(assessment.high_risk_flag).lower()}'
            f'</HighRisk>',
            '    <ForestCoverMetrics>',
            f'      <ForestCoverPct>{assessment.forest_cover_pct:.2f}'
            f'</ForestCoverPct>',
            f'      <CanopyHeightM>{assessment.canopy_height_m:.2f}'
            f'</CanopyHeightM>',
            f'      <MeetsFAOHeight>{str(assessment.meets_fao_height).lower()}'
            f'</MeetsFAOHeight>',
            f'      <BiomassAGB>{assessment.biomass_agb_mg_per_ha:.2f}'
            f'</BiomassAGB>',
            f'      <CarbonStock>{assessment.carbon_stock_tc_per_ha:.2f}'
            f'</CarbonStock>',
            '    </ForestCoverMetrics>',
            '    <FragmentationAnalysis>',
            f'      <Level>{_escape_xml(assessment.fragmentation_level)}'
            f'</Level>',
            f'      <RiskScore>{assessment.deforestation_risk_score:.3f}'
            f'</RiskScore>',
            '    </FragmentationAnalysis>',
            '    <DataQuality>',
            f'      <Level>{_escape_xml(assessment.data_quality_level)}'
            f'</Level>',
            f'      <SourceCount>{assessment.source_count}</SourceCount>',
            '    </DataQuality>',
            '  </PlotAssessment>',
            '  <Provenance>',
            f'    <SHA256>{_escape_xml(assessment.provenance_hash)}</SHA256>',
            '  </Provenance>',
            '</EUDRDueDiligenceStatement>',
        ]

        return "\n".join(xml_lines)

    # ------------------------------------------------------------------
    # Public API: Evidence Compilation
    # ------------------------------------------------------------------

    def compile_evidence(
        self,
        canopy_analysis: Dict[str, Any],
        fragmentation_analysis: Dict[str, Any],
        biomass_analysis: Dict[str, Any],
        change_detection: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Gather all analysis results into a single evidence bundle.

        Args:
            canopy_analysis: Canopy height results.
            fragmentation_analysis: Fragmentation metrics.
            biomass_analysis: Biomass estimation results.
            change_detection: Change detection results.

        Returns:
            Compiled evidence dictionary.
        """
        evidence = {
            "canopy_analysis": canopy_analysis,
            "fragmentation_analysis": fragmentation_analysis,
            "biomass_analysis": biomass_analysis,
            "change_detection": change_detection,
            "compiled_at": str(utcnow()),
        }
        evidence["evidence_hash"] = _compute_hash(evidence)
        return evidence

    # ------------------------------------------------------------------
    # Public API: Regulatory Context
    # ------------------------------------------------------------------

    def add_regulatory_context(
        self,
        verdict: str,
    ) -> Dict[str, str]:
        """Annotate report with applicable EUDR articles.

        Selects regulatory references relevant to the compliance
        verdict.

        Args:
            verdict: Compliance verdict string.

        Returns:
            Dictionary of applicable regulatory references.
        """
        references = {
            "article_2_5": EUDR_REFERENCES["article_2_5"],
            "article_2_6": EUDR_REFERENCES["article_2_6"],
            "article_9": EUDR_REFERENCES["article_9"],
            "article_31": EUDR_REFERENCES["article_31"],
            "fao_forest_def": EUDR_REFERENCES["fao_forest_def"],
        }

        if verdict == VERDICT_DEFORESTED:
            references["article_2_1"] = EUDR_REFERENCES["article_2_1"]
            references["article_4_2"] = EUDR_REFERENCES["article_4_2"]
            references["article_10"] = EUDR_REFERENCES["article_10"]

        if verdict == VERDICT_DEGRADED:
            references["article_2_3"] = EUDR_REFERENCES["article_2_3"]
            references["article_4_2"] = EUDR_REFERENCES["article_4_2"]
            references["article_11"] = EUDR_REFERENCES["article_11"]

        if verdict == VERDICT_DEFORESTATION_FREE:
            references["article_4_2"] = EUDR_REFERENCES["article_4_2"]

        if verdict == VERDICT_MANUAL_REVIEW:
            references["article_10"] = EUDR_REFERENCES["article_10"]
            references["article_11"] = EUDR_REFERENCES["article_11"]

        return references

    # ------------------------------------------------------------------
    # Public API: High-Risk Flagging
    # ------------------------------------------------------------------

    def auto_flag_high_risk(
        self,
        verdict: str,
        risk_score: float = 0.0,
    ) -> bool:
        """Flag reports requiring manual review.

        A report is flagged as high-risk if:
        1. Verdict is DEFORESTED or DEGRADED, OR
        2. Risk score exceeds the degradation threshold.

        Args:
            verdict: Compliance verdict.
            risk_score: Deforestation risk score.

        Returns:
            True if report should be flagged for manual review.
        """
        if verdict in (VERDICT_DEFORESTED, VERDICT_DEGRADED):
            return True
        if risk_score >= self.degradation_risk_threshold:
            return True
        return False

    # ------------------------------------------------------------------
    # Public API: Report Completeness Validation
    # ------------------------------------------------------------------

    def validate_report_completeness(
        self,
        report: Dict[str, Any],
        required_fields: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """Check all required fields are present before export.

        Args:
            report: Report dictionary to validate.
            required_fields: List of required field names. Uses
                REQUIRED_PLOT_FIELDS if None.

        Returns:
            Tuple of (is_complete: bool, missing_fields: list).
        """
        fields = required_fields or REQUIRED_PLOT_FIELDS
        missing: List[str] = []

        for field_name in fields:
            if field_name not in report:
                missing.append(field_name)
            elif report[field_name] is None:
                missing.append(field_name)
            elif isinstance(report[field_name], str) and not report[field_name]:
                missing.append(field_name)

        is_complete = len(missing) == 0

        if not is_complete:
            logger.warning(
                "Report validation failed: missing fields %s", missing
            )

        return is_complete, missing

    # ------------------------------------------------------------------
    # Internal: Verdict Determination
    # ------------------------------------------------------------------

    def _determine_verdict(
        self,
        risk_score: float,
        confidence: float,
        forest_cover_pct: float,
        fragmentation_level: str,
    ) -> str:
        """Determine compliance verdict from metrics.

        Decision rules (evaluated in order):
        1. If confidence < LOW threshold: INSUFFICIENT_DATA
        2. If risk >= deforestation threshold AND confidence >= HIGH:
           DEFORESTED
        3. If risk >= degradation threshold AND confidence >= HIGH:
           DEGRADED
        4. If risk < degradation threshold AND confidence >= HIGH:
           DEFORESTATION_FREE
        5. Otherwise: MANUAL_REVIEW_REQUIRED

        Args:
            risk_score: Deforestation risk score (0-1).
            confidence: Overall confidence (0-1).
            forest_cover_pct: Current forest cover percentage.
            fragmentation_level: Fragmentation classification.

        Returns:
            Verdict string.
        """
        if confidence < CONFIDENCE_THRESHOLD_LOW:
            return VERDICT_INSUFFICIENT_DATA

        if (
            risk_score >= self.deforestation_risk_threshold
            and confidence >= self.confidence_threshold
        ):
            return VERDICT_DEFORESTED

        if (
            risk_score >= self.degradation_risk_threshold
            and confidence >= self.confidence_threshold
        ):
            return VERDICT_DEGRADED

        if (
            risk_score < self.degradation_risk_threshold
            and confidence >= self.confidence_threshold
        ):
            return VERDICT_DEFORESTATION_FREE

        return VERDICT_MANUAL_REVIEW

    def _determine_evidence_verdict(
        self,
        evidence: Dict[str, Any],
        confidence: float,
    ) -> str:
        """Determine verdict from compiled evidence.

        Extracts risk indicators from evidence sub-analyses and
        delegates to _determine_verdict.

        Args:
            evidence: Compiled evidence dict.
            confidence: Overall confidence.

        Returns:
            Verdict string.
        """
        change = evidence.get("change_detection", {})
        risk = float(change.get("deforestation_risk_score", 0.0))

        frag = evidence.get("fragmentation_analysis", {})
        frag_level = frag.get("fragmentation_level", "")

        canopy = evidence.get("canopy_analysis", {})
        cover_pct = float(canopy.get("forest_cover_pct", 0.0))

        return self._determine_verdict(risk, confidence, cover_pct, frag_level)

    # ------------------------------------------------------------------
    # Internal: Data Quality Assessment
    # ------------------------------------------------------------------

    def _assess_data_quality(
        self,
        confidence: float,
        source_count: int,
    ) -> str:
        """Assess data quality level from confidence and source count.

        Args:
            confidence: Confidence score.
            source_count: Number of data sources.

        Returns:
            Quality level string.
        """
        for level in ["HIGH", "MEDIUM", "LOW"]:
            spec = DATA_QUALITY_LEVELS[level]
            if (
                confidence >= spec["min_confidence"]
                and source_count >= spec["min_sources"]
            ):
                return level

        return "INSUFFICIENT"

    def _assess_evidence_quality(
        self,
        evidence: Dict[str, Any],
        confidence: float,
    ) -> Dict[str, Any]:
        """Assess evidence quality with detailed breakdown.

        Args:
            evidence: Compiled evidence dict.
            confidence: Overall confidence.

        Returns:
            Quality assessment dictionary.
        """
        analyses_present = sum(
            1 for key in ["canopy_analysis", "fragmentation_analysis",
                          "biomass_analysis", "change_detection"]
            if evidence.get(key)
        )

        level = self._assess_data_quality(confidence, analyses_present)

        return {
            "quality_level": level,
            "confidence": round(confidence, 3),
            "analyses_present": analyses_present,
            "analyses_total": 4,
            "completeness_pct": round(analyses_present / 4 * 100, 1),
            "description": DATA_QUALITY_LEVELS.get(
                level, DATA_QUALITY_LEVELS["INSUFFICIENT"]
            )["description"],
        }

    # ------------------------------------------------------------------
    # Internal: Regulatory Reference Selection
    # ------------------------------------------------------------------

    def _select_regulatory_references(
        self,
        verdict: str,
    ) -> Dict[str, str]:
        """Select regulatory references for a verdict."""
        return self.add_regulatory_context(verdict)

    # ------------------------------------------------------------------
    # Internal: Before/After Comparison
    # ------------------------------------------------------------------

    def _build_before_after(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build before/after comparison data.

        Args:
            baseline: Metrics at cutoff date.
            current: Current metrics.

        Returns:
            Comparison dictionary with deltas.
        """
        comparison: Dict[str, Any] = {
            "cutoff_date": "2020-12-31",
            "baseline": baseline,
            "current": current,
            "changes": {},
        }

        numeric_keys = [
            "forest_cover_pct", "canopy_height_m", "biomass_agb",
            "carbon_stock", "edge_density",
        ]
        for key in numeric_keys:
            base_val = baseline.get(key)
            curr_val = current.get(key)
            if base_val is not None and curr_val is not None:
                delta = float(curr_val) - float(base_val)
                pct = (
                    (delta / float(base_val) * 100.0)
                    if float(base_val) != 0 else 0.0
                )
                comparison["changes"][key] = {
                    "baseline": round(float(base_val), 2),
                    "current": round(float(curr_val), 2),
                    "absolute_change": round(delta, 2),
                    "percentage_change": round(pct, 2),
                }

        return comparison

    # ------------------------------------------------------------------
    # Internal: Period Change Detection
    # ------------------------------------------------------------------

    def _detect_period_change(
        self,
        prev: Dict[str, Any],
        curr: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Detect significant change between two consecutive periods.

        A change is significant if forest cover drops by more than
        5 percentage points or biomass drops by more than 20%.

        Args:
            prev: Previous period metrics.
            curr: Current period metrics.

        Returns:
            Change dict if significant, None otherwise.
        """
        prev_cover = prev.get("forest_cover_pct", 0)
        curr_cover = curr.get("forest_cover_pct", 0)
        cover_delta = curr_cover - prev_cover

        prev_agb = prev.get("biomass_agb", 0)
        curr_agb = curr.get("biomass_agb", 0)
        agb_delta = curr_agb - prev_agb
        agb_pct = (
            (agb_delta / prev_agb * 100.0)
            if prev_agb > 0 else 0.0
        )

        is_significant = (
            cover_delta < -5.0
            or agb_pct < -20.0
        )

        if not is_significant:
            return None

        return {
            "from_date": prev.get("date", ""),
            "to_date": curr.get("date", ""),
            "forest_cover_change_pct": round(cover_delta, 2),
            "biomass_change_pct": round(agb_pct, 2),
            "type": "DEFORESTATION" if cover_delta < -5.0 else "DEGRADATION",
        }

    # ------------------------------------------------------------------
    # Internal: Batch Statistics
    # ------------------------------------------------------------------

    def _compute_batch_statistics(
        self,
        assessments: List[PlotAssessment],
    ) -> Dict[str, Any]:
        """Compute aggregate statistics for batch assessments.

        Args:
            assessments: List of PlotAssessment.

        Returns:
            Statistics dictionary.
        """
        if not assessments:
            return {}

        n = len(assessments)

        return {
            "mean_confidence": round(
                sum(a.confidence for a in assessments) / n, 3
            ),
            "mean_forest_cover_pct": round(
                sum(a.forest_cover_pct for a in assessments) / n, 2
            ),
            "mean_risk_score": round(
                sum(a.deforestation_risk_score for a in assessments) / n, 3
            ),
            "mean_biomass_agb": round(
                sum(a.biomass_agb_mg_per_ha for a in assessments) / n, 2
            ),
            "min_confidence": round(
                min(a.confidence for a in assessments), 3
            ),
            "max_risk_score": round(
                max(a.deforestation_risk_score for a in assessments), 3
            ),
            "sources_used_distribution": self._count_sources(assessments),
        }

    def _count_sources(
        self,
        assessments: List[PlotAssessment],
    ) -> Dict[str, int]:
        """Count source usage across assessments."""
        counts: Dict[str, int] = {}
        for a in assessments:
            for source in a.sources_used:
                counts[source] = counts.get(source, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Internal: Statistics Helper
    # ------------------------------------------------------------------

    def _compute_stat_summary(
        self,
        values: List[float],
    ) -> Dict[str, float]:
        """Compute min/max/mean/median for a list of values."""
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean_val = sum(sorted_vals) / n

        if n % 2 == 1:
            median_val = sorted_vals[n // 2]
        else:
            median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

        return {
            "min": round(min(sorted_vals), 2),
            "max": round(max(sorted_vals), 2),
            "mean": round(mean_val, 2),
            "median": round(median_val, 2),
        }

# ---------------------------------------------------------------------------
# XML Escaping Helper
# ---------------------------------------------------------------------------

def _escape_xml(text: str) -> str:
    """Escape special XML characters in a string.

    Args:
        text: Raw text to escape.

    Returns:
        XML-safe string with &, <, >, ", ' escaped.
    """
    if not text:
        return ""
    result = text.replace("&", "&amp;")
    result = result.replace("<", "&lt;")
    result = result.replace(">", "&gt;")
    result = result.replace('"', "&quot;")
    result = result.replace("'", "&apos;")
    return result

# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "ComplianceReporter",
    "PlotAssessment",
    "BatchReport",
    "EvidencePackage",
    "DashboardData",
    "VERDICT_DEFORESTATION_FREE",
    "VERDICT_DEFORESTED",
    "VERDICT_DEGRADED",
    "VERDICT_INSUFFICIENT_DATA",
    "VERDICT_MANUAL_REVIEW",
    "FORMAT_JSON",
    "FORMAT_PDF",
    "FORMAT_CSV",
    "FORMAT_EUDR_XML",
    "VALID_FORMATS",
    "EUDR_REFERENCES",
    "DATA_QUALITY_LEVELS",
    "METHODOLOGY_DESCRIPTIONS",
    "CONFIDENCE_THRESHOLD_HIGH",
    "CONFIDENCE_THRESHOLD_LOW",
]
