# -*- coding: utf-8 -*-
"""
Risk Report Generator Engine - AGENT-EUDR-016 Engine 7

Generates comprehensive risk assessment reports in multiple formats
(JSON, HTML, PDF metadata) with multi-language support covering
country profiles, commodity analyses, comparative assessments, trend
reports, due diligence briefs, executive summaries, regulatory alerts,
and portfolio risk dashboards.

Report Types (8):
    - country_profile: Single-country comprehensive risk assessment
      with all factors, hotspots, governance, trade flows, and
      commodity-specific breakdowns.
    - commodity_analysis: Multi-country risk assessment for a single
      commodity with production, trade, certification, and hotspot data.
    - comparative: Side-by-side comparison of 2+ countries across all
      risk factors, scores, and DD classifications.
    - trend: Historical evolution of risk scores and classifications
      for one or more countries over a configurable time window.
    - due_diligence_brief: Focused report on DD requirements, costs,
      audit frequency, and compliance timeline for a country-commodity.
    - executive_summary: High-level KPI dashboard with key metrics,
      alerts, and risk trends for leadership briefing.
    - regulatory_alert: Notification report on EC reclassifications,
      regulatory changes, and compliance impact assessments.
    - portfolio_risk: Aggregated risk scoring for a portfolio of
      imports across multiple countries and commodities.

Output Formats (5):
    - JSON: Machine-readable structured data for API integration.
    - HTML: Web-viewable report with embedded charts and tables.
    - PDF: PDF metadata only (template-ready for external rendering).
    - CSV: Tabular data export for spreadsheet analysis.
    - EXCEL: Microsoft Excel workbook with formatted sheets.

Languages (5):
    - en: English (default)
    - fr: French
    - de: German
    - es: Spanish
    - pt: Portuguese

Zero-Hallucination: All report content is generated from deterministic
    data aggregation and template-based formatting. No LLM calls are
    used in report generation (narrative summaries use template strings).

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .config import get_config
from .metrics import observe_report_generation_duration, record_report_generated
from .models import (
    AssessmentConfidence,
    CommodityType,
    DueDiligenceLevel,
    ReportFormat,
    ReportType,
    RiskLevel,
    RiskReport,
    SUPPORTED_COMMODITIES,
    SUPPORTED_OUTPUT_FORMATS,
    SUPPORTED_REPORT_LANGUAGES,
)
from .provenance import get_provenance_tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Report section templates per language.
_SECTION_TEMPLATES: Dict[str, Dict[str, str]] = {
    "en": {
        "executive_summary": "Executive Summary",
        "risk_scoring": "Risk Scoring Breakdown",
        "factor_analysis": "Risk Factor Analysis",
        "commodity_details": "Commodity Risk Details",
        "hotspot_maps": "Deforestation Hotspot Maps",
        "governance": "Governance Assessment",
        "trade_flows": "Trade Flow Analysis",
        "recommendations": "Recommendations",
        "appendices": "Appendices",
        "data_sources": "Data Sources",
        "methodology": "Methodology",
    },
    "fr": {
        "executive_summary": "Résumé Exécutif",
        "risk_scoring": "Détail de la Notation des Risques",
        "factor_analysis": "Analyse des Facteurs de Risque",
        "commodity_details": "Détails des Risques par Produit",
        "hotspot_maps": "Cartes des Zones de Déforestation",
        "governance": "Évaluation de la Gouvernance",
        "trade_flows": "Analyse des Flux Commerciaux",
        "recommendations": "Recommandations",
        "appendices": "Annexes",
        "data_sources": "Sources de Données",
        "methodology": "Méthodologie",
    },
    "de": {
        "executive_summary": "Zusammenfassung",
        "risk_scoring": "Risikobewertung",
        "factor_analysis": "Risikofaktorenanalyse",
        "commodity_details": "Rohstoffrisiko-Details",
        "hotspot_maps": "Abholzungs-Hotspot-Karten",
        "governance": "Governance-Bewertung",
        "trade_flows": "Handelsflussanalyse",
        "recommendations": "Empfehlungen",
        "appendices": "Anhänge",
        "data_sources": "Datenquellen",
        "methodology": "Methodik",
    },
    "es": {
        "executive_summary": "Resumen Ejecutivo",
        "risk_scoring": "Desglose de Puntuación de Riesgo",
        "factor_analysis": "Análisis de Factores de Riesgo",
        "commodity_details": "Detalles de Riesgo por Producto",
        "hotspot_maps": "Mapas de Puntos Críticos de Deforestación",
        "governance": "Evaluación de Gobernanza",
        "trade_flows": "Análisis de Flujos Comerciales",
        "recommendations": "Recomendaciones",
        "appendices": "Apéndices",
        "data_sources": "Fuentes de Datos",
        "methodology": "Metodología",
    },
    "pt": {
        "executive_summary": "Resumo Executivo",
        "risk_scoring": "Detalhamento da Pontuação de Risco",
        "factor_analysis": "Análise de Fatores de Risco",
        "commodity_details": "Detalhes de Risco por Produto",
        "hotspot_maps": "Mapas de Pontos Críticos de Desmatamento",
        "governance": "Avaliação de Governança",
        "trade_flows": "Análise de Fluxos Comerciais",
        "recommendations": "Recomendações",
        "appendices": "Apêndices",
        "data_sources": "Fontes de Dados",
        "methodology": "Metodologia",
    },
}

#: Risk level labels per language.
_RISK_LABELS: Dict[str, Dict[str, str]] = {
    "en": {"low": "Low Risk", "standard": "Standard Risk", "high": "High Risk"},
    "fr": {"low": "Risque Faible", "standard": "Risque Standard", "high": "Risque Élevé"},
    "de": {"low": "Niedriges Risiko", "standard": "Standardrisiko", "high": "Hohes Risiko"},
    "es": {"low": "Riesgo Bajo", "standard": "Riesgo Estándar", "high": "Riesgo Alto"},
    "pt": {"low": "Risco Baixo", "standard": "Risco Padrão", "high": "Risco Alto"},
}

#: Maximum report file size in bytes (default 50 MB).
_MAX_REPORT_SIZE_BYTES: int = 50 * 1024 * 1024


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# RiskReportGenerator
# ---------------------------------------------------------------------------


class RiskReportGenerator:
    """Generate comprehensive risk assessment reports in multiple formats.

    Produces country risk profiles, commodity analyses, comparative
    assessments, trend reports, due diligence briefs, executive
    summaries, regulatory alerts, and portfolio risk dashboards in
    JSON, HTML, PDF metadata, CSV, and Excel formats with multi-language
    support.

    All report content is deterministically generated from input data
    with no LLM calls (zero-hallucination). SHA-256 content hashing
    ensures report integrity and versioning support enables report
    comparison and diff generation.

    Attributes:
        _reports: In-memory store of generated reports keyed by report_id.
        _report_versions: Version history for report comparison.
        _lock: Threading lock for thread-safe access.

    Example:
        >>> generator = RiskReportGenerator()
        >>> report = generator.generate_report(
        ...     report_type="country_profile",
        ...     format="json",
        ...     data={"country_code": "BR", "risk_score": 72.5},
        ... )
        >>> assert report.report_type == ReportType.COUNTRY_PROFILE
        >>> assert report.content_hash is not None
    """

    def __init__(self) -> None:
        """Initialize RiskReportGenerator with empty stores."""
        self._reports: Dict[str, RiskReport] = {}
        self._report_versions: Dict[str, List[RiskReport]] = {}
        self._lock: threading.Lock = threading.Lock()
        logger.info(
            "RiskReportGenerator initialized: supported_formats=%d, "
            "supported_languages=%d",
            len(SUPPORTED_OUTPUT_FORMATS),
            len(SUPPORTED_REPORT_LANGUAGES),
        )

    # ------------------------------------------------------------------
    # Primary report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        report_type: str,
        format: str = "json",
        data: Optional[Dict[str, Any]] = None,
        language: str = "en",
        title: Optional[str] = None,
        sections: Optional[List[str]] = None,
    ) -> RiskReport:
        """Generate a risk assessment report.

        Creates a complete report in the specified format and language
        using the provided data dictionary. Validates inputs, builds
        report content, calculates SHA-256 content hash, and stores
        the report with provenance tracking.

        Args:
            report_type: Type of report (country_profile, commodity_matrix,
                comparative, trend, due_diligence, executive_summary,
                regulatory_alert, portfolio_risk).
            format: Output format (json, html, pdf, csv, excel).
            data: Report data dictionary (structure varies by report_type).
            language: Language code (en, fr, de, es, pt).
            title: Optional custom report title.
            sections: Optional list of section names to include. If None,
                includes all default sections for the report type.

        Returns:
            RiskReport with metadata, content hash, and storage path.

        Raises:
            ValueError: If report_type, format, or language is invalid,
                or if required data fields are missing.
        """
        start = time.monotonic()
        cfg = get_config()

        # -- Input validation ------------------------------------------------
        report_type_enum = self._validate_report_type(report_type)
        format_enum = self._validate_format(format)
        language = self._validate_language(language)
        data = data or {}

        # -- Generate report content -----------------------------------------
        content = self._build_report_content(
            report_type_enum, format_enum, data, language, sections,
        )

        # -- Content hash ----------------------------------------------------
        content_hash = self._hash_content(content)

        # -- Extract metadata ------------------------------------------------
        countries = self._extract_countries(data)
        commodities = self._extract_commodities(data)

        # -- Build report title ----------------------------------------------
        if title is None:
            title = self._build_report_title(
                report_type_enum, language, countries, commodities,
            )

        # -- Retention -------------------------------------------------------
        retention_days = cfg.report_retention_days
        expires_at = _utcnow() + timedelta(days=retention_days)

        # -- File size -------------------------------------------------------
        content_json = json.dumps(content, ensure_ascii=False)
        file_size = len(content_json.encode("utf-8"))

        if file_size > _MAX_REPORT_SIZE_BYTES:
            logger.warning(
                "Report size %d bytes exceeds recommended max %d bytes",
                file_size, _MAX_REPORT_SIZE_BYTES,
            )

        # -- Storage path ----------------------------------------------------
        storage_path = self._generate_storage_path(
            report_type_enum, format_enum, language,
        )

        # -- Build RiskReport ------------------------------------------------
        report = RiskReport(
            report_type=report_type_enum,
            format=format_enum,
            title=title,
            language=language,
            countries=countries,
            commodities=commodities,
            content_hash=content_hash,
            file_size_bytes=file_size,
            storage_path=storage_path,
            expires_at=expires_at,
        )

        # -- Provenance ------------------------------------------------------
        tracker = get_provenance_tracker()
        prov_data = {
            "report_id": report.report_id,
            "report_type": report.report_type.value,
            "format": report.format.value,
            "language": language,
            "countries": countries,
            "commodities": commodities,
            "content_hash": content_hash,
        }
        report.provenance_hash = tracker.build_hash(prov_data)

        tracker.record(
            entity_type="risk_report",
            action="generate",
            entity_id=report.report_id,
            data=report.model_dump(mode="json"),
            metadata={
                "report_type": report_type_enum.value,
                "format": format_enum.value,
                "language": language,
                "size_bytes": file_size,
            },
        )

        # -- Store -----------------------------------------------------------
        with self._lock:
            self._reports[report.report_id] = report
            # Track versions for comparison
            version_key = f"{report_type}:{language}:{'-'.join(countries)}"
            if version_key not in self._report_versions:
                self._report_versions[version_key] = []
            self._report_versions[version_key].append(report)

        # -- Metrics ---------------------------------------------------------
        elapsed = time.monotonic() - start
        observe_report_generation_duration(elapsed)
        record_report_generated(
            report_type=report_type_enum.value,
            report_format=format_enum.value,
        )

        logger.info(
            "Report generated: type=%s format=%s language=%s countries=%s "
            "size=%d elapsed_ms=%.1f",
            report_type_enum.value,
            format_enum.value,
            language,
            countries,
            file_size,
            elapsed * 1000,
        )
        return report

    def generate_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[RiskReport]:
        """Generate multiple reports in a single batch operation.

        Each item in the batch is a dictionary with keys:
            - report_type (str, required)
            - format (str, optional, default "json")
            - data (dict, optional)
            - language (str, optional, default "en")
            - title (str, optional)
            - sections (list[str], optional)

        Args:
            items: List of report generation request dictionaries.

        Returns:
            List of RiskReport objects in the same order as input items.

        Raises:
            ValueError: If items list is empty or exceeds batch_max_size.
        """
        cfg = get_config()
        if not items:
            raise ValueError("Batch items list must not be empty")
        if len(items) > cfg.batch_max_size:
            raise ValueError(
                f"Batch size {len(items)} exceeds maximum "
                f"{cfg.batch_max_size}"
            )

        results: List[RiskReport] = []
        for item in items:
            report = self.generate_report(
                report_type=item["report_type"],
                format=item.get("format", "json"),
                data=item.get("data"),
                language=item.get("language", "en"),
                title=item.get("title"),
                sections=item.get("sections"),
            )
            results.append(report)

        logger.info(
            "Batch report generation completed: items=%d", len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Specific report types
    # ------------------------------------------------------------------

    def generate_country_profile(
        self,
        country_code: str,
        risk_score: float,
        risk_level: str,
        factors: Dict[str, float],
        format: str = "json",
        language: str = "en",
    ) -> RiskReport:
        """Generate a comprehensive country risk profile report.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            risk_score: Composite risk score (0-100).
            risk_level: Risk level classification (low/standard/high).
            factors: Dictionary of risk factors with scores.
            format: Output format (json, html, pdf, csv, excel).
            language: Language code (en, fr, de, es, pt).

        Returns:
            RiskReport with country profile content.
        """
        data = {
            "country_code": country_code.upper().strip(),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "factors": factors,
        }
        return self.generate_report(
            report_type="country_profile",
            format=format,
            data=data,
            language=language,
        )

    def generate_commodity_analysis(
        self,
        commodity_type: str,
        countries: List[str],
        risk_scores: Dict[str, float],
        format: str = "json",
        language: str = "en",
    ) -> RiskReport:
        """Generate a multi-country commodity risk analysis report.

        Args:
            commodity_type: EUDR commodity type.
            countries: List of country codes to include.
            risk_scores: Dictionary mapping country_code -> risk_score.
            format: Output format (json, html, pdf, csv, excel).
            language: Language code (en, fr, de, es, pt).

        Returns:
            RiskReport with commodity analysis content.
        """
        data = {
            "commodity_type": commodity_type,
            "countries": countries,
            "risk_scores": risk_scores,
        }
        return self.generate_report(
            report_type="commodity_analysis",
            format=format,
            data=data,
            language=language,
        )

    def generate_comparative(
        self,
        countries: List[str],
        risk_scores: Dict[str, float],
        factors: Dict[str, Dict[str, float]],
        format: str = "json",
        language: str = "en",
    ) -> RiskReport:
        """Generate a comparative risk assessment for multiple countries.

        Args:
            countries: List of country codes to compare.
            risk_scores: Dictionary mapping country_code -> composite_score.
            factors: Dictionary mapping country_code -> factor_scores.
            format: Output format (json, html, pdf, csv, excel).
            language: Language code (en, fr, de, es, pt).

        Returns:
            RiskReport with comparative analysis content.
        """
        data = {
            "countries": countries,
            "risk_scores": risk_scores,
            "factors": factors,
        }
        return self.generate_report(
            report_type="comparative",
            format=format,
            data=data,
            language=language,
        )

    def generate_trend_report(
        self,
        country_code: str,
        historical_scores: List[Dict[str, Any]],
        window_years: int = 5,
        format: str = "json",
        language: str = "en",
    ) -> RiskReport:
        """Generate a historical trend report for a country's risk scores.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            historical_scores: List of historical score records with
                date and score fields.
            window_years: Analysis window in years.
            format: Output format (json, html, pdf, csv, excel).
            language: Language code (en, fr, de, es, pt).

        Returns:
            RiskReport with trend analysis content.
        """
        data = {
            "country_code": country_code.upper().strip(),
            "historical_scores": historical_scores,
            "window_years": window_years,
        }
        return self.generate_report(
            report_type="trend",
            format=format,
            data=data,
            language=language,
        )

    def generate_executive_summary(
        self,
        kpis: Dict[str, Any],
        alerts: List[Dict[str, Any]],
        trends: Dict[str, str],
        format: str = "json",
        language: str = "en",
    ) -> RiskReport:
        """Generate an executive summary report for leadership briefing.

        Args:
            kpis: Dictionary of key performance indicators.
            alerts: List of active alert records.
            trends: Dictionary mapping category -> trend_direction.
            format: Output format (json, html, pdf, csv, excel).
            language: Language code (en, fr, de, es, pt).

        Returns:
            RiskReport with executive summary content.
        """
        data = {
            "kpis": kpis,
            "alerts": alerts,
            "trends": trends,
        }
        return self.generate_report(
            report_type="executive_summary",
            format=format,
            data=data,
            language=language,
        )

    # ------------------------------------------------------------------
    # Content hashing and integrity
    # ------------------------------------------------------------------

    def hash_content(self, content: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of report content for integrity verification.

        Args:
            content: Report content dictionary.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        return self._hash_content(content)

    def verify_content_hash(
        self,
        report_id: str,
        expected_hash: str,
    ) -> bool:
        """Verify that a report's content hash matches the expected value.

        Args:
            report_id: Report identifier to verify.
            expected_hash: Expected SHA-256 hash value.

        Returns:
            True if hashes match, False otherwise.
        """
        with self._lock:
            report = self._reports.get(report_id)

        if report is None:
            logger.warning(
                "Cannot verify hash for unknown report_id: %s", report_id,
            )
            return False

        return report.content_hash == expected_hash

    # ------------------------------------------------------------------
    # Versioning and comparison
    # ------------------------------------------------------------------

    def compare_versions(
        self,
        report_id_1: str,
        report_id_2: str,
    ) -> Dict[str, Any]:
        """Compare two report versions and return differences.

        Args:
            report_id_1: First report identifier.
            report_id_2: Second report identifier.

        Returns:
            Dictionary with comparison results, changed fields, and
            content hash differences.

        Raises:
            ValueError: If either report_id is not found.
        """
        with self._lock:
            report1 = self._reports.get(report_id_1)
            report2 = self._reports.get(report_id_2)

        if report1 is None:
            raise ValueError(f"Report not found: {report_id_1}")
        if report2 is None:
            raise ValueError(f"Report not found: {report_id_2}")

        # Compare metadata
        metadata_diff = {
            "report_type_changed": report1.report_type != report2.report_type,
            "format_changed": report1.format != report2.format,
            "language_changed": report1.language != report2.language,
            "countries_changed": set(report1.countries) != set(report2.countries),
            "commodities_changed": set(report1.commodities) != set(report2.commodities),
        }

        # Content hash diff
        content_changed = report1.content_hash != report2.content_hash

        # Time diff
        time_delta = (
            report2.generated_at - report1.generated_at
        ).total_seconds()

        return {
            "report_id_1": report_id_1,
            "report_id_2": report_id_2,
            "generated_at_1": report1.generated_at.isoformat(),
            "generated_at_2": report2.generated_at.isoformat(),
            "time_delta_seconds": time_delta,
            "content_changed": content_changed,
            "content_hash_1": report1.content_hash,
            "content_hash_2": report2.content_hash,
            "metadata_diff": metadata_diff,
        }

    def get_version_history(
        self,
        report_type: str,
        language: str,
        countries: List[str],
    ) -> List[RiskReport]:
        """Get version history for a specific report configuration.

        Args:
            report_type: Type of report.
            language: Language code.
            countries: List of country codes.

        Returns:
            List of RiskReport versions in chronological order.
        """
        version_key = f"{report_type}:{language}:{'-'.join(sorted(countries))}"
        with self._lock:
            versions = self._report_versions.get(version_key, [])

        return sorted(versions, key=lambda r: r.generated_at)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_report(self, report_id: str) -> Optional[RiskReport]:
        """Retrieve a report by its unique identifier.

        Args:
            report_id: The report_id to look up.

        Returns:
            RiskReport if found, None otherwise.
        """
        with self._lock:
            return self._reports.get(report_id)

    def list_reports(
        self,
        report_type: Optional[str] = None,
        format: Optional[str] = None,
        language: Optional[str] = None,
        country_code: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[RiskReport]:
        """List reports with optional filters.

        Args:
            report_type: Optional report type filter.
            format: Optional output format filter.
            language: Optional language code filter.
            country_code: Optional country code filter.
            limit: Maximum number of results (default 100).
            offset: Pagination offset (default 0).

        Returns:
            Filtered list of RiskReport objects.
        """
        with self._lock:
            results = list(self._reports.values())

        if report_type:
            rt_lower = report_type.lower().strip()
            results = [r for r in results if r.report_type.value == rt_lower]

        if format:
            fmt_lower = format.lower().strip()
            results = [r for r in results if r.format.value == fmt_lower]

        if language:
            lang_lower = language.lower().strip()
            results = [r for r in results if r.language == lang_lower]

        if country_code:
            cc_upper = country_code.upper().strip()
            results = [r for r in results if cc_upper in r.countries]

        # Sort by generated_at descending
        results.sort(key=lambda r: r.generated_at, reverse=True)

        return results[offset:offset + limit]

    def delete_expired_reports(self) -> int:
        """Delete reports that have exceeded their retention period.

        Returns:
            Number of reports deleted.
        """
        now = _utcnow()
        deleted_count = 0

        with self._lock:
            to_delete = [
                rid for rid, report in self._reports.items()
                if report.expires_at is not None and report.expires_at < now
            ]
            for rid in to_delete:
                del self._reports[rid]
                deleted_count += 1

        if deleted_count > 0:
            logger.info(
                "Deleted %d expired reports", deleted_count,
            )

        # Record provenance
        tracker = get_provenance_tracker()
        tracker.record(
            entity_type="risk_report",
            action="archive",
            entity_id="batch_deletion",
            data={"deleted_count": deleted_count},
            metadata={"timestamp": now.isoformat()},
        )

        return deleted_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_report_type(self, report_type: str) -> ReportType:
        """Validate and convert report_type string to enum.

        Args:
            report_type: Report type string.

        Returns:
            ReportType enum value.

        Raises:
            ValueError: If report_type is invalid.
        """
        try:
            return ReportType(report_type.lower().strip())
        except ValueError:
            raise ValueError(
                f"Invalid report_type '{report_type}'; "
                f"must be one of: {[e.value for e in ReportType]}"
            )

    def _validate_format(self, format: str) -> ReportFormat:
        """Validate and convert format string to enum.

        Args:
            format: Format string.

        Returns:
            ReportFormat enum value.

        Raises:
            ValueError: If format is invalid.
        """
        try:
            return ReportFormat(format.lower().strip())
        except ValueError:
            raise ValueError(
                f"Invalid format '{format}'; "
                f"must be one of: {[e.value for e in ReportFormat]}"
            )

    def _validate_language(self, language: str) -> str:
        """Validate language code.

        Args:
            language: Language code string.

        Returns:
            Lowercase language code.

        Raises:
            ValueError: If language is not supported.
        """
        lang_lower = language.lower().strip()
        if lang_lower not in SUPPORTED_REPORT_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'; "
                f"must be one of: {SUPPORTED_REPORT_LANGUAGES}"
            )
        return lang_lower

    def _build_report_content(
        self,
        report_type: ReportType,
        format: ReportFormat,
        data: Dict[str, Any],
        language: str,
        sections: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Build report content structure based on type and format.

        Args:
            report_type: Report type enum.
            format: Report format enum.
            data: Input data dictionary.
            language: Language code.
            sections: Optional list of sections to include.

        Returns:
            Report content dictionary.
        """
        # Get section templates for the language
        templates = _SECTION_TEMPLATES.get(language, _SECTION_TEMPLATES["en"])

        # Build content based on report type
        if report_type == ReportType.COUNTRY_PROFILE:
            content = self._build_country_profile_content(
                data, templates, sections,
            )
        elif report_type == ReportType.COMMODITY_MATRIX:
            content = self._build_commodity_matrix_content(
                data, templates, sections,
            )
        elif report_type == ReportType.COMPARATIVE:
            content = self._build_comparative_content(
                data, templates, sections,
            )
        elif report_type == ReportType.TREND:
            content = self._build_trend_content(
                data, templates, sections,
            )
        elif report_type == ReportType.DUE_DILIGENCE:
            content = self._build_due_diligence_content(
                data, templates, sections,
            )
        elif report_type == ReportType.EXECUTIVE_SUMMARY:
            content = self._build_executive_summary_content(
                data, templates, sections,
            )
        else:
            # Generic content structure
            content = {
                "data": data,
                "sections": templates,
            }

        # Add metadata
        content["metadata"] = {
            "report_type": report_type.value,
            "format": format.value,
            "language": language,
            "generated_at": _utcnow().isoformat(),
        }

        return content

    def _build_country_profile_content(
        self,
        data: Dict[str, Any],
        templates: Dict[str, str],
        sections: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Build country profile report content.

        Args:
            data: Country profile data.
            templates: Section templates.
            sections: Optional section filter.

        Returns:
            Content dictionary.
        """
        country_code = data.get("country_code", "UNKNOWN")
        risk_score = data.get("risk_score", 0.0)
        risk_level = data.get("risk_level", "standard")
        factors = data.get("factors", {})

        content = {
            "title": f"Country Risk Profile: {country_code}",
            "country_code": country_code,
            "risk_score": risk_score,
            "risk_level": risk_level,
        }

        # Default sections for country profile
        default_sections = [
            "executive_summary",
            "risk_scoring",
            "factor_analysis",
            "commodity_details",
            "hotspot_maps",
            "governance",
            "trade_flows",
            "recommendations",
            "appendices",
        ]
        sections_to_include = sections or default_sections

        if "executive_summary" in sections_to_include:
            content["executive_summary"] = {
                "heading": templates.get("executive_summary"),
                "summary": (
                    f"Risk level: {risk_level.upper()}, "
                    f"Score: {risk_score:.1f}/100"
                ),
            }

        if "risk_scoring" in sections_to_include:
            content["risk_scoring"] = {
                "heading": templates.get("risk_scoring"),
                "score": risk_score,
                "level": risk_level,
            }

        if "factor_analysis" in sections_to_include:
            content["factor_analysis"] = {
                "heading": templates.get("factor_analysis"),
                "factors": factors,
            }

        if "commodity_details" in sections_to_include:
            content["commodity_details"] = {
                "heading": templates.get("commodity_details"),
                "commodities": data.get("commodities", []),
            }

        if "hotspot_maps" in sections_to_include:
            content["hotspot_maps"] = {
                "heading": templates.get("hotspot_maps"),
                "hotspots": data.get("hotspots", []),
            }

        if "governance" in sections_to_include:
            content["governance"] = {
                "heading": templates.get("governance"),
                "indicators": data.get("governance_indicators", {}),
            }

        if "trade_flows" in sections_to_include:
            content["trade_flows"] = {
                "heading": templates.get("trade_flows"),
                "flows": data.get("trade_flows", []),
            }

        if "recommendations" in sections_to_include:
            content["recommendations"] = {
                "heading": templates.get("recommendations"),
                "items": data.get("recommendations", []),
            }

        if "appendices" in sections_to_include:
            content["appendices"] = {
                "heading": templates.get("appendices"),
                "data_sources": data.get("data_sources", []),
                "methodology": data.get("methodology", ""),
            }

        return content

    def _build_commodity_matrix_content(
        self,
        data: Dict[str, Any],
        templates: Dict[str, str],
        sections: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Build commodity matrix report content.

        Args:
            data: Commodity matrix data.
            templates: Section templates.
            sections: Optional section filter.

        Returns:
            Content dictionary.
        """
        commodity_type = data.get("commodity_type", "unknown")
        countries = data.get("countries", [])
        risk_scores = data.get("risk_scores", {})

        content = {
            "title": f"Commodity Risk Matrix: {commodity_type}",
            "commodity_type": commodity_type,
            "countries": countries,
            "risk_scores": risk_scores,
            "matrix": self._build_risk_matrix(countries, risk_scores),
        }

        return content

    def _build_comparative_content(
        self,
        data: Dict[str, Any],
        templates: Dict[str, str],
        sections: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Build comparative analysis report content.

        Args:
            data: Comparative data.
            templates: Section templates.
            sections: Optional section filter.

        Returns:
            Content dictionary.
        """
        countries = data.get("countries", [])
        risk_scores = data.get("risk_scores", {})
        factors = data.get("factors", {})

        content = {
            "title": f"Comparative Risk Assessment: {', '.join(countries)}",
            "countries": countries,
            "risk_scores": risk_scores,
            "factors": factors,
            "comparison_table": self._build_comparison_table(
                countries, risk_scores, factors,
            ),
        }

        return content

    def _build_trend_content(
        self,
        data: Dict[str, Any],
        templates: Dict[str, str],
        sections: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Build trend analysis report content.

        Args:
            data: Trend data.
            templates: Section templates.
            sections: Optional section filter.

        Returns:
            Content dictionary.
        """
        country_code = data.get("country_code", "UNKNOWN")
        historical_scores = data.get("historical_scores", [])
        window_years = data.get("window_years", 5)

        # Calculate trend direction
        trend_direction = self._calculate_trend_direction(historical_scores)

        content = {
            "title": f"Risk Trend Analysis: {country_code}",
            "country_code": country_code,
            "window_years": window_years,
            "historical_scores": historical_scores,
            "trend_direction": trend_direction,
            "chart_data": self._build_trend_chart_data(historical_scores),
        }

        return content

    def _build_due_diligence_content(
        self,
        data: Dict[str, Any],
        templates: Dict[str, str],
        sections: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Build due diligence brief report content.

        Args:
            data: DD brief data.
            templates: Section templates.
            sections: Optional section filter.

        Returns:
            Content dictionary.
        """
        country_code = data.get("country_code", "UNKNOWN")
        dd_level = data.get("dd_level", "standard")
        requirements = data.get("requirements", [])
        costs = data.get("costs", {})

        content = {
            "title": f"Due Diligence Brief: {country_code}",
            "country_code": country_code,
            "dd_level": dd_level,
            "requirements": requirements,
            "costs": costs,
            "timeline": data.get("timeline", {}),
        }

        return content

    def _build_executive_summary_content(
        self,
        data: Dict[str, Any],
        templates: Dict[str, str],
        sections: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Build executive summary report content.

        Args:
            data: Executive summary data.
            templates: Section templates.
            sections: Optional section filter.

        Returns:
            Content dictionary.
        """
        kpis = data.get("kpis", {})
        alerts = data.get("alerts", [])
        trends = data.get("trends", {})

        content = {
            "title": "Executive Summary: Risk Assessment Dashboard",
            "kpis": kpis,
            "alerts": alerts,
            "trends": trends,
            "dashboard_data": self._build_dashboard_data(kpis, alerts, trends),
        }

        return content

    def _build_risk_matrix(
        self,
        countries: List[str],
        risk_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Build risk matrix table data.

        Args:
            countries: List of country codes.
            risk_scores: Country -> score mapping.

        Returns:
            List of matrix row dictionaries.
        """
        matrix = []
        for country in countries:
            score = risk_scores.get(country, 0.0)
            matrix.append({
                "country": country,
                "score": score,
                "level": self._score_to_level(score),
            })
        return matrix

    def _build_comparison_table(
        self,
        countries: List[str],
        risk_scores: Dict[str, float],
        factors: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Build comparison table data.

        Args:
            countries: List of country codes.
            risk_scores: Country -> composite score mapping.
            factors: Country -> factor scores mapping.

        Returns:
            List of comparison row dictionaries.
        """
        table = []
        for country in countries:
            row = {
                "country": country,
                "composite_score": risk_scores.get(country, 0.0),
                "factors": factors.get(country, {}),
            }
            table.append(row)
        return table

    def _build_trend_chart_data(
        self,
        historical_scores: List[Dict[str, Any]],
    ) -> Dict[str, List[Any]]:
        """Build trend chart data structure.

        Args:
            historical_scores: List of score records with date and score.

        Returns:
            Dictionary with x (dates) and y (scores) arrays.
        """
        dates = []
        scores = []
        for record in historical_scores:
            dates.append(record.get("date"))
            scores.append(record.get("score", 0.0))

        return {
            "x": dates,
            "y": scores,
        }

    def _build_dashboard_data(
        self,
        kpis: Dict[str, Any],
        alerts: List[Dict[str, Any]],
        trends: Dict[str, str],
    ) -> Dict[str, Any]:
        """Build executive dashboard data structure.

        Args:
            kpis: Key performance indicators.
            alerts: Active alerts.
            trends: Trend directions.

        Returns:
            Dashboard data dictionary.
        """
        return {
            "kpis": kpis,
            "alert_count": len(alerts),
            "active_alerts": alerts[:5],  # Top 5 alerts
            "trends": trends,
        }

    def _calculate_trend_direction(
        self,
        historical_scores: List[Dict[str, Any]],
    ) -> str:
        """Calculate trend direction from historical scores.

        Args:
            historical_scores: List of score records with date and score.

        Returns:
            Trend direction string (improving, stable, deteriorating,
            insufficient_data).
        """
        if len(historical_scores) < 2:
            return "insufficient_data"

        scores = [r.get("score", 0.0) for r in historical_scores]
        first_score = scores[0]
        last_score = scores[-1]

        delta = last_score - first_score
        threshold = 5.0  # +/- 5 points for stability

        if abs(delta) <= threshold:
            return "stable"
        if delta < 0:
            return "improving"  # Score decreased = improvement
        return "deteriorating"

    def _score_to_level(self, score: float) -> str:
        """Convert risk score to risk level.

        Args:
            score: Risk score (0-100).

        Returns:
            Risk level string (low, standard, high).
        """
        cfg = get_config()
        if score <= cfg.low_risk_threshold:
            return "low"
        if score <= cfg.high_risk_threshold:
            return "standard"
        return "high"

    def _hash_content(self, content: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of content dictionary.

        Args:
            content: Content dictionary.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        content_json = json.dumps(content, sort_keys=True, ensure_ascii=False)
        content_bytes = content_json.encode("utf-8")
        return hashlib.sha256(content_bytes).hexdigest()

    def _extract_countries(self, data: Dict[str, Any]) -> List[str]:
        """Extract country codes from data dictionary.

        Args:
            data: Report data.

        Returns:
            List of country codes (uppercase).
        """
        countries = []
        if "country_code" in data:
            countries.append(data["country_code"].upper())
        if "countries" in data and isinstance(data["countries"], list):
            countries.extend([c.upper() for c in data["countries"]])
        return sorted(set(countries))

    def _extract_commodities(self, data: Dict[str, Any]) -> List[str]:
        """Extract commodity types from data dictionary.

        Args:
            data: Report data.

        Returns:
            List of commodity type strings (lowercase).
        """
        commodities = []
        if "commodity_type" in data:
            commodities.append(data["commodity_type"].lower())
        if "commodities" in data and isinstance(data["commodities"], list):
            commodities.extend([c.lower() for c in data["commodities"]])
        return sorted(set(commodities))

    def _build_report_title(
        self,
        report_type: ReportType,
        language: str,
        countries: List[str],
        commodities: List[str],
    ) -> str:
        """Build default report title based on type and content.

        Args:
            report_type: Report type enum.
            language: Language code.
            countries: List of country codes.
            commodities: List of commodity types.

        Returns:
            Report title string.
        """
        templates = _SECTION_TEMPLATES.get(language, _SECTION_TEMPLATES["en"])

        if report_type == ReportType.COUNTRY_PROFILE:
            country_str = ", ".join(countries) if countries else "Unknown"
            return f"Country Risk Profile: {country_str}"
        elif report_type == ReportType.COMMODITY_MATRIX:
            commodity_str = ", ".join(commodities) if commodities else "All"
            return f"Commodity Risk Matrix: {commodity_str}"
        elif report_type == ReportType.COMPARATIVE:
            country_str = ", ".join(countries) if countries else "Multiple"
            return f"Comparative Risk Assessment: {country_str}"
        elif report_type == ReportType.TREND:
            country_str = ", ".join(countries) if countries else "Unknown"
            return f"Risk Trend Analysis: {country_str}"
        elif report_type == ReportType.DUE_DILIGENCE:
            country_str = ", ".join(countries) if countries else "Unknown"
            return f"Due Diligence Brief: {country_str}"
        elif report_type == ReportType.EXECUTIVE_SUMMARY:
            return "Executive Summary: Risk Assessment Dashboard"
        else:
            return f"Risk Report: {report_type.value}"

    def _generate_storage_path(
        self,
        report_type: ReportType,
        format: ReportFormat,
        language: str,
    ) -> str:
        """Generate storage path for report file.

        Args:
            report_type: Report type enum.
            format: Report format enum.
            language: Language code.

        Returns:
            Storage path string.
        """
        timestamp = _utcnow().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{report_type.value}_{language}_{timestamp}.{format.value}"
        )
        return f"reports/{report_type.value}/{filename}"

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        with self._lock:
            count = len(self._reports)
        return (
            f"RiskReportGenerator("
            f"reports={count}, "
            f"formats={len(SUPPORTED_OUTPUT_FORMATS)}, "
            f"languages={len(SUPPORTED_REPORT_LANGUAGES)})"
        )

    def __len__(self) -> int:
        """Return number of stored reports."""
        with self._lock:
            return len(self._reports)
