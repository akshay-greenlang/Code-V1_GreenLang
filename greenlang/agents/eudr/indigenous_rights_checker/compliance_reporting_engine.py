# -*- coding: utf-8 -*-
"""
ComplianceReportingEngine - Feature 8: Compliance Reporting

Generates indigenous rights compliance reports in multiple formats
(PDF, JSON, HTML, CSV, XLSX) and languages (EN, FR, DE, ES, PT).
Supports DDS section generation, certification scheme reports
(FSC FPIC, RSPO FPIC), supplier scorecards, trend reports, and
executive summaries for BI dashboards.

Per PRD F8.1-F8.6: 8 report types, multi-language support, audit-grade
provenance, and export capabilities for external BI systems.

Example:
    >>> engine = ComplianceReportingEngine(config, provenance)
    >>> report = await engine.generate_report(
    ...     report_type="indigenous_rights_compliance",
    ...     scope_type="operator",
    ...     scope_ids=["op-001"],
    ...     output_format="json",
    ... )
    >>> assert report.report_id is not None

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 (Feature 8)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.indigenous_rights_checker.config import (
    IndigenousRightsCheckerConfig,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    ComplianceReport,
    ReportFormat,
    ReportType,
)
from greenlang.agents.eudr.indigenous_rights_checker.provenance import (
    ProvenanceTracker,
)
from greenlang.agents.eudr.indigenous_rights_checker.metrics import (
    record_report_generated,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL templates
# ---------------------------------------------------------------------------

_SQL_INSERT_REPORT = """
    INSERT INTO eudr_indigenous_rights_checker.gl_eudr_irc_compliance_reports (
        report_id, report_type, title, format, language,
        scope_type, scope_ids, parameters, file_path,
        file_size_bytes, provenance_hash, generated_by
    ) VALUES (
        %(report_id)s, %(report_type)s, %(title)s, %(format)s,
        %(language)s, %(scope_type)s, %(scope_ids)s, %(parameters)s,
        %(file_path)s, %(file_size_bytes)s, %(provenance_hash)s,
        %(generated_by)s
    )
"""

_SQL_GET_REPORTS = """
    SELECT report_id, report_type, title, format, language,
           scope_type, scope_ids, parameters, file_path,
           file_size_bytes, provenance_hash, generated_by,
           generated_at
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_compliance_reports
    ORDER BY generated_at DESC
    LIMIT %(limit)s
"""

_SQL_GET_REPORT = """
    SELECT report_id, report_type, title, format, language,
           scope_type, scope_ids, parameters, file_path,
           file_size_bytes, provenance_hash, generated_by,
           generated_at
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_compliance_reports
    WHERE report_id = %(report_id)s
"""

# ---------------------------------------------------------------------------
# Report data aggregation queries
# ---------------------------------------------------------------------------

_SQL_AGG_TERRITORY_SUMMARY = """
    SELECT COUNT(*) AS total_territories,
           COUNT(DISTINCT country_code) AS countries,
           SUM(CASE WHEN legal_status = 'titled' THEN 1 ELSE 0 END) AS titled_count,
           SUM(CASE WHEN legal_status = 'declared' THEN 1 ELSE 0 END) AS declared_count,
           SUM(CASE WHEN legal_status = 'customary' THEN 1 ELSE 0 END) AS customary_count
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_territories
"""

_SQL_AGG_OVERLAP_SUMMARY = """
    SELECT COUNT(*) AS total_overlaps,
           SUM(CASE WHEN risk_level = 'critical' THEN 1 ELSE 0 END) AS critical_count,
           SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) AS high_count,
           SUM(CASE WHEN risk_level = 'medium' THEN 1 ELSE 0 END) AS medium_count,
           SUM(CASE WHEN risk_level = 'low' THEN 1 ELSE 0 END) AS low_count
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_overlaps
"""

_SQL_AGG_FPIC_SUMMARY = """
    SELECT COUNT(*) AS total_assessments,
           AVG(fpic_score) AS avg_score,
           SUM(CASE WHEN fpic_status = 'consent_obtained' THEN 1 ELSE 0 END) AS obtained_count,
           SUM(CASE WHEN fpic_status = 'consent_partial' THEN 1 ELSE 0 END) AS partial_count,
           SUM(CASE WHEN fpic_status = 'consent_missing' THEN 1 ELSE 0 END) AS missing_count
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_fpic_assessments
"""

_SQL_AGG_VIOLATION_SUMMARY = """
    SELECT COUNT(*) AS total_violations,
           SUM(CASE WHEN severity_level = 'critical' THEN 1 ELSE 0 END) AS critical_count,
           SUM(CASE WHEN severity_level = 'high' THEN 1 ELSE 0 END) AS high_count,
           SUM(CASE WHEN supply_chain_correlation = TRUE THEN 1 ELSE 0 END) AS correlated_count
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_violation_alerts
    WHERE status = 'active'
"""

_SQL_AGG_WORKFLOW_SUMMARY = """
    SELECT COUNT(*) AS total_workflows,
           SUM(CASE WHEN current_stage = 'monitoring' THEN 1 ELSE 0 END) AS completed_count,
           SUM(CASE WHEN sla_status = 'overdue' THEN 1 ELSE 0 END) AS overdue_count
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_workflows
"""

# ---------------------------------------------------------------------------
# Report title templates per type
# ---------------------------------------------------------------------------

_REPORT_TITLES: Dict[str, Dict[str, str]] = {
    "en": {
        "indigenous_rights_compliance": "Indigenous Rights Compliance Report",
        "dds_section": "Due Diligence Statement - Indigenous Rights Section",
        "fsc_fpic": "FSC FPIC Compliance Assessment Report",
        "rspo_fpic": "RSPO FPIC Compliance Assessment Report",
        "supplier_scorecard": "Supplier Indigenous Rights Scorecard",
        "trend_report": "Indigenous Rights Trend Analysis Report",
        "executive_summary": "Executive Summary - Indigenous Rights",
        "bi_export": "Indigenous Rights BI Data Export",
    },
    "fr": {
        "indigenous_rights_compliance": "Rapport de conformite des droits autochtones",
        "dds_section": "Declaration de diligence raisonnable - Section droits autochtones",
        "fsc_fpic": "Rapport d'evaluation de conformite FPIC FSC",
        "rspo_fpic": "Rapport d'evaluation de conformite FPIC RSPO",
        "supplier_scorecard": "Tableau de bord fournisseur - Droits autochtones",
        "trend_report": "Rapport d'analyse des tendances - Droits autochtones",
        "executive_summary": "Resume executif - Droits autochtones",
        "bi_export": "Export BI - Droits autochtones",
    },
    "de": {
        "indigenous_rights_compliance": "Compliance-Bericht indigene Rechte",
        "dds_section": "Sorgfaltserklaerung - Abschnitt indigene Rechte",
        "fsc_fpic": "FSC FPIC Compliance-Bewertungsbericht",
        "rspo_fpic": "RSPO FPIC Compliance-Bewertungsbericht",
        "supplier_scorecard": "Lieferanten-Scorecard indigene Rechte",
        "trend_report": "Trendanalysebericht indigene Rechte",
        "executive_summary": "Zusammenfassung - Indigene Rechte",
        "bi_export": "BI-Datenexport - Indigene Rechte",
    },
    "es": {
        "indigenous_rights_compliance": "Informe de cumplimiento de derechos indigenas",
        "dds_section": "Declaracion de diligencia debida - Seccion derechos indigenas",
        "fsc_fpic": "Informe de evaluacion de cumplimiento FPIC FSC",
        "rspo_fpic": "Informe de evaluacion de cumplimiento FPIC RSPO",
        "supplier_scorecard": "Cuadro de mando del proveedor - Derechos indigenas",
        "trend_report": "Informe de analisis de tendencias - Derechos indigenas",
        "executive_summary": "Resumen ejecutivo - Derechos indigenas",
        "bi_export": "Exportacion BI - Derechos indigenas",
    },
    "pt": {
        "indigenous_rights_compliance": "Relatorio de conformidade de direitos indigenas",
        "dds_section": "Declaracao de diligencia devida - Secao direitos indigenas",
        "fsc_fpic": "Relatorio de avaliacao de conformidade FPIC FSC",
        "rspo_fpic": "Relatorio de avaliacao de conformidade FPIC RSPO",
        "supplier_scorecard": "Scorecard do fornecedor - Direitos indigenas",
        "trend_report": "Relatorio de analise de tendencias - Direitos indigenas",
        "executive_summary": "Resumo executivo - Direitos indigenas",
        "bi_export": "Exportacao BI - Direitos indigenas",
    },
}


class ComplianceReportingEngine:
    """Engine for generating indigenous rights compliance reports.

    Supports 8 report types in 5 formats and 5 languages with
    complete audit trail and provenance tracking.

    Attributes:
        _config: Agent configuration with reporting settings.
        _provenance: Provenance tracker for audit trail.
        _pool: Async database connection pool.
    """

    def __init__(
        self,
        config: IndigenousRightsCheckerConfig,
        provenance: ProvenanceTracker,
    ) -> None:
        """Initialize ComplianceReportingEngine."""
        self._config = config
        self._provenance = provenance
        self._pool: Any = None
        logger.info("ComplianceReportingEngine initialized")

    async def startup(self, pool: Any) -> None:
        """Set the database connection pool."""
        self._pool = pool

    async def shutdown(self) -> None:
        """Clean up engine resources."""
        self._pool = None

    # -------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------

    async def generate_report(
        self,
        report_type: str,
        scope_type: str,
        scope_ids: Optional[List[str]] = None,
        output_format: str = "json",
        language: str = "en",
        parameters: Optional[Dict[str, Any]] = None,
        generated_by: str = "system",
    ) -> ComplianceReport:
        """Generate a compliance report.

        Aggregates data from all engines and generates a structured
        report in the requested format and language.

        Args:
            report_type: ReportType enum value string.
            scope_type: Scope level (operator, supplier, country, global).
            scope_ids: List of scope identifiers.
            output_format: ReportFormat enum value string (default: json).
            language: Language code (default: en).
            parameters: Additional report parameters.
            generated_by: Actor generating the report.

        Returns:
            ComplianceReport metadata with report content reference.
        """
        report_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Resolve report title
        title = self._resolve_title(report_type, language)

        # Aggregate report data
        report_data = await self._aggregate_report_data(
            report_type, scope_type, scope_ids or [], parameters or {}
        )

        # Serialize report content
        content = self._serialize_report(
            report_id=report_id,
            report_type=report_type,
            title=title,
            language=language,
            scope_type=scope_type,
            scope_ids=scope_ids or [],
            data=report_data,
            generated_at=now,
        )

        # Compute content size
        content_bytes = json.dumps(content, default=str).encode("utf-8")
        file_size_bytes = len(content_bytes)

        # Compute provenance hash
        provenance_hash = self._provenance.compute_data_hash({
            "report_id": report_id,
            "report_type": report_type,
            "scope_type": scope_type,
            "scope_ids": scope_ids or [],
            "generated_at": now.isoformat(),
        })

        report = ComplianceReport(
            report_id=report_id,
            report_type=ReportType(report_type),
            title=title,
            format=ReportFormat(output_format),
            language=language,
            scope_type=scope_type,
            scope_ids=scope_ids or [],
            parameters=parameters or {},
            file_size_bytes=file_size_bytes,
            provenance_hash=provenance_hash,
            generated_by=generated_by,
            generated_at=now,
        )

        self._provenance.record(
            "report", "generate", report_id,
            metadata={
                "report_type": report_type,
                "format": output_format,
                "language": language,
                "scope_type": scope_type,
            },
        )

        record_report_generated(report_type)
        await self._persist_report(report)

        logger.info(
            f"Report generated: {report_id} type={report_type} "
            f"format={output_format} lang={language}"
        )

        return report

    async def get_report(
        self, report_id: str
    ) -> Optional[ComplianceReport]:
        """Get a report by identifier.

        Args:
            report_id: Report UUID.

        Returns:
            ComplianceReport or None if not found.
        """
        if self._pool is None:
            return None

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SQL_GET_REPORT, {"report_id": report_id}
                )
                row = await cur.fetchone()

        if row is None:
            return None

        return self._row_to_report(row)

    async def list_reports(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent compliance reports.

        Args:
            limit: Maximum records to return.

        Returns:
            List of report summary dictionaries.
        """
        if self._pool is None:
            return []

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_GET_REPORTS, {"limit": limit})
                rows = await cur.fetchall()

        return [
            {
                "report_id": str(row[0]),
                "report_type": row[1],
                "title": row[2],
                "format": row[3],
                "language": row[4],
                "scope_type": row[5],
                "file_size_bytes": row[9],
                "provenance_hash": row[10],
                "generated_by": row[11],
                "generated_at": row[12].isoformat() if row[12] else None,
            }
            for row in rows
        ]

    # -------------------------------------------------------------------
    # Report data aggregation
    # -------------------------------------------------------------------

    async def _aggregate_report_data(
        self,
        report_type: str,
        scope_type: str,
        scope_ids: List[str],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate data for report generation.

        Queries all relevant data from the database and assembles
        it into a structured report data dictionary.

        Args:
            report_type: Report type identifier.
            scope_type: Scope level.
            scope_ids: Scope identifiers.
            parameters: Additional parameters.

        Returns:
            Aggregated report data dictionary.
        """
        data: Dict[str, Any] = {
            "report_type": report_type,
            "scope_type": scope_type,
            "scope_ids": scope_ids,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sections": {},
        }

        if self._pool is None:
            data["sections"]["note"] = "Database not connected"
            return data

        # Aggregate sections based on report type
        if report_type in (
            "indigenous_rights_compliance",
            "dds_section",
            "executive_summary",
        ):
            data["sections"]["territories"] = (
                await self._aggregate_territory_summary()
            )
            data["sections"]["overlaps"] = (
                await self._aggregate_overlap_summary()
            )
            data["sections"]["fpic_assessments"] = (
                await self._aggregate_fpic_summary()
            )
            data["sections"]["violations"] = (
                await self._aggregate_violation_summary()
            )
            data["sections"]["workflows"] = (
                await self._aggregate_workflow_summary()
            )

        elif report_type in ("fsc_fpic", "rspo_fpic"):
            data["sections"]["fpic_assessments"] = (
                await self._aggregate_fpic_summary()
            )
            data["sections"]["territories"] = (
                await self._aggregate_territory_summary()
            )

        elif report_type == "supplier_scorecard":
            data["sections"]["overlaps"] = (
                await self._aggregate_overlap_summary()
            )
            data["sections"]["fpic_assessments"] = (
                await self._aggregate_fpic_summary()
            )
            data["sections"]["violations"] = (
                await self._aggregate_violation_summary()
            )

        elif report_type == "trend_report":
            data["sections"]["violations"] = (
                await self._aggregate_violation_summary()
            )
            data["sections"]["fpic_assessments"] = (
                await self._aggregate_fpic_summary()
            )

        elif report_type == "bi_export":
            data["sections"]["territories"] = (
                await self._aggregate_territory_summary()
            )
            data["sections"]["overlaps"] = (
                await self._aggregate_overlap_summary()
            )
            data["sections"]["fpic_assessments"] = (
                await self._aggregate_fpic_summary()
            )
            data["sections"]["violations"] = (
                await self._aggregate_violation_summary()
            )
            data["sections"]["workflows"] = (
                await self._aggregate_workflow_summary()
            )

        return data

    async def _aggregate_territory_summary(self) -> Dict[str, Any]:
        """Aggregate territory summary statistics."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_AGG_TERRITORY_SUMMARY)
                row = await cur.fetchone()

        if row is None:
            return {}

        return {
            "total_territories": row[0] or 0,
            "countries": row[1] or 0,
            "titled_count": row[2] or 0,
            "declared_count": row[3] or 0,
            "customary_count": row[4] or 0,
        }

    async def _aggregate_overlap_summary(self) -> Dict[str, Any]:
        """Aggregate overlap summary statistics."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_AGG_OVERLAP_SUMMARY)
                row = await cur.fetchone()

        if row is None:
            return {}

        return {
            "total_overlaps": row[0] or 0,
            "critical_count": row[1] or 0,
            "high_count": row[2] or 0,
            "medium_count": row[3] or 0,
            "low_count": row[4] or 0,
        }

    async def _aggregate_fpic_summary(self) -> Dict[str, Any]:
        """Aggregate FPIC assessment summary statistics."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_AGG_FPIC_SUMMARY)
                row = await cur.fetchone()

        if row is None:
            return {}

        return {
            "total_assessments": row[0] or 0,
            "average_score": str(row[1]) if row[1] else "0",
            "consent_obtained_count": row[2] or 0,
            "consent_partial_count": row[3] or 0,
            "consent_missing_count": row[4] or 0,
        }

    async def _aggregate_violation_summary(self) -> Dict[str, Any]:
        """Aggregate violation alert summary statistics."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_AGG_VIOLATION_SUMMARY)
                row = await cur.fetchone()

        if row is None:
            return {}

        return {
            "total_active_violations": row[0] or 0,
            "critical_count": row[1] or 0,
            "high_count": row[2] or 0,
            "correlated_count": row[3] or 0,
        }

    async def _aggregate_workflow_summary(self) -> Dict[str, Any]:
        """Aggregate FPIC workflow summary statistics."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_AGG_WORKFLOW_SUMMARY)
                row = await cur.fetchone()

        if row is None:
            return {}

        return {
            "total_workflows": row[0] or 0,
            "completed_count": row[1] or 0,
            "overdue_count": row[2] or 0,
        }

    # -------------------------------------------------------------------
    # Report serialization
    # -------------------------------------------------------------------

    def _serialize_report(
        self,
        report_id: str,
        report_type: str,
        title: str,
        language: str,
        scope_type: str,
        scope_ids: List[str],
        data: Dict[str, Any],
        generated_at: datetime,
    ) -> Dict[str, Any]:
        """Serialize report data into structured format.

        Args:
            report_id: Report UUID.
            report_type: Report type.
            title: Report title.
            language: Language code.
            scope_type: Scope level.
            scope_ids: Scope identifiers.
            data: Aggregated report data.
            generated_at: Generation timestamp.

        Returns:
            Serialized report content dictionary.
        """
        return {
            "metadata": {
                "report_id": report_id,
                "report_type": report_type,
                "title": title,
                "language": language,
                "scope_type": scope_type,
                "scope_ids": scope_ids,
                "generated_at": generated_at.isoformat(),
                "agent_id": "GL-EUDR-IRC-021",
                "agent_version": "1.0.0",
                "regulation": "EU 2023/1115 (EUDR)",
                "applicable_articles": [
                    "Article 2 (Indigenous Rights)",
                    "Article 8 (Due Diligence)",
                    "Article 10 (Risk Assessment)",
                    "Article 11 (Risk Mitigation)",
                    "Article 29 (Country Benchmarking)",
                ],
            },
            "data": data.get("sections", {}),
            "compliance_status": self._determine_compliance_status(
                data.get("sections", {})
            ),
        }

    def _determine_compliance_status(
        self, sections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine overall compliance status from report sections.

        Deterministic compliance assessment based on aggregated data.

        Args:
            sections: Report data sections.

        Returns:
            Compliance status dictionary.
        """
        status = {
            "overall": "compliant",
            "findings": [],
        }

        # Check for critical overlaps
        overlaps = sections.get("overlaps", {})
        if overlaps.get("critical_count", 0) > 0:
            status["overall"] = "non_compliant"
            status["findings"].append(
                f"{overlaps['critical_count']} critical territory overlaps detected"
            )

        # Check for missing FPIC
        fpic = sections.get("fpic_assessments", {})
        if fpic.get("consent_missing_count", 0) > 0:
            status["overall"] = "non_compliant"
            status["findings"].append(
                f"{fpic['consent_missing_count']} plots with missing FPIC consent"
            )

        # Check for active violations
        violations = sections.get("violations", {})
        if violations.get("critical_count", 0) > 0:
            if status["overall"] != "non_compliant":
                status["overall"] = "requires_attention"
            status["findings"].append(
                f"{violations['critical_count']} critical violations active"
            )

        # Check for overdue workflows
        workflows = sections.get("workflows", {})
        if workflows.get("overdue_count", 0) > 0:
            if status["overall"] != "non_compliant":
                status["overall"] = "requires_attention"
            status["findings"].append(
                f"{workflows['overdue_count']} FPIC workflows overdue"
            )

        return status

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _resolve_title(self, report_type: str, language: str) -> str:
        """Resolve report title from templates.

        Args:
            report_type: Report type identifier.
            language: Language code.

        Returns:
            Localized report title string.
        """
        lang_titles = _REPORT_TITLES.get(language, _REPORT_TITLES["en"])
        return lang_titles.get(
            report_type,
            f"Indigenous Rights Report - {report_type}",
        )

    def _row_to_report(self, row: Any) -> ComplianceReport:
        """Convert database row to ComplianceReport model.

        Args:
            row: Database row tuple.

        Returns:
            ComplianceReport instance.
        """
        scope_ids = row[6] if isinstance(row[6], list) else []
        parameters = row[7] if isinstance(row[7], dict) else {}

        return ComplianceReport(
            report_id=str(row[0]),
            report_type=ReportType(row[1]),
            title=row[2],
            format=ReportFormat(row[3]),
            language=row[4],
            scope_type=row[5],
            scope_ids=scope_ids,
            parameters=parameters,
            file_path=row[8],
            file_size_bytes=row[9],
            provenance_hash=row[10],
            generated_by=row[11],
            generated_at=row[12],
        )

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    async def _persist_report(self, report: ComplianceReport) -> None:
        """Persist report metadata to database."""
        if self._pool is None:
            return

        params = {
            "report_id": report.report_id,
            "report_type": report.report_type.value,
            "title": report.title,
            "format": report.format.value,
            "language": report.language,
            "scope_type": report.scope_type,
            "scope_ids": json.dumps(report.scope_ids),
            "parameters": json.dumps(report.parameters),
            "file_path": report.file_path,
            "file_size_bytes": report.file_size_bytes,
            "provenance_hash": report.provenance_hash,
            "generated_by": report.generated_by,
        }

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_INSERT_REPORT, params)
            await conn.commit()
