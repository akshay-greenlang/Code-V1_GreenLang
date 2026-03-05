"""
CDP Report Generator -- Multi-Format Report Generation

This module generates CDP Climate Change disclosure reports in multiple formats:
PDF, Excel, XML (CDP ORS), and JSON.  It produces executive summaries, full
response exports, submission completeness validation, verification evidence
packages, and pre-submission compliance checks.

Key capabilities:
  - PDF report with all responses, scoring, and gap analysis
  - Excel tabular export with per-module worksheets
  - XML export for CDP Online Response System (ORS) submission
  - JSON export for programmatic consumption
  - Executive summary generation
  - Submission completeness validation
  - Verification evidence package assembly
  - Pre-submission compliance checks

Supported standards:
  - CDP ORS XML Schema v3.x
  - CDP Reporting Guidance 2025/2026

Example:
    >>> generator = ReportGenerator(config)
    >>> report = generator.generate_report(questionnaire_id, org_id,
    ...     format=ReportFormat.PDF, include_scoring=True)
    >>> print(report["file_path"])
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import (
    CDPAppConfig,
    CDPModule,
    MODULE_DEFINITIONS,
    ReportFormat,
    ResponseStatus,
    SCORING_CATEGORY_WEIGHTS,
    ScoringLevel,
)
from .models import (
    Response,
    Submission,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report Section Definitions
# ---------------------------------------------------------------------------

REPORT_SECTIONS = [
    {"id": "cover", "name": "Cover Page", "order": 0},
    {"id": "executive_summary", "name": "Executive Summary", "order": 1},
    {"id": "organization_profile", "name": "Organization Profile", "order": 2},
    {"id": "scoring_overview", "name": "Scoring Overview", "order": 3},
    {"id": "module_responses", "name": "Module Responses", "order": 4},
    {"id": "gap_analysis", "name": "Gap Analysis", "order": 5},
    {"id": "benchmarking", "name": "Benchmarking", "order": 6},
    {"id": "verification_status", "name": "Verification Status", "order": 7},
    {"id": "transition_plan", "name": "Transition Plan", "order": 8},
    {"id": "appendices", "name": "Appendices", "order": 9},
]

# CDP ORS XML namespace
CDP_ORS_NAMESPACE = "http://www.cdp.net/ors/schema/v3"


class ReportGenerator:
    """
    CDP Report Generator -- produces multi-format disclosure reports.

    Generates PDF, Excel, XML (ORS), and JSON reports from questionnaire
    responses.  Supports executive summaries, scoring breakdowns, gap
    analysis inclusion, and submission validation.

    Attributes:
        config: Application configuration.
        _submissions: Submission store keyed by ID.
        _by_questionnaire: Questionnaire ID -> submission IDs.

    Example:
        >>> generator = ReportGenerator(config)
        >>> report = generator.generate_report("q-1", "org-1", ReportFormat.JSON)
    """

    def __init__(self, config: CDPAppConfig) -> None:
        """Initialize the Report Generator."""
        self.config = config
        self._submissions: Dict[str, Submission] = {}
        self._by_questionnaire: Dict[str, List[str]] = {}
        logger.info("ReportGenerator initialized")

    # ------------------------------------------------------------------
    # Report Generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        questionnaire_id: str,
        org_id: str,
        report_format: ReportFormat = ReportFormat.JSON,
        include_scoring: bool = True,
        include_gap_analysis: bool = True,
        include_benchmarking: bool = False,
        include_executive_summary: bool = True,
        responses: Optional[List[Response]] = None,
        scoring_result: Optional[Dict[str, Any]] = None,
        gap_analysis: Optional[Dict[str, Any]] = None,
        benchmark_data: Optional[Dict[str, Any]] = None,
        org_data: Optional[Dict[str, Any]] = None,
        verification_data: Optional[Dict[str, Any]] = None,
        transition_plan_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a CDP disclosure report in the specified format.

        Args:
            questionnaire_id: Questionnaire ID.
            org_id: Organization ID.
            report_format: Output format (PDF, Excel, XML, JSON).
            include_scoring: Include scoring breakdown.
            include_gap_analysis: Include gap analysis section.
            include_benchmarking: Include peer benchmarking.
            include_executive_summary: Include executive summary.
            responses: List of questionnaire responses.
            scoring_result: Pre-computed scoring result.
            gap_analysis: Pre-computed gap analysis.
            benchmark_data: Pre-computed benchmark data.
            org_data: Organization profile data.
            verification_data: Verification status data.
            transition_plan_data: Transition plan data.

        Returns:
            Report generation result with content and metadata.
        """
        start_time = _now()
        responses = responses or []

        # Build report content
        report_content = self._build_report_content(
            questionnaire_id=questionnaire_id,
            org_id=org_id,
            responses=responses,
            scoring_result=scoring_result,
            gap_analysis=gap_analysis,
            benchmark_data=benchmark_data,
            org_data=org_data,
            verification_data=verification_data,
            transition_plan_data=transition_plan_data,
            include_scoring=include_scoring,
            include_gap_analysis=include_gap_analysis,
            include_benchmarking=include_benchmarking,
            include_executive_summary=include_executive_summary,
        )

        # Format-specific rendering
        if report_format == ReportFormat.PDF:
            rendered = self._render_pdf(report_content)
        elif report_format == ReportFormat.EXCEL:
            rendered = self._render_excel(report_content)
        elif report_format == ReportFormat.XML:
            rendered = self._render_xml(report_content)
        else:
            rendered = self._render_json(report_content)

        end_time = _now()
        processing_ms = (end_time - start_time).total_seconds() * 1000

        # Compute provenance hash
        provenance = _sha256(json.dumps(report_content, default=str, sort_keys=True))

        file_ext = report_format.value
        file_name = f"cdp_report_{org_id}_{questionnaire_id[:8]}.{file_ext}"
        file_path = f"{self.config.report_storage_path}{file_name}"

        result = {
            "questionnaire_id": questionnaire_id,
            "org_id": org_id,
            "format": report_format.value,
            "file_name": file_name,
            "file_path": file_path,
            "sections_included": rendered.get("sections", []),
            "total_responses": len(responses),
            "total_pages_estimate": rendered.get("page_count", 0),
            "generated_at": end_time.isoformat(),
            "processing_ms": round(processing_ms, 1),
            "provenance_hash": provenance,
            "content": rendered.get("content"),
        }

        logger.info(
            "Generated %s report for org %s, questionnaire %s: %d responses, %.0fms",
            report_format.value, org_id, questionnaire_id[:8],
            len(responses), processing_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Executive Summary
    # ------------------------------------------------------------------

    def generate_executive_summary(
        self,
        questionnaire_id: str,
        org_id: str,
        scoring_result: Optional[Dict[str, Any]] = None,
        gap_analysis: Optional[Dict[str, Any]] = None,
        verification_data: Optional[Dict[str, Any]] = None,
        org_data: Optional[Dict[str, Any]] = None,
        year: int = 2026,
    ) -> Dict[str, Any]:
        """
        Generate an executive summary of the CDP disclosure.

        Provides a concise overview of scoring, key gaps, verification
        status, and year-over-year progression for C-suite consumption.

        Returns:
            Executive summary data.
        """
        summary: Dict[str, Any] = {
            "org_id": org_id,
            "questionnaire_id": questionnaire_id,
            "year": year,
            "generated_at": _now().isoformat(),
        }

        # Organization overview
        if org_data:
            summary["organization"] = {
                "name": org_data.get("name", "Unknown"),
                "sector": org_data.get("gics_sector", ""),
                "country": org_data.get("country", ""),
            }

        # Scoring overview
        if scoring_result:
            summary["scoring"] = {
                "overall_score": scoring_result.get("overall_score_pct", 0.0),
                "level": scoring_result.get("overall_level", "D-"),
                "band": scoring_result.get("overall_band", "Disclosure"),
                "a_eligible": scoring_result.get("a_eligible", False),
                "completion_pct": scoring_result.get("completion_pct", 0.0),
            }

            # Top 3 strongest and weakest categories
            cat_scores = scoring_result.get("category_scores", [])
            if cat_scores:
                sorted_cats = sorted(cat_scores, key=lambda c: c.get("score_pct", 0), reverse=True)
                summary["strongest_categories"] = [
                    {"id": c.get("category_id"), "name": c.get("category_name"), "score": c.get("score_pct")}
                    for c in sorted_cats[:3]
                ]
                summary["weakest_categories"] = [
                    {"id": c.get("category_id"), "name": c.get("category_name"), "score": c.get("score_pct")}
                    for c in sorted_cats[-3:]
                ]

        # Gap summary
        if gap_analysis:
            summary["gap_summary"] = {
                "total_gaps": gap_analysis.get("total_gaps", 0),
                "critical_gaps": gap_analysis.get("critical_gaps", 0),
                "potential_uplift": gap_analysis.get("total_potential_uplift", 0.0),
                "projected_score": gap_analysis.get("projected_score", 0.0),
            }

        # Verification summary
        if verification_data:
            a_check = verification_data.get("a_level_assessment", {})
            a_reqs = a_check.get("a_requirements", {})
            summary["verification"] = {
                "areq03_met": a_reqs.get("AREQ03", {}).get("met", False),
                "areq04_met": a_reqs.get("AREQ04", {}).get("met", False),
                "completeness_pct": verification_data.get("completeness_pct", 0.0),
            }

        # Key recommendations
        summary["recommendations"] = self._generate_key_recommendations(
            scoring_result, gap_analysis, verification_data,
        )

        return summary

    # ------------------------------------------------------------------
    # Submission Validation
    # ------------------------------------------------------------------

    def validate_submission(
        self,
        questionnaire_id: str,
        org_id: str,
        responses: Optional[List[Response]] = None,
        year: int = 2026,
    ) -> Dict[str, Any]:
        """
        Validate questionnaire completeness for submission.

        Checks all required questions are answered, all responses
        are approved, and no critical validation errors exist.

        Args:
            questionnaire_id: Questionnaire ID.
            org_id: Organization ID.
            responses: List of responses to validate.
            year: Reporting year.

        Returns:
            Validation result with errors and warnings.
        """
        responses = responses or []
        errors: List[str] = []
        warnings: List[str] = []

        # Check overall response count
        total = len(responses)
        answered = sum(1 for r in responses if r.content or r.table_data or r.numeric_value or r.selected_options)
        approved = sum(1 for r in responses if r.status == ResponseStatus.APPROVED)

        if total == 0:
            errors.append("No responses found. Cannot submit an empty questionnaire.")

        # Check unanswered required questions
        unanswered = total - answered
        if unanswered > 0:
            errors.append(f"{unanswered} required question(s) are not yet answered")

        # Check unapproved responses
        unapproved = answered - approved
        if unapproved > 0:
            warnings.append(f"{unapproved} response(s) have not been approved by a reviewer")

        # Check module completeness
        module_question_counts: Dict[str, int] = {}
        module_answered_counts: Dict[str, int] = {}

        for resp in responses:
            mod = resp.module_code.value
            module_question_counts[mod] = module_question_counts.get(mod, 0) + 1
            if resp.content or resp.table_data or resp.numeric_value or resp.selected_options:
                module_answered_counts[mod] = module_answered_counts.get(mod, 0) + 1

        for mod_code, mod_def in MODULE_DEFINITIONS.items():
            if mod_def.get("required", False):
                q_count = module_question_counts.get(mod_code, 0)
                a_count = module_answered_counts.get(mod_code, 0)
                if q_count > 0 and a_count < q_count:
                    pct = round(a_count / q_count * 100, 0)
                    warnings.append(
                        f"Module {mod_code} ({mod_def['name']}) is {pct:.0f}% complete ({a_count}/{q_count})"
                    )

        # Check for draft responses (should be at least IN_REVIEW)
        draft_count = sum(1 for r in responses if r.status == ResponseStatus.DRAFT)
        if draft_count > 0:
            errors.append(f"{draft_count} response(s) are still in draft status")

        # Check for returned responses
        returned_count = sum(1 for r in responses if r.status == ResponseStatus.RETURNED)
        if returned_count > 0:
            errors.append(f"{returned_count} response(s) have been returned and need revision")

        # Deadline check
        deadline = date(year, self.config.submission_deadline_month, self.config.submission_deadline_day)
        days_remaining = (deadline - date.today()).days
        if days_remaining < 0:
            warnings.append(f"Submission deadline has passed ({deadline.isoformat()})")
        elif days_remaining <= 7:
            warnings.append(f"Only {days_remaining} day(s) remaining until submission deadline")

        # Completeness percentage
        completeness_pct = 0.0
        if total > 0:
            completeness_pct = round(answered / total * 100, 1)

        is_valid = len(errors) == 0

        return {
            "questionnaire_id": questionnaire_id,
            "org_id": org_id,
            "year": year,
            "valid": is_valid,
            "completeness_pct": completeness_pct,
            "total_questions": total,
            "answered_questions": answered,
            "approved_questions": approved,
            "errors": errors,
            "warnings": warnings,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "submission_deadline": deadline.isoformat(),
            "days_remaining": max(0, days_remaining),
            "validated_at": _now().isoformat(),
        }

    # ------------------------------------------------------------------
    # CDP ORS XML Export
    # ------------------------------------------------------------------

    def export_xml(
        self,
        questionnaire_id: str,
        org_id: str,
        responses: Optional[List[Response]] = None,
        org_data: Optional[Dict[str, Any]] = None,
        year: int = 2026,
        validate_before_export: bool = True,
    ) -> Dict[str, Any]:
        """
        Export questionnaire in CDP ORS-compatible XML format.

        Args:
            questionnaire_id: Questionnaire ID.
            org_id: Organization ID.
            responses: List of responses.
            org_data: Organization profile data.
            year: Reporting year.
            validate_before_export: Run validation before export.

        Returns:
            XML export result with content and validation status.
        """
        responses = responses or []

        # Optional pre-validation
        validation = None
        if validate_before_export:
            validation = self.validate_submission(
                questionnaire_id, org_id, responses, year,
            )

        # Build XML structure
        xml_content = self._build_xml_content(
            questionnaire_id, org_id, responses, org_data, year,
        )

        provenance = _sha256(xml_content)
        file_name = f"cdp_ors_{org_id}_{year}.xml"
        file_path = f"{self.config.report_storage_path}{file_name}"

        return {
            "questionnaire_id": questionnaire_id,
            "org_id": org_id,
            "year": year,
            "format": "xml",
            "file_name": file_name,
            "file_path": file_path,
            "content": xml_content,
            "response_count": len(responses),
            "validation": validation,
            "provenance_hash": provenance,
            "exported_at": _now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Excel Export
    # ------------------------------------------------------------------

    def export_excel(
        self,
        questionnaire_id: str,
        org_id: str,
        responses: Optional[List[Response]] = None,
        scoring_result: Optional[Dict[str, Any]] = None,
        year: int = 2026,
    ) -> Dict[str, Any]:
        """
        Export questionnaire responses as Excel workbook structure.

        Creates a multi-worksheet structure with one sheet per module
        and summary sheets for scoring and completion.

        Returns:
            Excel export result with worksheet definitions.
        """
        responses = responses or []

        worksheets = self._build_excel_worksheets(
            responses, scoring_result, year,
        )

        file_name = f"cdp_export_{org_id}_{year}.xlsx"
        file_path = f"{self.config.report_storage_path}{file_name}"

        return {
            "questionnaire_id": questionnaire_id,
            "org_id": org_id,
            "year": year,
            "format": "excel",
            "file_name": file_name,
            "file_path": file_path,
            "worksheets": worksheets,
            "worksheet_count": len(worksheets),
            "response_count": len(responses),
            "exported_at": _now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Verification Evidence Package
    # ------------------------------------------------------------------

    def assemble_verification_package(
        self,
        questionnaire_id: str,
        org_id: str,
        verification_records: Optional[List[Dict[str, Any]]] = None,
        evidence_attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Assemble a verification evidence package for submission.

        Collects all verification statements, evidence attachments,
        and metadata into a single package for CDP submission.

        Args:
            questionnaire_id: Questionnaire ID.
            org_id: Organization ID.
            verification_records: List of verification records.
            evidence_attachments: List of evidence file references.

        Returns:
            Verification evidence package.
        """
        verification_records = verification_records or []
        evidence_attachments = evidence_attachments or []

        package: Dict[str, Any] = {
            "package_id": _new_id(),
            "questionnaire_id": questionnaire_id,
            "org_id": org_id,
            "assembled_at": _now().isoformat(),
            "verification_statements": [],
            "evidence_files": [],
            "summary": {
                "total_verification_records": len(verification_records),
                "total_evidence_files": len(evidence_attachments),
                "scopes_covered": set(),
            },
        }

        # Process verification records
        for rec in verification_records:
            package["verification_statements"].append({
                "scope": rec.get("scope", ""),
                "assurance_level": rec.get("assurance_level", ""),
                "coverage_pct": rec.get("coverage_pct", 0),
                "verifier_name": rec.get("verifier_name", ""),
                "verifier_organization": rec.get("verifier_organization", ""),
                "standard": rec.get("verification_standard", ""),
                "statement_date": rec.get("statement_date", ""),
            })
            scope = rec.get("scope", "")
            if scope:
                package["summary"]["scopes_covered"].add(scope)

        # Process evidence attachments
        for att in evidence_attachments:
            package["evidence_files"].append({
                "file_name": att.get("file_name", ""),
                "file_type": att.get("file_type", ""),
                "file_size_bytes": att.get("file_size_bytes", 0),
                "description": att.get("description", ""),
                "uploaded_at": att.get("uploaded_at", ""),
            })

        # Convert set to list for serialization
        package["summary"]["scopes_covered"] = sorted(package["summary"]["scopes_covered"])

        # Provenance hash for the package
        package["provenance_hash"] = _sha256(
            json.dumps(package, default=str, sort_keys=True)
        )

        logger.info(
            "Assembled verification package for org %s: %d records, %d files",
            org_id, len(verification_records), len(evidence_attachments),
        )
        return package

    # ------------------------------------------------------------------
    # Submission Management
    # ------------------------------------------------------------------

    def create_submission(
        self,
        questionnaire_id: str,
        org_id: str,
        year: int,
        submitted_by: Optional[str] = None,
        responses: Optional[List[Response]] = None,
    ) -> Submission:
        """
        Create a submission record after validation.

        Args:
            questionnaire_id: Questionnaire ID.
            org_id: Organization ID.
            year: Reporting year.
            submitted_by: User submitting.
            responses: Responses to validate.

        Returns:
            Created Submission record.
        """
        responses = responses or []

        # Run validation
        validation = self.validate_submission(
            questionnaire_id, org_id, responses, year,
        )

        status = "validated" if validation["valid"] else "draft"

        submission = Submission(
            questionnaire_id=questionnaire_id,
            org_id=org_id,
            year=year,
            status=status,
            completeness_pct=Decimal(str(validation["completeness_pct"])),
            validation_errors=validation["errors"],
            validation_warnings=validation["warnings"],
            submitted_by=submitted_by,
        )

        if validation["valid"]:
            submission.submitted_at = _now()
            submission.status = "submitted"
            submission.confirmation_number = f"CDP-{year}-{_new_id()[:8].upper()}"

        self._submissions[submission.id] = submission
        if questionnaire_id not in self._by_questionnaire:
            self._by_questionnaire[questionnaire_id] = []
        self._by_questionnaire[questionnaire_id].append(submission.id)

        logger.info(
            "Created submission %s for org %s, year %d: status=%s, completeness=%.1f%%",
            submission.id, org_id, year, submission.status,
            float(submission.completeness_pct),
        )
        return submission

    def get_submission(self, submission_id: str) -> Optional[Submission]:
        """Get a submission by ID."""
        return self._submissions.get(submission_id)

    def get_submissions_for_questionnaire(
        self,
        questionnaire_id: str,
    ) -> List[Submission]:
        """Get all submissions for a questionnaire."""
        sub_ids = self._by_questionnaire.get(questionnaire_id, [])
        return [self._submissions[sid] for sid in sub_ids if sid in self._submissions]

    # ------------------------------------------------------------------
    # Internal: Report Content Building
    # ------------------------------------------------------------------

    def _build_report_content(
        self,
        questionnaire_id: str,
        org_id: str,
        responses: List[Response],
        scoring_result: Optional[Dict[str, Any]],
        gap_analysis: Optional[Dict[str, Any]],
        benchmark_data: Optional[Dict[str, Any]],
        org_data: Optional[Dict[str, Any]],
        verification_data: Optional[Dict[str, Any]],
        transition_plan_data: Optional[Dict[str, Any]],
        include_scoring: bool,
        include_gap_analysis: bool,
        include_benchmarking: bool,
        include_executive_summary: bool,
    ) -> Dict[str, Any]:
        """Build the structured report content."""
        content: Dict[str, Any] = {
            "metadata": {
                "questionnaire_id": questionnaire_id,
                "org_id": org_id,
                "generated_at": _now().isoformat(),
                "standard": "CDP Climate Change Questionnaire 2025/2026",
                "version": self.config.version,
            },
            "sections": [],
        }

        # Cover page
        content["cover"] = {
            "title": "CDP Climate Change Disclosure Report",
            "organization": org_data.get("name", "Organization") if org_data else "Organization",
            "year": self.config.default_questionnaire_year,
            "date": date.today().isoformat(),
        }
        content["sections"].append("cover")

        # Executive summary
        if include_executive_summary:
            content["executive_summary"] = self.generate_executive_summary(
                questionnaire_id, org_id,
                scoring_result=scoring_result,
                gap_analysis=gap_analysis,
                verification_data=verification_data,
                org_data=org_data,
            )
            content["sections"].append("executive_summary")

        # Organization profile
        if org_data:
            content["organization_profile"] = org_data
            content["sections"].append("organization_profile")

        # Scoring
        if include_scoring and scoring_result:
            content["scoring"] = scoring_result
            content["sections"].append("scoring_overview")

        # Module responses
        module_responses = self._group_responses_by_module(responses)
        content["module_responses"] = module_responses
        content["sections"].append("module_responses")

        # Gap analysis
        if include_gap_analysis and gap_analysis:
            content["gap_analysis"] = gap_analysis
            content["sections"].append("gap_analysis")

        # Benchmarking
        if include_benchmarking and benchmark_data:
            content["benchmarking"] = benchmark_data
            content["sections"].append("benchmarking")

        # Verification
        if verification_data:
            content["verification"] = verification_data
            content["sections"].append("verification_status")

        # Transition plan
        if transition_plan_data:
            content["transition_plan"] = transition_plan_data
            content["sections"].append("transition_plan")

        return content

    def _group_responses_by_module(
        self,
        responses: List[Response],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group responses by module code for structured output."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for resp in responses:
            mod = resp.module_code.value
            if mod not in grouped:
                grouped[mod] = []

            grouped[mod].append({
                "question_number": resp.question_number,
                "question_id": resp.question_id,
                "status": resp.status.value,
                "content": resp.content,
                "table_data": resp.table_data,
                "numeric_value": float(resp.numeric_value) if resp.numeric_value else None,
                "selected_options": resp.selected_options,
                "is_auto_populated": resp.is_auto_populated,
                "evidence_count": len(resp.evidence),
                "total_score": resp.total_score,
            })

        return grouped

    # ------------------------------------------------------------------
    # Internal: Format Renderers
    # ------------------------------------------------------------------

    def _render_pdf(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render report content to PDF format.

        In production this would use a PDF library (e.g. ReportLab, WeasyPrint).
        Here we return the structured content with PDF metadata.
        """
        page_count = self._estimate_page_count(content)
        return {
            "format": "pdf",
            "page_count": page_count,
            "sections": content.get("sections", []),
            "content": content,
        }

    def _render_excel(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Render report content to Excel worksheet structure."""
        worksheets = self._build_excel_worksheets(
            [], None, self.config.default_questionnaire_year,
        )
        # Add module response data
        module_responses = content.get("module_responses", {})
        for mod_code, responses in module_responses.items():
            mod_name = MODULE_DEFINITIONS.get(mod_code, {}).get("name", mod_code)
            worksheets.append({
                "sheet_name": f"{mod_code} - {mod_name}"[:31],
                "headers": ["Question", "Status", "Content", "Score", "Evidence"],
                "rows": [
                    [
                        r["question_number"],
                        r["status"],
                        (r["content"] or "")[:500],
                        r["total_score"],
                        r["evidence_count"],
                    ]
                    for r in responses
                ],
                "row_count": len(responses),
            })

        return {
            "format": "excel",
            "sections": content.get("sections", []),
            "content": {"worksheets": worksheets},
            "page_count": len(worksheets),
        }

    def _render_xml(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Render report content to XML format."""
        xml_string = self._build_xml_content(
            content["metadata"]["questionnaire_id"],
            content["metadata"]["org_id"],
            [],
            content.get("organization_profile"),
            self.config.default_questionnaire_year,
        )
        return {
            "format": "xml",
            "sections": content.get("sections", []),
            "content": xml_string,
            "page_count": 1,
        }

    def _render_json(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Render report content to JSON format."""
        return {
            "format": "json",
            "sections": content.get("sections", []),
            "content": content,
            "page_count": 1,
        }

    # ------------------------------------------------------------------
    # Internal: XML Builder
    # ------------------------------------------------------------------

    def _build_xml_content(
        self,
        questionnaire_id: str,
        org_id: str,
        responses: List[Response],
        org_data: Optional[Dict[str, Any]],
        year: int,
    ) -> str:
        """
        Build CDP ORS-compatible XML content.

        Produces a simplified XML structure following the CDP ORS schema
        for electronic submission.
        """
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<CDPResponse xmlns="{CDP_ORS_NAMESPACE}" year="{year}">',
            '  <Organization>',
            f'    <OrgId>{org_id}</OrgId>',
        ]

        if org_data:
            lines.append(f'    <Name>{self._xml_escape(org_data.get("name", ""))}</Name>')
            lines.append(f'    <Sector>{self._xml_escape(org_data.get("gics_sector", ""))}</Sector>')
            lines.append(f'    <Country>{self._xml_escape(org_data.get("country", ""))}</Country>')

        lines.append('  </Organization>')
        lines.append(f'  <Questionnaire id="{questionnaire_id}">')

        # Group responses by module
        by_module: Dict[str, List[Response]] = {}
        for resp in responses:
            mod = resp.module_code.value
            if mod not in by_module:
                by_module[mod] = []
            by_module[mod].append(resp)

        for mod_code in sorted(by_module.keys()):
            mod_name = MODULE_DEFINITIONS.get(mod_code, {}).get("name", mod_code)
            lines.append(f'    <Module code="{mod_code}" name="{self._xml_escape(mod_name)}">')

            for resp in by_module[mod_code]:
                lines.append(f'      <Response questionNumber="{resp.question_number}">')
                lines.append(f'        <Status>{resp.status.value}</Status>')

                if resp.content:
                    lines.append(f'        <Content><![CDATA[{resp.content}]]></Content>')
                if resp.numeric_value is not None:
                    lines.append(f'        <NumericValue>{resp.numeric_value}</NumericValue>')
                if resp.selected_options:
                    lines.append('        <SelectedOptions>')
                    for opt in resp.selected_options:
                        lines.append(f'          <Option>{self._xml_escape(opt)}</Option>')
                    lines.append('        </SelectedOptions>')
                if resp.table_data:
                    lines.append('        <TableData>')
                    for row in resp.table_data:
                        lines.append('          <Row>')
                        for col, val in row.items():
                            lines.append(f'            <Cell column="{self._xml_escape(col)}">{self._xml_escape(str(val))}</Cell>')
                        lines.append('          </Row>')
                    lines.append('        </TableData>')

                lines.append('      </Response>')

            lines.append('    </Module>')

        lines.append('  </Questionnaire>')
        lines.append('</CDPResponse>')

        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Internal: Excel Worksheet Builder
    # ------------------------------------------------------------------

    def _build_excel_worksheets(
        self,
        responses: List[Response],
        scoring_result: Optional[Dict[str, Any]],
        year: int,
    ) -> List[Dict[str, Any]]:
        """Build Excel worksheet structure."""
        worksheets: List[Dict[str, Any]] = []

        # Summary worksheet
        worksheets.append({
            "sheet_name": "Summary",
            "headers": ["Metric", "Value"],
            "rows": [
                ["Report Year", year],
                ["Total Responses", len(responses)],
                ["Generated At", _now().isoformat()],
            ],
            "row_count": 3,
        })

        # Scoring worksheet
        if scoring_result:
            cat_rows = []
            for cat in scoring_result.get("category_scores", []):
                cat_rows.append([
                    cat.get("category_id", ""),
                    cat.get("category_name", ""),
                    cat.get("score_pct", 0),
                    cat.get("weight_management", 0),
                    cat.get("level", ""),
                ])
            worksheets.append({
                "sheet_name": "Scoring",
                "headers": ["Category ID", "Category Name", "Score %", "Weight", "Level"],
                "rows": cat_rows,
                "row_count": len(cat_rows),
            })

        return worksheets

    # ------------------------------------------------------------------
    # Internal: Helpers
    # ------------------------------------------------------------------

    def _estimate_page_count(self, content: Dict[str, Any]) -> int:
        """Estimate PDF page count from content."""
        pages = 2  # Cover + summary
        module_responses = content.get("module_responses", {})
        for mod, responses in module_responses.items():
            # Roughly 3 responses per page
            pages += max(1, len(responses) // 3)
        if content.get("scoring"):
            pages += 2
        if content.get("gap_analysis"):
            pages += 2
        if content.get("benchmarking"):
            pages += 1
        if content.get("verification"):
            pages += 1
        if content.get("transition_plan"):
            pages += 1
        return pages

    def _xml_escape(self, text: str) -> str:
        """Escape special characters for XML output."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    def _generate_key_recommendations(
        self,
        scoring_result: Optional[Dict[str, Any]],
        gap_analysis: Optional[Dict[str, Any]],
        verification_data: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate top recommendations for the executive summary."""
        recommendations: List[Dict[str, Any]] = []

        # Scoring-based recommendations
        if scoring_result:
            score = scoring_result.get("overall_score_pct", 0)
            if score < 40:
                recommendations.append({
                    "priority": "critical",
                    "area": "Overall Disclosure",
                    "recommendation": "Complete all required questions to achieve at least C-level scoring",
                })
            elif score < 60:
                recommendations.append({
                    "priority": "high",
                    "area": "Management Level",
                    "recommendation": "Focus on targets, emissions reduction initiatives, and verification to reach B-level",
                })
            elif score < 80:
                recommendations.append({
                    "priority": "medium",
                    "area": "Leadership Level",
                    "recommendation": "Address A-level requirements: public transition plan, complete verification, and SBTi target",
                })

        # Gap-based recommendations
        if gap_analysis:
            critical_gaps = gap_analysis.get("critical_gaps", 0)
            if critical_gaps > 0:
                recommendations.append({
                    "priority": "critical",
                    "area": "Gap Resolution",
                    "recommendation": f"Address {critical_gaps} critical gap(s) to unlock significant score improvement",
                })

        # Verification-based recommendations
        if verification_data:
            a_check = verification_data.get("a_level_assessment", {})
            a_reqs = a_check.get("a_requirements", {})
            if not a_reqs.get("AREQ03", {}).get("met", False):
                recommendations.append({
                    "priority": "high",
                    "area": "Verification",
                    "recommendation": "Obtain third-party verification for 100% of Scope 1 and Scope 2 emissions",
                })
            if not a_reqs.get("AREQ04", {}).get("met", False):
                recommendations.append({
                    "priority": "high",
                    "area": "Scope 3 Verification",
                    "recommendation": "Verify at least 70% of one Scope 3 category for A-level eligibility",
                })

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda r: priority_order.get(r.get("priority", "low"), 4))

        return recommendations[:5]  # Top 5 recommendations
