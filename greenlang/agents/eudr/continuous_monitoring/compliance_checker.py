# -*- coding: utf-8 -*-
"""
Compliance Checker Engine - AGENT-EUDR-033

Automated EUDR compliance verification engine that audits operators
against Article 8 freshness requirements, risk assessment validity,
due diligence statement currency, and data completeness.

Zero-Hallucination:
    - All compliance checks are deterministic rule evaluations
    - Date freshness uses pure datetime arithmetic
    - Score computation uses Decimal arithmetic only

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-033 (GL-EUDR-CM-033)
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import ContinuousMonitoringConfig, get_config
from .models import (
    AGENT_ID,
    EUDR_ARTICLES_MONITORED,
    ActionRecommendation,
    ComplianceAuditRecord,
    ComplianceCheckItem,
    ComplianceStatus,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


class ComplianceChecker:
    """Automated EUDR compliance verification engine.

    Runs comprehensive compliance audits against EUDR article
    requirements, evaluating data freshness, risk assessment
    validity, and due diligence statement currency.

    Example:
        >>> checker = ComplianceChecker()
        >>> audit = await checker.run_compliance_audit(
        ...     operator_id="OP-001",
        ...     operator_data={"dds_date": "2026-01-01", "risk_assessments": []},
        ... )
        >>> assert audit.overall_score >= 0
    """

    def __init__(
        self, config: Optional[ContinuousMonitoringConfig] = None,
    ) -> None:
        """Initialize ComplianceChecker engine."""
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._audits: Dict[str, ComplianceAuditRecord] = {}
        logger.info("ComplianceChecker engine initialized")

    async def run_compliance_audit(
        self,
        operator_id: str,
        operator_data: Dict[str, Any],
    ) -> ComplianceAuditRecord:
        """Run a comprehensive compliance audit.

        Args:
            operator_id: Operator identifier.
            operator_data: Operator compliance data.

        Returns:
            ComplianceAuditRecord with audit results.
        """
        start_time = time.monotonic()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        audit_id = str(uuid.uuid4())

        check_items: List[ComplianceCheckItem] = []

        # Article 8 freshness
        art8_checks = await self.check_article_8_freshness(operator_data)
        check_items.extend(art8_checks)

        # Risk assessment validity
        ra_checks = await self.verify_risk_assessments(operator_data)
        check_items.extend(ra_checks)

        # Due diligence statements
        dds_checks = await self.validate_due_diligence_statements(operator_data)
        check_items.extend(dds_checks)

        # Additional article checks
        additional_checks = self._check_additional_articles(operator_data)
        check_items.extend(additional_checks)

        # Compute scores
        passed = sum(1 for c in check_items if c.status == ComplianceStatus.COMPLIANT)
        failed = sum(1 for c in check_items if c.status == ComplianceStatus.NON_COMPLIANT)
        total = len(check_items)

        overall_score = (
            (Decimal(str(passed)) / Decimal(str(total))) * Decimal("100")
            if total > 0 else Decimal("0")
        ).quantize(Decimal("0.01"))

        # Determine overall status
        if overall_score >= self.config.compliance_pass_threshold:
            overall_status = ComplianceStatus.COMPLIANT
        elif failed == 0:
            overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT

        # Determine article-level statuses
        art8_status = self._aggregate_check_status(
            [c for c in check_items if "Article 8" in c.article_reference]
        )
        ra_status = self._aggregate_check_status(
            [c for c in check_items if "risk_assessment" in c.check_id]
        )
        dds_status = self._aggregate_check_status(
            [c for c in check_items if "due_diligence" in c.check_id]
        )

        # Build recommendations
        recommendations = self._build_recommendations(check_items)

        next_audit = now + timedelta(days=self.config.compliance_audit_interval_days)

        record = ComplianceAuditRecord(
            audit_id=audit_id,
            operator_id=operator_id,
            compliance_status=overall_status,
            overall_score=overall_score,
            checks_passed=passed,
            checks_failed=failed,
            checks_total=total,
            check_items=check_items,
            article_8_status=art8_status,
            risk_assessment_status=ra_status,
            due_diligence_status=dds_status,
            recommendations=recommendations,
            audited_at=now,
            next_audit_date=next_audit,
        )

        record.provenance_hash = self._provenance.compute_hash({
            "audit_id": audit_id,
            "operator_id": operator_id,
            "overall_score": str(overall_score),
            "checks_total": total,
            "created_at": now.isoformat(),
        })

        self._provenance.record(
            entity_type="compliance_audit",
            action="audit",
            entity_id=audit_id,
            actor=AGENT_ID,
            metadata={
                "operator_id": operator_id,
                "score": str(overall_score),
                "passed": passed,
                "failed": failed,
            },
        )

        self._audits[audit_id] = record
        elapsed = time.monotonic() - start_time
        logger.info(
            "Compliance audit %s: score=%s, %d/%d passed (%.3fs)",
            audit_id, overall_score, passed, total, elapsed,
        )
        return record

    async def check_article_8_freshness(
        self,
        operator_data: Dict[str, Any],
    ) -> List[ComplianceCheckItem]:
        """Check Article 8 data freshness requirements.

        Article 8 requires operators to keep due diligence information
        current and updated at regular intervals.

        Args:
            operator_data: Operator compliance data.

        Returns:
            List of Article 8 compliance check items.
        """
        checks: List[ComplianceCheckItem] = []
        now = datetime.now(timezone.utc).replace(microsecond=0)
        max_age_days = self.config.article_8_freshness_max_days

        # Check DDS date
        dds_date_str = operator_data.get("dds_date")
        if dds_date_str:
            try:
                dds_date = self._parse_date(dds_date_str)
                age_days = (now - dds_date).days
                is_fresh = age_days <= max_age_days

                checks.append(ComplianceCheckItem(
                    check_id="art8_dds_freshness",
                    article_reference="Article 8 - Due Diligence Statement Freshness",
                    description=f"DDS last updated {age_days} days ago (max {max_age_days})",
                    status=ComplianceStatus.COMPLIANT if is_fresh else ComplianceStatus.NON_COMPLIANT,
                    details={"age_days": age_days, "max_days": max_age_days, "dds_date": str(dds_date)},
                ))
            except (ValueError, TypeError):
                checks.append(ComplianceCheckItem(
                    check_id="art8_dds_freshness",
                    article_reference="Article 8 - Due Diligence Statement Freshness",
                    description="Unable to parse DDS date",
                    status=ComplianceStatus.NON_COMPLIANT,
                    details={"error": "Invalid date format"},
                ))
        else:
            checks.append(ComplianceCheckItem(
                check_id="art8_dds_freshness",
                article_reference="Article 8 - Due Diligence Statement Freshness",
                description="No DDS date provided",
                status=ComplianceStatus.NON_COMPLIANT,
                details={"error": "Missing dds_date"},
            ))

        # Check supply chain data freshness
        sc_date_str = operator_data.get("supply_chain_last_updated")
        if sc_date_str:
            try:
                sc_date = self._parse_date(sc_date_str)
                age_days = (now - sc_date).days
                is_fresh = age_days <= max_age_days

                checks.append(ComplianceCheckItem(
                    check_id="art8_supply_chain_freshness",
                    article_reference="Article 8 - Supply Chain Data Freshness",
                    description=f"Supply chain data {age_days} days old (max {max_age_days})",
                    status=ComplianceStatus.COMPLIANT if is_fresh else ComplianceStatus.NON_COMPLIANT,
                    details={"age_days": age_days, "max_days": max_age_days},
                ))
            except (ValueError, TypeError):
                checks.append(ComplianceCheckItem(
                    check_id="art8_supply_chain_freshness",
                    article_reference="Article 8 - Supply Chain Data Freshness",
                    description="Unable to parse supply chain update date",
                    status=ComplianceStatus.PENDING_REVIEW,
                ))

        return checks

    async def verify_risk_assessments(
        self,
        operator_data: Dict[str, Any],
    ) -> List[ComplianceCheckItem]:
        """Verify risk assessment validity and currency.

        Args:
            operator_data: Operator compliance data.

        Returns:
            List of risk assessment compliance check items.
        """
        checks: List[ComplianceCheckItem] = []
        now = datetime.now(timezone.utc).replace(microsecond=0)
        validity_days = self.config.risk_assessment_validity_days

        risk_assessments = operator_data.get("risk_assessments", [])

        if not risk_assessments:
            checks.append(ComplianceCheckItem(
                check_id="risk_assessment_existence",
                article_reference="Article 10 - Risk Assessment Requirement",
                description="No risk assessments found",
                status=ComplianceStatus.NON_COMPLIANT,
                details={"error": "No risk assessments on file"},
            ))
            return checks

        for i, ra in enumerate(risk_assessments):
            ra_id = ra.get("assessment_id", f"ra_{i}")
            ra_date_str = ra.get("assessment_date")
            ra_scope = ra.get("scope", "unknown")

            if ra_date_str:
                try:
                    ra_date = self._parse_date(ra_date_str)
                    age_days = (now - ra_date).days
                    is_valid = age_days <= validity_days

                    checks.append(ComplianceCheckItem(
                        check_id=f"risk_assessment_{ra_id}",
                        article_reference="Article 10 - Risk Assessment Validity",
                        description=f"Risk assessment {ra_id} ({ra_scope}): {age_days} days old (max {validity_days})",
                        status=ComplianceStatus.COMPLIANT if is_valid else ComplianceStatus.EXPIRED,
                        details={
                            "assessment_id": ra_id,
                            "age_days": age_days,
                            "max_days": validity_days,
                            "scope": ra_scope,
                        },
                    ))
                except (ValueError, TypeError):
                    checks.append(ComplianceCheckItem(
                        check_id=f"risk_assessment_{ra_id}",
                        article_reference="Article 10 - Risk Assessment Validity",
                        description=f"Unable to parse date for assessment {ra_id}",
                        status=ComplianceStatus.PENDING_REVIEW,
                    ))
            else:
                checks.append(ComplianceCheckItem(
                    check_id=f"risk_assessment_{ra_id}",
                    article_reference="Article 10 - Risk Assessment Validity",
                    description=f"No date for assessment {ra_id}",
                    status=ComplianceStatus.PENDING_REVIEW,
                ))

        return checks

    async def validate_due_diligence_statements(
        self,
        operator_data: Dict[str, Any],
    ) -> List[ComplianceCheckItem]:
        """Validate due diligence statements for currency.

        Args:
            operator_data: Operator compliance data.

        Returns:
            List of due diligence compliance check items.
        """
        checks: List[ComplianceCheckItem] = []
        now = datetime.now(timezone.utc).replace(microsecond=0)
        max_age = self.config.due_diligence_statement_max_age_days

        statements = operator_data.get("due_diligence_statements", [])

        if not statements:
            checks.append(ComplianceCheckItem(
                check_id="due_diligence_existence",
                article_reference="Article 4 - Due Diligence Statement Requirement",
                description="No due diligence statements found",
                status=ComplianceStatus.NON_COMPLIANT,
                details={"error": "No DDS on file"},
            ))
            return checks

        for i, dds in enumerate(statements):
            dds_id = dds.get("statement_id", f"dds_{i}")
            dds_date_str = dds.get("statement_date")
            commodity = dds.get("commodity", "unknown")

            if dds_date_str:
                try:
                    dds_date = self._parse_date(dds_date_str)
                    age_days = (now - dds_date).days
                    is_valid = age_days <= max_age

                    checks.append(ComplianceCheckItem(
                        check_id=f"due_diligence_{dds_id}",
                        article_reference="Article 4 - Due Diligence Statement Currency",
                        description=f"DDS {dds_id} ({commodity}): {age_days} days old (max {max_age})",
                        status=ComplianceStatus.COMPLIANT if is_valid else ComplianceStatus.EXPIRED,
                        details={
                            "statement_id": dds_id,
                            "age_days": age_days,
                            "max_days": max_age,
                            "commodity": commodity,
                        },
                    ))
                except (ValueError, TypeError):
                    checks.append(ComplianceCheckItem(
                        check_id=f"due_diligence_{dds_id}",
                        article_reference="Article 4 - Due Diligence Statement Currency",
                        description=f"Unable to parse date for DDS {dds_id}",
                        status=ComplianceStatus.PENDING_REVIEW,
                    ))
            else:
                checks.append(ComplianceCheckItem(
                    check_id=f"due_diligence_{dds_id}",
                    article_reference="Article 4 - Due Diligence Statement Currency",
                    description=f"No date for DDS {dds_id}",
                    status=ComplianceStatus.PENDING_REVIEW,
                ))

            # Check completeness
            required_fields = ["commodity", "origin_country", "supplier_info"]
            missing = [f for f in required_fields if not dds.get(f)]
            if missing:
                checks.append(ComplianceCheckItem(
                    check_id=f"due_diligence_{dds_id}_completeness",
                    article_reference="Article 4 - Due Diligence Statement Completeness",
                    description=f"DDS {dds_id} missing fields: {', '.join(missing)}",
                    status=ComplianceStatus.NON_COMPLIANT,
                    details={"missing_fields": missing},
                ))
            else:
                checks.append(ComplianceCheckItem(
                    check_id=f"due_diligence_{dds_id}_completeness",
                    article_reference="Article 4 - Due Diligence Statement Completeness",
                    description=f"DDS {dds_id} has all required fields",
                    status=ComplianceStatus.COMPLIANT,
                ))

        return checks

    def _check_additional_articles(
        self, operator_data: Dict[str, Any],
    ) -> List[ComplianceCheckItem]:
        """Check additional EUDR article compliance."""
        checks: List[ComplianceCheckItem] = []

        # Article 12 - Record keeping
        retention_years = operator_data.get("retention_years", 0)
        checks.append(ComplianceCheckItem(
            check_id="art12_record_keeping",
            article_reference="Article 12 - Record Keeping",
            description=f"Record retention: {retention_years} years (minimum 5)",
            status=(
                ComplianceStatus.COMPLIANT if retention_years >= 5
                else ComplianceStatus.NON_COMPLIANT
            ),
            details={"retention_years": retention_years, "minimum": 5},
        ))

        # Article 14 - Competent authority notification
        has_ca_registration = operator_data.get("competent_authority_registered", False)
        checks.append(ComplianceCheckItem(
            check_id="art14_ca_registration",
            article_reference="Article 14 - Competent Authority Registration",
            description="Competent authority registration status",
            status=(
                ComplianceStatus.COMPLIANT if has_ca_registration
                else ComplianceStatus.NON_COMPLIANT
            ),
            details={"registered": has_ca_registration},
        ))

        return checks

    @staticmethod
    def _aggregate_check_status(checks: List[ComplianceCheckItem]) -> ComplianceStatus:
        """Aggregate multiple check statuses into one."""
        if not checks:
            return ComplianceStatus.PENDING_REVIEW

        statuses = [c.status for c in checks]
        if all(s == ComplianceStatus.COMPLIANT for s in statuses):
            return ComplianceStatus.COMPLIANT
        if any(s == ComplianceStatus.NON_COMPLIANT for s in statuses):
            return ComplianceStatus.NON_COMPLIANT
        if any(s == ComplianceStatus.EXPIRED for s in statuses):
            return ComplianceStatus.EXPIRED
        return ComplianceStatus.PARTIALLY_COMPLIANT

    @staticmethod
    def _build_recommendations(checks: List[ComplianceCheckItem]) -> List[ActionRecommendation]:
        """Build recommendations from failed checks."""
        recommendations: List[ActionRecommendation] = []
        failed = [c for c in checks if c.status in (
            ComplianceStatus.NON_COMPLIANT, ComplianceStatus.EXPIRED,
        )]

        for check in failed:
            priority = "critical" if "Article 4" in check.article_reference else "high"
            recommendations.append(ActionRecommendation(
                action=f"Address: {check.description}",
                priority=priority,
                deadline_days=14 if priority == "critical" else 30,
                category="compliance",
            ))

        return recommendations

    @staticmethod
    def _parse_date(date_val: Any) -> datetime:
        """Parse a date string or datetime to timezone-aware datetime."""
        if isinstance(date_val, datetime):
            if date_val.tzinfo is None:
                return date_val.replace(tzinfo=timezone.utc)
            return date_val
        date_str = str(date_val).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(date_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    async def get_audit(self, audit_id: str) -> Optional[ComplianceAuditRecord]:
        """Retrieve a compliance audit record by ID."""
        return self._audits.get(audit_id)

    async def list_audits(
        self,
        operator_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ComplianceAuditRecord]:
        """List compliance audit records with optional filters."""
        results = list(self._audits.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        if status:
            results = [r for r in results if r.compliance_status.value == status]
        return results

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "ComplianceChecker",
            "status": "healthy",
            "audit_count": len(self._audits),
        }
