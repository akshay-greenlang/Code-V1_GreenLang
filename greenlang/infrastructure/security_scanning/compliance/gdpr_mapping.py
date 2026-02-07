# -*- coding: utf-8 -*-
"""
GDPR Compliance Mapping - SEC-007

Maps security findings to GDPR (General Data Protection Regulation) technical
controls. Focuses on Article 25 (Data Protection by Design) and Article 32
(Security of Processing) requirements.

GDPR Articles Covered:
    - Article 25: Data protection by design and by default
    - Article 32: Security of processing
    - Article 33: Notification of personal data breach
    - Article 35: Data protection impact assessment

Example:
    >>> gdpr = GDPRCompliance()
    >>> results = await gdpr.check_controls(scan_results)
    >>> report = gdpr.generate_report(results)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.security_scanning.compliance.base import (
    ComplianceFramework,
    ComplianceReport,
    ControlCategory,
    ControlDefinition,
    ControlResult,
    ControlStatus,
    FindingControlMapping,
    FrameworkType,
)

logger = logging.getLogger(__name__)


class GDPRCompliance(ComplianceFramework):
    """GDPR compliance framework implementation.

    Maps security scanning findings to GDPR technical control requirements.
    Focuses on security-related articles that can be verified through
    automated scanning.

    Attributes:
        framework_type: Always FrameworkType.GDPR.
        controls: Dictionary of GDPR control definitions.
        finding_mappings: Mappings from security findings to controls.

    Example:
        >>> gdpr = GDPRCompliance()
        >>> results = await gdpr.check_controls(scan_findings)
        >>> report = gdpr.generate_report(results)
    """

    def __init__(self) -> None:
        """Initialize GDPR compliance framework."""
        super().__init__(FrameworkType.GDPR)

    def _initialize_controls(self) -> None:
        """Initialize GDPR technical control definitions."""
        self.controls = {
            # Article 25 - Data protection by design and by default
            "Art.25.1": ControlDefinition(
                control_id="Art.25.1",
                name="Data Protection by Design",
                description=(
                    "Taking into account the state of the art, the cost of implementation "
                    "and the nature, scope, context and purposes of processing as well as "
                    "the risks of varying likelihood and severity for rights and freedoms "
                    "of natural persons posed by the processing, the controller shall "
                    "implement appropriate technical and organisational measures designed "
                    "to implement data-protection principles."
                ),
                category=ControlCategory.DATA_PROTECTION,
                requirements=[
                    "Implement data protection principles in design",
                    "Minimize personal data collection",
                    "Implement pseudonymization where appropriate",
                    "Secure personal data processing",
                ],
                automated_checks=[
                    "pii_scan",
                    "sast_scan",
                    "encryption_verification",
                ],
                evidence_types=[
                    "pii_scan_results",
                    "privacy_design_documentation",
                    "data_minimization_evidence",
                ],
            ),
            "Art.25.2": ControlDefinition(
                control_id="Art.25.2",
                name="Data Protection by Default",
                description=(
                    "The controller shall implement appropriate technical and "
                    "organisational measures for ensuring that, by default, only "
                    "personal data which are necessary for each specific purpose "
                    "of the processing are processed."
                ),
                category=ControlCategory.DATA_PROTECTION,
                requirements=[
                    "Default to minimal data collection",
                    "Limit data retention by default",
                    "Restrict data accessibility by default",
                ],
                automated_checks=[
                    "pii_scan",
                    "configuration_audit",
                ],
                evidence_types=[
                    "default_settings_documentation",
                    "access_control_matrix",
                ],
            ),
            # Article 32 - Security of processing
            "Art.32.1.a": ControlDefinition(
                control_id="Art.32.1.a",
                name="Pseudonymization and Encryption",
                description=(
                    "The ability to ensure the ongoing confidentiality, integrity, "
                    "availability and resilience of processing systems and services, "
                    "including the pseudonymization and encryption of personal data."
                ),
                category=ControlCategory.ENCRYPTION,
                requirements=[
                    "Encrypt personal data at rest",
                    "Encrypt personal data in transit",
                    "Implement pseudonymization techniques",
                    "Protect encryption keys",
                ],
                automated_checks=[
                    "encryption_verification",
                    "tls_configuration_scan",
                    "iac_security_scan",
                ],
                evidence_types=[
                    "encryption_configuration",
                    "key_management_policy",
                    "tls_certificate_inventory",
                ],
            ),
            "Art.32.1.b": ControlDefinition(
                control_id="Art.32.1.b",
                name="System Confidentiality and Integrity",
                description=(
                    "The ability to ensure the ongoing confidentiality, integrity, "
                    "availability and resilience of processing systems and services."
                ),
                category=ControlCategory.VULNERABILITY_MANAGEMENT,
                requirements=[
                    "Protect against unauthorized access",
                    "Maintain system integrity",
                    "Ensure system availability",
                    "Build resilient systems",
                ],
                automated_checks=[
                    "vulnerability_scan",
                    "dependency_scan",
                    "container_scan",
                    "sast_scan",
                ],
                evidence_types=[
                    "vulnerability_report",
                    "availability_metrics",
                    "incident_history",
                ],
            ),
            "Art.32.1.c": ControlDefinition(
                control_id="Art.32.1.c",
                name="Data Restoration Capability",
                description=(
                    "The ability to restore the availability and access to personal "
                    "data in a timely manner in the event of a physical or technical "
                    "incident."
                ),
                category=ControlCategory.INCIDENT_RESPONSE,
                requirements=[
                    "Implement backup procedures",
                    "Test restoration capabilities",
                    "Define recovery time objectives",
                    "Document disaster recovery procedures",
                ],
                automated_checks=[
                    "backup_verification",
                    "dr_configuration_audit",
                ],
                evidence_types=[
                    "backup_logs",
                    "restoration_test_records",
                    "dr_plan_documentation",
                ],
            ),
            "Art.32.1.d": ControlDefinition(
                control_id="Art.32.1.d",
                name="Security Testing and Assessment",
                description=(
                    "A process for regularly testing, assessing and evaluating the "
                    "effectiveness of technical and organisational measures for ensuring "
                    "the security of the processing."
                ),
                category=ControlCategory.LOGGING_MONITORING,
                requirements=[
                    "Conduct regular security testing",
                    "Assess control effectiveness",
                    "Perform penetration testing",
                    "Review security measures",
                ],
                automated_checks=[
                    "vulnerability_scan",
                    "dast_scan",
                    "security_assessment_audit",
                ],
                evidence_types=[
                    "security_test_reports",
                    "penetration_test_results",
                    "assessment_schedule",
                ],
            ),
            # Article 32.2 - Risk assessment
            "Art.32.2": ControlDefinition(
                control_id="Art.32.2",
                name="Risk-Based Security Measures",
                description=(
                    "In assessing the appropriate level of security account shall be "
                    "taken of the risks that are presented by processing, in particular "
                    "from accidental or unlawful destruction, loss, alteration, "
                    "unauthorised disclosure of, or access to personal data."
                ),
                category=ControlCategory.VULNERABILITY_MANAGEMENT,
                requirements=[
                    "Assess processing risks",
                    "Implement appropriate security measures",
                    "Protect against data breaches",
                    "Prevent unauthorized access",
                ],
                automated_checks=[
                    "vulnerability_scan",
                    "secret_detection",
                    "access_control_review",
                ],
                evidence_types=[
                    "risk_assessment_documentation",
                    "security_measure_inventory",
                ],
            ),
            # Article 33 - Breach notification
            "Art.33": ControlDefinition(
                control_id="Art.33",
                name="Breach Detection and Notification",
                description=(
                    "In the case of a personal data breach, the controller shall "
                    "without undue delay and, where feasible, not later than 72 hours "
                    "after having become aware of it, notify the personal data breach "
                    "to the supervisory authority."
                ),
                category=ControlCategory.INCIDENT_RESPONSE,
                requirements=[
                    "Detect data breaches promptly",
                    "Assess breach severity",
                    "Maintain breach notification procedures",
                    "Document breach incidents",
                ],
                automated_checks=[
                    "secret_detection",
                    "pii_scan",
                    "security_monitoring_audit",
                ],
                evidence_types=[
                    "breach_detection_logs",
                    "incident_response_procedures",
                    "notification_records",
                ],
            ),
            # Article 35 - DPIA
            "Art.35": ControlDefinition(
                control_id="Art.35",
                name="Data Protection Impact Assessment",
                description=(
                    "Where a type of processing is likely to result in a high risk to "
                    "the rights and freedoms of natural persons, the controller shall "
                    "carry out an assessment of the impact of the envisaged processing "
                    "operations on the protection of personal data."
                ),
                category=ControlCategory.DATA_PROTECTION,
                requirements=[
                    "Identify high-risk processing",
                    "Conduct impact assessments",
                    "Evaluate processing necessity",
                    "Assess risks to data subjects",
                ],
                automated_checks=[
                    "pii_scan",
                    "data_classification_audit",
                ],
                evidence_types=[
                    "dpia_documentation",
                    "risk_mitigation_measures",
                ],
            ),
        }

    def _initialize_mappings(self) -> None:
        """Initialize finding-to-control mappings for GDPR."""
        self.finding_mappings = [
            # PII exposure is critical for GDPR
            FindingControlMapping(
                finding_type="pii_exposure",
                scanner="pii",
                control_ids=[
                    "Art.25.1", "Art.25.2", "Art.32.1.a",
                    "Art.32.2", "Art.33", "Art.35"
                ],
                severity_weight=2.0,  # High weight for PII findings
            ),
            # Secret detection
            FindingControlMapping(
                finding_type="secret",
                scanner="gitleaks",
                control_ids=["Art.32.1.a", "Art.32.1.b", "Art.32.2", "Art.33"],
                severity_weight=1.5,
            ),
            FindingControlMapping(
                finding_type="secret",
                scanner="trufflehog",
                control_ids=["Art.32.1.a", "Art.32.1.b", "Art.32.2", "Art.33"],
                severity_weight=1.5,
            ),
            # Code vulnerabilities
            FindingControlMapping(
                finding_type="code_vulnerability",
                scanner="bandit",
                control_ids=["Art.25.1", "Art.32.1.b", "Art.32.1.d"],
                severity_weight=1.0,
            ),
            FindingControlMapping(
                finding_type="code_vulnerability",
                scanner="semgrep",
                control_ids=["Art.25.1", "Art.32.1.b", "Art.32.1.d"],
                severity_weight=1.0,
            ),
            # Dependency vulnerabilities
            FindingControlMapping(
                finding_type="dependency_vulnerability",
                scanner="trivy",
                control_ids=["Art.32.1.b", "Art.32.1.d", "Art.32.2"],
                severity_weight=1.0,
            ),
            # IaC misconfigurations
            FindingControlMapping(
                finding_type="iac_misconfiguration",
                scanner="tfsec",
                control_ids=["Art.32.1.a", "Art.32.1.b"],
                severity_weight=0.8,
            ),
            FindingControlMapping(
                finding_type="iac_misconfiguration",
                scanner="checkov",
                control_ids=["Art.32.1.a", "Art.32.1.b"],
                severity_weight=0.8,
            ),
            # Container vulnerabilities
            FindingControlMapping(
                finding_type="container_vulnerability",
                scanner="trivy-container",
                control_ids=["Art.32.1.b", "Art.32.2"],
                severity_weight=1.0,
            ),
            # DAST findings
            FindingControlMapping(
                finding_type="web_vulnerability",
                scanner="zap",
                control_ids=["Art.32.1.b", "Art.32.1.d", "Art.32.2"],
                severity_weight=1.2,
            ),
        ]

    async def check_controls(
        self,
        scan_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ControlResult]:
        """Check GDPR controls against scan results.

        Args:
            scan_results: List of security scan findings.
            context: Additional context for evaluation.

        Returns:
            List of ControlResult for each GDPR control.
        """
        context = context or {}
        results: List[ControlResult] = []

        # Group findings by control
        control_findings: Dict[str, List[Dict[str, Any]]] = {
            ctrl_id: [] for ctrl_id in self.controls.keys()
        }

        # Identify PII-related findings with higher weight
        pii_findings = [
            f for f in scan_results
            if f.get("type") == "pii_exposure"
            or f.get("scanner") == "pii"
            or "pii" in str(f.get("rule_id", "")).lower()
        ]

        for finding in scan_results:
            scanner = finding.get("scanner", finding.get("tool", "unknown"))
            finding_type = finding.get("type", finding.get("finding_type", "unknown"))

            for mapping in self.finding_mappings:
                if mapping.scanner == scanner or mapping.finding_type == finding_type:
                    for ctrl_id in mapping.control_ids:
                        if ctrl_id in control_findings:
                            control_findings[ctrl_id].append(finding)

        # Evaluate each control
        for ctrl_id, control in self.controls.items():
            findings = control_findings.get(ctrl_id, [])
            result = await self._evaluate_control(
                control, findings, context, pii_findings
            )
            results.append(result)

        logger.info(
            "GDPR control check complete: %d controls evaluated, %d passed",
            len(results),
            sum(1 for r in results if r.passed),
        )

        return results

    async def _evaluate_control(
        self,
        control: ControlDefinition,
        findings: List[Dict[str, Any]],
        context: Dict[str, Any],
        pii_findings: List[Dict[str, Any]],
    ) -> ControlResult:
        """Evaluate a single GDPR control.

        Args:
            control: The control definition.
            findings: Findings related to this control.
            context: Additional evaluation context.
            pii_findings: PII-specific findings for heightened scrutiny.

        Returns:
            ControlResult for the control.
        """
        # Calculate score with PII weighting
        score = self._calculate_control_score(control.control_id, findings)

        # PII findings have extra impact on data protection controls
        if pii_findings and control.category == ControlCategory.DATA_PROTECTION:
            pii_penalty = min(len(pii_findings) * 0.1, 0.5)
            score = max(0.0, score - pii_penalty)

        status = self._determine_control_status(score, findings)

        # Check for critical PII exposure
        pii_critical = [
            f for f in findings
            if f.get("type") == "pii_exposure"
            and f.get("severity", "").upper() in ("CRITICAL", "HIGH")
        ]
        if pii_critical:
            status = ControlStatus.FAIL

        recommendations = self._generate_gdpr_recommendations(
            control.control_id, findings, pii_findings
        )

        evidence: List[Dict[str, Any]] = []
        if findings:
            evidence.append({
                "type": "scan_findings",
                "count": len(findings),
                "pii_related": sum(
                    1 for f in findings if f.get("type") == "pii_exposure"
                ),
                "severities": self._count_by_severity(findings),
            })

        for evidence_type in control.evidence_types:
            if evidence_type in context:
                evidence.append({
                    "type": evidence_type,
                    "data": context[evidence_type],
                })

        return ControlResult(
            control_id=control.control_id,
            status=status,
            score=score,
            findings=findings,
            evidence=evidence,
            recommendations=recommendations,
            details={
                "control_name": control.name,
                "category": control.category.value,
                "findings_count": len(findings),
                "pii_findings_count": len([
                    f for f in findings if f.get("type") == "pii_exposure"
                ]),
                "gdpr_article": control.control_id,
            },
        )

    def _count_by_severity(
        self, findings: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Count findings by severity level."""
        counts: Dict[str, int] = {
            "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0
        }
        for finding in findings:
            severity = finding.get("severity", "LOW").upper()
            if severity in counts:
                counts[severity] += 1
        return counts

    def _generate_gdpr_recommendations(
        self,
        control_id: str,
        findings: List[Dict[str, Any]],
        pii_findings: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate GDPR-specific recommendations.

        Args:
            control_id: The control being evaluated.
            findings: Findings related to the control.
            pii_findings: PII-specific findings.

        Returns:
            List of recommendation strings.
        """
        recommendations = self._generate_recommendations(control_id, findings)

        # Add GDPR-specific recommendations for PII findings
        if pii_findings:
            recommendations.insert(
                0,
                f"URGENT: {len(pii_findings)} potential PII exposures detected. "
                "Review immediately for GDPR compliance and potential breach notification obligations."
            )

        if "Art.25" in control_id and findings:
            recommendations.append(
                "Review data protection by design principles. Consider implementing "
                "additional data minimization and pseudonymization measures."
            )

        if "Art.32" in control_id and findings:
            recommendations.append(
                "Document security measures and their appropriateness to the risks "
                "as required by Article 32(2)."
            )

        if "Art.33" in control_id and findings:
            recommendations.append(
                "Ensure breach detection capabilities are adequate for the 72-hour "
                "notification requirement."
            )

        return recommendations

    def generate_report(
        self,
        results: List[ControlResult],
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> ComplianceReport:
        """Generate GDPR compliance report.

        Args:
            results: List of control check results.
            period_start: Start of the assessment period.
            period_end: End of the assessment period.

        Returns:
            Complete GDPR ComplianceReport.
        """
        report = ComplianceReport(
            framework=FrameworkType.GDPR,
            period_start=period_start or datetime.now(timezone.utc),
            period_end=period_end or datetime.now(timezone.utc),
            control_results=results,
        )

        report.calculate_score()
        report.calculate_status()

        # Aggregate findings
        report.findings_by_severity = {
            "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0
        }
        pii_finding_count = 0
        for result in results:
            for finding in result.findings:
                severity = finding.get("severity", "LOW").upper()
                if severity in report.findings_by_severity:
                    report.findings_by_severity[severity] += 1
                if finding.get("type") == "pii_exposure":
                    pii_finding_count += 1

        report.gaps = self._identify_gaps(results)
        report.summary = self._generate_summary(report, pii_finding_count)

        all_recommendations: List[str] = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        report.recommendations = list(set(all_recommendations))[:10]

        report.metadata = {
            "framework_version": "GDPR (EU 2016/679)",
            "articles_covered": ["25", "32", "33", "35"],
            "controls_evaluated": len(results),
            "assessment_type": "automated",
            "pii_findings_detected": pii_finding_count,
            "dpo_notification_required": pii_finding_count > 0,
        }

        logger.info(
            "GDPR report generated: id=%s status=%s score=%.2f pii_findings=%d",
            report.report_id,
            report.overall_status.value,
            report.overall_score,
            pii_finding_count,
        )

        return report

    def _identify_gaps(
        self, results: List[ControlResult]
    ) -> List[Dict[str, Any]]:
        """Identify compliance gaps from control results."""
        gaps: List[Dict[str, Any]] = []

        for result in results:
            if result.status in (ControlStatus.FAIL, ControlStatus.PARTIAL):
                control = self.get_control(result.control_id)
                if control:
                    gaps.append({
                        "control_id": result.control_id,
                        "control_name": control.name,
                        "status": result.status.value,
                        "score": result.score,
                        "category": control.category.value,
                        "findings_count": len(result.findings),
                        "pii_related": result.details.get("pii_findings_count", 0) > 0,
                        "priority": "CRITICAL" if result.details.get("pii_findings_count", 0) > 0 else (
                            "HIGH" if result.status == ControlStatus.FAIL else "MEDIUM"
                        ),
                        "remediation_guidance": result.recommendations,
                    })

        # Sort by priority (PII issues first)
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        gaps.sort(key=lambda g: (priority_order.get(g["priority"], 3), -g["findings_count"]))

        return gaps

    def _generate_summary(
        self, report: ComplianceReport, pii_finding_count: int
    ) -> str:
        """Generate executive summary for GDPR report."""
        passed = len(report.passed_controls)
        failed = len(report.failed_controls)
        partial = len(report.partial_controls)
        total = len(report.control_results)

        summary_parts = [
            f"GDPR Technical Compliance Assessment Summary",
            f"",
            f"Assessment Period: {report.period_start.strftime('%Y-%m-%d') if report.period_start else 'N/A'} "
            f"to {report.period_end.strftime('%Y-%m-%d') if report.period_end else 'N/A'}",
            f"",
            f"Overall Status: {report.overall_status.value}",
            f"Overall Score: {report.overall_score:.1%}",
            f"",
        ]

        if pii_finding_count > 0:
            summary_parts.extend([
                f"** ATTENTION: {pii_finding_count} potential PII exposures detected **",
                f"Review required for Article 33 breach notification obligations.",
                f"",
            ])

        summary_parts.extend([
            f"Control Results:",
            f"  - Passed: {passed}/{total} ({passed/total*100:.1f}%)" if total else "  - No controls evaluated",
            f"  - Failed: {failed}/{total}" if total else "",
            f"  - Partial: {partial}/{total}" if total else "",
            f"",
            f"Articles Assessed:",
            f"  - Article 25 (Data Protection by Design): {'PASS' if any(r.passed for r in report.control_results if 'Art.25' in r.control_id) else 'NEEDS ATTENTION'}",
            f"  - Article 32 (Security of Processing): {'PASS' if any(r.passed for r in report.control_results if 'Art.32' in r.control_id) else 'NEEDS ATTENTION'}",
            f"",
            f"Findings by Severity:",
            f"  - Critical: {report.findings_by_severity.get('CRITICAL', 0)}",
            f"  - High: {report.findings_by_severity.get('HIGH', 0)}",
            f"  - Medium: {report.findings_by_severity.get('MEDIUM', 0)}",
            f"  - Low: {report.findings_by_severity.get('LOW', 0)}",
        ])

        if report.gaps:
            summary_parts.append(f"")
            summary_parts.append(f"Compliance Gaps Identified: {len(report.gaps)}")
            for gap in report.gaps[:3]:
                summary_parts.append(
                    f"  - {gap['control_id']}: {gap['control_name']} ({gap['status']})"
                )

        return "\n".join(summary_parts)
