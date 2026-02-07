# -*- coding: utf-8 -*-
"""
ISO 27001 Compliance Mapping - SEC-007

Maps security findings to ISO/IEC 27001:2022 Annex A controls.
Focuses on controls most relevant to security scanning and vulnerability
management.

ISO 27001 Controls Covered:
    - A.5.3: Segregation of duties
    - A.8.9: Configuration management
    - A.8.23: Web filtering
    - A.8.28: Secure coding
    - A.12.6.1: Management of technical vulnerabilities
    - A.14.2.1: Secure development policy
    - A.14.2.5: Secure system engineering principles
    - A.18.2.3: Technical compliance review

Example:
    >>> iso27001 = ISO27001Compliance()
    >>> results = await iso27001.check_controls(scan_results)
    >>> report = iso27001.generate_report(results)

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


class ISO27001Compliance(ComplianceFramework):
    """ISO/IEC 27001:2022 compliance framework implementation.

    Maps security scanning findings to ISO 27001 Annex A controls.
    Provides automated compliance checks and report generation for
    ISO 27001 certification audits.

    Attributes:
        framework_type: Always FrameworkType.ISO27001.
        controls: Dictionary of ISO 27001 control definitions.
        finding_mappings: Mappings from security findings to controls.

    Example:
        >>> iso27001 = ISO27001Compliance()
        >>> results = await iso27001.check_controls(scan_findings)
        >>> report = iso27001.generate_report(results)
    """

    def __init__(self) -> None:
        """Initialize ISO 27001 compliance framework."""
        super().__init__(FrameworkType.ISO27001)

    def _initialize_controls(self) -> None:
        """Initialize ISO 27001:2022 Annex A controls."""
        self.controls = {
            # A.5 - Organizational controls
            "A.5.3": ControlDefinition(
                control_id="A.5.3",
                name="Segregation of duties",
                description=(
                    "Conflicting duties and areas of responsibility shall be "
                    "segregated to reduce opportunities for unauthorized or "
                    "unintentional modification or misuse of the organization's assets."
                ),
                category=ControlCategory.ACCESS_CONTROL,
                requirements=[
                    "Separate security scanning from development",
                    "Independent review of security findings",
                    "Segregated access to production systems",
                ],
                automated_checks=[
                    "access_control_review",
                    "role_separation_audit",
                ],
                evidence_types=[
                    "access_matrix",
                    "role_definitions",
                ],
            ),
            # A.8 - Technological controls
            "A.8.9": ControlDefinition(
                control_id="A.8.9",
                name="Configuration management",
                description=(
                    "Configurations, including security configurations, of hardware, "
                    "software, services and networks shall be established, documented, "
                    "implemented, monitored and reviewed."
                ),
                category=ControlCategory.CHANGE_MANAGEMENT,
                requirements=[
                    "Document security configurations",
                    "Review configuration changes",
                    "Monitor for configuration drift",
                    "Secure default configurations",
                ],
                automated_checks=[
                    "iac_security_scan",
                    "configuration_audit",
                ],
                evidence_types=[
                    "iac_scan_results",
                    "configuration_baseline",
                ],
            ),
            "A.8.23": ControlDefinition(
                control_id="A.8.23",
                name="Web filtering",
                description=(
                    "Access to external websites shall be managed to reduce "
                    "exposure to malicious content."
                ),
                category=ControlCategory.NETWORK_SECURITY,
                requirements=[
                    "Filter malicious web content",
                    "Monitor web traffic",
                    "Block known bad domains",
                ],
                automated_checks=[
                    "network_scan",
                    "dast_scan",
                ],
                evidence_types=[
                    "web_filter_config",
                    "blocked_domains_list",
                ],
            ),
            "A.8.28": ControlDefinition(
                control_id="A.8.28",
                name="Secure coding",
                description=(
                    "Secure coding principles shall be applied to software development."
                ),
                category=ControlCategory.SECURE_DEVELOPMENT,
                requirements=[
                    "Apply secure coding standards",
                    "Perform code reviews",
                    "Use static analysis tools",
                    "Train developers on security",
                ],
                automated_checks=[
                    "sast_scan",
                    "code_review_audit",
                ],
                evidence_types=[
                    "sast_results",
                    "coding_standards",
                    "training_records",
                ],
            ),
            # A.12 - Operations security (legacy numbering still widely used)
            "A.12.6.1": ControlDefinition(
                control_id="A.12.6.1",
                name="Management of technical vulnerabilities",
                description=(
                    "Information about technical vulnerabilities of information "
                    "systems being used shall be obtained in a timely fashion, "
                    "the organization's exposure to such vulnerabilities evaluated "
                    "and appropriate measures taken to address the associated risk."
                ),
                category=ControlCategory.VULNERABILITY_MANAGEMENT,
                requirements=[
                    "Maintain vulnerability inventory",
                    "Assess vulnerability severity",
                    "Prioritize vulnerability remediation",
                    "Track remediation progress",
                    "Define remediation SLAs",
                ],
                automated_checks=[
                    "vulnerability_scan",
                    "dependency_scan",
                    "container_scan",
                    "dast_scan",
                ],
                evidence_types=[
                    "vulnerability_report",
                    "remediation_tracking",
                    "sla_compliance_report",
                ],
            ),
            # A.14 - System acquisition, development and maintenance
            "A.14.2.1": ControlDefinition(
                control_id="A.14.2.1",
                name="Secure development policy",
                description=(
                    "Rules for the development of software and systems shall be "
                    "established and applied to developments within the organization."
                ),
                category=ControlCategory.SECURE_DEVELOPMENT,
                requirements=[
                    "Define secure development policy",
                    "Implement security gates in SDLC",
                    "Require security testing",
                    "Review security of changes",
                ],
                automated_checks=[
                    "sast_scan",
                    "dependency_scan",
                    "secret_detection",
                    "ci_cd_audit",
                ],
                evidence_types=[
                    "sdlc_documentation",
                    "security_gate_results",
                    "ci_cd_configuration",
                ],
            ),
            "A.14.2.5": ControlDefinition(
                control_id="A.14.2.5",
                name="Secure system engineering principles",
                description=(
                    "Principles for engineering secure systems shall be established, "
                    "documented, maintained and applied to any information system "
                    "development activities."
                ),
                category=ControlCategory.SECURE_DEVELOPMENT,
                requirements=[
                    "Apply defense in depth",
                    "Implement least privilege",
                    "Secure by default",
                    "Fail securely",
                ],
                automated_checks=[
                    "sast_scan",
                    "iac_security_scan",
                    "container_scan",
                ],
                evidence_types=[
                    "architecture_documentation",
                    "security_design_review",
                ],
            ),
            # A.18 - Compliance
            "A.18.2.3": ControlDefinition(
                control_id="A.18.2.3",
                name="Technical compliance review",
                description=(
                    "Information systems shall be regularly reviewed for compliance "
                    "with the organization's information security policies and standards."
                ),
                category=ControlCategory.LOGGING_MONITORING,
                requirements=[
                    "Conduct regular compliance reviews",
                    "Document review findings",
                    "Track remediation of gaps",
                    "Report to management",
                ],
                automated_checks=[
                    "compliance_scan",
                    "configuration_audit",
                    "policy_enforcement_check",
                ],
                evidence_types=[
                    "compliance_report",
                    "gap_analysis",
                    "management_review_records",
                ],
            ),
        }

    def _initialize_mappings(self) -> None:
        """Initialize finding-to-control mappings for ISO 27001."""
        self.finding_mappings = [
            # SAST findings
            FindingControlMapping(
                finding_type="code_vulnerability",
                scanner="bandit",
                control_ids=["A.8.28", "A.14.2.1", "A.14.2.5"],
                severity_weight=1.0,
            ),
            FindingControlMapping(
                finding_type="code_vulnerability",
                scanner="semgrep",
                control_ids=["A.8.28", "A.14.2.1", "A.14.2.5"],
                severity_weight=1.0,
            ),
            # Dependency vulnerabilities
            FindingControlMapping(
                finding_type="dependency_vulnerability",
                scanner="trivy",
                control_ids=["A.12.6.1", "A.14.2.1"],
                severity_weight=1.0,
            ),
            FindingControlMapping(
                finding_type="dependency_vulnerability",
                scanner="snyk",
                control_ids=["A.12.6.1", "A.14.2.1"],
                severity_weight=1.0,
            ),
            # Secret detection
            FindingControlMapping(
                finding_type="secret",
                scanner="gitleaks",
                control_ids=["A.5.3", "A.8.28", "A.14.2.1"],
                severity_weight=1.5,
            ),
            # Container vulnerabilities
            FindingControlMapping(
                finding_type="container_vulnerability",
                scanner="trivy-container",
                control_ids=["A.12.6.1", "A.8.9"],
                severity_weight=1.0,
            ),
            # IaC misconfigurations
            FindingControlMapping(
                finding_type="iac_misconfiguration",
                scanner="tfsec",
                control_ids=["A.8.9", "A.14.2.5"],
                severity_weight=0.8,
            ),
            FindingControlMapping(
                finding_type="iac_misconfiguration",
                scanner="checkov",
                control_ids=["A.8.9", "A.14.2.5", "A.18.2.3"],
                severity_weight=0.8,
            ),
            # DAST findings
            FindingControlMapping(
                finding_type="web_vulnerability",
                scanner="zap",
                control_ids=["A.8.23", "A.12.6.1", "A.14.2.1"],
                severity_weight=1.2,
            ),
        ]

    async def check_controls(
        self,
        scan_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ControlResult]:
        """Check ISO 27001 controls against scan results.

        Args:
            scan_results: List of security scan findings.
            context: Additional context for evaluation.

        Returns:
            List of ControlResult for each ISO 27001 control.
        """
        context = context or {}
        results: List[ControlResult] = []

        # Group findings by control
        control_findings: Dict[str, List[Dict[str, Any]]] = {
            ctrl_id: [] for ctrl_id in self.controls.keys()
        }

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
            result = await self._evaluate_control(control, findings, context)
            results.append(result)

        logger.info(
            "ISO 27001 control check complete: %d controls evaluated, %d passed",
            len(results),
            sum(1 for r in results if r.passed),
        )

        return results

    async def _evaluate_control(
        self,
        control: ControlDefinition,
        findings: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> ControlResult:
        """Evaluate a single ISO 27001 control.

        Args:
            control: The control definition.
            findings: Findings related to this control.
            context: Additional evaluation context.

        Returns:
            ControlResult for the control.
        """
        score = self._calculate_control_score(control.control_id, findings)
        status = self._determine_control_status(score, findings)
        recommendations = self._generate_recommendations(control.control_id, findings)

        evidence: List[Dict[str, Any]] = []
        if findings:
            evidence.append({
                "type": "scan_findings",
                "count": len(findings),
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
                "critical_count": sum(
                    1 for f in findings
                    if f.get("severity", "").upper() == "CRITICAL"
                ),
            },
        )

    def _count_by_severity(
        self, findings: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Count findings by severity level."""
        counts: Dict[str, int] = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "INFO": 0,
        }
        for finding in findings:
            severity = finding.get("severity", "LOW").upper()
            if severity in counts:
                counts[severity] += 1
        return counts

    def generate_report(
        self,
        results: List[ControlResult],
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> ComplianceReport:
        """Generate ISO 27001 compliance report.

        Args:
            results: List of control check results.
            period_start: Start of the assessment period.
            period_end: End of the assessment period.

        Returns:
            Complete ISO 27001 ComplianceReport.
        """
        report = ComplianceReport(
            framework=FrameworkType.ISO27001,
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
        for result in results:
            for finding in result.findings:
                severity = finding.get("severity", "LOW").upper()
                if severity in report.findings_by_severity:
                    report.findings_by_severity[severity] += 1

        report.gaps = self._identify_gaps(results)
        report.summary = self._generate_summary(report)

        all_recommendations: List[str] = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        report.recommendations = list(set(all_recommendations))[:10]

        report.metadata = {
            "framework_version": "ISO/IEC 27001:2022",
            "annex_a_version": "2022",
            "controls_evaluated": len(results),
            "assessment_type": "automated",
        }

        logger.info(
            "ISO 27001 report generated: id=%s status=%s score=%.2f",
            report.report_id,
            report.overall_status.value,
            report.overall_score,
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
                        "priority": "HIGH" if result.status == ControlStatus.FAIL else "MEDIUM",
                        "remediation_guidance": result.recommendations,
                    })

        gaps.sort(key=lambda g: (0 if g["priority"] == "HIGH" else 1, -g["findings_count"]))
        return gaps

    def _generate_summary(self, report: ComplianceReport) -> str:
        """Generate executive summary for ISO 27001 report."""
        passed = len(report.passed_controls)
        failed = len(report.failed_controls)
        partial = len(report.partial_controls)
        total = len(report.control_results)

        summary_parts = [
            f"ISO/IEC 27001:2022 Compliance Assessment Summary",
            f"",
            f"Assessment Period: {report.period_start.strftime('%Y-%m-%d') if report.period_start else 'N/A'} "
            f"to {report.period_end.strftime('%Y-%m-%d') if report.period_end else 'N/A'}",
            f"",
            f"Overall Status: {report.overall_status.value}",
            f"Overall Score: {report.overall_score:.1%}",
            f"",
            f"Control Results (Annex A):",
            f"  - Passed: {passed}/{total} ({passed/total*100:.1f}%)" if total else "  - No controls evaluated",
            f"  - Failed: {failed}/{total}" if total else "",
            f"  - Partial: {partial}/{total}" if total else "",
            f"",
            f"Findings by Severity:",
            f"  - Critical: {report.findings_by_severity.get('CRITICAL', 0)}",
            f"  - High: {report.findings_by_severity.get('HIGH', 0)}",
            f"  - Medium: {report.findings_by_severity.get('MEDIUM', 0)}",
            f"  - Low: {report.findings_by_severity.get('LOW', 0)}",
        ]

        if report.gaps:
            summary_parts.append(f"")
            summary_parts.append(f"Non-Conformities Identified: {len(report.gaps)}")
            for gap in report.gaps[:3]:
                summary_parts.append(
                    f"  - {gap['control_id']}: {gap['control_name']} ({gap['status']})"
                )

        return "\n".join(summary_parts)
