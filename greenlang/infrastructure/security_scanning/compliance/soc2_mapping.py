# -*- coding: utf-8 -*-
"""
SOC 2 Type II Compliance Mapping - SEC-007

Maps security findings to SOC 2 Type II Trust Service Criteria controls.
Focuses on CC6 (Logical and Physical Access Controls) and CC7 (System Operations)
criteria most relevant to security scanning.

SOC 2 Controls Covered:
    - CC6.1: Logical access security software, infrastructure, and architectures
    - CC6.6: Logical access security measures
    - CC6.7: Transmission protection
    - CC6.8: Secure disposal
    - CC7.1: System availability monitoring
    - CC7.2: Change management

Example:
    >>> soc2 = SOC2Compliance()
    >>> results = await soc2.check_controls(scan_results)
    >>> report = soc2.generate_report(results)

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


class SOC2Compliance(ComplianceFramework):
    """SOC 2 Type II compliance framework implementation.

    Maps security scanning findings to SOC 2 Trust Service Criteria controls.
    Provides automated compliance checks and report generation for SOC 2 audits.

    Attributes:
        framework_type: Always FrameworkType.SOC2.
        controls: Dictionary of SOC 2 control definitions.
        finding_mappings: Mappings from security findings to controls.

    Example:
        >>> soc2 = SOC2Compliance()
        >>> results = await soc2.check_controls(scan_findings)
        >>> if all(r.passed for r in results):
        ...     print("SOC 2 compliant!")
    """

    def __init__(self) -> None:
        """Initialize SOC 2 compliance framework."""
        super().__init__(FrameworkType.SOC2)

    def _initialize_controls(self) -> None:
        """Initialize SOC 2 Trust Service Criteria controls."""
        self.controls = {
            # CC6 - Logical and Physical Access Controls
            "CC6.1": ControlDefinition(
                control_id="CC6.1",
                name="Logical Access Security",
                description=(
                    "The entity implements logical access security software, "
                    "infrastructure, and architectures over protected information "
                    "assets to protect them from security events to meet the "
                    "entity's objectives."
                ),
                category=ControlCategory.ACCESS_CONTROL,
                requirements=[
                    "Identify and authenticate users before granting access",
                    "Restrict access to authorized users",
                    "Implement access control policies",
                    "Protect against unauthorized access attempts",
                    "Monitor and log access events",
                ],
                automated_checks=[
                    "vulnerability_scan",
                    "secret_detection",
                    "access_control_review",
                    "authentication_config",
                ],
                evidence_types=[
                    "vulnerability_report",
                    "access_logs",
                    "authentication_config",
                    "secret_scan_results",
                ],
            ),
            "CC6.6": ControlDefinition(
                control_id="CC6.6",
                name="Logical Access Security Measures",
                description=(
                    "The entity implements logical access security measures to "
                    "protect against threats from sources outside its system "
                    "boundaries."
                ),
                category=ControlCategory.ACCESS_CONTROL,
                requirements=[
                    "Restrict external system access points",
                    "Implement network security controls",
                    "Monitor external access attempts",
                    "Protect against malware and threats",
                ],
                automated_checks=[
                    "network_scan",
                    "iac_security_scan",
                    "container_scan",
                ],
                evidence_types=[
                    "network_diagram",
                    "firewall_rules",
                    "iac_scan_results",
                ],
            ),
            "CC6.7": ControlDefinition(
                control_id="CC6.7",
                name="Transmission Protection",
                description=(
                    "The entity restricts the transmission, movement, and removal "
                    "of information to authorized internal and external users and "
                    "processes, and protects it during transmission."
                ),
                category=ControlCategory.ENCRYPTION,
                requirements=[
                    "Encrypt data in transit",
                    "Implement secure transmission protocols",
                    "Control data movement",
                    "Protect against interception",
                ],
                automated_checks=[
                    "tls_configuration_scan",
                    "encryption_verification",
                    "iac_security_scan",
                ],
                evidence_types=[
                    "encryption_config",
                    "certificate_inventory",
                    "data_flow_diagram",
                ],
            ),
            "CC6.8": ControlDefinition(
                control_id="CC6.8",
                name="Secure Disposal",
                description=(
                    "The entity implements controls to prevent unauthorized and "
                    "malicious software from being introduced."
                ),
                category=ControlCategory.DATA_PROTECTION,
                requirements=[
                    "Control software deployment",
                    "Verify software integrity",
                    "Scan for malicious code",
                    "Implement change controls for software",
                ],
                automated_checks=[
                    "sast_scan",
                    "dependency_scan",
                    "container_scan",
                    "supply_chain_verification",
                ],
                evidence_types=[
                    "sast_results",
                    "sbom",
                    "signature_verification",
                ],
            ),
            # CC7 - System Operations
            "CC7.1": ControlDefinition(
                control_id="CC7.1",
                name="Vulnerability Management",
                description=(
                    "To meet its objectives, the entity uses detection and "
                    "monitoring procedures to identify (1) changes to configurations "
                    "that result in the introduction of new vulnerabilities, and "
                    "(2) susceptibilities to newly discovered vulnerabilities."
                ),
                category=ControlCategory.VULNERABILITY_MANAGEMENT,
                requirements=[
                    "Conduct regular vulnerability assessments",
                    "Monitor for new vulnerabilities",
                    "Track vulnerability remediation",
                    "Assess risk from vulnerabilities",
                    "Prioritize and address critical vulnerabilities",
                ],
                automated_checks=[
                    "vulnerability_scan",
                    "dependency_scan",
                    "container_scan",
                    "dast_scan",
                ],
                evidence_types=[
                    "vulnerability_scan_report",
                    "remediation_tracking",
                    "patch_management_logs",
                ],
            ),
            "CC7.2": ControlDefinition(
                control_id="CC7.2",
                name="Security Monitoring",
                description=(
                    "The entity monitors system components and the operation of "
                    "those components for anomalies that are indicative of malicious "
                    "acts, natural disasters, and errors affecting the entity's "
                    "ability to meet its objectives."
                ),
                category=ControlCategory.LOGGING_MONITORING,
                requirements=[
                    "Monitor for security events",
                    "Log security-relevant activities",
                    "Alert on security anomalies",
                    "Review security logs",
                    "Respond to security events",
                ],
                automated_checks=[
                    "sast_scan",
                    "secret_detection",
                    "pii_scan",
                    "audit_log_review",
                ],
                evidence_types=[
                    "security_logs",
                    "alert_configurations",
                    "incident_response_records",
                ],
            ),
        }

    def _initialize_mappings(self) -> None:
        """Initialize finding-to-control mappings for SOC 2."""
        self.finding_mappings = [
            # SAST findings
            FindingControlMapping(
                finding_type="code_vulnerability",
                scanner="bandit",
                control_ids=["CC6.1", "CC6.8", "CC7.1", "CC7.2"],
                severity_weight=1.0,
            ),
            FindingControlMapping(
                finding_type="code_vulnerability",
                scanner="semgrep",
                control_ids=["CC6.1", "CC6.8", "CC7.1", "CC7.2"],
                severity_weight=1.0,
            ),
            # Dependency vulnerabilities
            FindingControlMapping(
                finding_type="dependency_vulnerability",
                scanner="trivy",
                control_ids=["CC7.1", "CC6.8"],
                severity_weight=1.0,
            ),
            FindingControlMapping(
                finding_type="dependency_vulnerability",
                scanner="snyk",
                control_ids=["CC7.1", "CC6.8"],
                severity_weight=1.0,
            ),
            # Secret detection
            FindingControlMapping(
                finding_type="secret",
                scanner="gitleaks",
                control_ids=["CC6.1", "CC6.7", "CC7.2"],
                severity_weight=1.5,  # Higher weight for secrets
            ),
            FindingControlMapping(
                finding_type="secret",
                scanner="trufflehog",
                control_ids=["CC6.1", "CC6.7", "CC7.2"],
                severity_weight=1.5,
            ),
            # Container vulnerabilities
            FindingControlMapping(
                finding_type="container_vulnerability",
                scanner="trivy-container",
                control_ids=["CC7.1", "CC6.6", "CC6.8"],
                severity_weight=1.0,
            ),
            # IaC misconfigurations
            FindingControlMapping(
                finding_type="iac_misconfiguration",
                scanner="tfsec",
                control_ids=["CC6.6", "CC6.7"],
                severity_weight=0.8,
            ),
            FindingControlMapping(
                finding_type="iac_misconfiguration",
                scanner="checkov",
                control_ids=["CC6.6", "CC6.7"],
                severity_weight=0.8,
            ),
            # DAST findings
            FindingControlMapping(
                finding_type="web_vulnerability",
                scanner="zap",
                control_ids=["CC6.1", "CC6.6", "CC7.1"],
                severity_weight=1.2,
            ),
            # PII exposure
            FindingControlMapping(
                finding_type="pii_exposure",
                scanner="pii",
                control_ids=["CC6.7", "CC7.2"],
                severity_weight=1.5,
            ),
        ]

    async def check_controls(
        self,
        scan_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ControlResult]:
        """Check SOC 2 controls against scan results.

        Args:
            scan_results: List of security scan findings.
            context: Additional context for evaluation.

        Returns:
            List of ControlResult for each SOC 2 control.
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

            # Map finding to controls
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
            "SOC 2 control check complete: %d controls evaluated, %d passed",
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
        """Evaluate a single SOC 2 control.

        Args:
            control: The control definition.
            findings: Findings related to this control.
            context: Additional evaluation context.

        Returns:
            ControlResult for the control.
        """
        # Calculate score
        score = self._calculate_control_score(control.control_id, findings)

        # Determine status
        status = self._determine_control_status(score, findings)

        # Generate recommendations
        recommendations = self._generate_recommendations(control.control_id, findings)

        # Build evidence list
        evidence: List[Dict[str, Any]] = []
        if findings:
            evidence.append({
                "type": "scan_findings",
                "count": len(findings),
                "severities": self._count_by_severity(findings),
            })

        # Add context-based evidence
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
                "high_count": sum(
                    1 for f in findings
                    if f.get("severity", "").upper() == "HIGH"
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
            else:
                counts["LOW"] += 1
        return counts

    def generate_report(
        self,
        results: List[ControlResult],
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> ComplianceReport:
        """Generate SOC 2 compliance report.

        Args:
            results: List of control check results.
            period_start: Start of the assessment period.
            period_end: End of the assessment period.

        Returns:
            Complete SOC 2 ComplianceReport.
        """
        report = ComplianceReport(
            framework=FrameworkType.SOC2,
            period_start=period_start or datetime.now(timezone.utc),
            period_end=period_end or datetime.now(timezone.utc),
            control_results=results,
        )

        # Calculate overall score and status
        report.calculate_score()
        report.calculate_status()

        # Aggregate findings by severity
        report.findings_by_severity = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "INFO": 0,
        }
        for result in results:
            for finding in result.findings:
                severity = finding.get("severity", "LOW").upper()
                if severity in report.findings_by_severity:
                    report.findings_by_severity[severity] += 1

        # Identify gaps
        report.gaps = self._identify_gaps(results)

        # Generate summary
        report.summary = self._generate_summary(report)

        # Compile recommendations
        all_recommendations: List[str] = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        report.recommendations = list(set(all_recommendations))[:10]  # Top 10 unique

        # Add metadata
        report.metadata = {
            "framework_version": "SOC 2 Type II (2023)",
            "trust_service_categories": ["Security"],
            "controls_evaluated": len(results),
            "assessment_type": "automated",
        }

        logger.info(
            "SOC 2 report generated: id=%s status=%s score=%.2f",
            report.report_id,
            report.overall_status.value,
            report.overall_score,
        )

        return report

    def _identify_gaps(
        self, results: List[ControlResult]
    ) -> List[Dict[str, Any]]:
        """Identify compliance gaps from control results.

        Args:
            results: Control check results.

        Returns:
            List of identified gaps.
        """
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

        # Sort by priority and findings count
        gaps.sort(
            key=lambda g: (
                0 if g["priority"] == "HIGH" else 1,
                -g["findings_count"],
            )
        )

        return gaps

    def _generate_summary(self, report: ComplianceReport) -> str:
        """Generate executive summary for the report.

        Args:
            report: The compliance report.

        Returns:
            Summary text.
        """
        passed = len(report.passed_controls)
        failed = len(report.failed_controls)
        partial = len(report.partial_controls)
        total = len(report.control_results)

        summary_parts = [
            f"SOC 2 Type II Compliance Assessment Summary",
            f"",
            f"Assessment Period: {report.period_start.strftime('%Y-%m-%d') if report.period_start else 'N/A'} "
            f"to {report.period_end.strftime('%Y-%m-%d') if report.period_end else 'N/A'}",
            f"",
            f"Overall Status: {report.overall_status.value}",
            f"Overall Score: {report.overall_score:.1%}",
            f"",
            f"Control Results:",
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
            summary_parts.append(f"Compliance Gaps Identified: {len(report.gaps)}")
            for gap in report.gaps[:3]:
                summary_parts.append(
                    f"  - {gap['control_id']}: {gap['control_name']} ({gap['status']})"
                )

        return "\n".join(summary_parts)
