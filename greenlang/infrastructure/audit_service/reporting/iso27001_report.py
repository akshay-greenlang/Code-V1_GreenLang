# -*- coding: utf-8 -*-
"""
ISO 27001 Compliance Report Generator - SEC-005

Generates ISO 27001 Information Security Management System (ISMS)
compliance reports covering:
- A.9: Access Control
- A.12: Operations Security
- A.18: Compliance

Author: GreenLang Framework Team
Date: February 2026
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ISOControl:
    """ISO 27001 control definition."""

    def __init__(
        self,
        control_id: str,
        name: str,
        objective: str,
        category: str,
        event_types: List[str],
        implementation_guidance: str,
    ):
        self.control_id = control_id
        self.name = name
        self.objective = objective
        self.category = category
        self.event_types = event_types
        self.implementation_guidance = implementation_guidance


# ISO 27001:2022 Annex A Controls
ISO27001_CONTROLS = [
    # A.9: Access Control
    ISOControl(
        control_id="A.9.1.1",
        name="Access Control Policy",
        objective="An access control policy shall be established, documented and reviewed based on business and information security requirements.",
        category="A.9",
        event_types=["policy_created", "policy_updated", "policy_reviewed"],
        implementation_guidance="Document access control policies and review periodically.",
    ),
    ISOControl(
        control_id="A.9.1.2",
        name="Access to Networks and Network Services",
        objective="Users shall only be provided with access to the network and network services that they have been specifically authorized to use.",
        category="A.9",
        event_types=["network_access_granted", "network_access_denied", "vpn_connected"],
        implementation_guidance="Implement network access controls and VPN policies.",
    ),
    ISOControl(
        control_id="A.9.2.1",
        name="User Registration and De-registration",
        objective="A formal user registration and de-registration process shall be implemented to enable assignment of access rights.",
        category="A.9",
        event_types=["user_created", "user_deleted", "user_disabled"],
        implementation_guidance="Implement formal user lifecycle management.",
    ),
    ISOControl(
        control_id="A.9.2.2",
        name="User Access Provisioning",
        objective="A formal user access provisioning process shall be implemented to assign or revoke access rights.",
        category="A.9",
        event_types=["role_assigned", "role_revoked", "permission_granted", "permission_revoked"],
        implementation_guidance="Implement role-based access control with approval workflows.",
    ),
    ISOControl(
        control_id="A.9.2.3",
        name="Management of Privileged Access Rights",
        objective="The allocation and use of privileged access rights shall be restricted and controlled.",
        category="A.9",
        event_types=["admin_role_assigned", "elevated_access", "privilege_escalation"],
        implementation_guidance="Implement privileged access management (PAM).",
    ),
    ISOControl(
        control_id="A.9.4.1",
        name="Information Access Restriction",
        objective="Access to information and application system functions shall be restricted in accordance with the access control policy.",
        category="A.9",
        event_types=["data_access", "access_denied", "permission_check"],
        implementation_guidance="Implement fine-grained access controls.",
    ),
    ISOControl(
        control_id="A.9.4.2",
        name="Secure Log-on Procedures",
        objective="Where required by the access control policy, access to systems shall be controlled by a secure log-on procedure.",
        category="A.9",
        event_types=["login_success", "login_failure", "mfa_verified"],
        implementation_guidance="Implement secure authentication with MFA.",
    ),

    # A.12: Operations Security
    ISOControl(
        control_id="A.12.1.1",
        name="Documented Operating Procedures",
        objective="Operating procedures shall be documented and made available to all users who need them.",
        category="A.12",
        event_types=["procedure_created", "procedure_updated", "runbook_executed"],
        implementation_guidance="Maintain documented operational procedures.",
    ),
    ISOControl(
        control_id="A.12.1.2",
        name="Change Management",
        objective="Changes to the organization, business processes and information processing facilities shall be controlled.",
        category="A.12",
        event_types=["config_changed", "deployment_started", "deployment_completed", "rollback"],
        implementation_guidance="Implement formal change management processes.",
    ),
    ISOControl(
        control_id="A.12.4.1",
        name="Event Logging",
        objective="Event logs recording user activities, exceptions, faults and information security events shall be produced, kept and regularly reviewed.",
        category="A.12",
        event_types=["*"],  # All events
        implementation_guidance="Implement comprehensive event logging.",
    ),
    ISOControl(
        control_id="A.12.4.2",
        name="Protection of Log Information",
        objective="Logging facilities and log information shall be protected against tampering and unauthorized access.",
        category="A.12",
        event_types=["log_access", "log_tamper_attempt", "audit_config_changed"],
        implementation_guidance="Protect audit logs with integrity controls.",
    ),
    ISOControl(
        control_id="A.12.4.3",
        name="Administrator and Operator Logs",
        objective="System administrator and system operator activities shall be logged and the logs protected and regularly reviewed.",
        category="A.12",
        event_types=["admin_action", "system_config_changed", "backup_performed"],
        implementation_guidance="Log and review all administrative activities.",
    ),

    # A.18: Compliance
    ISOControl(
        control_id="A.18.1.1",
        name="Identification of Applicable Legislation",
        objective="All relevant legislative, regulatory and contractual requirements shall be explicitly identified.",
        category="A.18",
        event_types=["compliance_requirement_added", "regulation_updated"],
        implementation_guidance="Maintain compliance requirements registry.",
    ),
    ISOControl(
        control_id="A.18.1.3",
        name="Protection of Records",
        objective="Records shall be protected from loss, destruction, falsification, unauthorized access and unauthorized release.",
        category="A.18",
        event_types=["record_created", "record_accessed", "record_modified", "retention_applied"],
        implementation_guidance="Implement records management with retention policies.",
    ),
    ISOControl(
        control_id="A.18.2.1",
        name="Independent Review of Information Security",
        objective="The organization's approach to managing information security shall be reviewed independently.",
        category="A.18",
        event_types=["audit_started", "audit_completed", "finding_reported"],
        implementation_guidance="Conduct regular independent security reviews.",
    ),
]


class ControlEvidence:
    """Evidence collected for a control."""

    def __init__(self, control: ISOControl):
        self.control = control
        self.events: List[Dict[str, Any]] = []
        self.event_count = 0
        self.compliance_level: str = "unknown"  # full, partial, none, not_applicable
        self.findings: List[str] = []
        self.effectiveness_score: float = 0.0


class ISO27001ReportGenerator:
    """Generates ISO 27001 ISMS compliance reports.

    Collects audit evidence for ISO 27001 Annex A controls and generates
    comprehensive compliance reports for ISMS certification audits.
    """

    def __init__(self, repository: Optional[Any] = None):
        """Initialize the report generator.

        Args:
            repository: Optional audit event repository.
        """
        self._repository = repository
        self._controls = ISO27001_CONTROLS

    async def generate(
        self,
        period_start: datetime,
        period_end: datetime,
        organization_id: Optional[str] = None,
        format: Any = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bytes:
        """Generate ISO 27001 compliance report.

        Args:
            period_start: Report period start.
            period_end: Report period end.
            organization_id: Optional organization filter.
            format: Output format.
            progress_callback: Optional progress callback.

        Returns:
            Report content as bytes.
        """
        logger.info(
            "Generating ISO 27001 report for period %s to %s",
            period_start.isoformat(),
            period_end.isoformat(),
        )

        # Collect evidence for each control
        evidence_map: Dict[str, ControlEvidence] = {}
        total_controls = len(self._controls)

        for i, control in enumerate(self._controls):
            evidence = await self._collect_evidence(
                control, period_start, period_end, organization_id
            )
            evidence_map[control.control_id] = evidence

            if progress_callback:
                progress = (i + 1) / total_controls * 80
                progress_callback(progress)

        # Generate Statement of Applicability
        soa = self._generate_soa(evidence_map)

        if progress_callback:
            progress_callback(85)

        # Calculate ISMS maturity
        maturity_assessment = self._assess_maturity(evidence_map)

        if progress_callback:
            progress_callback(90)

        # Build report
        report = self._build_report(
            evidence_map=evidence_map,
            soa=soa,
            maturity_assessment=maturity_assessment,
            period_start=period_start,
            period_end=period_end,
            organization_id=organization_id,
        )

        if progress_callback:
            progress_callback(95)

        # Format output
        format_value = format.value if hasattr(format, "value") else str(format)
        if format_value == "json":
            return json.dumps(report, indent=2, default=str).encode("utf-8")
        elif format_value == "html":
            return self._to_html(report).encode("utf-8")
        else:
            return json.dumps(report, indent=2, default=str).encode("utf-8")

    async def _collect_evidence(
        self,
        control: ISOControl,
        period_start: datetime,
        period_end: datetime,
        organization_id: Optional[str],
    ) -> ControlEvidence:
        """Collect audit evidence for a control.

        Args:
            control: ISO control definition.
            period_start: Period start.
            period_end: Period end.
            organization_id: Organization filter.

        Returns:
            Collected evidence.
        """
        evidence = ControlEvidence(control)

        if self._repository is None:
            # Generate sample data
            evidence.event_count = 200
            evidence.compliance_level = "full"
            evidence.effectiveness_score = 0.92
            return evidence

        try:
            from greenlang.infrastructure.audit_service.models import (
                SearchQuery,
                TimeRange,
            )

            # Handle wildcard event types
            event_types = None if "*" in control.event_types else control.event_types

            query = SearchQuery(
                time_range=TimeRange(start=period_start, end=period_end),
                event_types=event_types,
            )

            events, total, _ = await self._repository.search(
                query=query,
                limit=500,
                offset=0,
            )

            evidence.events = [
                {
                    "timestamp": e.performed_at.isoformat(),
                    "event_type": e.event_type,
                    "outcome": e.outcome.value,
                }
                for e in events
            ]
            evidence.event_count = total

            # Calculate compliance level
            if evidence.event_count == 0:
                evidence.compliance_level = "not_applicable"
            else:
                success_rate = sum(
                    1 for e in events if e.outcome.value == "success"
                ) / max(len(events), 1)

                if success_rate >= 0.95:
                    evidence.compliance_level = "full"
                    evidence.effectiveness_score = success_rate
                elif success_rate >= 0.70:
                    evidence.compliance_level = "partial"
                    evidence.effectiveness_score = success_rate
                    evidence.findings.append(
                        f"Control effectiveness below target: {success_rate:.0%}"
                    )
                else:
                    evidence.compliance_level = "none"
                    evidence.effectiveness_score = success_rate
                    evidence.findings.append(
                        f"Control not effectively implemented: {success_rate:.0%}"
                    )

        except Exception as exc:
            logger.error("Failed to collect evidence for %s: %s", control.control_id, exc)
            evidence.compliance_level = "none"
            evidence.findings.append(f"Error collecting evidence: {exc}")

        return evidence

    def _generate_soa(
        self, evidence_map: Dict[str, ControlEvidence]
    ) -> Dict[str, Any]:
        """Generate Statement of Applicability.

        Args:
            evidence_map: Evidence for each control.

        Returns:
            SOA data.
        """
        controls = []
        for control_id, evidence in evidence_map.items():
            controls.append({
                "control_id": control_id,
                "name": evidence.control.name,
                "applicable": evidence.compliance_level != "not_applicable",
                "implemented": evidence.compliance_level in ("full", "partial"),
                "justification": self._get_applicability_justification(evidence),
                "reference": f"Evidence from {evidence.event_count} audit events",
            })

        applicable_count = sum(1 for c in controls if c["applicable"])
        implemented_count = sum(1 for c in controls if c["implemented"])

        return {
            "controls": controls,
            "summary": {
                "total_controls": len(controls),
                "applicable": applicable_count,
                "not_applicable": len(controls) - applicable_count,
                "implemented": implemented_count,
                "implementation_rate": implemented_count / max(applicable_count, 1) * 100,
            },
        }

    def _get_applicability_justification(self, evidence: ControlEvidence) -> str:
        """Get applicability justification for a control.

        Args:
            evidence: Control evidence.

        Returns:
            Justification text.
        """
        if evidence.compliance_level == "not_applicable":
            return "Control not applicable based on risk assessment."
        elif evidence.compliance_level == "full":
            return "Fully implemented with documented evidence."
        elif evidence.compliance_level == "partial":
            return "Partially implemented; remediation in progress."
        else:
            return "Not yet implemented; planned for future phase."

    def _assess_maturity(
        self, evidence_map: Dict[str, ControlEvidence]
    ) -> Dict[str, Any]:
        """Assess ISMS maturity level.

        Args:
            evidence_map: Evidence for each control.

        Returns:
            Maturity assessment data.
        """
        categories = {}
        for control_id, evidence in evidence_map.items():
            category = evidence.control.category
            if category not in categories:
                categories[category] = {
                    "controls": 0,
                    "full": 0,
                    "partial": 0,
                    "none": 0,
                    "effectiveness_sum": 0.0,
                }

            categories[category]["controls"] += 1
            categories[category][evidence.compliance_level] = (
                categories[category].get(evidence.compliance_level, 0) + 1
            )
            categories[category]["effectiveness_sum"] += evidence.effectiveness_score

        # Calculate category scores
        category_scores = {}
        for cat, data in categories.items():
            avg_effectiveness = data["effectiveness_sum"] / max(data["controls"], 1)
            category_scores[cat] = {
                "maturity_level": self._calculate_maturity_level(avg_effectiveness),
                "effectiveness": round(avg_effectiveness * 100, 1),
                "controls_full": data["full"],
                "controls_partial": data["partial"],
                "controls_none": data["none"],
            }

        # Overall maturity
        overall_effectiveness = sum(
            e.effectiveness_score for e in evidence_map.values()
        ) / max(len(evidence_map), 1)

        return {
            "overall_maturity_level": self._calculate_maturity_level(overall_effectiveness),
            "overall_effectiveness": round(overall_effectiveness * 100, 1),
            "category_assessments": category_scores,
            "maturity_scale": {
                "1": "Initial - Ad-hoc processes",
                "2": "Developing - Documented processes",
                "3": "Defined - Standardized processes",
                "4": "Managed - Measured and controlled",
                "5": "Optimized - Continuous improvement",
            },
        }

    def _calculate_maturity_level(self, effectiveness: float) -> int:
        """Calculate maturity level from effectiveness score.

        Args:
            effectiveness: Effectiveness score (0-1).

        Returns:
            Maturity level (1-5).
        """
        if effectiveness >= 0.95:
            return 5
        elif effectiveness >= 0.85:
            return 4
        elif effectiveness >= 0.70:
            return 3
        elif effectiveness >= 0.50:
            return 2
        else:
            return 1

    def _build_report(
        self,
        evidence_map: Dict[str, ControlEvidence],
        soa: Dict[str, Any],
        maturity_assessment: Dict[str, Any],
        period_start: datetime,
        period_end: datetime,
        organization_id: Optional[str],
    ) -> Dict[str, Any]:
        """Build the complete report structure.

        Args:
            evidence_map: Evidence for each control.
            soa: Statement of Applicability.
            maturity_assessment: Maturity assessment.
            period_start: Period start.
            period_end: Period end.
            organization_id: Organization ID.

        Returns:
            Complete report data.
        """
        # Control details by category
        control_details = {}
        for control_id, evidence in evidence_map.items():
            category = evidence.control.category
            if category not in control_details:
                control_details[category] = []

            control_details[category].append({
                "control_id": control_id,
                "name": evidence.control.name,
                "objective": evidence.control.objective,
                "compliance_level": evidence.compliance_level,
                "effectiveness_score": round(evidence.effectiveness_score * 100, 1),
                "event_count": evidence.event_count,
                "findings": evidence.findings,
                "implementation_guidance": evidence.control.implementation_guidance,
            })

        return {
            "report_type": "iso27001",
            "version": "1.0",
            "executive_summary": {
                "report_title": "ISO 27001 ISMS Compliance Report",
                "standard_version": "ISO/IEC 27001:2022",
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "overall_maturity": maturity_assessment["overall_maturity_level"],
                "overall_effectiveness": maturity_assessment["overall_effectiveness"],
                "soa_implementation_rate": soa["summary"]["implementation_rate"],
            },
            "statement_of_applicability": soa,
            "maturity_assessment": maturity_assessment,
            "control_assessments": {
                "a9_access_control": control_details.get("A.9", []),
                "a12_operations_security": control_details.get("A.12", []),
                "a18_compliance": control_details.get("A.18", []),
            },
            "recommendations": self._generate_recommendations(evidence_map),
            "metadata": {
                "organization_id": organization_id,
                "generator_version": "1.0.0",
            },
        }

    def _generate_recommendations(
        self, evidence_map: Dict[str, ControlEvidence]
    ) -> List[Dict[str, Any]]:
        """Generate improvement recommendations.

        Args:
            evidence_map: Evidence for each control.

        Returns:
            List of recommendations.
        """
        recommendations = []

        for control_id, evidence in evidence_map.items():
            if evidence.compliance_level in ("none", "partial"):
                recommendations.append({
                    "control_id": control_id,
                    "priority": "high" if evidence.compliance_level == "none" else "medium",
                    "finding": evidence.findings[0] if evidence.findings else "Control requires improvement",
                    "recommendation": evidence.control.implementation_guidance,
                    "expected_outcome": f"Achieve full compliance for {evidence.control.name}",
                })

        # Sort by priority
        recommendations.sort(key=lambda r: 0 if r["priority"] == "high" else 1)

        return recommendations[:10]  # Top 10 recommendations

    def _to_html(self, report: Dict[str, Any]) -> str:
        """Convert report to HTML format.

        Args:
            report: Report data.

        Returns:
            HTML string.
        """
        summary = report.get("executive_summary", {})
        maturity = report.get("maturity_assessment", {})

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ISO 27001 ISMS Compliance Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a237e; }}
        h2 {{ color: #283593; border-bottom: 2px solid #3949ab; padding-bottom: 10px; }}
        .maturity-badge {{ display: inline-block; padding: 10px 20px; border-radius: 20px; font-size: 24px; font-weight: bold; }}
        .level-5 {{ background: #1b5e20; color: white; }}
        .level-4 {{ background: #2e7d32; color: white; }}
        .level-3 {{ background: #f9a825; color: black; }}
        .level-2 {{ background: #ef6c00; color: white; }}
        .level-1 {{ background: #c62828; color: white; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: #e8eaf6; padding: 20px; border-radius: 8px; text-align: center; }}
        .summary-value {{ font-size: 32px; font-weight: bold; color: #3949ab; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #c5cae9; padding: 12px; text-align: left; }}
        th {{ background: #3949ab; color: white; }}
        .compliance-full {{ color: #1b5e20; }}
        .compliance-partial {{ color: #f57c00; }}
        .compliance-none {{ color: #c62828; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{summary.get('report_title', 'ISO 27001 Compliance Report')}</h1>
        <p><strong>Standard:</strong> {summary.get('standard_version', 'ISO/IEC 27001:2022')}</p>
        <p><strong>Report Period:</strong> {summary.get('period_start', 'N/A')} to {summary.get('period_end', 'N/A')}</p>

        <h2>ISMS Maturity Assessment</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="maturity-badge level-{maturity.get('overall_maturity_level', 1)}">
                    Level {maturity.get('overall_maturity_level', 1)}
                </div>
                <p>Overall Maturity</p>
            </div>
            <div class="summary-card">
                <div class="summary-value">{maturity.get('overall_effectiveness', 0)}%</div>
                <p>Overall Effectiveness</p>
            </div>
            <div class="summary-card">
                <div class="summary-value">{summary.get('soa_implementation_rate', 0):.0f}%</div>
                <p>SOA Implementation</p>
            </div>
        </div>

        <h2>Statement of Applicability Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Count</th>
            </tr>
            <tr><td>Total Controls</td><td>{report.get('statement_of_applicability', {}).get('summary', {}).get('total_controls', 0)}</td></tr>
            <tr><td>Applicable</td><td>{report.get('statement_of_applicability', {}).get('summary', {}).get('applicable', 0)}</td></tr>
            <tr><td>Implemented</td><td>{report.get('statement_of_applicability', {}).get('summary', {}).get('implemented', 0)}</td></tr>
        </table>

        <h2>Top Recommendations</h2>
        <table>
            <tr><th>Priority</th><th>Control</th><th>Recommendation</th></tr>
"""
        for rec in report.get("recommendations", [])[:5]:
            html += f"<tr><td>{rec['priority'].upper()}</td><td>{rec['control_id']}</td><td>{rec['recommendation']}</td></tr>"

        html += """
        </table>

        <footer>
            <p><small>Generated by GreenLang Audit Service - ISO 27001 Report Generator</small></p>
        </footer>
    </div>
</body>
</html>"""

        return html
