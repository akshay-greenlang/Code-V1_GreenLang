"""
Certification Report Generation

Generates comprehensive certification reports including:
- Certification scorecard
- Dimension breakdown
- Recommendations
- Compliance matrix

Example:
    >>> from certification.reports import ReportGenerator
    >>> generator = ReportGenerator()
    >>> scorecard = generator.generate_scorecard(certification_report)
    >>> generator.export_html(scorecard, "report.html")

"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DimensionScore:
    """Score for a single dimension."""
    name: str
    score: float
    weight: float
    weighted_score: float
    passed: bool
    threshold: float
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class CertificationScorecard:
    """Certification scorecard summary."""
    agent_id: str
    agent_version: str
    certification_id: str
    overall_score: float
    certification_level: str
    is_certified: bool
    timestamp: datetime
    valid_until: Optional[datetime]
    dimension_scores: List[DimensionScore] = field(default_factory=list)
    critical_findings: List[str] = field(default_factory=list)
    top_recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_dimension_by_name(self, name: str) -> Optional[DimensionScore]:
        """Get dimension score by name."""
        for dim in self.dimension_scores:
            if dim.name == name:
                return dim
        return None


@dataclass
class ComplianceItem:
    """Single compliance requirement."""
    requirement_id: str
    description: str
    status: str  # compliant, non_compliant, partial, n/a
    dimension: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class ComplianceMatrix:
    """Compliance matrix for regulatory requirements."""
    framework: str  # e.g., "GHG Protocol", "ISO 14064"
    items: List[ComplianceItem] = field(default_factory=list)
    compliance_rate: float = 0.0

    @property
    def compliant_count(self) -> int:
        """Count compliant items."""
        return sum(1 for item in self.items if item.status == "compliant")

    @property
    def total_applicable(self) -> int:
        """Count applicable items."""
        return sum(1 for item in self.items if item.status != "n/a")


@dataclass
class Recommendation:
    """Single recommendation for improvement."""
    priority: str  # critical, high, medium, low
    dimension: str
    description: str
    impact: str
    effort: str  # low, medium, high
    remediation: str


class RecommendationEngine:
    """
    Generates prioritized recommendations from certification results.

    Analyzes dimension scores and findings to produce actionable
    recommendations for improvement.
    """

    # Recommendation templates
    RECOMMENDATION_TEMPLATES = {
        "technical_accuracy": {
            "low_score": {
                "description": "Improve calculation accuracy",
                "remediation": "Review formulas against reference implementations and add unit tests",
                "impact": "Critical for regulatory compliance",
                "effort": "medium",
            },
            "missing_tests": {
                "description": "Add comprehensive golden tests",
                "remediation": "Create test cases covering all calculation paths with known values",
                "impact": "Enables verification of accuracy",
                "effort": "medium",
            },
        },
        "data_credibility": {
            "low_score": {
                "description": "Improve data source documentation",
                "remediation": "Document all emission factors sources with version and date",
                "impact": "Required for audit compliance",
                "effort": "low",
            },
            "missing_provenance": {
                "description": "Implement provenance tracking",
                "remediation": "Add SHA-256 provenance hash to all outputs",
                "impact": "Enables full audit trail",
                "effort": "medium",
            },
        },
        "safety_compliance": {
            "low_score": {
                "description": "Address safety compliance gaps",
                "remediation": "Review and implement relevant NFPA/IEC/OSHA requirements",
                "impact": "Critical for operational safety",
                "effort": "high",
            },
        },
        "regulatory_alignment": {
            "low_score": {
                "description": "Improve regulatory framework alignment",
                "remediation": "Map outputs to GHG Protocol scopes and ensure CSRD compatibility",
                "impact": "Required for regulatory reporting",
                "effort": "medium",
            },
        },
        "performance": {
            "low_score": {
                "description": "Optimize performance",
                "remediation": "Profile slow operations and implement caching where appropriate",
                "impact": "Improves user experience and scalability",
                "effort": "medium",
            },
        },
    }

    def __init__(self):
        """Initialize recommendation engine."""
        logger.info("RecommendationEngine initialized")

    def generate_recommendations(
        self,
        dimension_scores: List[DimensionScore],
        max_recommendations: int = 10,
    ) -> List[Recommendation]:
        """
        Generate prioritized recommendations.

        Args:
            dimension_scores: List of dimension scores
            max_recommendations: Maximum number of recommendations

        Returns:
            List of prioritized recommendations
        """
        recommendations = []

        # Sort dimensions by score (lowest first)
        sorted_dims = sorted(dimension_scores, key=lambda d: d.score)

        for dim in sorted_dims:
            if not dim.passed:
                # Critical priority for failed dimensions
                rec = self._create_recommendation(dim, "critical")
                if rec:
                    recommendations.append(rec)
            elif dim.score < 80:
                # High priority for low scores
                rec = self._create_recommendation(dim, "high")
                if rec:
                    recommendations.append(rec)
            elif dim.score < 90:
                # Medium priority for moderate scores
                rec = self._create_recommendation(dim, "medium")
                if rec:
                    recommendations.append(rec)

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 4))

        return recommendations[:max_recommendations]

    def _create_recommendation(
        self,
        dim: DimensionScore,
        priority: str,
    ) -> Optional[Recommendation]:
        """Create recommendation for a dimension."""
        templates = self.RECOMMENDATION_TEMPLATES.get(dim.name, {})
        template = templates.get("low_score", {})

        if not template:
            # Generic recommendation
            return Recommendation(
                priority=priority,
                dimension=dim.name,
                description=f"Improve {dim.name.replace('_', ' ')} score",
                impact=f"Current score: {dim.score:.1f}/100",
                effort="medium",
                remediation="; ".join(dim.recommendations[:2]) if dim.recommendations else "Review and address findings",
            )

        return Recommendation(
            priority=priority,
            dimension=dim.name,
            description=template.get("description", f"Improve {dim.name}"),
            impact=template.get("impact", "Improves certification score"),
            effort=template.get("effort", "medium"),
            remediation=template.get("remediation", "Review dimension requirements"),
        )


class ReportGenerator:
    """
    Generates certification reports in various formats.

    Supports:
    - Scorecard generation
    - Compliance matrix
    - HTML export
    - JSON export
    - Markdown export
    """

    def __init__(self):
        """Initialize report generator."""
        self.recommendation_engine = RecommendationEngine()
        logger.info("ReportGenerator initialized")

    def generate_scorecard(
        self,
        certification_report: Any,  # CertificationReport
    ) -> CertificationScorecard:
        """
        Generate certification scorecard from report.

        Args:
            certification_report: Full certification report

        Returns:
            CertificationScorecard summary
        """
        dimension_scores = []

        for name, result in certification_report.dimension_results.items():
            dim_score = DimensionScore(
                name=name,
                score=result.score,
                weight=result.weight,
                weighted_score=result.weighted_score,
                passed=result.passed_threshold,
                threshold=result.threshold,
                findings=result.findings,
                recommendations=result.recommendations,
            )
            dimension_scores.append(dim_score)

        # Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            dimension_scores, max_recommendations=5
        )
        top_recs = [r.description for r in recommendations]

        return CertificationScorecard(
            agent_id=certification_report.agent_id,
            agent_version=certification_report.agent_version,
            certification_id=certification_report.certification_id,
            overall_score=certification_report.overall_score,
            certification_level=certification_report.certification_level.value,
            is_certified=certification_report.is_certified,
            timestamp=certification_report.timestamp,
            valid_until=certification_report.valid_until,
            dimension_scores=dimension_scores,
            critical_findings=certification_report.critical_findings,
            top_recommendations=top_recs,
        )

    def generate_compliance_matrix(
        self,
        certification_report: Any,
        framework: str = "GHG Protocol",
    ) -> ComplianceMatrix:
        """
        Generate compliance matrix for a framework.

        Args:
            certification_report: Full certification report
            framework: Regulatory framework name

        Returns:
            ComplianceMatrix with compliance status
        """
        items = []

        if framework == "GHG Protocol":
            items = self._generate_ghg_protocol_items(certification_report)
        elif framework == "ISO 14064":
            items = self._generate_iso_14064_items(certification_report)
        elif framework == "EU CSRD":
            items = self._generate_csrd_items(certification_report)

        compliant_count = sum(1 for i in items if i.status == "compliant")
        total = sum(1 for i in items if i.status != "n/a")
        rate = (compliant_count / total * 100) if total > 0 else 0.0

        return ComplianceMatrix(
            framework=framework,
            items=items,
            compliance_rate=rate,
        )

    def _generate_ghg_protocol_items(
        self, report: Any
    ) -> List[ComplianceItem]:
        """Generate GHG Protocol compliance items."""
        items = []

        # Check technical accuracy for calculation requirements
        tech_result = report.dimension_results.get("technical_accuracy")
        items.append(ComplianceItem(
            requirement_id="GHG-001",
            description="Accurate quantification of GHG emissions",
            status="compliant" if tech_result and tech_result.passed_threshold else "non_compliant",
            dimension="technical_accuracy",
            evidence=["Golden test pass rate", "Calculation validation"],
        ))

        # Check data credibility for source requirements
        data_result = report.dimension_results.get("data_credibility")
        items.append(ComplianceItem(
            requirement_id="GHG-002",
            description="Use of recognized emission factors",
            status="compliant" if data_result and data_result.passed_threshold else "non_compliant",
            dimension="data_credibility",
            evidence=["Source documentation", "Provenance tracking"],
        ))

        # Check uncertainty quantification
        uncert_result = report.dimension_results.get("uncertainty_quantification")
        items.append(ComplianceItem(
            requirement_id="GHG-003",
            description="Uncertainty assessment",
            status="compliant" if uncert_result and uncert_result.passed_threshold else "partial",
            dimension="uncertainty_quantification",
            evidence=["Uncertainty bounds", "Confidence intervals"],
        ))

        # Check auditability for documentation
        audit_result = report.dimension_results.get("auditability")
        items.append(ComplianceItem(
            requirement_id="GHG-004",
            description="Documentation and record keeping",
            status="compliant" if audit_result and audit_result.passed_threshold else "non_compliant",
            dimension="auditability",
            evidence=["Audit trail", "Provenance hash"],
        ))

        return items

    def _generate_iso_14064_items(
        self, report: Any
    ) -> List[ComplianceItem]:
        """Generate ISO 14064 compliance items."""
        items = []

        items.append(ComplianceItem(
            requirement_id="ISO-001",
            description="Organizational boundary definition",
            status="partial",
            dimension="regulatory_alignment",
            evidence=["Pack specification"],
        ))

        items.append(ComplianceItem(
            requirement_id="ISO-002",
            description="Quantification methodology",
            status="compliant" if report.dimension_results.get("technical_accuracy", {}) else "non_compliant",
            dimension="technical_accuracy",
            evidence=["Formula documentation"],
        ))

        return items

    def _generate_csrd_items(
        self, report: Any
    ) -> List[ComplianceItem]:
        """Generate EU CSRD compliance items."""
        items = []

        items.append(ComplianceItem(
            requirement_id="CSRD-001",
            description="Double materiality assessment readiness",
            status="partial",
            dimension="regulatory_alignment",
            evidence=["Regulatory alignment dimension"],
        ))

        return items

    def export_html(
        self,
        scorecard: CertificationScorecard,
        output_path: str,
    ) -> None:
        """
        Export scorecard to HTML format.

        Args:
            scorecard: Certification scorecard
            output_path: Output file path
        """
        html = self._generate_html(scorecard)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Exported HTML report to {output_path}")

    def _generate_html(self, scorecard: CertificationScorecard) -> str:
        """Generate HTML report content."""
        status_color = "#28a745" if scorecard.is_certified else "#dc3545"
        level_badge = self._get_level_badge(scorecard.certification_level)

        dimension_rows = ""
        for dim in sorted(scorecard.dimension_scores, key=lambda d: d.score, reverse=True):
            status_icon = "pass" if dim.passed else "fail"
            bar_width = int(dim.score)
            bar_color = self._get_score_color(dim.score)

            dimension_rows += f"""
            <tr>
                <td><span class="status-{status_icon}">{status_icon.upper()}</span></td>
                <td>{dim.name.replace('_', ' ').title()}</td>
                <td>{dim.score:.1f}</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {bar_width}%; background: {bar_color};"></div>
                    </div>
                </td>
                <td>{dim.threshold:.0f}</td>
                <td>{dim.weight * 100:.0f}%</td>
            </tr>
            """

        findings_html = ""
        if scorecard.critical_findings:
            findings_html = "<ul>" + "".join(
                f"<li>{f}</li>" for f in scorecard.critical_findings[:5]
            ) + "</ul>"
        else:
            findings_html = "<p>No critical findings.</p>"

        recommendations_html = ""
        if scorecard.top_recommendations:
            recommendations_html = "<ol>" + "".join(
                f"<li>{r}</li>" for r in scorecard.top_recommendations[:5]
            ) + "</ol>"
        else:
            recommendations_html = "<p>No recommendations at this time.</p>"

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Certification Report - {scorecard.agent_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .report-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #eee;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .title {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .certification-badge {{
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            color: white;
            background: {status_color};
        }}
        .score-card {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .score-item {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .score-value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }}
        .score-label {{
            color: #666;
            font-size: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .status-pass {{
            color: #28a745;
            font-weight: bold;
        }}
        .status-fail {{
            color: #dc3545;
            font-weight: bold;
        }}
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #eee;
            border-radius: 4px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 4px;
        }}
        .section {{
            margin: 30px 0;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        .level-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 14px;
        }}
        {level_badge['style']}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="header">
            <div>
                <div class="title">Agent Certification Report</div>
                <div style="color: #666;">
                    {scorecard.agent_id} v{scorecard.agent_version}
                </div>
            </div>
            <div class="certification-badge">
                {'CERTIFIED' if scorecard.is_certified else 'NOT CERTIFIED'}
            </div>
        </div>

        <div class="score-card">
            <div class="score-item">
                <div class="score-value">{scorecard.overall_score:.1f}</div>
                <div class="score-label">Overall Score</div>
            </div>
            <div class="score-item">
                <div class="score-value"><span class="level-badge">{scorecard.certification_level}</span></div>
                <div class="score-label">Certification Level</div>
            </div>
            <div class="score-item">
                <div class="score-value">{sum(1 for d in scorecard.dimension_scores if d.passed)}/12</div>
                <div class="score-label">Dimensions Passed</div>
            </div>
            <div class="score-item">
                <div class="score-value">{scorecard.valid_until.strftime('%Y-%m-%d') if scorecard.valid_until else 'N/A'}</div>
                <div class="score-label">Valid Until</div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Dimension Scores</div>
            <table>
                <thead>
                    <tr>
                        <th>Status</th>
                        <th>Dimension</th>
                        <th>Score</th>
                        <th>Progress</th>
                        <th>Threshold</th>
                        <th>Weight</th>
                    </tr>
                </thead>
                <tbody>
                    {dimension_rows}
                </tbody>
            </table>
        </div>

        <div class="section">
            <div class="section-title">Critical Findings</div>
            {findings_html}
        </div>

        <div class="section">
            <div class="section-title">Top Recommendations</div>
            {recommendations_html}
        </div>

        <div class="footer">
            <p>Certification ID: {scorecard.certification_id}</p>
            <p>Generated: {scorecard.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <p>GreenLang Agent Factory - Certification Pipeline v1.0.0</p>
        </div>
    </div>
</body>
</html>
        """

        return html

    def _get_level_badge(self, level: str) -> Dict[str, str]:
        """Get badge styling for certification level."""
        level_colors = {
            "PLATINUM": "#E5E4E2",
            "GOLD": "#FFD700",
            "SILVER": "#C0C0C0",
            "BRONZE": "#CD7F32",
            "PROVISIONAL": "#FFA500",
            "FAIL": "#dc3545",
        }
        color = level_colors.get(level, "#666")

        return {
            "style": f".level-badge {{ background: {color}; color: {'#333' if level in ['PLATINUM', 'GOLD', 'SILVER'] else 'white'}; }}"
        }

    def _get_score_color(self, score: float) -> str:
        """Get color for score value."""
        if score >= 90:
            return "#28a745"
        elif score >= 75:
            return "#ffc107"
        else:
            return "#dc3545"

    def export_json(
        self,
        scorecard: CertificationScorecard,
        output_path: str,
    ) -> None:
        """
        Export scorecard to JSON format.

        Args:
            scorecard: Certification scorecard
            output_path: Output file path
        """
        data = {
            "certification_id": scorecard.certification_id,
            "agent_id": scorecard.agent_id,
            "agent_version": scorecard.agent_version,
            "overall_score": scorecard.overall_score,
            "certification_level": scorecard.certification_level,
            "is_certified": scorecard.is_certified,
            "timestamp": scorecard.timestamp.isoformat(),
            "valid_until": scorecard.valid_until.isoformat() if scorecard.valid_until else None,
            "dimension_scores": [
                {
                    "name": d.name,
                    "score": d.score,
                    "weight": d.weight,
                    "weighted_score": d.weighted_score,
                    "passed": d.passed,
                    "threshold": d.threshold,
                }
                for d in scorecard.dimension_scores
            ],
            "critical_findings": scorecard.critical_findings,
            "top_recommendations": scorecard.top_recommendations,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported JSON report to {output_path}")

    def export_markdown(
        self,
        scorecard: CertificationScorecard,
        output_path: str,
    ) -> None:
        """
        Export scorecard to Markdown format.

        Args:
            scorecard: Certification scorecard
            output_path: Output file path
        """
        status = "CERTIFIED" if scorecard.is_certified else "NOT CERTIFIED"

        md = f"""# Agent Certification Report

## Summary

| Field | Value |
|-------|-------|
| Agent ID | {scorecard.agent_id} |
| Version | {scorecard.agent_version} |
| Certification ID | {scorecard.certification_id} |
| Overall Score | {scorecard.overall_score:.1f}/100 |
| Certification Level | {scorecard.certification_level} |
| Status | **{status}** |
| Valid Until | {scorecard.valid_until.strftime('%Y-%m-%d') if scorecard.valid_until else 'N/A'} |

## Dimension Scores

| Status | Dimension | Score | Threshold | Weight |
|--------|-----------|-------|-----------|--------|
"""

        for dim in sorted(scorecard.dimension_scores, key=lambda d: d.score, reverse=True):
            status_icon = "PASS" if dim.passed else "FAIL"
            md += f"| {status_icon} | {dim.name.replace('_', ' ').title()} | {dim.score:.1f} | {dim.threshold:.0f} | {dim.weight*100:.0f}% |\n"

        md += "\n## Critical Findings\n\n"
        if scorecard.critical_findings:
            for finding in scorecard.critical_findings:
                md += f"- {finding}\n"
        else:
            md += "No critical findings.\n"

        md += "\n## Recommendations\n\n"
        if scorecard.top_recommendations:
            for i, rec in enumerate(scorecard.top_recommendations, 1):
                md += f"{i}. {rec}\n"
        else:
            md += "No recommendations at this time.\n"

        md += f"""
---

*Generated: {scorecard.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}*

*GreenLang Agent Factory - Certification Pipeline v1.0.0*
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        logger.info(f"Exported Markdown report to {output_path}")
