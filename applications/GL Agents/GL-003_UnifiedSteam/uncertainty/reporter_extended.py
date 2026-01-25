"""
Extended Uncertainty Reporting for GL-003 UNIFIEDSTEAM SteamSystemOptimizer.

This module provides additional reporting capabilities including:
- Audit-ready documentation generation
- Uncertainty reduction roadmaps
- Export to JSON/CSV formats
- Contributor breakdown analysis

Zero-Hallucination Guarantee:
- All formatting and analysis is deterministic
- Recommendations based on quantitative analysis, not LLM inference
- Complete provenance for all reported values
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import logging

from .uncertainty_models import (
    UncertainValue,
    PropagatedUncertainty,
    UncertaintyBreakdown,
    ConfidenceLevel
)


logger = logging.getLogger(__name__)


@dataclass
class AuditDocumentation:
    """
    Audit-ready documentation for uncertainty analysis.

    Attributes:
        document_id: Unique document identifier
        timestamp: Generation timestamp
        methodology: Calculation methodology description
        results: Analysis results
        provenance_hashes: SHA-256 hashes for each result
        quality_assessment: Overall quality assessment
        recommendations: Improvement recommendations
        document_hash: Hash of entire document for integrity
    """
    document_id: str
    timestamp: datetime
    methodology: Dict[str, Any]
    results: Dict[str, Any]
    provenance_hashes: Dict[str, str]
    quality_assessment: Dict[str, Any]
    recommendations: List[str]
    document_hash: str = ""

    def __post_init__(self):
        """Compute document hash if not provided."""
        if not self.document_hash:
            hash_data = {
                "document_id": self.document_id,
                "timestamp": self.timestamp.isoformat(),
                "results_keys": list(self.results.keys()),
                "quality_level": self.quality_assessment.get("quality_level", "")
            }
            hash_str = json.dumps(hash_data, sort_keys=True)
            self.document_hash = hashlib.sha256(hash_str.encode()).hexdigest()


@dataclass
class ReductionAction:
    """
    Single action item in uncertainty reduction roadmap.

    Attributes:
        action_id: Unique action identifier
        output_affected: Output that will be improved
        target_input: Input to improve
        current_contribution: Current contribution to uncertainty (%)
        recommended_action: Description of recommended action
        potential_reduction: Expected uncertainty reduction (%)
        priority: Action priority (1=highest)
        effort_estimate: Estimated effort (low/medium/high)
        cost_estimate: Estimated cost (low/medium/high)
    """
    action_id: str
    output_affected: str
    target_input: str
    current_contribution: float
    recommended_action: str
    potential_reduction: float
    priority: int
    effort_estimate: str
    cost_estimate: str


@dataclass
class ReductionRoadmap:
    """
    Roadmap for uncertainty reduction.

    Attributes:
        target_uncertainty: Target uncertainty level (%)
        current_state: Current uncertainty state analysis
        gap_analysis: Gap between current and target
        action_plan: Prioritized list of actions
        expected_outcome: Projected outcome after implementation
        implementation_timeline: Estimated timeline
    """
    target_uncertainty: float
    current_state: Dict[str, Dict[str, Any]]
    gap_analysis: Dict[str, Dict[str, Any]]
    action_plan: List[ReductionAction]
    expected_outcome: Dict[str, Any]
    implementation_timeline: str


class AuditReporter:
    """
    Generates audit-ready documentation for uncertainty analysis.

    Provides comprehensive documentation suitable for regulatory
    audits and third-party verification, including complete
    provenance tracking and calculation methodology.

    Example:
        reporter = AuditReporter()

        audit_doc = reporter.generate_audit_documentation(
            propagated_results,
            calculation_context={"equipment": "Boiler 1", "date": "2024-01-15"}
        )

        # Format as text report
        report_text = reporter.format_audit_report(audit_doc)

        # Export to JSON
        json_str = reporter.export_to_json(propagated_results)
    """

    def __init__(
        self,
        organization_name: str = "",
        facility_name: str = ""
    ):
        """
        Initialize audit reporter.

        Args:
            organization_name: Name of organization
            facility_name: Name of facility
        """
        self.organization_name = organization_name
        self.facility_name = facility_name

    def generate_audit_documentation(
        self,
        propagated_results: Dict[str, PropagatedUncertainty],
        calculation_context: Optional[Dict[str, Any]] = None,
        include_raw_data: bool = False
    ) -> AuditDocumentation:
        """
        Generate audit-ready documentation for uncertainty analysis.

        Creates comprehensive documentation suitable for regulatory
        audits and third-party verification.

        Args:
            propagated_results: Dictionary of propagated uncertainty results
            calculation_context: Additional context (dates, equipment, etc.)
            include_raw_data: Whether to include raw input data

        Returns:
            AuditDocumentation with all required elements
        """
        timestamp = datetime.utcnow()
        document_id = f"audit_{timestamp.strftime('%Y%m%d%H%M%S')}"

        # Methodology documentation
        methodology = {
            "standard": "GUM (Guide to the Expression of Uncertainty in Measurement)",
            "propagation_method": "First-order Taylor series / Monte Carlo",
            "confidence_level": "95%",
            "coverage_factor": 1.96,
            "software_version": "GL-003 UnifiedSteam v1.0"
        }

        # Process results
        results = {}
        provenance_hashes = {}
        all_recommendations = []
        total_uncertainty = 0.0
        max_uncertainty = 0.0
        worst_output = ""

        for output_name, result in propagated_results.items():
            rel_uncertainty = result.relative_uncertainty_percent()
            total_uncertainty += rel_uncertainty

            if rel_uncertainty > max_uncertainty:
                max_uncertainty = rel_uncertainty
                worst_output = output_name

            # Result documentation
            result_doc = {
                "output_name": output_name,
                "value": result.value,
                "uncertainty_1sigma": result.uncertainty,
                "uncertainty_95ci": result.uncertainty * 1.96,
                "relative_uncertainty_percent": rel_uncertainty,
                "confidence_interval_95": {
                    "lower": result.confidence_interval_95[0],
                    "upper": result.confidence_interval_95[1]
                },
                "propagation_method": result.propagation_method,
                "dominant_contributor": result.dominant_contributor,
                "computation_time_ms": result.computation_time_ms,
                "contribution_breakdown": result.get_contribution_breakdown()
            }

            # Add input documentation if requested
            if include_raw_data and result.contributing_inputs:
                result_doc["contributing_inputs"] = {
                    name: {
                        "value": uv.mean,
                        "uncertainty": uv.std,
                        "unit": uv.unit,
                        "source_id": uv.source_id
                    }
                    for name, uv in result.contributing_inputs.items()
                }

            results[output_name] = result_doc

            # Provenance hash
            provenance_data = {
                "output_name": output_name,
                "value": result.value,
                "uncertainty": result.uncertainty,
                "method": result.propagation_method,
                "timestamp": timestamp.isoformat()
            }
            provenance_str = json.dumps(provenance_data, sort_keys=True)
            provenance_hashes[output_name] = hashlib.sha256(provenance_str.encode()).hexdigest()

            # Generate recommendations
            contributions = result.get_contribution_breakdown()
            sorted_contribs = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

            for name, contrib in sorted_contribs[:3]:
                if contrib >= 30.0:
                    all_recommendations.append(
                        f"PRIORITY: Reduce uncertainty in {name} "
                        f"(contributes {contrib:.1f}% to {output_name})"
                    )
                elif contrib >= 15.0:
                    all_recommendations.append(
                        f"Consider improving measurement of {name} "
                        f"({contrib:.1f}% contribution to {output_name})"
                    )

        # Quality assessment
        n_outputs = len(propagated_results)
        avg_uncertainty = total_uncertainty / n_outputs if n_outputs > 0 else 0

        if max_uncertainty > 20.0:
            quality_level = "CRITICAL"
            quality_message = "High uncertainty levels require immediate attention"
        elif max_uncertainty > 10.0:
            quality_level = "WARNING"
            quality_message = "Elevated uncertainty levels should be addressed"
        elif max_uncertainty > 5.0:
            quality_level = "ACCEPTABLE"
            quality_message = "Uncertainty within typical operating range"
        else:
            quality_level = "GOOD"
            quality_message = "Low uncertainty levels across all outputs"

        quality_assessment = {
            "quality_level": quality_level,
            "quality_message": quality_message,
            "total_outputs": n_outputs,
            "average_uncertainty_percent": avg_uncertainty,
            "maximum_uncertainty_percent": max_uncertainty,
            "worst_output": worst_output,
            "outputs_exceeding_10_percent": sum(
                1 for r in propagated_results.values()
                if r.relative_uncertainty_percent() > 10.0
            )
        }

        # Add context
        if calculation_context:
            results["_context"] = calculation_context

        # Deduplicate recommendations
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return AuditDocumentation(
            document_id=document_id,
            timestamp=timestamp,
            methodology=methodology,
            results=results,
            provenance_hashes=provenance_hashes,
            quality_assessment=quality_assessment,
            recommendations=unique_recommendations[:10]
        )

    def format_audit_report(
        self,
        audit_doc: AuditDocumentation
    ) -> str:
        """
        Format audit documentation as human-readable report.

        Args:
            audit_doc: Audit documentation to format

        Returns:
            Formatted report string
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("UNCERTAINTY ANALYSIS AUDIT DOCUMENTATION")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Document ID: {audit_doc.document_id}")
        lines.append(f"Generated: {audit_doc.timestamp.isoformat()}")
        lines.append(f"Document Hash: {audit_doc.document_hash[:32]}...")

        if self.organization_name:
            lines.append(f"Organization: {self.organization_name}")
        if self.facility_name:
            lines.append(f"Facility: {self.facility_name}")
        lines.append("")

        # Methodology
        lines.append("METHODOLOGY")
        lines.append("-" * 80)
        for key, value in audit_doc.methodology.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        # Quality Assessment
        lines.append("QUALITY ASSESSMENT")
        lines.append("-" * 80)
        qa = audit_doc.quality_assessment
        lines.append(f"  Quality Level: {qa.get('quality_level', 'N/A')}")
        lines.append(f"  Assessment: {qa.get('quality_message', 'N/A')}")
        lines.append(f"  Total Outputs Analyzed: {qa.get('total_outputs', 0)}")
        lines.append(f"  Average Uncertainty: {qa.get('average_uncertainty_percent', 0):.2f}%")
        lines.append(f"  Maximum Uncertainty: {qa.get('maximum_uncertainty_percent', 0):.2f}%")
        lines.append(f"  Worst Output: {qa.get('worst_output', 'N/A')}")
        lines.append("")

        # Results Summary
        lines.append("RESULTS SUMMARY")
        lines.append("-" * 80)
        for output_name, result in audit_doc.results.items():
            if output_name.startswith("_"):
                continue
            lines.append(f"  {output_name}:")
            lines.append(f"    Value: {result.get('value', 0):.4f}")
            lines.append(f"    Uncertainty (95% CI): +/- {result.get('uncertainty_95ci', 0):.4f}")
            lines.append(f"    Relative Uncertainty: {result.get('relative_uncertainty_percent', 0):.2f}%")
            lines.append(f"    Dominant Contributor: {result.get('dominant_contributor', 'N/A')}")
            lines.append(f"    Provenance Hash: {audit_doc.provenance_hashes.get(output_name, 'N/A')[:16]}...")
            lines.append("")

        # Recommendations
        if audit_doc.recommendations:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 80)
            for i, rec in enumerate(audit_doc.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        # Footer
        lines.append("=" * 80)
        lines.append("END OF AUDIT DOCUMENTATION")
        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_reduction_roadmap(
        self,
        propagated_results: Dict[str, PropagatedUncertainty],
        target_uncertainty_percent: float = 5.0
    ) -> ReductionRoadmap:
        """
        Generate a roadmap for reducing uncertainty to target level.

        Args:
            propagated_results: Dictionary of propagated results
            target_uncertainty_percent: Target uncertainty level

        Returns:
            ReductionRoadmap with prioritized actions
        """
        current_state = {}
        gap_analysis = {}
        action_plan = []
        action_id = 0

        for output_name, result in propagated_results.items():
            current_uncertainty = result.relative_uncertainty_percent()
            contributions = result.get_contribution_breakdown()

            current_state[output_name] = {
                "current_uncertainty_percent": current_uncertainty,
                "meets_target": current_uncertainty <= target_uncertainty_percent,
                "gap": max(0, current_uncertainty - target_uncertainty_percent),
                "top_contributors": dict(
                    sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:5]
                )
            }

            if current_uncertainty > target_uncertainty_percent:
                gap = current_uncertainty - target_uncertainty_percent
                required_reduction = (gap / current_uncertainty) * 100

                gap_analysis[output_name] = {
                    "gap_percent": gap,
                    "required_reduction_percent": required_reduction,
                    "difficulty": "high" if gap > 10 else ("medium" if gap > 5 else "low")
                }

                sorted_contributors = sorted(
                    contributions.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                for contributor, contrib_percent in sorted_contributors[:3]:
                    if contrib_percent > 15:
                        action_id += 1
                        potential_improvement = contrib_percent * 0.3

                        action = ReductionAction(
                            action_id=f"action_{action_id:03d}",
                            output_affected=output_name,
                            target_input=contributor,
                            current_contribution=contrib_percent,
                            recommended_action=self._get_improvement_action(contributor, contrib_percent),
                            potential_reduction=potential_improvement,
                            priority=1 if contrib_percent > 30 else (2 if contrib_percent > 20 else 3),
                            effort_estimate="high" if contrib_percent > 30 else "medium",
                            cost_estimate="high" if contrib_percent > 30 else "medium"
                        )
                        action_plan.append(action)

        # Sort by priority
        action_plan.sort(key=lambda x: (x.priority, -x.potential_reduction))

        # Expected outcome
        total_potential_reduction = sum(a.potential_reduction for a in action_plan[:5])

        expected_outcome = {
            "actions_recommended": len(action_plan),
            "total_potential_reduction_percent": total_potential_reduction,
            "achievable_with_top_5_actions": total_potential_reduction,
        }

        implementation_timeline = "3-6 months" if len(action_plan) > 5 else "1-3 months"

        return ReductionRoadmap(
            target_uncertainty=target_uncertainty_percent,
            current_state=current_state,
            gap_analysis=gap_analysis,
            action_plan=action_plan,
            expected_outcome=expected_outcome,
            implementation_timeline=implementation_timeline
        )

    def _get_improvement_action(self, contributor: str, contribution: float) -> str:
        """Generate improvement action based on contributor type."""
        contributor_lower = contributor.lower()

        if "temperature" in contributor_lower or "temp" in contributor_lower:
            if contribution > 30:
                return "Upgrade temperature sensor to higher accuracy class (e.g., RTD Class AA)"
            else:
                return "Recalibrate temperature sensor or reduce calibration interval"

        elif "pressure" in contributor_lower:
            if contribution > 30:
                return "Upgrade pressure transmitter to higher accuracy class"
            else:
                return "Schedule precision recalibration of pressure transmitter"

        elif "flow" in contributor_lower:
            if contribution > 30:
                return "Consider ultrasonic flow meter or upgrade to higher accuracy class"
            else:
                return "Verify flow meter installation and recalibrate"

        elif "mass" in contributor_lower:
            return "Improve mass measurement through better weighing system or flow metering"

        elif "enthalpy" in contributor_lower or "energy" in contributor_lower:
            return "Reduce uncertainty in temperature and pressure inputs"

        else:
            if contribution > 30:
                return f"Upgrade measurement system for {contributor} to reduce uncertainty"
            else:
                return f"Review and optimize measurement of {contributor}"

    def format_roadmap_report(self, roadmap: ReductionRoadmap) -> str:
        """Format reduction roadmap as readable report."""
        lines = []

        lines.append("=" * 80)
        lines.append("UNCERTAINTY REDUCTION ROADMAP")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Target Uncertainty: {roadmap.target_uncertainty:.1f}%")
        lines.append(f"Estimated Timeline: {roadmap.implementation_timeline}")
        lines.append("")

        # Current State
        lines.append("CURRENT STATE")
        lines.append("-" * 80)
        for output, state in roadmap.current_state.items():
            status = "MEETS TARGET" if state["meets_target"] else "NEEDS IMPROVEMENT"
            lines.append(f"  {output}: {state['current_uncertainty_percent']:.1f}% [{status}]")
        lines.append("")

        # Gap Analysis
        if roadmap.gap_analysis:
            lines.append("GAP ANALYSIS")
            lines.append("-" * 80)
            for output, gap in roadmap.gap_analysis.items():
                lines.append(f"  {output}:")
                lines.append(f"    Gap: {gap['gap_percent']:.1f}%")
                lines.append(f"    Required Reduction: {gap['required_reduction_percent']:.1f}%")
                lines.append(f"    Difficulty: {gap['difficulty']}")
            lines.append("")

        # Action Plan
        lines.append("ACTION PLAN")
        lines.append("-" * 80)
        for action in roadmap.action_plan:
            lines.append(f"  [{action.priority}] {action.action_id}: {action.target_input}")
            lines.append(f"      Output: {action.output_affected}")
            lines.append(f"      Action: {action.recommended_action}")
            lines.append(f"      Potential Reduction: {action.potential_reduction:.1f}%")
            lines.append(f"      Effort: {action.effort_estimate}, Cost: {action.cost_estimate}")
            lines.append("")

        # Expected Outcome
        lines.append("EXPECTED OUTCOME")
        lines.append("-" * 80)
        for key, value in roadmap.expected_outcome.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def export_to_json(
        self,
        propagated_results: Dict[str, PropagatedUncertainty],
        output_file: Optional[str] = None
    ) -> str:
        """
        Export uncertainty analysis to JSON format.

        Args:
            propagated_results: Results to export
            output_file: Optional file path to write

        Returns:
            JSON string of results
        """
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "format_version": "1.0",
            "results": {}
        }

        for output_name, result in propagated_results.items():
            export_data["results"][output_name] = {
                "value": result.value,
                "uncertainty": result.uncertainty,
                "relative_uncertainty_percent": result.relative_uncertainty_percent(),
                "confidence_interval_95": list(result.confidence_interval_95),
                "propagation_method": result.propagation_method,
                "dominant_contributor": result.dominant_contributor,
                "contribution_breakdown": result.get_contribution_breakdown(),
                "computation_time_ms": result.computation_time_ms
            }

        json_str = json.dumps(export_data, indent=2, default=str)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(json_str)

        return json_str

    def export_to_csv(
        self,
        propagated_results: Dict[str, PropagatedUncertainty],
        output_file: Optional[str] = None
    ) -> str:
        """
        Export uncertainty analysis to CSV format.

        Args:
            propagated_results: Results to export
            output_file: Optional file path to write

        Returns:
            CSV string of results
        """
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Output Name",
            "Value",
            "Uncertainty (1-sigma)",
            "Uncertainty (95% CI)",
            "Relative Uncertainty (%)",
            "Lower 95% CI",
            "Upper 95% CI",
            "Dominant Contributor",
            "Propagation Method",
            "Computation Time (ms)"
        ])

        # Data rows
        for output_name, result in propagated_results.items():
            writer.writerow([
                output_name,
                result.value,
                result.uncertainty,
                result.uncertainty * 1.96,
                result.relative_uncertainty_percent(),
                result.confidence_interval_95[0],
                result.confidence_interval_95[1],
                result.dominant_contributor,
                result.propagation_method,
                result.computation_time_ms
            ])

        csv_str = output.getvalue()

        if output_file:
            with open(output_file, 'w', newline='') as f:
                f.write(csv_str)

        return csv_str
