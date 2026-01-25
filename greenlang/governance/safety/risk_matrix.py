r"""
RiskMatrix - Comprehensive Risk Assessment Framework for Process Heat Agents

This module implements a 5x5 risk matrix with severity/likelihood assessment
per IEC 61511 functional safety standards. Integrates HAZOP and FMEA outputs
for comprehensive process safety management.

Key Features:
- 5x5 risk matrix (Severity 1-5, Likelihood 1-5)
- Risk level calculation: LOW/MEDIUM/HIGH/CRITICAL
- Risk color mapping for visualization (green/yellow/orange/red)
- Risk heatmap generation
- Risk aggregation and trending
- Risk register with status tracking
- IEC 61511 acceptance criteria
- Integration with HAZOP and FMEA

Risk Level Matrix:
    Severity\Likelihood   1        2        3        4        5
    1 (Minor)             LOW      LOW      LOW      MEDIUM   MEDIUM
    2 (Significant)       LOW      LOW      MEDIUM   MEDIUM   HIGH
    3 (Serious)           LOW      MEDIUM   MEDIUM   HIGH     HIGH
    4 (Major)             MEDIUM   MEDIUM   HIGH     HIGH     CRITICAL
    5 (Catastrophic)      MEDIUM   HIGH     HIGH     CRITICAL CRITICAL

IEC 61511 Acceptance Criteria:
- CRITICAL: Immediate mitigation required (SIL 3-4)
- HIGH: Action required within 30 days (SIL 2-3)
- MEDIUM: Action required within 90 days (SIL 1-2)
- LOW: Monitor and review (No SIL required)

Reference:
- IEC 61511-1:2016 Functional Safety - Safety Instrumented Systems
- IEC 61882:2016 Hazard and Operability Studies (HAZOP)
- IEC 60812:2018 Failure Mode and Effects Analysis (FMEA)

Example:
    >>> from greenlang.safety.risk_matrix import RiskMatrix, RiskRegister
    >>> risk_level = RiskMatrix.calculate_risk_level(severity=4, likelihood=4)
    >>> print(risk_level)  # Output: CRITICAL
    >>> color = RiskMatrix.get_risk_color(risk_level)
    >>> print(color)  # Output: red
"""

from typing import Dict, List, Optional, Any, ClassVar
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import uuid
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(str, Enum):
    """Risk classification per IEC 61511."""
    SAFETY = "safety"              # Personnel, equipment injury
    ENVIRONMENTAL = "environmental"  # Emissions, spills, pollution
    OPERATIONAL = "operational"    # Downtime, quality loss
    COMPLIANCE = "compliance"      # Regulatory, financial penalties


class RiskStatus(str, Enum):
    """Risk management status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    MITIGATED = "mitigated"
    CLOSED = "closed"
    ACCEPTED = "accepted"


class SafetyIntegrityLevel(str, Enum):
    """IEC 61511 Safety Integrity Level mapping."""
    SIL_4 = "sil_4"  # 10,000-100,000x risk reduction
    SIL_3 = "sil_3"  # 1,000-10,000x risk reduction
    SIL_2 = "sil_2"  # 100-1,000x risk reduction
    SIL_1 = "sil_1"  # 10-100x risk reduction
    NO_SIL = "no_sil"  # No SIL required


# =============================================================================
# DATA MODELS
# =============================================================================

class RiskData(BaseModel):
    """Risk assessment data model."""
    risk_id: str = Field(default_factory=lambda: f"RISK-{uuid.uuid4().hex[:6].upper()}")
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    category: RiskCategory = Field(...)
    severity: int = Field(..., ge=1, le=5, description="Severity 1-5")
    likelihood: int = Field(..., ge=1, le=5, description="Likelihood 1-5")
    source: str = Field(default="", description="HAZOP, FMEA, or other source")
    source_id: str = Field(default="", description="Cross-reference ID")
    mitigation_strategy: str = Field(default="")
    responsible_party: str = Field(default="")
    target_mitigation_date: Optional[datetime] = Field(None)
    actual_mitigation_date: Optional[datetime] = Field(None)
    status: RiskStatus = Field(default=RiskStatus.OPEN)
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    required_sil: SafetyIntegrityLevel = Field(default=SafetyIntegrityLevel.NO_SIL)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    residual_severity: Optional[int] = Field(None, ge=1, le=5)
    residual_likelihood: Optional[int] = Field(None, ge=1, le=5)
    provenance_hash: str = Field(default="")

    @field_validator('severity', 'likelihood', 'residual_severity', 'residual_likelihood')
    def validate_risk_scores(cls, v):
        """Validate risk scores are in valid range."""
        if v is not None and not (1 <= v <= 5):
            raise ValueError("Risk scores must be between 1 and 5")
        return v


class RiskMatrix:
    """
    5x5 Risk Matrix definition.

    Static utility class for risk level calculations and color/SIL mappings.
    """

    # Risk level matrix: [severity-1][likelihood-1]
    RISK_MATRIX: ClassVar[List[List[RiskLevel]]] = [
        [RiskLevel.LOW, RiskLevel.LOW, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.MEDIUM],
        [RiskLevel.LOW, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.HIGH],
        [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.HIGH],
        [RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.HIGH, RiskLevel.CRITICAL],
        [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.CRITICAL],
    ]

    # Color mapping for visualization
    COLOR_MAP: ClassVar[Dict[RiskLevel, str]] = {
        RiskLevel.LOW: "green",
        RiskLevel.MEDIUM: "yellow",
        RiskLevel.HIGH: "orange",
        RiskLevel.CRITICAL: "red",
    }

    # SIL mapping per IEC 61511
    SIL_MAP: ClassVar[Dict[RiskLevel, SafetyIntegrityLevel]] = {
        RiskLevel.LOW: SafetyIntegrityLevel.NO_SIL,
        RiskLevel.MEDIUM: SafetyIntegrityLevel.SIL_1,
        RiskLevel.HIGH: SafetyIntegrityLevel.SIL_2,
        RiskLevel.CRITICAL: SafetyIntegrityLevel.SIL_3,
    }

    # Acceptance criteria (days for action)
    ACCEPTANCE_CRITERIA: ClassVar[Dict[RiskLevel, int]] = {
        RiskLevel.CRITICAL: 7,  # Immediate action (max 7 days)
        RiskLevel.HIGH: 30,     # Within 30 days
        RiskLevel.MEDIUM: 90,   # Within 90 days
        RiskLevel.LOW: 365,     # Annual review
    }

    @staticmethod
    def calculate_risk_level(severity: int, likelihood: int) -> RiskLevel:
        """
        Calculate risk level from severity and likelihood.

        Args:
            severity: 1-5 (1=Minor, 5=Catastrophic)
            likelihood: 1-5 (1=Remote, 5=Almost Certain)

        Returns:
            RiskLevel: LOW, MEDIUM, HIGH, or CRITICAL

        Raises:
            ValueError: If scores are outside 1-5 range
        """
        if not (1 <= severity <= 5 and 1 <= likelihood <= 5):
            raise ValueError("Severity and likelihood must be 1-5")

        return RiskMatrix.RISK_MATRIX[severity - 1][likelihood - 1]

    @staticmethod
    def get_risk_color(risk_level: RiskLevel) -> str:
        """
        Get visualization color for risk level.

        Args:
            risk_level: Risk level enum

        Returns:
            Color string: green, yellow, orange, or red
        """
        return RiskMatrix.COLOR_MAP.get(risk_level, "gray")

    @staticmethod
    def get_required_sil(risk_level: RiskLevel) -> SafetyIntegrityLevel:
        """
        Get required Safety Integrity Level per IEC 61511.

        Args:
            risk_level: Risk level enum

        Returns:
            SafetyIntegrityLevel: SIL level or NO_SIL
        """
        return RiskMatrix.SIL_MAP.get(risk_level, SafetyIntegrityLevel.NO_SIL)

    @staticmethod
    def get_acceptance_days(risk_level: RiskLevel) -> int:
        """
        Get days allowed for risk acceptance per IEC 61511.

        Args:
            risk_level: Risk level enum

        Returns:
            Number of days allowed for mitigation action
        """
        return RiskMatrix.ACCEPTANCE_CRITERIA.get(risk_level, 365)

    @staticmethod
    def generate_heatmap(risks: List[RiskData]) -> Dict[str, Any]:
        """
        Generate risk heatmap visualization data.

        Args:
            risks: List of risk assessments

        Returns:
            Dictionary with heatmap data:
            {
                'matrix': 5x5 array of risk counts,
                'colors': 5x5 array of colors,
                'summary': category breakdown
            }
        """
        # Initialize 5x5 matrix
        heatmap = [[0 for _ in range(5)] for _ in range(5)]
        colors = [[RiskMatrix.COLOR_MAP[RiskLevel.LOW] for _ in range(5)] for _ in range(5)]

        # Populate matrix with risk counts
        for risk in risks:
            if risk.status == RiskStatus.CLOSED:
                continue  # Skip closed risks

            severity_idx = risk.severity - 1
            likelihood_idx = risk.likelihood - 1
            heatmap[severity_idx][likelihood_idx] += 1

            # Update color
            risk_level = RiskMatrix.calculate_risk_level(risk.severity, risk.likelihood)
            colors[severity_idx][likelihood_idx] = RiskMatrix.get_risk_color(risk_level)

        # Calculate summary by category
        category_summary = {cat.value: 0 for cat in RiskCategory}
        for risk in risks:
            if risk.status != RiskStatus.CLOSED:
                category_summary[risk.category.value] += 1

        return {
            "matrix": heatmap,
            "colors": colors,
            "summary": category_summary,
            "generated_at": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def aggregate_risks(risks: List[RiskData]) -> Dict[str, Any]:
        """
        Aggregate risk statistics.

        Args:
            risks: List of risks to aggregate

        Returns:
            Aggregation summary with metrics:
            {
                'total_risks': count,
                'critical': count,
                'high': count,
                'medium': count,
                'low': count,
                'average_severity': float,
                'average_likelihood': float,
                'risks_overdue': count,
                'risks_at_target': count
            }
        """
        open_risks = [r for r in risks if r.status != RiskStatus.CLOSED]

        if not open_risks:
            return {
                "total_risks": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "average_severity": 0.0,
                "average_likelihood": 0.0,
                "risks_overdue": 0,
                "risks_at_target": 0,
            }

        # Count by risk level
        level_counts = {
            RiskLevel.CRITICAL: sum(1 for r in open_risks if r.risk_level == RiskLevel.CRITICAL),
            RiskLevel.HIGH: sum(1 for r in open_risks if r.risk_level == RiskLevel.HIGH),
            RiskLevel.MEDIUM: sum(1 for r in open_risks if r.risk_level == RiskLevel.MEDIUM),
            RiskLevel.LOW: sum(1 for r in open_risks if r.risk_level == RiskLevel.LOW),
        }

        # Calculate deadlines
        now = datetime.utcnow()
        risks_overdue = 0
        risks_at_target = 0

        for risk in open_risks:
            if risk.target_mitigation_date:
                if risk.target_mitigation_date < now:
                    risks_overdue += 1
                elif risk.target_mitigation_date <= now + timedelta(days=14):
                    risks_at_target += 1

        return {
            "total_risks": len(open_risks),
            "critical": level_counts[RiskLevel.CRITICAL],
            "high": level_counts[RiskLevel.HIGH],
            "medium": level_counts[RiskLevel.MEDIUM],
            "low": level_counts[RiskLevel.LOW],
            "average_severity": statistics.mean([r.severity for r in open_risks]),
            "average_likelihood": statistics.mean([r.likelihood for r in open_risks]),
            "risks_overdue": risks_overdue,
            "risks_at_target": risks_at_target,
        }


# =============================================================================
# RISK REGISTER
# =============================================================================

class RiskRegister:
    """
    Risk Register for process heat agents.

    Implements risk tracking, status management, and compliance reporting
    per IEC 61511 functional safety standards.

    Example:
        >>> register = RiskRegister()
        >>> risk = RiskData(
        ...     title="High temperature excursion",
        ...     category=RiskCategory.SAFETY,
        ...     severity=4,
        ...     likelihood=2
        ... )
        >>> register.add_risk(risk)
    """

    def __init__(self):
        """Initialize risk register."""
        self.risks: Dict[str, RiskData] = {}
        self.audit_trail: List[Dict[str, Any]] = []
        logger.info("RiskRegister initialized")

    def add_risk(self, risk: RiskData) -> RiskData:
        """
        Add a new risk to the register.

        Args:
            risk: RiskData model

        Returns:
            Added risk with calculated fields

        Raises:
            ValueError: If risk already exists
        """
        if risk.risk_id in self.risks:
            raise ValueError(f"Risk already exists: {risk.risk_id}")

        # Calculate risk level and SIL
        risk.risk_level = RiskMatrix.calculate_risk_level(risk.severity, risk.likelihood)
        risk.required_sil = RiskMatrix.get_required_sil(risk.risk_level)

        # Set target mitigation date based on acceptance criteria
        if not risk.target_mitigation_date:
            days_allowed = RiskMatrix.get_acceptance_days(risk.risk_level)
            risk.target_mitigation_date = datetime.utcnow() + timedelta(days=days_allowed)

        # Calculate provenance hash
        risk.provenance_hash = self._calculate_provenance(risk)

        self.risks[risk.risk_id] = risk
        self._log_audit("CREATED", risk.risk_id, {"action": "risk_created"})

        logger.info(
            f"Risk added: {risk.risk_id} ({risk.risk_level.value}) - {risk.title}"
        )
        return risk

    def update_risk(self, risk_id: str, updates: Dict[str, Any]) -> RiskData:
        """
        Update risk status and details.

        Args:
            risk_id: Risk identifier
            updates: Dictionary of field updates

        Returns:
            Updated RiskData

        Raises:
            ValueError: If risk not found
        """
        if risk_id not in self.risks:
            raise ValueError(f"Risk not found: {risk_id}")

        risk = self.risks[risk_id]
        old_status = risk.status

        # Apply updates
        for key, value in updates.items():
            if hasattr(risk, key):
                setattr(risk, key, value)

        risk.updated_at = datetime.utcnow()

        # Recalculate risk level if severity/likelihood changed
        if "severity" in updates or "likelihood" in updates:
            risk.risk_level = RiskMatrix.calculate_risk_level(risk.severity, risk.likelihood)
            risk.required_sil = RiskMatrix.get_required_sil(risk.risk_level)

        # Recalculate residual risk if applicable
        if risk.residual_severity and risk.residual_likelihood:
            residual_level = RiskMatrix.calculate_risk_level(
                risk.residual_severity,
                risk.residual_likelihood
            )
            logger.info(
                f"Risk {risk_id} residual level: {residual_level.value}"
            )

        # Update provenance hash
        risk.provenance_hash = self._calculate_provenance(risk)

        # Log audit trail
        self._log_audit("UPDATED", risk_id, {
            "old_status": old_status.value,
            "new_status": risk.status.value,
            "updates": updates
        })

        logger.info(f"Risk updated: {risk_id} (Status: {old_status.value} -> {risk.status.value})")
        return risk

    def get_open_risks(self, category: Optional[RiskCategory] = None) -> List[RiskData]:
        """
        Get open risks, optionally filtered by category.

        Args:
            category: Optional RiskCategory filter

        Returns:
            List of open RiskData sorted by severity

        Example:
            >>> safety_risks = register.get_open_risks(RiskCategory.SAFETY)
        """
        open_risks = [
            r for r in self.risks.values()
            if r.status in [RiskStatus.OPEN, RiskStatus.IN_PROGRESS]
        ]

        if category:
            open_risks = [r for r in open_risks if r.category == category]

        # Sort by severity (descending)
        return sorted(open_risks, key=lambda r: r.severity, reverse=True)

    def get_critical_risks(self) -> List[RiskData]:
        """
        Get critical risks requiring immediate action.

        Returns:
            List of CRITICAL risks sorted by due date
        """
        critical = [
            r for r in self.risks.values()
            if r.risk_level == RiskLevel.CRITICAL and r.status != RiskStatus.CLOSED
        ]
        return sorted(critical, key=lambda r: r.target_mitigation_date or datetime.max)

    def get_overdue_risks(self) -> List[RiskData]:
        """
        Get risks with overdue mitigation targets.

        Returns:
            List of overdue risks
        """
        now = datetime.utcnow()
        overdue = [
            r for r in self.risks.values()
            if r.target_mitigation_date and r.target_mitigation_date < now
            and r.status != RiskStatus.CLOSED
        ]
        return sorted(overdue, key=lambda r: r.target_mitigation_date)

    def import_from_hazop(self, hazop_deviations: List[Dict[str, Any]]) -> List[RiskData]:
        """
        Import risks from HAZOP study results.

        Args:
            hazop_deviations: List of HAZOP deviation dictionaries

        Returns:
            List of created RiskData objects

        Example:
            >>> hazop_data = analyzer.get_high_risk_deviations(study_id)
            >>> risks = register.import_from_hazop(hazop_data)
        """
        created_risks = []

        for hazop in hazop_deviations:
            risk = RiskData(
                title=f"HAZOP: {hazop.get('deviation_description', 'Unknown')}",
                description=f"Consequences: {', '.join(hazop.get('consequences', []))}",
                category=RiskCategory.SAFETY,
                severity=hazop.get("severity", 1),
                likelihood=hazop.get("likelihood", 1),
                source="HAZOP",
                source_id=hazop.get("deviation_id", ""),
                mitigation_strategy="; ".join(hazop.get("recommendations", [])),
            )

            created_risks.append(self.add_risk(risk))

        logger.info(f"Imported {len(created_risks)} risks from HAZOP")
        return created_risks

    def import_from_fmea(self, failure_modes: List[Dict[str, Any]]) -> List[RiskData]:
        """
        Import risks from FMEA study results.

        Args:
            failure_modes: List of FMEA failure mode dictionaries

        Returns:
            List of created RiskData objects
        """
        created_risks = []

        for fm in failure_modes:
            # Map FMEA RPN to risk level
            rpn = fm.get("rpn", 0)
            if rpn >= 200:
                severity = 5
                likelihood = 4
            elif rpn >= 100:
                severity = 4
                likelihood = 3
            else:
                severity = fm.get("severity", 1)
                likelihood = fm.get("occurrence", 1)

            risk = RiskData(
                title=f"FMEA: {fm.get('component_name', 'Unknown')} - {fm.get('failure_mode', '')}",
                description=f"End Effect: {fm.get('end_effect', '')}",
                category=RiskCategory.OPERATIONAL,
                severity=min(severity, 5),
                likelihood=min(likelihood, 5),
                source="FMEA",
                source_id=fm.get("fm_id", ""),
                mitigation_strategy=fm.get("recommended_action", ""),
                responsible_party=fm.get("responsibility", ""),
            )

            created_risks.append(self.add_risk(risk))

        logger.info(f"Imported {len(created_risks)} risks from FMEA")
        return created_risks

    def generate_report(self, as_json: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive risk register report.

        Args:
            as_json: If True, return JSON-serializable dict

        Returns:
            Report dictionary with summary and details

        Example:
            >>> report = register.generate_report()
            >>> print(f"Total Risks: {report['summary']['total_risks']}")
        """
        now = datetime.utcnow()
        open_risks = [r for r in self.risks.values() if r.status != RiskStatus.CLOSED]
        critical_risks = self.get_critical_risks()
        overdue_risks = self.get_overdue_risks()

        # Category breakdown
        category_breakdown = {}
        for category in RiskCategory:
            count = sum(1 for r in open_risks if r.category == category)
            category_breakdown[category.value] = count

        # Risk trend (simplified - risks created last 30 days)
        thirty_days_ago = now - timedelta(days=30)
        new_risks_30d = sum(1 for r in open_risks if r.created_at > thirty_days_ago)

        return {
            "report_generated": now.isoformat(),
            "summary": RiskMatrix.aggregate_risks(open_risks),
            "critical_risks_count": len(critical_risks),
            "overdue_risks_count": len(overdue_risks),
            "category_breakdown": category_breakdown,
            "new_risks_30_days": new_risks_30d,
            "heatmap": RiskMatrix.generate_heatmap(open_risks),
            "critical_risks": [
                {
                    "risk_id": r.risk_id,
                    "title": r.title,
                    "risk_level": r.risk_level.value,
                    "required_sil": r.required_sil.value,
                    "target_date": r.target_mitigation_date.isoformat() if r.target_mitigation_date else None,
                    "status": r.status.value,
                }
                for r in critical_risks[:10]  # Top 10
            ],
            "overdue_risks": [
                {
                    "risk_id": r.risk_id,
                    "title": r.title,
                    "target_date": r.target_mitigation_date.isoformat() if r.target_mitigation_date else None,
                    "days_overdue": (now - r.target_mitigation_date).days if r.target_mitigation_date else 0,
                }
                for r in overdue_risks[:10]  # Top 10
            ],
            "audit_trail_summary": {
                "total_events": len(self.audit_trail),
                "last_event": self.audit_trail[-1]["timestamp"].isoformat() if self.audit_trail else None,
            }
        }

    def export_to_compliance_report(self, format_type: str = "text") -> str:
        """
        Export risk register as compliance report.

        Args:
            format_type: "text", "csv", or "json"

        Returns:
            Formatted report string

        Example:
            >>> report_text = register.export_to_compliance_report(format_type="text")
            >>> print(report_text)
        """
        report = self.generate_report()
        summary = report["summary"]

        if format_type == "text":
            lines = [
                "=" * 70,
                "RISK REGISTER COMPLIANCE REPORT",
                "=" * 70,
                f"Report Generated: {report['report_generated']}",
                "",
                "EXECUTIVE SUMMARY",
                "-" * 70,
                f"Total Open Risks: {summary['total_risks']}",
                f"  CRITICAL: {summary['critical']} (Immediate action required)",
                f"  HIGH:     {summary['high']} (Action within 30 days)",
                f"  MEDIUM:   {summary['medium']} (Action within 90 days)",
                f"  LOW:      {summary['low']} (Monitor and review)",
                "",
                f"Risks Overdue for Mitigation: {report['overdue_risks_count']}",
                f"New Risks Last 30 Days: {report['new_risks_30_days']}",
                "",
                "CRITICAL RISKS REQUIRING IMMEDIATE ACTION",
                "-" * 70,
            ]

            for risk in report["critical_risks"]:
                lines.append(
                    f"  {risk['risk_id']}: {risk['title']}\n"
                    f"    Required SIL: {risk['required_sil']}\n"
                    f"    Target: {risk['target_date']}"
                )

            return "\n".join(lines)

        elif format_type == "csv":
            lines = ["risk_id,title,category,severity,likelihood,status,target_date"]
            for risk in self.risks.values():
                if risk.status != RiskStatus.CLOSED:
                    lines.append(
                        f"{risk.risk_id},{risk.title},{risk.category.value},"
                        f"{risk.severity},{risk.likelihood},{risk.status.value},"
                        f"{risk.target_mitigation_date.isoformat() if risk.target_mitigation_date else ''}"
                    )
            return "\n".join(lines)

        else:  # json
            import json
            return json.dumps(report, indent=2, default=str)

    def _calculate_provenance(self, risk: RiskData) -> str:
        """Calculate provenance hash for audit trail."""
        data_str = f"{risk.risk_id}|{risk.title}|{risk.severity}|{risk.likelihood}|{risk.status.value}"
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _log_audit(self, event_type: str, risk_id: str, details: Dict[str, Any]) -> None:
        """Log event to audit trail."""
        self.audit_trail.append({
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "risk_id": risk_id,
            "details": details,
        })


if __name__ == "__main__":
    # Example usage
    print("RiskMatrix and RiskRegister modules loaded")
