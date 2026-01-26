# -*- coding: utf-8 -*-
"""
GL-OPS-X-005: Continuous Improvement Agent
===========================================

Identifies improvement opportunities in emissions reduction, operational
efficiency, and cost savings through data-driven analysis.

Capabilities:
    - Gap analysis against targets and benchmarks
    - Root cause analysis for performance issues
    - Improvement opportunity identification
    - Impact assessment and prioritization
    - Implementation tracking
    - Progress monitoring and reporting

Zero-Hallucination Guarantees:
    - All analysis uses deterministic calculations
    - Complete provenance tracking with SHA-256 hashes
    - No LLM calls in the calculation path
    - All recommendations traceable to source data

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class ImprovementCategory(str, Enum):
    """Categories of improvement opportunities."""
    EMISSIONS_REDUCTION = "emissions_reduction"
    ENERGY_EFFICIENCY = "energy_efficiency"
    COST_REDUCTION = "cost_reduction"
    PROCESS_OPTIMIZATION = "process_optimization"
    WASTE_REDUCTION = "waste_reduction"
    RENEWABLE_ADOPTION = "renewable_adoption"
    EQUIPMENT_UPGRADE = "equipment_upgrade"
    BEHAVIORAL = "behavioral"


class ImprovementPriority(str, Enum):
    """Priority levels for improvements."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


class ImplementationStatus(str, Enum):
    """Status of improvement implementation."""
    IDENTIFIED = "identified"
    ASSESSED = "assessed"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class ImpactType(str, Enum):
    """Types of impact from improvements."""
    EMISSIONS = "emissions"  # kg CO2e
    ENERGY = "energy"  # kWh
    COST = "cost"  # currency
    WATER = "water"  # liters
    WASTE = "waste"  # kg


# =============================================================================
# Pydantic Models
# =============================================================================

class PerformanceMetric(BaseModel):
    """A performance metric for analysis."""
    metric_id: str = Field(..., description="Metric identifier")
    name: str = Field(..., description="Metric name")
    facility_id: str = Field(..., description="Facility identifier")

    # Values
    current_value: float = Field(..., description="Current value")
    target_value: float = Field(..., description="Target value")
    baseline_value: float = Field(..., description="Baseline value")

    # Unit
    unit: str = Field(..., description="Unit of measurement")

    # Time period
    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")

    # Gap analysis
    gap_to_target: Optional[float] = Field(None, description="Gap to target")
    gap_percent: Optional[float] = Field(None, description="Gap percentage")
    trend: Optional[str] = Field(None, description="Trend direction")


class ImpactAssessment(BaseModel):
    """Assessment of improvement impact."""
    impact_type: ImpactType = Field(..., description="Type of impact")
    annual_reduction: float = Field(..., description="Annual reduction amount")
    unit: str = Field(..., description="Unit of measurement")
    confidence: float = Field(default=0.8, ge=0, le=1, description="Assessment confidence")
    payback_years: Optional[float] = Field(None, description="Payback period in years")
    npv: Optional[float] = Field(None, description="Net present value")
    assumptions: List[str] = Field(default_factory=list, description="Key assumptions")


class ImprovementOpportunity(BaseModel):
    """An identified improvement opportunity."""
    opportunity_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    facility_id: str = Field(..., description="Facility identifier")
    category: ImprovementCategory = Field(..., description="Improvement category")
    priority: ImprovementPriority = Field(default=ImprovementPriority.MEDIUM)
    status: ImplementationStatus = Field(default=ImplementationStatus.IDENTIFIED)

    # Description
    title: str = Field(..., description="Opportunity title")
    description: str = Field(..., description="Detailed description")
    root_cause: Optional[str] = Field(None, description="Root cause if applicable")

    # Impact
    impacts: List[ImpactAssessment] = Field(default_factory=list)
    total_emissions_reduction_kg: float = Field(default=0.0, description="Total emissions reduction")
    total_cost_savings: float = Field(default=0.0, description="Total cost savings")
    total_energy_savings_kwh: float = Field(default=0.0, description="Total energy savings")

    # Implementation
    implementation_cost: float = Field(default=0.0, ge=0, description="Implementation cost")
    implementation_effort: str = Field(default="medium", description="low/medium/high")
    implementation_time_months: int = Field(default=3, ge=0, description="Implementation timeline")

    # Requirements
    required_resources: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)

    # Tracking
    identified_date: datetime = Field(default_factory=DeterministicClock.now)
    target_completion_date: Optional[datetime] = Field(None)
    actual_completion_date: Optional[datetime] = Field(None)
    progress_percent: float = Field(default=0.0, ge=0, le=100)

    # Source
    source_metrics: List[str] = Field(default_factory=list, description="Source metric IDs")
    confidence_score: float = Field(default=0.8, ge=0, le=1)

    # Metadata
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GapAnalysis(BaseModel):
    """Result of gap analysis."""
    facility_id: str = Field(..., description="Facility identifier")
    analysis_date: datetime = Field(default_factory=DeterministicClock.now)

    # Gaps identified
    emissions_gap_kg: float = Field(default=0.0, description="Emissions gap to target")
    energy_gap_kwh: float = Field(default=0.0, description="Energy gap to target")
    cost_gap: float = Field(default=0.0, description="Cost gap to target")

    # Gap percentages
    emissions_gap_percent: float = Field(default=0.0)
    energy_gap_percent: float = Field(default=0.0)
    cost_gap_percent: float = Field(default=0.0)

    # Metrics analyzed
    metrics_analyzed: int = Field(default=0)
    metrics_on_target: int = Field(default=0)
    metrics_below_target: int = Field(default=0)

    # Contributing factors
    top_contributors: List[Dict[str, Any]] = Field(default_factory=list)


class ImprovementInput(BaseModel):
    """Input for the Continuous Improvement Agent."""
    operation: str = Field(..., description="Operation to perform")
    metrics: List[PerformanceMetric] = Field(default_factory=list, description="Performance metrics")
    facility_id: Optional[str] = Field(None, description="Facility identifier")
    opportunity_id: Optional[str] = Field(None, description="Opportunity ID")
    opportunity: Optional[ImprovementOpportunity] = Field(None, description="Opportunity data")
    new_status: Optional[ImplementationStatus] = Field(None, description="New status")
    progress_percent: Optional[float] = Field(None, description="Progress percentage")
    category_filter: Optional[ImprovementCategory] = Field(None)
    priority_filter: Optional[ImprovementPriority] = Field(None)

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Validate operation is supported."""
        valid_ops = {
            'analyze_gaps', 'identify_opportunities', 'assess_impact',
            'prioritize', 'add_opportunity', 'update_status',
            'get_opportunities', 'get_progress', 'get_summary',
            'get_statistics'
        }
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class ImprovementOutput(BaseModel):
    """Output from the Continuous Improvement Agent."""
    success: bool = Field(..., description="Whether operation succeeded")
    operation: str = Field(..., description="Operation performed")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


# =============================================================================
# Continuous Improvement Agent Implementation
# =============================================================================

class ContinuousImprovementAgent(BaseAgent):
    """
    GL-OPS-X-005: Continuous Improvement Agent

    Identifies improvement opportunities in emissions reduction, operational
    efficiency, and cost savings through data-driven analysis.

    Zero-Hallucination Guarantees:
        - All analysis uses deterministic calculations
        - Complete provenance tracking with SHA-256 hashes
        - No LLM calls in the calculation path
        - All recommendations traceable to source data

    Usage:
        agent = ContinuousImprovementAgent()

        # Analyze gaps
        result = agent.run({
            "operation": "analyze_gaps",
            "facility_id": "FAC-001",
            "metrics": [...]
        })

        # Identify opportunities
        result = agent.run({
            "operation": "identify_opportunities",
            "facility_id": "FAC-001",
            "metrics": [...]
        })
    """

    AGENT_ID = "GL-OPS-X-005"
    AGENT_NAME = "Continuous Improvement Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Continuous Improvement Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Continuous improvement opportunity identification",
                version=self.VERSION,
                parameters={
                    "gap_threshold_percent": 5.0,
                    "min_savings_threshold": 1000.0,
                    "default_payback_threshold_years": 3.0,
                }
            )
        super().__init__(config)

        # Opportunities storage
        self._opportunities: Dict[str, ImprovementOpportunity] = {}

        # Gap analysis results
        self._gap_analyses: Dict[str, GapAnalysis] = {}

        # Performance metrics history
        self._metrics_history: Dict[str, List[PerformanceMetric]] = defaultdict(list)

        # Statistics
        self._total_opportunities_identified = 0
        self._total_savings_identified = 0.0
        self._total_emissions_reduction_identified = 0.0

        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute continuous improvement operations."""
        start_time = time.time()

        try:
            ci_input = ImprovementInput(**input_data)
            operation = ci_input.operation

            result_data = self._route_operation(ci_input)

            provenance_hash = self._compute_provenance_hash(input_data, result_data)
            processing_time_ms = (time.time() - start_time) * 1000

            output = ImprovementOutput(
                success=True,
                operation=operation,
                data=result_data,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
            )

        except Exception as e:
            self.logger.error(f"Continuous improvement operation failed: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000

            return AgentResult(
                success=False,
                error=str(e),
                data={
                    "operation": input_data.get("operation", "unknown"),
                    "processing_time_ms": processing_time_ms,
                },
            )

    def _route_operation(self, ci_input: ImprovementInput) -> Dict[str, Any]:
        """Route to appropriate operation handler."""
        operation = ci_input.operation

        if operation == "analyze_gaps":
            return self._handle_analyze_gaps(ci_input.facility_id, ci_input.metrics)
        elif operation == "identify_opportunities":
            return self._handle_identify_opportunities(ci_input.facility_id, ci_input.metrics)
        elif operation == "assess_impact":
            return self._handle_assess_impact(ci_input.opportunity_id)
        elif operation == "prioritize":
            return self._handle_prioritize(ci_input.facility_id)
        elif operation == "add_opportunity":
            return self._handle_add_opportunity(ci_input.opportunity)
        elif operation == "update_status":
            return self._handle_update_status(
                ci_input.opportunity_id,
                ci_input.new_status,
                ci_input.progress_percent,
            )
        elif operation == "get_opportunities":
            return self._handle_get_opportunities(
                ci_input.facility_id,
                ci_input.category_filter,
                ci_input.priority_filter,
            )
        elif operation == "get_progress":
            return self._handle_get_progress(ci_input.facility_id)
        elif operation == "get_summary":
            return self._handle_get_summary(ci_input.facility_id)
        elif operation == "get_statistics":
            return self._handle_get_statistics()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # =========================================================================
    # Gap Analysis
    # =========================================================================

    def _handle_analyze_gaps(
        self,
        facility_id: Optional[str],
        metrics: List[PerformanceMetric],
    ) -> Dict[str, Any]:
        """Analyze performance gaps against targets."""
        if not facility_id:
            return {"error": "facility_id is required"}

        # Store metrics
        for metric in metrics:
            self._metrics_history[facility_id].append(metric)
            # Calculate gap
            metric.gap_to_target = metric.current_value - metric.target_value
            if metric.target_value != 0:
                metric.gap_percent = (metric.gap_to_target / metric.target_value) * 100

        # Perform gap analysis
        gap_analysis = GapAnalysis(facility_id=facility_id)

        emissions_gaps = []
        energy_gaps = []
        cost_gaps = []

        for metric in metrics:
            # Classify metrics by type and aggregate gaps
            name_lower = metric.name.lower()

            if "emission" in name_lower or "co2" in name_lower:
                if metric.gap_to_target:
                    emissions_gaps.append(metric)
            elif "energy" in name_lower or "kwh" in name_lower:
                if metric.gap_to_target:
                    energy_gaps.append(metric)
            elif "cost" in name_lower:
                if metric.gap_to_target:
                    cost_gaps.append(metric)

        # Calculate totals
        gap_analysis.emissions_gap_kg = sum(
            m.gap_to_target for m in emissions_gaps if m.gap_to_target
        )
        gap_analysis.energy_gap_kwh = sum(
            m.gap_to_target for m in energy_gaps if m.gap_to_target
        )
        gap_analysis.cost_gap = sum(
            m.gap_to_target for m in cost_gaps if m.gap_to_target
        )

        # Count metrics
        gap_analysis.metrics_analyzed = len(metrics)
        gap_analysis.metrics_on_target = sum(
            1 for m in metrics
            if m.gap_percent is not None and abs(m.gap_percent) < self.config.parameters.get("gap_threshold_percent", 5.0)
        )
        gap_analysis.metrics_below_target = gap_analysis.metrics_analyzed - gap_analysis.metrics_on_target

        # Identify top contributors
        all_gaps = [(m, abs(m.gap_percent or 0)) for m in metrics if m.gap_percent]
        all_gaps.sort(key=lambda x: x[1], reverse=True)

        gap_analysis.top_contributors = [
            {
                "metric_id": m.metric_id,
                "name": m.name,
                "gap_percent": round(m.gap_percent, 2) if m.gap_percent else 0,
                "current_value": m.current_value,
                "target_value": m.target_value,
            }
            for m, _ in all_gaps[:5]
        ]

        # Store analysis
        self._gap_analyses[facility_id] = gap_analysis

        return gap_analysis.model_dump()

    # =========================================================================
    # Opportunity Identification
    # =========================================================================

    def _handle_identify_opportunities(
        self,
        facility_id: Optional[str],
        metrics: List[PerformanceMetric],
    ) -> Dict[str, Any]:
        """Identify improvement opportunities from performance data."""
        if not facility_id:
            return {"error": "facility_id is required"}

        # First perform gap analysis
        gap_result = self._handle_analyze_gaps(facility_id, metrics)

        opportunities = []
        gap_threshold = self.config.parameters.get("gap_threshold_percent", 5.0)

        for metric in metrics:
            if metric.gap_percent is None:
                continue

            # Only identify opportunities for significant gaps
            if abs(metric.gap_percent) <= gap_threshold:
                continue

            # Determine category and create opportunity
            category = self._categorize_metric(metric)
            opportunity = self._create_opportunity_from_metric(metric, category)

            if opportunity:
                self._opportunities[opportunity.opportunity_id] = opportunity
                opportunities.append(opportunity)
                self._total_opportunities_identified += 1
                self._total_savings_identified += opportunity.total_cost_savings
                self._total_emissions_reduction_identified += opportunity.total_emissions_reduction_kg

        return {
            "facility_id": facility_id,
            "opportunities_identified": len(opportunities),
            "opportunities": [o.model_dump() for o in opportunities],
            "gap_analysis": gap_result,
        }

    def _categorize_metric(self, metric: PerformanceMetric) -> ImprovementCategory:
        """Categorize a metric to determine improvement category."""
        name_lower = metric.name.lower()

        if "emission" in name_lower or "co2" in name_lower or "ghg" in name_lower:
            return ImprovementCategory.EMISSIONS_REDUCTION
        elif "energy" in name_lower or "kwh" in name_lower or "electricity" in name_lower:
            return ImprovementCategory.ENERGY_EFFICIENCY
        elif "cost" in name_lower or "spend" in name_lower:
            return ImprovementCategory.COST_REDUCTION
        elif "waste" in name_lower:
            return ImprovementCategory.WASTE_REDUCTION
        elif "renewable" in name_lower or "solar" in name_lower or "wind" in name_lower:
            return ImprovementCategory.RENEWABLE_ADOPTION
        else:
            return ImprovementCategory.PROCESS_OPTIMIZATION

    def _create_opportunity_from_metric(
        self,
        metric: PerformanceMetric,
        category: ImprovementCategory,
    ) -> Optional[ImprovementOpportunity]:
        """Create improvement opportunity from a metric gap."""
        if metric.gap_to_target is None or metric.gap_to_target == 0:
            return None

        # Calculate potential improvement
        gap = abs(metric.gap_to_target)
        improvement_target = gap * 0.5  # Target 50% gap closure

        # Determine priority based on gap size
        if metric.gap_percent and abs(metric.gap_percent) > 20:
            priority = ImprovementPriority.HIGH
        elif metric.gap_percent and abs(metric.gap_percent) > 10:
            priority = ImprovementPriority.MEDIUM
        else:
            priority = ImprovementPriority.LOW

        # Create impact assessment
        impacts = []
        emissions_reduction = 0.0
        cost_savings = 0.0
        energy_savings = 0.0

        if category == ImprovementCategory.EMISSIONS_REDUCTION:
            emissions_reduction = improvement_target * 12  # Annualize
            impacts.append(ImpactAssessment(
                impact_type=ImpactType.EMISSIONS,
                annual_reduction=emissions_reduction,
                unit="kg CO2e",
            ))
        elif category == ImprovementCategory.ENERGY_EFFICIENCY:
            energy_savings = improvement_target * 12
            cost_savings = energy_savings * 0.10  # $0.10/kWh
            emissions_reduction = energy_savings * 0.4  # 400g CO2/kWh
            impacts.extend([
                ImpactAssessment(
                    impact_type=ImpactType.ENERGY,
                    annual_reduction=energy_savings,
                    unit="kWh",
                ),
                ImpactAssessment(
                    impact_type=ImpactType.COST,
                    annual_reduction=cost_savings,
                    unit="USD",
                ),
            ])
        elif category == ImprovementCategory.COST_REDUCTION:
            cost_savings = improvement_target * 12
            impacts.append(ImpactAssessment(
                impact_type=ImpactType.COST,
                annual_reduction=cost_savings,
                unit="USD",
            ))

        # Estimate implementation cost
        implementation_cost = cost_savings * 2  # 2-year payback estimate

        opportunity = ImprovementOpportunity(
            facility_id=metric.facility_id,
            category=category,
            priority=priority,
            title=f"Improve {metric.name}",
            description=f"Reduce {metric.name} from {metric.current_value:.2f} to {metric.target_value:.2f} {metric.unit}. "
                       f"Current gap: {metric.gap_percent:.1f}%",
            impacts=impacts,
            total_emissions_reduction_kg=round(emissions_reduction, 2),
            total_cost_savings=round(cost_savings, 2),
            total_energy_savings_kwh=round(energy_savings, 2),
            implementation_cost=round(implementation_cost, 2),
            source_metrics=[metric.metric_id],
            confidence_score=0.75,
        )

        return opportunity

    # =========================================================================
    # Impact Assessment
    # =========================================================================

    def _handle_assess_impact(self, opportunity_id: Optional[str]) -> Dict[str, Any]:
        """Assess detailed impact of an opportunity."""
        if not opportunity_id:
            return {"error": "opportunity_id is required"}

        if opportunity_id not in self._opportunities:
            return {"error": f"Opportunity not found: {opportunity_id}"}

        opp = self._opportunities[opportunity_id]

        # Calculate additional metrics
        total_annual_savings = opp.total_cost_savings

        if opp.implementation_cost > 0 and total_annual_savings > 0:
            payback_years = opp.implementation_cost / total_annual_savings

            # Simple NPV calculation (5-year, 8% discount rate)
            discount_rate = 0.08
            npv = -opp.implementation_cost
            for year in range(1, 6):
                npv += total_annual_savings / ((1 + discount_rate) ** year)

            # Update impacts with financial metrics
            for impact in opp.impacts:
                if impact.impact_type == ImpactType.COST:
                    impact.payback_years = round(payback_years, 2)
                    impact.npv = round(npv, 2)
        else:
            payback_years = None
            npv = None

        opp.status = ImplementationStatus.ASSESSED

        return {
            "opportunity_id": opportunity_id,
            "category": opp.category.value,
            "priority": opp.priority.value,
            "impacts": [i.model_dump() for i in opp.impacts],
            "total_emissions_reduction_kg": opp.total_emissions_reduction_kg,
            "total_cost_savings": opp.total_cost_savings,
            "total_energy_savings_kwh": opp.total_energy_savings_kwh,
            "implementation_cost": opp.implementation_cost,
            "payback_years": payback_years,
            "npv": npv,
        }

    # =========================================================================
    # Prioritization
    # =========================================================================

    def _handle_prioritize(self, facility_id: Optional[str]) -> Dict[str, Any]:
        """Prioritize opportunities for a facility."""
        opportunities = list(self._opportunities.values())

        if facility_id:
            opportunities = [o for o in opportunities if o.facility_id == facility_id]

        # Score each opportunity
        scored = []
        for opp in opportunities:
            score = self._calculate_priority_score(opp)
            scored.append((opp, score))

        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update priorities based on ranking
        for i, (opp, score) in enumerate(scored):
            if i < len(scored) * 0.1:  # Top 10%
                opp.priority = ImprovementPriority.CRITICAL
            elif i < len(scored) * 0.3:  # Top 30%
                opp.priority = ImprovementPriority.HIGH
            elif i < len(scored) * 0.6:  # Top 60%
                opp.priority = ImprovementPriority.MEDIUM
            else:
                opp.priority = ImprovementPriority.LOW

        return {
            "prioritized_opportunities": [
                {
                    "opportunity_id": opp.opportunity_id,
                    "title": opp.title,
                    "priority": opp.priority.value,
                    "score": round(score, 2),
                    "emissions_reduction_kg": opp.total_emissions_reduction_kg,
                    "cost_savings": opp.total_cost_savings,
                }
                for opp, score in scored
            ],
            "total_opportunities": len(scored),
        }

    def _calculate_priority_score(self, opp: ImprovementOpportunity) -> float:
        """Calculate priority score for an opportunity."""
        score = 0.0

        # Emissions impact (40% weight)
        if opp.total_emissions_reduction_kg > 10000:
            score += 40
        elif opp.total_emissions_reduction_kg > 1000:
            score += 30
        elif opp.total_emissions_reduction_kg > 100:
            score += 20
        else:
            score += 10

        # Financial impact (30% weight)
        if opp.total_cost_savings > 0 and opp.implementation_cost > 0:
            payback = opp.implementation_cost / opp.total_cost_savings
            if payback < 1:
                score += 30
            elif payback < 2:
                score += 25
            elif payback < 3:
                score += 20
            elif payback < 5:
                score += 10
            else:
                score += 5
        elif opp.total_cost_savings > 0:
            score += 25

        # Implementation effort (20% weight)
        effort_scores = {"low": 20, "medium": 15, "high": 10}
        score += effort_scores.get(opp.implementation_effort, 10)

        # Confidence (10% weight)
        score += opp.confidence_score * 10

        return score

    # =========================================================================
    # Opportunity Management
    # =========================================================================

    def _handle_add_opportunity(
        self, opportunity: Optional[ImprovementOpportunity]
    ) -> Dict[str, Any]:
        """Add a new improvement opportunity."""
        if not opportunity:
            return {"error": "opportunity is required"}

        self._opportunities[opportunity.opportunity_id] = opportunity
        self._total_opportunities_identified += 1
        self._total_savings_identified += opportunity.total_cost_savings
        self._total_emissions_reduction_identified += opportunity.total_emissions_reduction_kg

        return {
            "opportunity_id": opportunity.opportunity_id,
            "added": True,
            "total_opportunities": len(self._opportunities),
        }

    def _handle_update_status(
        self,
        opportunity_id: Optional[str],
        new_status: Optional[ImplementationStatus],
        progress_percent: Optional[float],
    ) -> Dict[str, Any]:
        """Update opportunity status and progress."""
        if not opportunity_id:
            return {"error": "opportunity_id is required"}

        if opportunity_id not in self._opportunities:
            return {"error": f"Opportunity not found: {opportunity_id}"}

        opp = self._opportunities[opportunity_id]

        if new_status:
            opp.status = new_status

        if progress_percent is not None:
            opp.progress_percent = progress_percent

        if new_status == ImplementationStatus.COMPLETED:
            opp.actual_completion_date = DeterministicClock.now()
            opp.progress_percent = 100.0

        return {
            "opportunity_id": opportunity_id,
            "status": opp.status.value,
            "progress_percent": opp.progress_percent,
            "updated": True,
        }

    def _handle_get_opportunities(
        self,
        facility_id: Optional[str],
        category_filter: Optional[ImprovementCategory],
        priority_filter: Optional[ImprovementPriority],
    ) -> Dict[str, Any]:
        """Get improvement opportunities with optional filters."""
        opportunities = list(self._opportunities.values())

        if facility_id:
            opportunities = [o for o in opportunities if o.facility_id == facility_id]

        if category_filter:
            opportunities = [o for o in opportunities if o.category == category_filter]

        if priority_filter:
            opportunities = [o for o in opportunities if o.priority == priority_filter]

        # Sort by priority and status
        priority_order = {
            ImprovementPriority.CRITICAL: 0,
            ImprovementPriority.HIGH: 1,
            ImprovementPriority.MEDIUM: 2,
            ImprovementPriority.LOW: 3,
            ImprovementPriority.OPTIONAL: 4,
        }
        opportunities.sort(key=lambda o: priority_order.get(o.priority, 5))

        return {
            "opportunities": [o.model_dump() for o in opportunities],
            "count": len(opportunities),
        }

    # =========================================================================
    # Progress Tracking
    # =========================================================================

    def _handle_get_progress(self, facility_id: Optional[str]) -> Dict[str, Any]:
        """Get implementation progress summary."""
        opportunities = list(self._opportunities.values())

        if facility_id:
            opportunities = [o for o in opportunities if o.facility_id == facility_id]

        # Count by status
        by_status = defaultdict(int)
        for opp in opportunities:
            by_status[opp.status.value] += 1

        # Calculate progress metrics
        completed = [o for o in opportunities if o.status == ImplementationStatus.COMPLETED]
        in_progress = [o for o in opportunities if o.status == ImplementationStatus.IN_PROGRESS]

        realized_emissions_reduction = sum(o.total_emissions_reduction_kg for o in completed)
        realized_cost_savings = sum(o.total_cost_savings for o in completed)

        potential_emissions_reduction = sum(o.total_emissions_reduction_kg for o in in_progress)
        potential_cost_savings = sum(o.total_cost_savings for o in in_progress)

        # Average progress of in-progress items
        avg_progress = (
            sum(o.progress_percent for o in in_progress) / len(in_progress)
            if in_progress else 0.0
        )

        return {
            "total_opportunities": len(opportunities),
            "by_status": dict(by_status),
            "completion_rate": round(len(completed) / len(opportunities) * 100, 2) if opportunities else 0,
            "average_progress_percent": round(avg_progress, 2),
            "realized_emissions_reduction_kg": round(realized_emissions_reduction, 2),
            "realized_cost_savings": round(realized_cost_savings, 2),
            "potential_emissions_reduction_kg": round(potential_emissions_reduction, 2),
            "potential_cost_savings": round(potential_cost_savings, 2),
        }

    def _handle_get_summary(self, facility_id: Optional[str]) -> Dict[str, Any]:
        """Get comprehensive improvement summary."""
        opportunities = list(self._opportunities.values())

        if facility_id:
            opportunities = [o for o in opportunities if o.facility_id == facility_id]

        # By category
        by_category = defaultdict(lambda: {"count": 0, "emissions": 0.0, "savings": 0.0})
        for opp in opportunities:
            cat = opp.category.value
            by_category[cat]["count"] += 1
            by_category[cat]["emissions"] += opp.total_emissions_reduction_kg
            by_category[cat]["savings"] += opp.total_cost_savings

        # By priority
        by_priority = defaultdict(int)
        for opp in opportunities:
            by_priority[opp.priority.value] += 1

        # Top opportunities by emissions reduction
        top_by_emissions = sorted(
            opportunities,
            key=lambda o: o.total_emissions_reduction_kg,
            reverse=True
        )[:5]

        # Top opportunities by cost savings
        top_by_savings = sorted(
            opportunities,
            key=lambda o: o.total_cost_savings,
            reverse=True
        )[:5]

        return {
            "total_opportunities": len(opportunities),
            "total_emissions_reduction_kg": sum(o.total_emissions_reduction_kg for o in opportunities),
            "total_cost_savings": sum(o.total_cost_savings for o in opportunities),
            "total_energy_savings_kwh": sum(o.total_energy_savings_kwh for o in opportunities),
            "by_category": dict(by_category),
            "by_priority": dict(by_priority),
            "top_by_emissions": [
                {"id": o.opportunity_id, "title": o.title, "reduction": o.total_emissions_reduction_kg}
                for o in top_by_emissions
            ],
            "top_by_savings": [
                {"id": o.opportunity_id, "title": o.title, "savings": o.total_cost_savings}
                for o in top_by_savings
            ],
        }

    # =========================================================================
    # Statistics
    # =========================================================================

    def _handle_get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_opportunities_identified": self._total_opportunities_identified,
            "total_savings_identified": round(self._total_savings_identified, 2),
            "total_emissions_reduction_identified": round(self._total_emissions_reduction_identified, 2),
            "current_opportunities": len(self._opportunities),
            "gap_analyses_performed": len(self._gap_analyses),
            "facilities_analyzed": len(self._metrics_history),
        }

    # =========================================================================
    # Provenance
    # =========================================================================

    def _compute_provenance_hash(
        self, input_data: Dict[str, Any], output_data: Dict[str, Any]
    ) -> str:
        """Compute SHA-256 hash for audit trail."""
        provenance_str = json.dumps(
            {"input": input_data, "output": output_data},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()[:16]
