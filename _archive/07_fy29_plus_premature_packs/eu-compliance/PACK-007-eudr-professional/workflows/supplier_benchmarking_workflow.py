# -*- coding: utf-8 -*-
"""
Supplier Benchmarking Workflow
================================

Four-phase quarterly supplier benchmarking workflow for performance comparison
and engagement planning.

This workflow enables:
- Comparative supplier performance analysis
- Peer group benchmarking (industry, geography, commodity)
- Automated scorecard generation
- Data-driven supplier engagement strategies

Phases:
    1. Data Aggregation - Collect supplier KPIs across portfolio
    2. Peer Group Analysis - Compare suppliers to industry benchmarks
    3. Scorecard Generation - Create visual performance scorecards
    4. Engagement Planning - Generate targeted improvement plans

Regulatory Context:
    EUDR Article 8 requires operators to assess suppliers as part of due diligence.
    Benchmarking enables systematic supplier performance monitoring and continuous
    improvement programs that demonstrate "adequate and proportionate" risk management.

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import random
import statistics
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class Phase(str, Enum):
    """Workflow phases."""
    DATA_AGGREGATION = "data_aggregation"
    PEER_GROUP_ANALYSIS = "peer_group_analysis"
    SCORECARD_GENERATION = "scorecard_generation"
    ENGAGEMENT_PLANNING = "engagement_planning"


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PerformanceTier(str, Enum):
    """Supplier performance tiers."""
    LEADING = "leading"
    STRONG = "strong"
    AVERAGE = "average"
    NEEDS_IMPROVEMENT = "needs_improvement"
    AT_RISK = "at_risk"


# =============================================================================
# DATA MODELS
# =============================================================================


class SupplierBenchmarkingConfig(BaseModel):
    """Configuration for supplier benchmarking workflow."""
    benchmarking_period_months: int = Field(default=3, ge=1, description="Benchmarking period")
    peer_group_min_size: int = Field(default=10, ge=3, description="Minimum peer group size")
    include_external_benchmarks: bool = Field(default=True, description="Include industry benchmarks")
    performance_metrics: List[str] = Field(
        default_factory=lambda: [
            "certification_rate",
            "audit_score",
            "data_quality",
            "response_time",
            "deforestation_alerts",
        ],
        description="Metrics to benchmark",
    )
    operator_id: Optional[str] = Field(None, description="Operator context")


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase: Phase = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Execution duration")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


class WorkflowContext(BaseModel):
    """Shared context passed between workflow phases."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique execution ID")
    config: SupplierBenchmarkingConfig = Field(default_factory=SupplierBenchmarkingConfig)
    phase_results: List[PhaseResult] = Field(default_factory=list, description="Completed phase results")
    state: Dict[str, Any] = Field(default_factory=dict, description="Shared state data")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Workflow start time")

    class Config:
        arbitrary_types_allowed = True


class WorkflowResult(BaseModel):
    """Complete result from the supplier benchmarking workflow."""
    workflow_name: str = Field(default="supplier_benchmarking", description="Workflow identifier")
    phases: List[PhaseResult] = Field(default_factory=list, description="All phase results")
    overall_status: PhaseStatus = Field(..., description="Overall workflow status")
    total_duration_seconds: float = Field(default=0.0, ge=0.0, description="Total execution time")
    provenance_hash: str = Field(default="", description="Workflow-level provenance hash")
    execution_id: str = Field(..., description="Execution identifier")
    suppliers_benchmarked: int = Field(default=0, ge=0, description="Suppliers analyzed")
    peer_groups: int = Field(default=0, ge=0, description="Peer groups created")
    scorecards_generated: int = Field(default=0, ge=0, description="Scorecards created")
    engagement_plans: List[Dict[str, Any]] = Field(default_factory=list, description="Engagement actions")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


# =============================================================================
# SUPPLIER BENCHMARKING WORKFLOW
# =============================================================================


class SupplierBenchmarkingWorkflow:
    """
    Four-phase supplier benchmarking workflow.

    Provides quarterly supplier performance analysis with:
    - Multi-dimensional KPI aggregation
    - Peer group segmentation (industry, geography, commodity)
    - Visual scorecard generation
    - Automated engagement planning

    Example:
        >>> config = SupplierBenchmarkingConfig(
        ...     benchmarking_period_months=3,
        ...     peer_group_min_size=10,
        ... )
        >>> workflow = SupplierBenchmarkingWorkflow(config)
        >>> result = await workflow.run(WorkflowContext(config=config))
        >>> assert result.overall_status == PhaseStatus.COMPLETED
    """

    def __init__(self, config: Optional[SupplierBenchmarkingConfig] = None) -> None:
        """Initialize the supplier benchmarking workflow."""
        self.config = config or SupplierBenchmarkingConfig()
        self.logger = logging.getLogger(f"{__name__}.SupplierBenchmarkingWorkflow")

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """
        Execute the full 4-phase supplier benchmarking workflow.

        Args:
            context: Workflow context with configuration and initial state.

        Returns:
            WorkflowResult with benchmarks, scorecards, and engagement plans.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting supplier benchmarking workflow execution_id=%s period=%d months",
            context.execution_id,
            self.config.benchmarking_period_months,
        )

        context.config = self.config

        phase_handlers = [
            (Phase.DATA_AGGREGATION, self._phase_1_data_aggregation),
            (Phase.PEER_GROUP_ANALYSIS, self._phase_2_peer_group_analysis),
            (Phase.SCORECARD_GENERATION, self._phase_3_scorecard_generation),
            (Phase.ENGAGEMENT_PLANNING, self._phase_4_engagement_planning),
        ]

        overall_status = PhaseStatus.COMPLETED

        for phase, handler in phase_handlers:
            phase_start = datetime.utcnow()
            self.logger.info("Starting phase: %s", phase.value)

            try:
                phase_result = await handler(context)
                phase_result.duration_seconds = (datetime.utcnow() - phase_start).total_seconds()
                phase_result.timestamp = datetime.utcnow()
            except Exception as exc:
                self.logger.error("Phase '%s' failed: %s", phase.value, exc, exc_info=True)
                phase_result = PhaseResult(
                    phase=phase,
                    status=PhaseStatus.FAILED,
                    data={"error": str(exc)},
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    provenance_hash=self._hash({"error": str(exc)}),
                    timestamp=datetime.utcnow(),
                )

            context.phase_results.append(phase_result)

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                self.logger.error("Phase '%s' failed; halting workflow.", phase.value)
                break

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        # Extract final outputs
        suppliers = context.state.get("suppliers", [])
        peer_groups = context.state.get("peer_groups", [])
        scorecards = context.state.get("scorecards", [])
        engagement_plans = context.state.get("engagement_plans", [])

        provenance = self._hash({
            "execution_id": context.execution_id,
            "phases": [p.provenance_hash for p in context.phase_results],
            "period_months": self.config.benchmarking_period_months,
        })

        self.logger.info(
            "Supplier benchmarking workflow finished execution_id=%s status=%s "
            "suppliers=%d scorecards=%d",
            context.execution_id,
            overall_status.value,
            len(suppliers),
            len(scorecards),
        )

        return WorkflowResult(
            phases=context.phase_results,
            overall_status=overall_status,
            total_duration_seconds=total_duration,
            provenance_hash=provenance,
            execution_id=context.execution_id,
            suppliers_benchmarked=len(suppliers),
            peer_groups=len(peer_groups),
            scorecards_generated=len(scorecards),
            engagement_plans=engagement_plans,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Data Aggregation
    # -------------------------------------------------------------------------

    async def _phase_1_data_aggregation(self, context: WorkflowContext) -> PhaseResult:
        """
        Collect supplier KPIs across portfolio.

        Aggregates:
        - Certification status and validity
        - Audit scores (last 12 months)
        - Data quality metrics (completeness, timeliness)
        - Response time to information requests
        - Deforestation/compliance alert counts
        - DDS submission history
        """
        phase = Phase.DATA_AGGREGATION
        period_months = self.config.benchmarking_period_months

        self.logger.info("Aggregating supplier data (period=%d months)", period_months)

        await asyncio.sleep(0.05)

        # Simulate supplier data collection
        supplier_count = random.randint(20, 150)
        suppliers = []

        for i in range(supplier_count):
            supplier = {
                "supplier_id": f"SUP-{uuid.uuid4().hex[:8]}",
                "supplier_name": f"Supplier {i+1}",
                "country": random.choice(["BR", "ID", "CO", "PE", "MY", "TH", "VN", "GH"]),
                "commodity": random.choice(["cocoa", "coffee", "oil_palm", "soya", "cattle"]),
                "metrics": {
                    "certification_rate": random.uniform(0.0, 1.0),
                    "audit_score": random.uniform(40, 100),
                    "data_quality": random.uniform(0.5, 1.0),
                    "response_time_days": random.uniform(1, 30),
                    "deforestation_alerts": random.randint(0, 10),
                },
                "dds_count": random.randint(0, 20),
                "last_audit_date": (datetime.utcnow() - timedelta(days=random.randint(0, 365))).isoformat(),
            }
            suppliers.append(supplier)

        context.state["suppliers"] = suppliers
        context.state["period_start"] = (datetime.utcnow() - timedelta(days=period_months * 30)).isoformat()
        context.state["period_end"] = datetime.utcnow().isoformat()

        provenance = self._hash({
            "phase": phase.value,
            "supplier_count": len(suppliers),
            "period_months": period_months,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "suppliers_aggregated": len(suppliers),
                "period_months": period_months,
                "metrics_collected": len(self.config.performance_metrics),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Peer Group Analysis
    # -------------------------------------------------------------------------

    async def _phase_2_peer_group_analysis(self, context: WorkflowContext) -> PhaseResult:
        """
        Compare suppliers to industry benchmarks.

        Peer grouping dimensions:
        - Commodity type
        - Country/region
        - Company size (volume/revenue)
        - Certification level

        For each peer group, calculate:
        - Median, mean, P25, P75 for each metric
        - Identify top/bottom performers
        """
        phase = Phase.PEER_GROUP_ANALYSIS
        suppliers = context.state.get("suppliers", [])

        self.logger.info("Analyzing peer groups from %d suppliers", len(suppliers))

        # Group suppliers by commodity
        peer_groups_by_commodity: Dict[str, List[Dict[str, Any]]] = {}
        for supplier in suppliers:
            commodity = supplier.get("commodity", "unknown")
            if commodity not in peer_groups_by_commodity:
                peer_groups_by_commodity[commodity] = []
            peer_groups_by_commodity[commodity].append(supplier)

        # Calculate benchmarks for each peer group
        peer_groups = []
        for commodity, group_suppliers in peer_groups_by_commodity.items():
            if len(group_suppliers) < self.config.peer_group_min_size:
                continue

            benchmarks = self._calculate_benchmarks(group_suppliers, self.config.performance_metrics)

            peer_group = {
                "peer_group_id": f"PG-{commodity.upper()}-{uuid.uuid4().hex[:6]}",
                "commodity": commodity,
                "supplier_count": len(group_suppliers),
                "benchmarks": benchmarks,
            }
            peer_groups.append(peer_group)

        context.state["peer_groups"] = peer_groups

        provenance = self._hash({
            "phase": phase.value,
            "peer_group_count": len(peer_groups),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "peer_groups_created": len(peer_groups),
                "commodities": list(peer_groups_by_commodity.keys()),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Scorecard Generation
    # -------------------------------------------------------------------------

    async def _phase_3_scorecard_generation(self, context: WorkflowContext) -> PhaseResult:
        """
        Create visual performance scorecards.

        Scorecard elements:
        - Overall performance tier (leading, strong, average, needs improvement, at risk)
        - Metric-by-metric comparison to peer group median
        - Trend analysis (improving, stable, declining)
        - Certification status
        - Compliance alert summary
        """
        phase = Phase.SCORECARD_GENERATION
        suppliers = context.state.get("suppliers", [])
        peer_groups = context.state.get("peer_groups", [])

        self.logger.info("Generating scorecards for %d suppliers", len(suppliers))

        # Build commodity -> peer group lookup
        commodity_benchmarks = {
            pg["commodity"]: pg["benchmarks"]
            for pg in peer_groups
        }

        scorecards = []
        for supplier in suppliers:
            commodity = supplier.get("commodity", "unknown")
            benchmarks = commodity_benchmarks.get(commodity, {})

            # Calculate performance tier
            tier = self._calculate_performance_tier(supplier, benchmarks)

            # Calculate metric comparisons
            metric_comparisons = self._compare_to_benchmarks(supplier, benchmarks)

            scorecard = {
                "scorecard_id": f"SC-{uuid.uuid4().hex[:8]}",
                "supplier_id": supplier["supplier_id"],
                "supplier_name": supplier["supplier_name"],
                "performance_tier": tier.value,
                "overall_score": self._calculate_overall_score(metric_comparisons),
                "metric_comparisons": metric_comparisons,
                "generated_at": datetime.utcnow().isoformat(),
            }
            scorecards.append(scorecard)

        context.state["scorecards"] = scorecards

        # Count by tier
        tier_distribution = {}
        for sc in scorecards:
            tier = sc["performance_tier"]
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

        provenance = self._hash({
            "phase": phase.value,
            "scorecard_count": len(scorecards),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "scorecards_generated": len(scorecards),
                "tier_distribution": tier_distribution,
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Engagement Planning
    # -------------------------------------------------------------------------

    async def _phase_4_engagement_planning(self, context: WorkflowContext) -> PhaseResult:
        """
        Generate targeted improvement plans.

        Engagement strategies:
        - Leading/Strong: Recognition program, best practice sharing
        - Average: Capability building, peer learning
        - Needs Improvement: Corrective action plan, increased audits
        - At Risk: Performance improvement plan, supplier replacement evaluation
        """
        phase = Phase.ENGAGEMENT_PLANNING
        scorecards = context.state.get("scorecards", [])

        self.logger.info("Planning engagement for %d suppliers", len(scorecards))

        engagement_plans = []

        for scorecard in scorecards:
            tier = scorecard["performance_tier"]
            supplier_id = scorecard["supplier_id"]
            supplier_name = scorecard["supplier_name"]

            actions = self._generate_engagement_actions(tier, scorecard)

            plan = {
                "plan_id": f"EP-{uuid.uuid4().hex[:8]}",
                "supplier_id": supplier_id,
                "supplier_name": supplier_name,
                "performance_tier": tier,
                "actions": actions,
                "priority": self._determine_engagement_priority(tier),
                "timeline_months": self._determine_timeline(tier),
                "created_at": datetime.utcnow().isoformat(),
            }
            engagement_plans.append(plan)

        context.state["engagement_plans"] = engagement_plans

        # Count by priority
        priority_distribution = {}
        for plan in engagement_plans:
            priority = plan["priority"]
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1

        provenance = self._hash({
            "phase": phase.value,
            "plan_count": len(engagement_plans),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "engagement_plans_created": len(engagement_plans),
                "priority_distribution": priority_distribution,
            },
            provenance_hash=provenance,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _calculate_benchmarks(
        self, suppliers: List[Dict[str, Any]], metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate benchmark statistics for peer group."""
        benchmarks = {}

        for metric in metrics:
            values = [s["metrics"].get(metric, 0) for s in suppliers]
            if not values:
                continue

            benchmarks[metric] = {
                "mean": round(statistics.mean(values), 2),
                "median": round(statistics.median(values), 2),
                "p25": round(statistics.quantiles(values, n=4)[0], 2) if len(values) > 1 else values[0],
                "p75": round(statistics.quantiles(values, n=4)[2], 2) if len(values) > 1 else values[0],
                "min": round(min(values), 2),
                "max": round(max(values), 2),
            }

        return benchmarks

    def _calculate_performance_tier(
        self, supplier: Dict[str, Any], benchmarks: Dict[str, Any]
    ) -> PerformanceTier:
        """Calculate supplier performance tier."""
        if not benchmarks:
            return PerformanceTier.AVERAGE

        # Compare metrics to peer benchmarks
        scores = []
        for metric, benchmark in benchmarks.items():
            supplier_value = supplier["metrics"].get(metric, 0)
            median = benchmark.get("median", 0)

            # Invert score for metrics where lower is better
            if metric in ("response_time_days", "deforestation_alerts"):
                if median > 0:
                    score = (median / max(supplier_value, 0.1)) * 100
                else:
                    score = 50
            else:
                if median > 0:
                    score = (supplier_value / median) * 100
                else:
                    score = 50

            scores.append(min(200, max(0, score)))

        avg_score = sum(scores) / len(scores) if scores else 50

        if avg_score >= 120:
            return PerformanceTier.LEADING
        elif avg_score >= 100:
            return PerformanceTier.STRONG
        elif avg_score >= 70:
            return PerformanceTier.AVERAGE
        elif avg_score >= 50:
            return PerformanceTier.NEEDS_IMPROVEMENT
        return PerformanceTier.AT_RISK

    def _compare_to_benchmarks(
        self, supplier: Dict[str, Any], benchmarks: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compare supplier metrics to benchmarks."""
        comparisons = []

        for metric, benchmark in benchmarks.items():
            supplier_value = supplier["metrics"].get(metric, 0)
            median = benchmark.get("median", 0)

            comparison = {
                "metric": metric,
                "supplier_value": round(supplier_value, 2),
                "peer_median": median,
                "peer_p25": benchmark.get("p25", 0),
                "peer_p75": benchmark.get("p75", 0),
                "percentile_rank": self._calculate_percentile_rank(
                    supplier_value, benchmark
                ),
            }
            comparisons.append(comparison)

        return comparisons

    def _calculate_percentile_rank(
        self, value: float, benchmark: Dict[str, float]
    ) -> int:
        """Calculate approximate percentile rank."""
        p25 = benchmark.get("p25", 0)
        median = benchmark.get("median", 0)
        p75 = benchmark.get("p75", 0)

        if value <= p25:
            return 25
        elif value <= median:
            return 50
        elif value <= p75:
            return 75
        return 90

    def _calculate_overall_score(self, comparisons: List[Dict[str, Any]]) -> float:
        """Calculate overall performance score."""
        if not comparisons:
            return 50.0

        percentiles = [c.get("percentile_rank", 50) for c in comparisons]
        return round(sum(percentiles) / len(percentiles), 1)

    def _generate_engagement_actions(
        self, tier: str, scorecard: Dict[str, Any]
    ) -> List[str]:
        """Generate engagement actions based on performance tier."""
        actions = []

        if tier == PerformanceTier.LEADING.value:
            actions.append("Invite to supplier excellence program")
            actions.append("Request best practice case study")
            actions.append("Offer preferred supplier status")
        elif tier == PerformanceTier.STRONG.value:
            actions.append("Maintain current engagement level")
            actions.append("Share peer benchmark report")
            actions.append("Offer certification support if needed")
        elif tier == PerformanceTier.AVERAGE.value:
            actions.append("Provide peer benchmark report with improvement targets")
            actions.append("Offer capability building workshop")
            actions.append("Increase audit frequency to semi-annual")
        elif tier == PerformanceTier.NEEDS_IMPROVEMENT.value:
            actions.append("Develop 6-month corrective action plan")
            actions.append("Require monthly progress reporting")
            actions.append("Schedule on-site improvement assessment")
            actions.append("Provide technical assistance for certification")
        else:  # AT_RISK
            actions.append("Initiate performance improvement plan (PIP)")
            actions.append("Conduct immediate on-site audit")
            actions.append("Evaluate supplier replacement options")
            actions.append("Reduce order volume pending improvement")

        return actions

    def _determine_engagement_priority(self, tier: str) -> str:
        """Determine engagement priority."""
        priority_map = {
            PerformanceTier.LEADING.value: "low",
            PerformanceTier.STRONG.value: "low",
            PerformanceTier.AVERAGE.value: "medium",
            PerformanceTier.NEEDS_IMPROVEMENT.value: "high",
            PerformanceTier.AT_RISK.value: "critical",
        }
        return priority_map.get(tier, "medium")

    def _determine_timeline(self, tier: str) -> int:
        """Determine engagement timeline in months."""
        timeline_map = {
            PerformanceTier.LEADING.value: 12,
            PerformanceTier.STRONG.value: 12,
            PerformanceTier.AVERAGE.value: 6,
            PerformanceTier.NEEDS_IMPROVEMENT.value: 6,
            PerformanceTier.AT_RISK.value: 3,
        }
        return timeline_map.get(tier, 6)

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
