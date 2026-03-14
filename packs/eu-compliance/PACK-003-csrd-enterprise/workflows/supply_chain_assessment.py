# -*- coding: utf-8 -*-
"""
Supply Chain Assessment Workflow
===================================

5-phase supply chain ESG assessment workflow for CSRD Enterprise Pack.
Builds multi-tier supplier graphs, dispatches ESG questionnaires, scores
suppliers across E/S/G dimensions, assigns risk tiers, and generates
corrective action plans with Scope 3 upstream emission estimation.

Phases:
    1. Supplier Mapping: Build multi-tier (1-4) supply chain graph
    2. Questionnaire Dispatch: Automated ESG questionnaire with follow-up reminders
    3. Response Processing: Parse, validate, and score questionnaire responses
    4. Risk Assessment: Score suppliers E/S/G (0-100), assign risk tiers
    5. Improvement Planning: Corrective action plans for high-risk suppliers

Author: GreenLang Team
Version: 3.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class SupplierTier(str, Enum):
    """Supply chain tier classification."""

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"


class RiskTier(str, Enum):
    """Supplier risk tier classification."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class QuestionnaireStatus(str, Enum):
    """Questionnaire dispatch status."""

    DRAFT = "draft"
    SENT = "sent"
    REMINDER_SENT = "reminder_sent"
    RECEIVED = "received"
    EXPIRED = "expired"
    DECLINED = "declined"


class ActionPriority(str, Enum):
    """Corrective action priority."""

    IMMEDIATE = "immediate"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration in seconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class SupplierProfile(BaseModel):
    """Profile of a supplier in the supply chain graph."""

    supplier_id: str = Field(default_factory=lambda: f"sup-{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="Supplier name")
    tier: SupplierTier = Field(default=SupplierTier.TIER_1, description="Supply chain tier")
    country: str = Field(default="", description="ISO 3166-1 alpha-2 country code")
    industry: str = Field(default="", description="Industry sector")
    spend_eur: float = Field(default=0.0, ge=0.0, description="Annual spend in EUR")
    parent_supplier_id: Optional[str] = Field(
        None, description="Parent supplier ID (for tier > 1)"
    )
    products_services: List[str] = Field(
        default_factory=list, description="Products/services supplied"
    )
    contact_email: str = Field(default="", description="Supplier contact email")
    certifications: List[str] = Field(
        default_factory=list, description="ESG certifications held"
    )


class ESGScore(BaseModel):
    """ESG scores for a supplier."""

    environmental: float = Field(default=0.0, ge=0.0, le=100.0, description="Environmental score")
    social: float = Field(default=0.0, ge=0.0, le=100.0, description="Social score")
    governance: float = Field(default=0.0, ge=0.0, le=100.0, description="Governance score")
    overall: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall ESG score")
    risk_tier: RiskTier = Field(default=RiskTier.MEDIUM, description="Risk tier assignment")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Score confidence level"
    )


class CorrectiveAction(BaseModel):
    """Corrective action plan for a supplier."""

    action_id: str = Field(default_factory=lambda: f"ca-{uuid.uuid4().hex[:8]}")
    supplier_id: str = Field(..., description="Target supplier")
    supplier_name: str = Field(default="", description="Supplier name")
    priority: ActionPriority = Field(..., description="Action priority")
    dimension: str = Field(
        ..., description="ESG dimension (environmental, social, governance)"
    )
    description: str = Field(..., description="Action description")
    target_improvement: float = Field(
        default=0.0, ge=0.0, description="Target score improvement points"
    )
    deadline_days: int = Field(default=90, ge=1, description="Deadline in days")
    estimated_scope3_impact_tco2e: float = Field(
        default=0.0, description="Estimated Scope 3 reduction in tCO2e"
    )
    monitoring_frequency: str = Field(
        default="quarterly", description="Monitoring frequency"
    )


class SupplyChainAssessmentConfig(BaseModel):
    """Configuration for supply chain assessment."""

    entity_id: str = Field(..., description="Entity to assess")
    tenant_id: str = Field(default="", description="Tenant isolation ID")
    reporting_year: int = Field(default=2025, ge=2024, le=2050)
    max_tier_depth: int = Field(default=4, ge=1, le=6, description="Max tier depth to map")
    questionnaire_deadline_days: int = Field(
        default=30, ge=7, le=90, description="Questionnaire response deadline"
    )
    reminder_intervals_days: List[int] = Field(
        default_factory=lambda: [7, 14, 21],
        description="Days after dispatch to send reminders",
    )
    minimum_response_rate: float = Field(
        default=50.0, ge=0.0, le=100.0, description="Min response rate to proceed (%)"
    )
    risk_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "CRITICAL": 30.0,
            "HIGH": 50.0,
            "MEDIUM": 70.0,
        },
        description="ESG score thresholds for risk tiers",
    )
    scope3_categories: List[int] = Field(
        default_factory=lambda: [1, 2, 4, 9],
        description="Scope 3 categories to estimate (1=PG&S, 2=Capital, 4=Transport, 9=Downstream)",
    )
    include_scope3_estimation: bool = Field(
        default=True, description="Whether to estimate Scope 3 from supplier data"
    )


class SupplyChainAssessmentResult(BaseModel):
    """Complete result from supply chain assessment workflow."""

    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(default="supply_chain_assessment")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    entity_id: str = Field(default="", description="Entity assessed")
    phases: List[PhaseResult] = Field(default_factory=list, description="Per-phase results")
    total_duration_seconds: float = Field(default=0.0)
    suppliers_mapped: int = Field(default=0, description="Total suppliers in graph")
    suppliers_by_tier: Dict[str, int] = Field(default_factory=dict)
    questionnaire_response_rate: float = Field(default=0.0)
    suppliers_scored: int = Field(default=0)
    average_esg_scores: Dict[str, float] = Field(
        default_factory=dict, description="Average E/S/G scores"
    )
    risk_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Suppliers per risk tier"
    )
    critical_suppliers: List[str] = Field(
        default_factory=list, description="Critical-risk supplier IDs"
    )
    corrective_actions: List[CorrectiveAction] = Field(
        default_factory=list, description="Generated corrective actions"
    )
    scope3_upstream_tco2e: float = Field(
        default=0.0, description="Estimated Scope 3 upstream emissions"
    )
    provenance_hash: str = Field(default="", description="SHA-256 of complete output")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class SupplyChainAssessmentWorkflow:
    """
    5-phase supply chain ESG assessment workflow.

    Builds a multi-tier supply chain graph, dispatches automated ESG
    questionnaires with follow-up reminders, scores suppliers across
    Environmental, Social, and Governance dimensions (0-100 each),
    assigns risk tiers, generates corrective action plans for high-risk
    suppliers, and estimates Scope 3 upstream emissions.

    Risk tier assignment:
        CRITICAL: Overall ESG score < 30
        HIGH: Overall ESG score 30-50
        MEDIUM: Overall ESG score 50-70
        LOW: Overall ESG score >= 70

    Attributes:
        workflow_id: Unique execution identifier.
        config: Optional EnterprisePackConfig.
        _suppliers: Mapped supplier profiles.
        _scores: ESG scores per supplier.
        _corrective_actions: Generated corrective actions.

    Example:
        >>> workflow = SupplyChainAssessmentWorkflow()
        >>> config = SupplyChainAssessmentConfig(entity_id="entity-001")
        >>> result = await workflow.execute("entity-001", config)
        >>> assert result.suppliers_mapped > 0
        >>> assert result.scope3_upstream_tco2e >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        Initialize the supply chain assessment workflow.

        Args:
            config: Optional EnterprisePackConfig.
        """
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._suppliers: List[SupplierProfile] = []
        self._scores: Dict[str, ESGScore] = {}
        self._corrective_actions: List[CorrectiveAction] = []
        self._context: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        entity_id: str = "",
        config: Optional[SupplyChainAssessmentConfig] = None,
    ) -> SupplyChainAssessmentResult:
        """
        Execute the 5-phase supply chain assessment workflow.

        Args:
            entity_id: Entity to assess (fallback if config not provided).
            config: Full assessment configuration.

        Returns:
            SupplyChainAssessmentResult with supplier graph, scores,
            risk tiers, corrective actions, and Scope 3 estimates.
        """
        if config is None:
            config = SupplyChainAssessmentConfig(entity_id=entity_id or "default")

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting supply chain assessment workflow %s for entity=%s depth=%d",
            self.workflow_id, config.entity_id, config.max_tier_depth,
        )

        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Supplier Mapping
            p1 = await self._phase_1_supplier_mapping(config)
            phase_results.append(p1)
            if p1.status == PhaseStatus.FAILED:
                overall_status = WorkflowStatus.FAILED
                raise RuntimeError("Supplier mapping failed")

            # Phase 2: Questionnaire Dispatch
            p2 = await self._phase_2_questionnaire_dispatch(config)
            phase_results.append(p2)

            # Phase 3: Response Processing
            p3 = await self._phase_3_response_processing(config)
            phase_results.append(p3)

            # Phase 4: Risk Assessment
            p4 = await self._phase_4_risk_assessment(config)
            phase_results.append(p4)

            # Phase 5: Improvement Planning
            p5 = await self._phase_5_improvement_planning(config)
            phase_results.append(p5)

            overall_status = WorkflowStatus.COMPLETED

        except RuntimeError:
            if overall_status != WorkflowStatus.FAILED:
                overall_status = WorkflowStatus.FAILED
        except Exception as exc:
            self.logger.critical(
                "Supply chain assessment %s failed: %s",
                self.workflow_id, str(exc), exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="workflow_error",
                status=PhaseStatus.FAILED,
                errors=[str(exc)],
                provenance_hash=self._hash_data({"error": str(exc)}),
            ))

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        # Build summary
        suppliers_by_tier: Dict[str, int] = {}
        for s in self._suppliers:
            tier = s.tier.value
            suppliers_by_tier[tier] = suppliers_by_tier.get(tier, 0) + 1

        risk_distribution: Dict[str, int] = {}
        for score in self._scores.values():
            tier = score.risk_tier.value
            risk_distribution[tier] = risk_distribution.get(tier, 0) + 1

        critical_supplier_ids = [
            sid for sid, score in self._scores.items()
            if score.risk_tier == RiskTier.CRITICAL
        ]

        avg_scores = self._compute_average_scores()
        scope3_total = self._context.get("scope3_upstream_tco2e", 0.0)

        provenance = self._hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in phase_results],
        })

        self.logger.info(
            "Supply chain assessment %s finished status=%s suppliers=%d "
            "scored=%d critical=%d scope3=%.1f tCO2e in %.1fs",
            self.workflow_id, overall_status.value,
            len(self._suppliers), len(self._scores),
            len(critical_supplier_ids), scope3_total, total_duration,
        )

        return SupplyChainAssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            entity_id=config.entity_id,
            phases=phase_results,
            total_duration_seconds=total_duration,
            suppliers_mapped=len(self._suppliers),
            suppliers_by_tier=suppliers_by_tier,
            questionnaire_response_rate=self._context.get("response_rate", 0.0),
            suppliers_scored=len(self._scores),
            average_esg_scores=avg_scores,
            risk_distribution=risk_distribution,
            critical_suppliers=critical_supplier_ids,
            corrective_actions=self._corrective_actions,
            scope3_upstream_tco2e=scope3_total,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Supplier Mapping
    # -------------------------------------------------------------------------

    async def _phase_1_supplier_mapping(
        self, config: SupplyChainAssessmentConfig
    ) -> PhaseResult:
        """
        Build multi-tier supply chain graph from procurement data.

        Constructs a graph of suppliers across tiers 1-4 (configurable),
        connecting upstream relationships through parent-child links.

        Agents invoked:
            - greenlang.agents.data.erp_connector_agent (procurement data)
            - greenlang.agents.eudr.chain_of_custody (traceability)
            - greenlang.agents.eudr.network_analyzer (graph construction)

        Steps:
            1. Extract Tier 1 suppliers from procurement/ERP data
            2. Discover upstream tiers through supplier relationships
            3. Enrich supplier profiles with industry, country, certifications
            4. Build directed graph with spend allocation
        """
        phase_name = "supplier_mapping"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Tier 1 from ERP
        tier1_suppliers = await self._extract_tier1_suppliers(config.entity_id)
        self._suppliers.extend(tier1_suppliers)
        outputs["tier_1_count"] = len(tier1_suppliers)

        # Step 2: Discover upstream tiers
        for tier_num in range(2, config.max_tier_depth + 1):
            tier_enum = SupplierTier(f"tier_{tier_num}")
            parent_ids = [
                s.supplier_id for s in self._suppliers
                if s.tier.value == f"tier_{tier_num - 1}"
            ]
            upstream = await self._discover_upstream_suppliers(
                parent_ids, tier_enum
            )
            self._suppliers.extend(upstream)
            outputs[f"tier_{tier_num}_count"] = len(upstream)

            if not upstream:
                self.logger.info("No tier %d suppliers discovered", tier_num)
                break

        # Step 3: Enrich profiles
        enriched = await self._enrich_supplier_profiles(self._suppliers)
        outputs["profiles_enriched"] = enriched.get("count", 0)

        # Step 4: Build graph
        graph = await self._build_supply_chain_graph(self._suppliers)
        outputs["graph_nodes"] = graph.get("nodes", 0)
        outputs["graph_edges"] = graph.get("edges", 0)
        outputs["total_suppliers"] = len(self._suppliers)

        self._context["supplier_graph"] = graph

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name, status=status, duration_seconds=duration,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Questionnaire Dispatch
    # -------------------------------------------------------------------------

    async def _phase_2_questionnaire_dispatch(
        self, config: SupplyChainAssessmentConfig
    ) -> PhaseResult:
        """
        Dispatch ESG questionnaires to suppliers with follow-up reminders.

        Sends standardized ESG questionnaires covering environmental impact,
        social practices, and governance policies. Includes automated
        reminders at configurable intervals.

        Agents invoked:
            - greenlang.agents.data.supplier_questionnaire_processor

        Steps:
            1. Generate questionnaire for each supplier tier
            2. Dispatch to Tier 1 suppliers first
            3. Schedule follow-up reminders
            4. Track dispatch and delivery status
        """
        phase_name = "questionnaire_dispatch"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        dispatch_results: Dict[str, QuestionnaireStatus] = {}
        total_sent = 0

        # Step 1-2: Dispatch per tier
        for tier_value in ["tier_1", "tier_2", "tier_3", "tier_4"]:
            tier_suppliers = [
                s for s in self._suppliers if s.tier.value == tier_value
            ]
            if not tier_suppliers:
                continue

            for supplier in tier_suppliers:
                if not supplier.contact_email:
                    dispatch_results[supplier.supplier_id] = QuestionnaireStatus.EXPIRED
                    warnings.append(
                        f"No contact email for supplier {supplier.supplier_id} ({supplier.name})"
                    )
                    continue

                result = await self._dispatch_questionnaire(
                    supplier, config.questionnaire_deadline_days
                )
                dispatch_results[supplier.supplier_id] = QuestionnaireStatus(
                    result.get("status", "sent")
                )
                total_sent += 1

        outputs["questionnaires_sent"] = total_sent
        outputs["suppliers_without_contact"] = sum(
            1 for s in dispatch_results.values()
            if s == QuestionnaireStatus.EXPIRED
        )

        # Step 3: Schedule reminders
        reminders = await self._schedule_reminders(
            dispatch_results, config.reminder_intervals_days
        )
        outputs["reminders_scheduled"] = reminders.get("scheduled", 0)

        # Step 4: Delivery status
        outputs["dispatch_status"] = {
            s.value: sum(1 for v in dispatch_results.values() if v == s)
            for s in QuestionnaireStatus
            if any(v == s for v in dispatch_results.values())
        }

        self._context["dispatch_results"] = dispatch_results

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name, status=status, duration_seconds=duration,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Response Processing
    # -------------------------------------------------------------------------

    async def _phase_3_response_processing(
        self, config: SupplyChainAssessmentConfig
    ) -> PhaseResult:
        """
        Parse, validate, and score questionnaire responses.

        Processes received questionnaire responses, validates completeness,
        extracts structured ESG data, and computes preliminary scores.

        Agents invoked:
            - greenlang.agents.data.supplier_questionnaire_processor
            - greenlang.agents.data.validation_rule_engine

        Steps:
            1. Collect received responses
            2. Validate response completeness
            3. Extract structured ESG data points
            4. Compute preliminary E/S/G scores per response
            5. Check response rate against minimum threshold
        """
        phase_name = "response_processing"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        dispatch = self._context.get("dispatch_results", {})
        total_dispatched = sum(1 for v in dispatch.values() if v == QuestionnaireStatus.SENT)

        # Step 1: Collect responses
        responses = await self._collect_responses(dispatch)
        received_count = responses.get("received", 0)
        outputs["responses_received"] = received_count

        # Step 2: Validate completeness
        validation_results = await self._validate_responses(responses)
        outputs["valid_responses"] = validation_results.get("valid", 0)
        outputs["incomplete_responses"] = validation_results.get("incomplete", 0)

        # Step 3: Extract ESG data
        esg_data = await self._extract_esg_data(responses)
        outputs["data_points_extracted"] = esg_data.get("data_points", 0)

        # Step 4: Preliminary scores
        preliminary_scores = await self._compute_preliminary_scores(esg_data)
        outputs["suppliers_scored_preliminary"] = len(preliminary_scores)

        self._context["esg_data"] = esg_data
        self._context["preliminary_scores"] = preliminary_scores

        # Step 5: Response rate check
        response_rate = (received_count / max(total_dispatched, 1)) * 100.0
        outputs["response_rate"] = round(response_rate, 1)
        self._context["response_rate"] = round(response_rate, 1)

        if response_rate < config.minimum_response_rate:
            warnings.append(
                f"Response rate {response_rate:.1f}% below minimum "
                f"{config.minimum_response_rate:.1f}%"
            )

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name, status=status, duration_seconds=duration,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Risk Assessment
    # -------------------------------------------------------------------------

    async def _phase_4_risk_assessment(
        self, config: SupplyChainAssessmentConfig
    ) -> PhaseResult:
        """
        Score suppliers on E/S/G dimensions and assign risk tiers.

        Computes comprehensive Environmental, Social, and Governance scores
        (0-100 each) for each supplier using questionnaire data, public
        data, and certification status. Assigns risk tiers based on
        configurable thresholds.

        Agents invoked:
            - greenlang.agents.eudr.supplier_risk_scorer
            - greenlang.agents.eudr.geographic_sourcing_analyzer
            - greenlang.agents.eudr.certification_validator

        Steps:
            1. Score Environmental dimension (emissions, resource use, waste)
            2. Score Social dimension (labor, human rights, community)
            3. Score Governance dimension (ethics, transparency, compliance)
            4. Compute overall weighted ESG score
            5. Assign risk tiers per configurable thresholds
            6. Estimate Scope 3 upstream emissions from supplier data
        """
        phase_name = "risk_assessment"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        preliminary = self._context.get("preliminary_scores", {})
        thresholds = config.risk_thresholds

        # Steps 1-4: Score each supplier
        for supplier in self._suppliers:
            prelim = preliminary.get(supplier.supplier_id, {})
            e_score = await self._score_environmental(supplier, prelim)
            s_score = await self._score_social(supplier, prelim)
            g_score = await self._score_governance(supplier, prelim)

            overall = (e_score * 0.40 + s_score * 0.30 + g_score * 0.30)

            # Step 5: Risk tier assignment
            if overall < thresholds.get("CRITICAL", 30.0):
                risk_tier = RiskTier.CRITICAL
            elif overall < thresholds.get("HIGH", 50.0):
                risk_tier = RiskTier.HIGH
            elif overall < thresholds.get("MEDIUM", 70.0):
                risk_tier = RiskTier.MEDIUM
            else:
                risk_tier = RiskTier.LOW

            confidence = prelim.get("confidence", 0.5) if prelim else 0.3

            self._scores[supplier.supplier_id] = ESGScore(
                environmental=round(e_score, 1),
                social=round(s_score, 1),
                governance=round(g_score, 1),
                overall=round(overall, 1),
                risk_tier=risk_tier,
                confidence=confidence,
            )

        outputs["suppliers_scored"] = len(self._scores)
        outputs["risk_distribution"] = {}
        for score in self._scores.values():
            tier = score.risk_tier.value
            outputs["risk_distribution"][tier] = outputs["risk_distribution"].get(tier, 0) + 1

        outputs["critical_count"] = outputs["risk_distribution"].get("CRITICAL", 0)
        outputs["high_count"] = outputs["risk_distribution"].get("HIGH", 0)

        if outputs["critical_count"] > 0:
            warnings.append(
                f"{outputs['critical_count']} critical-risk suppliers identified"
            )

        # Step 6: Scope 3 estimation
        if config.include_scope3_estimation:
            scope3_estimate = await self._estimate_scope3_upstream(
                self._suppliers, self._scores, config.scope3_categories
            )
            outputs["scope3_upstream_tco2e"] = scope3_estimate.get("total_tco2e", 0.0)
            outputs["scope3_by_category"] = scope3_estimate.get("by_category", {})
            outputs["scope3_by_tier"] = scope3_estimate.get("by_tier", {})
            self._context["scope3_upstream_tco2e"] = scope3_estimate.get("total_tco2e", 0.0)
        else:
            outputs["scope3_upstream_tco2e"] = 0.0

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name, status=status, duration_seconds=duration,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Improvement Planning
    # -------------------------------------------------------------------------

    async def _phase_5_improvement_planning(
        self, config: SupplyChainAssessmentConfig
    ) -> PhaseResult:
        """
        Generate corrective action plans for high-risk suppliers.

        Creates targeted improvement plans for CRITICAL and HIGH risk
        suppliers, with specific actions per ESG dimension, deadlines,
        monitoring frequency, and estimated Scope 3 impact.

        Steps:
            1. Identify suppliers requiring corrective action
            2. Generate dimension-specific actions per supplier
            3. Estimate Scope 3 impact of each action
            4. Set deadlines and monitoring schedules
            5. Generate improvement plan report
        """
        phase_name = "improvement_planning"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Identify suppliers needing action
        actionable_suppliers = [
            (sid, score) for sid, score in self._scores.items()
            if score.risk_tier in (RiskTier.CRITICAL, RiskTier.HIGH)
        ]
        outputs["suppliers_requiring_action"] = len(actionable_suppliers)

        # Steps 2-4: Generate actions
        for supplier_id, score in actionable_suppliers:
            supplier = next(
                (s for s in self._suppliers if s.supplier_id == supplier_id), None
            )
            if supplier is None:
                continue

            # Environmental actions
            if score.environmental < 50:
                actions = await self._generate_environmental_actions(supplier, score)
                self._corrective_actions.extend(actions)

            # Social actions
            if score.social < 50:
                actions = await self._generate_social_actions(supplier, score)
                self._corrective_actions.extend(actions)

            # Governance actions
            if score.governance < 50:
                actions = await self._generate_governance_actions(supplier, score)
                self._corrective_actions.extend(actions)

        outputs["corrective_actions_generated"] = len(self._corrective_actions)
        outputs["actions_by_priority"] = {}
        for action in self._corrective_actions:
            p = action.priority.value
            outputs["actions_by_priority"][p] = outputs["actions_by_priority"].get(p, 0) + 1

        outputs["total_scope3_impact_tco2e"] = sum(
            a.estimated_scope3_impact_tco2e for a in self._corrective_actions
        )

        # Step 5: Improvement plan report
        report = await self._generate_improvement_report(
            self._corrective_actions, self._scores
        )
        outputs["improvement_report_id"] = report.get("report_id", "")

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name, status=status, duration_seconds=duration,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Score Helpers
    # -------------------------------------------------------------------------

    def _compute_average_scores(self) -> Dict[str, float]:
        """Compute average E/S/G scores across all scored suppliers."""
        if not self._scores:
            return {"environmental": 0.0, "social": 0.0, "governance": 0.0, "overall": 0.0}
        n = len(self._scores)
        return {
            "environmental": round(sum(s.environmental for s in self._scores.values()) / n, 1),
            "social": round(sum(s.social for s in self._scores.values()) / n, 1),
            "governance": round(sum(s.governance for s in self._scores.values()) / n, 1),
            "overall": round(sum(s.overall for s in self._scores.values()) / n, 1),
        }

    # -------------------------------------------------------------------------
    # Agent Simulation Stubs
    # -------------------------------------------------------------------------

    async def _extract_tier1_suppliers(
        self, entity_id: str
    ) -> List[SupplierProfile]:
        """Extract Tier 1 suppliers from procurement data."""
        return [
            SupplierProfile(
                name=f"Supplier-T1-{i}",
                tier=SupplierTier.TIER_1,
                country="DE" if i % 2 == 0 else "FR",
                industry="manufacturing",
                spend_eur=100000.0 * (i + 1),
                contact_email=f"contact-t1-{i}@supplier.com",
            )
            for i in range(10)
        ]

    async def _discover_upstream_suppliers(
        self, parent_ids: List[str], tier: SupplierTier
    ) -> List[SupplierProfile]:
        """Discover upstream suppliers at a given tier."""
        return [
            SupplierProfile(
                name=f"Supplier-{tier.value}-{i}",
                tier=tier,
                country="CN" if i % 3 == 0 else "IN",
                industry="raw_materials",
                parent_supplier_id=parent_ids[i % len(parent_ids)] if parent_ids else None,
                contact_email=f"contact-{tier.value}-{i}@supplier.com",
            )
            for i in range(max(len(parent_ids) // 2, 2))
        ]

    async def _enrich_supplier_profiles(
        self, suppliers: List[SupplierProfile]
    ) -> Dict[str, Any]:
        """Enrich supplier profiles with external data."""
        return {"count": len(suppliers)}

    async def _build_supply_chain_graph(
        self, suppliers: List[SupplierProfile]
    ) -> Dict[str, Any]:
        """Build directed supply chain graph."""
        edges = sum(1 for s in suppliers if s.parent_supplier_id)
        return {"nodes": len(suppliers), "edges": edges}

    async def _dispatch_questionnaire(
        self, supplier: SupplierProfile, deadline_days: int
    ) -> Dict[str, Any]:
        """Dispatch questionnaire to a supplier."""
        return {"status": "sent", "deadline": (datetime.utcnow() + timedelta(days=deadline_days)).isoformat()}

    async def _schedule_reminders(
        self, dispatch: Dict[str, QuestionnaireStatus],
        intervals: List[int],
    ) -> Dict[str, Any]:
        """Schedule follow-up reminders."""
        sent_count = sum(1 for v in dispatch.values() if v == QuestionnaireStatus.SENT)
        return {"scheduled": sent_count * len(intervals)}

    async def _collect_responses(
        self, dispatch: Dict[str, QuestionnaireStatus]
    ) -> Dict[str, Any]:
        """Collect questionnaire responses."""
        sent = sum(1 for v in dispatch.values() if v == QuestionnaireStatus.SENT)
        received = int(sent * 0.7)
        return {"received": received, "total_sent": sent}

    async def _validate_responses(self, responses: Dict) -> Dict[str, Any]:
        """Validate response completeness."""
        received = responses.get("received", 0)
        return {"valid": int(received * 0.9), "incomplete": int(received * 0.1)}

    async def _extract_esg_data(self, responses: Dict) -> Dict[str, Any]:
        """Extract structured ESG data from responses."""
        return {"data_points": responses.get("received", 0) * 25}

    async def _compute_preliminary_scores(
        self, esg_data: Dict
    ) -> Dict[str, Dict[str, Any]]:
        """Compute preliminary scores from questionnaire data."""
        scores = {}
        for supplier in self._suppliers:
            scores[supplier.supplier_id] = {
                "environmental": 65.0,
                "social": 60.0,
                "governance": 70.0,
                "confidence": 0.7,
            }
        return scores

    async def _score_environmental(
        self, supplier: SupplierProfile, prelim: Dict
    ) -> float:
        """Score environmental dimension (0-100)."""
        base = prelim.get("environmental", 50.0)
        cert_bonus = 5.0 if any("ISO14001" in c for c in supplier.certifications) else 0.0
        return min(100.0, base + cert_bonus)

    async def _score_social(
        self, supplier: SupplierProfile, prelim: Dict
    ) -> float:
        """Score social dimension (0-100)."""
        base = prelim.get("social", 50.0)
        cert_bonus = 5.0 if any("SA8000" in c for c in supplier.certifications) else 0.0
        return min(100.0, base + cert_bonus)

    async def _score_governance(
        self, supplier: SupplierProfile, prelim: Dict
    ) -> float:
        """Score governance dimension (0-100)."""
        return prelim.get("governance", 50.0)

    async def _estimate_scope3_upstream(
        self, suppliers: List[SupplierProfile],
        scores: Dict[str, ESGScore],
        categories: List[int],
    ) -> Dict[str, Any]:
        """Estimate Scope 3 upstream emissions from supplier data."""
        total_spend = sum(s.spend_eur for s in suppliers)
        # Spend-based estimation: industry-average EF per EUR
        ef_per_eur = 0.0005  # tCO2e per EUR spent (example)
        total_tco2e = total_spend * ef_per_eur
        return {
            "total_tco2e": round(total_tco2e, 2),
            "by_category": {f"cat_{c}": round(total_tco2e / len(categories), 2) for c in categories},
            "by_tier": {
                "tier_1": round(total_tco2e * 0.6, 2),
                "tier_2": round(total_tco2e * 0.25, 2),
                "tier_3": round(total_tco2e * 0.1, 2),
                "tier_4": round(total_tco2e * 0.05, 2),
            },
        }

    async def _generate_environmental_actions(
        self, supplier: SupplierProfile, score: ESGScore
    ) -> List[CorrectiveAction]:
        """Generate environmental corrective actions."""
        return [CorrectiveAction(
            supplier_id=supplier.supplier_id,
            supplier_name=supplier.name,
            priority=ActionPriority.IMMEDIATE if score.risk_tier == RiskTier.CRITICAL else ActionPriority.HIGH,
            dimension="environmental",
            description=f"Implement emissions reduction program at {supplier.name}",
            target_improvement=20.0,
            deadline_days=90 if score.risk_tier == RiskTier.CRITICAL else 180,
            estimated_scope3_impact_tco2e=supplier.spend_eur * 0.0001,
            monitoring_frequency="monthly",
        )]

    async def _generate_social_actions(
        self, supplier: SupplierProfile, score: ESGScore
    ) -> List[CorrectiveAction]:
        """Generate social corrective actions."""
        return [CorrectiveAction(
            supplier_id=supplier.supplier_id,
            supplier_name=supplier.name,
            priority=ActionPriority.HIGH,
            dimension="social",
            description=f"Conduct social compliance audit at {supplier.name}",
            target_improvement=15.0,
            deadline_days=120,
            monitoring_frequency="quarterly",
        )]

    async def _generate_governance_actions(
        self, supplier: SupplierProfile, score: ESGScore
    ) -> List[CorrectiveAction]:
        """Generate governance corrective actions."""
        return [CorrectiveAction(
            supplier_id=supplier.supplier_id,
            supplier_name=supplier.name,
            priority=ActionPriority.MEDIUM,
            dimension="governance",
            description=f"Establish anti-corruption policy at {supplier.name}",
            target_improvement=10.0,
            deadline_days=180,
            monitoring_frequency="semi-annual",
        )]

    async def _generate_improvement_report(
        self, actions: List[CorrectiveAction], scores: Dict[str, ESGScore]
    ) -> Dict[str, Any]:
        """Generate improvement plan summary report."""
        return {"report_id": f"impr-{uuid.uuid4().hex[:8]}"}

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _hash_data(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
