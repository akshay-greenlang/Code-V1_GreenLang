# -*- coding: utf-8 -*-
"""
Multi-Entity Consolidation Workflow
=========================================

4-phase workflow for consolidating Scope 3 inventories across multi-entity
corporate structures within PACK-043 Scope 3 Complete Pack.

Phases:
    1. ENTITY_MAPPING             -- Build entity hierarchy (parent ->
                                     subsidiaries -> JVs -> franchises).
    2. BOUNDARY_DEFINITION        -- Define consolidation approach per entity
                                     (equity share, operational control,
                                     financial control).
    3. PROPORTIONAL_CONSOLIDATION -- Apply ownership % to entity-level Scope 3
                                     results to produce group totals.
    4. INTERCOMPANY_ELIMINATION   -- Detect and remove inter-company double-
                                     counting from consolidated totals.

The workflow follows GreenLang zero-hallucination principles: every ownership
percentage, proportional allocation, and double-counting adjustment uses
deterministic arithmetic. SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Corporate Standard -- Chapter 3 (Setting Organizational Boundaries)
    GHG Protocol Scope 3 Standard -- Chapter 5 (Organizational Boundaries)
    IFRS S2 / ESRS E1 -- Consolidation requirements
    ISO 14064-1:2018 -- Clause 5.1 (Organizational boundaries)

Schedule: annually at group-level inventory consolidation
Estimated duration: 4-8 hours

Author: GreenLang Platform Team
Version: 43.0.0
"""

_MODULE_VERSION: str = "43.0.0"

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

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


class EntityType(str, Enum):
    """Types of corporate entities."""

    PARENT = "parent"
    SUBSIDIARY = "subsidiary"
    JOINT_VENTURE = "joint_venture"
    ASSOCIATE = "associate"
    FRANCHISE = "franchise"
    LEASED_OPERATION = "leased_operation"
    INVESTEE = "investee"


class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approaches."""

    EQUITY_SHARE = "equity_share"
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"


class DoubleCountType(str, Enum):
    """Types of inter-company double counting."""

    INTRA_GROUP_SALE = "intra_group_sale"
    SHARED_SERVICE = "shared_service"
    INTERNAL_TRANSPORT = "internal_transport"
    TRANSFER_PRICING = "transfer_pricing"
    COMMON_SUPPLIER = "common_supplier"


class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories (1-15)."""

    CAT_01_PURCHASED_GOODS = "cat_01_purchased_goods_services"
    CAT_02_CAPITAL_GOODS = "cat_02_capital_goods"
    CAT_03_FUEL_ENERGY = "cat_03_fuel_energy_related"
    CAT_04_UPSTREAM_TRANSPORT = "cat_04_upstream_transport"
    CAT_05_WASTE = "cat_05_waste_in_operations"
    CAT_06_BUSINESS_TRAVEL = "cat_06_business_travel"
    CAT_07_COMMUTING = "cat_07_employee_commuting"
    CAT_08_UPSTREAM_LEASED = "cat_08_upstream_leased_assets"
    CAT_09_DOWNSTREAM_TRANSPORT = "cat_09_downstream_transport"
    CAT_10_PROCESSING = "cat_10_processing_sold_products"
    CAT_11_USE_SOLD = "cat_11_use_of_sold_products"
    CAT_12_END_OF_LIFE = "cat_12_end_of_life_treatment"
    CAT_13_DOWNSTREAM_LEASED = "cat_13_downstream_leased_assets"
    CAT_14_FRANCHISES = "cat_14_franchises"
    CAT_15_INVESTMENTS = "cat_15_investments"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class WorkflowState(BaseModel):
    """Persistent state for checkpoint/resume."""

    workflow_id: str = Field(default="")
    current_phase: int = Field(default=0)
    phase_statuses: Dict[str, str] = Field(default_factory=dict)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default="")
    updated_at: str = Field(default="")


class EntityRecord(BaseModel):
    """Corporate entity record."""

    entity_id: str = Field(default_factory=lambda: f"ent-{uuid.uuid4().hex[:8]}")
    entity_name: str = Field(default="")
    entity_type: EntityType = Field(default=EntityType.SUBSIDIARY)
    parent_entity_id: str = Field(default="", description="ID of parent entity")
    country: str = Field(default="", description="ISO 3166-1 alpha-2")
    ownership_pct: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Equity ownership percentage",
    )
    has_operational_control: bool = Field(default=False)
    has_financial_control: bool = Field(default=False)
    revenue_usd: float = Field(default=0.0, ge=0.0)
    employee_count: int = Field(default=0, ge=0)


class EntityScope3Data(BaseModel):
    """Scope 3 emission data for a single entity."""

    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=2025)
    total_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    category_emissions: Dict[str, float] = Field(
        default_factory=dict,
        description="Category enum value -> tCO2e",
    )
    data_source: str = Field(default="pack_042")


class EntityBoundary(BaseModel):
    """Consolidation boundary definition for an entity."""

    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    entity_type: EntityType = Field(default=EntityType.SUBSIDIARY)
    approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL
    )
    inclusion_factor: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Factor to apply (0.0 = exclude, 1.0 = full)",
    )
    rationale: str = Field(default="")


class ConsolidatedCategory(BaseModel):
    """Consolidated emissions for a single Scope 3 category."""

    category: str = Field(default="")
    raw_total_tco2e: float = Field(default=0.0, ge=0.0)
    consolidated_tco2e: float = Field(default=0.0, ge=0.0)
    entity_contributions: Dict[str, float] = Field(
        default_factory=dict, description="Entity ID -> tCO2e contribution"
    )
    double_count_adjustment_tco2e: float = Field(default=0.0, ge=0.0)


class DoubleCountFlag(BaseModel):
    """Detected double counting instance."""

    flag_id: str = Field(default_factory=lambda: f"dc-{uuid.uuid4().hex[:8]}")
    entity_a_id: str = Field(default="")
    entity_a_name: str = Field(default="")
    entity_b_id: str = Field(default="")
    entity_b_name: str = Field(default="")
    double_count_type: DoubleCountType = Field(default=DoubleCountType.INTRA_GROUP_SALE)
    category: str = Field(default="")
    estimated_overlap_tco2e: float = Field(default=0.0, ge=0.0)
    resolution: str = Field(default="")
    resolved: bool = Field(default=False)


class IntercompanyTransaction(BaseModel):
    """Inter-company transaction record for double-counting detection."""

    transaction_id: str = Field(
        default_factory=lambda: f"txn-{uuid.uuid4().hex[:8]}"
    )
    seller_entity_id: str = Field(default="")
    buyer_entity_id: str = Field(default="")
    transaction_type: DoubleCountType = Field(default=DoubleCountType.INTRA_GROUP_SALE)
    amount_usd: float = Field(default=0.0, ge=0.0)
    estimated_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_category: str = Field(default="")
    description: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class MultiEntityInput(BaseModel):
    """Input data model for MultiEntityWorkflow."""

    organization_name: str = Field(default="", description="Group / parent name")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL,
    )
    entities: List[EntityRecord] = Field(default_factory=list)
    entity_scope3_data: List[EntityScope3Data] = Field(default_factory=list)
    intercompany_transactions: List[IntercompanyTransaction] = Field(
        default_factory=list
    )
    boundary_overrides: Dict[str, str] = Field(
        default_factory=dict,
        description="Entity ID -> override approach (e.g. equity_share)",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class MultiEntityOutput(BaseModel):
    """Complete output from MultiEntityWorkflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="multi_entity")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_name: str = Field(default="")
    reporting_year: int = Field(default=2025)
    consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL
    )
    entity_count: int = Field(default=0, ge=0)
    hierarchy_depth: int = Field(default=0, ge=0)
    entity_boundaries: List[EntityBoundary] = Field(default_factory=list)
    consolidated_categories: List[ConsolidatedCategory] = Field(default_factory=list)
    raw_total_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    consolidated_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    double_count_flags: List[DoubleCountFlag] = Field(default_factory=list)
    total_double_count_eliminated_tco2e: float = Field(default=0.0, ge=0.0)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class MultiEntityWorkflow:
    """
    4-phase multi-entity Scope 3 consolidation workflow.

    Builds entity hierarchy, defines consolidation boundaries per entity,
    applies proportional ownership to entity-level Scope 3 results, and
    detects/eliminates inter-company double counting.

    Zero-hallucination: ownership percentages, inclusion factors, and
    double-counting adjustments are all deterministic arithmetic on
    auditable input data.

    Attributes:
        workflow_id: Unique execution identifier.
        _entities: Entity records.
        _boundaries: Entity boundary definitions.
        _consolidated: Category-level consolidated results.
        _double_counts: Detected double-counting flags.
        _phase_results: Ordered phase outputs.
        _state: Checkpoint/resume state.

    Example:
        >>> wf = MultiEntityWorkflow()
        >>> inp = MultiEntityInput(
        ...     entities=[
        ...         EntityRecord(entity_name="Parent Co", entity_type=EntityType.PARENT),
        ...         EntityRecord(entity_name="Sub A", entity_type=EntityType.SUBSIDIARY,
        ...                      ownership_pct=80.0),
        ...     ],
        ...     entity_scope3_data=[...],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_NAMES: List[str] = [
        "entity_mapping",
        "boundary_definition",
        "proportional_consolidation",
        "intercompany_elimination",
    ]

    PHASE_WEIGHTS: Dict[str, float] = {
        "entity_mapping": 15.0,
        "boundary_definition": 20.0,
        "proportional_consolidation": 35.0,
        "intercompany_elimination": 30.0,
    }

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize MultiEntityWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._entities: List[EntityRecord] = []
        self._entity_map: Dict[str, EntityRecord] = {}
        self._children_map: Dict[str, List[str]] = {}
        self._hierarchy_depth: int = 0
        self._boundaries: List[EntityBoundary] = []
        self._boundary_map: Dict[str, EntityBoundary] = {}
        self._consolidated: List[ConsolidatedCategory] = []
        self._double_counts: List[DoubleCountFlag] = []
        self._phase_results: List[PhaseResult] = []
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[MultiEntityInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> MultiEntityOutput:
        """
        Execute the 4-phase multi-entity consolidation workflow.

        Args:
            input_data: Full input model.
            config: Optional configuration overrides.

        Returns:
            MultiEntityOutput with consolidated Scope 3 totals.
        """
        if input_data is None:
            input_data = MultiEntityInput()

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting multi-entity workflow %s org=%s entities=%d",
            self.workflow_id,
            input_data.organization_name,
            len(input_data.entities),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING
        self._update_progress(0.0)

        try:
            phase1 = await self._execute_with_retry(
                self._phase_entity_mapping, input_data, phase_number=1
            )
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 1 failed: {phase1.errors}")
            self._update_progress(15.0)

            phase2 = await self._execute_with_retry(
                self._phase_boundary_definition, input_data, phase_number=2
            )
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 2 failed: {phase2.errors}")
            self._update_progress(35.0)

            phase3 = await self._execute_with_retry(
                self._phase_proportional_consolidation, input_data, phase_number=3
            )
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 3 failed: {phase3.errors}")
            self._update_progress(70.0)

            phase4 = await self._execute_with_retry(
                self._phase_intercompany_elimination, input_data, phase_number=4
            )
            self._phase_results.append(phase4)
            if phase4.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 4 failed: {phase4.errors}")
            self._update_progress(100.0)

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Multi-entity workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(
                PhaseResult(
                    phase_name="error", phase_number=0,
                    status=PhaseStatus.FAILED, errors=[str(exc)],
                )
            )

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        raw_total = sum(cc.raw_total_tco2e for cc in self._consolidated)
        cons_total = sum(cc.consolidated_tco2e for cc in self._consolidated)
        dc_total = sum(dc.estimated_overlap_tco2e for dc in self._double_counts if dc.resolved)

        result = MultiEntityOutput(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_name=input_data.organization_name,
            reporting_year=input_data.reporting_year,
            consolidation_approach=input_data.consolidation_approach,
            entity_count=len(self._entities),
            hierarchy_depth=self._hierarchy_depth,
            entity_boundaries=self._boundaries,
            consolidated_categories=self._consolidated,
            raw_total_scope3_tco2e=round(raw_total, 2),
            consolidated_scope3_tco2e=round(cons_total, 2),
            double_count_flags=self._double_counts,
            total_double_count_eliminated_tco2e=round(dc_total, 2),
            progress_pct=self._state.progress_pct,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Multi-entity workflow %s completed in %.2fs status=%s "
            "raw=%.1f cons=%.1f dc_elim=%.1f tCO2e",
            self.workflow_id, elapsed, overall_status.value,
            raw_total, cons_total, dc_total,
        )
        return result

    def get_state(self) -> WorkflowState:
        """Return current workflow state for checkpoint/resume."""
        return self._state.model_copy()

    async def resume(
        self, state: WorkflowState, input_data: MultiEntityInput
    ) -> MultiEntityOutput:
        """Resume workflow from a saved checkpoint state."""
        self._state = state
        self.workflow_id = state.workflow_id
        return await self.execute(input_data)

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: MultiEntityInput, phase_number: int
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s",
                        phase_number, attempt, self.MAX_RETRIES, exc,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Entity Mapping
    # -------------------------------------------------------------------------

    async def _phase_entity_mapping(
        self, input_data: MultiEntityInput
    ) -> PhaseResult:
        """Build entity hierarchy."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._entities = list(input_data.entities)
        self._entity_map = {e.entity_id: e for e in self._entities}

        # Build parent -> children map
        self._children_map = {}
        for entity in self._entities:
            if entity.parent_entity_id:
                self._children_map.setdefault(
                    entity.parent_entity_id, []
                ).append(entity.entity_id)

        # Find root entities (parents with no parent_entity_id)
        roots = [e for e in self._entities if not e.parent_entity_id]
        if not roots:
            warnings.append(
                "No root entity found (no entity with empty parent_entity_id); "
                "treating first entity as root"
            )
            if self._entities:
                self._entities[0].parent_entity_id = ""
                roots = [self._entities[0]]

        # Compute hierarchy depth
        self._hierarchy_depth = self._compute_depth(roots)

        # Entity type distribution
        type_dist: Dict[str, int] = {}
        for e in self._entities:
            t = e.entity_type.value
            type_dist[t] = type_dist.get(t, 0) + 1

        # Country distribution
        country_dist: Dict[str, int] = {}
        for e in self._entities:
            c = e.country or "unknown"
            country_dist[c] = country_dist.get(c, 0) + 1

        outputs["total_entities"] = len(self._entities)
        outputs["root_entities"] = len(roots)
        outputs["hierarchy_depth"] = self._hierarchy_depth
        outputs["entity_type_distribution"] = type_dist
        outputs["country_distribution"] = country_dist
        outputs["total_group_revenue_usd"] = round(
            sum(e.revenue_usd for e in self._entities), 2
        )

        self._state.phase_statuses["entity_mapping"] = "completed"
        self._state.current_phase = 1

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 EntityMapping: %d entities, depth=%d, roots=%d",
            len(self._entities), self._hierarchy_depth, len(roots),
        )
        return PhaseResult(
            phase_name="entity_mapping", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Boundary Definition
    # -------------------------------------------------------------------------

    async def _phase_boundary_definition(
        self, input_data: MultiEntityInput
    ) -> PhaseResult:
        """Define consolidation approach per entity."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._boundaries = []
        self._boundary_map = {}
        default_approach = input_data.consolidation_approach

        for entity in self._entities:
            # Check for override
            override = input_data.boundary_overrides.get(entity.entity_id)
            if override:
                approach = ConsolidationApproach(override)
            else:
                approach = default_approach

            # Calculate inclusion factor based on approach
            inclusion = self._calculate_inclusion_factor(entity, approach)
            rationale = self._generate_boundary_rationale(entity, approach, inclusion)

            boundary = EntityBoundary(
                entity_id=entity.entity_id,
                entity_name=entity.entity_name,
                entity_type=entity.entity_type,
                approach=approach,
                inclusion_factor=round(inclusion, 4),
                rationale=rationale,
            )
            self._boundaries.append(boundary)
            self._boundary_map[entity.entity_id] = boundary

        # Approach distribution
        approach_dist: Dict[str, int] = {}
        for b in self._boundaries:
            a = b.approach.value
            approach_dist[a] = approach_dist.get(a, 0) + 1

        outputs["boundaries_defined"] = len(self._boundaries)
        outputs["approach_distribution"] = approach_dist
        outputs["entities_fully_included"] = sum(
            1 for b in self._boundaries if b.inclusion_factor == 1.0
        )
        outputs["entities_partially_included"] = sum(
            1 for b in self._boundaries if 0.0 < b.inclusion_factor < 1.0
        )
        outputs["entities_excluded"] = sum(
            1 for b in self._boundaries if b.inclusion_factor == 0.0
        )

        self._state.phase_statuses["boundary_definition"] = "completed"
        self._state.current_phase = 2

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 BoundaryDefinition: %d boundaries, full=%d partial=%d excluded=%d",
            len(self._boundaries),
            outputs["entities_fully_included"],
            outputs["entities_partially_included"],
            outputs["entities_excluded"],
        )
        return PhaseResult(
            phase_name="boundary_definition", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Proportional Consolidation
    # -------------------------------------------------------------------------

    async def _phase_proportional_consolidation(
        self, input_data: MultiEntityInput
    ) -> PhaseResult:
        """Apply ownership % to entity-level Scope 3 results."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Build scope3 data lookup
        scope3_lookup: Dict[str, EntityScope3Data] = {
            sd.entity_id: sd for sd in input_data.entity_scope3_data
        }

        # Collect all categories
        all_categories: Set[str] = set()
        for sd in input_data.entity_scope3_data:
            all_categories.update(sd.category_emissions.keys())

        self._consolidated = []

        for cat in sorted(all_categories):
            raw_total = 0.0
            cons_total = 0.0
            contributions: Dict[str, float] = {}

            for entity in self._entities:
                sd = scope3_lookup.get(entity.entity_id)
                if not sd:
                    continue

                cat_emissions = sd.category_emissions.get(cat, 0.0)
                raw_total += cat_emissions

                boundary = self._boundary_map.get(entity.entity_id)
                factor = boundary.inclusion_factor if boundary else 1.0
                contribution = cat_emissions * factor
                cons_total += contribution
                contributions[entity.entity_id] = round(contribution, 4)

            self._consolidated.append(ConsolidatedCategory(
                category=cat,
                raw_total_tco2e=round(raw_total, 4),
                consolidated_tco2e=round(cons_total, 4),
                entity_contributions=contributions,
            ))

        total_raw = sum(cc.raw_total_tco2e for cc in self._consolidated)
        total_cons = sum(cc.consolidated_tco2e for cc in self._consolidated)

        entities_with_data = len(scope3_lookup)
        entities_without_data = len(self._entities) - entities_with_data
        if entities_without_data > 0:
            warnings.append(
                f"{entities_without_data} entities have no Scope 3 data; "
                f"they contribute zero to consolidated totals"
            )

        outputs["categories_consolidated"] = len(self._consolidated)
        outputs["raw_total_scope3_tco2e"] = round(total_raw, 2)
        outputs["consolidated_scope3_tco2e"] = round(total_cons, 2)
        outputs["proportional_reduction_pct"] = round(
            ((total_raw - total_cons) / total_raw * 100.0) if total_raw > 0 else 0.0, 2
        )
        outputs["entities_with_data"] = entities_with_data
        outputs["entities_without_data"] = entities_without_data

        self._state.phase_statuses["proportional_consolidation"] = "completed"
        self._state.current_phase = 3

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ProportionalConsolidation: raw=%.1f cons=%.1f tCO2e, "
            "%d categories",
            total_raw, total_cons, len(self._consolidated),
        )
        return PhaseResult(
            phase_name="proportional_consolidation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Intercompany Elimination
    # -------------------------------------------------------------------------

    async def _phase_intercompany_elimination(
        self, input_data: MultiEntityInput
    ) -> PhaseResult:
        """Detect and remove inter-company double-counting."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._double_counts = []

        # Process explicit intercompany transactions
        for txn in input_data.intercompany_transactions:
            seller = self._entity_map.get(txn.seller_entity_id)
            buyer = self._entity_map.get(txn.buyer_entity_id)

            if not seller or not buyer:
                warnings.append(
                    f"Inter-company transaction {txn.transaction_id} references "
                    f"unknown entity; skipping"
                )
                continue

            # Both entities must be within boundary
            seller_boundary = self._boundary_map.get(txn.seller_entity_id)
            buyer_boundary = self._boundary_map.get(txn.buyer_entity_id)

            if not seller_boundary or not buyer_boundary:
                continue
            if seller_boundary.inclusion_factor == 0 or buyer_boundary.inclusion_factor == 0:
                continue

            # The overlap is the emissions from the transaction
            overlap = txn.estimated_emissions_tco2e * min(
                seller_boundary.inclusion_factor,
                buyer_boundary.inclusion_factor,
            )

            flag = DoubleCountFlag(
                entity_a_id=txn.seller_entity_id,
                entity_a_name=seller.entity_name,
                entity_b_id=txn.buyer_entity_id,
                entity_b_name=buyer.entity_name,
                double_count_type=txn.transaction_type,
                category=txn.scope3_category,
                estimated_overlap_tco2e=round(overlap, 4),
                resolution=f"Eliminate {overlap:.2f} tCO2e from consolidated total "
                           f"(retain at seller entity level)",
                resolved=True,
            )
            self._double_counts.append(flag)

        # Auto-detect common patterns: parent-subsidiary intra-group sales
        self._detect_implicit_double_counts(input_data, warnings)

        # Apply eliminations to consolidated categories
        total_eliminated = 0.0
        for dc in self._double_counts:
            if not dc.resolved:
                continue
            if dc.category:
                for cc in self._consolidated:
                    if cc.category == dc.category:
                        cc.double_count_adjustment_tco2e += dc.estimated_overlap_tco2e
                        cc.consolidated_tco2e = max(
                            cc.consolidated_tco2e - dc.estimated_overlap_tco2e, 0.0
                        )
                        cc.consolidated_tco2e = round(cc.consolidated_tco2e, 4)
                        break
            total_eliminated += dc.estimated_overlap_tco2e

        outputs["intercompany_transactions_processed"] = len(
            input_data.intercompany_transactions
        )
        outputs["double_count_flags_raised"] = len(self._double_counts)
        outputs["double_count_flags_resolved"] = sum(
            1 for dc in self._double_counts if dc.resolved
        )
        outputs["total_eliminated_tco2e"] = round(total_eliminated, 2)
        outputs["final_consolidated_scope3_tco2e"] = round(
            sum(cc.consolidated_tco2e for cc in self._consolidated), 2
        )
        outputs["double_count_types"] = dict(
            sorted(
                {
                    dc.double_count_type.value: sum(
                        1 for d in self._double_counts
                        if d.double_count_type == dc.double_count_type
                    )
                    for dc in self._double_counts
                }.items()
            )
        )

        self._state.phase_statuses["intercompany_elimination"] = "completed"
        self._state.current_phase = 4

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 IntercompanyElimination: %d flags, %.1f tCO2e eliminated, "
            "final=%.1f tCO2e",
            len(self._double_counts),
            total_eliminated,
            outputs["final_consolidated_scope3_tco2e"],
        )
        return PhaseResult(
            phase_name="intercompany_elimination", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _compute_depth(self, roots: List[EntityRecord]) -> int:
        """Compute hierarchy depth via BFS."""
        if not roots:
            return 0
        max_depth = 1
        queue: List[Tuple[str, int]] = [(r.entity_id, 1) for r in roots]
        while queue:
            entity_id, depth = queue.pop(0)
            max_depth = max(max_depth, depth)
            for child_id in self._children_map.get(entity_id, []):
                queue.append((child_id, depth + 1))
        return max_depth

    def _calculate_inclusion_factor(
        self, entity: EntityRecord, approach: ConsolidationApproach
    ) -> float:
        """Calculate inclusion factor based on consolidation approach."""
        if approach == ConsolidationApproach.EQUITY_SHARE:
            return entity.ownership_pct / 100.0
        elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
            return 1.0 if entity.has_operational_control else 0.0
        elif approach == ConsolidationApproach.FINANCIAL_CONTROL:
            return 1.0 if entity.has_financial_control else 0.0
        return 1.0

    def _generate_boundary_rationale(
        self,
        entity: EntityRecord,
        approach: ConsolidationApproach,
        inclusion: float,
    ) -> str:
        """Generate rationale for boundary inclusion decision."""
        if approach == ConsolidationApproach.EQUITY_SHARE:
            return (
                f"Equity share approach: {entity.ownership_pct}% ownership "
                f"-> {inclusion * 100:.1f}% inclusion"
            )
        elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
            status = "included (100%)" if inclusion > 0 else "excluded"
            return (
                f"Operational control approach: {entity.entity_name} "
                f"{'has' if entity.has_operational_control else 'does not have'} "
                f"operational control -> {status}"
            )
        else:
            status = "included (100%)" if inclusion > 0 else "excluded"
            return (
                f"Financial control approach: {entity.entity_name} "
                f"{'has' if entity.has_financial_control else 'does not have'} "
                f"financial control -> {status}"
            )

    def _detect_implicit_double_counts(
        self,
        input_data: MultiEntityInput,
        warnings: List[str],
    ) -> None:
        """Detect implicit double-counting from entity hierarchy."""
        # Check for subsidiaries reporting to same categories as parent
        scope3_lookup = {sd.entity_id: sd for sd in input_data.entity_scope3_data}

        for parent_id, child_ids in self._children_map.items():
            parent_data = scope3_lookup.get(parent_id)
            if not parent_data:
                continue

            for child_id in child_ids:
                child_data = scope3_lookup.get(child_id)
                if not child_data:
                    continue

                # Find overlapping categories
                parent_cats = set(parent_data.category_emissions.keys())
                child_cats = set(child_data.category_emissions.keys())
                overlap_cats = parent_cats & child_cats

                for cat in overlap_cats:
                    parent_val = parent_data.category_emissions.get(cat, 0.0)
                    child_val = child_data.category_emissions.get(cat, 0.0)

                    # Simple heuristic: if child's contribution is > 5% of
                    # parent's for same category, flag potential double count
                    if parent_val > 0 and child_val / parent_val > 0.05:
                        parent_ent = self._entity_map.get(parent_id)
                        child_ent = self._entity_map.get(child_id)
                        if parent_ent and child_ent:
                            self._double_counts.append(DoubleCountFlag(
                                entity_a_id=parent_id,
                                entity_a_name=parent_ent.entity_name,
                                entity_b_id=child_id,
                                entity_b_name=child_ent.entity_name,
                                double_count_type=DoubleCountType.INTRA_GROUP_SALE,
                                category=cat,
                                estimated_overlap_tco2e=0.0,
                                resolution="Review required: potential parent-subsidiary overlap",
                                resolved=False,
                            ))

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state."""
        self._entities = []
        self._entity_map = {}
        self._children_map = {}
        self._hierarchy_depth = 0
        self._boundaries = []
        self._boundary_map = {}
        self._consolidated = []
        self._double_counts = []
        self._phase_results = []
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )

    def _update_progress(self, pct: float) -> None:
        """Update progress percentage in state."""
        self._state.progress_pct = min(pct, 100.0)
        self._state.updated_at = datetime.utcnow().isoformat()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: MultiEntityOutput) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += f"|{result.workflow_id}|{result.consolidated_scope3_tco2e}"
        chain += f"|{result.total_double_count_eliminated_tco2e}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
