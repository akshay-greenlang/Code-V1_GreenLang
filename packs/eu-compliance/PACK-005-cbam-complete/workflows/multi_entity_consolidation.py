# -*- coding: utf-8 -*-
"""
Multi-Entity Consolidation Workflow
======================================

Five-phase quarterly consolidation workflow for CBAM obligations across
multi-entity corporate groups operating in multiple EU member states.
Gathers per-entity import data, runs per-entity CBAM calculations,
consolidates at group level with de minimis checks, allocates costs
using configurable methods, and generates comprehensive reporting.

Regulatory Context:
    Per EU CBAM Regulation 2023/956:
    - Article 2(11): Each authorized declarant submits individually, but
      corporate groups may coordinate certificate purchasing.
    - Article 5: De minimis threshold per consignment per goods category
      (EUR 150 or 150 kg net mass).
    - Article 22(2): Quarterly holding >= 50% of estimated annual obligation.
    - Cross-entity certificate netting is NOT allowed per regulation
      (certificates are non-transferable), but cost allocation for
      internal management reporting is permitted.

Cost Allocation Methods:
    - VOLUME: Allocate by imported volume (tonnes)
    - REVENUE: Allocate by entity revenue share
    - PROFIT_CENTER: Allocate by profit center assignment
    - EQUAL: Equal allocation across all entities
    - CUSTOM: Custom allocation weights provided by user

Phases:
    1. DataCollection - Gather import data per entity, validate completeness
    2. EntityCalculation - Run CBAM calculations per entity
    3. GroupAggregation - Consolidate obligations, de minimis, netting
    4. CostAllocation - Distribute costs using configured method
    5. Reporting - Consolidated and per-entity reporting

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class CostAllocationMethod(str, Enum):
    """Certificate cost allocation method for group entities."""
    VOLUME = "VOLUME"
    REVENUE = "REVENUE"
    PROFIT_CENTER = "PROFIT_CENTER"
    EQUAL = "EQUAL"
    CUSTOM = "CUSTOM"


class DeMinimisStatus(str, Enum):
    """De minimis threshold assessment status."""
    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    ABOVE_THRESHOLD = "ABOVE_THRESHOLD"
    BORDERLINE = "BORDERLINE"


# =============================================================================
# CONSTANTS
# =============================================================================

DE_MINIMIS_VALUE_EUR = Decimal("150.00")
DE_MINIMIS_MASS_KG = Decimal("150.00")
QUARTERLY_HOLDING_PCT = Decimal("0.50")


# =============================================================================
# DATA MODELS - SHARED
# =============================================================================


class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(...)
    execution_timestamp: datetime = Field(default_factory=datetime.utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        """Store phase outputs for downstream consumption."""
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        """Retrieve outputs from a previous phase."""
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Record phase status for checkpoint/resume."""
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase has already completed."""
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(...)
    status: WorkflowStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# DATA MODELS - MULTI-ENTITY CONSOLIDATION
# =============================================================================


class EntityImportData(BaseModel):
    """Import data for a single group entity."""
    entity_id: str = Field(..., description="Entity identifier")
    entity_name: str = Field(default="")
    member_state: str = Field(..., description="EU member state ISO code")
    import_records: List[Dict[str, Any]] = Field(default_factory=list)
    total_imports_tonnes: float = Field(default=0.0, ge=0)
    total_imports_value_eur: float = Field(default=0.0, ge=0)
    goods_categories: List[str] = Field(default_factory=list)
    data_complete: bool = Field(default=False)
    missing_fields: List[str] = Field(default_factory=list)
    contact_email: Optional[str] = Field(None)


class EntityCalculationResult(BaseModel):
    """CBAM calculation result for a single entity."""
    entity_id: str = Field(...)
    entity_name: str = Field(default="")
    total_embedded_emissions_tco2e: float = Field(default=0.0, ge=0)
    direct_emissions_tco2e: float = Field(default=0.0, ge=0)
    indirect_emissions_tco2e: float = Field(default=0.0, ge=0)
    certificate_obligation: float = Field(default=0.0, ge=0)
    free_allocation_deduction: float = Field(default=0.0, ge=0)
    carbon_price_deduction: float = Field(default=0.0, ge=0)
    net_obligation: float = Field(default=0.0, ge=0)
    de_minimis_status: DeMinimisStatus = Field(
        default=DeMinimisStatus.ABOVE_THRESHOLD
    )
    estimated_cost_eur: float = Field(default=0.0, ge=0)
    by_goods_category: Dict[str, float] = Field(default_factory=dict)


class CostAllocationEntry(BaseModel):
    """Cost allocation for a single entity."""
    entity_id: str = Field(...)
    entity_name: str = Field(default="")
    allocation_weight: float = Field(default=0.0, ge=0)
    allocated_cost_eur: float = Field(default=0.0, ge=0)
    allocated_certificates: float = Field(default=0.0, ge=0)
    allocation_method: str = Field(default="")
    justification: str = Field(default="")


class MultiEntityConsolidationInput(BaseModel):
    """Input configuration for multi-entity consolidation."""
    organization_id: str = Field(..., description="Group organization ID")
    group_name: str = Field(default="", description="Corporate group name")
    reporting_year: int = Field(..., ge=2026, le=2050)
    reporting_quarter: int = Field(..., ge=1, le=4)
    entities: List[EntityImportData] = Field(
        ..., min_length=1, description="Entity data list"
    )
    cost_allocation_method: CostAllocationMethod = Field(
        default=CostAllocationMethod.VOLUME
    )
    custom_allocation_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Custom weights by entity_id (for CUSTOM method)"
    )
    entity_revenues: Dict[str, float] = Field(
        default_factory=dict,
        description="Entity revenues for REVENUE method"
    )
    ets_price_eur: float = Field(default=80.0, ge=0)
    enable_cross_entity_netting: bool = Field(default=False)
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_year")
    @classmethod
    def validate_year(cls, v: int) -> int:
        """Validate reporting year is in CBAM definitive period."""
        if v < 2026:
            raise ValueError("CBAM definitive period starts 2026")
        return v


class MultiEntityConsolidationResult(WorkflowResult):
    """Complete result from multi-entity consolidation."""
    entity_count: int = Field(default=0)
    group_total_obligation_tco2e: float = Field(default=0.0)
    group_total_cost_eur: float = Field(default=0.0)
    entities_above_de_minimis: int = Field(default=0)
    entities_below_de_minimis: int = Field(default=0)
    cost_allocation_method: str = Field(default="")
    entity_summaries: List[Dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================


class DataCollectionPhase:
    """
    Phase 1: Data Collection.

    Gathers import data from all group entities across EU member states,
    validates completeness per entity, and tracks missing data for
    follow-up with entity contacts.
    """

    PHASE_NAME = "data_collection"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute data collection phase.

        Args:
            context: Workflow context with entity data.

        Returns:
            PhaseResult with per-entity data status and completeness.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            entities = config.get("entities", [])
            year = config.get("reporting_year", 0)
            quarter = config.get("reporting_quarter", 0)

            entity_statuses: List[Dict[str, Any]] = []
            complete_count = 0
            incomplete_count = 0
            total_imports = Decimal("0")
            total_value = Decimal("0")

            for entity in entities:
                entity_id = entity.get("entity_id", "")
                entity_name = entity.get("entity_name", entity_id)
                data_complete = entity.get("data_complete", False)
                missing = entity.get("missing_fields", [])
                imports_tonnes = Decimal(str(
                    entity.get("total_imports_tonnes", 0)
                ))
                imports_value = Decimal(str(
                    entity.get("total_imports_value_eur", 0)
                ))

                # Validate required fields
                required_fields = [
                    "entity_id", "member_state", "total_imports_tonnes"
                ]
                validation_errors = []
                for field in required_fields:
                    if not entity.get(field):
                        validation_errors.append(f"Missing required: {field}")

                if validation_errors:
                    data_complete = False
                    missing = list(set(missing + validation_errors))

                entity_status = {
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "member_state": entity.get("member_state", ""),
                    "data_complete": data_complete,
                    "missing_fields": missing,
                    "import_records_count": len(
                        entity.get("import_records", [])
                    ),
                    "total_imports_tonnes": float(imports_tonnes),
                    "total_imports_value_eur": float(imports_value),
                    "goods_categories": entity.get("goods_categories", []),
                }
                entity_statuses.append(entity_status)

                if data_complete and not validation_errors:
                    complete_count += 1
                else:
                    incomplete_count += 1
                    warnings.append(
                        f"Entity '{entity_name}' ({entity_id}): "
                        f"incomplete data - {', '.join(missing)}"
                    )

                total_imports += imports_tonnes
                total_value += imports_value

            outputs["entity_statuses"] = entity_statuses
            outputs["total_entities"] = len(entities)
            outputs["entities_complete"] = complete_count
            outputs["entities_incomplete"] = incomplete_count
            outputs["total_imports_tonnes"] = float(total_imports)
            outputs["total_imports_value_eur"] = float(total_value)
            outputs["completeness_pct"] = (
                round(complete_count / max(len(entities), 1) * 100, 1)
            )
            outputs["member_states"] = list(set(
                e.get("member_state", "") for e in entities if e.get("member_state")
            ))

            if incomplete_count > 0:
                warnings.append(
                    f"{incomplete_count}/{len(entities)} entities have "
                    f"incomplete data"
                )

            status = PhaseStatus.COMPLETED
            records = sum(
                len(e.get("import_records", [])) for e in entities
            )

        except Exception as exc:
            logger.error("DataCollection failed: %s", exc, exc_info=True)
            errors.append(f"Data collection failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )


class EntityCalculationPhase:
    """
    Phase 2: Entity Calculation.

    Runs the PACK-004 CBAM calculation engine for each entity
    individually, computing embedded emissions, certificate obligations,
    deductions, and de minimis status.
    """

    PHASE_NAME = "entity_calculation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute per-entity CBAM calculation.

        Args:
            context: Workflow context with entity data.

        Returns:
            PhaseResult with per-entity calculation results.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            entities = config.get("entities", [])
            ets_price = Decimal(str(config.get("ets_price_eur", 80)))
            data_output = context.get_phase_output("data_collection")
            entity_statuses = data_output.get("entity_statuses", [])

            entity_results: List[Dict[str, Any]] = []
            total_obligation = Decimal("0")
            total_emissions = Decimal("0")

            for entity in entities:
                entity_id = entity.get("entity_id", "")
                entity_name = entity.get("entity_name", entity_id)

                # Check if entity data was marked complete
                status_entry = next(
                    (s for s in entity_statuses if s.get("entity_id") == entity_id),
                    {}
                )
                if not status_entry.get("data_complete", False):
                    warnings.append(
                        f"Calculating entity '{entity_name}' with "
                        f"incomplete data"
                    )

                # Run calculation engine (deterministic, zero-hallucination)
                calc_result = await self._calculate_entity(entity, ets_price)
                entity_results.append(calc_result)

                total_emissions += Decimal(str(
                    calc_result.get("total_embedded_emissions_tco2e", 0)
                ))
                total_obligation += Decimal(str(
                    calc_result.get("net_obligation", 0)
                ))

            outputs["entity_results"] = entity_results
            outputs["total_embedded_emissions_tco2e"] = float(total_emissions)
            outputs["total_net_obligation_tco2e"] = float(total_obligation)
            outputs["entities_calculated"] = len(entity_results)

            status = PhaseStatus.COMPLETED
            records = len(entities)

        except Exception as exc:
            logger.error("EntityCalculation failed: %s", exc, exc_info=True)
            errors.append(f"Entity calculation failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    async def _calculate_entity(
        self,
        entity: Dict[str, Any],
        ets_price: Decimal,
    ) -> Dict[str, Any]:
        """
        Run CBAM calculation for a single entity.

        Computes embedded emissions, applies deductions for free
        allocation and carbon price paid, checks de minimis, and
        estimates certificate cost.
        """
        imports = entity.get("import_records", [])
        total_embedded = Decimal("0")
        direct = Decimal("0")
        indirect = Decimal("0")
        by_category: Dict[str, Decimal] = {}

        for record in imports:
            embedded = Decimal(str(
                record.get("embedded_emissions_tco2e", 0)
            ))
            total_embedded += embedded
            direct += Decimal(str(record.get("direct_emissions_tco2e", 0)))
            indirect += Decimal(str(record.get("indirect_emissions_tco2e", 0)))
            category = record.get("goods_category", "other")
            by_category[category] = by_category.get(
                category, Decimal("0")
            ) + embedded

        # Default emission estimate if no records
        if total_embedded == 0 and entity.get("total_imports_tonnes", 0) > 0:
            tonnes = Decimal(str(entity.get("total_imports_tonnes", 0)))
            # Default emission factor placeholder: 2.0 tCO2e/tonne
            total_embedded = tonnes * Decimal("2.0")
            direct = total_embedded

        # Deductions
        free_alloc = Decimal("0")
        carbon_price = Decimal("0")
        for record in imports:
            free_alloc += Decimal(str(
                record.get("free_allocation_deduction_tco2e", 0)
            ))
            carbon_price += Decimal(str(
                record.get("carbon_price_deduction_tco2e", 0)
            ))

        net_obligation = total_embedded - free_alloc - carbon_price
        if net_obligation < 0:
            net_obligation = Decimal("0")

        # De minimis check (per entity aggregate)
        import_value = Decimal(str(entity.get("total_imports_value_eur", 0)))
        import_mass = Decimal(str(entity.get("total_imports_tonnes", 0))) * 1000
        if import_value < DE_MINIMIS_VALUE_EUR and import_mass < DE_MINIMIS_MASS_KG:
            de_minimis = DeMinimisStatus.BELOW_THRESHOLD
        elif import_value < DE_MINIMIS_VALUE_EUR * 2 or import_mass < DE_MINIMIS_MASS_KG * 2:
            de_minimis = DeMinimisStatus.BORDERLINE
        else:
            de_minimis = DeMinimisStatus.ABOVE_THRESHOLD

        cost = (net_obligation * ets_price).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return {
            "entity_id": entity.get("entity_id", ""),
            "entity_name": entity.get("entity_name", ""),
            "total_embedded_emissions_tco2e": float(total_embedded),
            "direct_emissions_tco2e": float(direct),
            "indirect_emissions_tco2e": float(indirect),
            "certificate_obligation": float(total_embedded),
            "free_allocation_deduction": float(free_alloc),
            "carbon_price_deduction": float(carbon_price),
            "net_obligation": float(net_obligation),
            "de_minimis_status": de_minimis.value,
            "estimated_cost_eur": float(cost),
            "by_goods_category": {
                k: float(v) for k, v in by_category.items()
            },
        }


class GroupAggregationPhase:
    """
    Phase 3: Group Aggregation.

    Consolidates all entity obligations, applies group-level de minimis
    check, performs cross-entity certificate netting (for reporting
    purposes), and identifies consolidation adjustments.
    """

    PHASE_NAME = "group_aggregation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute group aggregation phase.

        Args:
            context: Workflow context with per-entity calculations.

        Returns:
            PhaseResult with consolidated group totals.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            calc_output = context.get_phase_output("entity_calculation")
            entity_results = calc_output.get("entity_results", [])
            enable_netting = config.get("enable_cross_entity_netting", False)

            # Aggregate totals
            group_emissions = Decimal("0")
            group_obligation = Decimal("0")
            group_free_alloc = Decimal("0")
            group_carbon_price = Decimal("0")
            group_cost = Decimal("0")
            entities_above = 0
            entities_below = 0
            by_member_state: Dict[str, Decimal] = {}
            by_category: Dict[str, Decimal] = {}

            entities = config.get("entities", [])
            entity_map = {e.get("entity_id"): e for e in entities}

            for result in entity_results:
                emissions = Decimal(str(
                    result.get("total_embedded_emissions_tco2e", 0)
                ))
                obligation = Decimal(str(result.get("net_obligation", 0)))
                free_alloc = Decimal(str(
                    result.get("free_allocation_deduction", 0)
                ))
                carbon_price = Decimal(str(
                    result.get("carbon_price_deduction", 0)
                ))
                cost = Decimal(str(result.get("estimated_cost_eur", 0)))
                de_minimis = result.get("de_minimis_status", "")

                group_emissions += emissions
                group_obligation += obligation
                group_free_alloc += free_alloc
                group_carbon_price += carbon_price
                group_cost += cost

                if de_minimis == DeMinimisStatus.BELOW_THRESHOLD.value:
                    entities_below += 1
                else:
                    entities_above += 1

                # Aggregate by member state
                entity_id = result.get("entity_id", "")
                entity_data = entity_map.get(entity_id, {})
                ms = entity_data.get("member_state", "unknown")
                by_member_state[ms] = by_member_state.get(
                    ms, Decimal("0")
                ) + obligation

                # Aggregate by goods category
                for cat, val in result.get("by_goods_category", {}).items():
                    by_category[cat] = by_category.get(
                        cat, Decimal("0")
                    ) + Decimal(str(val))

            outputs["group_total_emissions_tco2e"] = float(group_emissions)
            outputs["group_net_obligation_tco2e"] = float(group_obligation)
            outputs["group_free_allocation_tco2e"] = float(group_free_alloc)
            outputs["group_carbon_price_deduction_tco2e"] = float(
                group_carbon_price
            )
            outputs["group_estimated_cost_eur"] = float(group_cost)
            outputs["entities_above_de_minimis"] = entities_above
            outputs["entities_below_de_minimis"] = entities_below

            # Group-level de minimis
            if group_obligation <= 0:
                outputs["group_de_minimis_status"] = "NOT_APPLICABLE"
            else:
                outputs["group_de_minimis_status"] = "ABOVE_THRESHOLD"

            outputs["by_member_state"] = {
                k: float(v) for k, v in by_member_state.items()
            }
            outputs["by_goods_category"] = {
                k: float(v) for k, v in by_category.items()
            }

            # Cross-entity netting analysis (reporting only)
            if enable_netting:
                netting_result = self._analyze_netting(entity_results)
                outputs["netting_analysis"] = netting_result
                warnings.append(
                    "Cross-entity netting is for internal reporting only. "
                    "CBAM certificates are non-transferable per Article 20."
                )

            # Consolidation adjustments
            adjustments = self._identify_adjustments(entity_results)
            outputs["consolidation_adjustments"] = adjustments

            status = PhaseStatus.COMPLETED
            records = len(entity_results)

        except Exception as exc:
            logger.error("GroupAggregation failed: %s", exc, exc_info=True)
            errors.append(f"Group aggregation failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _analyze_netting(
        self, entity_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze potential cross-entity certificate netting."""
        surplus_entities = []
        deficit_entities = []
        for r in entity_results:
            obligation = Decimal(str(r.get("net_obligation", 0)))
            if obligation > 0:
                deficit_entities.append({
                    "entity_id": r.get("entity_id", ""),
                    "deficit_tco2e": float(obligation),
                })
            else:
                surplus_entities.append({
                    "entity_id": r.get("entity_id", ""),
                    "surplus_tco2e": float(abs(obligation)),
                })
        return {
            "surplus_entities": surplus_entities,
            "deficit_entities": deficit_entities,
            "netting_possible": len(surplus_entities) > 0 and len(deficit_entities) > 0,
            "note": "For internal reporting only - certificates non-transferable",
        }

    def _identify_adjustments(
        self, entity_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify consolidation adjustments needed."""
        adjustments = []
        for r in entity_results:
            de_minimis = r.get("de_minimis_status", "")
            if de_minimis == DeMinimisStatus.BELOW_THRESHOLD.value:
                adjustments.append({
                    "entity_id": r.get("entity_id", ""),
                    "type": "de_minimis_exclusion",
                    "description": (
                        f"Entity below de minimis threshold; "
                        f"obligation zeroed"
                    ),
                    "original_obligation": r.get("net_obligation", 0),
                    "adjusted_obligation": 0.0,
                })
        return adjustments


class CostAllocationPhase:
    """
    Phase 4: Cost Allocation.

    Distributes CBAM certificate costs to entities using the configured
    allocation method. Generates allocation justification for audit.
    """

    PHASE_NAME = "cost_allocation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute cost allocation phase.

        Args:
            context: Workflow context with group aggregation outputs.

        Returns:
            PhaseResult with per-entity cost allocations.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            method_str = config.get(
                "cost_allocation_method",
                CostAllocationMethod.VOLUME.value,
            )
            method = CostAllocationMethod(method_str)
            entities = config.get("entities", [])
            custom_weights = config.get("custom_allocation_weights", {})
            entity_revenues = config.get("entity_revenues", {})

            agg_output = context.get_phase_output("group_aggregation")
            calc_output = context.get_phase_output("entity_calculation")
            entity_results = calc_output.get("entity_results", [])
            group_cost = Decimal(str(
                agg_output.get("group_estimated_cost_eur", 0)
            ))
            group_obligation = Decimal(str(
                agg_output.get("group_net_obligation_tco2e", 0)
            ))

            # Compute weights per method
            weights = self._compute_weights(
                method, entities, entity_results,
                custom_weights, entity_revenues,
            )
            outputs["allocation_method"] = method.value
            outputs["weights"] = weights

            # Allocate costs
            allocations: List[Dict[str, Any]] = []
            total_allocated = Decimal("0")

            for entity_id, weight in weights.items():
                entity_data = next(
                    (e for e in entities if e.get("entity_id") == entity_id),
                    {}
                )
                weight_dec = Decimal(str(weight))
                allocated_cost = (group_cost * weight_dec).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                allocated_certs = (group_obligation * weight_dec).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )
                total_allocated += allocated_cost

                allocation = {
                    "entity_id": entity_id,
                    "entity_name": entity_data.get("entity_name", entity_id),
                    "allocation_weight": float(weight_dec),
                    "allocated_cost_eur": float(allocated_cost),
                    "allocated_certificates": float(allocated_certs),
                    "allocation_method": method.value,
                    "justification": self._build_justification(
                        method, entity_id, weight_dec, entity_data,
                    ),
                }
                allocations.append(allocation)

            # Handle rounding difference
            rounding_diff = group_cost - total_allocated
            if abs(rounding_diff) > Decimal("0.01") and allocations:
                allocations[-1]["allocated_cost_eur"] = float(
                    Decimal(str(allocations[-1]["allocated_cost_eur"])) + rounding_diff
                )
                allocations[-1]["rounding_adjustment_eur"] = float(
                    rounding_diff
                )

            outputs["allocations"] = allocations
            outputs["total_allocated_cost_eur"] = float(
                group_cost
            )
            outputs["allocation_count"] = len(allocations)

            status = PhaseStatus.COMPLETED
            records = len(allocations)

        except Exception as exc:
            logger.error("CostAllocation failed: %s", exc, exc_info=True)
            errors.append(f"Cost allocation failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _compute_weights(
        self,
        method: CostAllocationMethod,
        entities: List[Dict[str, Any]],
        entity_results: List[Dict[str, Any]],
        custom_weights: Dict[str, float],
        entity_revenues: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute allocation weights based on method."""
        entity_ids = [e.get("entity_id", "") for e in entities]

        if method == CostAllocationMethod.VOLUME:
            return self._weights_by_volume(entities)
        elif method == CostAllocationMethod.REVENUE:
            return self._weights_by_revenue(entity_ids, entity_revenues)
        elif method == CostAllocationMethod.PROFIT_CENTER:
            return self._weights_by_obligation(entity_ids, entity_results)
        elif method == CostAllocationMethod.EQUAL:
            return self._weights_equal(entity_ids)
        elif method == CostAllocationMethod.CUSTOM:
            return self._weights_custom(entity_ids, custom_weights)
        else:
            return self._weights_equal(entity_ids)

    def _weights_by_volume(
        self, entities: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Weight by imported volume (tonnes)."""
        total = sum(
            Decimal(str(e.get("total_imports_tonnes", 0))) for e in entities
        )
        if total <= 0:
            return self._weights_equal(
                [e.get("entity_id", "") for e in entities]
            )
        return {
            e.get("entity_id", ""): float(
                Decimal(str(e.get("total_imports_tonnes", 0))) / total
            )
            for e in entities
        }

    def _weights_by_revenue(
        self, entity_ids: List[str], revenues: Dict[str, float]
    ) -> Dict[str, float]:
        """Weight by entity revenue share."""
        total = sum(Decimal(str(revenues.get(eid, 0))) for eid in entity_ids)
        if total <= 0:
            return self._weights_equal(entity_ids)
        return {
            eid: float(
                Decimal(str(revenues.get(eid, 0))) / total
            )
            for eid in entity_ids
        }

    def _weights_by_obligation(
        self,
        entity_ids: List[str],
        entity_results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Weight by individual net obligation (profit center proxy)."""
        result_map = {
            r.get("entity_id", ""): r for r in entity_results
        }
        total = sum(
            Decimal(str(result_map.get(eid, {}).get("net_obligation", 0)))
            for eid in entity_ids
        )
        if total <= 0:
            return self._weights_equal(entity_ids)
        return {
            eid: float(
                Decimal(str(
                    result_map.get(eid, {}).get("net_obligation", 0)
                )) / total
            )
            for eid in entity_ids
        }

    def _weights_equal(self, entity_ids: List[str]) -> Dict[str, float]:
        """Equal allocation across entities."""
        count = max(len(entity_ids), 1)
        weight = 1.0 / count
        return {eid: weight for eid in entity_ids}

    def _weights_custom(
        self, entity_ids: List[str], custom: Dict[str, float]
    ) -> Dict[str, float]:
        """Custom weights provided by user."""
        total = sum(Decimal(str(custom.get(eid, 0))) for eid in entity_ids)
        if total <= 0:
            return self._weights_equal(entity_ids)
        return {
            eid: float(
                Decimal(str(custom.get(eid, 0))) / total
            )
            for eid in entity_ids
        }

    def _build_justification(
        self,
        method: CostAllocationMethod,
        entity_id: str,
        weight: Decimal,
        entity_data: Dict[str, Any],
    ) -> str:
        """Build audit-ready allocation justification."""
        pct = (weight * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        if method == CostAllocationMethod.VOLUME:
            tonnes = entity_data.get("total_imports_tonnes", 0)
            return (
                f"Allocated {pct}% based on import volume "
                f"({tonnes:.2f} tonnes)"
            )
        elif method == CostAllocationMethod.REVENUE:
            return f"Allocated {pct}% based on entity revenue share"
        elif method == CostAllocationMethod.PROFIT_CENTER:
            return (
                f"Allocated {pct}% based on individual CBAM "
                f"obligation (profit center)"
            )
        elif method == CostAllocationMethod.EQUAL:
            return f"Allocated {pct}% using equal distribution method"
        elif method == CostAllocationMethod.CUSTOM:
            return f"Allocated {pct}% using custom-defined weights"
        return f"Allocated {pct}%"


class ConsolidationReportingPhase:
    """
    Phase 5: Reporting.

    Generates group consolidation report plus per-entity declaration
    summaries, cost allocation statements, and entity contact
    notifications.
    """

    PHASE_NAME = "reporting"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute reporting phase.

        Args:
            context: Workflow context with all prior phase outputs.

        Returns:
            PhaseResult with reports and notifications.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            org_id = config.get("organization_id", "")
            group_name = config.get("group_name", org_id)
            year = config.get("reporting_year", 0)
            quarter = config.get("reporting_quarter", 0)

            agg_output = context.get_phase_output("group_aggregation")
            calc_output = context.get_phase_output("entity_calculation")
            alloc_output = context.get_phase_output("cost_allocation")

            # Group consolidation report
            consolidation_report = {
                "report_id": str(uuid.uuid4()),
                "report_type": "group_cbam_consolidation",
                "group_name": group_name,
                "organization_id": org_id,
                "reporting_year": year,
                "reporting_quarter": quarter,
                "generated_at": datetime.utcnow().isoformat(),
                "group_totals": {
                    "total_emissions_tco2e": agg_output.get(
                        "group_total_emissions_tco2e", 0
                    ),
                    "net_obligation_tco2e": agg_output.get(
                        "group_net_obligation_tco2e", 0
                    ),
                    "estimated_cost_eur": agg_output.get(
                        "group_estimated_cost_eur", 0
                    ),
                    "entities_count": len(
                        calc_output.get("entity_results", [])
                    ),
                    "entities_above_de_minimis": agg_output.get(
                        "entities_above_de_minimis", 0
                    ),
                    "entities_below_de_minimis": agg_output.get(
                        "entities_below_de_minimis", 0
                    ),
                },
                "by_member_state": agg_output.get("by_member_state", {}),
                "by_goods_category": agg_output.get("by_goods_category", {}),
                "cost_allocation": {
                    "method": alloc_output.get("allocation_method", ""),
                    "allocations": alloc_output.get("allocations", []),
                },
                "adjustments": agg_output.get(
                    "consolidation_adjustments", []
                ),
            }
            outputs["consolidation_report"] = consolidation_report
            outputs["consolidation_report_id"] = consolidation_report["report_id"]

            # Per-entity declaration summaries
            entity_summaries: List[Dict[str, Any]] = []
            entity_results = calc_output.get("entity_results", [])
            allocations = alloc_output.get("allocations", [])
            alloc_map = {a.get("entity_id"): a for a in allocations}

            for er in entity_results:
                entity_id = er.get("entity_id", "")
                alloc = alloc_map.get(entity_id, {})
                summary = {
                    "entity_id": entity_id,
                    "entity_name": er.get("entity_name", ""),
                    "emissions_tco2e": er.get(
                        "total_embedded_emissions_tco2e", 0
                    ),
                    "net_obligation_tco2e": er.get("net_obligation", 0),
                    "de_minimis_status": er.get("de_minimis_status", ""),
                    "allocated_cost_eur": alloc.get("allocated_cost_eur", 0),
                    "allocated_certificates": alloc.get(
                        "allocated_certificates", 0
                    ),
                    "allocation_justification": alloc.get(
                        "justification", ""
                    ),
                }
                entity_summaries.append(summary)

            outputs["entity_summaries"] = entity_summaries

            # Entity notifications
            entities = config.get("entities", [])
            notifications: List[Dict[str, Any]] = []
            for entity in entities:
                entity_id = entity.get("entity_id", "")
                contact = entity.get("contact_email", "")
                if contact:
                    notification = {
                        "notification_id": str(uuid.uuid4()),
                        "entity_id": entity_id,
                        "entity_name": entity.get("entity_name", entity_id),
                        "recipient": contact,
                        "type": "consolidation_summary",
                        "subject": (
                            f"CBAM Q{quarter}/{year} Consolidation - "
                            f"{entity.get('entity_name', entity_id)}"
                        ),
                        "sent_at": datetime.utcnow().isoformat(),
                    }
                    notifications.append(notification)

            outputs["entity_notifications"] = notifications
            outputs["notifications_sent"] = len(notifications)

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("ConsolidationReporting failed: %s", exc, exc_info=True)
            errors.append(f"Consolidation reporting failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================


class MultiEntityConsolidationWorkflow:
    """
    Five-phase quarterly multi-entity consolidation workflow.

    Orchestrates CBAM obligation consolidation across a corporate group
    with multiple entities operating in different EU member states.
    Supports five cost allocation methods and checkpoint/resume.

    Attributes:
        workflow_id: Unique execution identifier.
        _phases: Ordered phase executors.
        _progress_callback: Optional progress notification callback.

    Example:
        >>> wf = MultiEntityConsolidationWorkflow()
        >>> input_data = MultiEntityConsolidationInput(
        ...     organization_id="group-123",
        ...     reporting_year=2026,
        ...     reporting_quarter=1,
        ...     entities=[entity1, entity2],
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "multi_entity_consolidation"

    PHASE_ORDER = [
        "data_collection",
        "entity_calculation",
        "group_aggregation",
        "cost_allocation",
        "reporting",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize multi-entity consolidation workflow.

        Args:
            progress_callback: Optional callback(phase, message, pct).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "data_collection": DataCollectionPhase(),
            "entity_calculation": EntityCalculationPhase(),
            "group_aggregation": GroupAggregationPhase(),
            "cost_allocation": CostAllocationPhase(),
            "reporting": ConsolidationReportingPhase(),
        }

    async def run(
        self, input_data: MultiEntityConsolidationInput
    ) -> MultiEntityConsolidationResult:
        """
        Execute the 5-phase multi-entity consolidation workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            MultiEntityConsolidationResult with per-entity and group data.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting multi-entity consolidation %s for org=%s "
            "year=%d Q%d entities=%d",
            self.workflow_id, input_data.organization_id,
            input_data.reporting_year, input_data.reporting_quarter,
            len(input_data.entities),
        )

        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=self._build_config(input_data),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            if context.is_phase_completed(phase_name):
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(phase_name, f"Starting: {phase_name}", pct)
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_result = await self._phases[phase_name].execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, phase_result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, phase_result.status)
                    if phase_name in ("data_collection", "entity_calculation"):
                        overall_status = WorkflowStatus.FAILED
                        break

                context.errors.extend(phase_result.errors)
                context.warnings.extend(phase_result.warnings)

            except Exception as exc:
                logger.error(
                    "Phase '%s' raised: %s", phase_name, exc, exc_info=True
                )
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = (
                WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL
            )

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context, input_data)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress(
            "workflow", f"Workflow {overall_status.value}", 1.0
        )
        logger.info(
            "Multi-entity consolidation %s finished status=%s in %.1fs",
            self.workflow_id, overall_status.value, total_duration,
        )

        return MultiEntityConsolidationResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            entity_count=len(input_data.entities),
            group_total_obligation_tco2e=summary.get(
                "group_net_obligation_tco2e", 0.0
            ),
            group_total_cost_eur=summary.get(
                "group_estimated_cost_eur", 0.0
            ),
            entities_above_de_minimis=summary.get(
                "entities_above_de_minimis", 0
            ),
            entities_below_de_minimis=summary.get(
                "entities_below_de_minimis", 0
            ),
            cost_allocation_method=input_data.cost_allocation_method.value,
            entity_summaries=summary.get("entity_summaries", []),
        )

    def _build_config(
        self, input_data: MultiEntityConsolidationInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        return {
            "organization_id": input_data.organization_id,
            "group_name": input_data.group_name,
            "reporting_year": input_data.reporting_year,
            "reporting_quarter": input_data.reporting_quarter,
            "entities": [e.model_dump() for e in input_data.entities],
            "cost_allocation_method": input_data.cost_allocation_method.value,
            "custom_allocation_weights": input_data.custom_allocation_weights,
            "entity_revenues": input_data.entity_revenues,
            "ets_price_eur": input_data.ets_price_eur,
            "enable_cross_entity_netting": input_data.enable_cross_entity_netting,
        }

    def _build_summary(
        self,
        context: WorkflowContext,
        input_data: MultiEntityConsolidationInput,
    ) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        agg = context.get_phase_output("group_aggregation")
        alloc = context.get_phase_output("cost_allocation")
        reporting = context.get_phase_output("reporting")
        return {
            "group_net_obligation_tco2e": agg.get(
                "group_net_obligation_tco2e", 0.0
            ),
            "group_estimated_cost_eur": agg.get(
                "group_estimated_cost_eur", 0.0
            ),
            "entities_above_de_minimis": agg.get(
                "entities_above_de_minimis", 0
            ),
            "entities_below_de_minimis": agg.get(
                "entities_below_de_minimis", 0
            ),
            "allocation_method": alloc.get("allocation_method", ""),
            "entity_summaries": reporting.get("entity_summaries", []),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)


# =============================================================================
# UTILITIES
# =============================================================================


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    serialized = str(data).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
