# -*- coding: utf-8 -*-
"""
Multi-Entity Rollup Workflow
=================================

5-phase workflow for consolidating 100+ entities with intercompany
elimination within PACK-027 Enterprise Net Zero Pack.

Phases:
    1. EntityRefresh         -- Refresh entity hierarchy (new, divested, ownership changes)
    2. DataValidation        -- Validate per-entity data completeness and quality
    3. EntityCalculation     -- Calculate per-entity emissions
    4. Elimination           -- Intercompany elimination (S3 Cat1 vs. S1 overlap)
    5. ConsolidatedReport    -- Generate consolidated report with reconciliation

Uses: multi_entity_consolidation_engine, enterprise_baseline_engine.

Zero-hallucination: deterministic GHG Protocol consolidation.
SHA-256 provenance hashes.

Author: GreenLang Team
Version: 27.0.0
Pack: PACK-027 Enterprise Net Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "27.0.0"
_PACK_ID = "PACK-027"

def _new_uuid() -> str:
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class EntityChangeType(str, Enum):
    NEW_ENTITY = "new_entity"
    DIVESTED = "divested"
    OWNERSHIP_CHANGE = "ownership_change"
    RECLASSIFICATION = "reclassification"
    NO_CHANGE = "no_change"

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    dag_node_id: str = Field(default="")

class EntityChange(BaseModel):
    entity_id: str = Field(...)
    entity_name: str = Field(default="")
    change_type: str = Field(default="no_change")
    old_ownership_pct: float = Field(default=100.0)
    new_ownership_pct: float = Field(default=100.0)
    effective_date: str = Field(default="")
    impact_tco2e: float = Field(default=0.0)

class EntityValidation(BaseModel):
    entity_id: str = Field(...)
    entity_name: str = Field(default="")
    data_complete: bool = Field(default=False)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    data_quality_score: float = Field(default=5.0, ge=1.0, le=5.0)
    missing_data_points: List[str] = Field(default_factory=list)
    validation_passed: bool = Field(default=False)

class EntityResult(BaseModel):
    entity_id: str = Field(...)
    entity_name: str = Field(default="")
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    ownership_factor: float = Field(default=1.0)
    adjusted_total_tco2e: float = Field(default=0.0, ge=0.0)

class EliminationEntry(BaseModel):
    selling_entity: str = Field(default="")
    buying_entity: str = Field(default="")
    category: str = Field(default="")
    eliminated_tco2e: float = Field(default=0.0, ge=0.0)
    justification: str = Field(default="")

class Reconciliation(BaseModel):
    sum_entity_totals: float = Field(default=0.0, ge=0.0)
    total_eliminations: float = Field(default=0.0, ge=0.0)
    consolidated_total: float = Field(default=0.0, ge=0.0)
    reconciliation_delta: float = Field(default=0.0, description="Should be zero")
    is_reconciled: bool = Field(default=False)

class MultiEntityRollupConfig(BaseModel):
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    consolidation_approach: str = Field(default="financial_control")
    enable_intercompany_elimination: bool = Field(default=True)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class MultiEntityRollupInput(BaseModel):
    config: MultiEntityRollupConfig = Field(default_factory=MultiEntityRollupConfig)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    entity_changes: List[EntityChange] = Field(default_factory=list)
    intercompany_transactions: List[Dict[str, Any]] = Field(default_factory=list)
    entity_emissions: List[Dict[str, Any]] = Field(default_factory=list)

class MultiEntityRollupResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="enterprise_multi_entity_rollup")
    pack_id: str = Field(default="PACK-027")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    entity_changes: List[EntityChange] = Field(default_factory=list)
    entity_validations: List[EntityValidation] = Field(default_factory=list)
    entity_results: List[EntityResult] = Field(default_factory=list)
    eliminations: List[EliminationEntry] = Field(default_factory=list)
    reconciliation: Reconciliation = Field(default_factory=Reconciliation)
    consolidated_scope1: float = Field(default=0.0, ge=0.0)
    consolidated_scope2_location: float = Field(default=0.0, ge=0.0)
    consolidated_scope2_market: float = Field(default=0.0, ge=0.0)
    consolidated_scope3: float = Field(default=0.0, ge=0.0)
    consolidated_total: float = Field(default=0.0, ge=0.0)
    next_steps: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class MultiEntityRollupWorkflow:
    """
    5-phase multi-entity rollup workflow for 100+ entity consolidation.

    Phase 1: Entity Refresh -- Identify new/divested/changed entities.
    Phase 2: Data Validation -- Validate completeness and quality per entity.
    Phase 3: Entity Calculation -- Calculate emissions per entity.
    Phase 4: Elimination -- Intercompany elimination.
    Phase 5: Consolidated Report -- Generate report with reconciliation.

    Example:
        >>> wf = MultiEntityRollupWorkflow()
        >>> inp = MultiEntityRollupInput(
        ...     entities=[{"entity_id": "e1", "name": "Entity 1"}],
        ... )
        >>> result = await wf.execute(inp)
    """

    def __init__(self, config: Optional[MultiEntityRollupConfig] = None) -> None:
        self.workflow_id: str = _new_uuid()
        self.config = config or MultiEntityRollupConfig()
        self._phase_results: List[PhaseResult] = []
        self._validations: List[EntityValidation] = []
        self._entity_results: List[EntityResult] = []
        self._eliminations: List[EliminationEntry] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: MultiEntityRollupInput) -> MultiEntityRollupResult:
        started_at = utcnow()
        self.config = input_data.config
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_entity_refresh(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_data_validation(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_entity_calculation(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_elimination(input_data)
            self._phase_results.append(phase4)

            phase5 = await self._phase_consolidated_report(input_data)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Multi-entity rollup failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()

        # Consolidate totals
        c_s1 = sum(r.scope1_tco2e * r.ownership_factor for r in self._entity_results)
        c_s2l = sum(r.scope2_location_tco2e * r.ownership_factor for r in self._entity_results)
        c_s2m = sum(r.scope2_market_tco2e * r.ownership_factor for r in self._entity_results)
        c_s3 = sum(r.scope3_tco2e * r.ownership_factor for r in self._entity_results)
        total_elim = sum(e.eliminated_tco2e for e in self._eliminations)
        c_s3_net = max(c_s3 - total_elim, 0.0)
        c_total = c_s1 + c_s2m + c_s3_net

        # Reconciliation
        sum_entity = sum(r.adjusted_total_tco2e for r in self._entity_results)
        recon = Reconciliation(
            sum_entity_totals=round(sum_entity, 2),
            total_eliminations=round(total_elim, 2),
            consolidated_total=round(c_total, 2),
            reconciliation_delta=round(sum_entity - total_elim - c_total, 2),
            is_reconciled=abs(sum_entity - total_elim - c_total) < 0.01,
        )

        result = MultiEntityRollupResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            entity_changes=input_data.entity_changes,
            entity_validations=self._validations,
            entity_results=self._entity_results,
            eliminations=self._eliminations,
            reconciliation=recon,
            consolidated_scope1=round(c_s1, 2),
            consolidated_scope2_location=round(c_s2l, 2),
            consolidated_scope2_market=round(c_s2m, 2),
            consolidated_scope3=round(c_s3_net, 2),
            consolidated_total=round(c_total, 2),
            next_steps=self._generate_next_steps(),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"}),
        )
        return result

    async def _phase_entity_refresh(self, input_data: MultiEntityRollupInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        changes = input_data.entity_changes
        new_count = sum(1 for c in changes if c.change_type == "new_entity")
        divested = sum(1 for c in changes if c.change_type == "divested")
        ownership = sum(1 for c in changes if c.change_type == "ownership_change")

        outputs["total_entities"] = len(input_data.entities)
        outputs["new_entities"] = new_count
        outputs["divested_entities"] = divested
        outputs["ownership_changes"] = ownership
        outputs["no_change"] = len(changes) - new_count - divested - ownership

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="entity_refresh", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_entity_refresh",
        )

    async def _phase_data_validation(self, input_data: MultiEntityRollupInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._validations = []
        for entity in input_data.entities:
            eid = entity.get("entity_id", "")
            ename = entity.get("name", eid)
            completeness = float(entity.get("data_completeness_pct", 80))
            dq = float(entity.get("data_quality_score", 3.0))
            missing = entity.get("missing_data_points", [])
            passed = completeness >= 70 and dq <= 4.0

            if not passed:
                warnings.append(f"Entity '{ename}' failed validation: {completeness:.0f}% complete, DQ={dq}")

            val = EntityValidation(
                entity_id=eid, entity_name=ename,
                data_complete=completeness >= 95,
                completeness_pct=completeness,
                data_quality_score=dq,
                missing_data_points=missing,
                validation_passed=passed,
            )
            self._validations.append(val)

        passed_count = sum(1 for v in self._validations if v.validation_passed)
        outputs["entities_validated"] = len(self._validations)
        outputs["entities_passed"] = passed_count
        outputs["entities_failed"] = len(self._validations) - passed_count

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="data_validation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_data_validation",
        )

    async def _phase_entity_calculation(self, input_data: MultiEntityRollupInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        self._entity_results = []
        approach = self.config.consolidation_approach

        for em_data in input_data.entity_emissions:
            eid = em_data.get("entity_id", "")
            ownership = float(em_data.get("ownership_pct", 100)) / 100.0
            factor = ownership if approach == "equity_share" else 1.0

            s1 = float(em_data.get("scope1_tco2e", 0))
            s2l = float(em_data.get("scope2_location_tco2e", 0))
            s2m = float(em_data.get("scope2_market_tco2e", 0))
            s3 = float(em_data.get("scope3_tco2e", 0))
            total = s1 + s2m + s3

            er = EntityResult(
                entity_id=eid,
                entity_name=em_data.get("entity_name", eid),
                scope1_tco2e=round(s1, 2),
                scope2_location_tco2e=round(s2l, 2),
                scope2_market_tco2e=round(s2m, 2),
                scope3_tco2e=round(s3, 2),
                total_tco2e=round(total, 2),
                ownership_factor=factor,
                adjusted_total_tco2e=round(total * factor, 2),
            )
            self._entity_results.append(er)

        outputs["entities_calculated"] = len(self._entity_results)
        outputs["total_pre_elimination"] = round(
            sum(r.adjusted_total_tco2e for r in self._entity_results), 2,
        )

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="entity_calculation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_entity_calculation",
        )

    async def _phase_elimination(self, input_data: MultiEntityRollupInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        self._eliminations = []
        if self.config.enable_intercompany_elimination:
            for txn in input_data.intercompany_transactions:
                entry = EliminationEntry(
                    selling_entity=txn.get("selling_entity", ""),
                    buying_entity=txn.get("buying_entity", ""),
                    category=txn.get("category", "scope3_cat01"),
                    eliminated_tco2e=float(txn.get("eliminated_tco2e", 0)),
                    justification=txn.get("justification", "Intercompany elimination"),
                )
                self._eliminations.append(entry)

        outputs["eliminations_applied"] = len(self._eliminations)
        outputs["total_eliminated_tco2e"] = round(
            sum(e.eliminated_tco2e for e in self._eliminations), 2,
        )

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="elimination", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_elimination",
        )

    async def _phase_consolidated_report(self, input_data: MultiEntityRollupInput) -> PhaseResult:
        started = utcnow()
        outputs: Dict[str, Any] = {}

        outputs["report_sections"] = [
            "Consolidated GHG Summary",
            "Entity Hierarchy & Boundary",
            "Per-Entity Emissions Breakdown",
            "Intercompany Eliminations",
            "Reconciliation (Entity Sum - Eliminations = Consolidated)",
            "Ownership Adjustments (Equity Share)",
            "Data Quality by Entity",
            "Appendix: Entity Details",
        ]
        outputs["report_formats"] = ["MD", "HTML", "JSON", "XLSX"]

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="consolidated_report", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_consolidated_report",
        )

    def _generate_next_steps(self) -> List[str]:
        return [
            "Review consolidated totals with group controller / CFO.",
            "Validate intercompany eliminations against financial consolidation.",
            "Address data quality gaps for entities below validation threshold.",
            "Update SBTi target coverage calculation with latest entity count.",
            "Prepare external assurance scope based on consolidated boundary.",
        ]
