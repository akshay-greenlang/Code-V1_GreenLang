# -*- coding: utf-8 -*-
"""
MultiEntityOrchestrator - 100+ Entity Hierarchy Management for PACK-027
============================================================================

Enterprise orchestrator for managing multi-entity GHG consolidation
across 100+ subsidiaries, joint ventures, associates, and SPVs.
Supports GHG Protocol consolidation approaches (financial control,
operational control, equity share) with intercompany elimination
and base year recalculation on structural changes.

Features:
    - 3 consolidation approaches per GHG Protocol
    - Intercompany transaction elimination
    - Base year recalculation (5% significance threshold)
    - M&A impact assessment (acquisitions/divestitures)
    - Entity-level and group-level reporting
    - Ownership % tracking with effective dates
    - SHA-256 provenance tracking

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-027 Enterprise Net Zero Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConsolidationApproach(str, Enum):
    FINANCIAL_CONTROL = "financial_control"
    OPERATIONAL_CONTROL = "operational_control"
    EQUITY_SHARE = "equity_share"

class EntityType(str, Enum):
    PARENT = "parent"
    WHOLLY_OWNED = "wholly_owned"
    MAJORITY_OWNED = "majority_owned"
    JOINT_VENTURE = "joint_venture"
    ASSOCIATE = "associate"
    SPV = "spv"
    FRANCHISE = "franchise"

class StructuralChangeType(str, Enum):
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    ORGANIC_GROWTH = "organic_growth"
    OUTSOURCING = "outsourcing"
    INSOURCING = "insourcing"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class MultiEntityConfig(BaseModel):
    pack_id: str = Field(default="PACK-027")
    consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL
    )
    base_year: int = Field(default=2023, ge=2015, le=2025)
    significance_threshold_pct: float = Field(default=5.0, ge=1.0, le=10.0)
    max_entities: int = Field(default=500, ge=1, le=5000)
    enable_intercompany_elimination: bool = Field(default=True)
    enable_provenance: bool = Field(default=True)

class EntityDefinition(BaseModel):
    entity_id: str = Field(default_factory=_new_uuid)
    entity_name: str = Field(default="")
    entity_type: EntityType = Field(default=EntityType.WHOLLY_OWNED)
    parent_entity_id: Optional[str] = Field(None)
    country: str = Field(default="")
    region: str = Field(default="")
    ownership_pct: float = Field(default=100.0, ge=0.0, le=100.0)
    has_operational_control: bool = Field(default=True)
    has_financial_control: bool = Field(default=True)
    effective_from: str = Field(default="2023-01-01")
    effective_to: Optional[str] = Field(None)
    sector: str = Field(default="")
    employee_count: int = Field(default=0)
    scope1_tco2e: float = Field(default=0.0)
    scope2_tco2e: float = Field(default=0.0)
    scope3_tco2e: float = Field(default=0.0)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reporting_currency: str = Field(default="USD")

class IntercompanyTransaction(BaseModel):
    transaction_id: str = Field(default_factory=_new_uuid)
    selling_entity_id: str = Field(default="")
    buying_entity_id: str = Field(default="")
    transaction_type: str = Field(default="goods")
    emissions_tco2e: float = Field(default=0.0)
    amount: float = Field(default=0.0)
    currency: str = Field(default="USD")
    period: str = Field(default="")

class StructuralChange(BaseModel):
    change_id: str = Field(default_factory=_new_uuid)
    change_type: StructuralChangeType = Field(...)
    entity_name: str = Field(default="")
    effective_date: str = Field(default="")
    emissions_impact_tco2e: float = Field(default=0.0)
    impact_pct_of_base_year: float = Field(default=0.0)
    requires_recalculation: bool = Field(default=False)

class ConsolidationResult(BaseModel):
    result_id: str = Field(default_factory=_new_uuid)
    consolidation_approach: str = Field(default="")
    reporting_year: int = Field(default=2025)
    total_entities: int = Field(default=0)
    entities_in_boundary: int = Field(default=0)
    entities_excluded: int = Field(default=0)
    group_scope1_tco2e: float = Field(default=0.0)
    group_scope2_tco2e: float = Field(default=0.0)
    group_scope3_tco2e: float = Field(default=0.0)
    group_total_tco2e: float = Field(default=0.0)
    intercompany_eliminations_tco2e: float = Field(default=0.0)
    by_entity: List[Dict[str, Any]] = Field(default_factory=list)
    by_region: Dict[str, float] = Field(default_factory=dict)
    by_sector: Dict[str, float] = Field(default_factory=dict)
    base_year_recalculations: List[Dict[str, Any]] = Field(default_factory=list)
    data_quality_score: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# MultiEntityOrchestrator
# ---------------------------------------------------------------------------

class MultiEntityOrchestrator:
    """Manage 100+ entity hierarchy for GHG consolidation.

    Handles organizational boundary definition, consolidation
    under financial/operational/equity approaches, intercompany
    elimination, and base year recalculation for structural changes.

    Example:
        >>> config = MultiEntityConfig(
        ...     consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
        ... )
        >>> orch = MultiEntityOrchestrator(config)
        >>> orch.add_entity(EntityDefinition(entity_name="UK Subsidiary", ...))
        >>> result = orch.consolidate(2025)
    """

    def __init__(self, config: Optional[MultiEntityConfig] = None) -> None:
        self.config = config or MultiEntityConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._entities: Dict[str, EntityDefinition] = {}
        self._intercompany: List[IntercompanyTransaction] = []
        self._structural_changes: List[StructuralChange] = []
        self.logger.info(
            "MultiEntityOrchestrator initialized: approach=%s, max=%d",
            self.config.consolidation_approach.value, self.config.max_entities,
        )

    def add_entity(self, entity: EntityDefinition) -> str:
        """Add an entity to the hierarchy."""
        if len(self._entities) >= self.config.max_entities:
            raise ValueError(f"Maximum entity limit ({self.config.max_entities}) reached")
        self._entities[entity.entity_id] = entity
        self.logger.info("Entity added: %s (%s)", entity.entity_name, entity.entity_type.value)
        return entity.entity_id

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity from the hierarchy."""
        if entity_id in self._entities:
            del self._entities[entity_id]
            return True
        return False

    def add_intercompany_transaction(self, txn: IntercompanyTransaction) -> str:
        """Register an intercompany transaction for elimination."""
        self._intercompany.append(txn)
        return txn.transaction_id

    def register_structural_change(self, change: StructuralChange) -> str:
        """Register a structural change (M&A) for base year recalculation."""
        if change.impact_pct_of_base_year >= self.config.significance_threshold_pct:
            change.requires_recalculation = True
        self._structural_changes.append(change)
        self.logger.info(
            "Structural change registered: %s (%s), impact=%.1f%%, recalc=%s",
            change.entity_name, change.change_type.value,
            change.impact_pct_of_base_year, change.requires_recalculation,
        )
        return change.change_id

    def consolidate(self, reporting_year: int) -> ConsolidationResult:
        """Consolidate emissions across all entities."""
        start = time.monotonic()

        result = ConsolidationResult(
            consolidation_approach=self.config.consolidation_approach.value,
            reporting_year=reporting_year,
            total_entities=len(self._entities),
        )

        in_boundary = 0
        excluded = 0
        scope1 = scope2 = scope3 = 0.0
        by_region: Dict[str, float] = {}
        by_sector: Dict[str, float] = {}
        entity_details: List[Dict[str, Any]] = []

        for entity in self._entities.values():
            included, share = self._apply_consolidation(entity)
            if not included:
                excluded += 1
                continue

            in_boundary += 1
            e_scope1 = entity.scope1_tco2e * share
            e_scope2 = entity.scope2_tco2e * share
            e_scope3 = entity.scope3_tco2e * share
            e_total = e_scope1 + e_scope2 + e_scope3

            scope1 += e_scope1
            scope2 += e_scope2
            scope3 += e_scope3

            region = entity.region or entity.country or "Unknown"
            by_region[region] = by_region.get(region, 0.0) + e_total

            sector = entity.sector or "General"
            by_sector[sector] = by_sector.get(sector, 0.0) + e_total

            entity_details.append({
                "entity_id": entity.entity_id,
                "entity_name": entity.entity_name,
                "entity_type": entity.entity_type.value,
                "ownership_pct": entity.ownership_pct,
                "consolidation_share": share,
                "scope1_tco2e": round(e_scope1, 2),
                "scope2_tco2e": round(e_scope2, 2),
                "scope3_tco2e": round(e_scope3, 2),
                "total_tco2e": round(e_total, 2),
            })

        # Intercompany elimination
        elimination = 0.0
        if self.config.enable_intercompany_elimination:
            elimination = sum(txn.emissions_tco2e for txn in self._intercompany)
            scope3 -= elimination

        # Base year recalculations
        recalculations = [
            {
                "change_id": c.change_id,
                "change_type": c.change_type.value,
                "entity_name": c.entity_name,
                "impact_pct": c.impact_pct_of_base_year,
                "requires_recalculation": c.requires_recalculation,
            }
            for c in self._structural_changes if c.requires_recalculation
        ]

        total = scope1 + scope2 + scope3
        dq_scores = [e.data_quality_score for e in self._entities.values() if e.data_quality_score > 0]
        avg_dq = sum(dq_scores) / len(dq_scores) if dq_scores else 0.0

        result.entities_in_boundary = in_boundary
        result.entities_excluded = excluded
        result.group_scope1_tco2e = round(scope1, 2)
        result.group_scope2_tco2e = round(scope2, 2)
        result.group_scope3_tco2e = round(scope3, 2)
        result.group_total_tco2e = round(total, 2)
        result.intercompany_eliminations_tco2e = round(elimination, 2)
        result.by_entity = entity_details
        result.by_region = {k: round(v, 2) for k, v in by_region.items()}
        result.by_sector = {k: round(v, 2) for k, v in by_sector.items()}
        result.base_year_recalculations = recalculations
        result.data_quality_score = round(avg_dq, 3)

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Consolidation complete: %d/%d entities, S1=%.2f S2=%.2f S3=%.2f, "
            "total=%.2f, eliminations=%.2f",
            in_boundary, len(self._entities),
            scope1, scope2, scope3, total, elimination,
        )
        return result

    def compare_approaches(self, reporting_year: int) -> Dict[str, ConsolidationResult]:
        """Compare emissions under all 3 consolidation approaches."""
        results: Dict[str, ConsolidationResult] = {}
        original = self.config.consolidation_approach

        for approach in ConsolidationApproach:
            self.config.consolidation_approach = approach
            results[approach.value] = self.consolidate(reporting_year)

        self.config.consolidation_approach = original
        return results

    def get_entity_tree(self) -> Dict[str, Any]:
        """Get the entity hierarchy as a tree structure."""
        roots = [
            e for e in self._entities.values()
            if e.parent_entity_id is None or e.entity_type == EntityType.PARENT
        ]
        return {
            "total_entities": len(self._entities),
            "roots": [
                {"entity_id": r.entity_id, "name": r.entity_name, "type": r.entity_type.value}
                for r in roots
            ],
            "consolidation_approach": self.config.consolidation_approach.value,
        }

    def get_orchestrator_status(self) -> Dict[str, Any]:
        return {
            "pack_id": self.config.pack_id,
            "total_entities": len(self._entities),
            "max_entities": self.config.max_entities,
            "consolidation_approach": self.config.consolidation_approach.value,
            "intercompany_transactions": len(self._intercompany),
            "structural_changes": len(self._structural_changes),
            "base_year": self.config.base_year,
        }

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _apply_consolidation(self, entity: EntityDefinition) -> tuple:
        """Determine if entity is in boundary and its consolidation share."""
        approach = self.config.consolidation_approach

        if approach == ConsolidationApproach.FINANCIAL_CONTROL:
            if entity.has_financial_control:
                return True, 1.0
            return False, 0.0

        elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
            if entity.has_operational_control:
                return True, 1.0
            return False, 0.0

        elif approach == ConsolidationApproach.EQUITY_SHARE:
            if entity.ownership_pct > 0:
                return True, entity.ownership_pct / 100.0
            return False, 0.0

        return False, 0.0
