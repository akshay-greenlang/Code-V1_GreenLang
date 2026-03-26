# -*- coding: utf-8 -*-
"""
Boundary Definition Workflow
====================================

5-phase workflow for GHG organisational boundary definition covering entity
mapping, ownership chain resolution, consolidation approach application,
materiality check, and boundary lock within PACK-049 Multi-Site Management.

Phases:
    1. EntityMapping            -- Map legal entities to facilities and sites,
                                   building the entity-facility hierarchy.
    2. OwnershipChain           -- Define ownership percentages and control
                                   relationships through the corporate tree.
    3. ConsolidationApproach    -- Apply the selected consolidation approach
                                   (equity share / financial control /
                                   operational control) per GHG Protocol Ch. 3.
    4. MaterialityCheck         -- Assess materiality of each entity, flag
                                   de minimis exclusions and document rationale.
    5. BoundaryLock             -- Lock the boundary for the reporting period,
                                   generate documentation and provenance hash.

Regulatory Basis:
    GHG Protocol Corporate Standard (Ch. 3) -- Setting Organisational Boundaries
    ISO 14064-1:2018 (Cl. 5.1) -- Organisational boundaries
    CSRD / ESRS 1 (2024) -- Consolidation scope
    SEC Climate Rules (2024) -- Registrant boundary definition

Author: GreenLang Team
Version: 49.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


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


class BoundaryPhase(str, Enum):
    ENTITY_MAPPING = "entity_mapping"
    OWNERSHIP_CHAIN = "ownership_chain"
    CONSOLIDATION_APPROACH = "consolidation_approach"
    MATERIALITY_CHECK = "materiality_check"
    BOUNDARY_LOCK = "boundary_lock"


class ConsolidationApproach(str, Enum):
    EQUITY_SHARE = "equity_share"
    FINANCIAL_CONTROL = "financial_control"
    OPERATIONAL_CONTROL = "operational_control"


class EntityType(str, Enum):
    PARENT = "parent"
    SUBSIDIARY = "subsidiary"
    JOINT_VENTURE = "joint_venture"
    ASSOCIATE = "associate"
    FRANCHISE = "franchise"
    SPECIAL_PURPOSE = "special_purpose"


class ControlType(str, Enum):
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    NONE = "none"


class MaterialityClassification(str, Enum):
    MATERIAL = "material"
    IMMATERIAL = "immaterial"
    DE_MINIMIS = "de_minimis"
    EXCLUDED = "excluded"


class BoundaryLockStatus(str, Enum):
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    LOCKED = "locked"
    SUPERSEDED = "superseded"


# =============================================================================
# REFERENCE DATA
# =============================================================================

DE_MINIMIS_THRESHOLD_PCT = Decimal("1.0")  # <1% of total = de minimis
IMMATERIAL_THRESHOLD_PCT = Decimal("5.0")  # <5% = immaterial (may still include)

CONSOLIDATION_RULES: Dict[str, Dict[str, Any]] = {
    "equity_share": {
        "description": "Report GHG emissions proportional to equity share in each entity.",
        "ghg_protocol_ref": "Chapter 3, Table 1",
        "includes_jv": True,
        "includes_associate": True,
        "minimum_equity_pct": Decimal("0"),
    },
    "financial_control": {
        "description": "Report 100% of emissions from entities over which financial control.",
        "ghg_protocol_ref": "Chapter 3, Table 1",
        "includes_jv": False,
        "includes_associate": False,
        "minimum_equity_pct": Decimal("50"),
    },
    "operational_control": {
        "description": "Report 100% of emissions from entities over which operational control.",
        "ghg_protocol_ref": "Chapter 3, Table 1",
        "includes_jv": False,
        "includes_associate": False,
        "minimum_equity_pct": Decimal("0"),
    },
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    phase_name: str = Field(...)
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class LegalEntity(BaseModel):
    """A legal entity in the corporate structure."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_id: str = Field(default_factory=_new_uuid)
    entity_name: str = Field(...)
    entity_type: EntityType = Field(EntityType.SUBSIDIARY)
    parent_entity_id: Optional[str] = Field(None)
    country_code: str = Field("")
    jurisdiction: str = Field("")
    incorporation_date: str = Field("")
    is_active: bool = Field(True)


class EntityFacilityMapping(BaseModel):
    """Maps a legal entity to its physical facilities / sites."""
    entity_id: str = Field(...)
    entity_name: str = Field("")
    facility_ids: List[str] = Field(default_factory=list)
    facility_names: List[str] = Field(default_factory=list)
    facility_count: int = Field(0)


class OwnershipLink(BaseModel):
    """A single ownership / control link between entities."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    parent_entity_id: str = Field(...)
    child_entity_id: str = Field(...)
    ownership_pct: Decimal = Field(Decimal("100.00"), ge=Decimal("0"), le=Decimal("100"))
    has_financial_control: bool = Field(False)
    has_operational_control: bool = Field(False)
    relationship_type: EntityType = Field(EntityType.SUBSIDIARY)
    effective_date: str = Field("")
    notes: str = Field("")


class ConsolidationResult(BaseModel):
    """Result of applying a consolidation approach to an entity."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_id: str = Field(...)
    entity_name: str = Field("")
    approach: ConsolidationApproach = Field(ConsolidationApproach.OPERATIONAL_CONTROL)
    ownership_pct: Decimal = Field(Decimal("100.00"))
    reporting_pct: Decimal = Field(Decimal("100.00"), description="% of emissions to report")
    is_included: bool = Field(True)
    inclusion_reason: str = Field("")
    exclusion_reason: str = Field("")
    control_type: ControlType = Field(ControlType.OPERATIONAL)


class MaterialityAssessment(BaseModel):
    """Materiality assessment for a single entity."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_id: str = Field(...)
    entity_name: str = Field("")
    estimated_emissions_tco2e: Decimal = Field(Decimal("0"))
    share_of_total_pct: Decimal = Field(Decimal("0"))
    classification: MaterialityClassification = Field(MaterialityClassification.MATERIAL)
    rationale: str = Field("")
    is_included_override: Optional[bool] = Field(None)


class BoundaryDocument(BaseModel):
    """Final locked boundary document."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    boundary_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    approach: ConsolidationApproach = Field(ConsolidationApproach.OPERATIONAL_CONTROL)
    total_entities: int = Field(0)
    included_entities: int = Field(0)
    excluded_entities: int = Field(0)
    total_facilities: int = Field(0)
    entity_list: List[Dict[str, Any]] = Field(default_factory=list)
    exclusions_list: List[Dict[str, Any]] = Field(default_factory=list)
    lock_status: BoundaryLockStatus = Field(BoundaryLockStatus.DRAFT)
    locked_at: str = Field("")
    locked_by: str = Field("")
    change_log: List[str] = Field(default_factory=list)
    provenance_hash: str = Field("")


class BoundaryDefinitionInput(BaseModel):
    """Input for the boundary definition workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    organisation_id: str = Field(...)
    organisation_name: str = Field("")
    reporting_year: int = Field(...)
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL
    )
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    ownership_links: List[Dict[str, Any]] = Field(default_factory=list)
    entity_facility_map: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_emissions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-entity emission estimates for materiality"
    )
    prior_year_boundary: Optional[Dict[str, Any]] = Field(None)
    lock_boundary: bool = Field(True, description="Lock on completion")
    locked_by: str = Field("system", description="User locking the boundary")
    skip_phases: List[str] = Field(default_factory=list)


class BoundaryDefinitionResult(BaseModel):
    """Output from the boundary definition workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    workflow_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    boundary_document: Optional[BoundaryDocument] = Field(None)
    entity_count: int = Field(0)
    included_count: int = Field(0)
    excluded_count: int = Field(0)
    yoy_changes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(0.0)
    provenance_hash: str = Field("")
    started_at: str = Field("")
    completed_at: str = Field("")


# =============================================================================
# WORKFLOW CLASS
# =============================================================================


class BoundaryDefinitionWorkflow:
    """
    5-phase organisational boundary definition workflow.

    Maps entities to facilities, resolves ownership chains, applies the
    selected consolidation approach, assesses materiality, and locks the
    boundary with full audit trail and SHA-256 provenance.

    Example:
        >>> wf = BoundaryDefinitionWorkflow()
        >>> inp = BoundaryDefinitionInput(
        ...     organisation_id="ORG-001", reporting_year=2025,
        ...     entities=[{"entity_name": "HQ Corp", "entity_type": "parent"}],
        ... )
        >>> result = wf.execute(inp)
    """

    PHASE_ORDER: List[BoundaryPhase] = [
        BoundaryPhase.ENTITY_MAPPING,
        BoundaryPhase.OWNERSHIP_CHAIN,
        BoundaryPhase.CONSOLIDATION_APPROACH,
        BoundaryPhase.MATERIALITY_CHECK,
        BoundaryPhase.BOUNDARY_LOCK,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._entities: List[LegalEntity] = []
        self._mappings: List[EntityFacilityMapping] = []
        self._links: List[OwnershipLink] = []
        self._consolidation: Dict[str, ConsolidationResult] = {}
        self._materiality: Dict[str, MaterialityAssessment] = {}

    def execute(self, input_data: BoundaryDefinitionInput) -> BoundaryDefinitionResult:
        """Execute the full 5-phase boundary definition workflow."""
        start = _utcnow()
        result = BoundaryDefinitionResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            status=WorkflowStatus.RUNNING,
            started_at=start.isoformat(),
        )

        phase_methods = {
            BoundaryPhase.ENTITY_MAPPING: self._phase_entity_mapping,
            BoundaryPhase.OWNERSHIP_CHAIN: self._phase_ownership_chain,
            BoundaryPhase.CONSOLIDATION_APPROACH: self._phase_consolidation_approach,
            BoundaryPhase.MATERIALITY_CHECK: self._phase_materiality_check,
            BoundaryPhase.BOUNDARY_LOCK: self._phase_boundary_lock,
        }

        for idx, phase in enumerate(self.PHASE_ORDER, 1):
            if phase.value in input_data.skip_phases:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx, status=PhaseStatus.SKIPPED,
                ))
                continue

            phase_start = _utcnow()
            try:
                phase_out = phase_methods[phase](input_data, result)
                elapsed = (_utcnow() - phase_start).total_seconds()
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
                    outputs=phase_out, provenance_hash=_compute_hash(str(phase_out)),
                ))
            except Exception as exc:
                elapsed = (_utcnow() - phase_start).total_seconds()
                logger.error("Phase %s failed: %s", phase.value, exc, exc_info=True)
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.FAILED, duration_seconds=elapsed,
                    errors=[str(exc)],
                ))
                result.status = WorkflowStatus.FAILED
                result.errors.append(f"Phase {phase.value}: {exc}")
                break

        if result.status != WorkflowStatus.FAILED:
            result.status = WorkflowStatus.COMPLETED

        end = _utcnow()
        result.completed_at = end.isoformat()
        result.duration_seconds = (end - start).total_seconds()
        result.provenance_hash = _compute_hash(
            f"{result.workflow_id}|{result.organisation_id}|"
            f"{result.included_count}|{result.completed_at}"
        )
        return result

    # -----------------------------------------------------------------
    # PHASE 1 -- ENTITY MAPPING
    # -----------------------------------------------------------------

    def _phase_entity_mapping(
        self, input_data: BoundaryDefinitionInput, result: BoundaryDefinitionResult,
    ) -> Dict[str, Any]:
        """Map legal entities to facilities."""
        logger.info("Phase 1 -- Entity Mapping: %d entities", len(input_data.entities))
        entities: List[LegalEntity] = []

        for raw in input_data.entities:
            try:
                et = EntityType(raw.get("entity_type", "subsidiary"))
            except ValueError:
                et = EntityType.SUBSIDIARY

            ent = LegalEntity(
                entity_id=raw.get("entity_id", _new_uuid()),
                entity_name=raw.get("entity_name", "Unknown"),
                entity_type=et,
                parent_entity_id=raw.get("parent_entity_id"),
                country_code=raw.get("country_code", ""),
                jurisdiction=raw.get("jurisdiction", ""),
                incorporation_date=raw.get("incorporation_date", ""),
                is_active=raw.get("is_active", True),
            )
            entities.append(ent)

        self._entities = entities
        result.entity_count = len(entities)

        # Build facility mappings
        mappings: List[EntityFacilityMapping] = []
        for efm in input_data.entity_facility_map:
            eid = efm.get("entity_id", "")
            fids = efm.get("facility_ids", [])
            fnames = efm.get("facility_names", [])
            mapping = EntityFacilityMapping(
                entity_id=eid,
                entity_name=efm.get("entity_name", ""),
                facility_ids=fids,
                facility_names=fnames,
                facility_count=len(fids),
            )
            mappings.append(mapping)

        self._mappings = mappings
        total_facilities = sum(m.facility_count for m in mappings)

        logger.info("Mapped %d entities to %d facilities", len(entities), total_facilities)
        return {
            "entities_mapped": len(entities),
            "facilities_mapped": total_facilities,
            "entity_types": self._count_by(entities, lambda e: e.entity_type.value),
        }

    def _count_by(self, items: List[Any], key_fn: Any) -> Dict[str, int]:
        """Count items by a key function."""
        counts: Dict[str, int] = {}
        for item in items:
            k = key_fn(item)
            counts[k] = counts.get(k, 0) + 1
        return counts

    # -----------------------------------------------------------------
    # PHASE 2 -- OWNERSHIP CHAIN
    # -----------------------------------------------------------------

    def _phase_ownership_chain(
        self, input_data: BoundaryDefinitionInput, result: BoundaryDefinitionResult,
    ) -> Dict[str, Any]:
        """Define ownership percentages and control relationships."""
        logger.info("Phase 2 -- Ownership Chain: %d links", len(input_data.ownership_links))
        links: List[OwnershipLink] = []

        for raw in input_data.ownership_links:
            try:
                rel = EntityType(raw.get("relationship_type", "subsidiary"))
            except ValueError:
                rel = EntityType.SUBSIDIARY

            try:
                own_pct = Decimal(str(raw.get("ownership_pct", "100")))
            except Exception:
                own_pct = Decimal("100.00")

            link = OwnershipLink(
                parent_entity_id=raw.get("parent_entity_id", ""),
                child_entity_id=raw.get("child_entity_id", ""),
                ownership_pct=own_pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                has_financial_control=raw.get("has_financial_control", own_pct > Decimal("50")),
                has_operational_control=raw.get("has_operational_control", True),
                relationship_type=rel,
                effective_date=raw.get("effective_date", ""),
                notes=raw.get("notes", ""),
            )
            links.append(link)

        self._links = links

        # Compute effective ownership for each entity through chain
        effective_ownership = self._compute_effective_ownership(links)

        jv_count = sum(1 for l in links if l.relationship_type == EntityType.JOINT_VENTURE)
        avg_own = Decimal("0")
        if links:
            avg_own = (sum(l.ownership_pct for l in links) / Decimal(str(len(links)))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        logger.info("Ownership chain: %d links, %d JVs, avg ownership %.1f%%",
                     len(links), jv_count, float(avg_own))
        return {
            "ownership_links": len(links),
            "joint_ventures": jv_count,
            "average_ownership_pct": float(avg_own),
            "effective_ownership_count": len(effective_ownership),
        }

    def _compute_effective_ownership(
        self, links: List[OwnershipLink]
    ) -> Dict[str, Decimal]:
        """Compute effective ownership for each child entity through parent chain."""
        child_to_parent: Dict[str, List[OwnershipLink]] = {}
        for link in links:
            child_to_parent.setdefault(link.child_entity_id, []).append(link)

        effective: Dict[str, Decimal] = {}
        for entity in self._entities:
            eid = entity.entity_id
            if eid not in child_to_parent:
                effective[eid] = Decimal("100.00")
            else:
                # Take the direct ownership (simplified: use direct parent link)
                direct_links = child_to_parent[eid]
                if direct_links:
                    effective[eid] = direct_links[0].ownership_pct
                else:
                    effective[eid] = Decimal("100.00")
        return effective

    # -----------------------------------------------------------------
    # PHASE 3 -- CONSOLIDATION APPROACH
    # -----------------------------------------------------------------

    def _phase_consolidation_approach(
        self, input_data: BoundaryDefinitionInput, result: BoundaryDefinitionResult,
    ) -> Dict[str, Any]:
        """Apply the consolidation approach per GHG Protocol Chapter 3."""
        approach = input_data.consolidation_approach
        rules = CONSOLIDATION_RULES.get(approach.value, {})
        logger.info("Phase 3 -- Consolidation Approach: %s", approach.value)

        link_lookup: Dict[str, OwnershipLink] = {}
        for link in self._links:
            link_lookup[link.child_entity_id] = link

        consolidation: Dict[str, ConsolidationResult] = {}
        included = 0
        excluded = 0

        for entity in self._entities:
            eid = entity.entity_id
            link = link_lookup.get(eid)
            own_pct = link.ownership_pct if link else Decimal("100.00")
            has_fin = link.has_financial_control if link else True
            has_ops = link.has_operational_control if link else True

            is_included = True
            reporting_pct = Decimal("100.00")
            exclusion_reason = ""
            inclusion_reason = ""
            ctrl = ControlType.OPERATIONAL

            if approach == ConsolidationApproach.EQUITY_SHARE:
                reporting_pct = own_pct
                is_included = own_pct > Decimal("0")
                inclusion_reason = f"Equity share: {own_pct}% of emissions reported"
                ctrl = ControlType.NONE
                if not is_included:
                    exclusion_reason = "Zero equity interest"

            elif approach == ConsolidationApproach.FINANCIAL_CONTROL:
                ctrl = ControlType.FINANCIAL
                if has_fin:
                    reporting_pct = Decimal("100.00")
                    inclusion_reason = "Financial control: 100% of emissions reported"
                else:
                    is_included = False
                    reporting_pct = Decimal("0")
                    exclusion_reason = "No financial control"

            elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
                ctrl = ControlType.OPERATIONAL
                if has_ops:
                    reporting_pct = Decimal("100.00")
                    inclusion_reason = "Operational control: 100% of emissions reported"
                else:
                    is_included = False
                    reporting_pct = Decimal("0")
                    exclusion_reason = "No operational control"

            # Check entity type compatibility
            if entity.entity_type == EntityType.JOINT_VENTURE and not rules.get("includes_jv", False):
                if approach != ConsolidationApproach.EQUITY_SHARE:
                    is_included = has_ops if approach == ConsolidationApproach.OPERATIONAL_CONTROL else has_fin
                    if not is_included:
                        exclusion_reason = f"JV excluded under {approach.value} (no control)"

            if is_included:
                included += 1
            else:
                excluded += 1

            cr = ConsolidationResult(
                entity_id=eid,
                entity_name=entity.entity_name,
                approach=approach,
                ownership_pct=own_pct,
                reporting_pct=reporting_pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                is_included=is_included,
                inclusion_reason=inclusion_reason,
                exclusion_reason=exclusion_reason,
                control_type=ctrl,
            )
            consolidation[eid] = cr

        self._consolidation = consolidation
        result.included_count = included
        result.excluded_count = excluded

        logger.info("Consolidation: %d included, %d excluded under %s",
                     included, excluded, approach.value)
        return {
            "approach": approach.value,
            "included": included,
            "excluded": excluded,
            "approach_description": rules.get("description", ""),
        }

    # -----------------------------------------------------------------
    # PHASE 4 -- MATERIALITY CHECK
    # -----------------------------------------------------------------

    def _phase_materiality_check(
        self, input_data: BoundaryDefinitionInput, result: BoundaryDefinitionResult,
    ) -> Dict[str, Any]:
        """Assess materiality per entity and flag exclusions."""
        logger.info("Phase 4 -- Materiality Check")
        emission_lookup: Dict[str, Decimal] = {}
        for rec in input_data.estimated_emissions:
            eid = rec.get("entity_id", "")
            try:
                emission_lookup[eid] = Decimal(str(rec.get("estimated_emissions_tco2e", "0")))
            except Exception:
                emission_lookup[eid] = Decimal("0")

        total_emissions = sum(emission_lookup.values()) or Decimal("1")
        materiality: Dict[str, MaterialityAssessment] = {}
        material_count = 0
        immaterial_count = 0
        de_minimis_count = 0

        for entity in self._entities:
            eid = entity.entity_id
            cons = self._consolidation.get(eid)
            if cons and not cons.is_included:
                continue

            estimated = emission_lookup.get(eid, Decimal("0"))
            share = (estimated / total_emissions * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ) if total_emissions > Decimal("0") else Decimal("0")

            if share < DE_MINIMIS_THRESHOLD_PCT:
                classification = MaterialityClassification.DE_MINIMIS
                rationale = f"Emissions share {share}% < {DE_MINIMIS_THRESHOLD_PCT}% de minimis threshold"
                de_minimis_count += 1
            elif share < IMMATERIAL_THRESHOLD_PCT:
                classification = MaterialityClassification.IMMATERIAL
                rationale = f"Emissions share {share}% < {IMMATERIAL_THRESHOLD_PCT}% materiality threshold"
                immaterial_count += 1
            else:
                classification = MaterialityClassification.MATERIAL
                rationale = f"Emissions share {share}% -- material to the inventory"
                material_count += 1

            ma = MaterialityAssessment(
                entity_id=eid,
                entity_name=entity.entity_name,
                estimated_emissions_tco2e=estimated,
                share_of_total_pct=share,
                classification=classification,
                rationale=rationale,
            )
            materiality[eid] = ma

        self._materiality = materiality

        # Flag de minimis exclusions
        for eid, ma in materiality.items():
            if ma.classification == MaterialityClassification.DE_MINIMIS:
                result.warnings.append(
                    f"Entity {ma.entity_name} is de minimis ({ma.share_of_total_pct}%) "
                    f"-- consider exclusion"
                )

        logger.info("Materiality: %d material, %d immaterial, %d de minimis",
                     material_count, immaterial_count, de_minimis_count)
        return {
            "material_count": material_count,
            "immaterial_count": immaterial_count,
            "de_minimis_count": de_minimis_count,
            "total_estimated_emissions": float(total_emissions),
        }

    # -----------------------------------------------------------------
    # PHASE 5 -- BOUNDARY LOCK
    # -----------------------------------------------------------------

    def _phase_boundary_lock(
        self, input_data: BoundaryDefinitionInput, result: BoundaryDefinitionResult,
    ) -> Dict[str, Any]:
        """Lock the boundary and generate documentation."""
        logger.info("Phase 5 -- Boundary Lock")
        now_iso = _utcnow().isoformat()

        # Build entity list for document
        entity_list: List[Dict[str, Any]] = []
        exclusions_list: List[Dict[str, Any]] = []

        for entity in self._entities:
            eid = entity.entity_id
            cons = self._consolidation.get(eid)
            mat = self._materiality.get(eid)

            entry = {
                "entity_id": eid,
                "entity_name": entity.entity_name,
                "entity_type": entity.entity_type.value,
                "country_code": entity.country_code,
                "ownership_pct": float(cons.ownership_pct) if cons else 100.0,
                "reporting_pct": float(cons.reporting_pct) if cons else 100.0,
                "is_included": cons.is_included if cons else True,
                "materiality": mat.classification.value if mat else "material",
                "emissions_share_pct": float(mat.share_of_total_pct) if mat else 0.0,
            }

            if cons and cons.is_included:
                entity_list.append(entry)
            else:
                entry["exclusion_reason"] = cons.exclusion_reason if cons else ""
                exclusions_list.append(entry)

        # Detect YoY changes
        yoy_changes: List[str] = []
        if input_data.prior_year_boundary:
            prior_entities = set(
                e.get("entity_id", "")
                for e in input_data.prior_year_boundary.get("entity_list", [])
            )
            current_entities = set(e["entity_id"] for e in entity_list)
            added = current_entities - prior_entities
            removed = prior_entities - current_entities
            for a in added:
                yoy_changes.append(f"NEW: Entity {a} added to boundary")
            for r in removed:
                yoy_changes.append(f"REMOVED: Entity {r} removed from boundary")

        result.yoy_changes = yoy_changes

        # Determine lock status
        lock_status = BoundaryLockStatus.LOCKED if input_data.lock_boundary else BoundaryLockStatus.DRAFT

        total_fac = sum(m.facility_count for m in self._mappings)

        prov_data = (
            f"{input_data.organisation_id}|{input_data.reporting_year}|"
            f"{input_data.consolidation_approach.value}|{len(entity_list)}|"
            f"{len(exclusions_list)}|{now_iso}"
        )
        prov_hash = _compute_hash(prov_data)

        doc = BoundaryDocument(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            approach=input_data.consolidation_approach,
            total_entities=len(self._entities),
            included_entities=len(entity_list),
            excluded_entities=len(exclusions_list),
            total_facilities=total_fac,
            entity_list=entity_list,
            exclusions_list=exclusions_list,
            lock_status=lock_status,
            locked_at=now_iso if lock_status == BoundaryLockStatus.LOCKED else "",
            locked_by=input_data.locked_by if lock_status == BoundaryLockStatus.LOCKED else "",
            change_log=yoy_changes,
            provenance_hash=prov_hash,
        )
        result.boundary_document = doc

        logger.info(
            "Boundary %s: %d included, %d excluded, %d facilities, %d YoY changes",
            lock_status.value, len(entity_list), len(exclusions_list),
            total_fac, len(yoy_changes),
        )
        return {
            "lock_status": lock_status.value,
            "included_entities": len(entity_list),
            "excluded_entities": len(exclusions_list),
            "total_facilities": total_fac,
            "yoy_changes": len(yoy_changes),
            "provenance_hash": prov_hash,
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "BoundaryDefinitionWorkflow",
    "BoundaryDefinitionInput",
    "BoundaryDefinitionResult",
    "BoundaryPhase",
    "ConsolidationApproach",
    "EntityType",
    "ControlType",
    "MaterialityClassification",
    "BoundaryLockStatus",
    "LegalEntity",
    "EntityFacilityMapping",
    "OwnershipLink",
    "ConsolidationResult",
    "MaterialityAssessment",
    "BoundaryDocument",
    "PhaseResult",
    "PhaseStatus",
    "WorkflowStatus",
]
