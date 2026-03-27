# -*- coding: utf-8 -*-
"""
Entity Mapping Workflow
====================================

5-phase workflow for corporate entity discovery and organisational boundary
mapping per GHG Protocol Corporate Standard Chapter 3 within PACK-050
GHG Consolidation Pack.

Phases:
    1. EntityDiscovery           -- Discover all corporate entities from
                                    ERP/legal registry data sources, producing
                                    a deduplicated entity candidate list.
    2. OwnershipChainResolution  -- Resolve ownership chains and calculate
                                    effective ownership percentages through
                                    multi-tier holding structures.
    3. ControlAssessment         -- Assess operational and financial control
                                    for each entity per GHG Protocol criteria.
    4. MaterialityScreening      -- Apply materiality threshold to identify
                                    and exclude immaterial entities from the
                                    consolidation boundary.
    5. RegistryLock              -- Lock the entity registry for the reporting
                                    period with provenance hash and audit trail.

Regulatory Basis:
    GHG Protocol Corporate Standard (Ch. 3) -- Setting Organisational Boundaries
    GHG Protocol Corporate Standard (Ch. 4) -- Setting Operational Boundaries
    ISO 14064-1:2018 (Cl. 5.1) -- Organisational boundaries
    CSRD / ESRS E1 -- Climate change disclosure (scope & boundary)

Author: GreenLang Team
Version: 50.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

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


class EntityMappingPhase(str, Enum):
    """Entity mapping workflow phases."""
    ENTITY_DISCOVERY = "entity_discovery"
    OWNERSHIP_CHAIN_RESOLUTION = "ownership_chain_resolution"
    CONTROL_ASSESSMENT = "control_assessment"
    MATERIALITY_SCREENING = "materiality_screening"
    REGISTRY_LOCK = "registry_lock"


class EntityType(str, Enum):
    """Legal entity type classification."""
    PARENT = "parent"
    SUBSIDIARY = "subsidiary"
    JOINT_VENTURE = "joint_venture"
    ASSOCIATE = "associate"
    BRANCH = "branch"
    SPECIAL_PURPOSE = "special_purpose"
    OTHER = "other"


class ControlType(str, Enum):
    """GHG Protocol control classification."""
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    NO_CONTROL = "no_control"


class MaterialityClassification(str, Enum):
    """Materiality assessment outcome."""
    MATERIAL = "material"
    IMMATERIAL = "immaterial"
    BORDERLINE = "borderline"


class RegistryLockStatus(str, Enum):
    """Entity registry lock status."""
    UNLOCKED = "unlocked"
    LOCKED = "locked"
    LOCKED_WITH_EXCEPTIONS = "locked_with_exceptions"


# =============================================================================
# REFERENCE DATA
# =============================================================================

DEFAULT_MATERIALITY_THRESHOLD_PCT = Decimal("1.0")
DEFAULT_EMISSIONS_THRESHOLD_TCO2E = Decimal("500")

ENTITY_SOURCE_PRIORITY: Dict[str, int] = {
    "erp": 1,
    "legal_registry": 2,
    "manual": 3,
    "subsidiary_register": 1,
    "annual_report": 2,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class CandidateEntity(BaseModel):
    """A discovered candidate entity before ownership resolution."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_id: str = Field(default_factory=_new_uuid, description="Unique entity ID")
    entity_name: str = Field(..., description="Legal entity name")
    entity_type: EntityType = Field(EntityType.SUBSIDIARY, description="Entity type")
    jurisdiction: str = Field("", description="Legal jurisdiction / country code")
    registration_number: str = Field("", description="Company registration number")
    parent_entity_id: str = Field("", description="Direct parent entity ID")
    direct_ownership_pct: Decimal = Field(
        Decimal("100.00"), ge=Decimal("0"), le=Decimal("100"),
        description="Direct ownership percentage"
    )
    source: str = Field("manual", description="Discovery data source")
    is_active: bool = Field(True, description="Entity active status")
    discovered_at: str = Field(default_factory=lambda: _utcnow().isoformat())


class OwnershipChain(BaseModel):
    """Resolved ownership chain for an entity."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field("", description="Entity name")
    chain: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ownership chain from entity to ultimate parent"
    )
    direct_ownership_pct: Decimal = Field(Decimal("100.00"))
    effective_ownership_pct: Decimal = Field(
        Decimal("100.00"), description="Effective ownership after chain multiplication"
    )
    chain_depth: int = Field(0, description="Number of tiers in ownership chain")
    is_circular: bool = Field(False, description="Circular ownership detected")


class ControlAssessmentResult(BaseModel):
    """Control assessment for a single entity."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field("", description="Entity name")
    has_operational_control: bool = Field(False, description="Operational control flag")
    has_financial_control: bool = Field(False, description="Financial control flag")
    control_type: ControlType = Field(ControlType.NO_CONTROL)
    effective_ownership_pct: Decimal = Field(Decimal("0"))
    board_representation: bool = Field(False, description="Board seat held")
    management_authority: bool = Field(False, description="Management authority present")
    policy_setting_power: bool = Field(False, description="Can set operating policies")
    assessment_notes: str = Field("", description="Assessment rationale")


class MaterialityAssessment(BaseModel):
    """Materiality assessment for a single entity."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field("", description="Entity name")
    estimated_emissions_tco2e: Decimal = Field(Decimal("0"))
    emissions_share_pct: Decimal = Field(Decimal("0"))
    materiality_threshold_pct: Decimal = Field(DEFAULT_MATERIALITY_THRESHOLD_PCT)
    classification: MaterialityClassification = Field(MaterialityClassification.MATERIAL)
    is_included: bool = Field(True, description="Included in consolidation boundary")
    exclusion_rationale: str = Field("", description="Reason for exclusion if immaterial")


class LockedEntity(BaseModel):
    """A fully resolved and locked entity in the registry."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_id: str = Field(...)
    entity_name: str = Field("")
    entity_type: EntityType = Field(EntityType.SUBSIDIARY)
    jurisdiction: str = Field("")
    effective_ownership_pct: Decimal = Field(Decimal("100.00"))
    control_type: ControlType = Field(ControlType.OPERATIONAL_CONTROL)
    materiality: MaterialityClassification = Field(MaterialityClassification.MATERIAL)
    is_included: bool = Field(True)
    locked_at: str = Field("")
    provenance_hash: str = Field("")


class EntityMappingInput(BaseModel):
    """Input for the entity mapping workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    organisation_id: str = Field(..., description="Organisation identifier")
    organisation_name: str = Field("", description="Organisation display name")
    reporting_year: int = Field(..., description="Reporting year")
    entity_data: List[Dict[str, Any]] = Field(
        default_factory=list, description="Raw entity data from ERP/legal sources"
    )
    ownership_links: List[Dict[str, Any]] = Field(
        default_factory=list, description="Ownership link records"
    )
    control_indicators: List[Dict[str, Any]] = Field(
        default_factory=list, description="Control assessment indicators per entity"
    )
    emissions_estimates: List[Dict[str, Any]] = Field(
        default_factory=list, description="Estimated emissions per entity for materiality"
    )
    materiality_threshold_pct: Decimal = Field(
        DEFAULT_MATERIALITY_THRESHOLD_PCT,
        description="Materiality threshold percentage"
    )
    skip_phases: List[str] = Field(default_factory=list, description="Phase names to skip")


class EntityMappingResult(BaseModel):
    """Output from the entity mapping workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workflow_id: str = Field(default_factory=_new_uuid, description="Workflow run ID")
    organisation_id: str = Field("", description="Organisation ID")
    reporting_year: int = Field(0)
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    locked_entities: List[LockedEntity] = Field(default_factory=list)
    total_discovered: int = Field(0)
    total_included: int = Field(0)
    total_excluded: int = Field(0)
    registry_lock_status: RegistryLockStatus = Field(RegistryLockStatus.UNLOCKED)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(0.0)
    provenance_hash: str = Field("")
    started_at: str = Field("")
    completed_at: str = Field("")


# =============================================================================
# WORKFLOW CLASS
# =============================================================================


class EntityMappingWorkflow:
    """
    5-phase entity mapping workflow for GHG consolidation.

    Discovers corporate entities, resolves ownership chains, assesses
    control, screens for materiality, and locks the registry for the
    reporting period with full SHA-256 provenance.

    Example:
        >>> wf = EntityMappingWorkflow()
        >>> inp = EntityMappingInput(
        ...     organisation_id="ORG-001",
        ...     reporting_year=2025,
        ...     entity_data=[{"entity_name": "Sub A", "jurisdiction": "DE"}],
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_ORDER: List[EntityMappingPhase] = [
        EntityMappingPhase.ENTITY_DISCOVERY,
        EntityMappingPhase.OWNERSHIP_CHAIN_RESOLUTION,
        EntityMappingPhase.CONTROL_ASSESSMENT,
        EntityMappingPhase.MATERIALITY_SCREENING,
        EntityMappingPhase.REGISTRY_LOCK,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EntityMappingWorkflow."""
        self.config = config or {}
        self._candidates: List[CandidateEntity] = []
        self._ownership_chains: Dict[str, OwnershipChain] = {}
        self._control_assessments: Dict[str, ControlAssessmentResult] = {}
        self._materiality: Dict[str, MaterialityAssessment] = {}

    # -----------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # -----------------------------------------------------------------

    def execute(self, input_data: EntityMappingInput) -> EntityMappingResult:
        """
        Execute the full 5-phase entity mapping workflow.

        Args:
            input_data: Validated workflow input.

        Returns:
            EntityMappingResult with locked entities and provenance.
        """
        start = _utcnow()
        result = EntityMappingResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            status=WorkflowStatus.RUNNING,
            started_at=start.isoformat(),
        )

        phase_methods = {
            EntityMappingPhase.ENTITY_DISCOVERY: self._phase_entity_discovery,
            EntityMappingPhase.OWNERSHIP_CHAIN_RESOLUTION: self._phase_ownership_chain,
            EntityMappingPhase.CONTROL_ASSESSMENT: self._phase_control_assessment,
            EntityMappingPhase.MATERIALITY_SCREENING: self._phase_materiality_screening,
            EntityMappingPhase.REGISTRY_LOCK: self._phase_registry_lock,
        }

        for idx, phase in enumerate(self.PHASE_ORDER, 1):
            if phase.value in input_data.skip_phases:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.SKIPPED,
                ))
                continue

            phase_start = _utcnow()
            try:
                phase_out = phase_methods[phase](input_data, result)
                elapsed = (_utcnow() - phase_start).total_seconds()
                ph_hash = _compute_hash(str(phase_out))
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
                    outputs=phase_out, provenance_hash=ph_hash,
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
                result.errors.append(f"Phase {phase.value} failed: {exc}")
                break

        if result.status != WorkflowStatus.FAILED:
            result.status = WorkflowStatus.COMPLETED

        end = _utcnow()
        result.completed_at = end.isoformat()
        result.duration_seconds = (end - start).total_seconds()
        result.provenance_hash = _compute_hash(
            f"{result.workflow_id}|{result.organisation_id}|"
            f"{result.total_included}|{result.completed_at}"
        )
        return result

    # -----------------------------------------------------------------
    # PHASE 1 -- ENTITY DISCOVERY
    # -----------------------------------------------------------------

    def _phase_entity_discovery(
        self,
        input_data: EntityMappingInput,
        result: EntityMappingResult,
    ) -> Dict[str, Any]:
        """
        Discover all corporate entities from ERP/legal registry data.

        Parses raw entity dicts, deduplicates by name+jurisdiction,
        and prioritises sources.
        """
        logger.info("Phase 1 -- Entity Discovery: %d raw entities", len(input_data.entity_data))
        seen: Dict[str, CandidateEntity] = {}
        duplicates = 0

        for raw in input_data.entity_data:
            name = raw.get("entity_name", "").strip()
            jurisdiction = raw.get("jurisdiction", "").strip().upper()
            if not name:
                result.warnings.append(f"Skipped entity with empty name: {raw}")
                continue

            dedup_key = f"{name}|{jurisdiction}".lower()
            source = raw.get("source", "manual")
            source_priority = ENTITY_SOURCE_PRIORITY.get(source, 99)

            if dedup_key in seen:
                existing_priority = ENTITY_SOURCE_PRIORITY.get(seen[dedup_key].source, 99)
                if source_priority < existing_priority:
                    seen[dedup_key].source = source
                duplicates += 1
                continue

            try:
                entity_type = EntityType(raw.get("entity_type", "subsidiary"))
            except ValueError:
                entity_type = EntityType.OTHER

            candidate = CandidateEntity(
                entity_name=name,
                entity_type=entity_type,
                jurisdiction=jurisdiction,
                registration_number=raw.get("registration_number", ""),
                parent_entity_id=raw.get("parent_entity_id", ""),
                direct_ownership_pct=self._dec(raw.get("direct_ownership_pct", "100")),
                source=source,
                is_active=raw.get("is_active", True),
            )
            seen[dedup_key] = candidate

        self._candidates = list(seen.values())
        result.total_discovered = len(self._candidates)

        type_dist: Dict[str, int] = {}
        for c in self._candidates:
            type_dist[c.entity_type.value] = type_dist.get(c.entity_type.value, 0) + 1

        logger.info("Discovery: %d entities found, %d duplicates removed",
                     len(self._candidates), duplicates)
        return {
            "entities_discovered": len(self._candidates),
            "duplicates_removed": duplicates,
            "entity_type_distribution": type_dist,
            "sources_used": list({c.source for c in self._candidates}),
        }

    # -----------------------------------------------------------------
    # PHASE 2 -- OWNERSHIP CHAIN RESOLUTION
    # -----------------------------------------------------------------

    def _phase_ownership_chain(
        self,
        input_data: EntityMappingInput,
        result: EntityMappingResult,
    ) -> Dict[str, Any]:
        """
        Resolve ownership chains and calculate effective ownership
        percentages through multi-tier holding structures.
        """
        logger.info("Phase 2 -- Ownership Chain Resolution: %d entities", len(self._candidates))

        # Build parent lookup from explicit links and candidate data
        parent_map: Dict[str, Dict[str, Any]] = {}
        for link in input_data.ownership_links:
            child_id = link.get("child_entity_id", "")
            if child_id:
                parent_map[child_id] = link

        # Also build name-to-id index
        name_to_id: Dict[str, str] = {}
        for c in self._candidates:
            name_to_id[c.entity_name.lower()] = c.entity_id

        chains: Dict[str, OwnershipChain] = {}
        circular_count = 0

        for entity in self._candidates:
            chain_links: List[Dict[str, Any]] = []
            effective_pct = entity.direct_ownership_pct
            visited: set = {entity.entity_id}
            current_id = entity.parent_entity_id
            is_circular = False
            depth = 0

            while current_id and depth < 20:
                if current_id in visited:
                    is_circular = True
                    circular_count += 1
                    result.warnings.append(
                        f"Circular ownership detected for {entity.entity_name}"
                    )
                    break
                visited.add(current_id)

                link_data = parent_map.get(current_id, {})
                link_pct = self._dec(link_data.get("ownership_pct", "100"))
                effective_pct = (effective_pct * link_pct / Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                chain_links.append({
                    "entity_id": current_id,
                    "ownership_pct": float(link_pct),
                })
                current_id = link_data.get("parent_entity_id", "")
                depth += 1

            chain = OwnershipChain(
                entity_id=entity.entity_id,
                entity_name=entity.entity_name,
                chain=chain_links,
                direct_ownership_pct=entity.direct_ownership_pct,
                effective_ownership_pct=effective_pct,
                chain_depth=len(chain_links),
                is_circular=is_circular,
            )
            chains[entity.entity_id] = chain

        self._ownership_chains = chains

        avg_depth = Decimal("0")
        if chains:
            total_depth = sum(c.chain_depth for c in chains.values())
            avg_depth = (Decimal(str(total_depth)) / Decimal(str(len(chains)))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        logger.info("Ownership resolved: %d chains, %d circular, avg depth %.1f",
                     len(chains), circular_count, float(avg_depth))
        return {
            "chains_resolved": len(chains),
            "circular_detected": circular_count,
            "average_chain_depth": float(avg_depth),
            "max_chain_depth": max((c.chain_depth for c in chains.values()), default=0),
        }

    # -----------------------------------------------------------------
    # PHASE 3 -- CONTROL ASSESSMENT
    # -----------------------------------------------------------------

    def _phase_control_assessment(
        self,
        input_data: EntityMappingInput,
        result: EntityMappingResult,
    ) -> Dict[str, Any]:
        """
        Assess operational and financial control for each entity
        per GHG Protocol Corporate Standard Chapter 3 criteria.
        """
        logger.info("Phase 3 -- Control Assessment: %d entities", len(self._candidates))

        indicator_lookup: Dict[str, Dict[str, Any]] = {}
        for ind in input_data.control_indicators:
            eid = ind.get("entity_id", "")
            if eid:
                indicator_lookup[eid] = ind

        assessments: Dict[str, ControlAssessmentResult] = {}
        control_dist: Dict[str, int] = {}

        for entity in self._candidates:
            indicators = indicator_lookup.get(entity.entity_id, {})
            chain = self._ownership_chains.get(entity.entity_id)
            eff_pct = chain.effective_ownership_pct if chain else entity.direct_ownership_pct

            board_rep = bool(indicators.get("board_representation", False))
            mgmt_auth = bool(indicators.get("management_authority", False))
            policy_power = bool(indicators.get("policy_setting_power", False))

            has_op_control = self._determine_operational_control(
                mgmt_auth, policy_power, indicators
            )
            has_fin_control = self._determine_financial_control(
                eff_pct, board_rep, indicators
            )

            if has_op_control:
                control = ControlType.OPERATIONAL_CONTROL
            elif has_fin_control:
                control = ControlType.FINANCIAL_CONTROL
            else:
                control = ControlType.NO_CONTROL

            notes = self._build_control_notes(control, eff_pct, has_op_control, has_fin_control)

            assessment = ControlAssessmentResult(
                entity_id=entity.entity_id,
                entity_name=entity.entity_name,
                has_operational_control=has_op_control,
                has_financial_control=has_fin_control,
                control_type=control,
                effective_ownership_pct=eff_pct,
                board_representation=board_rep,
                management_authority=mgmt_auth,
                policy_setting_power=policy_power,
                assessment_notes=notes,
            )
            assessments[entity.entity_id] = assessment
            control_dist[control.value] = control_dist.get(control.value, 0) + 1

        self._control_assessments = assessments

        logger.info("Control assessment: %s", control_dist)
        return {
            "entities_assessed": len(assessments),
            "control_distribution": control_dist,
            "operational_control_count": control_dist.get("operational_control", 0),
            "financial_control_count": control_dist.get("financial_control", 0),
            "no_control_count": control_dist.get("no_control", 0),
        }

    def _determine_operational_control(
        self, mgmt_auth: bool, policy_power: bool, indicators: Dict[str, Any]
    ) -> bool:
        """Determine operational control per GHG Protocol criteria."""
        if mgmt_auth and policy_power:
            return True
        has_ops = bool(indicators.get("operates_facility", False))
        has_env_policy = bool(indicators.get("sets_environmental_policy", False))
        if has_ops and has_env_policy:
            return True
        return bool(indicators.get("has_operational_control", False))

    def _determine_financial_control(
        self, eff_pct: Decimal, board_rep: bool, indicators: Dict[str, Any]
    ) -> bool:
        """Determine financial control per GHG Protocol criteria."""
        if eff_pct > Decimal("50"):
            return True
        if board_rep and eff_pct >= Decimal("20"):
            return True
        return bool(indicators.get("has_financial_control", False))

    def _build_control_notes(
        self, control: ControlType, eff_pct: Decimal,
        has_op: bool, has_fin: bool
    ) -> str:
        """Build deterministic control assessment notes."""
        if control == ControlType.OPERATIONAL_CONTROL:
            return (
                f"Operational control confirmed at {eff_pct}% effective ownership. "
                f"Entity has authority to introduce and implement operating policies."
            )
        elif control == ControlType.FINANCIAL_CONTROL:
            return (
                f"Financial control confirmed at {eff_pct}% effective ownership. "
                f"Entity has ability to direct financial and operating policies."
            )
        return (
            f"No control at {eff_pct}% effective ownership. "
            f"Op control: {has_op}, Fin control: {has_fin}."
        )

    # -----------------------------------------------------------------
    # PHASE 4 -- MATERIALITY SCREENING
    # -----------------------------------------------------------------

    def _phase_materiality_screening(
        self,
        input_data: EntityMappingInput,
        result: EntityMappingResult,
    ) -> Dict[str, Any]:
        """
        Apply materiality threshold to exclude immaterial entities
        from the consolidation boundary.
        """
        logger.info("Phase 4 -- Materiality Screening")

        # Build emissions estimate lookup
        emissions_lookup: Dict[str, Decimal] = {}
        for est in input_data.emissions_estimates:
            eid = est.get("entity_id", "")
            if eid:
                emissions_lookup[eid] = self._dec(est.get("estimated_emissions_tco2e", "0"))

        total_estimated = sum(emissions_lookup.values()) or Decimal("1")
        threshold_pct = input_data.materiality_threshold_pct

        assessments: Dict[str, MaterialityAssessment] = {}
        included = 0
        excluded = 0
        borderline = 0

        for entity in self._candidates:
            entity_emissions = emissions_lookup.get(entity.entity_id, Decimal("0"))
            share_pct = (entity_emissions / total_estimated * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            if share_pct >= threshold_pct:
                classification = MaterialityClassification.MATERIAL
                is_included = True
                included += 1
            elif share_pct >= threshold_pct * Decimal("0.5"):
                classification = MaterialityClassification.BORDERLINE
                is_included = True
                included += 1
                borderline += 1
                result.warnings.append(
                    f"Borderline materiality for {entity.entity_name}: {share_pct}%"
                )
            else:
                classification = MaterialityClassification.IMMATERIAL
                is_included = False
                excluded += 1

            # Always include entities with control even if immaterial
            control = self._control_assessments.get(entity.entity_id)
            if control and control.control_type != ControlType.NO_CONTROL and not is_included:
                is_included = True
                classification = MaterialityClassification.MATERIAL
                included += 1
                excluded -= 1
                result.warnings.append(
                    f"Entity {entity.entity_name} below threshold but has control -- included"
                )

            exclusion_rationale = ""
            if not is_included:
                exclusion_rationale = (
                    f"Emissions share {share_pct}% below materiality threshold "
                    f"{threshold_pct}% and no control relationship."
                )

            assessments[entity.entity_id] = MaterialityAssessment(
                entity_id=entity.entity_id,
                entity_name=entity.entity_name,
                estimated_emissions_tco2e=entity_emissions,
                emissions_share_pct=share_pct,
                materiality_threshold_pct=threshold_pct,
                classification=classification,
                is_included=is_included,
                exclusion_rationale=exclusion_rationale,
            )

        self._materiality = assessments

        logger.info("Materiality: %d included, %d excluded, %d borderline",
                     included, excluded, borderline)
        return {
            "entities_screened": len(assessments),
            "included_count": included,
            "excluded_count": excluded,
            "borderline_count": borderline,
            "threshold_pct": float(threshold_pct),
            "total_estimated_tco2e": float(total_estimated),
        }

    # -----------------------------------------------------------------
    # PHASE 5 -- REGISTRY LOCK
    # -----------------------------------------------------------------

    def _phase_registry_lock(
        self,
        input_data: EntityMappingInput,
        result: EntityMappingResult,
    ) -> Dict[str, Any]:
        """
        Lock the entity registry for the reporting period with
        provenance hashes and audit trail.
        """
        logger.info("Phase 5 -- Registry Lock")
        now_iso = _utcnow().isoformat()
        locked_entities: List[LockedEntity] = []
        included = 0
        excluded = 0

        for entity in self._candidates:
            materiality = self._materiality.get(entity.entity_id)
            control = self._control_assessments.get(entity.entity_id)
            chain = self._ownership_chains.get(entity.entity_id)

            is_included = materiality.is_included if materiality else True
            mat_class = materiality.classification if materiality else MaterialityClassification.MATERIAL
            ctrl_type = control.control_type if control else ControlType.OPERATIONAL_CONTROL
            eff_pct = chain.effective_ownership_pct if chain else entity.direct_ownership_pct

            prov_input = (
                f"{entity.entity_id}|{entity.entity_name}|{eff_pct}|"
                f"{ctrl_type.value}|{mat_class.value}|{now_iso}"
            )
            prov_hash = _compute_hash(prov_input)

            locked = LockedEntity(
                entity_id=entity.entity_id,
                entity_name=entity.entity_name,
                entity_type=entity.entity_type,
                jurisdiction=entity.jurisdiction,
                effective_ownership_pct=eff_pct,
                control_type=ctrl_type,
                materiality=mat_class,
                is_included=is_included,
                locked_at=now_iso,
                provenance_hash=prov_hash,
            )
            locked_entities.append(locked)

            if is_included:
                included += 1
            else:
                excluded += 1

        # Determine lock status
        has_exceptions = any(
            m.classification == MaterialityClassification.BORDERLINE
            for m in self._materiality.values()
        )
        lock_status = (
            RegistryLockStatus.LOCKED_WITH_EXCEPTIONS
            if has_exceptions
            else RegistryLockStatus.LOCKED
        )

        result.locked_entities = locked_entities
        result.total_included = included
        result.total_excluded = excluded
        result.registry_lock_status = lock_status

        logger.info("Registry locked: %d included, %d excluded, status=%s",
                     included, excluded, lock_status.value)
        return {
            "locked_count": len(locked_entities),
            "included_count": included,
            "excluded_count": excluded,
            "lock_status": lock_status.value,
            "locked_at": now_iso,
        }

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    def _dec(self, value: Any) -> Decimal:
        """Safely parse to Decimal."""
        if value is None:
            return Decimal("0")
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal("0")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "EntityMappingWorkflow",
    "EntityMappingInput",
    "EntityMappingResult",
    "EntityMappingPhase",
    "PhaseStatus",
    "WorkflowStatus",
    "EntityType",
    "ControlType",
    "MaterialityClassification",
    "RegistryLockStatus",
    "CandidateEntity",
    "OwnershipChain",
    "ControlAssessmentResult",
    "MaterialityAssessment",
    "LockedEntity",
    "PhaseResult",
]
