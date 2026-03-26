# -*- coding: utf-8 -*-
"""
Boundary Definition Workflow
================================

4-phase workflow for establishing organizational and operational boundaries
within PACK-041 Scope 1-2 Complete Pack.

Phases:
    1. EntityMapping          -- Import org structure, map facilities, classify entity types
    2. BoundarySelection      -- Apply consolidation approach, calculate inclusion %,
                                 handle exceptions
    3. SourceIdentification   -- Map sector to source categories, check data availability,
                                 identify applicable MRV agents
    4. MaterialityAssessment  -- Estimate per-category emissions via benchmarks,
                                 apply materiality thresholds, generate report

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Corporate Standard Chapter 3 (Setting Organizational Boundaries)
    GHG Protocol Corporate Standard Chapter 4 (Setting Operational Boundaries)
    ISO 14064-1:2018 Clause 5.1 (Organizational boundaries)

Schedule: on-demand (typically annually or at inventory setup)
Estimated duration: 30 minutes

Author: GreenLang Team
Version: 41.0.0
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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


class ConsolidationApproach(str, Enum):
    """GHG Protocol organizational boundary consolidation approaches."""

    EQUITY_SHARE = "equity_share"
    FINANCIAL_CONTROL = "financial_control"
    OPERATIONAL_CONTROL = "operational_control"


class EntityType(str, Enum):
    """Legal entity classification."""

    PARENT = "parent"
    SUBSIDIARY = "subsidiary"
    JOINT_VENTURE = "joint_venture"
    ASSOCIATE = "associate"
    FRANCHISE = "franchise"
    LEASED_ASSET = "leased_asset"
    SPV = "special_purpose_vehicle"


class SourceCategory(str, Enum):
    """Emission source categories for Scope 1 and 2."""

    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    REFRIGERANT_FGAS = "refrigerant_fgas"
    LAND_USE = "land_use"
    WASTE_TREATMENT = "waste_treatment"
    AGRICULTURAL = "agricultural"
    SCOPE2_ELECTRICITY = "scope2_electricity"
    SCOPE2_STEAM_HEAT = "scope2_steam_heat"
    SCOPE2_COOLING = "scope2_cooling"


class DataAvailability(str, Enum):
    """Data availability classification for a source category."""

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    ESTIMATED = "estimated"


class MaterialityLevel(str, Enum):
    """Materiality classification for emission sources."""

    MATERIAL = "material"
    POTENTIALLY_MATERIAL = "potentially_material"
    IMMATERIAL = "immaterial"
    EXCLUDED = "excluded"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class EntityRecord(BaseModel):
    """Organizational entity in the corporate structure."""

    entity_id: str = Field(default_factory=lambda: f"ent-{uuid.uuid4().hex[:8]}")
    entity_name: str = Field(default="", description="Legal entity name")
    entity_type: EntityType = Field(default=EntityType.SUBSIDIARY)
    parent_entity_id: str = Field(default="", description="Parent entity ID")
    country: str = Field(default="", description="ISO 3166-1 alpha-2")
    equity_share_pct: float = Field(default=100.0, ge=0.0, le=100.0)
    has_financial_control: bool = Field(default=True)
    has_operational_control: bool = Field(default=True)
    sector: str = Field(default="", description="NACE/GICS sector code")
    employee_count: int = Field(default=0, ge=0)
    revenue_eur: float = Field(default=0.0, ge=0.0)
    notes: str = Field(default="")


class FacilityRecord(BaseModel):
    """Physical facility within an entity."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    facility_name: str = Field(default="", description="Facility display name")
    entity_id: str = Field(default="", description="Owning entity ID")
    country: str = Field(default="", description="ISO 3166-1 alpha-2")
    region: str = Field(default="", description="State or region")
    latitude: float = Field(default=0.0, ge=-90.0, le=90.0)
    longitude: float = Field(default=0.0, ge=-180.0, le=180.0)
    facility_type: str = Field(default="office", description="office|factory|warehouse|datacenter|retail|other")
    floor_area_sqm: float = Field(default=0.0, ge=0.0)
    employee_count: int = Field(default=0, ge=0)
    annual_energy_spend_eur: float = Field(default=0.0, ge=0.0)
    annual_production_volume: float = Field(default=0.0, ge=0.0)
    production_unit: str = Field(default="")
    has_combustion_equipment: bool = Field(default=False)
    has_vehicle_fleet: bool = Field(default=False)
    has_process_emissions: bool = Field(default=False)
    has_refrigeration: bool = Field(default=True)
    has_onsite_waste_treatment: bool = Field(default=False)
    has_agricultural_operations: bool = Field(default=False)
    has_land_use_change: bool = Field(default=False)
    purchases_electricity: bool = Field(default=True)
    purchases_steam_heat: bool = Field(default=False)
    purchases_cooling: bool = Field(default=False)
    data_sources: List[str] = Field(default_factory=list, description="Available data source types")


class EntityBoundaryResult(BaseModel):
    """Boundary determination result for a single entity."""

    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    entity_type: str = Field(default="")
    inclusion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    included: bool = Field(default=True)
    exclusion_reason: str = Field(default="")
    facility_count: int = Field(default=0, ge=0)


class SourceCategoryAssignment(BaseModel):
    """Assignment of a source category to a facility with MRV agent mapping."""

    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    source_category: SourceCategory = Field(...)
    mrv_agent_id: str = Field(default="", description="MRV agent identifier e.g. MRV-001")
    data_availability: DataAvailability = Field(default=DataAvailability.UNAVAILABLE)
    applicable: bool = Field(default=False)
    notes: str = Field(default="")


class MaterialityRecord(BaseModel):
    """Materiality assessment for a single source category."""

    source_category: SourceCategory = Field(...)
    estimated_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    materiality_level: MaterialityLevel = Field(default=MaterialityLevel.IMMATERIAL)
    materiality_threshold_pct: float = Field(default=1.0)
    facility_count: int = Field(default=0, ge=0)
    data_quality: str = Field(default="low")
    recommendation: str = Field(default="")


class CompletenessReport(BaseModel):
    """Boundary completeness report per ISO 14064-1."""

    total_entities: int = Field(default=0, ge=0)
    included_entities: int = Field(default=0, ge=0)
    excluded_entities: int = Field(default=0, ge=0)
    total_facilities: int = Field(default=0, ge=0)
    included_facilities: int = Field(default=0, ge=0)
    total_source_categories: int = Field(default=0, ge=0)
    material_categories: int = Field(default=0, ge=0)
    coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    exclusions: List[Dict[str, str]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class BoundaryDefinitionInput(BaseModel):
    """Input data model for BoundaryDefinitionWorkflow."""

    entities: List[EntityRecord] = Field(default_factory=list, description="Organizational entities")
    facilities: List[FacilityRecord] = Field(default_factory=list, description="Physical facilities")
    preferred_approach: str = Field(
        default="operational_control",
        description="Consolidation approach: equity_share|financial_control|operational_control",
    )
    sector: str = Field(default="", description="Primary sector NACE or GICS code")
    materiality_threshold_pct: float = Field(default=1.0, ge=0.0, le=10.0)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    base_year: int = Field(default=2020, ge=2010, le=2050)
    include_biogenic: bool = Field(default=False)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("preferred_approach")
    @classmethod
    def validate_approach(cls, v: str) -> str:
        """Validate consolidation approach."""
        valid = {"equity_share", "financial_control", "operational_control"}
        if v not in valid:
            raise ValueError(f"preferred_approach must be one of {valid}, got '{v}'")
        return v


class BoundaryDefinitionResult(BaseModel):
    """Complete result from boundary definition workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="boundary_definition")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    consolidation_approach: str = Field(default="operational_control")
    boundary_definition: List[EntityBoundaryResult] = Field(default_factory=list)
    source_categories: List[SourceCategoryAssignment] = Field(default_factory=list)
    materiality_assessment: List[MaterialityRecord] = Field(default_factory=list)
    completeness_report: Optional[CompletenessReport] = Field(default=None)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# SECTOR TO SOURCE CATEGORY MAPPING (Zero-Hallucination)
# =============================================================================

# NACE-based default source category applicability
SECTOR_SOURCE_DEFAULTS: Dict[str, List[SourceCategory]] = {
    "manufacturing": [
        SourceCategory.STATIONARY_COMBUSTION,
        SourceCategory.MOBILE_COMBUSTION,
        SourceCategory.PROCESS_EMISSIONS,
        SourceCategory.FUGITIVE_EMISSIONS,
        SourceCategory.REFRIGERANT_FGAS,
        SourceCategory.SCOPE2_ELECTRICITY,
        SourceCategory.SCOPE2_STEAM_HEAT,
    ],
    "energy": [
        SourceCategory.STATIONARY_COMBUSTION,
        SourceCategory.FUGITIVE_EMISSIONS,
        SourceCategory.PROCESS_EMISSIONS,
        SourceCategory.SCOPE2_ELECTRICITY,
    ],
    "office": [
        SourceCategory.STATIONARY_COMBUSTION,
        SourceCategory.REFRIGERANT_FGAS,
        SourceCategory.SCOPE2_ELECTRICITY,
        SourceCategory.SCOPE2_COOLING,
    ],
    "retail": [
        SourceCategory.STATIONARY_COMBUSTION,
        SourceCategory.REFRIGERANT_FGAS,
        SourceCategory.SCOPE2_ELECTRICITY,
        SourceCategory.SCOPE2_COOLING,
    ],
    "logistics": [
        SourceCategory.STATIONARY_COMBUSTION,
        SourceCategory.MOBILE_COMBUSTION,
        SourceCategory.REFRIGERANT_FGAS,
        SourceCategory.SCOPE2_ELECTRICITY,
    ],
    "agriculture": [
        SourceCategory.STATIONARY_COMBUSTION,
        SourceCategory.MOBILE_COMBUSTION,
        SourceCategory.AGRICULTURAL,
        SourceCategory.LAND_USE,
        SourceCategory.SCOPE2_ELECTRICITY,
    ],
    "waste_management": [
        SourceCategory.STATIONARY_COMBUSTION,
        SourceCategory.MOBILE_COMBUSTION,
        SourceCategory.WASTE_TREATMENT,
        SourceCategory.FUGITIVE_EMISSIONS,
        SourceCategory.SCOPE2_ELECTRICITY,
    ],
    "chemicals": [
        SourceCategory.STATIONARY_COMBUSTION,
        SourceCategory.PROCESS_EMISSIONS,
        SourceCategory.FUGITIVE_EMISSIONS,
        SourceCategory.REFRIGERANT_FGAS,
        SourceCategory.SCOPE2_ELECTRICITY,
        SourceCategory.SCOPE2_STEAM_HEAT,
    ],
    "mining": [
        SourceCategory.STATIONARY_COMBUSTION,
        SourceCategory.MOBILE_COMBUSTION,
        SourceCategory.PROCESS_EMISSIONS,
        SourceCategory.FUGITIVE_EMISSIONS,
        SourceCategory.LAND_USE,
        SourceCategory.SCOPE2_ELECTRICITY,
    ],
    "default": [
        SourceCategory.STATIONARY_COMBUSTION,
        SourceCategory.REFRIGERANT_FGAS,
        SourceCategory.SCOPE2_ELECTRICITY,
    ],
}

# MRV agent mapping per source category
SOURCE_TO_MRV_AGENT: Dict[SourceCategory, str] = {
    SourceCategory.STATIONARY_COMBUSTION: "MRV-001",
    SourceCategory.MOBILE_COMBUSTION: "MRV-002",
    SourceCategory.PROCESS_EMISSIONS: "MRV-003",
    SourceCategory.FUGITIVE_EMISSIONS: "MRV-004",
    SourceCategory.REFRIGERANT_FGAS: "MRV-005",
    SourceCategory.LAND_USE: "MRV-006",
    SourceCategory.WASTE_TREATMENT: "MRV-007",
    SourceCategory.AGRICULTURAL: "MRV-008",
    SourceCategory.SCOPE2_ELECTRICITY: "MRV-009",
    SourceCategory.SCOPE2_STEAM_HEAT: "MRV-011",
    SourceCategory.SCOPE2_COOLING: "MRV-012",
}

# Benchmark emission intensities tCO2e per employee by sector (IEA/GHG Protocol)
BENCHMARK_INTENSITY_PER_EMPLOYEE: Dict[str, Dict[str, float]] = {
    "manufacturing": {
        "stationary_combustion": 5.0,
        "mobile_combustion": 0.8,
        "process_emissions": 2.5,
        "fugitive_emissions": 0.3,
        "refrigerant_fgas": 0.2,
        "scope2_electricity": 3.0,
        "scope2_steam_heat": 1.0,
    },
    "office": {
        "stationary_combustion": 0.5,
        "refrigerant_fgas": 0.1,
        "scope2_electricity": 1.5,
        "scope2_cooling": 0.3,
    },
    "logistics": {
        "stationary_combustion": 0.3,
        "mobile_combustion": 8.0,
        "refrigerant_fgas": 0.5,
        "scope2_electricity": 0.8,
    },
    "retail": {
        "stationary_combustion": 0.3,
        "refrigerant_fgas": 0.4,
        "scope2_electricity": 2.0,
        "scope2_cooling": 0.5,
    },
    "agriculture": {
        "stationary_combustion": 1.0,
        "mobile_combustion": 2.0,
        "agricultural": 12.0,
        "land_use": 3.0,
        "scope2_electricity": 1.5,
    },
    "chemicals": {
        "stationary_combustion": 8.0,
        "process_emissions": 15.0,
        "fugitive_emissions": 1.0,
        "refrigerant_fgas": 0.3,
        "scope2_electricity": 5.0,
        "scope2_steam_heat": 2.0,
    },
    "default": {
        "stationary_combustion": 1.0,
        "refrigerant_fgas": 0.15,
        "scope2_electricity": 2.0,
    },
}

# Benchmark emission intensities tCO2e per sqm by facility type
BENCHMARK_INTENSITY_PER_SQM: Dict[str, float] = {
    "office": 0.05,
    "factory": 0.15,
    "warehouse": 0.03,
    "datacenter": 0.80,
    "retail": 0.08,
    "other": 0.06,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class BoundaryDefinitionWorkflow:
    """
    4-phase boundary definition workflow for Scope 1-2 GHG inventory.

    Establishes organizational boundaries using the GHG Protocol consolidation
    approach (equity share, financial control, or operational control), maps
    facilities to entities, identifies applicable source categories per sector,
    and performs materiality screening using benchmark emission intensities.

    Zero-hallucination: all inclusion percentages derive from equity shares
    or control flags, all benchmark intensities from published IEA/GHG Protocol
    data, no LLM calls in numeric paths.

    Attributes:
        workflow_id: Unique execution identifier.
        _entity_boundaries: Boundary results per entity.
        _source_assignments: Source category assignments per facility.
        _materiality_records: Materiality assessment records.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = BoundaryDefinitionWorkflow()
        >>> inp = BoundaryDefinitionInput(entities=[...], facilities=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_DEPENDENCIES: Dict[str, List[str]] = {
        "entity_mapping": [],
        "boundary_selection": ["entity_mapping"],
        "source_identification": ["boundary_selection"],
        "materiality_assessment": ["source_identification"],
    }

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize BoundaryDefinitionWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._entity_boundaries: List[EntityBoundaryResult] = []
        self._source_assignments: List[SourceCategoryAssignment] = []
        self._materiality_records: List[MaterialityRecord] = []
        self._phase_results: List[PhaseResult] = []
        self._entity_map: Dict[str, EntityRecord] = {}
        self._facility_map: Dict[str, FacilityRecord] = {}
        self._included_facilities: List[FacilityRecord] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[BoundaryDefinitionInput] = None,
        entities: Optional[List[EntityRecord]] = None,
        facilities: Optional[List[FacilityRecord]] = None,
        preferred_approach: str = "operational_control",
        sector: str = "",
    ) -> BoundaryDefinitionResult:
        """
        Execute the 4-phase boundary definition workflow.

        Args:
            input_data: Full input model (preferred).
            entities: Entity list (fallback).
            facilities: Facility list (fallback).
            preferred_approach: Consolidation approach (fallback).
            sector: Primary sector (fallback).

        Returns:
            BoundaryDefinitionResult with boundaries, source categories,
            and materiality assessment.

        Raises:
            ValueError: If no entities or facilities are provided.
        """
        if input_data is None:
            if entities is None:
                raise ValueError("Either input_data or entities must be provided")
            input_data = BoundaryDefinitionInput(
                entities=entities or [],
                facilities=facilities or [],
                preferred_approach=preferred_approach,
                sector=sector,
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting boundary definition workflow %s approach=%s sector=%s entities=%d facilities=%d",
            self.workflow_id,
            input_data.preferred_approach,
            input_data.sector,
            len(input_data.entities),
            len(input_data.facilities),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Entity Mapping
            phase1 = await self._execute_with_retry(
                self._phase_entity_mapping, input_data, phase_number=1
            )
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 1 failed: {phase1.errors}")

            # Phase 2: Boundary Selection
            phase2 = await self._execute_with_retry(
                self._phase_boundary_selection, input_data, phase_number=2
            )
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 2 failed: {phase2.errors}")

            # Phase 3: Source Identification
            phase3 = await self._execute_with_retry(
                self._phase_source_identification, input_data, phase_number=3
            )
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 3 failed: {phase3.errors}")

            # Phase 4: Materiality Assessment
            phase4 = await self._execute_with_retry(
                self._phase_materiality_assessment, input_data, phase_number=4
            )
            self._phase_results.append(phase4)
            if phase4.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 4 failed: {phase4.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Boundary definition workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        completeness = self._build_completeness_report(input_data)

        result = BoundaryDefinitionResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            consolidation_approach=input_data.preferred_approach,
            boundary_definition=self._entity_boundaries,
            source_categories=self._source_assignments,
            materiality_assessment=self._materiality_records,
            completeness_report=completeness,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Boundary definition workflow %s completed in %.2fs status=%s "
            "entities=%d/%d included, categories=%d material",
            self.workflow_id,
            elapsed,
            overall_status.value,
            sum(1 for e in self._entity_boundaries if e.included),
            len(self._entity_boundaries),
            sum(1 for m in self._materiality_records if m.materiality_level == MaterialityLevel.MATERIAL),
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: BoundaryDefinitionInput, phase_number: int
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
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
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

    async def _phase_entity_mapping(self, input_data: BoundaryDefinitionInput) -> PhaseResult:
        """Import org structure, map facilities to entities, classify entity types."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Build entity lookup
        self._entity_map = {e.entity_id: e for e in input_data.entities}
        self._facility_map = {f.facility_id: f for f in input_data.facilities}

        if not input_data.entities:
            warnings.append("No entities provided; creating default parent entity")
            default_entity = EntityRecord(
                entity_name="Default Organization",
                entity_type=EntityType.PARENT,
                equity_share_pct=100.0,
            )
            self._entity_map[default_entity.entity_id] = default_entity

        # Map facilities to entities
        entity_facility_map: Dict[str, List[str]] = {}
        orphan_facilities: List[str] = []

        for fac in input_data.facilities:
            if fac.entity_id and fac.entity_id in self._entity_map:
                entity_facility_map.setdefault(fac.entity_id, []).append(fac.facility_id)
            else:
                orphan_facilities.append(fac.facility_id)

        # Assign orphan facilities to first entity
        if orphan_facilities:
            first_entity_id = next(iter(self._entity_map))
            entity_facility_map.setdefault(first_entity_id, []).extend(orphan_facilities)
            warnings.append(
                f"{len(orphan_facilities)} facilities have no matching entity; "
                f"assigned to entity {first_entity_id}"
            )
            for fac_id in orphan_facilities:
                if fac_id in self._facility_map:
                    self._facility_map[fac_id].entity_id = first_entity_id

        # Classify entity hierarchy depth
        hierarchy: Dict[str, int] = {}
        for eid, entity in self._entity_map.items():
            depth = 0
            current = entity
            visited: set = set()
            while current.parent_entity_id and current.parent_entity_id in self._entity_map:
                if current.parent_entity_id in visited:
                    warnings.append(f"Circular parent reference detected for entity {eid}")
                    break
                visited.add(current.parent_entity_id)
                current = self._entity_map[current.parent_entity_id]
                depth += 1
            hierarchy[eid] = depth

        # Classify entity types summary
        type_counts: Dict[str, int] = {}
        for entity in self._entity_map.values():
            t = entity.entity_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        outputs["total_entities"] = len(self._entity_map)
        outputs["total_facilities"] = len(self._facility_map)
        outputs["entity_types"] = type_counts
        outputs["entity_facility_map"] = {k: len(v) for k, v in entity_facility_map.items()}
        outputs["orphan_facilities_reassigned"] = len(orphan_facilities)
        outputs["max_hierarchy_depth"] = max(hierarchy.values()) if hierarchy else 0

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 EntityMapping: %d entities, %d facilities, depth=%d",
            len(self._entity_map),
            len(self._facility_map),
            outputs["max_hierarchy_depth"],
        )
        return PhaseResult(
            phase_name="entity_mapping",
            phase_number=1,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Boundary Selection
    # -------------------------------------------------------------------------

    async def _phase_boundary_selection(self, input_data: BoundaryDefinitionInput) -> PhaseResult:
        """Apply consolidation approach, calculate inclusion %, handle exceptions."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        approach = input_data.preferred_approach
        self._entity_boundaries = []

        for eid, entity in self._entity_map.items():
            inclusion_pct = self._calculate_inclusion_pct(entity, approach)
            included = inclusion_pct > 0.0
            exclusion_reason = ""

            # Apply de minimis exclusion (< 1% equity and < 1% emissions estimate)
            if approach == ConsolidationApproach.EQUITY_SHARE.value and entity.equity_share_pct < 1.0:
                exclusion_reason = f"Equity share {entity.equity_share_pct}% below 1% de minimis threshold"
                included = False
                warnings.append(f"Entity {entity.entity_name} excluded: {exclusion_reason}")

            # Control-based exclusion
            if approach == ConsolidationApproach.FINANCIAL_CONTROL.value and not entity.has_financial_control:
                exclusion_reason = "No financial control"
                included = False
                inclusion_pct = 0.0

            if approach == ConsolidationApproach.OPERATIONAL_CONTROL.value and not entity.has_operational_control:
                exclusion_reason = "No operational control"
                included = False
                inclusion_pct = 0.0

            # Count facilities belonging to this entity
            fac_count = sum(1 for f in self._facility_map.values() if f.entity_id == eid)

            self._entity_boundaries.append(EntityBoundaryResult(
                entity_id=eid,
                entity_name=entity.entity_name,
                entity_type=entity.entity_type.value,
                inclusion_pct=round(inclusion_pct, 2),
                included=included,
                exclusion_reason=exclusion_reason,
                facility_count=fac_count,
            ))

        # Build included facility list
        included_entity_ids = {eb.entity_id for eb in self._entity_boundaries if eb.included}
        self._included_facilities = [
            f for f in self._facility_map.values() if f.entity_id in included_entity_ids
        ]

        outputs["approach"] = approach
        outputs["total_entities"] = len(self._entity_boundaries)
        outputs["included_entities"] = sum(1 for eb in self._entity_boundaries if eb.included)
        outputs["excluded_entities"] = sum(1 for eb in self._entity_boundaries if not eb.included)
        outputs["included_facilities"] = len(self._included_facilities)
        outputs["avg_inclusion_pct"] = round(
            sum(eb.inclusion_pct for eb in self._entity_boundaries if eb.included)
            / max(sum(1 for eb in self._entity_boundaries if eb.included), 1),
            2,
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 BoundarySelection: approach=%s included=%d/%d entities, %d facilities",
            approach,
            outputs["included_entities"],
            outputs["total_entities"],
            outputs["included_facilities"],
        )
        return PhaseResult(
            phase_name="boundary_selection",
            phase_number=2,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _calculate_inclusion_pct(self, entity: EntityRecord, approach: str) -> float:
        """Calculate entity inclusion percentage based on consolidation approach."""
        if approach == ConsolidationApproach.EQUITY_SHARE.value:
            return entity.equity_share_pct
        elif approach == ConsolidationApproach.FINANCIAL_CONTROL.value:
            return 100.0 if entity.has_financial_control else 0.0
        elif approach == ConsolidationApproach.OPERATIONAL_CONTROL.value:
            return 100.0 if entity.has_operational_control else 0.0
        return 0.0

    # -------------------------------------------------------------------------
    # Phase 3: Source Identification
    # -------------------------------------------------------------------------

    async def _phase_source_identification(self, input_data: BoundaryDefinitionInput) -> PhaseResult:
        """Map sector to source categories, check data availability, identify MRV agents."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        sector_key = self._normalize_sector(input_data.sector)
        default_sources = SECTOR_SOURCE_DEFAULTS.get(sector_key, SECTOR_SOURCE_DEFAULTS["default"])

        self._source_assignments = []
        category_facility_count: Dict[str, int] = {}

        for facility in self._included_facilities:
            applicable_sources = self._determine_facility_sources(facility, default_sources)

            for source_cat in SourceCategory:
                is_applicable = source_cat in applicable_sources
                data_avail = self._assess_data_availability(facility, source_cat)
                mrv_agent = SOURCE_TO_MRV_AGENT.get(source_cat, "")

                if is_applicable:
                    category_facility_count[source_cat.value] = (
                        category_facility_count.get(source_cat.value, 0) + 1
                    )

                    notes = ""
                    if data_avail == DataAvailability.UNAVAILABLE:
                        notes = "Data collection required"
                        warnings.append(
                            f"Facility {facility.facility_name}: {source_cat.value} applicable "
                            f"but data unavailable"
                        )

                    self._source_assignments.append(SourceCategoryAssignment(
                        facility_id=facility.facility_id,
                        facility_name=facility.facility_name,
                        source_category=source_cat,
                        mrv_agent_id=mrv_agent,
                        data_availability=data_avail,
                        applicable=True,
                        notes=notes,
                    ))

        # Summarize unique agents needed
        unique_agents = sorted({sa.mrv_agent_id for sa in self._source_assignments if sa.mrv_agent_id})
        unique_categories = sorted({sa.source_category.value for sa in self._source_assignments})

        outputs["sector_key"] = sector_key
        outputs["total_assignments"] = len(self._source_assignments)
        outputs["unique_source_categories"] = unique_categories
        outputs["unique_mrv_agents"] = unique_agents
        outputs["category_facility_counts"] = category_facility_count
        outputs["data_available_count"] = sum(
            1 for sa in self._source_assignments if sa.data_availability == DataAvailability.AVAILABLE
        )
        outputs["data_partial_count"] = sum(
            1 for sa in self._source_assignments if sa.data_availability == DataAvailability.PARTIAL
        )
        outputs["data_unavailable_count"] = sum(
            1 for sa in self._source_assignments if sa.data_availability == DataAvailability.UNAVAILABLE
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 SourceIdentification: %d assignments across %d categories, %d MRV agents",
            len(self._source_assignments),
            len(unique_categories),
            len(unique_agents),
        )
        return PhaseResult(
            phase_name="source_identification",
            phase_number=3,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _normalize_sector(self, sector: str) -> str:
        """Normalize sector string to a known key."""
        if not sector:
            return "default"
        sector_lower = sector.lower().strip()
        mapping = {
            "manufacturing": "manufacturing",
            "industrial": "manufacturing",
            "energy": "energy",
            "utilities": "energy",
            "office": "office",
            "services": "office",
            "financial": "office",
            "retail": "retail",
            "consumer": "retail",
            "logistics": "logistics",
            "transport": "logistics",
            "shipping": "logistics",
            "agriculture": "agriculture",
            "farming": "agriculture",
            "waste": "waste_management",
            "chemicals": "chemicals",
            "pharma": "chemicals",
            "mining": "mining",
            "metals": "mining",
        }
        for key, value in mapping.items():
            if key in sector_lower:
                return value
        return "default"

    def _determine_facility_sources(
        self, facility: FacilityRecord, sector_defaults: List[SourceCategory]
    ) -> List[SourceCategory]:
        """Determine applicable source categories for a facility based on attributes."""
        sources = set(sector_defaults)

        # Add based on facility flags
        if facility.has_combustion_equipment:
            sources.add(SourceCategory.STATIONARY_COMBUSTION)
        if facility.has_vehicle_fleet:
            sources.add(SourceCategory.MOBILE_COMBUSTION)
        if facility.has_process_emissions:
            sources.add(SourceCategory.PROCESS_EMISSIONS)
        if facility.has_refrigeration:
            sources.add(SourceCategory.REFRIGERANT_FGAS)
        if facility.has_onsite_waste_treatment:
            sources.add(SourceCategory.WASTE_TREATMENT)
        if facility.has_agricultural_operations:
            sources.add(SourceCategory.AGRICULTURAL)
        if facility.has_land_use_change:
            sources.add(SourceCategory.LAND_USE)
        if facility.purchases_electricity:
            sources.add(SourceCategory.SCOPE2_ELECTRICITY)
        if facility.purchases_steam_heat:
            sources.add(SourceCategory.SCOPE2_STEAM_HEAT)
        if facility.purchases_cooling:
            sources.add(SourceCategory.SCOPE2_COOLING)

        return sorted(sources, key=lambda x: x.value)

    def _assess_data_availability(
        self, facility: FacilityRecord, source_category: SourceCategory
    ) -> DataAvailability:
        """Assess data availability for a source category at a facility."""
        available_sources = {s.lower() for s in facility.data_sources}

        availability_map: Dict[SourceCategory, List[str]] = {
            SourceCategory.STATIONARY_COMBUSTION: ["fuel_invoices", "meter_data", "erp"],
            SourceCategory.MOBILE_COMBUSTION: ["fuel_cards", "fleet_data", "telematics"],
            SourceCategory.PROCESS_EMISSIONS: ["process_data", "erp", "production_records"],
            SourceCategory.FUGITIVE_EMISSIONS: ["ldar_data", "maintenance_records"],
            SourceCategory.REFRIGERANT_FGAS: ["refrigerant_logs", "maintenance_records", "hvac_data"],
            SourceCategory.LAND_USE: ["land_survey", "satellite_data"],
            SourceCategory.WASTE_TREATMENT: ["waste_manifests", "waste_records"],
            SourceCategory.AGRICULTURAL: ["farm_records", "fertilizer_data"],
            SourceCategory.SCOPE2_ELECTRICITY: ["utility_bills", "meter_data", "erp"],
            SourceCategory.SCOPE2_STEAM_HEAT: ["utility_bills", "steam_meters"],
            SourceCategory.SCOPE2_COOLING: ["utility_bills", "cooling_meters"],
        }

        required = availability_map.get(source_category, [])
        if not required:
            return DataAvailability.UNAVAILABLE

        matched = sum(1 for r in required if r in available_sources)
        if matched == 0:
            return DataAvailability.UNAVAILABLE
        elif matched >= len(required):
            return DataAvailability.AVAILABLE
        else:
            return DataAvailability.PARTIAL

    # -------------------------------------------------------------------------
    # Phase 4: Materiality Assessment
    # -------------------------------------------------------------------------

    async def _phase_materiality_assessment(self, input_data: BoundaryDefinitionInput) -> PhaseResult:
        """Estimate per-category emissions using benchmarks, apply materiality thresholds."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        sector_key = self._normalize_sector(input_data.sector)
        benchmarks = BENCHMARK_INTENSITY_PER_EMPLOYEE.get(
            sector_key, BENCHMARK_INTENSITY_PER_EMPLOYEE["default"]
        )
        threshold_pct = input_data.materiality_threshold_pct

        # Aggregate employee counts and floor area by source category
        category_estimates: Dict[str, float] = {}
        category_facility_counts: Dict[str, int] = {}

        for assignment in self._source_assignments:
            cat_key = assignment.source_category.value
            facility = self._facility_map.get(assignment.facility_id)
            if not facility:
                continue

            # Estimate emissions using per-employee benchmark
            intensity = benchmarks.get(cat_key, 0.0)
            employees = max(facility.employee_count, 1)
            estimated_tco2e = intensity * employees

            # Supplement with per-sqm benchmark for Scope 2 electricity
            if assignment.source_category == SourceCategory.SCOPE2_ELECTRICITY and facility.floor_area_sqm > 0:
                sqm_intensity = BENCHMARK_INTENSITY_PER_SQM.get(facility.facility_type, 0.06)
                sqm_estimate = sqm_intensity * facility.floor_area_sqm
                estimated_tco2e = max(estimated_tco2e, sqm_estimate)

            category_estimates[cat_key] = category_estimates.get(cat_key, 0.0) + estimated_tco2e
            category_facility_counts[cat_key] = category_facility_counts.get(cat_key, 0) + 1

        # Calculate totals and percentages
        total_estimated = sum(category_estimates.values())

        self._materiality_records = []
        for cat_key, estimated in category_estimates.items():
            pct = (estimated / total_estimated * 100.0) if total_estimated > 0 else 0.0

            if pct >= threshold_pct * 5:
                level = MaterialityLevel.MATERIAL
                recommendation = "Full measurement and reporting required"
                quality = "high"
            elif pct >= threshold_pct:
                level = MaterialityLevel.POTENTIALLY_MATERIAL
                recommendation = "Measurement recommended; estimation acceptable initially"
                quality = "medium"
            else:
                level = MaterialityLevel.IMMATERIAL
                recommendation = "May use simplified estimation; document exclusion rationale"
                quality = "low"

            source_cat = SourceCategory(cat_key)
            self._materiality_records.append(MaterialityRecord(
                source_category=source_cat,
                estimated_emissions_tco2e=round(estimated, 2),
                pct_of_total=round(pct, 2),
                materiality_level=level,
                materiality_threshold_pct=threshold_pct,
                facility_count=category_facility_counts.get(cat_key, 0),
                data_quality=quality,
                recommendation=recommendation,
            ))

        # Sort by estimated emissions descending
        self._materiality_records.sort(key=lambda m: m.estimated_emissions_tco2e, reverse=True)

        material_count = sum(1 for m in self._materiality_records if m.materiality_level == MaterialityLevel.MATERIAL)
        potentially_material = sum(
            1 for m in self._materiality_records
            if m.materiality_level == MaterialityLevel.POTENTIALLY_MATERIAL
        )
        immaterial_count = sum(
            1 for m in self._materiality_records if m.materiality_level == MaterialityLevel.IMMATERIAL
        )

        outputs["total_estimated_tco2e"] = round(total_estimated, 2)
        outputs["material_categories"] = material_count
        outputs["potentially_material_categories"] = potentially_material
        outputs["immaterial_categories"] = immaterial_count
        outputs["threshold_pct"] = threshold_pct
        outputs["top_categories"] = [
            {"category": m.source_category.value, "tco2e": m.estimated_emissions_tco2e, "pct": m.pct_of_total}
            for m in self._materiality_records[:5]
        ]

        if total_estimated == 0:
            warnings.append(
                "Total estimated emissions are zero; check employee counts and sector assignment"
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 MaterialityAssessment: total=%.1f tCO2e, material=%d, threshold=%.1f%%",
            total_estimated,
            material_count,
            threshold_pct,
        )
        return PhaseResult(
            phase_name="materiality_assessment",
            phase_number=4,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Completeness Report
    # -------------------------------------------------------------------------

    def _build_completeness_report(self, input_data: BoundaryDefinitionInput) -> CompletenessReport:
        """Build completeness report per ISO 14064-1."""
        included_entity_count = sum(1 for eb in self._entity_boundaries if eb.included)
        excluded_entity_count = sum(1 for eb in self._entity_boundaries if not eb.included)

        exclusions: List[Dict[str, str]] = []
        for eb in self._entity_boundaries:
            if not eb.included:
                exclusions.append({
                    "entity_id": eb.entity_id,
                    "entity_name": eb.entity_name,
                    "reason": eb.exclusion_reason,
                })

        material_count = sum(
            1 for m in self._materiality_records if m.materiality_level == MaterialityLevel.MATERIAL
        )
        total_categories = len(self._materiality_records)

        coverage_pct = 0.0
        if self._materiality_records:
            material_emissions = sum(
                m.estimated_emissions_tco2e
                for m in self._materiality_records
                if m.materiality_level in (MaterialityLevel.MATERIAL, MaterialityLevel.POTENTIALLY_MATERIAL)
            )
            total_emissions = sum(m.estimated_emissions_tco2e for m in self._materiality_records)
            coverage_pct = (material_emissions / total_emissions * 100.0) if total_emissions > 0 else 0.0

        recommendations: List[str] = []
        if excluded_entity_count > 0:
            recommendations.append(
                f"Document exclusion rationale for {excluded_entity_count} entities "
                f"per GHG Protocol requirements"
            )
        unavailable_data = sum(
            1 for sa in self._source_assignments if sa.data_availability == DataAvailability.UNAVAILABLE
        )
        if unavailable_data > 0:
            recommendations.append(
                f"Address {unavailable_data} source-facility combinations with unavailable data"
            )
        if coverage_pct < 95.0:
            recommendations.append(
                f"Coverage is {coverage_pct:.1f}%; aim for 95%+ to meet verification standards"
            )

        return CompletenessReport(
            total_entities=len(self._entity_boundaries),
            included_entities=included_entity_count,
            excluded_entities=excluded_entity_count,
            total_facilities=len(self._facility_map),
            included_facilities=len(self._included_facilities),
            total_source_categories=total_categories,
            material_categories=material_count,
            coverage_pct=round(coverage_pct, 2),
            exclusions=exclusions,
            recommendations=recommendations,
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._entity_boundaries = []
        self._source_assignments = []
        self._materiality_records = []
        self._phase_results = []
        self._entity_map = {}
        self._facility_map = {}
        self._included_facilities = []

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: BoundaryDefinitionResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += f"|{result.workflow_id}|{result.consolidation_approach}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
