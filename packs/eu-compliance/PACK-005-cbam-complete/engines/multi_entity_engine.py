# -*- coding: utf-8 -*-
"""
MultiEntityEngine - PACK-005 CBAM Complete Engine 3

Corporate group CBAM management with multi-EORI support. Handles
entity hierarchies, consolidated obligations, cost allocation,
de minimis calculations, financial guarantees, and delegated
compliance through customs brokers.

EU Member States:
    All 27 member states with National Competent Authority (NCA)
    identifiers are included for cross-border compliance coordination.

Features:
    - Multi-level entity hierarchy (parent-subsidiary-branch)
    - Consolidated CBAM obligation across group entities
    - Group-level de minimis assessment (200 EUR / 150 kg CO2e)
    - Cost allocation by proportional, headcount, or revenue method
    - Financial guarantee tracking per entity
    - Member state NCA coordination
    - Delegated compliance through customs brokers

Zero-Hallucination:
    - All calculations use Decimal arithmetic
    - De minimis thresholds from Regulation 2023/956 Article 2
    - Cost allocation via deterministic proportional formulas
    - SHA-256 provenance hash on all results

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-005 CBAM Complete
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

# ---------------------------------------------------------------------------
# Constants: EU-27 Member States with NCA Identifiers
# ---------------------------------------------------------------------------

EU_MEMBER_STATES: Dict[str, Dict[str, str]] = {
    "AT": {"name": "Austria", "nca": "AT-UBA", "nca_name": "Umweltbundesamt"},
    "BE": {"name": "Belgium", "nca": "BE-FOD", "nca_name": "FOD Leefmilieu"},
    "BG": {"name": "Bulgaria", "nca": "BG-MOEW", "nca_name": "Ministry of Environment and Water"},
    "HR": {"name": "Croatia", "nca": "HR-MZOE", "nca_name": "Ministry of Economy and Sustainable Development"},
    "CY": {"name": "Cyprus", "nca": "CY-DLE", "nca_name": "Department of Labour Inspection"},
    "CZ": {"name": "Czech Republic", "nca": "CZ-MZP", "nca_name": "Ministry of the Environment"},
    "DK": {"name": "Denmark", "nca": "DK-DEA", "nca_name": "Danish Energy Agency"},
    "EE": {"name": "Estonia", "nca": "EE-KIK", "nca_name": "Environmental Investment Centre"},
    "FI": {"name": "Finland", "nca": "FI-ELY", "nca_name": "Energy Authority"},
    "FR": {"name": "France", "nca": "FR-DGEC", "nca_name": "Direction Generale Energie et Climat"},
    "DE": {"name": "Germany", "nca": "DE-DEHSt", "nca_name": "Deutsche Emissionshandelsstelle"},
    "GR": {"name": "Greece", "nca": "GR-YPEN", "nca_name": "Ministry of Environment and Energy"},
    "HU": {"name": "Hungary", "nca": "HU-OKF", "nca_name": "National Disaster Management Authority"},
    "IE": {"name": "Ireland", "nca": "IE-EPA", "nca_name": "Environmental Protection Agency"},
    "IT": {"name": "Italy", "nca": "IT-ISPRA", "nca_name": "Institute for Environmental Protection and Research"},
    "LV": {"name": "Latvia", "nca": "LV-VARAM", "nca_name": "Ministry of Environmental Protection"},
    "LT": {"name": "Lithuania", "nca": "LT-AAA", "nca_name": "Environmental Protection Agency"},
    "LU": {"name": "Luxembourg", "nca": "LU-AEV", "nca_name": "Administration de l'Environnement"},
    "MT": {"name": "Malta", "nca": "MT-ERA", "nca_name": "Environment and Resources Authority"},
    "NL": {"name": "Netherlands", "nca": "NL-NEA", "nca_name": "Nederlandse Emissieautoriteit"},
    "PL": {"name": "Poland", "nca": "PL-KOBIZE", "nca_name": "National Centre for Emissions Management"},
    "PT": {"name": "Portugal", "nca": "PT-APA", "nca_name": "Portuguese Environment Agency"},
    "RO": {"name": "Romania", "nca": "RO-ANPM", "nca_name": "National Agency for Environmental Protection"},
    "SK": {"name": "Slovakia", "nca": "SK-MZP", "nca_name": "Ministry of Environment"},
    "SI": {"name": "Slovenia", "nca": "SI-ARSO", "nca_name": "Slovenian Environment Agency"},
    "ES": {"name": "Spain", "nca": "ES-MITECO", "nca_name": "Ministry for Ecological Transition"},
    "SE": {"name": "Sweden", "nca": "SE-NV", "nca_name": "Swedish Environmental Protection Agency"},
}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    """Type of entity in the corporate group."""
    PARENT = "parent"
    SUBSIDIARY = "subsidiary"
    BRANCH = "branch"
    JOINT_VENTURE = "joint_venture"
    ASSOCIATE = "associate"

class CostAllocationMethod(str, Enum):
    """Cost allocation method across group entities."""
    PROPORTIONAL_EMISSIONS = "proportional_emissions"
    PROPORTIONAL_IMPORTS = "proportional_imports"
    HEADCOUNT = "headcount"
    REVENUE = "revenue"
    EQUAL = "equal"

class GuaranteeStatus(str, Enum):
    """Status of financial guarantee."""
    ACTIVE = "active"
    PENDING = "pending"
    EXPIRED = "expired"
    INSUFFICIENT = "insufficient"
    RELEASED = "released"

class ComplianceStatus(str, Enum):
    """Compliance status of an entity."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    EXEMPT = "exempt"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """Entity in a corporate group."""
    entity_id: str = Field(default_factory=_new_uuid, description="Entity identifier")
    eori: str = Field(description="EORI number")
    name: str = Field(description="Entity legal name")
    entity_type: EntityType = Field(default=EntityType.SUBSIDIARY, description="Entity type")
    member_state: str = Field(description="EU member state code (ISO 3166-1 alpha-2)")
    parent_entity_id: Optional[str] = Field(default=None, description="Parent entity identifier")
    ownership_pct: Decimal = Field(default=Decimal("100"), description="Parent ownership percentage")
    annual_imports_eur: Decimal = Field(default=Decimal("0"), description="Annual CBAM import value in EUR")
    annual_emissions_tco2e: Decimal = Field(default=Decimal("0"), description="Annual embedded emissions tCO2e")
    headcount: int = Field(default=0, description="Entity headcount")
    annual_revenue_eur: Decimal = Field(default=Decimal("0"), description="Annual revenue in EUR")
    is_authorized_declarant: bool = Field(default=False, description="Whether authorized as CBAM declarant")
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.PENDING_REVIEW, description="Current compliance status"
    )
    created_at: datetime = Field(default_factory=utcnow, description="Creation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("ownership_pct", "annual_imports_eur", "annual_emissions_tco2e",
                     "annual_revenue_eur", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class EntityGroup(BaseModel):
    """Corporate group of entities for CBAM."""
    group_id: str = Field(default_factory=_new_uuid, description="Group identifier")
    group_name: str = Field(description="Corporate group name")
    parent_entity_id: Optional[str] = Field(default=None, description="Root parent entity ID")
    entities: Dict[str, Entity] = Field(default_factory=dict, description="Entities in group keyed by ID")
    hierarchy: Dict[str, List[str]] = Field(
        default_factory=dict, description="Parent-to-children hierarchy mapping"
    )
    created_at: datetime = Field(default_factory=utcnow, description="Creation timestamp")
    config: Dict[str, Any] = Field(default_factory=dict, description="Group configuration")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class ConsolidatedObligation(BaseModel):
    """Consolidated CBAM obligation across group entities."""
    obligation_id: str = Field(default_factory=_new_uuid, description="Obligation identifier")
    group_id: str = Field(description="Source group identifier")
    period: str = Field(description="Reporting period (e.g. '2026')")
    total_imports_eur: Decimal = Field(description="Total import value across group")
    total_emissions_tco2e: Decimal = Field(description="Total embedded emissions across group")
    total_certificates_needed: Decimal = Field(description="Total certificates needed")
    entity_obligations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-entity obligation breakdown"
    )
    de_minimis_entities: List[str] = Field(
        default_factory=list, description="Entities below de minimis thresholds"
    )
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_imports_eur", "total_emissions_tco2e", "total_certificates_needed", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class GroupDeMinimisResult(BaseModel):
    """De minimis assessment for a group of entities."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    group_id: str = Field(description="Group identifier")
    year: int = Field(description="Assessment year")
    value_threshold_eur: Decimal = Field(default=Decimal("150"), description="Value threshold per consignment")
    weight_threshold_kg_co2e: Decimal = Field(
        default=Decimal("150"), description="Emissions weight threshold per consignment"
    )
    entities_assessed: int = Field(description="Number of entities assessed")
    entities_exempt: int = Field(description="Number of entities below threshold")
    entities_subject: int = Field(description="Number of entities above threshold")
    entity_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-entity de minimis results"
    )
    total_exempt_emissions: Decimal = Field(description="Total exempt emissions")
    total_subject_emissions: Decimal = Field(description="Total subject emissions")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("value_threshold_eur", "weight_threshold_kg_co2e",
                     "total_exempt_emissions", "total_subject_emissions", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class CostAllocation(BaseModel):
    """Cost allocation result across group entities."""
    allocation_id: str = Field(default_factory=_new_uuid, description="Allocation identifier")
    group_id: str = Field(description="Group identifier")
    total_cost: Decimal = Field(description="Total cost to allocate")
    method: CostAllocationMethod = Field(description="Allocation method used")
    entity_allocations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-entity cost allocations"
    )
    allocated_total: Decimal = Field(description="Sum of allocations (verification)")
    residual: Decimal = Field(default=Decimal("0"), description="Unallocated residual due to rounding")
    allocated_at: datetime = Field(default_factory=utcnow, description="Allocation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_cost", "allocated_total", "residual", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class EntityDeclaration(BaseModel):
    """CBAM declaration for a single entity."""
    declaration_id: str = Field(default_factory=_new_uuid, description="Declaration identifier")
    entity_id: str = Field(description="Entity identifier")
    eori: str = Field(description="EORI number")
    entity_name: str = Field(description="Entity name")
    member_state: str = Field(description="Member state code")
    period: str = Field(description="Reporting period")
    total_imports_eur: Decimal = Field(description="Total import value")
    total_emissions_tco2e: Decimal = Field(description="Total embedded emissions")
    certificates_required: Decimal = Field(description="Certificates required")
    allocated_cost: Decimal = Field(default=Decimal("0"), description="Allocated group cost")
    nca_identifier: str = Field(default="", description="NCA identifier for submission")
    de_minimis_exempt: bool = Field(default=False, description="Whether de minimis exempt")
    created_at: datetime = Field(default_factory=utcnow, description="Creation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_imports_eur", "total_emissions_tco2e",
                     "certificates_required", "allocated_cost", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class FinancialGuaranteeRecord(BaseModel):
    """Financial guarantee record for an entity."""
    guarantee_id: str = Field(default_factory=_new_uuid, description="Guarantee identifier")
    group_id: str = Field(description="Group identifier")
    entity_id: str = Field(description="Entity identifier")
    guarantee_amount: Decimal = Field(description="Guarantee amount in EUR")
    required_amount: Decimal = Field(description="Required guarantee amount in EUR")
    surplus_deficit: Decimal = Field(description="Surplus (positive) or deficit (negative)")
    status: GuaranteeStatus = Field(description="Guarantee status")
    valid_from: datetime = Field(default_factory=utcnow, description="Guarantee validity start")
    valid_until: Optional[datetime] = Field(default=None, description="Guarantee validity end")
    issuer: str = Field(default="", description="Guarantee issuer (bank/insurer)")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("guarantee_amount", "required_amount", "surplus_deficit", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class MemberStateCoordination(BaseModel):
    """Member state coordination status for a group."""
    coordination_id: str = Field(default_factory=_new_uuid, description="Coordination identifier")
    group_id: str = Field(description="Group identifier")
    member_states_involved: List[str] = Field(default_factory=list, description="Member state codes")
    nca_contacts: List[Dict[str, str]] = Field(default_factory=list, description="NCA contact details")
    entities_by_state: Dict[str, List[str]] = Field(
        default_factory=dict, description="Entity IDs grouped by member state"
    )
    primary_nca: str = Field(default="", description="Primary NCA for coordination")
    coordination_status: str = Field(default="active", description="Coordination status")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class DelegatedComplianceStatus(BaseModel):
    """Status of delegated compliance through a broker."""
    status_id: str = Field(default_factory=_new_uuid, description="Status identifier")
    broker_id: str = Field(description="Broker identifier")
    broker_name: str = Field(default="", description="Broker name")
    client_entities: List[str] = Field(default_factory=list, description="Client entity IDs")
    total_managed_emissions: Decimal = Field(description="Total emissions under management")
    declarations_submitted: int = Field(default=0, description="Declarations submitted")
    declarations_pending: int = Field(default=0, description="Declarations pending")
    compliance_score: Decimal = Field(default=Decimal("0"), description="Overall compliance score (0-100)")
    delegation_valid_until: Optional[datetime] = Field(default=None, description="Delegation expiry")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_managed_emissions", "compliance_score", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class HierarchyResult(BaseModel):
    """Result of setting entity hierarchy."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    group_id: str = Field(description="Group identifier")
    parent_id: str = Field(description="Parent entity identifier")
    child_ids: List[str] = Field(default_factory=list, description="Child entity identifiers")
    hierarchy_depth: int = Field(default=0, description="Maximum hierarchy depth")
    total_entities: int = Field(default=0, description="Total entities in group")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class MultiEntityConfig(BaseModel):
    """Configuration for the MultiEntityEngine."""
    de_minimis_value_eur: Decimal = Field(
        default=Decimal("150"), description="De minimis value threshold per consignment (EUR)"
    )
    de_minimis_weight_kg_co2e: Decimal = Field(
        default=Decimal("150"), description="De minimis emissions threshold (kg CO2e)"
    )
    default_cost_allocation_method: CostAllocationMethod = Field(
        default=CostAllocationMethod.PROPORTIONAL_EMISSIONS,
        description="Default cost allocation method",
    )
    guarantee_coverage_pct: Decimal = Field(
        default=Decimal("100"), description="Required guarantee coverage percentage"
    )

# ---------------------------------------------------------------------------
# Pydantic model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

MultiEntityConfig.model_rebuild()
Entity.model_rebuild()
EntityGroup.model_rebuild()
ConsolidatedObligation.model_rebuild()
GroupDeMinimisResult.model_rebuild()
CostAllocation.model_rebuild()
EntityDeclaration.model_rebuild()
FinancialGuaranteeRecord.model_rebuild()
MemberStateCoordination.model_rebuild()
DelegatedComplianceStatus.model_rebuild()
HierarchyResult.model_rebuild()

# ---------------------------------------------------------------------------
# MultiEntityEngine
# ---------------------------------------------------------------------------

class MultiEntityEngine:
    """
    Corporate group CBAM management engine.

    Manages multi-entity CBAM compliance including entity hierarchies,
    consolidated obligations, cost allocation, and member state coordination
    across all 27 EU member states.

    Attributes:
        config: Engine configuration.
        _groups: In-memory group store.
        _guarantees: In-memory guarantee store.

    Example:
        >>> engine = MultiEntityEngine()
        >>> group = engine.create_group({"group_name": "Acme Corp Group"})
        >>> entity = engine.add_entity(group.group_id, {
        ...     "eori": "DE123456789000", "name": "Acme DE", "member_state": "DE"
        ... })
        >>> obligations = engine.consolidate_obligations(group.group_id, "2026")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MultiEntityEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = MultiEntityConfig(**config)
        elif config and isinstance(config, MultiEntityConfig):
            self.config = config
        else:
            self.config = MultiEntityConfig()

        self._groups: Dict[str, EntityGroup] = {}
        self._guarantees: Dict[str, FinancialGuaranteeRecord] = {}
        logger.info("MultiEntityEngine initialized (v%s)", _MODULE_VERSION)

    # -----------------------------------------------------------------------
    # Group Management
    # -----------------------------------------------------------------------

    def create_group(self, group_config: Dict[str, Any]) -> EntityGroup:
        """Create a new entity group.

        Args:
            group_config: Group configuration including 'group_name'.

        Returns:
            Newly created EntityGroup.

        Raises:
            ValueError: If group_name is missing.
        """
        name = group_config.get("group_name", "").strip()
        if not name:
            raise ValueError("group_name is required")

        group = EntityGroup(
            group_name=name,
            config=group_config,
        )
        group.provenance_hash = _compute_hash(group)
        self._groups[group.group_id] = group

        logger.info("Created group %s: %s", group.group_id, name)
        return group

    def add_entity(
        self, group_id: str, entity: Dict[str, Any]
    ) -> Entity:
        """Add an entity to a group.

        Args:
            group_id: Target group identifier.
            entity: Entity data including 'eori', 'name', 'member_state'.

        Returns:
            Newly created Entity.

        Raises:
            ValueError: If group not found or required fields missing.
        """
        if group_id not in self._groups:
            raise ValueError(f"Group {group_id} not found")

        eori = entity.get("eori", "").strip()
        name = entity.get("name", "").strip()
        member_state = entity.get("member_state", "").strip().upper()

        if not eori:
            raise ValueError("EORI is required")
        if not name:
            raise ValueError("Entity name is required")
        if member_state not in EU_MEMBER_STATES:
            raise ValueError(f"Invalid member state: {member_state}. Must be one of {list(EU_MEMBER_STATES.keys())}")

        new_entity = Entity(
            eori=eori,
            name=name,
            entity_type=EntityType(entity.get("entity_type", "subsidiary")),
            member_state=member_state,
            parent_entity_id=entity.get("parent_entity_id"),
            ownership_pct=_decimal(entity.get("ownership_pct", 100)),
            annual_imports_eur=_decimal(entity.get("annual_imports_eur", 0)),
            annual_emissions_tco2e=_decimal(entity.get("annual_emissions_tco2e", 0)),
            headcount=entity.get("headcount", 0),
            annual_revenue_eur=_decimal(entity.get("annual_revenue_eur", 0)),
            is_authorized_declarant=entity.get("is_authorized_declarant", False),
        )
        new_entity.provenance_hash = _compute_hash(new_entity)

        group = self._groups[group_id]
        group.entities[new_entity.entity_id] = new_entity

        if new_entity.entity_type == EntityType.PARENT and group.parent_entity_id is None:
            group.parent_entity_id = new_entity.entity_id

        group.provenance_hash = _compute_hash(group)

        logger.info(
            "Added entity %s (%s) to group %s, member_state=%s",
            new_entity.entity_id, name, group_id, member_state,
        )
        return new_entity

    # -----------------------------------------------------------------------
    # Hierarchy
    # -----------------------------------------------------------------------

    def set_hierarchy(
        self, group_id: str, parent_id: str, child_ids: List[str]
    ) -> HierarchyResult:
        """Set parent-child hierarchy within a group.

        Args:
            group_id: Group identifier.
            parent_id: Parent entity identifier.
            child_ids: List of child entity identifiers.

        Returns:
            HierarchyResult with hierarchy metadata.

        Raises:
            ValueError: If group or entities not found.
        """
        if group_id not in self._groups:
            raise ValueError(f"Group {group_id} not found")

        group = self._groups[group_id]

        if parent_id not in group.entities:
            raise ValueError(f"Parent entity {parent_id} not in group {group_id}")

        for cid in child_ids:
            if cid not in group.entities:
                raise ValueError(f"Child entity {cid} not in group {group_id}")
            group.entities[cid].parent_entity_id = parent_id

        group.hierarchy[parent_id] = child_ids
        group.provenance_hash = _compute_hash(group)

        depth = self._calculate_hierarchy_depth(group, parent_id)

        result = HierarchyResult(
            group_id=group_id,
            parent_id=parent_id,
            child_ids=child_ids,
            hierarchy_depth=depth,
            total_entities=len(group.entities),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Set hierarchy in group %s: parent=%s, children=%d, depth=%d",
            group_id, parent_id, len(child_ids), depth,
        )
        return result

    # -----------------------------------------------------------------------
    # Consolidated Obligations
    # -----------------------------------------------------------------------

    def consolidate_obligations(
        self, group_id: str, period: str
    ) -> ConsolidatedObligation:
        """Consolidate CBAM obligations across all entities in a group.

        Aggregates imports, emissions, and certificate requirements,
        accounting for ownership percentages and de minimis exemptions.

        Args:
            group_id: Group identifier.
            period: Reporting period string.

        Returns:
            ConsolidatedObligation with group-wide totals.

        Raises:
            ValueError: If group not found.
        """
        if group_id not in self._groups:
            raise ValueError(f"Group {group_id} not found")

        group = self._groups[group_id]
        total_imports = Decimal("0")
        total_emissions = Decimal("0")
        entity_obligations: List[Dict[str, Any]] = []
        de_minimis_entities: List[str] = []

        for eid, entity in group.entities.items():
            ownership_factor = (entity.ownership_pct / Decimal("100")).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )
            adj_imports = entity.annual_imports_eur * ownership_factor
            adj_emissions = entity.annual_emissions_tco2e * ownership_factor

            is_exempt = (
                adj_imports < self.config.de_minimis_value_eur
                and adj_emissions * Decimal("1000") < self.config.de_minimis_weight_kg_co2e
            )

            if is_exempt:
                de_minimis_entities.append(eid)
            else:
                total_imports += adj_imports
                total_emissions += adj_emissions

            entity_obligations.append({
                "entity_id": eid,
                "eori": entity.eori,
                "name": entity.name,
                "member_state": entity.member_state,
                "ownership_pct": str(entity.ownership_pct),
                "adjusted_imports_eur": str(adj_imports.quantize(Decimal("0.01"))),
                "adjusted_emissions_tco2e": str(adj_emissions.quantize(Decimal("0.001"))),
                "de_minimis_exempt": is_exempt,
                "certificates_required": str(adj_emissions.quantize(Decimal("0.001"))) if not is_exempt else "0",
            })

        result = ConsolidatedObligation(
            group_id=group_id,
            period=period,
            total_imports_eur=total_imports,
            total_emissions_tco2e=total_emissions,
            total_certificates_needed=total_emissions,
            entity_obligations=entity_obligations,
            de_minimis_entities=de_minimis_entities,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Consolidated obligations for group %s period %s: EUR %s, %s tCO2e, %d exempt",
            group_id, period, total_imports, total_emissions, len(de_minimis_entities),
        )
        return result

    # -----------------------------------------------------------------------
    # De Minimis Assessment
    # -----------------------------------------------------------------------

    def calculate_group_deminimis(
        self, group_id: str, year: int
    ) -> GroupDeMinimisResult:
        """Calculate de minimis exemption eligibility for each entity.

        Per Article 2(3), CBAM does not apply to goods with intrinsic value
        not exceeding EUR 150 per consignment, or where embedded emissions
        are below the weight threshold.

        Args:
            group_id: Group identifier.
            year: Assessment year.

        Returns:
            GroupDeMinimisResult with per-entity assessment.

        Raises:
            ValueError: If group not found.
        """
        if group_id not in self._groups:
            raise ValueError(f"Group {group_id} not found")

        group = self._groups[group_id]
        entity_results: List[Dict[str, Any]] = []
        exempt_count = 0
        subject_count = 0
        total_exempt_emissions = Decimal("0")
        total_subject_emissions = Decimal("0")

        for eid, entity in group.entities.items():
            below_value = entity.annual_imports_eur < self.config.de_minimis_value_eur
            below_weight = (
                entity.annual_emissions_tco2e * Decimal("1000") < self.config.de_minimis_weight_kg_co2e
            )
            is_exempt = below_value and below_weight

            if is_exempt:
                exempt_count += 1
                total_exempt_emissions += entity.annual_emissions_tco2e
            else:
                subject_count += 1
                total_subject_emissions += entity.annual_emissions_tco2e

            entity_results.append({
                "entity_id": eid,
                "eori": entity.eori,
                "name": entity.name,
                "annual_imports_eur": str(entity.annual_imports_eur),
                "annual_emissions_tco2e": str(entity.annual_emissions_tco2e),
                "below_value_threshold": below_value,
                "below_weight_threshold": below_weight,
                "exempt": is_exempt,
            })

        result = GroupDeMinimisResult(
            group_id=group_id,
            year=year,
            entities_assessed=len(group.entities),
            entities_exempt=exempt_count,
            entities_subject=subject_count,
            entity_results=entity_results,
            total_exempt_emissions=total_exempt_emissions,
            total_subject_emissions=total_subject_emissions,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "De minimis assessment for group %s year %d: %d exempt, %d subject",
            group_id, year, exempt_count, subject_count,
        )
        return result

    # -----------------------------------------------------------------------
    # Cost Allocation
    # -----------------------------------------------------------------------

    def allocate_costs(
        self, group_id: str, total_cost: Decimal, method: Optional[str] = None
    ) -> CostAllocation:
        """Allocate group CBAM costs across entities.

        Args:
            group_id: Group identifier.
            total_cost: Total cost to allocate in EUR.
            method: Allocation method. Defaults to config default.

        Returns:
            CostAllocation with per-entity breakdown.

        Raises:
            ValueError: If group not found or invalid method.
        """
        if group_id not in self._groups:
            raise ValueError(f"Group {group_id} not found")

        total_cost = _decimal(total_cost)
        if total_cost < Decimal("0"):
            raise ValueError("Total cost must not be negative")

        if method:
            try:
                alloc_method = CostAllocationMethod(method)
            except ValueError:
                raise ValueError(f"Invalid allocation method: {method}")
        else:
            alloc_method = self.config.default_cost_allocation_method

        group = self._groups[group_id]
        entities = list(group.entities.values())

        total_weight = Decimal("0")
        weights: Dict[str, Decimal] = {}

        for entity in entities:
            if alloc_method == CostAllocationMethod.PROPORTIONAL_EMISSIONS:
                w = entity.annual_emissions_tco2e
            elif alloc_method == CostAllocationMethod.PROPORTIONAL_IMPORTS:
                w = entity.annual_imports_eur
            elif alloc_method == CostAllocationMethod.HEADCOUNT:
                w = _decimal(entity.headcount)
            elif alloc_method == CostAllocationMethod.REVENUE:
                w = entity.annual_revenue_eur
            else:
                w = Decimal("1")
            weights[entity.entity_id] = w
            total_weight += w

        allocations: List[Dict[str, Any]] = []
        allocated_sum = Decimal("0")

        for entity in entities:
            fraction = (weights[entity.entity_id] / total_weight).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            ) if total_weight > 0 else Decimal("0")
            amount = (total_cost * fraction).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            allocated_sum += amount

            allocations.append({
                "entity_id": entity.entity_id,
                "eori": entity.eori,
                "name": entity.name,
                "weight": str(weights[entity.entity_id]),
                "fraction": str(fraction),
                "allocated_amount_eur": str(amount),
            })

        residual = total_cost - allocated_sum

        result = CostAllocation(
            group_id=group_id,
            total_cost=total_cost,
            method=alloc_method,
            entity_allocations=allocations,
            allocated_total=allocated_sum,
            residual=residual,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Allocated EUR %s across %d entities using %s, residual=%s",
            total_cost, len(entities), alloc_method.value, residual,
        )
        return result

    # -----------------------------------------------------------------------
    # Entity Declarations
    # -----------------------------------------------------------------------

    def generate_entity_declarations(
        self, group_id: str, period: str
    ) -> List[EntityDeclaration]:
        """Generate individual CBAM declarations for each entity in a group.

        Args:
            group_id: Group identifier.
            period: Reporting period.

        Returns:
            List of EntityDeclaration objects, one per non-exempt entity.

        Raises:
            ValueError: If group not found.
        """
        if group_id not in self._groups:
            raise ValueError(f"Group {group_id} not found")

        group = self._groups[group_id]
        declarations: List[EntityDeclaration] = []

        for eid, entity in group.entities.items():
            ms_info = EU_MEMBER_STATES.get(entity.member_state, {})
            nca_id = ms_info.get("nca", "")

            is_exempt = (
                entity.annual_imports_eur < self.config.de_minimis_value_eur
                and entity.annual_emissions_tco2e * Decimal("1000") < self.config.de_minimis_weight_kg_co2e
            )

            decl = EntityDeclaration(
                entity_id=eid,
                eori=entity.eori,
                entity_name=entity.name,
                member_state=entity.member_state,
                period=period,
                total_imports_eur=entity.annual_imports_eur,
                total_emissions_tco2e=entity.annual_emissions_tco2e,
                certificates_required=entity.annual_emissions_tco2e if not is_exempt else Decimal("0"),
                nca_identifier=nca_id,
                de_minimis_exempt=is_exempt,
            )
            decl.provenance_hash = _compute_hash(decl)
            declarations.append(decl)

        logger.info(
            "Generated %d declarations for group %s period %s",
            len(declarations), group_id, period,
        )
        return declarations

    # -----------------------------------------------------------------------
    # Financial Guarantee
    # -----------------------------------------------------------------------

    def manage_financial_guarantee(
        self, group_id: str, guarantee: Dict[str, Any]
    ) -> FinancialGuaranteeRecord:
        """Manage financial guarantee for an entity in the group.

        Args:
            group_id: Group identifier.
            guarantee: Guarantee data including 'entity_id', 'guarantee_amount',
                'required_amount', 'issuer'.

        Returns:
            FinancialGuaranteeRecord with status.

        Raises:
            ValueError: If group or entity not found.
        """
        if group_id not in self._groups:
            raise ValueError(f"Group {group_id} not found")

        entity_id = guarantee.get("entity_id", "")
        if entity_id not in self._groups[group_id].entities:
            raise ValueError(f"Entity {entity_id} not in group {group_id}")

        amount = _decimal(guarantee.get("guarantee_amount", 0))
        required = _decimal(guarantee.get("required_amount", 0))
        surplus = amount - required

        if amount <= Decimal("0"):
            status = GuaranteeStatus.PENDING
        elif surplus < Decimal("0"):
            status = GuaranteeStatus.INSUFFICIENT
        else:
            status = GuaranteeStatus.ACTIVE

        record = FinancialGuaranteeRecord(
            group_id=group_id,
            entity_id=entity_id,
            guarantee_amount=amount,
            required_amount=required,
            surplus_deficit=surplus,
            status=status,
            issuer=guarantee.get("issuer", ""),
        )
        record.provenance_hash = _compute_hash(record)
        self._guarantees[record.guarantee_id] = record

        logger.info(
            "Financial guarantee for entity %s: amount=%s, required=%s, status=%s",
            entity_id, amount, required, status.value,
        )
        return record

    # -----------------------------------------------------------------------
    # Member State Coordination
    # -----------------------------------------------------------------------

    def coordinate_member_states(
        self, group_id: str
    ) -> MemberStateCoordination:
        """Coordinate CBAM compliance across member states for a group.

        Identifies all member states involved, maps entities to NCAs,
        and determines the primary NCA for coordination.

        Args:
            group_id: Group identifier.

        Returns:
            MemberStateCoordination with cross-border details.

        Raises:
            ValueError: If group not found.
        """
        if group_id not in self._groups:
            raise ValueError(f"Group {group_id} not found")

        group = self._groups[group_id]
        states_involved: set = set()
        entities_by_state: Dict[str, List[str]] = defaultdict(list)
        nca_contacts: List[Dict[str, str]] = []

        for eid, entity in group.entities.items():
            ms = entity.member_state
            states_involved.add(ms)
            entities_by_state[ms].append(eid)

        for ms in sorted(states_involved):
            ms_info = EU_MEMBER_STATES.get(ms, {})
            nca_contacts.append({
                "member_state": ms,
                "country_name": ms_info.get("name", ""),
                "nca_identifier": ms_info.get("nca", ""),
                "nca_name": ms_info.get("nca_name", ""),
                "entity_count": str(len(entities_by_state[ms])),
            })

        primary_nca = ""
        max_entities = 0
        for ms, eids in entities_by_state.items():
            if len(eids) > max_entities:
                max_entities = len(eids)
                primary_nca = EU_MEMBER_STATES.get(ms, {}).get("nca", "")

        result = MemberStateCoordination(
            group_id=group_id,
            member_states_involved=sorted(states_involved),
            nca_contacts=nca_contacts,
            entities_by_state=dict(entities_by_state),
            primary_nca=primary_nca,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Member state coordination for group %s: %d states, primary NCA=%s",
            group_id, len(states_involved), primary_nca,
        )
        return result

    # -----------------------------------------------------------------------
    # Delegated Compliance
    # -----------------------------------------------------------------------

    def track_delegated_compliance(
        self, broker_id: str, client_entities: List[Dict[str, Any]]
    ) -> DelegatedComplianceStatus:
        """Track delegated compliance status through a customs broker.

        Args:
            broker_id: Customs broker/indirect representative identifier.
            client_entities: List of client entity dicts with 'entity_id',
                'emissions_tco2e', 'declarations_submitted', 'declarations_pending'.

        Returns:
            DelegatedComplianceStatus with aggregate metrics.
        """
        entity_ids: List[str] = []
        total_emissions = Decimal("0")
        total_submitted = 0
        total_pending = 0

        for client in client_entities:
            entity_ids.append(client.get("entity_id", ""))
            total_emissions += _decimal(client.get("emissions_tco2e", 0))
            total_submitted += client.get("declarations_submitted", 0)
            total_pending += client.get("declarations_pending", 0)

        total_declarations = total_submitted + total_pending
        compliance_score = Decimal("0")
        if total_declarations > 0:
            compliance_score = (
                _decimal(total_submitted) / _decimal(total_declarations) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        result = DelegatedComplianceStatus(
            broker_id=broker_id,
            client_entities=entity_ids,
            total_managed_emissions=total_emissions,
            declarations_submitted=total_submitted,
            declarations_pending=total_pending,
            compliance_score=compliance_score,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Delegated compliance for broker %s: %d clients, %s tCO2e, score=%s%%",
            broker_id, len(entity_ids), total_emissions, compliance_score,
        )
        return result

    # -----------------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------------

    def _calculate_hierarchy_depth(
        self, group: EntityGroup, node_id: str, current_depth: int = 0
    ) -> int:
        """Calculate maximum hierarchy depth from a node.

        Args:
            group: Entity group.
            node_id: Starting node.
            current_depth: Current recursion depth.

        Returns:
            Maximum depth from this node.
        """
        children = group.hierarchy.get(node_id, [])
        if not children:
            return current_depth
        return max(
            self._calculate_hierarchy_depth(group, cid, current_depth + 1)
            for cid in children
        )
