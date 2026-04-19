# -*- coding: utf-8 -*-
"""
CSDDDSetupWizard - Guided Configuration for PACK-019 CSDDD Readiness
========================================================================

Interactive and programmatic setup wizard that guides users through initial
CSDDD pack configuration. Detects sector from NACE codes, estimates CSDDD
scope applicability, suggests sector-appropriate presets, validates
configuration inputs, and generates implementation plans.

Legal References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - Article 2: Scope and company thresholds
    - Article 22: Climate transition plan requirements
    - Article 30: Transposition timeline (Group 1: 2027, Group 2: 2029)

Configuration Dimensions:
    - Sector classification (NACE code based)
    - Company size and scope determination
    - Value chain structure (upstream/downstream depth)
    - Priority adverse impact categories
    - Compliance timeline based on group classification

Sector Presets:
    - manufacturing     -- Industrial, process, chemicals (NACE C)
    - extractive        -- Mining, oil & gas (NACE B)
    - financial         -- Banks, insurance, asset management (NACE K)
    - agriculture       -- Farming, forestry, fishing (NACE A)
    - textiles          -- Fashion, garments, footwear (NACE C13-C15)
    - technology        -- IT, software, electronics (NACE J, C26)
    - multi_sector      -- Conglomerates and diversified businesses

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-019 CSDDD Readiness Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """Compute SHA-256 hash for provenance tracking."""
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

class SetupStatus(str, Enum):
    """Setup wizard execution status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class SectorType(str, Enum):
    """Business sector types for CSDDD configuration."""

    MANUFACTURING = "manufacturing"
    EXTRACTIVE = "extractive"
    FINANCIAL = "financial"
    AGRICULTURE = "agriculture"
    TEXTILES = "textiles"
    TECHNOLOGY = "technology"
    MULTI_SECTOR = "multi_sector"
    UNKNOWN = "unknown"

class CompanyGroup(str, Enum):
    """CSDDD company group classification per Article 2."""

    GROUP_1 = "group_1"
    GROUP_2 = "group_2"
    FRANCHISE = "franchise"
    OUT_OF_SCOPE = "out_of_scope"

class ValueChainDepth(str, Enum):
    """Value chain mapping depth configuration."""

    TIER_1_ONLY = "tier_1_only"
    TIER_1_AND_2 = "tier_1_and_2"
    FULL_UPSTREAM = "full_upstream"
    FULL_UPSTREAM_DOWNSTREAM = "full_upstream_downstream"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class WizardConfig(BaseModel):
    """Configuration for the setup wizard."""

    interactive_mode: bool = Field(default=False)
    auto_detect_sector: bool = Field(default=True)
    generate_output_file: bool = Field(default=True)
    output_dir: Optional[str] = Field(None)

class CompanyProfile(BaseModel):
    """Company profile for CSDDD scope determination."""

    company_name: str = Field(default="")
    nace_code: str = Field(default="")
    employee_count: int = Field(default=0, ge=0)
    net_turnover_eur: float = Field(default=0.0, ge=0.0)
    is_eu_incorporated: bool = Field(default=True)
    has_eu_operations: bool = Field(default=True)
    is_franchise_or_licensee: bool = Field(default=False)
    royalty_revenue_eur: float = Field(default=0.0, ge=0.0)
    countries_of_operation: List[str] = Field(default_factory=list)
    sector: SectorType = Field(default=SectorType.UNKNOWN)

    @field_validator("nace_code")
    @classmethod
    def validate_nace_code(cls, v: str) -> str:
        """Validate NACE code format."""
        if not v:
            return v
        if v[0].upper() not in "ABCDEFGHIJKLMN":
            raise ValueError(f"Invalid NACE division: {v[0]}")
        return v.upper()

class CSDDDConfiguration(BaseModel):
    """Generated CSDDD readiness configuration."""

    config_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-019")
    company_name: str = Field(default="")
    sector: SectorType = Field(default=SectorType.UNKNOWN)
    company_group: CompanyGroup = Field(default=CompanyGroup.OUT_OF_SCOPE)
    compliance_deadline: Optional[int] = Field(None)
    value_chain_depth: ValueChainDepth = Field(default=ValueChainDepth.TIER_1_ONLY)
    priority_impact_categories: List[str] = Field(default_factory=list)
    climate_transition_required: bool = Field(default=False)
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    preset_name: str = Field(default="")
    engines_enabled: List[str] = Field(default_factory=list)
    bridges_enabled: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class ImplementationPhase(BaseModel):
    """A phase in the CSDDD implementation plan."""

    phase_number: int = Field(default=0)
    phase_name: str = Field(default="")
    description: str = Field(default="")
    duration_months: int = Field(default=0)
    csddd_articles: List[str] = Field(default_factory=list)
    deliverables: List[str] = Field(default_factory=list)

class ImplementationPlan(BaseModel):
    """Generated CSDDD implementation plan."""

    plan_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field(default="")
    company_group: CompanyGroup = Field(default=CompanyGroup.OUT_OF_SCOPE)
    compliance_deadline: Optional[int] = Field(None)
    total_duration_months: int = Field(default=0)
    phases: List[ImplementationPhase] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class SetupResult(BaseModel):
    """Result of setup wizard execution."""

    wizard_id: str = Field(default_factory=_new_uuid)
    status: SetupStatus = Field(default=SetupStatus.NOT_STARTED)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    configuration: Optional[CSDDDConfiguration] = Field(None)
    implementation_plan: Optional[ImplementationPlan] = Field(None)
    validation_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# NACE Code Mapping
# ---------------------------------------------------------------------------

NACE_DIVISIONS: Dict[str, Dict[str, Any]] = {
    "A": {"description": "Agriculture, forestry and fishing", "sector": SectorType.AGRICULTURE},
    "B": {"description": "Mining and quarrying", "sector": SectorType.EXTRACTIVE},
    "C": {"description": "Manufacturing", "sector": SectorType.MANUFACTURING},
    "D": {"description": "Electricity, gas, steam supply", "sector": SectorType.EXTRACTIVE},
    "E": {"description": "Water supply, waste management", "sector": SectorType.MULTI_SECTOR},
    "F": {"description": "Construction", "sector": SectorType.MANUFACTURING},
    "G": {"description": "Wholesale and retail trade", "sector": SectorType.MULTI_SECTOR},
    "H": {"description": "Transportation and storage", "sector": SectorType.MULTI_SECTOR},
    "I": {"description": "Accommodation and food service", "sector": SectorType.MULTI_SECTOR},
    "J": {"description": "Information and communication", "sector": SectorType.TECHNOLOGY},
    "K": {"description": "Financial and insurance activities", "sector": SectorType.FINANCIAL},
    "L": {"description": "Real estate activities", "sector": SectorType.MULTI_SECTOR},
    "M": {"description": "Professional and technical", "sector": SectorType.TECHNOLOGY},
    "N": {"description": "Administrative and support", "sector": SectorType.MULTI_SECTOR},
}

# High-risk sectors per CSDDD Annex
HIGH_RISK_SECTORS: List[SectorType] = [
    SectorType.TEXTILES,
    SectorType.AGRICULTURE,
    SectorType.EXTRACTIVE,
    SectorType.MANUFACTURING,
]

# Sector-specific impact category priorities
SECTOR_IMPACT_PRIORITIES: Dict[SectorType, List[str]] = {
    SectorType.MANUFACTURING: [
        "environmental_pollution", "occupational_health_safety",
        "labour_rights", "supply_chain_human_rights",
    ],
    SectorType.EXTRACTIVE: [
        "environmental_degradation", "indigenous_rights",
        "community_displacement", "water_pollution", "corruption",
    ],
    SectorType.FINANCIAL: [
        "enabling_adverse_impacts", "money_laundering",
        "corruption", "data_privacy",
    ],
    SectorType.AGRICULTURE: [
        "deforestation", "forced_labour", "child_labour",
        "land_rights", "water_scarcity",
    ],
    SectorType.TEXTILES: [
        "forced_labour", "child_labour", "living_wage",
        "environmental_pollution", "gender_discrimination",
    ],
    SectorType.TECHNOLOGY: [
        "conflict_minerals", "data_privacy",
        "electronic_waste", "supply_chain_labour",
    ],
    SectorType.MULTI_SECTOR: [
        "human_rights", "environmental", "governance", "labour_rights",
    ],
}

# Sector preset configurations
SECTOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "manufacturing": {
        "value_chain_depth": ValueChainDepth.TIER_1_AND_2,
        "priority_categories": SECTOR_IMPACT_PRIORITIES[SectorType.MANUFACTURING],
        "engines_enabled": [
            "scope_engine", "impact_engine", "prevention_engine",
            "grievance_engine", "climate_engine", "liability_engine",
            "scorecard_engine", "reporting_engine",
        ],
        "bridges_enabled": [
            "csrd_pack_bridge", "mrv_bridge", "supply_chain_bridge",
            "data_bridge", "taxonomy_bridge",
        ],
    },
    "extractive": {
        "value_chain_depth": ValueChainDepth.FULL_UPSTREAM,
        "priority_categories": SECTOR_IMPACT_PRIORITIES[SectorType.EXTRACTIVE],
        "engines_enabled": [
            "scope_engine", "impact_engine", "prevention_engine",
            "grievance_engine", "climate_engine", "liability_engine",
            "scorecard_engine", "reporting_engine",
        ],
        "bridges_enabled": [
            "csrd_pack_bridge", "mrv_bridge", "eudr_bridge",
            "supply_chain_bridge", "data_bridge", "taxonomy_bridge",
        ],
    },
    "financial": {
        "value_chain_depth": ValueChainDepth.TIER_1_ONLY,
        "priority_categories": SECTOR_IMPACT_PRIORITIES[SectorType.FINANCIAL],
        "engines_enabled": [
            "scope_engine", "impact_engine", "prevention_engine",
            "grievance_engine", "liability_engine",
            "scorecard_engine", "reporting_engine",
        ],
        "bridges_enabled": [
            "csrd_pack_bridge", "supply_chain_bridge",
            "data_bridge", "green_claims_bridge",
        ],
    },
    "agriculture": {
        "value_chain_depth": ValueChainDepth.FULL_UPSTREAM,
        "priority_categories": SECTOR_IMPACT_PRIORITIES[SectorType.AGRICULTURE],
        "engines_enabled": [
            "scope_engine", "impact_engine", "prevention_engine",
            "grievance_engine", "climate_engine", "liability_engine",
            "scorecard_engine", "reporting_engine",
        ],
        "bridges_enabled": [
            "csrd_pack_bridge", "mrv_bridge", "eudr_bridge",
            "supply_chain_bridge", "data_bridge",
            "green_claims_bridge", "taxonomy_bridge",
        ],
    },
    "textiles": {
        "value_chain_depth": ValueChainDepth.FULL_UPSTREAM_DOWNSTREAM,
        "priority_categories": SECTOR_IMPACT_PRIORITIES[SectorType.TEXTILES],
        "engines_enabled": [
            "scope_engine", "impact_engine", "prevention_engine",
            "grievance_engine", "climate_engine", "liability_engine",
            "scorecard_engine", "reporting_engine",
        ],
        "bridges_enabled": [
            "csrd_pack_bridge", "mrv_bridge", "supply_chain_bridge",
            "data_bridge", "green_claims_bridge",
        ],
    },
    "technology": {
        "value_chain_depth": ValueChainDepth.TIER_1_AND_2,
        "priority_categories": SECTOR_IMPACT_PRIORITIES[SectorType.TECHNOLOGY],
        "engines_enabled": [
            "scope_engine", "impact_engine", "prevention_engine",
            "grievance_engine", "climate_engine",
            "scorecard_engine", "reporting_engine",
        ],
        "bridges_enabled": [
            "csrd_pack_bridge", "mrv_bridge", "supply_chain_bridge",
            "data_bridge", "taxonomy_bridge",
        ],
    },
    "multi_sector": {
        "value_chain_depth": ValueChainDepth.TIER_1_AND_2,
        "priority_categories": SECTOR_IMPACT_PRIORITIES[SectorType.MULTI_SECTOR],
        "engines_enabled": [
            "scope_engine", "impact_engine", "prevention_engine",
            "grievance_engine", "climate_engine", "liability_engine",
            "scorecard_engine", "reporting_engine",
        ],
        "bridges_enabled": [
            "csrd_pack_bridge", "mrv_bridge", "eudr_bridge",
            "supply_chain_bridge", "data_bridge",
            "green_claims_bridge", "taxonomy_bridge",
        ],
    },
}

# ---------------------------------------------------------------------------
# CSDDDSetupWizard
# ---------------------------------------------------------------------------

class CSDDDSetupWizard:
    """Guided configuration wizard for PACK-019 CSDDD Readiness Pack.

    Guides users through sector selection, scope determination, value chain
    structure, and impact priority configuration. Supports both interactive
    and programmatic usage.

    Attributes:
        config: Wizard configuration.
        base_path: Pack base directory path.

    Example:
        >>> wizard = CSDDDSetupWizard()
        >>> params = {"company_name": "Acme", "nace_code": "C",
        ...           "employee_count": 1500, "net_turnover_eur": 500_000_000}
        >>> result = wizard.create_configuration(params)
        >>> assert result.status == SetupStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[WizardConfig] = None,
        base_path: Optional[Path] = None,
    ) -> None:
        """Initialize CSDDDSetupWizard."""
        self.config = config or WizardConfig()
        self.base_path = base_path or Path(__file__).parent.parent
        logger.info(
            "CSDDDSetupWizard initialized (base_path=%s)", self.base_path
        )

    def create_configuration(
        self,
        params: Dict[str, Any],
    ) -> SetupResult:
        """Create a CSDDD configuration from provided parameters.

        Args:
            params: Dict with keys: company_name, nace_code, employee_count,
                    net_turnover_eur, is_eu_incorporated, has_eu_operations,
                    is_franchise_or_licensee, royalty_revenue_eur,
                    countries_of_operation, reporting_year.

        Returns:
            SetupResult with generated configuration and implementation plan.
        """
        result = SetupResult(
            started_at=utcnow(),
            status=SetupStatus.IN_PROGRESS,
        )

        try:
            # Build company profile
            profile = CompanyProfile(
                company_name=params.get("company_name", ""),
                nace_code=params.get("nace_code", ""),
                employee_count=params.get("employee_count", 0),
                net_turnover_eur=params.get("net_turnover_eur", 0.0),
                is_eu_incorporated=params.get("is_eu_incorporated", True),
                has_eu_operations=params.get("has_eu_operations", True),
                is_franchise_or_licensee=params.get("is_franchise_or_licensee", False),
                royalty_revenue_eur=params.get("royalty_revenue_eur", 0.0),
                countries_of_operation=params.get("countries_of_operation", []),
            )

            # Detect sector
            if self.config.auto_detect_sector and profile.nace_code:
                profile.sector = self._detect_sector(profile.nace_code)

            # Estimate scope
            scope = self.estimate_scope(profile)
            company_group = CompanyGroup(scope.get("company_group", "out_of_scope"))

            # Get sector defaults
            sector_name = profile.sector.value
            preset = self.get_preset(sector_name)

            # Build configuration
            configuration = CSDDDConfiguration(
                company_name=profile.company_name,
                sector=profile.sector,
                company_group=company_group,
                compliance_deadline=scope.get("compliance_deadline"),
                value_chain_depth=ValueChainDepth(
                    preset.get("value_chain_depth", ValueChainDepth.TIER_1_ONLY.value)
                ),
                priority_impact_categories=preset.get("priority_categories", []),
                climate_transition_required=company_group != CompanyGroup.OUT_OF_SCOPE,
                reporting_year=params.get("reporting_year", 2025),
                preset_name=sector_name,
                engines_enabled=preset.get("engines_enabled", []),
                bridges_enabled=preset.get("bridges_enabled", []),
            )

            # Validate
            validation_errors = self.validate_configuration(configuration)
            result.validation_errors = validation_errors

            if validation_errors:
                result.warnings.append(
                    f"Configuration has {len(validation_errors)} validation issues"
                )

            configuration.provenance_hash = _compute_hash(configuration)
            result.configuration = configuration

            # Generate implementation plan
            impl_plan = self.generate_implementation_plan(configuration)
            result.implementation_plan = impl_plan

            result.status = SetupStatus.COMPLETED

            logger.info(
                "Configuration created for %s (sector=%s, group=%s)",
                profile.company_name,
                profile.sector.value,
                company_group.value,
            )

        except Exception as exc:
            result.status = SetupStatus.FAILED
            result.validation_errors.append(str(exc))
            logger.error("Setup wizard failed: %s", str(exc), exc_info=True)

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                utcnow() - result.started_at
            ).total_seconds() * 1000
        result.provenance_hash = _compute_hash(result)
        return result

    def validate_configuration(
        self,
        config: CSDDDConfiguration,
    ) -> List[str]:
        """Validate a CSDDD configuration.

        Args:
            config: Configuration to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        if not config.company_name:
            errors.append("Company name is required")

        if config.company_group == CompanyGroup.OUT_OF_SCOPE:
            errors.append(
                "Company appears out of CSDDD scope; verify thresholds"
            )

        if not config.engines_enabled:
            errors.append("No engines enabled; at least scope_engine is required")

        if not config.priority_impact_categories:
            errors.append("No priority impact categories configured")

        if config.climate_transition_required and "climate_engine" not in config.engines_enabled:
            errors.append(
                "Climate transition plan required but climate_engine not enabled"
            )

        if config.reporting_year < 2025:
            errors.append(
                f"Reporting year {config.reporting_year} is before CSDDD application"
            )

        logger.info("Configuration validation: %d errors", len(errors))
        return errors

    def get_sector_defaults(self, sector: str) -> Dict[str, Any]:
        """Get sector-specific default configuration.

        Args:
            sector: Sector name string.

        Returns:
            Dict with sector defaults including impact priorities and depth.
        """
        try:
            sector_type = SectorType(sector)
        except ValueError:
            sector_type = SectorType.MULTI_SECTOR

        nace_info = None
        for code, info in NACE_DIVISIONS.items():
            if info["sector"] == sector_type:
                nace_info = info
                break

        return {
            "sector": sector_type.value,
            "sector_description": nace_info["description"] if nace_info else "",
            "is_high_risk": sector_type in HIGH_RISK_SECTORS,
            "priority_impact_categories": SECTOR_IMPACT_PRIORITIES.get(
                sector_type,
                SECTOR_IMPACT_PRIORITIES[SectorType.MULTI_SECTOR],
            ),
            "suggested_value_chain_depth": (
                ValueChainDepth.FULL_UPSTREAM.value
                if sector_type in HIGH_RISK_SECTORS
                else ValueChainDepth.TIER_1_AND_2.value
            ),
            "preset_available": sector in SECTOR_PRESETS,
        }

    def estimate_scope(self, profile: CompanyProfile) -> Dict[str, Any]:
        """Estimate CSDDD scope applicability for a company profile.

        Deterministic scope rules per Article 2 (zero-hallucination).

        Args:
            profile: Company profile with employee count and turnover.

        Returns:
            Dict with company_group, in_scope, compliance_deadline.
        """
        employees = profile.employee_count
        turnover = profile.net_turnover_eur
        is_franchise = profile.is_franchise_or_licensee
        royalty_revenue = profile.royalty_revenue_eur

        if employees > 1000 and turnover > 450_000_000:
            group = CompanyGroup.GROUP_1.value
            deadline = 2027
        elif employees > 500 and turnover > 150_000_000:
            group = CompanyGroup.GROUP_2.value
            deadline = 2029
        elif is_franchise and royalty_revenue > 80_000_000:
            group = CompanyGroup.FRANCHISE.value
            deadline = 2029
        else:
            group = CompanyGroup.OUT_OF_SCOPE.value
            deadline = None

        in_scope = group != CompanyGroup.OUT_OF_SCOPE.value

        return {
            "company_group": group,
            "in_scope": in_scope,
            "compliance_deadline": deadline,
            "employee_count": employees,
            "net_turnover_eur": turnover,
            "is_franchise": is_franchise,
        }

    def generate_implementation_plan(
        self,
        config: CSDDDConfiguration,
    ) -> ImplementationPlan:
        """Generate a phased CSDDD implementation plan.

        Args:
            config: CSDDD configuration.

        Returns:
            ImplementationPlan with phases, durations, and deliverables.
        """
        phases: List[ImplementationPhase] = [
            ImplementationPhase(
                phase_number=1,
                phase_name="Gap Assessment",
                description=(
                    "Assess current due diligence practices against "
                    "CSDDD requirements and identify gaps"
                ),
                duration_months=2,
                csddd_articles=["Art_5", "Art_6"],
                deliverables=[
                    "Gap analysis report",
                    "Stakeholder mapping",
                    "Risk area identification",
                ],
            ),
            ImplementationPhase(
                phase_number=2,
                phase_name="Policy Integration",
                description=(
                    "Integrate due diligence into corporate policies "
                    "and governance structures (Art 5)"
                ),
                duration_months=3,
                csddd_articles=["Art_5"],
                deliverables=[
                    "Updated corporate governance framework",
                    "Due diligence policy document",
                    "Code of conduct update",
                    "Board oversight mechanism",
                ],
            ),
            ImplementationPhase(
                phase_number=3,
                phase_name="Impact Identification",
                description=(
                    "Identify actual and potential adverse impacts "
                    "across own operations and value chain (Art 6-7)"
                ),
                duration_months=4,
                csddd_articles=["Art_6", "Art_7", "Art_11"],
                deliverables=[
                    "Adverse impact register",
                    "Impact prioritisation matrix",
                    "Stakeholder engagement plan",
                    "Value chain mapping",
                ],
            ),
            ImplementationPhase(
                phase_number=4,
                phase_name="Prevention & Mitigation",
                description=(
                    "Develop and implement prevention and mitigation measures "
                    "including contractual clauses (Art 8-9, 16)"
                ),
                duration_months=4,
                csddd_articles=["Art_8", "Art_9", "Art_16"],
                deliverables=[
                    "Corrective action plans",
                    "Supplier contractual clauses",
                    "Prevention measures implementation",
                    "Monitoring framework",
                ],
            ),
            ImplementationPhase(
                phase_number=5,
                phase_name="Grievance & Remediation",
                description=(
                    "Establish grievance mechanism and remediation "
                    "procedures (Art 10, 12)"
                ),
                duration_months=3,
                csddd_articles=["Art_10", "Art_12"],
                deliverables=[
                    "Grievance mechanism design",
                    "Remediation procedures",
                    "Complaint handling process",
                    "Whistleblower channel",
                ],
            ),
            ImplementationPhase(
                phase_number=6,
                phase_name="Monitoring & Reporting",
                description=(
                    "Implement monitoring systems and prepare "
                    "public reporting (Art 13-14)"
                ),
                duration_months=3,
                csddd_articles=["Art_13", "Art_14"],
                deliverables=[
                    "KPI monitoring dashboard",
                    "Annual due diligence report",
                    "Website disclosure",
                    "Effectiveness assessment",
                ],
            ),
        ]

        # Add climate transition phase if required
        if config.climate_transition_required:
            phases.append(ImplementationPhase(
                phase_number=7,
                phase_name="Climate Transition Plan",
                description=(
                    "Develop Paris-aligned climate transition plan "
                    "with interim targets (Art 22)"
                ),
                duration_months=4,
                csddd_articles=["Art_22"],
                deliverables=[
                    "Climate transition plan document",
                    "Emission reduction targets (SBTi-aligned)",
                    "Interim 2030 targets",
                    "Financial investment plan",
                    "Implementation roadmap",
                ],
            ))

        total_months = sum(p.duration_months for p in phases)

        recommendations = self._generate_plan_recommendations(config)

        plan = ImplementationPlan(
            company_name=config.company_name,
            company_group=config.company_group,
            compliance_deadline=config.compliance_deadline,
            total_duration_months=total_months,
            phases=phases,
            recommendations=recommendations,
        )
        plan.provenance_hash = _compute_hash(plan)

        logger.info(
            "Implementation plan generated: %d phases, %d months",
            len(phases),
            total_months,
        )
        return plan

    def get_preset(self, sector_name: str) -> Dict[str, Any]:
        """Get a sector-specific preset configuration.

        Args:
            sector_name: Sector preset name.

        Returns:
            Dict with preset configuration values.
        """
        preset = SECTOR_PRESETS.get(sector_name)
        if preset is None:
            logger.warning(
                "No preset for sector '%s', using multi_sector", sector_name
            )
            preset = SECTOR_PRESETS["multi_sector"]

        # Convert enum values for serialization
        result = dict(preset)
        if isinstance(result.get("value_chain_depth"), ValueChainDepth):
            result["value_chain_depth"] = result["value_chain_depth"].value
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_sector(self, nace_code: str) -> SectorType:
        """Detect sector from NACE code."""
        if not nace_code:
            return SectorType.UNKNOWN

        division = nace_code[0].upper()
        nace_info = NACE_DIVISIONS.get(division)

        if nace_info:
            # Special handling for textiles (C13-C15)
            if division == "C" and len(nace_code) >= 3:
                sub = nace_code[1:3]
                if sub in ("13", "14", "15"):
                    return SectorType.TEXTILES

            return nace_info["sector"]

        return SectorType.UNKNOWN

    def _generate_plan_recommendations(
        self,
        config: CSDDDConfiguration,
    ) -> List[str]:
        """Generate implementation plan recommendations."""
        recs: List[str] = []

        if config.company_group == CompanyGroup.GROUP_1:
            recs.append(
                "As a Group 1 company, compliance is required by 2027; "
                "begin implementation immediately"
            )
        elif config.company_group == CompanyGroup.GROUP_2:
            recs.append(
                "As a Group 2 company, compliance is required by 2029; "
                "start planning by end of 2026"
            )

        if config.sector in (SectorType.EXTRACTIVE, SectorType.AGRICULTURE, SectorType.TEXTILES):
            recs.append(
                f"As a {config.sector.value} sector company, enhanced due "
                f"diligence is recommended for supply chain human rights impacts"
            )

        if config.value_chain_depth in (
            ValueChainDepth.FULL_UPSTREAM,
            ValueChainDepth.FULL_UPSTREAM_DOWNSTREAM,
        ):
            recs.append(
                "Full value chain mapping requires dedicated supply chain "
                "due diligence resources; consider phased rollout"
            )

        if "eudr_bridge" in config.bridges_enabled:
            recs.append(
                "EUDR compliance data should be integrated into CSDDD "
                "environmental impact assessment for deforestation risks"
            )

        recs.append(
            "Engage external legal counsel for Art 26 civil liability "
            "risk assessment and D&O insurance review"
        )

        return recs
