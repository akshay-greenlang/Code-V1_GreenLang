# -*- coding: utf-8 -*-
"""
Audit Trail & Lineage Agent Models (AGENT-MRV-030)

This module provides comprehensive data models for the GreenLang Audit Trail &
Lineage Agent, which maintains an immutable, hash-chained audit log and directed
acyclic graph (DAG) of data lineage across all MRV pipeline stages.

Supports:
- 12 audit event types covering the full MRV lifecycle (ingestion through seal)
- 8 lineage depth levels (L1_SOURCE through L8_REPORTING)
- 10 lineage node types (source_data, activity_data, emission_factor, etc.)
- 8 lineage edge types (data_flow, factor_application, method_selection, etc.)
- 4 emission scopes (Scope 1, Scope 2 location, Scope 2 market, Scope 3)
- 15 Scope 3 categories (cat_1 through cat_15)
- 10 emission factor sources (DEFRA, EPA, IPCC, IEA, Ecoinvent, etc.)
- 8 calculation methodologies (emission_factor, mass_balance, etc.)
- 8 change types for recalculation impact analysis
- 4 change severity levels for materiality assessment
- 5 evidence package statuses (draft through rejected)
- 3 assurance levels (limited, reasonable, no_assurance)
- 4 compliance statuses (compliant, non_compliant, partial, not_applicable)
- 5 data quality tiers (tier_1 through tier_5)
- 4 signature algorithms (SHA256, SHA384, SHA512, ED25519)
- 9 compliance frameworks (GHG Protocol, ISO 14064, CSRD, CDP, SBTi, SB 253,
  SEC Climate, EU Taxonomy, GRI)
- SHA-256 hash-chained audit trail with tamper detection
- Directed acyclic graph lineage traversal (forward and backward)
- Evidence packaging for third-party assurance readiness
- Compliance trace mapping (framework requirement to evidence)
- Change detection with materiality impact assessment
- Recalculation cascade analysis with dry-run mode
- Batch audit event ingestion with validation-only mode
- Chain integrity verification with break-point detection
- 10-stage pipeline provenance tracking (VALIDATE through SEAL)

All numeric fields use Decimal for precision in regulatory calculations.
All models are frozen (immutable) for audit trail integrity.

Example:
    >>> from greenlang.agents.mrv.audit_trail_lineage.models import (
    ...     AuditEventInput, AuditEventType, EmissionScope,
    ... )
    >>> event = AuditEventInput(
    ...     event_type=AuditEventType.CALCULATION_INITIATED,
    ...     agent_id="GL-MRV-S1-001",
    ...     scope=EmissionScope.SCOPE_1,
    ...     organization_id="org-001",
    ...     reporting_year=2025,
    ...     calculation_id="calc-abc-123",
    ...     payload={"fuel_type": "natural_gas", "quantity_kg": "1500.00"},
    ... )
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from enum import Enum
from pydantic import Field, validator
from pydantic import ConfigDict
import hashlib
import json
from greenlang.schemas import GreenLangBase

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-X-042"
AGENT_COMPONENT: str = "AGENT-MRV-030"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_atl_"

# Decimal quantization constants
_QUANT_2DP = Decimal("0.01")
_QUANT_4DP = Decimal("0.0001")
_QUANT_8DP = Decimal("0.00000001")
_QUANT_10DP = Decimal("0.0000000001")

# ==============================================================================
# ENUMERATIONS (25)
# ==============================================================================


class AuditEventType(str, Enum):
    """Audit event types covering the full MRV lifecycle.

    Each event type represents a discrete, auditable step in the emissions
    calculation pipeline.  Events are recorded in chronological order and
    hash-chained to form a tamper-evident audit trail.

    Events follow the natural pipeline order:
    1. calculation_initiated    -- pipeline kicked off
    2. activity_data_ingested   -- raw activity data loaded
    3. emission_factor_resolved -- EF looked up / matched
    4. methodology_applied      -- calculation method selected
    5. uncertainty_quantified   -- uncertainty bounds computed
    6. allocation_performed     -- emissions allocated to scopes/categories
    7. double_counting_checked  -- DC rules evaluated
    8. compliance_validated     -- framework compliance checked
    9. aggregation_completed    -- subtotals / totals computed
    10. recalculation_triggered -- change triggers recalc cascade
    11. evidence_packaged       -- evidence bundle assembled
    12. audit_sealed            -- final hash seal applied
    """

    CALCULATION_INITIATED = "calculation_initiated"
    ACTIVITY_DATA_INGESTED = "activity_data_ingested"
    EMISSION_FACTOR_RESOLVED = "emission_factor_resolved"
    METHODOLOGY_APPLIED = "methodology_applied"
    UNCERTAINTY_QUANTIFIED = "uncertainty_quantified"
    ALLOCATION_PERFORMED = "allocation_performed"
    DOUBLE_COUNTING_CHECKED = "double_counting_checked"
    COMPLIANCE_VALIDATED = "compliance_validated"
    AGGREGATION_COMPLETED = "aggregation_completed"
    RECALCULATION_TRIGGERED = "recalculation_triggered"
    EVIDENCE_PACKAGED = "evidence_packaged"
    AUDIT_SEALED = "audit_sealed"


class LineageLevel(str, Enum):
    """Lineage depth levels from raw source to final reporting.

    The 8-level hierarchy allows precise tracing of how raw data is
    transformed into reported emissions values.  Each level represents
    a stage of increasing aggregation and interpretation.

    L1_SOURCE:       Raw data source (ERP, sensor, invoice, survey)
    L2_EXTRACTION:   Extracted data (parsed, decoded, digitised)
    L3_VALIDATION:   Validated data (schema-checked, range-checked)
    L4_NORMALIZATION: Normalized data (unit-converted, gap-filled)
    L5_CALCULATION:  Calculated emissions (EF applied, formula run)
    L6_AGGREGATION:  Aggregated totals (by scope, category, facility)
    L7_COMPLIANCE:   Compliance-mapped data (framework-specific views)
    L8_REPORTING:    Final reported values (assurance-ready, sealed)
    """

    L1_SOURCE = "L1_SOURCE"
    L2_EXTRACTION = "L2_EXTRACTION"
    L3_VALIDATION = "L3_VALIDATION"
    L4_NORMALIZATION = "L4_NORMALIZATION"
    L5_CALCULATION = "L5_CALCULATION"
    L6_AGGREGATION = "L6_AGGREGATION"
    L7_COMPLIANCE = "L7_COMPLIANCE"
    L8_REPORTING = "L8_REPORTING"


class LineageNodeType(str, Enum):
    """Types of nodes in the lineage directed acyclic graph.

    Each node represents a data artefact at a specific point in the
    MRV pipeline.  Nodes are connected by edges that describe the
    transformation relationship between them.
    """

    SOURCE_DATA = "source_data"
    ACTIVITY_DATA = "activity_data"
    EMISSION_FACTOR = "emission_factor"
    METHODOLOGY = "methodology"
    CALCULATION = "calculation"
    ALLOCATION = "allocation"
    AGGREGATION = "aggregation"
    COMPLIANCE_CHECK = "compliance_check"
    REPORT_ITEM = "report_item"
    EVIDENCE = "evidence"


class LineageEdgeType(str, Enum):
    """Types of edges in the lineage directed acyclic graph.

    Each edge describes how one data artefact was derived from or
    influenced by another.  Edges carry transformation metadata
    and confidence scores.
    """

    DATA_FLOW = "data_flow"
    FACTOR_APPLICATION = "factor_application"
    METHOD_SELECTION = "method_selection"
    ALLOCATION_LINK = "allocation_link"
    AGGREGATION_LINK = "aggregation_link"
    COMPLIANCE_LINK = "compliance_link"
    REPORT_LINK = "report_link"
    EVIDENCE_LINK = "evidence_link"


class EmissionScope(str, Enum):
    """GHG Protocol emission scopes.

    Scope 1: Direct emissions from owned/controlled sources.
    Scope 2 Location: Indirect emissions from purchased energy (grid average).
    Scope 2 Market: Indirect emissions from purchased energy (contractual).
    Scope 3: All other indirect emissions in the value chain.
    """

    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"


class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories (1-15).

    Covers all 15 categories per the GHG Protocol Corporate Value Chain
    (Scope 3) Accounting and Reporting Standard.  Categories 1-8 are
    upstream; categories 9-15 are downstream.
    """

    CAT_1 = "cat_1"
    CAT_2 = "cat_2"
    CAT_3 = "cat_3"
    CAT_4 = "cat_4"
    CAT_5 = "cat_5"
    CAT_6 = "cat_6"
    CAT_7 = "cat_7"
    CAT_8 = "cat_8"
    CAT_9 = "cat_9"
    CAT_10 = "cat_10"
    CAT_11 = "cat_11"
    CAT_12 = "cat_12"
    CAT_13 = "cat_13"
    CAT_14 = "cat_14"
    CAT_15 = "cat_15"


class EFSource(str, Enum):
    """Emission factor data sources for provenance tracking.

    Each source carries a different authority level and update cadence.
    The audit trail records which source was used for each EF lookup.
    """

    DEFRA = "DEFRA"
    EPA = "EPA"
    IPCC = "IPCC"
    IEA = "IEA"
    ECOINVENT = "Ecoinvent"
    ADEME = "ADEME"
    BEIS = "BEIS"
    EGRID = "eGRID"
    GABI = "GaBi"
    CUSTOM = "custom"


class CalculationMethodology(str, Enum):
    """Calculation methodologies used by MRV agents.

    Each methodology has different data requirements, accuracy levels,
    and auditability characteristics.  The audit trail records which
    methodology was applied for each calculation.
    """

    EMISSION_FACTOR = "emission_factor"
    MASS_BALANCE = "mass_balance"
    STOICHIOMETRIC = "stoichiometric"
    DIRECT_MEASUREMENT = "direct_measurement"
    SPEND_BASED = "spend_based"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"


class ChangeType(str, Enum):
    """Types of changes that may trigger recalculation.

    Changes are classified by what was modified in order to determine
    the scope of recalculation impact and whether materiality thresholds
    are breached.
    """

    EMISSION_FACTOR_UPDATE = "emission_factor_update"
    ACTIVITY_DATA_CORRECTION = "activity_data_correction"
    METHODOLOGY_CHANGE = "methodology_change"
    SCOPE_BOUNDARY_CHANGE = "scope_boundary_change"
    ORGANIZATIONAL_CHANGE = "organizational_change"
    BASE_YEAR_RECALCULATION = "base_year_recalculation"
    ERROR_CORRECTION = "error_correction"
    REGULATORY_UPDATE = "regulatory_update"


class ChangeSeverity(str, Enum):
    """Severity levels for detected changes.

    Severity determines whether a change requires immediate recalculation,
    can be deferred to the next reporting period, or is informational only.

    CRITICAL: Exceeds materiality threshold; immediate recalculation required.
    HIGH:     Near materiality threshold; recalculation recommended.
    MEDIUM:   Moderate impact; schedule for next reporting cycle.
    LOW:      Minor impact; log for record-keeping only.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidencePackageStatus(str, Enum):
    """Status of an evidence package for assurance readiness.

    Evidence packages move through a lifecycle from draft to final
    acceptance or rejection by the assurance provider.
    """

    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    SEALED = "sealed"
    REJECTED = "rejected"


class AssuranceLevel(str, Enum):
    """Assurance engagement level per ISAE 3000 / ISAE 3410.

    LIMITED:       Negative assurance ("nothing has come to our attention").
    REASONABLE:    Positive assurance ("in our opinion, fairly stated").
    NO_ASSURANCE:  No external assurance engagement.
    """

    LIMITED = "limited"
    REASONABLE = "reasonable"
    NO_ASSURANCE = "no_assurance"


class ComplianceStatus(str, Enum):
    """Compliance check result status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"


class DataQualityTier(str, Enum):
    """Data quality tier (1 = best, 5 = worst).

    Tier 1: Verified primary data (direct measurement, supplier-specific).
    Tier 2: Unverified primary data (self-reported, not assured).
    Tier 3: Average data with good proxy match (industry average, same region).
    Tier 4: Estimated data (spend-based, modelled, proxy with poor match).
    Tier 5: Extrapolated or default values (global defaults, outdated data).
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    TIER_5 = "tier_5"


class SignatureAlgorithm(str, Enum):
    """Cryptographic signature algorithms for evidence package signing.

    SHA256:  SHA-256 HMAC -- standard for internal signing.
    SHA384:  SHA-384 HMAC -- intermediate strength.
    SHA512:  SHA-512 HMAC -- high-strength signing.
    ED25519: Ed25519 digital signature -- public-key signing for external assurance.
    """

    SHA256 = "SHA256"
    SHA384 = "SHA384"
    SHA512 = "SHA512"
    ED25519 = "ED25519"


class AuditTrailStatus(str, Enum):
    """Overall status of an organization's audit trail.

    ACTIVE:     Trail is actively recording events.
    SEALED:     Trail has been sealed for the reporting period.
    ARCHIVED:   Trail has been archived after assurance completion.
    TAMPERED:   Chain integrity check detected a break.
    """

    ACTIVE = "active"
    SEALED = "sealed"
    ARCHIVED = "archived"
    TAMPERED = "tampered"


class VerificationResult(str, Enum):
    """Result of a chain integrity or evidence verification check.

    VERIFIED:   All hashes match; no tampering detected.
    FAILED:     One or more hashes do not match; possible tampering.
    INCOMPLETE: Verification could not be completed (missing events).
    """

    VERIFIED = "verified"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class TraversalDirection(str, Enum):
    """Direction of lineage graph traversal.

    FORWARD:  Trace from source data toward reporting (downstream).
    BACKWARD: Trace from reported value back to source data (upstream).
    """

    FORWARD = "forward"
    BACKWARD = "backward"


class GraphFormat(str, Enum):
    """Output format for lineage graph serialization.

    ADJACENCY_LIST: Node list + edge list (default, compact).
    DOT:            Graphviz DOT format for visualization.
    CYTOSCAPE:      Cytoscape.js JSON format for web rendering.
    MERMAID:        Mermaid.js flowchart syntax for Markdown embedding.
    """

    ADJACENCY_LIST = "adjacency_list"
    DOT = "dot"
    CYTOSCAPE = "cytoscape"
    MERMAID = "mermaid"


class ReportingPeriodType(str, Enum):
    """Reporting period granularity.

    ANNUAL:      Full fiscal or calendar year.
    SEMI_ANNUAL: Half-year reporting period.
    QUARTERLY:   Quarter-year reporting period.
    """

    ANNUAL = "annual"
    SEMI_ANNUAL = "semi_annual"
    QUARTERLY = "quarterly"


class RecalculationTrigger(str, Enum):
    """Triggers that initiate a recalculation cascade.

    Each trigger carries different propagation rules and materiality
    thresholds.  The audit trail records the trigger for each
    recalculation event.
    """

    EF_DATABASE_UPDATE = "ef_database_update"
    DATA_CORRECTION = "data_correction"
    METHODOLOGY_REVISION = "methodology_revision"
    BOUNDARY_ADJUSTMENT = "boundary_adjustment"
    STRUCTURAL_CHANGE = "structural_change"
    MANUAL_OVERRIDE = "manual_override"


class FrameworkIdentifier(str, Enum):
    """Regulatory and reporting framework identifiers.

    Each framework has specific disclosure requirements, data granularity
    expectations, and assurance level requirements.  The compliance trace
    maps evidence to individual framework requirements.
    """

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    SEC_CLIMATE = "sec_climate"
    EU_TAXONOMY = "eu_taxonomy"
    GRI = "gri"


class MaterialityThreshold(str, Enum):
    """Materiality threshold types for change impact assessment.

    QUANTITATIVE_5PCT: Change exceeds 5% of total reported emissions.
    QUANTITATIVE_1PCT: Change exceeds 1% of total reported emissions.
    DE_MINIMIS:        Change falls below the de minimis threshold.
    """

    QUANTITATIVE_5PCT = "quantitative_5pct"
    QUANTITATIVE_1PCT = "quantitative_1pct"
    DE_MINIMIS = "de_minimis"


class ChainIntegrityStatus(str, Enum):
    """Integrity status of the hash chain.

    INTACT:   All sequential hashes verified successfully.
    BROKEN:   One or more sequential hash links are broken.
    UNKNOWN:  Integrity has not yet been verified.
    """

    INTACT = "intact"
    BROKEN = "broken"
    UNKNOWN = "unknown"


class PipelineStage(str, Enum):
    """Processing pipeline stages for provenance tracking.

    The 10-stage pipeline matches the standard GreenLang MRV processing
    workflow.  Each stage produces one or more provenance records that
    are hash-chained together.
    """

    VALIDATE = "validate"
    CLASSIFY = "classify"
    NORMALIZE = "normalize"
    RESOLVE_EFS = "resolve_efs"
    CALCULATE = "calculate"
    ALLOCATE = "allocate"
    AGGREGATE = "aggregate"
    COMPLIANCE = "compliance"
    PROVENANCE = "provenance"
    SEAL = "seal"


# ==============================================================================
# CONSTANT TABLES
# ==============================================================================

# Scope 3 Category Metadata -- 15 categories with name and position
SCOPE3_CATEGORY_METADATA: Dict[str, Dict[str, str]] = {
    Scope3Category.CAT_1.value: {
        "number": "1", "name": "Purchased Goods and Services",
        "position": "upstream", "agent_component": "AGENT-MRV-014",
    },
    Scope3Category.CAT_2.value: {
        "number": "2", "name": "Capital Goods",
        "position": "upstream", "agent_component": "AGENT-MRV-015",
    },
    Scope3Category.CAT_3.value: {
        "number": "3", "name": "Fuel- and Energy-Related Activities",
        "position": "upstream", "agent_component": "AGENT-MRV-016",
    },
    Scope3Category.CAT_4.value: {
        "number": "4", "name": "Upstream Transportation and Distribution",
        "position": "upstream", "agent_component": "AGENT-MRV-017",
    },
    Scope3Category.CAT_5.value: {
        "number": "5", "name": "Waste Generated in Operations",
        "position": "upstream", "agent_component": "AGENT-MRV-018",
    },
    Scope3Category.CAT_6.value: {
        "number": "6", "name": "Business Travel",
        "position": "upstream", "agent_component": "AGENT-MRV-019",
    },
    Scope3Category.CAT_7.value: {
        "number": "7", "name": "Employee Commuting",
        "position": "upstream", "agent_component": "AGENT-MRV-020",
    },
    Scope3Category.CAT_8.value: {
        "number": "8", "name": "Upstream Leased Assets",
        "position": "upstream", "agent_component": "AGENT-MRV-021",
    },
    Scope3Category.CAT_9.value: {
        "number": "9", "name": "Downstream Transportation and Distribution",
        "position": "downstream", "agent_component": "AGENT-MRV-022",
    },
    Scope3Category.CAT_10.value: {
        "number": "10", "name": "Processing of Sold Products",
        "position": "downstream", "agent_component": "AGENT-MRV-023",
    },
    Scope3Category.CAT_11.value: {
        "number": "11", "name": "Use of Sold Products",
        "position": "downstream", "agent_component": "AGENT-MRV-024",
    },
    Scope3Category.CAT_12.value: {
        "number": "12", "name": "End-of-Life Treatment of Sold Products",
        "position": "downstream", "agent_component": "AGENT-MRV-025",
    },
    Scope3Category.CAT_13.value: {
        "number": "13", "name": "Downstream Leased Assets",
        "position": "downstream", "agent_component": "AGENT-MRV-026",
    },
    Scope3Category.CAT_14.value: {
        "number": "14", "name": "Franchises",
        "position": "downstream", "agent_component": "AGENT-MRV-027",
    },
    Scope3Category.CAT_15.value: {
        "number": "15", "name": "Investments",
        "position": "downstream", "agent_component": "AGENT-MRV-028",
    },
}

# Audit event type ordering -- maps event type to natural pipeline position
AUDIT_EVENT_ORDER: Dict[str, int] = {
    AuditEventType.CALCULATION_INITIATED.value: 1,
    AuditEventType.ACTIVITY_DATA_INGESTED.value: 2,
    AuditEventType.EMISSION_FACTOR_RESOLVED.value: 3,
    AuditEventType.METHODOLOGY_APPLIED.value: 4,
    AuditEventType.UNCERTAINTY_QUANTIFIED.value: 5,
    AuditEventType.ALLOCATION_PERFORMED.value: 6,
    AuditEventType.DOUBLE_COUNTING_CHECKED.value: 7,
    AuditEventType.COMPLIANCE_VALIDATED.value: 8,
    AuditEventType.AGGREGATION_COMPLETED.value: 9,
    AuditEventType.RECALCULATION_TRIGGERED.value: 10,
    AuditEventType.EVIDENCE_PACKAGED.value: 11,
    AuditEventType.AUDIT_SEALED.value: 12,
}

# Lineage level depth ordering
LINEAGE_LEVEL_DEPTH: Dict[str, int] = {
    LineageLevel.L1_SOURCE.value: 1,
    LineageLevel.L2_EXTRACTION.value: 2,
    LineageLevel.L3_VALIDATION.value: 3,
    LineageLevel.L4_NORMALIZATION.value: 4,
    LineageLevel.L5_CALCULATION.value: 5,
    LineageLevel.L6_AGGREGATION.value: 6,
    LineageLevel.L7_COMPLIANCE.value: 7,
    LineageLevel.L8_REPORTING.value: 8,
}

# Compliance framework requirements -- number of required data points
COMPLIANCE_FRAMEWORK_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    FrameworkIdentifier.GHG_PROTOCOL.value: {
        "name": "GHG Protocol Corporate Standard",
        "total_requirements": 42,
        "mandatory": True,
        "assurance_required": False,
        "version": "2015 (with 2023 amendments)",
    },
    FrameworkIdentifier.ISO_14064.value: {
        "name": "ISO 14064-1:2018",
        "total_requirements": 38,
        "mandatory": False,
        "assurance_required": True,
        "version": "2018",
    },
    FrameworkIdentifier.CSRD_ESRS.value: {
        "name": "CSRD ESRS E1 Climate Change",
        "total_requirements": 65,
        "mandatory": True,
        "assurance_required": True,
        "version": "ESRS E1 (2024)",
    },
    FrameworkIdentifier.CDP.value: {
        "name": "CDP Climate Change Questionnaire",
        "total_requirements": 55,
        "mandatory": False,
        "assurance_required": False,
        "version": "2024",
    },
    FrameworkIdentifier.SBTI.value: {
        "name": "Science Based Targets initiative",
        "total_requirements": 30,
        "mandatory": False,
        "assurance_required": False,
        "version": "SBTi v5.1 (2024)",
    },
    FrameworkIdentifier.SB_253.value: {
        "name": "California SB 253",
        "total_requirements": 48,
        "mandatory": True,
        "assurance_required": True,
        "version": "2023",
    },
    FrameworkIdentifier.SEC_CLIMATE.value: {
        "name": "SEC Climate Disclosure Rule",
        "total_requirements": 35,
        "mandatory": True,
        "assurance_required": True,
        "version": "2024",
    },
    FrameworkIdentifier.EU_TAXONOMY.value: {
        "name": "EU Taxonomy Regulation",
        "total_requirements": 28,
        "mandatory": True,
        "assurance_required": False,
        "version": "2020/852",
    },
    FrameworkIdentifier.GRI.value: {
        "name": "GRI 305: Emissions 2016",
        "total_requirements": 22,
        "mandatory": False,
        "assurance_required": False,
        "version": "2016 (2024 update)",
    },
}

# Change severity materiality thresholds
CHANGE_SEVERITY_THRESHOLDS: Dict[str, Dict[str, Decimal]] = {
    ChangeSeverity.CRITICAL.value: {
        "min_impact_pct": Decimal("5.00"),
        "recalculation_required": Decimal("1"),  # True
        "max_response_hours": Decimal("24"),
    },
    ChangeSeverity.HIGH.value: {
        "min_impact_pct": Decimal("1.00"),
        "recalculation_required": Decimal("1"),  # True
        "max_response_hours": Decimal("72"),
    },
    ChangeSeverity.MEDIUM.value: {
        "min_impact_pct": Decimal("0.10"),
        "recalculation_required": Decimal("0"),  # False, deferred
        "max_response_hours": Decimal("168"),
    },
    ChangeSeverity.LOW.value: {
        "min_impact_pct": Decimal("0.00"),
        "recalculation_required": Decimal("0"),  # False
        "max_response_hours": Decimal("720"),
    },
}

# Data quality tier to uncertainty mapping
DATA_QUALITY_UNCERTAINTY: Dict[str, Decimal] = {
    DataQualityTier.TIER_1.value: Decimal("5"),
    DataQualityTier.TIER_2.value: Decimal("15"),
    DataQualityTier.TIER_3.value: Decimal("30"),
    DataQualityTier.TIER_4.value: Decimal("50"),
    DataQualityTier.TIER_5.value: Decimal("60"),
}

# Agent ID to component mapping for cross-agent lineage resolution
MRV_AGENT_REGISTRY: Dict[str, Dict[str, str]] = {
    "GL-MRV-S1-001": {"component": "AGENT-MRV-001", "name": "Stationary Combustion"},
    "GL-MRV-S1-002": {"component": "AGENT-MRV-002", "name": "Refrigerants & F-Gas"},
    "GL-MRV-S1-003": {"component": "AGENT-MRV-003", "name": "Mobile Combustion"},
    "GL-MRV-S1-004": {"component": "AGENT-MRV-004", "name": "Process Emissions"},
    "GL-MRV-S1-005": {"component": "AGENT-MRV-005", "name": "Fugitive Emissions"},
    "GL-MRV-S1-006": {"component": "AGENT-MRV-006", "name": "Land Use Emissions"},
    "GL-MRV-S1-007": {"component": "AGENT-MRV-007", "name": "Waste Treatment"},
    "GL-MRV-S1-008": {"component": "AGENT-MRV-008", "name": "Agricultural Emissions"},
    "GL-MRV-S2-001": {"component": "AGENT-MRV-009", "name": "Scope 2 Location-Based"},
    "GL-MRV-S2-002": {"component": "AGENT-MRV-010", "name": "Scope 2 Market-Based"},
    "GL-MRV-S2-003": {"component": "AGENT-MRV-011", "name": "Steam/Heat Purchase"},
    "GL-MRV-S2-004": {"component": "AGENT-MRV-012", "name": "Cooling Purchase"},
    "GL-MRV-S2-005": {"component": "AGENT-MRV-013", "name": "Dual Reporting Reconciliation"},
    "GL-MRV-S3-001": {"component": "AGENT-MRV-014", "name": "Purchased Goods & Services"},
    "GL-MRV-S3-002": {"component": "AGENT-MRV-015", "name": "Capital Goods"},
    "GL-MRV-S3-003": {"component": "AGENT-MRV-016", "name": "Fuel & Energy Activities"},
    "GL-MRV-S3-004": {"component": "AGENT-MRV-017", "name": "Upstream Transportation"},
    "GL-MRV-S3-005": {"component": "AGENT-MRV-018", "name": "Waste Generated"},
    "GL-MRV-S3-006": {"component": "AGENT-MRV-019", "name": "Business Travel"},
    "GL-MRV-S3-007": {"component": "AGENT-MRV-020", "name": "Employee Commuting"},
    "GL-MRV-S3-008": {"component": "AGENT-MRV-021", "name": "Upstream Leased Assets"},
    "GL-MRV-S3-009": {"component": "AGENT-MRV-022", "name": "Downstream Transportation"},
    "GL-MRV-S3-010": {"component": "AGENT-MRV-023", "name": "Processing of Sold Products"},
    "GL-MRV-S3-011": {"component": "AGENT-MRV-024", "name": "Use of Sold Products"},
    "GL-MRV-S3-012": {"component": "AGENT-MRV-025", "name": "End-of-Life Treatment"},
    "GL-MRV-S3-013": {"component": "AGENT-MRV-026", "name": "Downstream Leased Assets"},
    "GL-MRV-S3-014": {"component": "AGENT-MRV-027", "name": "Franchises"},
    "GL-MRV-S3-015": {"component": "AGENT-MRV-028", "name": "Investments"},
    "GL-MRV-X-040": {"component": "AGENT-MRV-029", "name": "Scope 3 Category Mapper"},
    "GL-MRV-X-042": {"component": "AGENT-MRV-030", "name": "Audit Trail & Lineage"},
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def compute_event_hash(
    event_type: str,
    agent_id: str,
    payload: Dict[str, Any],
    prev_hash: str,
    timestamp: str,
) -> str:
    """Compute SHA-256 hash for an audit event in the chain.

    The hash covers all material fields plus the previous event hash,
    creating a tamper-evident chain.

    Args:
        event_type: The audit event type string.
        agent_id: The agent that produced the event.
        payload: Event payload dictionary.
        prev_hash: SHA-256 hash of the previous event in the chain.
        timestamp: ISO 8601 timestamp string.

    Returns:
        64-character lowercase hex SHA-256 hash string.

    Example:
        >>> compute_event_hash(
        ...     "calculation_initiated", "GL-MRV-S1-001",
        ...     {"fuel": "natural_gas"}, "0" * 64,
        ...     "2025-01-15T10:30:00Z",
        ... )
        'a1b2c3...'
    """
    canonical = json.dumps(
        {
            "event_type": event_type,
            "agent_id": agent_id,
            "payload": payload,
            "prev_hash": prev_hash,
            "timestamp": timestamp,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_lineage_hash(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> str:
    """Compute SHA-256 provenance hash for a lineage graph.

    Args:
        nodes: List of node dictionaries.
        edges: List of edge dictionaries.

    Returns:
        64-character lowercase hex SHA-256 hash string.

    Example:
        >>> compute_lineage_hash(
        ...     [{"id": "n1", "type": "source_data"}],
        ...     [{"source": "n1", "target": "n2"}],
        ... )
        'b2c3d4...'
    """
    canonical = json.dumps(
        {"nodes": nodes, "edges": edges},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_evidence_hash(
    organization_id: str,
    reporting_year: int,
    contents: Dict[str, Any],
) -> str:
    """Compute SHA-256 hash for an evidence package.

    Args:
        organization_id: Organization identifier string.
        reporting_year: Reporting year integer.
        contents: Evidence package contents dictionary.

    Returns:
        64-character lowercase hex SHA-256 hash string.

    Example:
        >>> compute_evidence_hash("org-001", 2025, {"total": "1000"})
        'c3d4e5...'
    """
    canonical = json.dumps(
        {
            "organization_id": organization_id,
            "reporting_year": reporting_year,
            "contents": contents,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def severity_from_impact(impact_pct: Decimal) -> ChangeSeverity:
    """Determine change severity from impact percentage.

    Args:
        impact_pct: Impact as a percentage of total emissions (Decimal).

    Returns:
        ChangeSeverity corresponding to the impact level.

    Example:
        >>> severity_from_impact(Decimal("6.5"))
        <ChangeSeverity.CRITICAL: 'critical'>
        >>> severity_from_impact(Decimal("0.05"))
        <ChangeSeverity.LOW: 'low'>
    """
    if impact_pct >= Decimal("5.00"):
        return ChangeSeverity.CRITICAL
    if impact_pct >= Decimal("1.00"):
        return ChangeSeverity.HIGH
    if impact_pct >= Decimal("0.10"):
        return ChangeSeverity.MEDIUM
    return ChangeSeverity.LOW


def lineage_level_depth(level: LineageLevel) -> int:
    """Return the numeric depth (1-8) for a lineage level.

    Args:
        level: A LineageLevel enum member.

    Returns:
        Integer 1-8.

    Example:
        >>> lineage_level_depth(LineageLevel.L5_CALCULATION)
        5
    """
    return LINEAGE_LEVEL_DEPTH.get(level.value, 0)


def category_name(category: Scope3Category) -> str:
    """Get the human-readable name for a Scope3Category.

    Args:
        category: A Scope3Category enum member.

    Returns:
        Human-readable category name string.

    Example:
        >>> category_name(Scope3Category.CAT_6)
        'Business Travel'
    """
    metadata = SCOPE3_CATEGORY_METADATA.get(category.value, {})
    return metadata.get("name", category.value)


# ==============================================================================
# PYDANTIC MODELS -- INPUT MODELS (10)
# ==============================================================================


class AuditEventInput(GreenLangBase):
    """Input for recording a single audit event in the hash chain.

    Each audit event captures a discrete, auditable action performed by
    an MRV agent during the emissions calculation pipeline.  Events are
    hash-chained to create a tamper-evident audit trail.

    Example:
        >>> event = AuditEventInput(
        ...     event_type=AuditEventType.CALCULATION_INITIATED,
        ...     agent_id="GL-MRV-S1-001",
        ...     scope=EmissionScope.SCOPE_1,
        ...     organization_id="org-001",
        ...     reporting_year=2025,
        ...     calculation_id="calc-abc-123",
        ...     payload={"fuel_type": "natural_gas"},
        ... )
    """

    event_type: AuditEventType = Field(
        ...,
        description="Type of audit event being recorded",
    )
    agent_id: str = Field(
        ..., min_length=1, max_length=50,
        description="Identifier of the MRV agent that produced this event",
    )
    scope: Optional[EmissionScope] = Field(
        default=None,
        description="Emission scope associated with this event",
    )
    category: Optional[Scope3Category] = Field(
        default=None,
        description="Scope 3 category if scope is scope_3",
    )
    organization_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030,
        description="Reporting year for the emission calculation",
    )
    calculation_id: Optional[str] = Field(
        default=None, max_length=200,
        description="Unique calculation identifier linking related events",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific payload data (inputs, outputs, parameters)",
    )
    data_quality_score: Optional[Decimal] = Field(
        default=None, ge=Decimal("1.0"), le=Decimal("5.0"),
        description="Data quality score (1.0 = best, 5.0 = worst)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tags, labels, correlation IDs)",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)

    @validator("category")
    def validate_category_requires_scope3(
        cls, v: Optional[Scope3Category], values: dict
    ) -> Optional[Scope3Category]:
        """Validate that category is only set when scope is scope_3."""
        scope = values.get("scope")
        if v is not None and scope is not None and scope != EmissionScope.SCOPE_3:
            raise ValueError(
                f"category can only be set when scope is scope_3, got scope={scope}"
            )
        return v

    @validator("data_quality_score")
    def quantize_dq_score(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Quantize data quality score to 2 decimal places."""
        if v is not None:
            return v.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)
        return v


class LineageNodeInput(GreenLangBase):
    """Input for creating a lineage node in the DAG.

    A lineage node represents a data artefact at a specific point in
    the MRV pipeline.  Nodes are typed and levelled to enable structured
    traversal and impact analysis.

    Example:
        >>> node = LineageNodeInput(
        ...     node_type=LineageNodeType.ACTIVITY_DATA,
        ...     level=LineageLevel.L1_SOURCE,
        ...     agent_id="GL-MRV-S1-001",
        ...     qualified_name="stationary_combustion.natural_gas.qty_kg",
        ...     value=Decimal("1500.00"),
        ...     unit="kg",
        ... )
    """

    node_type: LineageNodeType = Field(
        ...,
        description="Type of lineage node",
    )
    level: LineageLevel = Field(
        ...,
        description="Lineage depth level (L1_SOURCE through L8_REPORTING)",
    )
    agent_id: str = Field(
        ..., min_length=1, max_length=50,
        description="Identifier of the MRV agent that owns this node",
    )
    qualified_name: str = Field(
        ..., min_length=1, max_length=500,
        description="Dot-separated qualified name of the data artefact",
    )
    value: Optional[Decimal] = Field(
        default=None,
        description="Numeric value at this node (if applicable)",
    )
    unit: Optional[str] = Field(
        default=None, max_length=50,
        description="Unit of measure for the value",
    )
    data_quality_score: Optional[Decimal] = Field(
        default=None, ge=Decimal("1.0"), le=Decimal("5.0"),
        description="Data quality score (1.0 = best, 5.0 = worst)",
    )
    ef_source: Optional[EFSource] = Field(
        default=None,
        description="Emission factor source (if node is an emission_factor)",
    )
    methodology: Optional[CalculationMethodology] = Field(
        default=None,
        description="Calculation methodology (if node is a calculation)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the node",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)

    @validator("value")
    def quantize_value(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Quantize value to 8 decimal places for precision."""
        if v is not None:
            return v.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        return v


class LineageEdgeInput(GreenLangBase):
    """Input for creating a lineage edge in the DAG.

    Edges connect source nodes to target nodes and describe the
    transformation or dependency relationship between them.

    Example:
        >>> edge = LineageEdgeInput(
        ...     source_node_id="node-001",
        ...     target_node_id="node-002",
        ...     edge_type=LineageEdgeType.FACTOR_APPLICATION,
        ...     transformation_description="Apply DEFRA 2024 EF for natural gas",
        ...     confidence=Decimal("0.95"),
        ... )
    """

    source_node_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Identifier of the source node",
    )
    target_node_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Identifier of the target node",
    )
    edge_type: LineageEdgeType = Field(
        ...,
        description="Type of edge connecting the nodes",
    )
    transformation_description: Optional[str] = Field(
        default=None, max_length=1000,
        description="Human-readable description of the transformation",
    )
    confidence: Optional[Decimal] = Field(
        default=None, ge=Decimal("0.0"), le=Decimal("1.0"),
        description="Confidence score for this transformation (0.0-1.0)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional edge metadata",
    )

    model_config = ConfigDict(frozen=True)

    @validator("source_node_id")
    def validate_different_nodes(cls, v: str, values: dict) -> str:
        """Validate source and target are different nodes (no self-loops)."""
        # target_node_id may not be parsed yet; cross-field check at edge level
        return v

    @validator("confidence")
    def quantize_confidence(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Quantize confidence to 4 decimal places."""
        if v is not None:
            return v.quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)
        return v


class EvidencePackageRequest(GreenLangBase):
    """Request to assemble an evidence package for assurance readiness.

    An evidence package bundles all relevant audit events, lineage graphs,
    compliance traces, and supporting documentation for a specific
    organization and reporting period.

    Example:
        >>> req = EvidencePackageRequest(
        ...     organization_id="org-001",
        ...     reporting_year=2025,
        ...     frameworks=[FrameworkIdentifier.GHG_PROTOCOL, FrameworkIdentifier.CSRD_ESRS],
        ...     assurance_level=AssuranceLevel.REASONABLE,
        ... )
    """

    organization_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030,
        description="Reporting year",
    )
    frameworks: List[FrameworkIdentifier] = Field(
        ..., min_length=1,
        description="Target compliance frameworks for the evidence package",
    )
    scope_filter: Optional[List[EmissionScope]] = Field(
        default=None,
        description="Filter evidence by emission scope (None = all scopes)",
    )
    category_filter: Optional[List[Scope3Category]] = Field(
        default=None,
        description="Filter evidence by Scope 3 category (None = all categories)",
    )
    assurance_level: AssuranceLevel = Field(
        default=AssuranceLevel.LIMITED,
        description="Target assurance level",
    )
    include_signatures: bool = Field(
        default=True,
        description="Whether to include cryptographic signatures in the package",
    )
    signature_algorithm: SignatureAlgorithm = Field(
        default=SignatureAlgorithm.SHA256,
        description="Signature algorithm to use",
    )
    include_lineage_graphs: bool = Field(
        default=True,
        description="Whether to include full lineage graphs in the package",
    )
    graph_format: GraphFormat = Field(
        default=GraphFormat.ADJACENCY_LIST,
        description="Format for lineage graph serialization",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)


class ComplianceTraceRequest(GreenLangBase):
    """Request to trace compliance requirements to evidence.

    Maps each disclosure requirement of a specific framework to the
    audit events and lineage nodes that provide supporting evidence.

    Example:
        >>> req = ComplianceTraceRequest(
        ...     framework=FrameworkIdentifier.CSRD_ESRS,
        ...     organization_id="org-001",
        ...     reporting_year=2025,
        ...     data_points=["E1-6-a", "E1-6-b", "E1-6-c"],
        ... )
    """

    framework: FrameworkIdentifier = Field(
        ...,
        description="Target compliance framework",
    )
    organization_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030,
        description="Reporting year",
    )
    data_points: Optional[List[str]] = Field(
        default=None,
        description="Specific data points/requirements to trace (None = all)",
    )
    scope_filter: Optional[List[EmissionScope]] = Field(
        default=None,
        description="Filter by emission scope",
    )
    include_gaps: bool = Field(
        default=True,
        description="Whether to include gap analysis in the result",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)


class ChangeDetectionInput(GreenLangBase):
    """Input for registering a detected change that may trigger recalculation.

    Changes to emission factors, activity data, methodologies, or
    organizational boundaries are tracked and assessed for materiality
    impact.

    Example:
        >>> change = ChangeDetectionInput(
        ...     change_type=ChangeType.EMISSION_FACTOR_UPDATE,
        ...     affected_entity_type="emission_factor",
        ...     affected_entity_id="DEFRA-2024-natural-gas",
        ...     old_value=Decimal("2.0210"),
        ...     new_value=Decimal("2.0340"),
        ...     trigger=RecalculationTrigger.EF_DATABASE_UPDATE,
        ...     severity=ChangeSeverity.MEDIUM,
        ... )
    """

    change_type: ChangeType = Field(
        ...,
        description="Type of change detected",
    )
    affected_entity_type: str = Field(
        ..., min_length=1, max_length=200,
        description="Type of entity affected (emission_factor, activity_data, etc.)",
    )
    affected_entity_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Identifier of the affected entity",
    )
    old_value: Optional[Decimal] = Field(
        default=None,
        description="Previous value before the change",
    )
    new_value: Optional[Decimal] = Field(
        default=None,
        description="New value after the change",
    )
    old_value_str: Optional[str] = Field(
        default=None, max_length=2000,
        description="Previous value as string (for non-numeric changes)",
    )
    new_value_str: Optional[str] = Field(
        default=None, max_length=2000,
        description="New value as string (for non-numeric changes)",
    )
    trigger: RecalculationTrigger = Field(
        ...,
        description="What triggered this change",
    )
    severity: Optional[ChangeSeverity] = Field(
        default=None,
        description="Severity (auto-calculated if None based on impact)",
    )
    effective_date: Optional[str] = Field(
        default=None, max_length=10,
        description="Date the change takes effect (YYYY-MM-DD)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional change metadata",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)

    @validator("old_value")
    def quantize_old_value(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Quantize old_value to 10 decimal places."""
        if v is not None:
            return v.quantize(_QUANT_10DP, rounding=ROUND_HALF_UP)
        return v

    @validator("new_value")
    def quantize_new_value(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Quantize new_value to 10 decimal places."""
        if v is not None:
            return v.quantize(_QUANT_10DP, rounding=ROUND_HALF_UP)
        return v


class BatchAuditRequest(GreenLangBase):
    """Batch request for recording multiple audit events atomically.

    Supports validation-only mode (dry run) to check event consistency
    before committing to the hash chain.

    Example:
        >>> batch = BatchAuditRequest(
        ...     events=[event1, event2, event3],
        ...     validate_only=False,
        ... )
    """

    events: List[AuditEventInput] = Field(
        ..., min_length=1, max_length=10000,
        description="List of audit events to record",
    )
    validate_only: bool = Field(
        default=False,
        description="If True, validate events without committing to chain",
    )
    organization_id: Optional[str] = Field(
        default=None, max_length=200,
        description="Override organization_id for all events in the batch",
    )
    reporting_year: Optional[int] = Field(
        default=None, ge=2015, le=2030,
        description="Override reporting_year for all events in the batch",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)


class LineageQueryInput(GreenLangBase):
    """Input for querying the lineage graph from a starting node.

    Supports both forward (source-to-reporting) and backward
    (reporting-to-source) traversal with optional filtering by
    node type and lineage level.

    Example:
        >>> query = LineageQueryInput(
        ...     start_node_id="node-calc-001",
        ...     direction=TraversalDirection.BACKWARD,
        ...     max_depth=5,
        ...     node_type_filter=[LineageNodeType.EMISSION_FACTOR],
        ... )
    """

    start_node_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Starting node identifier for traversal",
    )
    direction: TraversalDirection = Field(
        default=TraversalDirection.BACKWARD,
        description="Traversal direction (forward or backward)",
    )
    max_depth: int = Field(
        default=8, ge=1, le=20,
        description="Maximum traversal depth",
    )
    node_type_filter: Optional[List[LineageNodeType]] = Field(
        default=None,
        description="Only include nodes of these types (None = all)",
    )
    level_filter: Optional[List[LineageLevel]] = Field(
        default=None,
        description="Only include nodes at these levels (None = all)",
    )
    include_edges: bool = Field(
        default=True,
        description="Whether to include edge metadata in results",
    )
    graph_format: GraphFormat = Field(
        default=GraphFormat.ADJACENCY_LIST,
        description="Output format for the lineage graph",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)


class ChainVerificationRequest(GreenLangBase):
    """Request to verify integrity of the audit event hash chain.

    Verifies sequential hash links between events from start_position
    to end_position.  Detects any tampered, missing, or out-of-order
    events.

    Example:
        >>> req = ChainVerificationRequest(
        ...     organization_id="org-001",
        ...     reporting_year=2025,
        ...     start_position=1,
        ...     end_position=500,
        ... )
    """

    organization_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2030,
        description="Reporting year",
    )
    start_position: int = Field(
        default=1, ge=1,
        description="Starting chain position (1-indexed)",
    )
    end_position: Optional[int] = Field(
        default=None, ge=1,
        description="Ending chain position (None = last event)",
    )
    recompute_hashes: bool = Field(
        default=True,
        description="Whether to recompute hashes from payloads for verification",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)

    @validator("end_position")
    def validate_end_after_start(
        cls, v: Optional[int], values: dict
    ) -> Optional[int]:
        """Validate end_position >= start_position when both are set."""
        start = values.get("start_position", 1)
        if v is not None and v < start:
            raise ValueError(
                f"end_position ({v}) must be >= start_position ({start})"
            )
        return v


class RecalculationRequest(GreenLangBase):
    """Request to analyse or execute a recalculation cascade.

    When a change is detected, this request determines which calculations
    are affected and optionally triggers a recalculation.  Supports
    dry-run mode for impact analysis without side effects.

    Example:
        >>> req = RecalculationRequest(
        ...     change_event_id="change-001",
        ...     affected_calculation_ids=["calc-001", "calc-002"],
        ...     cascade=True,
        ...     dry_run=True,
        ... )
    """

    change_event_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Identifier of the change event that triggered recalculation",
    )
    affected_calculation_ids: List[str] = Field(
        default_factory=list,
        description="Calculation IDs known to be affected (may be extended by cascade)",
    )
    cascade: bool = Field(
        default=True,
        description="Whether to cascade to downstream dependent calculations",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, analyse impact without executing recalculation",
    )
    materiality_threshold: MaterialityThreshold = Field(
        default=MaterialityThreshold.QUANTITATIVE_5PCT,
        description="Materiality threshold for determining significance",
    )
    organization_id: Optional[str] = Field(
        default=None, max_length=200,
        description="Organization identifier (inferred from change event if None)",
    )
    reporting_year: Optional[int] = Field(
        default=None, ge=2015, le=2030,
        description="Reporting year (inferred from change event if None)",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant identifier for multi-tenancy",
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# PYDANTIC MODELS -- OUTPUT MODELS (10)
# ==============================================================================


class AuditEventOutput(GreenLangBase):
    """Output from recording a single audit event.

    Contains the event hash, chain position, and verification status
    that confirm the event was successfully appended to the hash chain.

    Example:
        >>> output.event_hash
        'a1b2c3d4e5f6...'
        >>> output.chain_position
        42
    """

    event_id: str = Field(
        ..., description="Unique identifier for the recorded event",
    )
    event_type: AuditEventType = Field(
        ..., description="Type of event recorded",
    )
    event_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of this event",
    )
    chain_position: int = Field(
        ..., ge=1,
        description="Position of this event in the hash chain (1-indexed)",
    )
    prev_event_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the previous event in the chain",
    )
    verification_status: VerificationResult = Field(
        default=VerificationResult.VERIFIED,
        description="Verification status of the chain link",
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp of event creation",
    )
    processing_time_ms: Decimal = Field(
        ..., ge=0,
        description="Time taken to process and record this event (ms)",
    )
    agent_id: str = Field(
        default=AGENT_ID,
        description="Agent that recorded this event",
    )

    model_config = ConfigDict(frozen=True)

    @validator("processing_time_ms")
    def quantize_processing_time(cls, v: Decimal) -> Decimal:
        """Quantize processing time to 2 decimal places."""
        return v.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)


class LineageNodeOutput(GreenLangBase):
    """Output representation of a lineage node."""

    node_id: str = Field(
        ..., description="Unique identifier for the lineage node",
    )
    node_type: LineageNodeType = Field(
        ..., description="Type of lineage node",
    )
    level: LineageLevel = Field(
        ..., description="Lineage depth level",
    )
    qualified_name: str = Field(
        ..., description="Dot-separated qualified name",
    )
    value: Optional[Decimal] = Field(
        default=None, description="Numeric value at this node",
    )
    unit: Optional[str] = Field(
        default=None, description="Unit of measure",
    )
    agent_id: str = Field(
        ..., description="Agent that owns this node",
    )
    data_quality_score: Optional[Decimal] = Field(
        default=None, description="Data quality score",
    )
    node_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the node data",
    )
    created_at: str = Field(
        ..., description="ISO 8601 timestamp of node creation",
    )

    model_config = ConfigDict(frozen=True)


class LineageEdgeOutput(GreenLangBase):
    """Output representation of a lineage edge."""

    edge_id: str = Field(
        ..., description="Unique identifier for the lineage edge",
    )
    source_node_id: str = Field(
        ..., description="Source node identifier",
    )
    target_node_id: str = Field(
        ..., description="Target node identifier",
    )
    edge_type: LineageEdgeType = Field(
        ..., description="Type of edge",
    )
    transformation_description: Optional[str] = Field(
        default=None, description="Description of the transformation",
    )
    confidence: Optional[Decimal] = Field(
        default=None, description="Confidence score",
    )
    created_at: str = Field(
        ..., description="ISO 8601 timestamp of edge creation",
    )

    model_config = ConfigDict(frozen=True)


class LineageGraphOutput(GreenLangBase):
    """Output from a lineage graph query.

    Contains the complete subgraph of nodes and edges that were
    reached during traversal from the starting node.

    Example:
        >>> output.node_count
        15
        >>> output.total_depth
        6
    """

    graph_id: str = Field(
        ..., description="Unique identifier for this graph result",
    )
    nodes: List[LineageNodeOutput] = Field(
        default_factory=list,
        description="Nodes in the traversed subgraph",
    )
    edges: List[LineageEdgeOutput] = Field(
        default_factory=list,
        description="Edges in the traversed subgraph",
    )
    root_nodes: List[str] = Field(
        default_factory=list,
        description="Node IDs with no incoming edges (sources)",
    )
    leaf_nodes: List[str] = Field(
        default_factory=list,
        description="Node IDs with no outgoing edges (sinks)",
    )
    total_depth: int = Field(
        ..., ge=0,
        description="Maximum depth of the traversed subgraph",
    )
    node_count: int = Field(
        ..., ge=0,
        description="Total number of nodes in the subgraph",
    )
    edge_count: int = Field(
        ..., ge=0,
        description="Total number of edges in the subgraph",
    )
    provenance_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the complete graph for integrity verification",
    )

    model_config = ConfigDict(frozen=True)


class LineageChainOutput(GreenLangBase):
    """Output from a linear chain traversal (source to target path).

    Represents a single path through the lineage DAG from a source
    node to a target node, with all intermediate nodes ordered by
    depth level.

    Example:
        >>> output.total_depth
        5
        >>> output.ordered_nodes[0].level
        <LineageLevel.L1_SOURCE: 'L1_SOURCE'>
    """

    chain_id: str = Field(
        ..., description="Unique identifier for this lineage chain",
    )
    ordered_nodes: List[LineageNodeOutput] = Field(
        ...,
        description="Nodes ordered from source to target by depth level",
    )
    total_depth: int = Field(
        ..., ge=1,
        description="Number of nodes in the chain",
    )
    source_node: str = Field(
        ..., description="Node ID of the chain source (root)",
    )
    target_node: str = Field(
        ..., description="Node ID of the chain target (leaf)",
    )
    provenance_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the chain for integrity verification",
    )

    model_config = ConfigDict(frozen=True)


class EvidencePackageOutput(GreenLangBase):
    """Output from evidence package assembly.

    Contains the assembled evidence package metadata, completeness
    score, and cryptographic signature for assurance readiness.

    Example:
        >>> output.completeness_score
        Decimal('0.92')
        >>> output.status
        <EvidencePackageStatus.SEALED: 'sealed'>
    """

    package_id: str = Field(
        ..., description="Unique identifier for the evidence package",
    )
    organization_id: str = Field(
        ..., description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., description="Reporting year",
    )
    status: EvidencePackageStatus = Field(
        ..., description="Current status of the evidence package",
    )
    contents_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of package contents (event counts, node counts, etc.)",
    )
    completeness_score: Decimal = Field(
        ..., ge=Decimal("0.0"), le=Decimal("1.0"),
        description="Fraction of required evidence that is present (0.0-1.0)",
    )
    signature: Optional[str] = Field(
        default=None,
        description="Cryptographic signature of the package",
    )
    signature_algorithm: Optional[SignatureAlgorithm] = Field(
        default=None,
        description="Algorithm used for the signature",
    )
    package_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the complete evidence package",
    )
    framework_coverage: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Coverage percentage per framework (0.0-1.0)",
    )
    assurance_level: AssuranceLevel = Field(
        ..., description="Target assurance level",
    )
    created_at: str = Field(
        ..., description="ISO 8601 timestamp of package creation",
    )

    model_config = ConfigDict(frozen=True)

    @validator("completeness_score")
    def quantize_completeness(cls, v: Decimal) -> Decimal:
        """Quantize completeness score to 4 decimal places."""
        return v.quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)


class ComplianceTraceOutput(GreenLangBase):
    """Output from a compliance trace analysis.

    Maps framework requirements to evidence, identifies coverage
    gaps, and provides an overall compliance status assessment.

    Example:
        >>> output.coverage_pct
        Decimal('0.85')
        >>> output.gaps
        ['E1-6-c: Missing Scope 3 Cat 3 evidence']
    """

    framework: FrameworkIdentifier = Field(
        ..., description="Framework that was traced",
    )
    organization_id: str = Field(
        ..., description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., description="Reporting year",
    )
    requirements_total: int = Field(
        ..., ge=0,
        description="Total number of framework requirements",
    )
    requirements_covered: int = Field(
        ..., ge=0,
        description="Number of requirements with supporting evidence",
    )
    coverage_pct: Decimal = Field(
        ..., ge=Decimal("0.0"), le=Decimal("1.0"),
        description="Coverage ratio (requirements_covered / requirements_total)",
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="List of requirement gaps (requirement ID + description)",
    )
    evidence_map: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Map of requirement ID to list of supporting evidence IDs",
    )
    compliance_status: ComplianceStatus = Field(
        ..., description="Overall compliance status",
    )
    provenance_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the compliance trace result",
    )

    model_config = ConfigDict(frozen=True)

    @validator("coverage_pct")
    def quantize_coverage(cls, v: Decimal) -> Decimal:
        """Quantize coverage to 4 decimal places."""
        return v.quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)


class ChangeDetectionOutput(GreenLangBase):
    """Output from change detection and impact analysis.

    Assesses the impact of a detected change on reported emissions,
    determines materiality, and recommends whether recalculation
    is required.

    Example:
        >>> output.recalculation_required
        True
        >>> output.materiality_pct
        Decimal('2.35')
    """

    change_id: str = Field(
        ..., description="Unique identifier for the detected change",
    )
    change_type: ChangeType = Field(
        ..., description="Type of change detected",
    )
    severity: ChangeSeverity = Field(
        ..., description="Assessed severity of the change",
    )
    affected_calculations_count: int = Field(
        ..., ge=0,
        description="Number of calculations affected by this change",
    )
    materiality_pct: Decimal = Field(
        ..., ge=Decimal("0.0"),
        description="Impact as a percentage of total reported emissions",
    )
    recalculation_required: bool = Field(
        ..., description="Whether recalculation is required",
    )
    impact_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of impact by scope, category, and agent",
    )
    affected_scopes: List[EmissionScope] = Field(
        default_factory=list,
        description="Emission scopes affected by the change",
    )
    affected_categories: List[Scope3Category] = Field(
        default_factory=list,
        description="Scope 3 categories affected by the change",
    )
    provenance_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the change detection result",
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp of the change detection",
    )

    model_config = ConfigDict(frozen=True)

    @validator("materiality_pct")
    def quantize_materiality(cls, v: Decimal) -> Decimal:
        """Quantize materiality percentage to 4 decimal places."""
        return v.quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)


class ChainVerificationOutput(GreenLangBase):
    """Output from hash chain integrity verification.

    Reports whether the chain is intact, how many events were verified,
    and identifies any break points where hashes do not match.

    Example:
        >>> output.chain_status
        <ChainIntegrityStatus.INTACT: 'intact'>
        >>> output.break_points
        []
    """

    chain_status: ChainIntegrityStatus = Field(
        ..., description="Overall chain integrity status",
    )
    total_events: int = Field(
        ..., ge=0,
        description="Total number of events in the verified range",
    )
    verified_events: int = Field(
        ..., ge=0,
        description="Number of events that passed hash verification",
    )
    break_points: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of break point details (position, expected hash, actual hash)",
    )
    verification_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the verification result itself",
    )
    verified_at: str = Field(
        ..., description="ISO 8601 timestamp of verification completion",
    )
    verification_duration_ms: Decimal = Field(
        ..., ge=0,
        description="Duration of the verification process in milliseconds",
    )

    model_config = ConfigDict(frozen=True)

    @validator("verification_duration_ms")
    def quantize_duration(cls, v: Decimal) -> Decimal:
        """Quantize duration to 2 decimal places."""
        return v.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)


class AuditTrailSummary(GreenLangBase):
    """Summary of an organization's audit trail for a reporting period.

    Provides high-level metrics on audit trail completeness, chain
    integrity, lineage coverage, and compliance readiness.

    Example:
        >>> summary.total_events
        1523
        >>> summary.chain_integrity
        <ChainIntegrityStatus.INTACT: 'intact'>
    """

    organization_id: str = Field(
        ..., description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., description="Reporting year",
    )
    total_events: int = Field(
        ..., ge=0,
        description="Total number of audit events recorded",
    )
    events_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Event count breakdown by AuditEventType",
    )
    events_by_scope: Dict[str, int] = Field(
        default_factory=dict,
        description="Event count breakdown by EmissionScope",
    )
    events_by_agent: Dict[str, int] = Field(
        default_factory=dict,
        description="Event count breakdown by agent_id",
    )
    chain_integrity: ChainIntegrityStatus = Field(
        ..., description="Overall hash chain integrity status",
    )
    lineage_node_count: int = Field(
        default=0, ge=0,
        description="Total lineage nodes in the DAG",
    )
    lineage_edge_count: int = Field(
        default=0, ge=0,
        description="Total lineage edges in the DAG",
    )
    lineage_coverage: Decimal = Field(
        default=Decimal("0.0"), ge=Decimal("0.0"), le=Decimal("1.0"),
        description="Fraction of calculations with complete lineage (0.0-1.0)",
    )
    compliance_coverage: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Coverage ratio per framework (0.0-1.0)",
    )
    trail_status: AuditTrailStatus = Field(
        default=AuditTrailStatus.ACTIVE,
        description="Current status of the audit trail",
    )
    first_event_at: Optional[str] = Field(
        default=None, description="ISO 8601 timestamp of the first event",
    )
    last_updated: str = Field(
        ..., description="ISO 8601 timestamp of the last event or update",
    )
    provenance_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the summary for integrity",
    )

    model_config = ConfigDict(frozen=True)

    @validator("lineage_coverage")
    def quantize_lineage_coverage(cls, v: Decimal) -> Decimal:
        """Quantize lineage coverage to 4 decimal places."""
        return v.quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)


class ComplianceReport(GreenLangBase):
    """Full compliance report for a specific framework.

    Provides a comprehensive assessment of compliance readiness,
    evidence sufficiency, requirement-level detail, and actionable
    recommendations.

    Example:
        >>> report.compliance_status
        <ComplianceStatus.PARTIAL: 'partial'>
        >>> report.evidence_sufficiency
        Decimal('0.87')
    """

    framework: FrameworkIdentifier = Field(
        ..., description="Target compliance framework",
    )
    framework_name: str = Field(
        ..., description="Human-readable framework name",
    )
    organization_id: str = Field(
        ..., description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., description="Reporting year",
    )
    compliance_status: ComplianceStatus = Field(
        ..., description="Overall compliance status",
    )
    evidence_sufficiency: Decimal = Field(
        ..., ge=Decimal("0.0"), le=Decimal("1.0"),
        description="Fraction of requirements with sufficient evidence (0.0-1.0)",
    )
    requirements: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of requirements with status and evidence references",
    )
    gaps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of identified gaps with severity and remediation",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Prioritized recommendations for closing gaps",
    )
    assurance_readiness: Decimal = Field(
        ..., ge=Decimal("0.0"), le=Decimal("1.0"),
        description="Readiness score for external assurance (0.0-1.0)",
    )
    assurance_level_achievable: AssuranceLevel = Field(
        ..., description="Highest assurance level achievable with current evidence",
    )
    provenance_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the compliance report",
    )
    generated_at: str = Field(
        ..., description="ISO 8601 timestamp of report generation",
    )

    model_config = ConfigDict(frozen=True)

    @validator("evidence_sufficiency")
    def quantize_sufficiency(cls, v: Decimal) -> Decimal:
        """Quantize evidence sufficiency to 4 decimal places."""
        return v.quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)

    @validator("assurance_readiness")
    def quantize_readiness(cls, v: Decimal) -> Decimal:
        """Quantize assurance readiness to 4 decimal places."""
        return v.quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)


# ==============================================================================
# PYDANTIC MODELS -- AGGREGATED RESULT MODEL (1)
# ==============================================================================


class AuditTrailCalculationResult(GreenLangBase):
    """Top-level result from the Audit Trail & Lineage Agent.

    Encapsulates all outputs from a complete audit trail processing run,
    including event recording, lineage graph construction, compliance
    tracing, and chain integrity verification.

    This is the primary return type from the agent's ``process()`` method.

    Example:
        >>> result = agent.process(input_data)
        >>> result.chain_integrity
        <ChainIntegrityStatus.INTACT: 'intact'>
        >>> result.errors
        []
    """

    calculation_id: str = Field(
        ..., description="Unique identifier for this audit trail calculation run",
    )
    organization_id: str = Field(
        ..., description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., description="Reporting year",
    )
    events_recorded: int = Field(
        ..., ge=0,
        description="Total number of audit events recorded in this run",
    )
    lineage_nodes_created: int = Field(
        ..., ge=0,
        description="Total number of lineage nodes created in this run",
    )
    lineage_edges_created: int = Field(
        ..., ge=0,
        description="Total number of lineage edges created in this run",
    )
    compliance_traces: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Compliance coverage per framework (0.0-1.0)",
    )
    chain_integrity: ChainIntegrityStatus = Field(
        ..., description="Hash chain integrity after this run",
    )
    trail_status: AuditTrailStatus = Field(
        default=AuditTrailStatus.ACTIVE,
        description="Audit trail status after this run",
    )
    provenance_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 provenance hash of the complete result",
    )
    processing_time_ms: Decimal = Field(
        ..., ge=0,
        description="Total processing time in milliseconds",
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp of result creation",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of errors encountered during processing",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warnings generated during processing",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional result metadata",
    )
    pipeline_stage: PipelineStage = Field(
        default=PipelineStage.SEAL,
        description="Final pipeline stage reached",
    )

    model_config = ConfigDict(frozen=True)

    @validator("processing_time_ms")
    def quantize_processing_time(cls, v: Decimal) -> Decimal:
        """Quantize processing time to 2 decimal places."""
        return v.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)


# ==============================================================================
# PYDANTIC MODELS -- INTERNAL / PROVENANCE MODELS (4)
# ==============================================================================


class ProvenanceRecord(GreenLangBase):
    """Provenance record for the 10-stage pipeline audit trail.

    Each pipeline stage produces a provenance record that is
    hash-chained with the previous stage's record.
    """

    record_id: str = Field(
        ..., description="Unique provenance record identifier",
    )
    sha256_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the record data",
    )
    parent_hash: Optional[str] = Field(
        default=None, min_length=64, max_length=64,
        description="SHA-256 hash of the parent record (chain link)",
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp of record creation",
    )
    operation: str = Field(
        ..., description="Operation that produced this record",
    )
    stage: PipelineStage = Field(
        ..., description="Pipeline stage",
    )
    agent_id: str = Field(
        default=AGENT_ID,
        description="Agent that produced this record",
    )

    model_config = ConfigDict(frozen=True)


class BatchAuditResult(GreenLangBase):
    """Result from a batch audit event recording operation.

    Includes per-event results and aggregate statistics for the batch.
    """

    batch_id: str = Field(
        ..., description="Unique identifier for this batch operation",
    )
    total_events: int = Field(
        ..., ge=0, description="Total events submitted in the batch",
    )
    events_recorded: int = Field(
        ..., ge=0, description="Events successfully recorded",
    )
    events_failed: int = Field(
        ..., ge=0, description="Events that failed validation or recording",
    )
    validation_only: bool = Field(
        ..., description="Whether this was a validation-only (dry run) batch",
    )
    event_results: List[AuditEventOutput] = Field(
        default_factory=list,
        description="Per-event recording results",
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-event error details",
    )
    chain_integrity: ChainIntegrityStatus = Field(
        ..., description="Chain integrity after batch processing",
    )
    provenance_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the batch result",
    )
    processing_time_ms: Decimal = Field(
        ..., ge=0,
        description="Total batch processing time in milliseconds",
    )

    model_config = ConfigDict(frozen=True)

    @validator("processing_time_ms")
    def quantize_processing_time(cls, v: Decimal) -> Decimal:
        """Quantize processing time to 2 decimal places."""
        return v.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)


class RecalculationImpact(GreenLangBase):
    """Impact analysis result from a recalculation request.

    Details which calculations, scopes, and categories are affected
    by a change, and provides old vs new emission values.
    """

    recalculation_id: str = Field(
        ..., description="Unique identifier for this recalculation analysis",
    )
    change_event_id: str = Field(
        ..., description="Change event that triggered this analysis",
    )
    dry_run: bool = Field(
        ..., description="Whether this was a dry-run (no actual recalculation)",
    )
    affected_calculations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of affected calculations with old/new values",
    )
    total_affected: int = Field(
        ..., ge=0,
        description="Total number of affected calculations",
    )
    cascade_depth: int = Field(
        ..., ge=0,
        description="Maximum depth of the recalculation cascade",
    )
    total_impact_pct: Decimal = Field(
        ..., ge=Decimal("0.0"),
        description="Total impact as percentage of reported emissions",
    )
    old_total_tco2e: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Previous total emissions in tCO2e (before change)",
    )
    new_total_tco2e: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Revised total emissions in tCO2e (after change)",
    )
    delta_tco2e: Optional[Decimal] = Field(
        default=None,
        description="Change in total emissions in tCO2e (new - old)",
    )
    materiality_assessment: MaterialityThreshold = Field(
        ..., description="Materiality threshold classification",
    )
    recalculation_required: bool = Field(
        ..., description="Whether recalculation is required per materiality policy",
    )
    affected_scopes: List[EmissionScope] = Field(
        default_factory=list,
        description="Emission scopes affected",
    )
    affected_categories: List[Scope3Category] = Field(
        default_factory=list,
        description="Scope 3 categories affected",
    )
    provenance_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the recalculation impact analysis",
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp of the analysis",
    )

    model_config = ConfigDict(frozen=True)

    @validator("total_impact_pct")
    def quantize_impact(cls, v: Decimal) -> Decimal:
        """Quantize impact percentage to 4 decimal places."""
        return v.quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)


class DataQualityAssessment(GreenLangBase):
    """Data quality assessment summary for audit trail evidence.

    Aggregates data quality scores across all lineage nodes in the
    trail to provide an overall quality profile.
    """

    organization_id: str = Field(
        ..., description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., description="Reporting year",
    )
    overall_tier: DataQualityTier = Field(
        ..., description="Overall (weighted) data quality tier",
    )
    tier_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of nodes by data quality tier",
    )
    weighted_score: Decimal = Field(
        ..., ge=Decimal("1.0"), le=Decimal("5.0"),
        description="Emission-weighted average data quality score",
    )
    by_scope: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Average data quality score per emission scope",
    )
    by_agent: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Average data quality score per MRV agent",
    )
    improvement_priorities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Prioritized list of DQ improvement opportunities",
    )
    uncertainty_pct: Decimal = Field(
        ..., ge=0,
        description="Implied uncertainty percentage from data quality",
    )
    provenance_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of the DQ assessment",
    )

    model_config = ConfigDict(frozen=True)

    @validator("weighted_score")
    def quantize_weighted_score(cls, v: Decimal) -> Decimal:
        """Quantize weighted score to 2 decimal places."""
        return v.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)

    @validator("uncertainty_pct")
    def quantize_uncertainty(cls, v: Decimal) -> Decimal:
        """Quantize uncertainty to 2 decimal places."""
        return v.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)
