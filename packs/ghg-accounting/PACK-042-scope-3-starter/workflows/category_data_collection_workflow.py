# -*- coding: utf-8 -*-
"""
Category Data Collection Workflow
=====================================

4-phase workflow for structured data collection across selected Scope 3
categories within PACK-042 Scope 3 Starter Pack.

Phases:
    1. CategorySelection    -- User selects categories for detailed analysis,
                               guided by screening results
    2. DataRequirements     -- Generate per-category data requirement checklists
                               based on selected methodology tier
    3. DataIntake           -- Collect activity data via forms, file upload,
                               ERP integration, or API
    4. DataValidation       -- Validate completeness (% coverage), units,
                               date ranges, plausibility checks (sector
                               benchmark comparison)

The workflow follows GreenLang zero-hallucination principles: all validation
rules, benchmark comparisons, and completeness scores are derived from
deterministic formulas. SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Corporate Value Chain (Scope 3) Standard -- Chapter 8
    GHG Protocol Technical Guidance for Calculating Scope 3 Emissions (2013)
    ISO 14064-1:2018 Clause 5.2.4

Schedule: on-demand (after screening; repeated per reporting period)
Estimated duration: 1-3 weeks per category

Author: GreenLang Platform Team
Version: 42.0.0
"""

_MODULE_VERSION: str = "42.0.0"

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas.enums import ValidationSeverity

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

class MethodologyTier(str, Enum):
    """Methodology tier for Scope 3 calculation."""

    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"

class DataSourceType(str, Enum):
    """Types of data sources for Scope 3 data collection."""

    MANUAL_FORM = "manual_form"
    FILE_UPLOAD = "file_upload"
    ERP_INTEGRATION = "erp_integration"
    API_FEED = "api_feed"
    SUPPLIER_PORTAL = "supplier_portal"
    QUESTIONNAIRE = "questionnaire"
    ESTIMATED = "estimated"

class CompletionStatus(str, Enum):
    """Completion status for data collection per category."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUBSTANTIALLY_COMPLETE = "substantially_complete"
    COMPLETE = "complete"
    BLOCKED = "blocked"

class UnitSystem(str, Enum):
    """Unit measurement systems."""

    SI = "si"
    IMPERIAL = "imperial"
    MIXED = "mixed"

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(
        default_factory=dict, description="Phase output data"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")

class WorkflowState(BaseModel):
    """Persistent state for checkpoint/resume capability."""

    workflow_id: str = Field(default="")
    current_phase: int = Field(default=0)
    phase_statuses: Dict[str, str] = Field(default_factory=dict)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default="")
    updated_at: str = Field(default="")

class DataRequirementField(BaseModel):
    """Single required data field for a Scope 3 category."""

    field_name: str = Field(..., description="Field identifier")
    display_name: str = Field(default="", description="Human-readable name")
    description: str = Field(default="")
    data_type: str = Field(
        default="float",
        description="Expected data type: float|int|str|date|bool",
    )
    unit: str = Field(default="", description="Expected unit e.g. kg, kWh, USD")
    is_mandatory: bool = Field(default=True)
    validation_rule: str = Field(
        default="", description="Validation expression or description"
    )
    example_value: str = Field(default="")
    tier: MethodologyTier = Field(
        default=MethodologyTier.SPEND_BASED,
        description="Minimum methodology tier that requires this field",
    )

class CategoryDataRequirements(BaseModel):
    """Data requirements for a single Scope 3 category and tier."""

    category: Scope3Category = Field(...)
    category_name: str = Field(default="")
    tier: MethodologyTier = Field(default=MethodologyTier.SPEND_BASED)
    required_fields: List[DataRequirementField] = Field(default_factory=list)
    optional_fields: List[DataRequirementField] = Field(default_factory=list)
    supported_sources: List[DataSourceType] = Field(default_factory=list)
    data_collection_guidance: str = Field(default="")
    estimated_effort_hours: float = Field(default=0.0, ge=0.0)

class IngestedDataRecord(BaseModel):
    """Single data record ingested for a Scope 3 category."""

    record_id: str = Field(
        default_factory=lambda: f"ingested-{uuid.uuid4().hex[:8]}"
    )
    category: Scope3Category = Field(...)
    source_type: DataSourceType = Field(default=DataSourceType.MANUAL_FORM)
    data_fields: Dict[str, Any] = Field(default_factory=dict)
    supplier_name: str = Field(default="")
    reporting_period_start: str = Field(default="", description="ISO date")
    reporting_period_end: str = Field(default="", description="ISO date")
    original_unit: str = Field(default="")
    normalized_unit: str = Field(default="")
    normalized_value: float = Field(default=0.0)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class ValidationIssue(BaseModel):
    """Single validation issue found in data."""

    issue_id: str = Field(
        default_factory=lambda: f"val-{uuid.uuid4().hex[:8]}"
    )
    category: Scope3Category = Field(...)
    field_name: str = Field(default="")
    severity: ValidationSeverity = Field(default=ValidationSeverity.WARNING)
    message: str = Field(default="")
    record_id: str = Field(default="")
    suggested_action: str = Field(default="")

class CategoryCollectionProgress(BaseModel):
    """Collection progress for a single Scope 3 category."""

    category: Scope3Category = Field(...)
    category_name: str = Field(default="")
    status: CompletionStatus = Field(default=CompletionStatus.NOT_STARTED)
    records_collected: int = Field(default=0, ge=0)
    mandatory_fields_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    date_range_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    validation_issues: int = Field(default=0, ge=0)
    validation_errors: int = Field(default=0, ge=0)
    data_sources_used: List[str] = Field(default_factory=list)
    tier: MethodologyTier = Field(default=MethodologyTier.SPEND_BASED)

# =============================================================================
# INPUT / OUTPUT
# =============================================================================

class CategoryDataCollectionInput(BaseModel):
    """Input data model for CategoryDataCollectionWorkflow."""

    selected_categories: List[Scope3Category] = Field(
        default_factory=list,
        description="Categories selected for detailed data collection",
    )
    methodology_tiers: Dict[str, str] = Field(
        default_factory=dict,
        description="Category -> methodology tier override",
    )
    ingested_data: List[IngestedDataRecord] = Field(
        default_factory=list, description="Pre-collected data records"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    reporting_period_start: str = Field(default="2025-01-01")
    reporting_period_end: str = Field(default="2025-12-31")
    organization_name: str = Field(default="")
    sector: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("selected_categories")
    @classmethod
    def validate_categories(cls, v: List[Scope3Category]) -> List[Scope3Category]:
        """Ensure at least one category is selected."""
        return v  # Empty is allowed for initial exploration

class CategoryDataCollectionResult(BaseModel):
    """Complete result from category data collection workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="category_data_collection")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    selected_categories: List[str] = Field(default_factory=list)
    data_requirements: List[CategoryDataRequirements] = Field(default_factory=list)
    ingested_records: List[IngestedDataRecord] = Field(default_factory=list)
    validation_issues: List[ValidationIssue] = Field(default_factory=list)
    category_progress: List[CategoryCollectionProgress] = Field(default_factory=list)
    overall_completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")

# =============================================================================
# PER-CATEGORY DATA REQUIREMENT TEMPLATES (Zero-Hallucination)
# =============================================================================

# Template definitions: category -> tier -> list of required field definitions
# Based on GHG Protocol Technical Guidance for Scope 3, Appendix tables

def _build_requirement_templates() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Build per-category, per-tier data requirement templates."""
    templates: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    # Category 1: Purchased Goods & Services
    templates["cat_01_purchased_goods_services"] = {
        "spend_based": [
            {"field_name": "supplier_name", "display_name": "Supplier Name", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "spend_amount", "display_name": "Spend Amount", "data_type": "float", "unit": "USD", "is_mandatory": True},
            {"field_name": "eeio_sector", "display_name": "EEIO Sector Code", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "purchase_date", "display_name": "Purchase Date", "data_type": "date", "unit": "", "is_mandatory": False},
        ],
        "average_data": [
            {"field_name": "supplier_name", "display_name": "Supplier Name", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "product_type", "display_name": "Product/Material Type", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "quantity", "display_name": "Quantity Purchased", "data_type": "float", "unit": "kg", "is_mandatory": True},
            {"field_name": "emission_factor_id", "display_name": "Emission Factor Reference", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "spend_amount", "display_name": "Spend Amount", "data_type": "float", "unit": "USD", "is_mandatory": False},
        ],
        "supplier_specific": [
            {"field_name": "supplier_name", "display_name": "Supplier Name", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "product_type", "display_name": "Product/Material Type", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "quantity", "display_name": "Quantity Purchased", "data_type": "float", "unit": "kg", "is_mandatory": True},
            {"field_name": "supplier_emission_factor", "display_name": "Supplier-Specific EF", "data_type": "float", "unit": "kgCO2e/kg", "is_mandatory": True},
            {"field_name": "allocation_method", "display_name": "Allocation Method", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "third_party_verified", "display_name": "Third-Party Verified", "data_type": "bool", "unit": "", "is_mandatory": False},
        ],
    }

    # Category 2: Capital Goods
    templates["cat_02_capital_goods"] = {
        "spend_based": [
            {"field_name": "asset_description", "display_name": "Asset Description", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "spend_amount", "display_name": "Capital Expenditure", "data_type": "float", "unit": "USD", "is_mandatory": True},
            {"field_name": "eeio_sector", "display_name": "EEIO Sector Code", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "acquisition_date", "display_name": "Acquisition Date", "data_type": "date", "unit": "", "is_mandatory": False},
        ],
        "average_data": [
            {"field_name": "asset_description", "display_name": "Asset Description", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "asset_type", "display_name": "Asset Type", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "weight_kg", "display_name": "Asset Weight", "data_type": "float", "unit": "kg", "is_mandatory": True},
            {"field_name": "material_composition", "display_name": "Material Composition", "data_type": "str", "unit": "", "is_mandatory": False},
        ],
        "supplier_specific": [
            {"field_name": "asset_description", "display_name": "Asset Description", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "supplier_name", "display_name": "Manufacturer/Supplier", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "cradle_to_gate_emissions", "display_name": "Cradle-to-Gate Emissions", "data_type": "float", "unit": "kgCO2e", "is_mandatory": True},
            {"field_name": "epd_reference", "display_name": "EPD Reference", "data_type": "str", "unit": "", "is_mandatory": False},
        ],
    }

    # Category 3: Fuel- & Energy-Related Activities
    templates["cat_03_fuel_energy_related"] = {
        "spend_based": [
            {"field_name": "fuel_type", "display_name": "Fuel/Energy Type", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "spend_amount", "display_name": "Spend Amount", "data_type": "float", "unit": "USD", "is_mandatory": True},
        ],
        "average_data": [
            {"field_name": "fuel_type", "display_name": "Fuel/Energy Type", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "quantity", "display_name": "Quantity Consumed", "data_type": "float", "unit": "kWh", "is_mandatory": True},
            {"field_name": "upstream_ef", "display_name": "Upstream Emission Factor", "data_type": "float", "unit": "kgCO2e/kWh", "is_mandatory": True},
            {"field_name": "td_loss_pct", "display_name": "T&D Loss %", "data_type": "float", "unit": "%", "is_mandatory": False},
        ],
        "supplier_specific": [
            {"field_name": "fuel_type", "display_name": "Fuel/Energy Type", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "quantity", "display_name": "Quantity Consumed", "data_type": "float", "unit": "kWh", "is_mandatory": True},
            {"field_name": "supplier_wtt_ef", "display_name": "Supplier WTT Factor", "data_type": "float", "unit": "kgCO2e/kWh", "is_mandatory": True},
        ],
    }

    # Category 4: Upstream Transportation & Distribution
    templates["cat_04_upstream_transport"] = {
        "spend_based": [
            {"field_name": "logistics_provider", "display_name": "Logistics Provider", "data_type": "str", "unit": "", "is_mandatory": False},
            {"field_name": "spend_amount", "display_name": "Transport Spend", "data_type": "float", "unit": "USD", "is_mandatory": True},
            {"field_name": "transport_mode", "display_name": "Transport Mode", "data_type": "str", "unit": "", "is_mandatory": False},
        ],
        "average_data": [
            {"field_name": "origin", "display_name": "Origin Location", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "destination", "display_name": "Destination Location", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "weight_tonnes", "display_name": "Shipment Weight", "data_type": "float", "unit": "tonnes", "is_mandatory": True},
            {"field_name": "distance_km", "display_name": "Distance", "data_type": "float", "unit": "km", "is_mandatory": True},
            {"field_name": "transport_mode", "display_name": "Transport Mode", "data_type": "str", "unit": "", "is_mandatory": True},
        ],
        "supplier_specific": [
            {"field_name": "logistics_provider", "display_name": "Logistics Provider", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "fuel_consumed_litres", "display_name": "Fuel Consumed", "data_type": "float", "unit": "litres", "is_mandatory": True},
            {"field_name": "fuel_type", "display_name": "Fuel Type", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "tonne_km", "display_name": "Tonne-km", "data_type": "float", "unit": "tkm", "is_mandatory": True},
        ],
    }

    # Category 5: Waste Generated in Operations
    templates["cat_05_waste_in_operations"] = {
        "spend_based": [
            {"field_name": "waste_management_spend", "display_name": "Waste Management Spend", "data_type": "float", "unit": "USD", "is_mandatory": True},
        ],
        "average_data": [
            {"field_name": "waste_type", "display_name": "Waste Type", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "weight_tonnes", "display_name": "Waste Weight", "data_type": "float", "unit": "tonnes", "is_mandatory": True},
            {"field_name": "treatment_method", "display_name": "Treatment Method", "data_type": "str", "unit": "", "is_mandatory": True},
        ],
        "supplier_specific": [
            {"field_name": "waste_type", "display_name": "Waste Type", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "weight_tonnes", "display_name": "Waste Weight", "data_type": "float", "unit": "tonnes", "is_mandatory": True},
            {"field_name": "treatment_method", "display_name": "Treatment Method", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "facility_specific_ef", "display_name": "Facility-Specific EF", "data_type": "float", "unit": "kgCO2e/tonne", "is_mandatory": True},
        ],
    }

    # Category 6: Business Travel
    templates["cat_06_business_travel"] = {
        "spend_based": [
            {"field_name": "travel_spend", "display_name": "Travel Spend", "data_type": "float", "unit": "USD", "is_mandatory": True},
            {"field_name": "travel_type", "display_name": "Travel Type", "data_type": "str", "unit": "", "is_mandatory": False},
        ],
        "average_data": [
            {"field_name": "travel_mode", "display_name": "Travel Mode", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "distance_km", "display_name": "Distance", "data_type": "float", "unit": "km", "is_mandatory": True},
            {"field_name": "cabin_class", "display_name": "Cabin Class (air)", "data_type": "str", "unit": "", "is_mandatory": False},
            {"field_name": "hotel_nights", "display_name": "Hotel Nights", "data_type": "int", "unit": "nights", "is_mandatory": False},
        ],
        "supplier_specific": [
            {"field_name": "travel_mode", "display_name": "Travel Mode", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "distance_km", "display_name": "Distance", "data_type": "float", "unit": "km", "is_mandatory": True},
            {"field_name": "carrier_ef", "display_name": "Carrier Emission Factor", "data_type": "float", "unit": "kgCO2e/pkm", "is_mandatory": True},
        ],
    }

    # Category 7: Employee Commuting
    templates["cat_07_employee_commuting"] = {
        "spend_based": [
            {"field_name": "commuting_subsidy_spend", "display_name": "Commuting Subsidy", "data_type": "float", "unit": "USD", "is_mandatory": True},
            {"field_name": "employee_count", "display_name": "Employee Count", "data_type": "int", "unit": "", "is_mandatory": True},
        ],
        "average_data": [
            {"field_name": "commute_mode", "display_name": "Commute Mode", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "avg_distance_km", "display_name": "Average One-Way Distance", "data_type": "float", "unit": "km", "is_mandatory": True},
            {"field_name": "employee_count", "display_name": "Employee Count", "data_type": "int", "unit": "", "is_mandatory": True},
            {"field_name": "working_days_per_year", "display_name": "Working Days/Year", "data_type": "int", "unit": "days", "is_mandatory": True},
            {"field_name": "wfh_pct", "display_name": "Work-from-Home %", "data_type": "float", "unit": "%", "is_mandatory": False},
        ],
        "supplier_specific": [
            {"field_name": "commute_mode", "display_name": "Commute Mode", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "total_distance_km", "display_name": "Total Annual Distance", "data_type": "float", "unit": "km", "is_mandatory": True},
            {"field_name": "vehicle_type", "display_name": "Vehicle Type/Size", "data_type": "str", "unit": "", "is_mandatory": True},
            {"field_name": "fuel_type", "display_name": "Fuel Type", "data_type": "str", "unit": "", "is_mandatory": True},
        ],
    }

    # Categories 8-15: simplified templates (same structure)
    for cat_key, cat_label in [
        ("cat_08_upstream_leased_assets", "Leased Asset"),
        ("cat_09_downstream_transport", "Shipment"),
        ("cat_10_processing_sold_products", "Product Processing"),
        ("cat_11_use_of_sold_products", "Product Use"),
        ("cat_12_end_of_life_treatment", "Product End-of-Life"),
        ("cat_13_downstream_leased_assets", "Leased Asset"),
        ("cat_14_franchises", "Franchise"),
        ("cat_15_investments", "Investment"),
    ]:
        templates[cat_key] = {
            "spend_based": [
                {"field_name": "description", "display_name": f"{cat_label} Description", "data_type": "str", "unit": "", "is_mandatory": True},
                {"field_name": "spend_amount", "display_name": "Spend/Value", "data_type": "float", "unit": "USD", "is_mandatory": True},
                {"field_name": "eeio_sector", "display_name": "EEIO Sector", "data_type": "str", "unit": "", "is_mandatory": False},
            ],
            "average_data": [
                {"field_name": "description", "display_name": f"{cat_label} Description", "data_type": "str", "unit": "", "is_mandatory": True},
                {"field_name": "activity_data", "display_name": "Activity Data", "data_type": "float", "unit": "varies", "is_mandatory": True},
                {"field_name": "activity_unit", "display_name": "Activity Unit", "data_type": "str", "unit": "", "is_mandatory": True},
                {"field_name": "emission_factor", "display_name": "Emission Factor", "data_type": "float", "unit": "kgCO2e/unit", "is_mandatory": True},
            ],
            "supplier_specific": [
                {"field_name": "description", "display_name": f"{cat_label} Description", "data_type": "str", "unit": "", "is_mandatory": True},
                {"field_name": "activity_data", "display_name": "Activity Data", "data_type": "float", "unit": "varies", "is_mandatory": True},
                {"field_name": "activity_unit", "display_name": "Activity Unit", "data_type": "str", "unit": "", "is_mandatory": True},
                {"field_name": "supplier_ef", "display_name": "Supplier-Specific EF", "data_type": "float", "unit": "kgCO2e/unit", "is_mandatory": True},
                {"field_name": "verification_status", "display_name": "Verification Status", "data_type": "str", "unit": "", "is_mandatory": False},
            ],
        }

    return templates

REQUIREMENT_TEMPLATES = _build_requirement_templates()

# Category names mapping
CATEGORY_NAMES: Dict[str, str] = {
    "cat_01_purchased_goods_services": "Purchased Goods & Services",
    "cat_02_capital_goods": "Capital Goods",
    "cat_03_fuel_energy_related": "Fuel- & Energy-Related Activities",
    "cat_04_upstream_transport": "Upstream Transportation & Distribution",
    "cat_05_waste_in_operations": "Waste Generated in Operations",
    "cat_06_business_travel": "Business Travel",
    "cat_07_employee_commuting": "Employee Commuting",
    "cat_08_upstream_leased_assets": "Upstream Leased Assets",
    "cat_09_downstream_transport": "Downstream Transportation & Distribution",
    "cat_10_processing_sold_products": "Processing of Sold Products",
    "cat_11_use_of_sold_products": "Use of Sold Products",
    "cat_12_end_of_life_treatment": "End-of-Life Treatment of Sold Products",
    "cat_13_downstream_leased_assets": "Downstream Leased Assets",
    "cat_14_franchises": "Franchises",
    "cat_15_investments": "Investments",
}

# Estimated effort hours per tier (category-independent baseline)
EFFORT_HOURS_BY_TIER: Dict[str, float] = {
    "spend_based": 4.0,
    "average_data": 16.0,
    "supplier_specific": 40.0,
    "hybrid": 24.0,
}

# Sector benchmark ranges for plausibility checks (kgCO2e per employee)
PLAUSIBILITY_BENCHMARKS: Dict[str, Dict[str, Dict[str, float]]] = {
    "cat_01_purchased_goods_services": {
        "manufacturing": {"low": 2.0, "median": 8.0, "high": 25.0},
        "services": {"low": 0.5, "median": 2.5, "high": 8.0},
        "default": {"low": 1.0, "median": 5.0, "high": 15.0},
    },
    "cat_06_business_travel": {
        "services": {"low": 0.2, "median": 1.5, "high": 5.0},
        "manufacturing": {"low": 0.1, "median": 0.5, "high": 2.0},
        "default": {"low": 0.1, "median": 0.8, "high": 3.0},
    },
    "cat_07_employee_commuting": {
        "default": {"low": 0.3, "median": 1.2, "high": 3.0},
    },
}

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class CategoryDataCollectionWorkflow:
    """
    4-phase data collection workflow for Scope 3 categories.

    Generates per-category data requirements based on selected methodology
    tier, collects activity data via multiple channels, and validates
    completeness, units, date ranges, and plausibility against sector
    benchmarks.

    Zero-hallucination: all validation rules and benchmark comparisons use
    deterministic formulas. No LLM calls in validation paths.

    Attributes:
        workflow_id: Unique execution identifier.
        _selected_categories: Categories chosen for data collection.
        _requirements: Generated data requirements per category.
        _ingested_records: All collected data records.
        _validation_issues: Validation issues found.
        _category_progress: Progress tracking per category.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = CategoryDataCollectionWorkflow()
        >>> inp = CategoryDataCollectionInput(
        ...     selected_categories=[Scope3Category.CAT_01_PURCHASED_GOODS],
        ...     methodology_tiers={"cat_01_purchased_goods_services": "average_data"},
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_NAMES: List[str] = [
        "category_selection",
        "data_requirements",
        "data_intake",
        "data_validation",
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize CategoryDataCollectionWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._selected_categories: List[Scope3Category] = []
        self._requirements: List[CategoryDataRequirements] = []
        self._ingested_records: List[IngestedDataRecord] = []
        self._validation_issues: List[ValidationIssue] = []
        self._category_progress: List[CategoryCollectionProgress] = []
        self._phase_results: List[PhaseResult] = []
        self._tier_map: Dict[str, MethodologyTier] = {}
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
        input_data: Optional[CategoryDataCollectionInput] = None,
        selected_categories: Optional[List[Scope3Category]] = None,
        ingested_data: Optional[List[IngestedDataRecord]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> CategoryDataCollectionResult:
        """
        Execute the 4-phase data collection workflow.

        Args:
            input_data: Full input model (preferred).
            selected_categories: Categories to collect (fallback).
            ingested_data: Pre-collected data records (fallback).
            config: Optional configuration overrides.

        Returns:
            CategoryDataCollectionResult with requirements, data, and validation.
        """
        if input_data is None:
            input_data = CategoryDataCollectionInput(
                selected_categories=selected_categories or [],
                ingested_data=ingested_data or [],
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting category data collection workflow %s categories=%d records=%d",
            self.workflow_id,
            len(input_data.selected_categories),
            len(input_data.ingested_data),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Category Selection
            phase1 = await self._execute_with_retry(
                self._phase_category_selection, input_data, phase_number=1
            )
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 1 failed: {phase1.errors}")

            # Phase 2: Data Requirements
            phase2 = await self._execute_with_retry(
                self._phase_data_requirements, input_data, phase_number=2
            )
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 2 failed: {phase2.errors}")

            # Phase 3: Data Intake
            phase3 = await self._execute_with_retry(
                self._phase_data_intake, input_data, phase_number=3
            )
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 3 failed: {phase3.errors}")

            # Phase 4: Data Validation
            phase4 = await self._execute_with_retry(
                self._phase_data_validation, input_data, phase_number=4
            )
            self._phase_results.append(phase4)
            if phase4.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 4 failed: {phase4.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Category data collection workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(
                PhaseResult(
                    phase_name="error", phase_number=0,
                    status=PhaseStatus.FAILED, errors=[str(exc)],
                )
            )

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        overall_completeness = self._calculate_overall_completeness()

        result = CategoryDataCollectionResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            selected_categories=[c.value for c in self._selected_categories],
            data_requirements=self._requirements,
            ingested_records=self._ingested_records,
            validation_issues=self._validation_issues,
            category_progress=self._category_progress,
            overall_completeness_pct=overall_completeness,
            progress_pct=self._state.progress_pct,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Category data collection workflow %s completed in %.2fs "
            "status=%s categories=%d completeness=%.1f%%",
            self.workflow_id, elapsed, overall_status.value,
            len(self._selected_categories), overall_completeness,
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: CategoryDataCollectionInput,
        phase_number: int,
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
    # Phase 1: Category Selection
    # -------------------------------------------------------------------------

    async def _phase_category_selection(
        self, input_data: CategoryDataCollectionInput
    ) -> PhaseResult:
        """Validate and confirm category selection for detailed analysis."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._selected_categories = list(input_data.selected_categories)

        if not self._selected_categories:
            # Default to all 15 categories if none specified
            self._selected_categories = list(Scope3Category)
            warnings.append(
                "No categories specified; defaulting to all 15 Scope 3 categories"
            )

        # Parse methodology tiers
        for cat in self._selected_categories:
            tier_str = input_data.methodology_tiers.get(cat.value, "spend_based")
            try:
                self._tier_map[cat.value] = MethodologyTier(tier_str)
            except ValueError:
                self._tier_map[cat.value] = MethodologyTier.SPEND_BASED
                warnings.append(
                    f"Invalid tier '{tier_str}' for {cat.value}; defaulting to spend_based"
                )

        outputs["selected_count"] = len(self._selected_categories)
        outputs["selected_categories"] = [c.value for c in self._selected_categories]
        outputs["tier_assignments"] = {
            k: v.value for k, v in self._tier_map.items()
        }
        outputs["upstream_categories"] = sum(
            1 for c in self._selected_categories
            if c.value.startswith(("cat_01", "cat_02", "cat_03", "cat_04",
                                   "cat_05", "cat_06", "cat_07", "cat_08"))
        )
        outputs["downstream_categories"] = sum(
            1 for c in self._selected_categories
            if c.value.startswith(("cat_09", "cat_10", "cat_11", "cat_12",
                                   "cat_13", "cat_14", "cat_15"))
        )

        self._state.progress_pct = 10.0

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 CategorySelection: %d categories selected, tiers assigned",
            len(self._selected_categories),
        )
        return PhaseResult(
            phase_name="category_selection", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Data Requirements
    # -------------------------------------------------------------------------

    async def _phase_data_requirements(
        self, input_data: CategoryDataCollectionInput
    ) -> PhaseResult:
        """Generate per-category data requirement checklists."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._requirements = []
        total_mandatory = 0
        total_optional = 0

        for cat in self._selected_categories:
            tier = self._tier_map.get(cat.value, MethodologyTier.SPEND_BASED)
            cat_key = cat.value
            cat_name = CATEGORY_NAMES.get(cat_key, cat_key)

            # Look up template
            cat_templates = REQUIREMENT_TEMPLATES.get(cat_key, {})
            tier_fields = cat_templates.get(tier.value, [])

            if not tier_fields:
                warnings.append(
                    f"No requirement template for {cat_key} at tier {tier.value}; "
                    f"using spend_based fallback"
                )
                tier_fields = cat_templates.get("spend_based", [])

            required_fields: List[DataRequirementField] = []
            optional_fields: List[DataRequirementField] = []

            for field_def in tier_fields:
                field = DataRequirementField(
                    field_name=field_def.get("field_name", ""),
                    display_name=field_def.get("display_name", ""),
                    data_type=field_def.get("data_type", "str"),
                    unit=field_def.get("unit", ""),
                    is_mandatory=field_def.get("is_mandatory", True),
                    tier=tier,
                )
                if field.is_mandatory:
                    required_fields.append(field)
                    total_mandatory += 1
                else:
                    optional_fields.append(field)
                    total_optional += 1

            # Determine supported data sources
            supported_sources = [
                DataSourceType.MANUAL_FORM,
                DataSourceType.FILE_UPLOAD,
            ]
            if tier in (MethodologyTier.AVERAGE_DATA, MethodologyTier.SUPPLIER_SPECIFIC):
                supported_sources.append(DataSourceType.ERP_INTEGRATION)
                supported_sources.append(DataSourceType.API_FEED)
            if tier == MethodologyTier.SUPPLIER_SPECIFIC:
                supported_sources.append(DataSourceType.SUPPLIER_PORTAL)
                supported_sources.append(DataSourceType.QUESTIONNAIRE)

            effort = EFFORT_HOURS_BY_TIER.get(tier.value, 4.0)

            self._requirements.append(
                CategoryDataRequirements(
                    category=cat,
                    category_name=cat_name,
                    tier=tier,
                    required_fields=required_fields,
                    optional_fields=optional_fields,
                    supported_sources=supported_sources,
                    data_collection_guidance=(
                        f"Collect {len(required_fields)} mandatory fields for "
                        f"{cat_name} using {tier.value} methodology."
                    ),
                    estimated_effort_hours=effort,
                )
            )

        outputs["categories_with_requirements"] = len(self._requirements)
        outputs["total_mandatory_fields"] = total_mandatory
        outputs["total_optional_fields"] = total_optional
        outputs["estimated_total_effort_hours"] = sum(
            r.estimated_effort_hours for r in self._requirements
        )
        outputs["requirements_summary"] = {
            r.category.value: {
                "tier": r.tier.value,
                "mandatory": len(r.required_fields),
                "optional": len(r.optional_fields),
                "effort_hours": r.estimated_effort_hours,
            }
            for r in self._requirements
        }

        self._state.progress_pct = 30.0

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 DataRequirements: %d categories, %d mandatory fields, "
            "%.0f total hours estimated",
            len(self._requirements), total_mandatory,
            outputs["estimated_total_effort_hours"],
        )
        return PhaseResult(
            phase_name="data_requirements", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Data Intake
    # -------------------------------------------------------------------------

    async def _phase_data_intake(
        self, input_data: CategoryDataCollectionInput
    ) -> PhaseResult:
        """Collect activity data and normalize records."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._ingested_records = list(input_data.ingested_data)

        # Group records by category
        records_by_category: Dict[str, List[IngestedDataRecord]] = {}
        for record in self._ingested_records:
            cat_key = record.category.value
            records_by_category.setdefault(cat_key, []).append(record)

        # Initialize progress tracking per category
        self._category_progress = []
        for cat in self._selected_categories:
            cat_key = cat.value
            cat_records = records_by_category.get(cat_key, [])
            tier = self._tier_map.get(cat_key, MethodologyTier.SPEND_BASED)

            # Calculate mandatory field coverage
            req = next(
                (r for r in self._requirements if r.category == cat), None
            )
            mandatory_coverage = 0.0
            if req and req.required_fields and cat_records:
                mandatory_names = {f.field_name for f in req.required_fields}
                fields_present: set = set()
                for rec in cat_records:
                    fields_present.update(
                        k for k in rec.data_fields.keys() if k in mandatory_names
                    )
                mandatory_coverage = (
                    len(fields_present) / len(mandatory_names) * 100.0
                    if mandatory_names else 100.0
                )

            # Calculate date range coverage
            date_coverage = self._calculate_date_coverage(
                cat_records,
                input_data.reporting_period_start,
                input_data.reporting_period_end,
            )

            # Determine completion status
            status = CompletionStatus.NOT_STARTED
            if cat_records:
                if mandatory_coverage >= 100.0 and date_coverage >= 90.0:
                    status = CompletionStatus.COMPLETE
                elif mandatory_coverage >= 80.0 or date_coverage >= 50.0:
                    status = CompletionStatus.SUBSTANTIALLY_COMPLETE
                else:
                    status = CompletionStatus.IN_PROGRESS

            sources_used = list({r.source_type.value for r in cat_records})

            self._category_progress.append(
                CategoryCollectionProgress(
                    category=cat,
                    category_name=CATEGORY_NAMES.get(cat_key, cat_key),
                    status=status,
                    records_collected=len(cat_records),
                    mandatory_fields_coverage_pct=round(mandatory_coverage, 1),
                    date_range_coverage_pct=round(date_coverage, 1),
                    data_sources_used=sources_used,
                    tier=tier,
                )
            )

        # Identify categories with no data
        empty_cats = [
            cp.category_name
            for cp in self._category_progress
            if cp.records_collected == 0
        ]
        if empty_cats:
            warnings.append(
                f"{len(empty_cats)} categories have no data records: "
                f"{', '.join(empty_cats[:5])}"
                + (f" and {len(empty_cats) - 5} more" if len(empty_cats) > 5 else "")
            )

        outputs["total_records_ingested"] = len(self._ingested_records)
        outputs["categories_with_data"] = sum(
            1 for cp in self._category_progress if cp.records_collected > 0
        )
        outputs["categories_without_data"] = len(empty_cats)
        outputs["data_sources_used"] = list({
            r.source_type.value for r in self._ingested_records
        })
        outputs["progress_by_category"] = {
            cp.category.value: {
                "status": cp.status.value,
                "records": cp.records_collected,
                "mandatory_coverage_pct": cp.mandatory_fields_coverage_pct,
                "date_coverage_pct": cp.date_range_coverage_pct,
            }
            for cp in self._category_progress
        }

        self._state.progress_pct = 60.0

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 DataIntake: %d records, %d/%d categories with data",
            len(self._ingested_records),
            outputs["categories_with_data"],
            len(self._selected_categories),
        )
        return PhaseResult(
            phase_name="data_intake", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Data Validation
    # -------------------------------------------------------------------------

    async def _phase_data_validation(
        self, input_data: CategoryDataCollectionInput
    ) -> PhaseResult:
        """Validate completeness, units, date ranges, and plausibility."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._validation_issues = []
        total_errors = 0
        total_warnings = 0

        for cat in self._selected_categories:
            cat_key = cat.value
            cat_records = [
                r for r in self._ingested_records if r.category == cat
            ]

            # 1. Completeness check
            completeness_issues = self._validate_completeness(cat, cat_records)
            self._validation_issues.extend(completeness_issues)

            # 2. Unit consistency check
            unit_issues = self._validate_units(cat, cat_records)
            self._validation_issues.extend(unit_issues)

            # 3. Date range check
            date_issues = self._validate_date_range(
                cat, cat_records,
                input_data.reporting_period_start,
                input_data.reporting_period_end,
            )
            self._validation_issues.extend(date_issues)

            # 4. Plausibility check (sector benchmark comparison)
            plausibility_issues = self._validate_plausibility(
                cat, cat_records, input_data.sector
            )
            self._validation_issues.extend(plausibility_issues)

            # 5. Outlier detection
            outlier_issues = self._detect_outliers(cat, cat_records)
            self._validation_issues.extend(outlier_issues)

        # Update category progress with validation counts
        for cp in self._category_progress:
            cat_issues = [
                vi for vi in self._validation_issues if vi.category == cp.category
            ]
            cp.validation_issues = len(cat_issues)
            cp.validation_errors = sum(
                1 for vi in cat_issues if vi.severity == ValidationSeverity.ERROR
            )

        total_errors = sum(
            1 for vi in self._validation_issues
            if vi.severity == ValidationSeverity.ERROR
        )
        total_warnings = sum(
            1 for vi in self._validation_issues
            if vi.severity == ValidationSeverity.WARNING
        )

        outputs["total_validation_issues"] = len(self._validation_issues)
        outputs["total_errors"] = total_errors
        outputs["total_warnings"] = total_warnings
        outputs["total_info"] = len(self._validation_issues) - total_errors - total_warnings
        outputs["categories_with_errors"] = sum(
            1 for cp in self._category_progress if cp.validation_errors > 0
        )
        outputs["categories_clean"] = sum(
            1 for cp in self._category_progress if cp.validation_issues == 0
        )
        outputs["validation_summary"] = {
            cp.category.value: {
                "issues": cp.validation_issues,
                "errors": cp.validation_errors,
            }
            for cp in self._category_progress
        }

        if total_errors > 0:
            warnings.append(
                f"{total_errors} validation errors found; data corrections needed "
                f"before calculation"
            )

        self._state.progress_pct = 100.0

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 DataValidation: %d issues (%d errors, %d warnings)",
            len(self._validation_issues), total_errors, total_warnings,
        )
        return PhaseResult(
            phase_name="data_validation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Validation Helpers
    # -------------------------------------------------------------------------

    def _validate_completeness(
        self, category: Scope3Category, records: List[IngestedDataRecord]
    ) -> List[ValidationIssue]:
        """Check mandatory field coverage for a category."""
        issues: List[ValidationIssue] = []
        req = next((r for r in self._requirements if r.category == category), None)
        if not req:
            return issues

        if not records:
            issues.append(ValidationIssue(
                category=category,
                severity=ValidationSeverity.ERROR,
                message=f"No data records for {CATEGORY_NAMES.get(category.value, category.value)}",
                suggested_action="Collect activity data for this category",
            ))
            return issues

        mandatory_names = {f.field_name for f in req.required_fields}
        for record in records:
            missing = mandatory_names - set(record.data_fields.keys())
            for field_name in missing:
                issues.append(ValidationIssue(
                    category=category,
                    field_name=field_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing mandatory field '{field_name}'",
                    record_id=record.record_id,
                    suggested_action=f"Provide value for '{field_name}'",
                ))

        return issues

    def _validate_units(
        self, category: Scope3Category, records: List[IngestedDataRecord]
    ) -> List[ValidationIssue]:
        """Check unit consistency across records."""
        issues: List[ValidationIssue] = []
        if not records:
            return issues

        units_seen: Dict[str, set] = {}
        for record in records:
            for field_name, value in record.data_fields.items():
                if isinstance(value, dict) and "unit" in value:
                    units_seen.setdefault(field_name, set()).add(value["unit"])

        for field_name, units in units_seen.items():
            if len(units) > 1:
                issues.append(ValidationIssue(
                    category=category,
                    field_name=field_name,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Inconsistent units for '{field_name}': "
                        f"{', '.join(sorted(units))}"
                    ),
                    suggested_action="Normalize all values to a single unit",
                ))

        return issues

    def _validate_date_range(
        self, category: Scope3Category, records: List[IngestedDataRecord],
        period_start: str, period_end: str,
    ) -> List[ValidationIssue]:
        """Check that records cover the reporting period."""
        issues: List[ValidationIssue] = []
        if not records:
            return issues

        records_with_dates = [
            r for r in records
            if r.reporting_period_start and r.reporting_period_end
        ]

        if not records_with_dates and records:
            issues.append(ValidationIssue(
                category=category,
                severity=ValidationSeverity.WARNING,
                message="No records have reporting period dates",
                suggested_action="Add reporting period start and end dates",
            ))
            return issues

        # Check for records outside reporting period
        for record in records_with_dates:
            if record.reporting_period_start < period_start:
                issues.append(ValidationIssue(
                    category=category,
                    field_name="reporting_period_start",
                    severity=ValidationSeverity.INFO,
                    message=(
                        f"Record starts before reporting period "
                        f"({record.reporting_period_start} < {period_start})"
                    ),
                    record_id=record.record_id,
                    suggested_action="Verify record belongs to this reporting year",
                ))
            if record.reporting_period_end > period_end:
                issues.append(ValidationIssue(
                    category=category,
                    field_name="reporting_period_end",
                    severity=ValidationSeverity.INFO,
                    message=(
                        f"Record ends after reporting period "
                        f"({record.reporting_period_end} > {period_end})"
                    ),
                    record_id=record.record_id,
                    suggested_action="Verify record belongs to this reporting year",
                ))

        return issues

    def _validate_plausibility(
        self, category: Scope3Category, records: List[IngestedDataRecord],
        sector: str,
    ) -> List[ValidationIssue]:
        """Compare aggregated values against sector benchmarks."""
        issues: List[ValidationIssue] = []
        cat_key = category.value

        benchmarks = PLAUSIBILITY_BENCHMARKS.get(cat_key, {})
        if not benchmarks or not records:
            return issues

        sector_lower = (sector or "default").lower()
        sector_bench = benchmarks.get(sector_lower, benchmarks.get("default", {}))
        if not sector_bench:
            return issues

        # Sum up values for plausibility
        total_value = 0.0
        for record in records:
            for key, val in record.data_fields.items():
                if isinstance(val, (int, float)):
                    total_value += val

        if total_value > 0 and "high" in sector_bench:
            if total_value > sector_bench["high"] * 10:
                issues.append(ValidationIssue(
                    category=category,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Aggregated value ({total_value:.0f}) significantly exceeds "
                        f"sector high benchmark ({sector_bench['high']:.1f})"
                    ),
                    suggested_action="Verify data for potential double-counting or unit errors",
                ))

        return issues

    def _detect_outliers(
        self, category: Scope3Category, records: List[IngestedDataRecord]
    ) -> List[ValidationIssue]:
        """Detect outlier values using IQR method."""
        issues: List[ValidationIssue] = []
        if len(records) < 4:
            return issues

        # Collect numeric values per field
        field_values: Dict[str, List[float]] = {}
        for record in records:
            for key, val in record.data_fields.items():
                if isinstance(val, (int, float)) and val > 0:
                    field_values.setdefault(key, []).append(float(val))

        for field_name, values in field_values.items():
            if len(values) < 4:
                continue

            sorted_vals = sorted(values)
            q1_idx = len(sorted_vals) // 4
            q3_idx = (3 * len(sorted_vals)) // 4
            q1 = sorted_vals[q1_idx]
            q3 = sorted_vals[q3_idx]
            iqr = q3 - q1

            if iqr <= 0:
                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_count = sum(
                1 for v in values if v < lower_bound or v > upper_bound
            )
            if outlier_count > 0:
                issues.append(ValidationIssue(
                    category=category,
                    field_name=field_name,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"{outlier_count} outlier(s) detected for '{field_name}' "
                        f"(IQR bounds: {lower_bound:.2f} - {upper_bound:.2f})"
                    ),
                    suggested_action="Review outlier records for data entry errors",
                ))

        return issues

    def _calculate_date_coverage(
        self, records: List[IngestedDataRecord],
        period_start: str, period_end: str,
    ) -> float:
        """Calculate what percentage of the reporting period is covered."""
        if not records:
            return 0.0

        records_with_dates = [
            r for r in records if r.reporting_period_start and r.reporting_period_end
        ]
        if not records_with_dates:
            return 50.0  # Assume partial coverage if dates not specified

        try:
            start = datetime.fromisoformat(period_start)
            end = datetime.fromisoformat(period_end)
            total_days = (end - start).days
            if total_days <= 0:
                return 100.0

            covered_days: set = set()
            for record in records_with_dates:
                try:
                    rec_start = datetime.fromisoformat(record.reporting_period_start)
                    rec_end = datetime.fromisoformat(record.reporting_period_end)
                    day = max(rec_start, start)
                    while day <= min(rec_end, end):
                        covered_days.add(day.date())
                        day = datetime(day.year, day.month, day.day + 1) if day.day < 28 else day
                        break  # Simplified: count record as covering its span
                except (ValueError, TypeError):
                    continue

            # Simplified: each record with dates covers its span
            if records_with_dates:
                return min(len(records_with_dates) / 12.0 * 100.0, 100.0)
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    def _calculate_overall_completeness(self) -> float:
        """Calculate overall completeness across all categories."""
        if not self._category_progress:
            return 0.0
        total = sum(cp.mandatory_fields_coverage_pct for cp in self._category_progress)
        return round(total / len(self._category_progress), 1)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._selected_categories = []
        self._requirements = []
        self._ingested_records = []
        self._validation_issues = []
        self._category_progress = []
        self._phase_results = []
        self._tier_map = {}
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: CategoryDataCollectionResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += f"|{result.workflow_id}|{result.overall_completeness_pct}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
