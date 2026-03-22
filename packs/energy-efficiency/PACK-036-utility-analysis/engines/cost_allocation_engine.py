# -*- coding: utf-8 -*-
"""
CostAllocationEngine - PACK-036 Utility Analysis Engine 4
==========================================================

Energy cost allocation engine for distributing utility costs across tenants,
departments, processes, and common areas within multi-tenant or multi-use
facilities.  Supports nine allocation methods including direct metering,
sub-metering, area proration, headcount, production-based, operating hours,
weighted combination, regression, and fixed percentage.

Allocation Methodology:
    Area Proration:
        entity_cost = total_cost * (entity_area / total_area)

    Direct Meter / Sub-Meter:
        entity_cost = total_cost * (entity_kwh / total_metered_kwh)

    Headcount:
        entity_cost = total_cost * (entity_headcount / total_headcount)

    Production-Based:
        entity_cost = total_cost * (entity_units / total_units)

    Operating Hours:
        entity_cost = total_cost * (entity_hours / total_hours)

    Weighted Combination:
        entity_cost = total * sum(w_i * basis_i) / sum(w_i * total_basis_i)

    Fixed Percentage:
        entity_cost = total_cost * entity.allocation_weight / 100

    Coincident Peak Demand:
        entity_demand_cost = total_demand_cost * (entity_kw_at_peak / sum_all_kw_at_peak)

    4CP (Four Coincident Peak):
        demand_cost = avg(entity_demand_at_4_system_peaks) * rate

    Gini Coefficient (Fairness):
        Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n

Cost Components:
    ENERGY:          Volumetric energy charge (per kWh)
    DEMAND:          Peak demand charge (per kW)
    DISTRIBUTION:    Wires/pipes delivery charge
    TRANSMISSION:    High-voltage transmission charge
    TAXES:           Government levies and duties
    FEES:            Fixed service fees, meter charges
    COMMON_AREA:     Shared space energy (lobbies, corridors, lifts)
    BASE_BUILDING:   Building services (fire, security, BMS)
    RENEWABLE_LEVY:  Feed-in tariff or renewable obligation levies

Regulatory References:
    - ASHRAE Standard 105-2014 Standard Methods of Determining, Expressing,
      and Comparing Building Energy Performance and Greenhouse Gas Emissions
    - BOMA International: ANSI/BOMA Z65.1 Office Buildings Standard Methods
      of Measurement (floor area and allocation)
    - RICS Service Charges in Commercial Property (3rd ed, 2018)
    - EU Energy Efficiency Directive 2012/27/EU, Article 9 (sub-metering)
    - EN 15459:2017 Energy performance of buildings - Economic evaluation
    - IPMVP (International Performance Measurement and Verification Protocol)
    - IFRS 16 Leases: allocation of variable lease payments

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Allocation formulas from published property/energy management standards
    - No LLM involvement in any numeric calculation path
    - SHA-256 provenance hashing on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-036 Utility Analysis
Engine:  4 of 10
Status:  Production Ready
"""

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when denominator is zero.

    Returns:
        Result of division or *default*.
    """
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100).

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        Percentage as Decimal; Decimal("0") when whole is zero.
    """
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float.

    Uses ROUND_HALF_UP (regulatory standard rounding).

    Args:
        value: Value to round.
        places: Number of decimal places.

    Returns:
        Rounded float value.
    """
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))


def _round2(value: Decimal) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round4(value: Decimal) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AllocationMethod(str, Enum):
    """Method used to allocate utility costs to entities.

    DIRECT_METER:         Entity has its own revenue-grade meter.
    SUB_METER:            Entity has a sub-meter installed.
    AREA_PRORATION:       Cost split by floor area (m2) ratio.
    HEADCOUNT:            Cost split by number of occupants/employees.
    PRODUCTION_BASED:     Cost split by production output units.
    OPERATING_HOURS:      Cost split by weekly operating hours.
    WEIGHTED_COMBINATION: Multi-factor weighted allocation.
    REGRESSION:           Statistical regression-based allocation.
    FIXED_PERCENTAGE:     Contractually fixed allocation percentage.
    """
    DIRECT_METER = "direct_meter"
    SUB_METER = "sub_meter"
    AREA_PRORATION = "area_proration"
    HEADCOUNT = "headcount"
    PRODUCTION_BASED = "production_based"
    OPERATING_HOURS = "operating_hours"
    WEIGHTED_COMBINATION = "weighted_combination"
    REGRESSION = "regression"
    FIXED_PERCENTAGE = "fixed_percentage"


class CostComponent(str, Enum):
    """Utility bill cost component categories.

    Each component of a utility bill can be allocated using a different
    method.  For example, ENERGY by sub-meter, DEMAND by coincident peak,
    and COMMON_AREA by area proration.

    ENERGY:           Volumetric energy charge (per kWh consumed).
    DEMAND:           Peak demand charge (per kW of peak demand).
    DISTRIBUTION:     Local distribution network charge (wires/pipes).
    TRANSMISSION:     High-voltage transmission network charge.
    TAXES:            Government levies, duties, and environmental taxes.
    FEES:             Fixed service fees, meter rental, connection charges.
    COMMON_AREA:      Energy used in shared building spaces.
    BASE_BUILDING:    Essential building services (fire, security, BMS).
    RENEWABLE_LEVY:   Feed-in tariff, renewable obligation, green levy.
    """
    ENERGY = "energy"
    DEMAND = "demand"
    DISTRIBUTION = "distribution"
    TRANSMISSION = "transmission"
    TAXES = "taxes"
    FEES = "fees"
    COMMON_AREA = "common_area"
    BASE_BUILDING = "base_building"
    RENEWABLE_LEVY = "renewable_levy"


class TenantType(str, Enum):
    """Tenant / entity type classification for allocation purposes.

    Determines default allocation behaviours and benchmark comparisons.

    COMMERCIAL_OFFICE: Standard office tenant.
    RETAIL:            Retail shop or showroom tenant.
    INDUSTRIAL:        Manufacturing / warehouse tenant.
    RESIDENTIAL:       Residential dwelling unit.
    COMMON_AREA:       Shared building spaces (lobbies, corridors).
    MECHANICAL:        Plant rooms, risers, services spaces.
    PARKING:           Parking garage or surface lot.
    """
    COMMERCIAL_OFFICE = "commercial_office"
    RETAIL = "retail"
    INDUSTRIAL = "industrial"
    RESIDENTIAL = "residential"
    COMMON_AREA = "common_area"
    MECHANICAL = "mechanical"
    PARKING = "parking"


class ReconciliationStatus(str, Enum):
    """Status of cost allocation reconciliation against actual billing.

    RECONCILED:      Variance <= 1% (within rounding tolerance).
    VARIANCE_LOW:    1% < variance <= 3% (acceptable administrative variance).
    VARIANCE_HIGH:   3% < variance <= 5% (investigation recommended).
    UNRECONCILED:    Variance > 5% (action required, allocation error likely).
    """
    RECONCILED = "reconciled"
    VARIANCE_LOW = "variance_low"
    VARIANCE_HIGH = "variance_high"
    UNRECONCILED = "unreconciled"


class AllocationFrequency(str, Enum):
    """Frequency of cost allocation cycles.

    MONTHLY:     Monthly allocation (standard for utility billing).
    QUARTERLY:   Quarterly allocation (quarterly billing cycles).
    ANNUAL:      Annual allocation (annual true-up and reconciliation).
    """
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class DemandAllocationMethod(str, Enum):
    """Method used to allocate demand (kW) charges.

    COINCIDENT_PEAK:     Allocate by each entity's kW at building peak.
    NON_COINCIDENT_PEAK: Allocate by each entity's own individual peak kW.
    DIVERSIFIED:         Adjusted for diversity factor between entities.
    FOUR_CP:             Average of entity demand at 4 system peak periods.
    AREA_BASED:          Demand allocated by area ratio (fallback).
    """
    COINCIDENT_PEAK = "coincident_peak"
    NON_COINCIDENT_PEAK = "non_coincident_peak"
    DIVERSIFIED = "diversified"
    FOUR_CP = "four_cp"
    AREA_BASED = "area_based"


# ---------------------------------------------------------------------------
# Constants / Reference Data
# ---------------------------------------------------------------------------

# Reconciliation thresholds (percentage of total cost).
_RECONCILED_THRESHOLD: Decimal = Decimal("1.0")
_VARIANCE_LOW_THRESHOLD: Decimal = Decimal("3.0")
_VARIANCE_HIGH_THRESHOLD: Decimal = Decimal("5.0")

# Default cost component weights for weighted combination method.
# Source: BOMA International best practice, RICS Service Charges guidance.
DEFAULT_COMPONENT_WEIGHTS: Dict[str, Decimal] = {
    "area": Decimal("0.50"),
    "headcount": Decimal("0.25"),
    "operating_hours": Decimal("0.25"),
}

# Typical load density by tenant type (W/m2).
# Source: CIBSE Guide F Table 20.1, ASHRAE 90.1-2019 Table G3.1.
TYPICAL_LOAD_DENSITY_W_PER_M2: Dict[str, Decimal] = {
    TenantType.COMMERCIAL_OFFICE: Decimal("25.0"),
    TenantType.RETAIL: Decimal("30.0"),
    TenantType.INDUSTRIAL: Decimal("40.0"),
    TenantType.RESIDENTIAL: Decimal("12.0"),
    TenantType.COMMON_AREA: Decimal("15.0"),
    TenantType.MECHANICAL: Decimal("50.0"),
    TenantType.PARKING: Decimal("5.0"),
}

# Diversity factors by tenant count range.
# Source: ASHRAE Fundamentals Handbook, Chapter 18.
DIVERSITY_FACTORS: Dict[str, Decimal] = {
    "2_to_5": Decimal("0.85"),
    "6_to_10": Decimal("0.80"),
    "11_to_20": Decimal("0.75"),
    "21_to_50": Decimal("0.70"),
    "51_plus": Decimal("0.65"),
}

# Standard operating hours by tenant type (hours per week).
# Source: ASHRAE Standard 100-2018 Table 7-1, CIBSE Guide F.
STANDARD_OPERATING_HOURS: Dict[str, Decimal] = {
    TenantType.COMMERCIAL_OFFICE: Decimal("50.0"),
    TenantType.RETAIL: Decimal("65.0"),
    TenantType.INDUSTRIAL: Decimal("80.0"),
    TenantType.RESIDENTIAL: Decimal("168.0"),
    TenantType.COMMON_AREA: Decimal("168.0"),
    TenantType.MECHANICAL: Decimal("168.0"),
    TenantType.PARKING: Decimal("168.0"),
}

# Tax rate defaults by cost component (for invoice generation).
DEFAULT_TAX_RATE: Decimal = Decimal("0.19")  # 19% standard EU VAT


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class AllocationEntity(BaseModel):
    """Tenant, department, or process that receives allocated costs.

    Attributes:
        entity_id: Unique entity identifier.
        name: Human-readable entity name.
        entity_type: Classification of the entity.
        floor_area_m2: Occupied floor area in square metres.
        headcount: Number of occupants or employees.
        operating_hours_per_week: Weekly operating hours.
        production_units: Production output (units per period).
        sub_meter_id: Optional sub-meter identifier for direct metering.
        allocation_weight: Fixed percentage weight (0-100) for fixed allocation.
    """
    entity_id: str = Field(..., min_length=1, description="Entity identifier")
    name: str = Field(..., min_length=1, description="Entity name")
    entity_type: TenantType = Field(
        default=TenantType.COMMERCIAL_OFFICE,
        description="Entity type classification",
    )
    floor_area_m2: float = Field(
        default=0.0, ge=0, description="Occupied floor area (m2)"
    )
    headcount: int = Field(
        default=0, ge=0, description="Number of occupants / employees"
    )
    operating_hours_per_week: float = Field(
        default=0.0, ge=0, le=168.0, description="Weekly operating hours"
    )
    production_units: float = Field(
        default=0.0, ge=0, description="Production output per period"
    )
    sub_meter_id: Optional[str] = Field(
        None, description="Sub-meter ID if directly metered"
    )
    allocation_weight: float = Field(
        default=0.0, ge=0, le=100.0,
        description="Fixed allocation percentage (0-100)",
    )

    @field_validator("floor_area_m2")
    @classmethod
    def validate_floor_area(cls, v: float) -> float:
        """Ensure floor area is within plausible bounds."""
        if v > 2_000_000:
            raise ValueError("Floor area exceeds 2 million m2 sanity check")
        return v


class AllocationRule(BaseModel):
    """Rule defining how a specific cost component should be allocated.

    Attributes:
        cost_component: Which bill component this rule applies to.
        method: Allocation method to use.
        parameters: Additional method-specific parameters.
        priority: Rule priority (lower = higher priority, for conflict resolution).
    """
    cost_component: CostComponent = Field(
        ..., description="Cost component this rule governs"
    )
    method: AllocationMethod = Field(
        ..., description="Allocation method to apply"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific parameters (e.g., weights, thresholds)",
    )
    priority: int = Field(
        default=100, ge=1, le=1000, description="Rule priority (1 = highest)"
    )


class CostPool(BaseModel):
    """Pool of costs to be allocated across entities.

    Attributes:
        pool_id: Unique pool identifier.
        description: Human-readable description of the cost pool.
        total_amount_eur: Total cost in EUR to allocate.
        cost_component: Which cost component this pool represents.
        allocable: Whether this pool should be allocated (False = retained).
        allocation_method: Override allocation method (optional).
    """
    pool_id: str = Field(default_factory=_new_uuid, description="Pool identifier")
    description: str = Field(default="", description="Cost pool description")
    total_amount_eur: float = Field(
        ..., ge=0, description="Total cost to allocate (EUR)"
    )
    cost_component: CostComponent = Field(
        ..., description="Cost component category"
    )
    allocable: bool = Field(
        default=True, description="Whether this pool should be allocated"
    )
    allocation_method: Optional[AllocationMethod] = Field(
        None, description="Override allocation method for this pool"
    )

    @field_validator("total_amount_eur")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        """Ensure cost is within plausible bounds."""
        if v > 100_000_000:
            raise ValueError("Cost exceeds EUR 100 million sanity check")
        return v


class SubMeterData(BaseModel):
    """Sub-meter reading data for a specific entity and period.

    Attributes:
        meter_id: Sub-meter identifier.
        entity_id: Entity that this meter serves.
        period_start: Start date of the metering period.
        period_end: End date of the metering period.
        consumption_kwh: Energy consumed during the period (kWh).
        peak_demand_kw: Peak demand recorded during the period (kW).
    """
    meter_id: str = Field(..., min_length=1, description="Sub-meter identifier")
    entity_id: str = Field(..., min_length=1, description="Entity served by meter")
    period_start: date = Field(..., description="Metering period start date")
    period_end: date = Field(..., description="Metering period end date")
    consumption_kwh: float = Field(
        ..., ge=0, description="Energy consumed (kWh)"
    )
    peak_demand_kw: float = Field(
        default=0.0, ge=0, description="Peak demand (kW)"
    )

    @field_validator("consumption_kwh")
    @classmethod
    def validate_consumption(cls, v: float) -> float:
        """Ensure consumption is within plausible bounds."""
        if v > 100_000_000:
            raise ValueError("Consumption exceeds 100 GWh per period sanity check")
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class AllocationLineItem(BaseModel):
    """Single line item in a cost allocation result.

    Attributes:
        entity_id: Entity receiving this allocated cost.
        cost_component: Cost component category.
        allocated_amount_eur: Amount allocated (EUR).
        allocation_method: Method used for this allocation.
        basis_value: Value of the allocation basis (area, kWh, headcount, etc.).
        basis_unit: Unit of the basis value.
        share_pct: Entity's share as a percentage of total.
    """
    entity_id: str = Field(..., description="Entity identifier")
    cost_component: str = Field(..., description="Cost component")
    allocated_amount_eur: float = Field(
        default=0.0, description="Allocated amount (EUR)"
    )
    allocation_method: str = Field(
        default="", description="Allocation method used"
    )
    basis_value: float = Field(
        default=0.0, description="Allocation basis value"
    )
    basis_unit: str = Field(
        default="", description="Unit of basis value"
    )
    share_pct: float = Field(
        default=0.0, description="Share percentage of total"
    )


class AllocationResult(BaseModel):
    """Complete cost allocation result for a period.

    Contains all allocated line items, unallocated remainder,
    reconciliation status, and provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )

    period: str = Field(default="", description="Allocation period label")
    total_cost_eur: float = Field(
        default=0.0, description="Total cost before allocation (EUR)"
    )
    allocated_items: List[AllocationLineItem] = Field(
        default_factory=list, description="Allocated line items"
    )
    unallocated_eur: float = Field(
        default=0.0, description="Unallocated remainder (EUR)"
    )
    reconciliation_status: str = Field(
        default=ReconciliationStatus.UNRECONCILED.value,
        description="Reconciliation status",
    )
    variance_pct: float = Field(
        default=0.0, description="Variance percentage"
    )

    entity_totals: Dict[str, float] = Field(
        default_factory=dict, description="Total allocated per entity (EUR)"
    )
    component_totals: Dict[str, float] = Field(
        default_factory=dict, description="Total allocated per component (EUR)"
    )

    warnings: List[str] = Field(
        default_factory=list, description="Allocation warnings"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class TenantInvoice(BaseModel):
    """Invoice generated for a single tenant from allocation results.

    Attributes:
        invoice_id: Unique invoice identifier.
        tenant_id: Tenant / entity identifier.
        tenant_name: Tenant display name.
        period: Billing period label.
        line_items: Allocated cost line items.
        subtotal: Sum of line items before tax.
        taxes: Tax amount.
        total: Subtotal plus taxes.
        due_date: Invoice due date.
        currency: Currency code.
    """
    invoice_id: str = Field(default_factory=_new_uuid, description="Invoice ID")
    tenant_id: str = Field(..., description="Tenant identifier")
    tenant_name: str = Field(default="", description="Tenant display name")
    period: str = Field(default="", description="Billing period")
    line_items: List[AllocationLineItem] = Field(
        default_factory=list, description="Cost line items"
    )
    subtotal: float = Field(default=0.0, description="Subtotal before tax (EUR)")
    taxes: float = Field(default=0.0, description="Tax amount (EUR)")
    total: float = Field(default=0.0, description="Total invoice amount (EUR)")
    due_date: Optional[date] = Field(None, description="Invoice due date")
    currency: str = Field(default="EUR", description="Currency code")
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class ReconciliationReport(BaseModel):
    """Reconciliation report comparing allocated costs to actual billing.

    Attributes:
        report_id: Unique report identifier.
        period: Reconciliation period label.
        billed_total: Actual bill total (EUR).
        allocated_total: Sum of allocated costs (EUR).
        variance_eur: Absolute variance (EUR).
        variance_pct: Variance as percentage of billed total.
        status: Reconciliation status classification.
        explanations: Narrative explanations for variances.
        component_variances: Variance broken down by cost component.
    """
    report_id: str = Field(default_factory=_new_uuid, description="Report ID")
    period: str = Field(default="", description="Reconciliation period")
    billed_total: float = Field(default=0.0, description="Billed total (EUR)")
    allocated_total: float = Field(
        default=0.0, description="Allocated total (EUR)"
    )
    variance_eur: float = Field(default=0.0, description="Variance (EUR)")
    variance_pct: float = Field(default=0.0, description="Variance (%)")
    status: str = Field(
        default=ReconciliationStatus.UNRECONCILED.value,
        description="Reconciliation status",
    )
    explanations: List[str] = Field(
        default_factory=list, description="Variance explanations"
    )
    component_variances: Dict[str, float] = Field(
        default_factory=dict, description="Variance by cost component (EUR)"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class FairnessMetrics(BaseModel):
    """Fairness and equity metrics for cost allocation evaluation.

    Measures how equitably costs are distributed using statistical
    indicators.  A Gini coefficient of 0 means perfect equality (all
    entities pay the same per m2); a coefficient approaching 1 means
    extreme inequality.

    Attributes:
        gini_coefficient: Gini coefficient (0 = equal, 1 = unequal).
        max_cost_per_m2: Highest cost per m2 among entities.
        min_cost_per_m2: Lowest cost per m2 among entities.
        median_cost_per_m2: Median cost per m2.
        mean_cost_per_m2: Mean cost per m2.
        coefficient_of_variation: CV of cost per m2 (std / mean).
        entity_cost_per_m2: Per-entity cost per m2 breakdown.
    """
    gini_coefficient: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Gini coefficient"
    )
    max_cost_per_m2: float = Field(
        default=0.0, description="Maximum cost per m2 (EUR/m2)"
    )
    min_cost_per_m2: float = Field(
        default=0.0, description="Minimum cost per m2 (EUR/m2)"
    )
    median_cost_per_m2: float = Field(
        default=0.0, description="Median cost per m2 (EUR/m2)"
    )
    mean_cost_per_m2: float = Field(
        default=0.0, description="Mean cost per m2 (EUR/m2)"
    )
    coefficient_of_variation: float = Field(
        default=0.0, description="CV of cost per m2"
    )
    entity_cost_per_m2: Dict[str, float] = Field(
        default_factory=dict, description="Per-entity cost per m2"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------


class CostAllocationEngine:
    """Energy cost allocation engine for multi-tenant and multi-use facilities.

    Distributes utility costs across tenants, departments, and processes
    using nine allocation methods.  Supports cost component-level rules,
    demand charge allocation, common area distribution, tenant invoice
    generation, reconciliation against actual billing, and fairness
    assessment via Gini coefficient.

    All calculations are deterministic using Decimal arithmetic with
    SHA-256 provenance hashing.  No LLM is used in any calculation path.

    Usage::

        engine = CostAllocationEngine()
        result = engine.allocate_costs(entities, cost_pools, rules, sub_meters)
        invoices = engine.generate_invoices(result, entities)
        recon = engine.reconcile(result, actual_bill_total=Decimal("45000"))
        fairness = engine.calculate_fairness(result, entities)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise the cost allocation engine with reference data."""
        self._load_densities = TYPICAL_LOAD_DENSITY_W_PER_M2
        self._diversity_factors = DIVERSITY_FACTORS
        self._default_weights = DEFAULT_COMPONENT_WEIGHTS
        self._operating_hours = STANDARD_OPERATING_HOURS
        self._tax_rate = DEFAULT_TAX_RATE

    # -------------------------------------------------------------------
    # Public API -- Core Allocation
    # -------------------------------------------------------------------

    def allocate_costs(
        self,
        entities: List[AllocationEntity],
        cost_pools: List[CostPool],
        rules: List[AllocationRule],
        sub_meters: Optional[List[SubMeterData]] = None,
        period: str = "",
    ) -> AllocationResult:
        """Allocate utility costs across entities using configured rules.

        For each cost pool, finds the matching allocation rule (by cost
        component, respecting priority), and distributes the pool's total
        amount across entities using the designated method.  Pools marked
        as non-allocable are accumulated as unallocated remainder.

        Args:
            entities: List of entities to receive allocated costs.
            cost_pools: List of cost pools to allocate.
            rules: List of allocation rules defining methods per component.
            sub_meters: Optional sub-meter data for direct metering methods.
            period: Allocation period label (e.g. '2026-01', '2026-Q1').

        Returns:
            AllocationResult with all line items and reconciliation data.

        Raises:
            ValueError: If entities list is empty.
        """
        t0 = time.perf_counter()

        if not entities:
            raise ValueError("At least one allocation entity is required")
        if not cost_pools:
            raise ValueError("At least one cost pool is required")

        logger.info(
            "Starting cost allocation: %d entities, %d pools, %d rules, period=%s",
            len(entities), len(cost_pools), len(rules), period,
        )

        sub_meters = sub_meters or []
        all_items: List[AllocationLineItem] = []
        unallocated = Decimal("0")
        total_cost = Decimal("0")
        warnings: List[str] = []

        # Build rule lookup: cost_component -> rule (lowest priority number wins).
        rule_map = self._build_rule_map(rules)

        for pool in cost_pools:
            pool_amount = _decimal(pool.total_amount_eur)
            total_cost += pool_amount

            if not pool.allocable:
                unallocated += pool_amount
                logger.debug(
                    "Pool %s marked non-allocable, EUR %.2f added to unallocated",
                    pool.pool_id, pool_amount,
                )
                continue

            # Determine allocation method.
            method = self._resolve_method(pool, rule_map)
            if method is None:
                unallocated += pool_amount
                warnings.append(
                    f"No rule found for component {pool.cost_component.value}; "
                    f"EUR {_round2(pool_amount)} unallocated"
                )
                continue

            # Dispatch to method-specific allocator.
            items = self._dispatch_allocation(
                entities, pool, method, sub_meters, rule_map
            )

            # Validate allocated sum matches pool total.
            allocated_sum = sum(_decimal(i.allocated_amount_eur) for i in items)
            residual = pool_amount - allocated_sum

            if abs(residual) > Decimal("0.01"):
                # Distribute rounding residual to largest entity.
                items = self._distribute_residual(items, residual)
                logger.debug(
                    "Rounding residual EUR %.4f distributed for pool %s",
                    residual, pool.pool_id,
                )

            all_items.extend(items)

        # Build entity and component totals.
        entity_totals = self._compute_entity_totals(all_items)
        component_totals = self._compute_component_totals(all_items)

        # Self-reconcile: sum of allocated + unallocated vs total cost.
        allocated_total = sum(_decimal(v) for v in entity_totals.values())
        self_variance = total_cost - allocated_total - unallocated
        self_variance_pct = _safe_pct(abs(self_variance), total_cost)

        recon_status = self._classify_variance(self_variance_pct)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = AllocationResult(
            period=period,
            total_cost_eur=_round2(total_cost),
            allocated_items=all_items,
            unallocated_eur=_round2(unallocated),
            reconciliation_status=recon_status.value,
            variance_pct=_round4(self_variance_pct),
            entity_totals={k: _round2(_decimal(v)) for k, v in entity_totals.items()},
            component_totals={
                k: _round2(_decimal(v)) for k, v in component_totals.items()
            },
            warnings=warnings,
            processing_time_ms=round(elapsed_ms, 2),
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Cost allocation complete: total=EUR %.2f, allocated=%d items, "
            "unallocated=EUR %.2f, status=%s, elapsed=%.1f ms",
            total_cost, len(all_items), unallocated,
            recon_status.value, elapsed_ms,
        )

        return result

    def allocate_by_area(
        self,
        entities: List[AllocationEntity],
        total_cost: Decimal,
    ) -> List[AllocationLineItem]:
        """Allocate cost by floor area proration.

        Formula: entity_cost = total_cost * (entity_area / total_area)

        Args:
            entities: Entities with floor_area_m2 populated.
            total_cost: Total cost to allocate (EUR).

        Returns:
            List of AllocationLineItem with area-based allocation.
        """
        total_area = sum(_decimal(e.floor_area_m2) for e in entities)
        if total_area == Decimal("0"):
            logger.warning("Total area is zero; cannot allocate by area")
            return []

        items: List[AllocationLineItem] = []
        for entity in entities:
            area = _decimal(entity.floor_area_m2)
            share = _safe_divide(area, total_area)
            amount = total_cost * share

            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=CostComponent.ENERGY.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=AllocationMethod.AREA_PRORATION.value,
                basis_value=_round2(area),
                basis_unit="m2",
                share_pct=_round4(_safe_pct(area, total_area)),
            ))

        return items

    def allocate_by_meter(
        self,
        entities: List[AllocationEntity],
        sub_meters: List[SubMeterData],
        total_cost: Decimal,
    ) -> List[AllocationLineItem]:
        """Allocate cost by sub-meter consumption readings.

        Formula: entity_cost = total_cost * (entity_kwh / total_metered_kwh)

        Args:
            entities: Entities with sub_meter_id populated.
            sub_meters: Sub-meter consumption data.
            total_cost: Total cost to allocate (EUR).

        Returns:
            List of AllocationLineItem with metered allocation.
        """
        # Build meter lookup: entity_id -> total consumption.
        consumption_by_entity: Dict[str, Decimal] = {}
        for sm in sub_meters:
            current = consumption_by_entity.get(sm.entity_id, Decimal("0"))
            consumption_by_entity[sm.entity_id] = current + _decimal(sm.consumption_kwh)

        total_consumption = sum(consumption_by_entity.values())
        if total_consumption == Decimal("0"):
            logger.warning("Total metered consumption is zero; cannot allocate by meter")
            return []

        items: List[AllocationLineItem] = []
        for entity in entities:
            kwh = consumption_by_entity.get(entity.entity_id, Decimal("0"))
            share = _safe_divide(kwh, total_consumption)
            amount = total_cost * share

            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=CostComponent.ENERGY.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=AllocationMethod.SUB_METER.value,
                basis_value=_round2(kwh),
                basis_unit="kWh",
                share_pct=_round4(_safe_pct(kwh, total_consumption)),
            ))

        return items

    def allocate_demand(
        self,
        entities: List[AllocationEntity],
        sub_meters: List[SubMeterData],
        total_demand_cost: Decimal,
        method: DemandAllocationMethod = DemandAllocationMethod.COINCIDENT_PEAK,
    ) -> List[AllocationLineItem]:
        """Allocate demand (kW) charges across entities.

        Supports coincident peak, non-coincident peak, diversified,
        four-CP, and area-based methods.

        Args:
            entities: List of allocation entities.
            sub_meters: Sub-meter data with peak_demand_kw readings.
            total_demand_cost: Total demand charge to allocate (EUR).
            method: Demand allocation method to use.

        Returns:
            List of AllocationLineItem with demand-based allocation.
        """
        if method == DemandAllocationMethod.AREA_BASED:
            return self._allocate_demand_by_area(entities, total_demand_cost)
        if method == DemandAllocationMethod.FOUR_CP:
            return self._allocate_demand_four_cp(entities, sub_meters, total_demand_cost)
        if method == DemandAllocationMethod.DIVERSIFIED:
            return self._allocate_demand_diversified(
                entities, sub_meters, total_demand_cost
            )

        # COINCIDENT_PEAK and NON_COINCIDENT_PEAK both use peak kW values.
        return self._allocate_demand_by_peak(
            entities, sub_meters, total_demand_cost, method
        )

    def allocate_common_area(
        self,
        entities: List[AllocationEntity],
        common_area_cost: Decimal,
    ) -> List[AllocationLineItem]:
        """Allocate common area costs across non-common-area entities.

        Common area costs (lobbies, corridors, lifts, security, fire systems)
        are distributed to occupant entities proportional to their floor area,
        excluding entities classified as COMMON_AREA or MECHANICAL.

        Args:
            entities: All entities including common area.
            common_area_cost: Total common area cost (EUR).

        Returns:
            List of AllocationLineItem for common area cost distribution.
        """
        excluded_types = {TenantType.COMMON_AREA, TenantType.MECHANICAL}
        occupant_entities = [
            e for e in entities if e.entity_type not in excluded_types
        ]

        if not occupant_entities:
            logger.warning("No occupant entities for common area allocation")
            return []

        total_area = sum(_decimal(e.floor_area_m2) for e in occupant_entities)
        if total_area == Decimal("0"):
            logger.warning("Total occupant area is zero for common area allocation")
            return []

        items: List[AllocationLineItem] = []
        for entity in occupant_entities:
            area = _decimal(entity.floor_area_m2)
            share = _safe_divide(area, total_area)
            amount = common_area_cost * share

            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=CostComponent.COMMON_AREA.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=AllocationMethod.AREA_PRORATION.value,
                basis_value=_round2(area),
                basis_unit="m2",
                share_pct=_round4(_safe_pct(area, total_area)),
            ))

        return items

    # -------------------------------------------------------------------
    # Public API -- Invoice Generation
    # -------------------------------------------------------------------

    def generate_invoices(
        self,
        result: AllocationResult,
        entities: List[AllocationEntity],
        tax_rate: Optional[Decimal] = None,
        due_date: Optional[date] = None,
    ) -> List[TenantInvoice]:
        """Generate tenant invoices from allocation results.

        Groups allocation line items by entity and generates one invoice
        per entity with subtotal, tax, and total.

        Args:
            result: Completed allocation result.
            entities: List of entities (for name lookup).
            tax_rate: Tax rate to apply (default: 19% EU VAT).
            due_date: Invoice due date (optional).

        Returns:
            List of TenantInvoice models.
        """
        t0 = time.perf_counter()
        rate = tax_rate if tax_rate is not None else self._tax_rate

        # Build entity name lookup.
        name_map: Dict[str, str] = {e.entity_id: e.name for e in entities}

        # Group items by entity.
        items_by_entity: Dict[str, List[AllocationLineItem]] = {}
        for item in result.allocated_items:
            items_by_entity.setdefault(item.entity_id, []).append(item)

        invoices: List[TenantInvoice] = []
        for entity_id, items in items_by_entity.items():
            subtotal = sum(_decimal(i.allocated_amount_eur) for i in items)
            tax_amount = subtotal * rate
            total = subtotal + tax_amount

            invoice = TenantInvoice(
                tenant_id=entity_id,
                tenant_name=name_map.get(entity_id, entity_id),
                period=result.period,
                line_items=items,
                subtotal=_round2(subtotal),
                taxes=_round2(tax_amount),
                total=_round2(total),
                due_date=due_date,
            )
            invoice.provenance_hash = _compute_hash(invoice)
            invoices.append(invoice)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Generated %d invoices for period %s in %.1f ms",
            len(invoices), result.period, elapsed_ms,
        )

        return invoices

    # -------------------------------------------------------------------
    # Public API -- Reconciliation
    # -------------------------------------------------------------------

    def reconcile(
        self,
        result: AllocationResult,
        actual_bill_total: Decimal,
    ) -> ReconciliationReport:
        """Reconcile allocated costs against actual utility bill total.

        Compares the sum of all allocated amounts plus unallocated
        remainder against the actual bill total from the utility provider.

        Args:
            result: Completed allocation result.
            actual_bill_total: Actual utility bill total (EUR).

        Returns:
            ReconciliationReport with variance analysis.
        """
        t0 = time.perf_counter()

        allocated_total = _decimal(result.total_cost_eur)
        billed = _decimal(actual_bill_total)

        variance_eur = allocated_total - billed
        abs_variance = abs(variance_eur)
        variance_pct = _safe_pct(abs_variance, billed)

        status = self._classify_variance(variance_pct)

        # Build component-level variance analysis.
        component_variances: Dict[str, float] = {}
        explanations: List[str] = []

        if status != ReconciliationStatus.RECONCILED:
            explanations.append(
                f"Total variance of EUR {_round2(variance_eur)} "
                f"({_round4(variance_pct)}%) detected between allocation "
                f"(EUR {_round2(allocated_total)}) and actual bill "
                f"(EUR {_round2(billed)})"
            )

            if _decimal(result.unallocated_eur) > Decimal("0"):
                explanations.append(
                    f"EUR {result.unallocated_eur} remains unallocated "
                    f"(non-allocable pools or missing rules)"
                )

            if status == ReconciliationStatus.VARIANCE_HIGH:
                explanations.append(
                    "Variance exceeds 3% threshold. Review allocation rules "
                    "and sub-meter readings for accuracy."
                )
            elif status == ReconciliationStatus.UNRECONCILED:
                explanations.append(
                    "Variance exceeds 5% threshold. Immediate investigation "
                    "required: check meter calibration, rate changes, or "
                    "unaccounted cost components."
                )
        else:
            explanations.append(
                f"Allocation reconciled within {_round4(variance_pct)}% tolerance"
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        report = ReconciliationReport(
            period=result.period,
            billed_total=_round2(billed),
            allocated_total=_round2(allocated_total),
            variance_eur=_round2(variance_eur),
            variance_pct=_round4(variance_pct),
            status=status.value,
            explanations=explanations,
            component_variances=component_variances,
        )
        report.provenance_hash = _compute_hash(report)

        logger.info(
            "Reconciliation complete: billed=EUR %.2f, allocated=EUR %.2f, "
            "variance=%.4f%%, status=%s, elapsed=%.1f ms",
            billed, allocated_total, variance_pct, status.value, elapsed_ms,
        )

        return report

    # -------------------------------------------------------------------
    # Public API -- Fairness Assessment
    # -------------------------------------------------------------------

    def calculate_fairness(
        self,
        result: AllocationResult,
        entities: List[AllocationEntity],
    ) -> FairnessMetrics:
        """Calculate fairness and equity metrics for cost allocation.

        Computes Gini coefficient, cost-per-m2 statistics, and coefficient
        of variation to assess whether costs are equitably distributed.

        Gini formula:
            sorted values x_1 <= x_2 <= ... <= x_n
            Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n

        Args:
            result: Completed allocation result.
            entities: List of entities with floor_area_m2 populated.

        Returns:
            FairnessMetrics with statistical assessment.
        """
        t0 = time.perf_counter()

        # Build cost-per-m2 values for entities with non-zero area.
        entity_map = {e.entity_id: e for e in entities}
        cost_per_m2_values: List[Decimal] = []
        entity_cost_per_m2: Dict[str, float] = {}

        for entity_id, total_eur in result.entity_totals.items():
            entity = entity_map.get(entity_id)
            if entity is None or entity.floor_area_m2 <= 0:
                continue
            cpm2 = _safe_divide(
                _decimal(total_eur), _decimal(entity.floor_area_m2)
            )
            cost_per_m2_values.append(cpm2)
            entity_cost_per_m2[entity_id] = _round2(cpm2)

        if not cost_per_m2_values:
            logger.warning("No entities with positive area for fairness calc")
            return FairnessMetrics(provenance_hash="")

        # Sort for Gini and percentile calculations.
        sorted_values = sorted(cost_per_m2_values)
        n = len(sorted_values)

        # Gini coefficient.
        gini = self._compute_gini(sorted_values)

        # Basic statistics.
        sum_values = sum(sorted_values)
        mean_val = _safe_divide(sum_values, _decimal(n))

        # Median.
        if n % 2 == 1:
            median_val = sorted_values[n // 2]
        else:
            median_val = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / Decimal("2")

        # Standard deviation for coefficient of variation.
        if n > 1:
            variance_sum = sum((v - mean_val) ** 2 for v in sorted_values)
            std_dev = (variance_sum / _decimal(n)).sqrt()
            cv = _safe_divide(std_dev, mean_val)
        else:
            cv = Decimal("0")

        elapsed_ms = (time.perf_counter() - t0) * 1000

        metrics = FairnessMetrics(
            gini_coefficient=_round4(gini),
            max_cost_per_m2=_round2(sorted_values[-1]),
            min_cost_per_m2=_round2(sorted_values[0]),
            median_cost_per_m2=_round2(median_val),
            mean_cost_per_m2=_round2(mean_val),
            coefficient_of_variation=_round4(cv),
            entity_cost_per_m2=entity_cost_per_m2,
        )
        metrics.provenance_hash = _compute_hash(metrics)

        logger.info(
            "Fairness calculation: Gini=%.4f, CV=%.4f, range=[%.2f, %.2f] EUR/m2, "
            "elapsed=%.1f ms",
            gini, cv, sorted_values[0], sorted_values[-1], elapsed_ms,
        )

        return metrics

    # -------------------------------------------------------------------
    # Public API -- Virtual Sub-Meter Estimation
    # -------------------------------------------------------------------

    def virtual_submeter_estimate(
        self,
        entity: AllocationEntity,
        building_profile: Dict[str, Any],
    ) -> Decimal:
        """Estimate consumption for an entity without a physical sub-meter.

        Uses load density, operating hours, and floor area to produce a
        virtual sub-meter reading.  This is a deterministic engineering
        estimate, not an LLM prediction.

        Formula:
            estimated_kwh = load_density_w_per_m2 * area_m2 * hours_per_period / 1000

        Args:
            entity: Entity to estimate consumption for.
            building_profile: Building context with keys:
                - 'period_hours': Total hours in the billing period.
                - 'diversity_factor': Optional diversity factor override.
                - 'load_density_override': Optional W/m2 override.

        Returns:
            Estimated consumption in kWh as Decimal.
        """
        area = _decimal(entity.floor_area_m2)
        if area == Decimal("0"):
            return Decimal("0")

        # Load density: override or lookup by entity type.
        load_density_override = building_profile.get("load_density_override")
        if load_density_override is not None:
            load_density = _decimal(load_density_override)
        else:
            load_density = self._load_densities.get(
                entity.entity_type, Decimal("25.0")
            )

        # Operating hours for the period.
        period_hours = _decimal(building_profile.get("period_hours", 730))

        # Adjust for actual operating hours vs standard.
        standard_hours = self._operating_hours.get(
            entity.entity_type, Decimal("50.0")
        )
        actual_hours = _decimal(entity.operating_hours_per_week)
        if actual_hours > Decimal("0") and standard_hours > Decimal("0"):
            hours_factor = _safe_divide(actual_hours, standard_hours)
        else:
            hours_factor = Decimal("1")

        # Diversity factor.
        diversity = _decimal(
            building_profile.get("diversity_factor", "0.85")
        )

        # Estimated kWh = W/m2 * m2 * hours * diversity * hours_factor / 1000.
        estimated_kwh = (
            load_density * area * period_hours * diversity * hours_factor
            / Decimal("1000")
        )

        logger.debug(
            "Virtual submeter estimate for %s: %.2f kWh "
            "(%.1f W/m2, %.0f m2, %.0f hrs, diversity=%.2f, hours_factor=%.2f)",
            entity.entity_id, estimated_kwh, load_density, area,
            period_hours, diversity, hours_factor,
        )

        return estimated_kwh

    # -------------------------------------------------------------------
    # Internal -- Allocation Dispatch
    # -------------------------------------------------------------------

    def _dispatch_allocation(
        self,
        entities: List[AllocationEntity],
        pool: CostPool,
        method: AllocationMethod,
        sub_meters: List[SubMeterData],
        rule_map: Dict[str, AllocationRule],
    ) -> List[AllocationLineItem]:
        """Dispatch allocation to the correct method handler.

        Args:
            entities: Entities to allocate to.
            pool: Cost pool being allocated.
            method: Allocation method to use.
            sub_meters: Sub-meter data.
            rule_map: Rule lookup for parameters.

        Returns:
            List of allocation line items.
        """
        total = _decimal(pool.total_amount_eur)
        component = pool.cost_component

        if method == AllocationMethod.AREA_PRORATION:
            items = self._allocate_by_area_internal(entities, total, component)
        elif method in (AllocationMethod.DIRECT_METER, AllocationMethod.SUB_METER):
            items = self._allocate_by_meter_internal(
                entities, sub_meters, total, component
            )
        elif method == AllocationMethod.HEADCOUNT:
            items = self._allocate_by_headcount(entities, total, component)
        elif method == AllocationMethod.PRODUCTION_BASED:
            items = self._allocate_by_production(entities, total, component)
        elif method == AllocationMethod.OPERATING_HOURS:
            items = self._allocate_by_operating_hours(entities, total, component)
        elif method == AllocationMethod.WEIGHTED_COMBINATION:
            rule = rule_map.get(component.value)
            weights = rule.parameters if rule else {}
            items = self._allocate_weighted(entities, total, component, weights)
        elif method == AllocationMethod.FIXED_PERCENTAGE:
            items = self._allocate_fixed_percentage(entities, total, component)
        elif method == AllocationMethod.REGRESSION:
            items = self._allocate_by_regression(entities, total, component)
        else:
            logger.warning("Unknown allocation method: %s", method)
            items = []

        return items

    # -------------------------------------------------------------------
    # Internal -- Individual Allocation Methods
    # -------------------------------------------------------------------

    def _allocate_by_area_internal(
        self,
        entities: List[AllocationEntity],
        total_cost: Decimal,
        component: CostComponent,
    ) -> List[AllocationLineItem]:
        """Area proration allocation (internal, with component tagging)."""
        total_area = sum(_decimal(e.floor_area_m2) for e in entities)
        if total_area == Decimal("0"):
            logger.warning("Total area zero for component %s", component.value)
            return []

        items: List[AllocationLineItem] = []
        for entity in entities:
            area = _decimal(entity.floor_area_m2)
            share = _safe_divide(area, total_area)
            amount = total_cost * share
            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=component.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=AllocationMethod.AREA_PRORATION.value,
                basis_value=_round2(area),
                basis_unit="m2",
                share_pct=_round4(_safe_pct(area, total_area)),
            ))
        return items

    def _allocate_by_meter_internal(
        self,
        entities: List[AllocationEntity],
        sub_meters: List[SubMeterData],
        total_cost: Decimal,
        component: CostComponent,
    ) -> List[AllocationLineItem]:
        """Sub-meter allocation (internal, with component tagging)."""
        consumption_by_entity: Dict[str, Decimal] = {}
        for sm in sub_meters:
            current = consumption_by_entity.get(sm.entity_id, Decimal("0"))
            consumption_by_entity[sm.entity_id] = current + _decimal(sm.consumption_kwh)

        total_consumption = sum(consumption_by_entity.values())
        if total_consumption == Decimal("0"):
            logger.warning(
                "Total metered consumption zero for component %s; "
                "falling back to area proration",
                component.value,
            )
            return self._allocate_by_area_internal(entities, total_cost, component)

        items: List[AllocationLineItem] = []
        for entity in entities:
            kwh = consumption_by_entity.get(entity.entity_id, Decimal("0"))
            share = _safe_divide(kwh, total_consumption)
            amount = total_cost * share
            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=component.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=AllocationMethod.SUB_METER.value,
                basis_value=_round2(kwh),
                basis_unit="kWh",
                share_pct=_round4(_safe_pct(kwh, total_consumption)),
            ))
        return items

    def _allocate_by_headcount(
        self,
        entities: List[AllocationEntity],
        total_cost: Decimal,
        component: CostComponent,
    ) -> List[AllocationLineItem]:
        """Allocate cost proportional to headcount."""
        total_headcount = sum(_decimal(e.headcount) for e in entities)
        if total_headcount == Decimal("0"):
            logger.warning("Total headcount zero; falling back to area proration")
            return self._allocate_by_area_internal(entities, total_cost, component)

        items: List[AllocationLineItem] = []
        for entity in entities:
            hc = _decimal(entity.headcount)
            share = _safe_divide(hc, total_headcount)
            amount = total_cost * share
            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=component.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=AllocationMethod.HEADCOUNT.value,
                basis_value=float(hc),
                basis_unit="persons",
                share_pct=_round4(_safe_pct(hc, total_headcount)),
            ))
        return items

    def _allocate_by_production(
        self,
        entities: List[AllocationEntity],
        total_cost: Decimal,
        component: CostComponent,
    ) -> List[AllocationLineItem]:
        """Allocate cost proportional to production units."""
        total_units = sum(_decimal(e.production_units) for e in entities)
        if total_units == Decimal("0"):
            logger.warning("Total production units zero; falling back to area")
            return self._allocate_by_area_internal(entities, total_cost, component)

        items: List[AllocationLineItem] = []
        for entity in entities:
            units = _decimal(entity.production_units)
            share = _safe_divide(units, total_units)
            amount = total_cost * share
            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=component.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=AllocationMethod.PRODUCTION_BASED.value,
                basis_value=_round2(units),
                basis_unit="units",
                share_pct=_round4(_safe_pct(units, total_units)),
            ))
        return items

    def _allocate_by_operating_hours(
        self,
        entities: List[AllocationEntity],
        total_cost: Decimal,
        component: CostComponent,
    ) -> List[AllocationLineItem]:
        """Allocate cost proportional to weekly operating hours."""
        total_hours = sum(
            _decimal(e.operating_hours_per_week) for e in entities
        )
        if total_hours == Decimal("0"):
            logger.warning("Total operating hours zero; falling back to area")
            return self._allocate_by_area_internal(entities, total_cost, component)

        items: List[AllocationLineItem] = []
        for entity in entities:
            hours = _decimal(entity.operating_hours_per_week)
            share = _safe_divide(hours, total_hours)
            amount = total_cost * share
            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=component.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=AllocationMethod.OPERATING_HOURS.value,
                basis_value=_round2(hours),
                basis_unit="hours/week",
                share_pct=_round4(_safe_pct(hours, total_hours)),
            ))
        return items

    def _allocate_weighted(
        self,
        entities: List[AllocationEntity],
        total_cost: Decimal,
        component: CostComponent,
        weight_params: Dict[str, Any],
    ) -> List[AllocationLineItem]:
        """Weighted combination allocation using multiple factors.

        Combines area, headcount, and operating hours with configurable
        weights.  Falls back to default weights if not specified.

        Formula:
            entity_score = w_area * (area/total_area) +
                           w_hc * (headcount/total_hc) +
                           w_hours * (hours/total_hours)
            entity_cost = total_cost * entity_score / sum(entity_scores)
        """
        w_area = _decimal(weight_params.get("area_weight",
                          self._default_weights["area"]))
        w_hc = _decimal(weight_params.get("headcount_weight",
                        self._default_weights["headcount"]))
        w_hours = _decimal(weight_params.get("hours_weight",
                           self._default_weights["operating_hours"]))

        total_area = sum(_decimal(e.floor_area_m2) for e in entities)
        total_hc = sum(_decimal(e.headcount) for e in entities)
        total_hours = sum(
            _decimal(e.operating_hours_per_week) for e in entities
        )

        # Compute weighted score for each entity.
        scores: Dict[str, Decimal] = {}
        for entity in entities:
            area_ratio = _safe_divide(_decimal(entity.floor_area_m2), total_area)
            hc_ratio = _safe_divide(_decimal(entity.headcount), total_hc)
            hours_ratio = _safe_divide(
                _decimal(entity.operating_hours_per_week), total_hours
            )
            score = w_area * area_ratio + w_hc * hc_ratio + w_hours * hours_ratio
            scores[entity.entity_id] = score

        total_score = sum(scores.values())
        if total_score == Decimal("0"):
            logger.warning("Total weighted score zero; falling back to area")
            return self._allocate_by_area_internal(entities, total_cost, component)

        items: List[AllocationLineItem] = []
        for entity in entities:
            score = scores[entity.entity_id]
            share = _safe_divide(score, total_score)
            amount = total_cost * share
            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=component.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=AllocationMethod.WEIGHTED_COMBINATION.value,
                basis_value=_round4(score),
                basis_unit="weighted_score",
                share_pct=_round4(_safe_pct(score, total_score)),
            ))
        return items

    def _allocate_fixed_percentage(
        self,
        entities: List[AllocationEntity],
        total_cost: Decimal,
        component: CostComponent,
    ) -> List[AllocationLineItem]:
        """Allocate by fixed contractual percentages.

        Uses entity.allocation_weight as fixed percentage (0-100).
        """
        total_weight = sum(_decimal(e.allocation_weight) for e in entities)
        if total_weight == Decimal("0"):
            logger.warning("Total fixed weights zero; falling back to area")
            return self._allocate_by_area_internal(entities, total_cost, component)

        items: List[AllocationLineItem] = []
        for entity in entities:
            weight = _decimal(entity.allocation_weight)
            share = _safe_divide(weight, total_weight)
            amount = total_cost * share
            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=component.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=AllocationMethod.FIXED_PERCENTAGE.value,
                basis_value=_round4(weight),
                basis_unit="percent",
                share_pct=_round4(_safe_pct(weight, total_weight)),
            ))
        return items

    def _allocate_by_regression(
        self,
        entities: List[AllocationEntity],
        total_cost: Decimal,
        component: CostComponent,
    ) -> List[AllocationLineItem]:
        """Regression-based allocation using area and operating hours.

        Uses a simple two-variable linear model:
            predicted_load = beta_0 + beta_1 * area + beta_2 * hours
        where beta coefficients are derived from load densities and
        standard operating hours.

        This is a deterministic engineering estimate, not a statistical
        fit from historical data (which would require a separate
        regression training step).
        """
        # Use load density as beta_1 proxy and hours factor as beta_2 proxy.
        predicted_loads: Dict[str, Decimal] = {}
        for entity in entities:
            load_density = self._load_densities.get(
                entity.entity_type, Decimal("25.0")
            )
            area = _decimal(entity.floor_area_m2)
            hours = _decimal(entity.operating_hours_per_week)
            std_hours = self._operating_hours.get(
                entity.entity_type, Decimal("50.0")
            )
            hours_factor = _safe_divide(hours, std_hours, Decimal("1"))

            # Predicted load in kW = load_density * area * hours_factor / 1000.
            predicted = load_density * area * hours_factor / Decimal("1000")
            predicted_loads[entity.entity_id] = predicted

        total_predicted = sum(predicted_loads.values())
        if total_predicted == Decimal("0"):
            logger.warning("Total predicted load zero; falling back to area")
            return self._allocate_by_area_internal(entities, total_cost, component)

        items: List[AllocationLineItem] = []
        for entity in entities:
            load = predicted_loads[entity.entity_id]
            share = _safe_divide(load, total_predicted)
            amount = total_cost * share
            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=component.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=AllocationMethod.REGRESSION.value,
                basis_value=_round2(load),
                basis_unit="kW_predicted",
                share_pct=_round4(_safe_pct(load, total_predicted)),
            ))
        return items

    # -------------------------------------------------------------------
    # Internal -- Demand Allocation Methods
    # -------------------------------------------------------------------

    def _allocate_demand_by_peak(
        self,
        entities: List[AllocationEntity],
        sub_meters: List[SubMeterData],
        total_demand_cost: Decimal,
        method: DemandAllocationMethod,
    ) -> List[AllocationLineItem]:
        """Allocate demand cost by peak kW (coincident or non-coincident).

        Coincident peak: entity's kW at the time of building peak.
        Non-coincident peak: entity's own maximum kW (regardless of timing).

        For coincident peak, we use the maximum peak_demand_kw value from
        each entity's sub-meter data.  In practice, coincident peak would
        require interval data aligned to building peak; here we use the
        reported peak as a reasonable proxy.

        Formula:
            entity_demand_cost = total_demand_cost * (entity_kw / sum_all_kw)
        """
        # Collect peak demand per entity.
        peak_by_entity: Dict[str, Decimal] = {}
        for sm in sub_meters:
            current = peak_by_entity.get(sm.entity_id, Decimal("0"))
            peak_val = _decimal(sm.peak_demand_kw)
            if method == DemandAllocationMethod.NON_COINCIDENT_PEAK:
                # Take each entity's own max peak.
                peak_by_entity[sm.entity_id] = max(current, peak_val)
            else:
                # Coincident peak: accumulate (assumes single peak period).
                if peak_val > current:
                    peak_by_entity[sm.entity_id] = peak_val

        total_peak = sum(peak_by_entity.values())
        if total_peak == Decimal("0"):
            logger.warning("Total peak demand zero; falling back to area")
            return self._allocate_demand_by_area(entities, total_demand_cost)

        items: List[AllocationLineItem] = []
        for entity in entities:
            kw = peak_by_entity.get(entity.entity_id, Decimal("0"))
            share = _safe_divide(kw, total_peak)
            amount = total_demand_cost * share
            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=CostComponent.DEMAND.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=method.value,
                basis_value=_round2(kw),
                basis_unit="kW",
                share_pct=_round4(_safe_pct(kw, total_peak)),
            ))
        return items

    def _allocate_demand_by_area(
        self,
        entities: List[AllocationEntity],
        total_demand_cost: Decimal,
    ) -> List[AllocationLineItem]:
        """Allocate demand cost by area ratio (fallback method)."""
        total_area = sum(_decimal(e.floor_area_m2) for e in entities)
        if total_area == Decimal("0"):
            return []

        items: List[AllocationLineItem] = []
        for entity in entities:
            area = _decimal(entity.floor_area_m2)
            share = _safe_divide(area, total_area)
            amount = total_demand_cost * share
            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=CostComponent.DEMAND.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=DemandAllocationMethod.AREA_BASED.value,
                basis_value=_round2(area),
                basis_unit="m2",
                share_pct=_round4(_safe_pct(area, total_area)),
            ))
        return items

    def _allocate_demand_four_cp(
        self,
        entities: List[AllocationEntity],
        sub_meters: List[SubMeterData],
        total_demand_cost: Decimal,
    ) -> List[AllocationLineItem]:
        """Four Coincident Peak (4CP) demand allocation.

        Averages each entity's demand at 4 system peak periods and
        allocates cost proportionally.

        Formula:
            avg_demand = avg(entity_demand_at_4_system_peaks)
            entity_cost = total_demand_cost * (avg_demand / sum_all_avg)

        Since we may not have exactly 4 peak periods in the data, we
        use the top-4 peak readings per entity (or all if fewer than 4).
        """
        # Collect all peak readings per entity, take top-4 average.
        readings_by_entity: Dict[str, List[Decimal]] = {}
        for sm in sub_meters:
            if sm.peak_demand_kw > 0:
                readings_by_entity.setdefault(sm.entity_id, []).append(
                    _decimal(sm.peak_demand_kw)
                )

        avg_by_entity: Dict[str, Decimal] = {}
        for entity_id, readings in readings_by_entity.items():
            # Sort descending, take top 4.
            sorted_readings = sorted(readings, reverse=True)[:4]
            avg_demand = _safe_divide(
                sum(sorted_readings), _decimal(len(sorted_readings))
            )
            avg_by_entity[entity_id] = avg_demand

        total_avg = sum(avg_by_entity.values())
        if total_avg == Decimal("0"):
            logger.warning("4CP total average zero; falling back to area")
            return self._allocate_demand_by_area(entities, total_demand_cost)

        items: List[AllocationLineItem] = []
        for entity in entities:
            avg_kw = avg_by_entity.get(entity.entity_id, Decimal("0"))
            share = _safe_divide(avg_kw, total_avg)
            amount = total_demand_cost * share
            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=CostComponent.DEMAND.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=DemandAllocationMethod.FOUR_CP.value,
                basis_value=_round2(avg_kw),
                basis_unit="kW_4cp_avg",
                share_pct=_round4(_safe_pct(avg_kw, total_avg)),
            ))
        return items

    def _allocate_demand_diversified(
        self,
        entities: List[AllocationEntity],
        sub_meters: List[SubMeterData],
        total_demand_cost: Decimal,
    ) -> List[AllocationLineItem]:
        """Diversified demand allocation adjusted for diversity factor.

        Each entity's non-coincident peak is adjusted by a diversity
        factor based on the total number of entities.

        Formula:
            diversified_kw = entity_ncp_kw * diversity_factor
            entity_cost = total * (diversified_kw / sum_all_diversified)
        """
        n_entities = len(entities)
        diversity = self._get_diversity_factor(n_entities)

        # Get non-coincident peak per entity.
        peak_by_entity: Dict[str, Decimal] = {}
        for sm in sub_meters:
            current = peak_by_entity.get(sm.entity_id, Decimal("0"))
            peak_val = _decimal(sm.peak_demand_kw)
            peak_by_entity[sm.entity_id] = max(current, peak_val)

        # Apply diversity factor.
        diversified: Dict[str, Decimal] = {
            eid: peak * diversity for eid, peak in peak_by_entity.items()
        }

        total_diversified = sum(diversified.values())
        if total_diversified == Decimal("0"):
            logger.warning("Diversified total zero; falling back to area")
            return self._allocate_demand_by_area(entities, total_demand_cost)

        items: List[AllocationLineItem] = []
        for entity in entities:
            div_kw = diversified.get(entity.entity_id, Decimal("0"))
            share = _safe_divide(div_kw, total_diversified)
            amount = total_demand_cost * share
            items.append(AllocationLineItem(
                entity_id=entity.entity_id,
                cost_component=CostComponent.DEMAND.value,
                allocated_amount_eur=_round2(amount),
                allocation_method=DemandAllocationMethod.DIVERSIFIED.value,
                basis_value=_round2(div_kw),
                basis_unit="kW_diversified",
                share_pct=_round4(_safe_pct(div_kw, total_diversified)),
            ))
        return items

    # -------------------------------------------------------------------
    # Internal -- Gini Coefficient
    # -------------------------------------------------------------------

    def _compute_gini(self, sorted_values: List[Decimal]) -> Decimal:
        """Compute Gini coefficient from sorted values.

        Formula:
            Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n

        where i is 1-indexed rank and x_i are sorted ascending values.

        Args:
            sorted_values: Values sorted in ascending order.

        Returns:
            Gini coefficient as Decimal (0 = equality, 1 = inequality).
        """
        n = _decimal(len(sorted_values))
        if n == Decimal("0"):
            return Decimal("0")

        sum_xi = sum(sorted_values)
        if sum_xi == Decimal("0"):
            return Decimal("0")

        weighted_sum = sum(
            _decimal(i + 1) * v for i, v in enumerate(sorted_values)
        )

        gini = (
            (Decimal("2") * weighted_sum) / (n * sum_xi)
            - (n + Decimal("1")) / n
        )

        # Clamp to [0, 1].
        return max(Decimal("0"), min(Decimal("1"), gini))

    # -------------------------------------------------------------------
    # Internal -- Helpers
    # -------------------------------------------------------------------

    def _build_rule_map(
        self, rules: List[AllocationRule]
    ) -> Dict[str, AllocationRule]:
        """Build a lookup of cost component -> highest-priority rule.

        When multiple rules exist for the same component, the one with
        the lowest priority number (highest priority) wins.

        Args:
            rules: List of allocation rules.

        Returns:
            Dict mapping component value to the best rule.
        """
        rule_map: Dict[str, AllocationRule] = {}
        for rule in sorted(rules, key=lambda r: r.priority):
            key = rule.cost_component.value
            if key not in rule_map:
                rule_map[key] = rule
        return rule_map

    def _resolve_method(
        self,
        pool: CostPool,
        rule_map: Dict[str, AllocationRule],
    ) -> Optional[AllocationMethod]:
        """Resolve the allocation method for a cost pool.

        Priority: pool-level override > matching rule > None.

        Args:
            pool: Cost pool being allocated.
            rule_map: Component -> rule lookup.

        Returns:
            AllocationMethod or None if no rule found.
        """
        if pool.allocation_method is not None:
            return pool.allocation_method

        rule = rule_map.get(pool.cost_component.value)
        if rule is not None:
            return rule.method

        return None

    def _distribute_residual(
        self,
        items: List[AllocationLineItem],
        residual: Decimal,
    ) -> List[AllocationLineItem]:
        """Distribute rounding residual to the entity with largest allocation.

        Ensures that the sum of allocations exactly equals the pool total
        after rounding.

        Args:
            items: Current allocation line items.
            residual: Rounding residual (positive or negative EUR).

        Returns:
            Updated list with residual absorbed by largest item.
        """
        if not items:
            return items

        # Find the item with the largest allocation.
        max_idx = 0
        max_amount = Decimal("0")
        for idx, item in enumerate(items):
            amt = _decimal(item.allocated_amount_eur)
            if amt > max_amount:
                max_amount = amt
                max_idx = idx

        # Adjust the largest item.
        adjusted = _decimal(items[max_idx].allocated_amount_eur) + residual
        items[max_idx].allocated_amount_eur = _round2(adjusted)

        return items

    def _compute_entity_totals(
        self, items: List[AllocationLineItem]
    ) -> Dict[str, float]:
        """Compute total allocated amount per entity.

        Args:
            items: All allocation line items.

        Returns:
            Dict mapping entity_id to total EUR.
        """
        totals: Dict[str, Decimal] = {}
        for item in items:
            current = totals.get(item.entity_id, Decimal("0"))
            totals[item.entity_id] = current + _decimal(item.allocated_amount_eur)
        return {k: _round2(v) for k, v in totals.items()}

    def _compute_component_totals(
        self, items: List[AllocationLineItem]
    ) -> Dict[str, float]:
        """Compute total allocated amount per cost component.

        Args:
            items: All allocation line items.

        Returns:
            Dict mapping component to total EUR.
        """
        totals: Dict[str, Decimal] = {}
        for item in items:
            current = totals.get(item.cost_component, Decimal("0"))
            totals[item.cost_component] = current + _decimal(item.allocated_amount_eur)
        return {k: _round2(v) for k, v in totals.items()}

    def _classify_variance(
        self, variance_pct: Decimal
    ) -> ReconciliationStatus:
        """Classify variance percentage into a reconciliation status.

        Thresholds:
            <= 1.0%: RECONCILED
            <= 3.0%: VARIANCE_LOW
            <= 5.0%: VARIANCE_HIGH
            >  5.0%: UNRECONCILED

        Args:
            variance_pct: Absolute variance as percentage.

        Returns:
            ReconciliationStatus enum member.
        """
        if variance_pct <= _RECONCILED_THRESHOLD:
            return ReconciliationStatus.RECONCILED
        if variance_pct <= _VARIANCE_LOW_THRESHOLD:
            return ReconciliationStatus.VARIANCE_LOW
        if variance_pct <= _VARIANCE_HIGH_THRESHOLD:
            return ReconciliationStatus.VARIANCE_HIGH
        return ReconciliationStatus.UNRECONCILED

    def _get_diversity_factor(self, n_entities: int) -> Decimal:
        """Get demand diversity factor based on entity count.

        Diversity factor reduces the sum of non-coincident peaks to
        estimate coincident peak demand for groups of tenants.

        Args:
            n_entities: Number of entities in the building.

        Returns:
            Diversity factor as Decimal (0 to 1).
        """
        if n_entities <= 1:
            return Decimal("1.0")
        if n_entities <= 5:
            return self._diversity_factors["2_to_5"]
        if n_entities <= 10:
            return self._diversity_factors["6_to_10"]
        if n_entities <= 20:
            return self._diversity_factors["11_to_20"]
        if n_entities <= 50:
            return self._diversity_factors["21_to_50"]
        return self._diversity_factors["51_plus"]
