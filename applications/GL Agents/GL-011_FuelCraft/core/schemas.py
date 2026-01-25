# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT Data Schemas

Pydantic schemas for fuel properties, emissions, inventory, pricing,
contracts, and optimization results per the build guide specification.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
import hashlib
import json


class HeatingValueBasis(str, Enum):
    """Heating value convention."""
    LHV = "lhv"  # Lower Heating Value
    HHV = "hhv"  # Higher Heating Value


class FuelProperty(BaseModel):
    """Fuel physical and chemical properties."""

    fuel_id: str = Field(..., min_length=1)
    fuel_name: str = Field(...)
    lhv_mj_per_kg: float = Field(..., gt=0, le=60)
    hhv_mj_per_kg: Optional[float] = Field(None, gt=0, le=65)
    density_kg_per_m3_ref: float = Field(..., gt=500, le=1100)
    reference_temp_c: float = Field(default=15.0)
    sulfur_wt_pct: Optional[float] = Field(None, ge=0, le=5)
    ash_wt_pct: Optional[float] = Field(None, ge=0, le=1)
    water_content_pct: Optional[float] = Field(None, ge=0, le=5)
    flash_point_c: Optional[float] = Field(None, ge=-50, le=300)
    pour_point_c: Optional[float] = Field(None, ge=-50, le=50)
    viscosity_cst_50c: Optional[float] = Field(None, ge=0.1, le=700)
    effective_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EmissionFactor(BaseModel):
    """Carbon emission factor with boundary specification."""

    factor_id: str = Field(...)
    fuel_id: str = Field(...)
    ci_kgco2e_per_mj_ttw: float = Field(..., ge=0)
    ci_kgco2e_per_mj_wtt: Optional[float] = Field(None, ge=0)
    ci_kgco2e_per_mj_wtw: Optional[float] = Field(None, ge=0)
    pathway: Optional[str] = Field(None)
    gwp_version: str = Field(default="AR6")
    source: str = Field(default="default")
    effective_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expiry_date: Optional[datetime] = Field(None)


class InventoryState(BaseModel):
    """Tank inventory state with quality tracking."""

    tank_id: str = Field(...)
    fuel_id: str = Field(...)
    volume_m3_ref: float = Field(..., ge=0)
    temperature_c: float = Field(...)
    density_kg_per_m3: Optional[float] = Field(None)
    level_pct: Optional[float] = Field(None, ge=0, le=100)
    water_interface_m3: Optional[float] = Field(None, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Price(BaseModel):
    """Fuel price with market information."""

    fuel_id: str = Field(...)
    price_value: float = Field(..., ge=0)
    currency: str = Field(default="USD")
    unit: str = Field(default="per_mj")
    market_hub: Optional[str] = Field(None)
    price_type: str = Field(default="spot")  # spot, forward, contract
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: Optional[datetime] = Field(None)
    valid_to: Optional[datetime] = Field(None)


class Contract(BaseModel):
    """Fuel supply contract terms."""

    contract_id: str = Field(...)
    fuel_id: str = Field(...)
    supplier_id: str = Field(...)
    min_take_mj: Optional[float] = Field(None, ge=0)
    max_take_mj: Optional[float] = Field(None, ge=0)
    price_formula: Optional[str] = Field(None)
    fixed_price_per_mj: Optional[float] = Field(None, ge=0)
    take_or_pay_pct: Optional[float] = Field(None, ge=0, le=100)
    penalty_per_mj: Optional[float] = Field(None, ge=0)
    valid_from: datetime = Field(...)
    valid_to: datetime = Field(...)
    delivery_lead_days: Optional[int] = Field(None, ge=0)


class Demand(BaseModel):
    """Energy demand forecast."""

    period_start: datetime = Field(...)
    period_end: datetime = Field(...)
    demand_mwh_th: float = Field(..., ge=0)
    demand_mj: Optional[float] = Field(None, ge=0)
    site_id: str = Field(...)
    unit_id: Optional[str] = Field(None)
    confidence_pct: Optional[float] = Field(None, ge=0, le=100)


class BlendRatio(BaseModel):
    """Fuel blend composition."""

    fuel_id: str = Field(...)
    ratio: float = Field(..., ge=0, le=1)
    period_start: datetime = Field(...)
    period_end: datetime = Field(...)

    @field_validator('ratio')
    @classmethod
    def validate_ratio(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Ratio must be between 0 and 1')
        return round(v, 6)


class OptimizationResult(BaseModel):
    """Complete optimization solution."""

    run_id: str = Field(...)
    site_id: str = Field(...)
    solve_time_seconds: float = Field(..., ge=0)
    objective_value: float = Field(...)
    solver_status: str = Field(...)
    gap_pct: Optional[float] = Field(None, ge=0)

    # Solution components
    fuel_mix: Dict[str, float] = Field(default_factory=dict)
    blend_ratios: List[BlendRatio] = Field(default_factory=list)
    procurement_schedule: List[Dict[str, Any]] = Field(default_factory=list)

    # Costs
    total_cost: float = Field(...)
    purchase_cost: float = Field(default=0)
    logistics_cost: float = Field(default=0)
    penalty_cost: float = Field(default=0)
    carbon_cost: float = Field(default=0)

    # Carbon
    total_emissions_tco2e: float = Field(default=0)
    carbon_intensity_kgco2e_per_mj: float = Field(default=0)

    # Metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = Field(default="")

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "run_id": self.run_id,
            "objective_value": self.objective_value,
            "fuel_mix": self.fuel_mix,
            "timestamp": self.timestamp.isoformat()
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


class RunBundle(BaseModel):
    """Immutable run bundle for reproducibility."""

    run_id: str = Field(...)
    bundle_hash: str = Field(...)

    # Inputs
    input_snapshot_ids: Dict[str, str] = Field(default_factory=dict)

    # Versions
    model_version: str = Field(...)
    solver_version: str = Field(...)
    feature_schema_version: str = Field(...)

    # Outputs
    result: Optional[OptimizationResult] = Field(None)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(None)

    # Status
    status: str = Field(default="pending")
    error_message: Optional[str] = Field(None)
