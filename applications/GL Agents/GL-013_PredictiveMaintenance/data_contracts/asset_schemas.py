"""GL-013 PredictiveMaintenance: Asset Schemas - Version 1.0"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

class AssetStatus(str, Enum):
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    DECOMMISSIONED = "decommissioned"
    STANDBY = "standby"

class AssetCriticality(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class BearingType(str, Enum):
    BALL = "ball"
    ROLLER = "roller"
    SPHERICAL_ROLLER = "spherical_roller"
    TAPERED_ROLLER = "tapered_roller"
    NEEDLE = "needle"
    THRUST = "thrust"
    SLEEVE = "sleeve"

class MaintenanceType(str, Enum):
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    CONDITION_BASED = "condition_based"
    OVERHAUL = "overhaul"

class SiteAlias(BaseModel):
    model_config = ConfigDict(frozen=True)
    site_id: str
    site_name: str
    alias_type: str = "scada"
    tag_prefix: Optional[str] = None

class VendorAlias(BaseModel):
    model_config = ConfigDict(frozen=True)
    vendor_id: str
    vendor_name: str
    vendor_asset_id: str
    system_type: str = "cmms"

class NameplateData(BaseModel):
    model_config = ConfigDict(frozen=True)
    manufacturer: str
    model: str
    serial_number: str
    rated_power_kw: Optional[float] = Field(default=None, ge=0)
    rated_speed_rpm: Optional[float] = Field(default=None, ge=0)
    rated_voltage_v: Optional[float] = Field(default=None, ge=0)
    rated_current_a: Optional[float] = Field(default=None, ge=0)
    rated_frequency_hz: Optional[float] = Field(default=None, ge=0)
    efficiency_class: Optional[str] = None
    ip_rating: Optional[str] = None

class BearingInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    position: str
    bearing_type: BearingType
    manufacturer: Optional[str] = None
    part_number: Optional[str] = None
    bpfo: Optional[float] = Field(default=None, gt=0)
    bpfi: Optional[float] = Field(default=None, gt=0)
    bsf: Optional[float] = Field(default=None, gt=0)
    ftf: Optional[float] = Field(default=None, gt=0)
    install_date: Optional[datetime] = None
    expected_life_hours: Optional[float] = Field(default=None, ge=0)

class EquipmentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    equipment_type: str
    nameplate: NameplateData
    bearings: List[BearingInfo] = Field(default_factory=list)
    coupling_type: Optional[str] = None
    lubrication_type: Optional[str] = None
    cooling_type: Optional[str] = None

class InstallationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    installation_date: datetime
    installed_by: str
    location: str
    foundation_type: Optional[str] = None
    alignment_method: Optional[str] = None
    commissioning_date: Optional[datetime] = None

class OverhaulRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    overhaul_id: str
    overhaul_date: datetime
    overhaul_type: str
    performed_by: str
    components_replaced: List[str] = Field(default_factory=list)
    cost_usd: Optional[float] = Field(default=None, ge=0)
    downtime_hours: Optional[float] = Field(default=None, ge=0)
    next_overhaul_hours: Optional[float] = Field(default=None, ge=0)

class MaintenanceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    record_id: str
    maintenance_type: MaintenanceType
    date: datetime
    description: str
    performed_by: str
    work_order_id: Optional[str] = None
    parts_used: List[str] = Field(default_factory=list)
    cost_usd: Optional[float] = Field(default=None, ge=0)
    findings: Optional[str] = None

class OperatingLimits(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_speed_rpm: Optional[float] = Field(default=None, ge=0)
    min_speed_rpm: Optional[float] = Field(default=None, ge=0)
    max_temp_c: Optional[float] = None
    max_vibration_mm_s: Optional[float] = Field(default=None, ge=0)
    max_current_a: Optional[float] = Field(default=None, ge=0)
    min_flow_m3h: Optional[float] = Field(default=None, ge=0)
    max_pressure_bar: Optional[float] = Field(default=None, ge=0)

class Asset(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True, extra="forbid")
    schema_version: str = Field(default="1.0")
    asset_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    status: AssetStatus = AssetStatus.OPERATIONAL
    criticality: AssetCriticality = AssetCriticality.MEDIUM
    site_aliases: List[SiteAlias] = Field(default_factory=list)
    vendor_aliases: List[VendorAlias] = Field(default_factory=list)
    equipment_spec: Optional[EquipmentSpec] = None
    installation: Optional[InstallationRecord] = None
    overhaul_history: List[OverhaulRecord] = Field(default_factory=list)
    maintenance_history: List[MaintenanceRecord] = Field(default_factory=list)
    operating_limits: Optional[OperatingLimits] = None
    parent_asset_id: Optional[str] = None
    child_asset_ids: List[str] = Field(default_factory=list)
    tags: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AssetHierarchy(BaseModel):
    model_config = ConfigDict(frozen=True)
    root_asset_id: str
    assets: Dict[str, Asset] = Field(default_factory=dict)
    parent_child_map: Dict[str, List[str]] = Field(default_factory=dict)

ASSET_SCHEMAS = {"SiteAlias": SiteAlias, "VendorAlias": VendorAlias, "NameplateData": NameplateData, "BearingInfo": BearingInfo, "EquipmentSpec": EquipmentSpec, "InstallationRecord": InstallationRecord, "OverhaulRecord": OverhaulRecord, "MaintenanceRecord": MaintenanceRecord, "OperatingLimits": OperatingLimits, "Asset": Asset, "AssetHierarchy": AssetHierarchy}
__all__ = ["AssetStatus", "AssetCriticality", "BearingType", "MaintenanceType", "SiteAlias", "VendorAlias", "NameplateData", "BearingInfo", "EquipmentSpec", "InstallationRecord", "OverhaulRecord", "MaintenanceRecord", "OperatingLimits", "Asset", "AssetHierarchy", "ASSET_SCHEMAS"]
