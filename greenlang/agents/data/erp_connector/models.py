# -*- coding: utf-8 -*-
"""
ERP/Finance Connector Service Data Models - AGENT-DATA-003: ERP Connector

Pydantic v2 data models for the ERP/Finance Connector SDK. Re-exports the
Layer 1 enumerations and models from the foundation agent, and defines
additional SDK models for connection records, sync jobs, spend summaries,
Scope 3 summaries, emission results, currency rates, statistics, and
request wrappers.

Models:
    - Re-exported enums: ERPSystem, Scope3Category, TransactionType,
        SpendCategory
    - Re-exported Layer 1: ERPConnectionConfig, VendorMapping,
        MaterialMapping, PurchaseOrderLine, PurchaseOrder, SpendRecord,
        InventoryItem, ERPQueryInput, ERPQueryOutput
    - Re-exported constants: SPEND_TO_SCOPE3_MAPPING, DEFAULT_EMISSION_FACTORS
    - New enums: ConnectionStatus, SyncMode, EmissionMethodology
    - SDK models: ConnectionRecord, SyncJob, SpendSummary, Scope3Summary,
        EmissionResult, CurrencyRate, ERPStatistics
    - Request models: RegisterConnectionRequest, SyncSpendRequest,
        MapVendorRequest, CalculateEmissionsRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-003 ERP/Finance Connector
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Re-export Layer 1 enumerations
# ---------------------------------------------------------------------------

from greenlang.agents.data.erp_connector_agent import (
    ERPSystem,
    Scope3Category,
    TransactionType,
    SpendCategory,
)

# ---------------------------------------------------------------------------
# Re-export Layer 1 models
# ---------------------------------------------------------------------------

from greenlang.agents.data.erp_connector_agent import (
    ERPConnectionConfig,
    VendorMapping,
    MaterialMapping,
    PurchaseOrderLine,
    PurchaseOrder,
    SpendRecord,
    InventoryItem,
    ERPQueryInput,
    ERPQueryOutput,
)

# ---------------------------------------------------------------------------
# Re-export Layer 1 constants
# ---------------------------------------------------------------------------

from greenlang.agents.data.erp_connector_agent import (
    SPEND_TO_SCOPE3_MAPPING,
    DEFAULT_EMISSION_FACTORS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# New Enumerations
# =============================================================================


class ConnectionStatus(str, Enum):
    """Lifecycle status of an ERP connection."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    TESTING = "testing"
    INITIALIZING = "initializing"


class SyncMode(str, Enum):
    """Data synchronization modes for ERP data pulls."""

    FULL = "full"
    INCREMENTAL = "incremental"
    DELTA = "delta"
    MANUAL = "manual"


class EmissionMethodology(str, Enum):
    """Methodology for estimating emissions from spend data."""

    EEIO = "eeio"
    HYBRID = "hybrid"
    PROCESS_BASED = "process_based"
    SUPPLIER_SPECIFIC = "supplier_specific"


# =============================================================================
# SDK Data Models
# =============================================================================


class ConnectionRecord(BaseModel):
    """Persistent record of a registered ERP connection.

    Captures metadata about an ERP connection including status, sync
    history, and error tracking for monitoring and audit purposes.

    Attributes:
        connection_id: Unique identifier for this connection record.
        erp_system: ERP system type for this connection.
        host: ERP server hostname or endpoint URL.
        status: Current lifecycle status of the connection.
        last_sync: Timestamp of the most recent successful sync.
        sync_count: Total number of sync operations performed.
        error_count: Total number of errors encountered.
        created_at: Timestamp when the connection was registered.
        tenant_id: Tenant identifier for multi-tenant isolation.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    connection_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this connection record",
    )
    erp_system: ERPSystem = Field(
        default=ERPSystem.SIMULATED,
        description="ERP system type for this connection",
    )
    host: str = Field(
        ..., description="ERP server hostname or endpoint URL",
    )
    status: ConnectionStatus = Field(
        default=ConnectionStatus.INITIALIZING,
        description="Current lifecycle status of the connection",
    )
    last_sync: Optional[datetime] = Field(
        None, description="Timestamp of the most recent successful sync",
    )
    sync_count: int = Field(
        default=0, ge=0,
        description="Total number of sync operations performed",
    )
    error_count: int = Field(
        default=0, ge=0,
        description="Total number of errors encountered",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the connection was registered",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host is non-empty."""
        if not v or not v.strip():
            raise ValueError("host must be non-empty")
        return v


class SyncJob(BaseModel):
    """Record of a data synchronization job execution.

    Tracks the lifecycle, configuration, and results of a single
    ERP data sync operation for monitoring and audit purposes.

    Attributes:
        job_id: Unique identifier for this sync job.
        connection_id: ERP connection used for this sync.
        sync_mode: Synchronization mode used.
        query_type: Type of data queried (spend, po, inventory).
        status: Current lifecycle status of the sync job.
        records_synced: Number of records successfully synchronized.
        records_skipped: Number of records skipped during sync.
        errors: List of error messages encountered during sync.
        started_at: Timestamp when the sync job started.
        completed_at: Timestamp when the sync job finished.
        duration_seconds: Total sync duration in seconds.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this sync job",
    )
    connection_id: str = Field(
        ..., description="ERP connection used for this sync",
    )
    sync_mode: SyncMode = Field(
        default=SyncMode.FULL,
        description="Synchronization mode used",
    )
    query_type: str = Field(
        default="spend",
        description="Type of data queried (spend, po, inventory)",
    )
    status: str = Field(
        default="pending",
        description="Current lifecycle status of the sync job",
    )
    records_synced: int = Field(
        default=0, ge=0,
        description="Number of records successfully synchronized",
    )
    records_skipped: int = Field(
        default=0, ge=0,
        description="Number of records skipped during sync",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of error messages encountered during sync",
    )
    started_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the sync job started",
    )
    completed_at: Optional[datetime] = Field(
        None, description="Timestamp when the sync job finished",
    )
    duration_seconds: float = Field(
        default=0.0, ge=0.0,
        description="Total sync duration in seconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("connection_id")
    @classmethod
    def validate_connection_id(cls, v: str) -> str:
        """Validate connection_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("connection_id must be non-empty")
        return v


class SpendSummary(BaseModel):
    """Aggregated spend summary for a given period.

    Provides high-level spend statistics including total amounts,
    record counts, category breakdowns, and Scope 3 distribution.

    Attributes:
        period_start: Start date of the summary period.
        period_end: End date of the summary period.
        total_spend_usd: Total spend amount in USD.
        record_count: Total number of spend records in the period.
        vendor_count: Number of unique vendors in the period.
        category_breakdown: Spend amount by spend category.
        top_vendors: List of top vendors by spend amount.
        scope3_distribution: Spend amount by Scope 3 category.
    """

    period_start: date = Field(
        ..., description="Start date of the summary period",
    )
    period_end: date = Field(
        ..., description="End date of the summary period",
    )
    total_spend_usd: float = Field(
        default=0.0, ge=0.0,
        description="Total spend amount in USD",
    )
    record_count: int = Field(
        default=0, ge=0,
        description="Total number of spend records in the period",
    )
    vendor_count: int = Field(
        default=0, ge=0,
        description="Number of unique vendors in the period",
    )
    category_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Spend amount by spend category",
    )
    top_vendors: List[Dict[str, float]] = Field(
        default_factory=list,
        description="List of top vendors by spend amount",
    )
    scope3_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Spend amount by Scope 3 category",
    )

    model_config = {"extra": "forbid"}


class Scope3Summary(BaseModel):
    """Summary of spend and emissions for a single Scope 3 category.

    Provides detailed statistics for a single GHG Protocol Scope 3
    category including spend, vendor count, and estimated emissions.

    Attributes:
        category: Scope 3 category enum value.
        category_name: Human-readable Scope 3 category name.
        total_spend_usd: Total spend amount in USD for this category.
        record_count: Number of spend records in this category.
        vendor_count: Number of unique vendors in this category.
        estimated_emissions_kgco2e: Estimated emissions in kgCO2e.
        emission_factor_source: Source of the emission factor used.
        percentage_of_total: Percentage of total spend this represents.
    """

    category: Scope3Category = Field(
        ..., description="Scope 3 category enum value",
    )
    category_name: str = Field(
        default="", description="Human-readable Scope 3 category name",
    )
    total_spend_usd: float = Field(
        default=0.0, ge=0.0,
        description="Total spend amount in USD for this category",
    )
    record_count: int = Field(
        default=0, ge=0,
        description="Number of spend records in this category",
    )
    vendor_count: int = Field(
        default=0, ge=0,
        description="Number of unique vendors in this category",
    )
    estimated_emissions_kgco2e: float = Field(
        default=0.0, ge=0.0,
        description="Estimated emissions in kgCO2e",
    )
    emission_factor_source: str = Field(
        default="",
        description="Source of the emission factor used",
    )
    percentage_of_total: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of total spend this represents",
    )

    model_config = {"extra": "forbid"}


class EmissionResult(BaseModel):
    """Result of an emissions calculation for a single spend record.

    Contains the calculated emissions estimate with full traceability
    to the source spend record, emission factor, and methodology.

    Attributes:
        result_id: Unique identifier for this emission result.
        spend_record_id: Source spend record identifier.
        vendor_id: Vendor associated with the spend.
        amount_usd: Spend amount in USD used for calculation.
        emission_factor: Emission factor applied (kgCO2e per USD).
        methodology: Emissions calculation methodology used.
        estimated_kgco2e: Estimated emissions in kgCO2e.
        emission_factor_source: Source of the emission factor.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this emission result",
    )
    spend_record_id: str = Field(
        ..., description="Source spend record identifier",
    )
    vendor_id: str = Field(
        ..., description="Vendor associated with the spend",
    )
    amount_usd: float = Field(
        default=0.0, ge=0.0,
        description="Spend amount in USD used for calculation",
    )
    emission_factor: float = Field(
        default=0.0, ge=0.0,
        description="Emission factor applied (kgCO2e per USD)",
    )
    methodology: EmissionMethodology = Field(
        default=EmissionMethodology.EEIO,
        description="Emissions calculation methodology used",
    )
    estimated_kgco2e: float = Field(
        default=0.0, ge=0.0,
        description="Estimated emissions in kgCO2e",
    )
    emission_factor_source: str = Field(
        default="",
        description="Source of the emission factor",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("spend_record_id")
    @classmethod
    def validate_spend_record_id(cls, v: str) -> str:
        """Validate spend_record_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("spend_record_id must be non-empty")
        return v

    @field_validator("vendor_id")
    @classmethod
    def validate_vendor_id(cls, v: str) -> str:
        """Validate vendor_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("vendor_id must be non-empty")
        return v


class CurrencyRate(BaseModel):
    """Exchange rate between two currencies.

    Represents a point-in-time exchange rate for currency conversion
    with source attribution and effective date tracking.

    Attributes:
        from_currency: Source currency code (ISO 4217).
        to_currency: Target currency code (ISO 4217).
        rate: Exchange rate (units of to_currency per from_currency).
        source: Source of the exchange rate data.
        effective_date: Date this rate is effective.
    """

    from_currency: str = Field(
        ..., description="Source currency code (ISO 4217)",
    )
    to_currency: str = Field(
        ..., description="Target currency code (ISO 4217)",
    )
    rate: float = Field(
        ..., gt=0.0,
        description="Exchange rate (units of to_currency per from_currency)",
    )
    source: str = Field(
        default="", description="Source of the exchange rate data",
    )
    effective_date: date = Field(
        ..., description="Date this rate is effective",
    )

    model_config = {"extra": "forbid"}

    @field_validator("from_currency")
    @classmethod
    def validate_from_currency(cls, v: str) -> str:
        """Validate from_currency is non-empty."""
        if not v or not v.strip():
            raise ValueError("from_currency must be non-empty")
        return v

    @field_validator("to_currency")
    @classmethod
    def validate_to_currency(cls, v: str) -> str:
        """Validate to_currency is non-empty."""
        if not v or not v.strip():
            raise ValueError("to_currency must be non-empty")
        return v


class ERPStatistics(BaseModel):
    """Aggregated statistics for the ERP connector service.

    Provides high-level operational metrics for monitoring the
    overall health and activity of the ERP connector.

    Attributes:
        total_connections: Total number of registered ERP connections.
        active_connections: Number of currently active connections.
        total_syncs: Total number of sync operations performed.
        total_spend_records: Total number of spend records processed.
        total_purchase_orders: Total number of purchase orders processed.
        total_emissions_calculated: Total emissions calculations performed.
        scope3_coverage_pct: Percentage of spend mapped to Scope 3.
        avg_sync_duration_seconds: Average sync operation duration.
    """

    total_connections: int = Field(
        default=0, ge=0,
        description="Total number of registered ERP connections",
    )
    active_connections: int = Field(
        default=0, ge=0,
        description="Number of currently active connections",
    )
    total_syncs: int = Field(
        default=0, ge=0,
        description="Total number of sync operations performed",
    )
    total_spend_records: int = Field(
        default=0, ge=0,
        description="Total number of spend records processed",
    )
    total_purchase_orders: int = Field(
        default=0, ge=0,
        description="Total number of purchase orders processed",
    )
    total_emissions_calculated: int = Field(
        default=0, ge=0,
        description="Total emissions calculations performed",
    )
    scope3_coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of spend mapped to Scope 3",
    )
    avg_sync_duration_seconds: float = Field(
        default=0.0, ge=0.0,
        description="Average sync operation duration",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# Request Models
# =============================================================================


class RegisterConnectionRequest(BaseModel):
    """Request body for registering a new ERP connection.

    Attributes:
        erp_system: ERP system type for the connection.
        host: ERP server hostname or endpoint URL.
        port: ERP server port number.
        username: API username for authentication.
        client_id: Client or company identifier.
        company_code: Company code within the ERP system.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    erp_system: ERPSystem = Field(
        ..., description="ERP system type for the connection",
    )
    host: str = Field(
        ..., description="ERP server hostname or endpoint URL",
    )
    port: int = Field(
        default=443, ge=1, le=65535,
        description="ERP server port number",
    )
    username: str = Field(
        ..., description="API username for authentication",
    )
    client_id: Optional[str] = Field(
        None, description="Client or company identifier",
    )
    company_code: Optional[str] = Field(
        None, description="Company code within the ERP system",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host is non-empty."""
        if not v or not v.strip():
            raise ValueError("host must be non-empty")
        return v

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username is non-empty."""
        if not v or not v.strip():
            raise ValueError("username must be non-empty")
        return v


class SyncSpendRequest(BaseModel):
    """Request body for synchronizing spend data from an ERP connection.

    Attributes:
        connection_id: ERP connection to sync from.
        start_date: Start date for the spend data query.
        end_date: End date for the spend data query.
        vendor_ids: Optional list of vendor IDs to filter.
        cost_centers: Optional list of cost centers to filter.
        sync_mode: Synchronization mode to use.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    connection_id: str = Field(
        ..., description="ERP connection to sync from",
    )
    start_date: date = Field(
        ..., description="Start date for the spend data query",
    )
    end_date: date = Field(
        ..., description="End date for the spend data query",
    )
    vendor_ids: Optional[List[str]] = Field(
        None, description="Optional list of vendor IDs to filter",
    )
    cost_centers: Optional[List[str]] = Field(
        None, description="Optional list of cost centers to filter",
    )
    sync_mode: SyncMode = Field(
        default=SyncMode.FULL,
        description="Synchronization mode to use",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("connection_id")
    @classmethod
    def validate_connection_id(cls, v: str) -> str:
        """Validate connection_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("connection_id must be non-empty")
        return v


class MapVendorRequest(BaseModel):
    """Request body for mapping a vendor to a Scope 3 category.

    Attributes:
        vendor_id: Vendor identifier to map.
        vendor_name: Human-readable vendor name.
        primary_category: Primary Scope 3 category for this vendor.
        spend_category: High-level spend category for this vendor.
        emission_factor: Optional vendor-specific emission factor.
    """

    vendor_id: str = Field(
        ..., description="Vendor identifier to map",
    )
    vendor_name: str = Field(
        ..., description="Human-readable vendor name",
    )
    primary_category: Scope3Category = Field(
        ..., description="Primary Scope 3 category for this vendor",
    )
    spend_category: SpendCategory = Field(
        ..., description="High-level spend category for this vendor",
    )
    emission_factor: Optional[float] = Field(
        None, ge=0.0,
        description="Optional vendor-specific emission factor (kgCO2e/USD)",
    )

    model_config = {"extra": "forbid"}

    @field_validator("vendor_id")
    @classmethod
    def validate_vendor_id(cls, v: str) -> str:
        """Validate vendor_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("vendor_id must be non-empty")
        return v

    @field_validator("vendor_name")
    @classmethod
    def validate_vendor_name(cls, v: str) -> str:
        """Validate vendor_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("vendor_name must be non-empty")
        return v


class CalculateEmissionsRequest(BaseModel):
    """Request body for calculating emissions from spend data.

    Attributes:
        connection_id: ERP connection whose spend data to use.
        start_date: Start date for the emissions calculation period.
        end_date: End date for the emissions calculation period.
        methodology: Emissions calculation methodology to use.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    connection_id: str = Field(
        ..., description="ERP connection whose spend data to use",
    )
    start_date: date = Field(
        ..., description="Start date for the emissions calculation period",
    )
    end_date: date = Field(
        ..., description="End date for the emissions calculation period",
    )
    methodology: EmissionMethodology = Field(
        default=EmissionMethodology.EEIO,
        description="Emissions calculation methodology to use",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("connection_id")
    @classmethod
    def validate_connection_id(cls, v: str) -> str:
        """Validate connection_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("connection_id must be non-empty")
        return v


__all__ = [
    # Re-exported enums
    "ERPSystem",
    "Scope3Category",
    "TransactionType",
    "SpendCategory",
    # Re-exported Layer 1 models
    "ERPConnectionConfig",
    "VendorMapping",
    "MaterialMapping",
    "PurchaseOrderLine",
    "PurchaseOrder",
    "SpendRecord",
    "InventoryItem",
    "ERPQueryInput",
    "ERPQueryOutput",
    # Re-exported constants
    "SPEND_TO_SCOPE3_MAPPING",
    "DEFAULT_EMISSION_FACTORS",
    # New enums
    "ConnectionStatus",
    "SyncMode",
    "EmissionMethodology",
    # SDK models
    "ConnectionRecord",
    "SyncJob",
    "SpendSummary",
    "Scope3Summary",
    "EmissionResult",
    "CurrencyRate",
    "ERPStatistics",
    # Request models
    "RegisterConnectionRequest",
    "SyncSpendRequest",
    "MapVendorRequest",
    "CalculateEmissionsRequest",
]
