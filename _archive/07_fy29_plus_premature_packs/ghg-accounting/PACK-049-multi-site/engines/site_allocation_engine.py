# -*- coding: utf-8 -*-
"""
PACK-049 GHG Multi-Site Management Pack - Site Allocation Engine
=================================================================

Allocates shared-services, landlord-tenant, cogeneration, district
system, and virtual power purchase agreement (VPPA) emissions across
sites within an organisational portfolio.  Ensures that every tonne
of CO2e is attributed to exactly one cost centre / tenant / beneficiary,
preventing double-counting and under-reporting.

Allocation Methods:
    Shared Services:
        allocated_i = source_emissions * (key_i / SUM(key_j for j in targets))
        Validates: SUM(allocated_i) == source_emissions (within tolerance)

    Landlord-Tenant Split (GHG Protocol):
        tenant_share  = tenant_area + (common_area * tenant_area / total_area)
        tenant_emissions = whole_building * tenant_share / total_area
        landlord_emissions = whole_building - tenant_emissions

    Cogeneration / CHP (ISO 8302):
        Efficiency method:
            elec_emissions = fuel * (elec_output / (elec_output + heat_output))
            heat_emissions = fuel * (heat_output / (elec_output + heat_output))
        Energy content method:
            Same formula but uses primary-energy-weighted outputs

    District System:
        Per-site allocation proportional to metered consumption

    VPPA Allocation:
        Distribute contractual certificates proportional to allocation keys

Provenance:
    SHA-256 hash on every AllocationResult guaranteeing audit trail.

Regulatory References:
    - GHG Protocol Corporate Standard (2004, rev 2015) Ch 8 - Allocating Emissions
    - ISO 14064-1:2018 Clause 5.2.3 - Allocation of GHG data
    - EU CSRD / ESRS E1 - Disclosure of allocation methodologies
    - PCAF v3 2024 - Attribution of financed emissions
    - GHG Protocol Scope 2 Guidance (2015) - Contractual instruments

Zero-Hallucination:
    - All allocations use Decimal arithmetic with explicit rounding
    - No LLM involvement in any numeric pathway
    - Provenance hash on every result model

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-049 GHG Multi-Site Management
Engine:  6 of 10
Status:  Production Ready
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")
_TOLERANCE = Decimal("0.0001")
_DP6 = Decimal("0.000001")
_DP10 = Decimal("0.0000000001")

def _safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Divide with zero-guard, returning Decimal('0') when denominator is zero."""
    if denominator == _ZERO:
        return _ZERO
    return (numerator / denominator).quantize(_DP6, rounding=ROUND_HALF_UP)

def _quantise(value: Decimal, precision: Decimal = _DP6) -> Decimal:
    """Quantise a Decimal to the requested precision."""
    return value.quantize(precision, rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AllocationType(str, Enum):
    """Supported emission allocation types."""
    SHARED_SERVICES = "SHARED_SERVICES"
    LANDLORD_TENANT = "LANDLORD_TENANT"
    COGENERATION = "COGENERATION"
    DISTRICT_SYSTEM = "DISTRICT_SYSTEM"
    VPPA = "VPPA"

class AllocationMethod(str, Enum):
    """Supported allocation key methods."""
    FLOOR_AREA = "FLOOR_AREA"
    HEADCOUNT = "HEADCOUNT"
    REVENUE = "REVENUE"
    PRODUCTION_OUTPUT = "PRODUCTION_OUTPUT"
    ENERGY_CONSUMPTION = "ENERGY_CONSUMPTION"
    OPERATING_HOURS = "OPERATING_HOURS"
    METERED_CONSUMPTION = "METERED_CONSUMPTION"
    CONTRACTUAL = "CONTRACTUAL"
    CUSTOM = "CUSTOM"

class CogenerationMethod(str, Enum):
    """Cogeneration emission allocation method per ISO 8302."""
    EFFICIENCY = "EFFICIENCY"
    ENERGY_CONTENT = "ENERGY_CONTENT"

class AllocationStatus(str, Enum):
    """Status of an allocation run."""
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class AllocationConfig(BaseModel):
    """Configuration for a single allocation run."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config_id: str = Field(default_factory=_new_uuid, description="Unique configuration identifier")
    allocation_type: str = Field(
        ..., description="Type of allocation (SHARED_SERVICES, LANDLORD_TENANT, etc.)"
    )
    method: str = Field(
        ..., description="Allocation key method (FLOOR_AREA, HEADCOUNT, etc.)"
    )
    source_site_id: Optional[str] = Field(
        None, description="Source site providing the shared emissions (if applicable)"
    )
    target_site_ids: List[str] = Field(
        default_factory=list, description="Target sites receiving allocated emissions"
    )
    allocation_keys: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Allocation keys per target site (site_id -> key value)",
    )

    @field_validator("allocation_type")
    @classmethod
    def validate_allocation_type(cls, v: str) -> str:
        """Validate allocation type is supported."""
        allowed = {e.value for e in AllocationType}
        if v.upper() not in allowed:
            raise ValueError(f"allocation_type must be one of {sorted(allowed)}, got '{v}'")
        return v.upper()

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate method is supported."""
        allowed = {e.value for e in AllocationMethod}
        if v.upper() not in allowed:
            raise ValueError(f"method must be one of {sorted(allowed)}, got '{v}'")
        return v.upper()

class AllocationResult(BaseModel):
    """Result of an allocation operation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    config_id: str = Field(..., description="Configuration ID that produced this result")
    allocation_type: str = Field(..., description="Type of allocation performed")
    method: str = Field(..., description="Allocation method used")
    allocated_amounts: Dict[str, Decimal] = Field(
        default_factory=dict, description="Allocated emissions per target site (tCO2e)"
    )
    total_allocated: Decimal = Field(
        _ZERO, description="Total emissions allocated across all targets (tCO2e)"
    )
    unallocated_remainder: Decimal = Field(
        _ZERO, description="Emissions not allocated (should be zero or within tolerance)"
    )
    created_at: datetime = Field(default_factory=utcnow, description="Timestamp of allocation")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash after initialisation."""
        if not self.provenance_hash:
            payload = (
                f"{self.result_id}|{self.config_id}|{self.allocation_type}|"
                f"{self.method}|{self.total_allocated}|{self.unallocated_remainder}"
            )
            for site_id in sorted(self.allocated_amounts):
                payload += f"|{site_id}={self.allocated_amounts[site_id]}"
            self.provenance_hash = _compute_hash(payload)

class LandlordTenantSplit(BaseModel):
    """Result of a landlord-tenant emission split."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    site_id: str = Field(..., description="Site / building identifier")
    whole_building_emissions: Decimal = Field(
        ..., description="Whole-building emissions before split (tCO2e)"
    )
    tenant_floor_area: Decimal = Field(
        ..., ge=_ZERO, description="Tenant occupied floor area (m2)"
    )
    total_floor_area: Decimal = Field(
        ..., gt=_ZERO, description="Total building floor area (m2)"
    )
    common_area_pct: Decimal = Field(
        _ZERO, ge=_ZERO, le=_HUNDRED,
        description="Common area percentage (0-100)",
    )
    tenant_emissions: Decimal = Field(
        _ZERO, description="Emissions attributed to tenant (tCO2e)"
    )
    landlord_emissions: Decimal = Field(
        _ZERO, description="Emissions attributed to landlord (tCO2e)"
    )
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash after initialisation."""
        if not self.provenance_hash:
            payload = (
                f"{self.site_id}|{self.whole_building_emissions}|"
                f"{self.tenant_floor_area}|{self.total_floor_area}|"
                f"{self.common_area_pct}|{self.tenant_emissions}|"
                f"{self.landlord_emissions}"
            )
            self.provenance_hash = _compute_hash(payload)

class CogenerationAllocation(BaseModel):
    """Result of a cogeneration / CHP emission allocation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    site_id: str = Field(..., description="Site hosting the CHP plant")
    total_fuel_emissions: Decimal = Field(
        ..., description="Total fuel-input emissions (tCO2e)"
    )
    electricity_output_kwh: Decimal = Field(
        ..., ge=_ZERO, description="Useful electricity output (kWh)"
    )
    heat_output_kwh: Decimal = Field(
        ..., ge=_ZERO, description="Useful heat output (kWh)"
    )
    method: str = Field(
        ..., description="Allocation method (EFFICIENCY or ENERGY_CONTENT)"
    )
    electricity_emissions: Decimal = Field(
        _ZERO, description="Emissions allocated to electricity (tCO2e)"
    )
    heat_emissions: Decimal = Field(
        _ZERO, description="Emissions allocated to heat (tCO2e)"
    )
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash after initialisation."""
        if not self.provenance_hash:
            payload = (
                f"{self.site_id}|{self.total_fuel_emissions}|"
                f"{self.electricity_output_kwh}|{self.heat_output_kwh}|"
                f"{self.method}|{self.electricity_emissions}|{self.heat_emissions}"
            )
            self.provenance_hash = _compute_hash(payload)

class VPPACertificate(BaseModel):
    """A single virtual power purchase agreement certificate."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    certificate_id: str = Field(default_factory=_new_uuid, description="Certificate identifier")
    volume_mwh: Decimal = Field(..., gt=_ZERO, description="Volume in MWh")
    emission_factor: Decimal = Field(
        _ZERO, ge=_ZERO, description="Emission factor (tCO2e/MWh)"
    )
    source_description: str = Field("", description="Source of renewable energy")
    vintage_year: int = Field(2026, ge=2015, le=2040, description="Certificate vintage year")

class DistrictConsumption(BaseModel):
    """Metered consumption for a site connected to a district system."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    site_id: str = Field(..., description="Connected site identifier")
    consumption_kwh: Decimal = Field(..., ge=_ZERO, description="Metered consumption (kWh)")
    period: str = Field("", description="Reporting period identifier")

class AllocationSummary(BaseModel):
    """Summary of a batch of allocation results."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    summary_id: str = Field(default_factory=_new_uuid, description="Summary identifier")
    total_source_emissions: Decimal = Field(_ZERO, description="Total source emissions (tCO2e)")
    total_allocated: Decimal = Field(_ZERO, description="Total allocated (tCO2e)")
    total_unallocated: Decimal = Field(_ZERO, description="Total unallocated remainder (tCO2e)")
    allocation_count: int = Field(0, description="Number of allocation operations")
    completeness_pct: Decimal = Field(_ZERO, description="Allocation completeness (%)")
    by_type: Dict[str, Decimal] = Field(default_factory=dict, description="Allocated by type")
    by_site: Dict[str, Decimal] = Field(default_factory=dict, description="Allocated by site")
    status: str = Field("COMPLETE", description="Overall status")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

class CompletenessCheck(BaseModel):
    """Result of allocation completeness verification."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_total: Decimal = Field(..., description="Source total emissions (tCO2e)")
    allocated_total: Decimal = Field(..., description="Allocated total emissions (tCO2e)")
    difference: Decimal = Field(_ZERO, description="Difference (tCO2e)")
    difference_pct: Decimal = Field(_ZERO, description="Difference percentage")
    within_tolerance: bool = Field(True, description="Whether difference is within tolerance")
    tolerance_used: Decimal = Field(_TOLERANCE, description="Tolerance value used")
    status: str = Field("PASS", description="PASS or FAIL")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SiteAllocationEngine:
    """
    Allocates shared emissions across sites in a multi-site portfolio.

    Supports five allocation types:
        1. Shared Services -- proportional key allocation
        2. Landlord-Tenant -- floor-area-based split with common areas
        3. Cogeneration -- CHP efficiency / energy-content allocation
        4. District System -- metered-consumption proportional split
        5. VPPA -- contractual certificate allocation

    All arithmetic uses Decimal to eliminate floating-point drift.
    Every result carries a SHA-256 provenance hash.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        tolerance: Decimal = _TOLERANCE,
        rounding_precision: Decimal = _DP6,
    ) -> None:
        """
        Initialise the SiteAllocationEngine.

        Args:
            tolerance: Acceptable rounding tolerance for completeness checks.
            rounding_precision: Decimal precision for quantisation.
        """
        self._tolerance = tolerance
        self._precision = rounding_precision
        logger.info(
            "SiteAllocationEngine v%s initialised (tolerance=%s, precision=%s)",
            _MODULE_VERSION, tolerance, rounding_precision,
        )

    # ----------------------------------------------- shared services
    def allocate_shared_services(
        self,
        config: AllocationConfig,
        source_emissions: Decimal,
    ) -> AllocationResult:
        """
        Distribute shared-service emissions proportionally using allocation keys.

        For each target site *i*:
            allocated_i = source_emissions * (key_i / total_keys)

        Any rounding remainder is assigned to the last target site to
        maintain a zero-remainder balance.

        Args:
            config: Allocation configuration including target sites and keys.
            source_emissions: Total source emissions to allocate (tCO2e).

        Returns:
            AllocationResult with per-site amounts and provenance hash.

        Raises:
            ValueError: If keys are empty or contain negative values.
        """
        logger.info(
            "Allocating shared services: config=%s source=%s",
            config.config_id, source_emissions,
        )
        if not config.allocation_keys:
            raise ValueError("allocation_keys must not be empty for shared-service allocation")
        if source_emissions < _ZERO:
            raise ValueError("source_emissions must be non-negative")

        # Validate keys
        for site_id, key_val in config.allocation_keys.items():
            if key_val < _ZERO:
                raise ValueError(
                    f"Allocation key for site '{site_id}' is negative ({key_val})"
                )

        total_keys = sum(config.allocation_keys.values())
        if total_keys == _ZERO:
            raise ValueError("Sum of allocation keys is zero; cannot allocate")

        allocated: Dict[str, Decimal] = {}
        running_total = _ZERO
        sorted_sites = sorted(config.allocation_keys.keys())

        for idx, site_id in enumerate(sorted_sites):
            key_val = config.allocation_keys[site_id]
            proportion = _safe_divide(key_val, total_keys)
            amount = _quantise(source_emissions * proportion, self._precision)

            if idx == len(sorted_sites) - 1:
                # Assign remainder to last site to ensure completeness
                amount = source_emissions - running_total
                amount = _quantise(amount, self._precision)

            allocated[site_id] = amount
            running_total += amount

        total_allocated = sum(allocated.values())
        remainder = source_emissions - total_allocated

        result = AllocationResult(
            config_id=config.config_id,
            allocation_type=AllocationType.SHARED_SERVICES.value,
            method=config.method,
            allocated_amounts=allocated,
            total_allocated=total_allocated,
            unallocated_remainder=_quantise(remainder, self._precision),
        )

        logger.info(
            "Shared services allocated: total=%s sites=%d remainder=%s",
            total_allocated, len(allocated), result.unallocated_remainder,
        )
        return result

    # ----------------------------------------------- landlord-tenant
    def calculate_landlord_tenant_split(
        self,
        site_id: str,
        whole_building_emissions: Decimal,
        tenant_floor_area: Decimal,
        total_floor_area: Decimal,
        common_area_pct: Decimal = _ZERO,
    ) -> LandlordTenantSplit:
        """
        Split whole-building emissions between landlord and tenant.

        Formula:
            common_area_m2 = total_floor_area * common_area_pct / 100
            private_area   = total_floor_area - common_area_m2
            tenant_share   = tenant_floor_area + (common_area_m2 * tenant_floor_area / private_area)
            tenant_emissions  = whole_building * (tenant_share / total_floor_area)
            landlord_emissions = whole_building - tenant_emissions

        When common_area_pct is zero the split degenerates to a simple
        floor-area ratio.

        Args:
            site_id: Building / site identifier.
            whole_building_emissions: Total whole-building emissions (tCO2e).
            tenant_floor_area: Tenant occupied area (m2).
            total_floor_area: Total building gross internal area (m2).
            common_area_pct: Common area as a percentage of total (0-100).

        Returns:
            LandlordTenantSplit with computed emissions and provenance hash.

        Raises:
            ValueError: If areas are inconsistent or common_area_pct out of range.
        """
        logger.info(
            "Landlord-tenant split: site=%s building=%s tenant_area=%s total=%s common=%s%%",
            site_id, whole_building_emissions, tenant_floor_area, total_floor_area, common_area_pct,
        )
        if total_floor_area <= _ZERO:
            raise ValueError("total_floor_area must be positive")
        if tenant_floor_area < _ZERO:
            raise ValueError("tenant_floor_area must be non-negative")
        if tenant_floor_area > total_floor_area:
            raise ValueError("tenant_floor_area cannot exceed total_floor_area")
        if common_area_pct < _ZERO or common_area_pct > _HUNDRED:
            raise ValueError("common_area_pct must be between 0 and 100")
        if whole_building_emissions < _ZERO:
            raise ValueError("whole_building_emissions must be non-negative")

        common_area_m2 = _quantise(
            total_floor_area * common_area_pct / _HUNDRED, self._precision
        )
        private_area = total_floor_area - common_area_m2

        if private_area <= _ZERO:
            # Edge case: entire building is common area
            # Allocate proportionally to tenant_area / total_area
            tenant_share = tenant_floor_area
        else:
            # Tenant gets their floor area plus a share of common area
            common_area_share = _safe_divide(
                common_area_m2 * tenant_floor_area, private_area
            )
            tenant_share = tenant_floor_area + common_area_share

        tenant_emissions = _quantise(
            whole_building_emissions * _safe_divide(tenant_share, total_floor_area),
            self._precision,
        )
        landlord_emissions = _quantise(
            whole_building_emissions - tenant_emissions, self._precision
        )

        result = LandlordTenantSplit(
            site_id=site_id,
            whole_building_emissions=whole_building_emissions,
            tenant_floor_area=tenant_floor_area,
            total_floor_area=total_floor_area,
            common_area_pct=common_area_pct,
            tenant_emissions=tenant_emissions,
            landlord_emissions=landlord_emissions,
        )

        logger.info(
            "Landlord-tenant result: tenant=%s landlord=%s hash=%s",
            tenant_emissions, landlord_emissions, result.provenance_hash[:12],
        )
        return result

    # ----------------------------------------------- cogeneration
    def allocate_cogeneration(
        self,
        site_id: str,
        total_fuel_emissions: Decimal,
        electricity_output_kwh: Decimal,
        heat_output_kwh: Decimal,
        method: str = CogenerationMethod.EFFICIENCY.value,
    ) -> CogenerationAllocation:
        """
        Allocate cogeneration (CHP) emissions between electricity and heat.

        Efficiency Method (default):
            proportion_elec = elec_output / (elec_output + heat_output)
            proportion_heat = heat_output / (elec_output + heat_output)
            elec_emissions  = fuel_emissions * proportion_elec
            heat_emissions  = fuel_emissions * proportion_heat

        Energy Content Method:
            Same formula but the outputs are expected to be expressed in
            primary-energy-equivalent terms by the caller.

        Both methods ensure:
            elec_emissions + heat_emissions == total_fuel_emissions

        Args:
            site_id: Site hosting the CHP plant.
            total_fuel_emissions: Total fuel-input emissions (tCO2e).
            electricity_output_kwh: Useful electricity output (kWh).
            heat_output_kwh: Useful heat output (kWh).
            method: EFFICIENCY or ENERGY_CONTENT.

        Returns:
            CogenerationAllocation with split emissions and provenance hash.

        Raises:
            ValueError: If outputs are both zero or method is unsupported.
        """
        logger.info(
            "Cogeneration allocation: site=%s fuel=%s elec=%s heat=%s method=%s",
            site_id, total_fuel_emissions, electricity_output_kwh, heat_output_kwh, method,
        )
        method_upper = method.upper()
        valid_methods = {e.value for e in CogenerationMethod}
        if method_upper not in valid_methods:
            raise ValueError(f"method must be one of {sorted(valid_methods)}, got '{method}'")
        if total_fuel_emissions < _ZERO:
            raise ValueError("total_fuel_emissions must be non-negative")
        if electricity_output_kwh < _ZERO:
            raise ValueError("electricity_output_kwh must be non-negative")
        if heat_output_kwh < _ZERO:
            raise ValueError("heat_output_kwh must be non-negative")

        total_output = electricity_output_kwh + heat_output_kwh
        if total_output == _ZERO:
            raise ValueError(
                "Cannot allocate cogeneration emissions when both electricity and heat output are zero"
            )

        proportion_elec = _safe_divide(electricity_output_kwh, total_output)
        elec_emissions = _quantise(
            total_fuel_emissions * proportion_elec, self._precision
        )
        heat_emissions = _quantise(
            total_fuel_emissions - elec_emissions, self._precision
        )

        result = CogenerationAllocation(
            site_id=site_id,
            total_fuel_emissions=total_fuel_emissions,
            electricity_output_kwh=electricity_output_kwh,
            heat_output_kwh=heat_output_kwh,
            method=method_upper,
            electricity_emissions=elec_emissions,
            heat_emissions=heat_emissions,
        )

        logger.info(
            "Cogeneration result: elec=%s heat=%s hash=%s",
            elec_emissions, heat_emissions, result.provenance_hash[:12],
        )
        return result

    # ----------------------------------------------- district system
    def allocate_district_system(
        self,
        system_emissions: Decimal,
        connected_sites: List[str],
        consumption_data: List[DistrictConsumption],
    ) -> List[AllocationResult]:
        """
        Allocate district heating/cooling system emissions to connected sites.

        Each site's share is proportional to its metered consumption:
            share_i = system_emissions * (consumption_i / total_consumption)

        Sites listed in connected_sites but absent from consumption_data
        receive zero allocation with a logged warning.

        Args:
            system_emissions: Total district system emissions (tCO2e).
            connected_sites: List of site IDs connected to the district system.
            consumption_data: Metered consumption records per site.

        Returns:
            List of AllocationResult, one per connected site.

        Raises:
            ValueError: If system_emissions is negative or no consumption data.
        """
        logger.info(
            "District system allocation: emissions=%s sites=%d consumption_records=%d",
            system_emissions, len(connected_sites), len(consumption_data),
        )
        if system_emissions < _ZERO:
            raise ValueError("system_emissions must be non-negative")
        if not connected_sites:
            raise ValueError("connected_sites must not be empty")

        # Build consumption lookup
        consumption_map: Dict[str, Decimal] = {}
        for record in consumption_data:
            if record.site_id in consumption_map:
                consumption_map[record.site_id] += record.consumption_kwh
            else:
                consumption_map[record.site_id] = record.consumption_kwh

        total_consumption = sum(consumption_map.values())
        if total_consumption == _ZERO:
            logger.warning("Total district consumption is zero; all sites get zero allocation")

        results: List[AllocationResult] = []
        running_total = _ZERO
        sorted_sites = sorted(connected_sites)

        for idx, site_id in enumerate(sorted_sites):
            site_consumption = consumption_map.get(site_id, _ZERO)
            if site_id not in consumption_map:
                logger.warning(
                    "Site %s listed as connected but has no consumption data", site_id
                )

            if total_consumption == _ZERO:
                amount = _ZERO
            elif idx == len(sorted_sites) - 1:
                amount = _quantise(system_emissions - running_total, self._precision)
            else:
                proportion = _safe_divide(site_consumption, total_consumption)
                amount = _quantise(system_emissions * proportion, self._precision)

            allocated_amounts = {site_id: amount}
            running_total += amount

            config_id = _new_uuid()
            result = AllocationResult(
                config_id=config_id,
                allocation_type=AllocationType.DISTRICT_SYSTEM.value,
                method=AllocationMethod.METERED_CONSUMPTION.value,
                allocated_amounts=allocated_amounts,
                total_allocated=amount,
                unallocated_remainder=_ZERO,
            )
            results.append(result)

        logger.info(
            "District system allocated: total=%s across %d sites",
            running_total, len(results),
        )
        return results

    # ----------------------------------------------- VPPA
    def allocate_vppa(
        self,
        vppa_certificates: List[VPPACertificate],
        beneficiary_sites: List[str],
        allocation_keys: Dict[str, Decimal],
    ) -> List[AllocationResult]:
        """
        Allocate virtual power purchase agreement certificates to beneficiary sites.

        Total VPPA volume (MWh) is distributed across beneficiary sites
        proportional to their allocation keys.  The emission benefit is:
            benefit_i = total_volume * (key_i / total_keys) * weighted_ef

        Where weighted_ef is the volume-weighted average emission factor
        across all certificates.

        Args:
            vppa_certificates: List of VPPA certificates.
            beneficiary_sites: List of beneficiary site IDs.
            allocation_keys: Per-site allocation keys (site_id -> value).

        Returns:
            List of AllocationResult, one per beneficiary site.

        Raises:
            ValueError: If inputs are empty or keys are inconsistent.
        """
        logger.info(
            "VPPA allocation: certificates=%d sites=%d",
            len(vppa_certificates), len(beneficiary_sites),
        )
        if not vppa_certificates:
            raise ValueError("vppa_certificates must not be empty")
        if not beneficiary_sites:
            raise ValueError("beneficiary_sites must not be empty")
        if not allocation_keys:
            raise ValueError("allocation_keys must not be empty")

        # Calculate total volume and weighted emission factor
        total_volume_mwh = sum(c.volume_mwh for c in vppa_certificates)
        weighted_ef_sum = sum(
            c.volume_mwh * c.emission_factor for c in vppa_certificates
        )
        weighted_ef = _safe_divide(weighted_ef_sum, total_volume_mwh)

        total_emissions = _quantise(total_volume_mwh * weighted_ef, self._precision)
        total_keys = sum(allocation_keys.get(s, _ZERO) for s in beneficiary_sites)

        if total_keys == _ZERO:
            raise ValueError("Sum of allocation keys for beneficiary sites is zero")

        results: List[AllocationResult] = []
        running_total = _ZERO
        sorted_sites = sorted(beneficiary_sites)

        for idx, site_id in enumerate(sorted_sites):
            key_val = allocation_keys.get(site_id, _ZERO)
            if idx == len(sorted_sites) - 1:
                amount = _quantise(total_emissions - running_total, self._precision)
            else:
                proportion = _safe_divide(key_val, total_keys)
                amount = _quantise(total_emissions * proportion, self._precision)

            running_total += amount
            allocated_amounts = {site_id: amount}

            config_id = _new_uuid()
            result = AllocationResult(
                config_id=config_id,
                allocation_type=AllocationType.VPPA.value,
                method=AllocationMethod.CONTRACTUAL.value,
                allocated_amounts=allocated_amounts,
                total_allocated=amount,
                unallocated_remainder=_ZERO,
            )
            results.append(result)

        logger.info(
            "VPPA allocated: total=%s weighted_ef=%s across %d sites",
            running_total, weighted_ef, len(results),
        )
        return results

    # ----------------------------------------------- summary
    def get_allocation_summary(
        self,
        results: List[AllocationResult],
    ) -> AllocationSummary:
        """
        Produce a summary across a list of allocation results.

        Aggregates total allocated, remainder, counts by type, and per-site totals.

        Args:
            results: List of AllocationResult from various allocation operations.

        Returns:
            AllocationSummary with aggregated totals and provenance hash.
        """
        logger.info("Building allocation summary for %d results", len(results))

        total_allocated = _ZERO
        total_unallocated = _ZERO
        by_type: Dict[str, Decimal] = {}
        by_site: Dict[str, Decimal] = {}

        for r in results:
            total_allocated += r.total_allocated
            total_unallocated += r.unallocated_remainder

            alloc_type = r.allocation_type
            by_type[alloc_type] = by_type.get(alloc_type, _ZERO) + r.total_allocated

            for site_id, amount in r.allocated_amounts.items():
                by_site[site_id] = by_site.get(site_id, _ZERO) + amount

        source_total = total_allocated + total_unallocated
        completeness = (
            _quantise(_safe_divide(total_allocated, source_total) * _HUNDRED, self._precision)
            if source_total > _ZERO
            else _HUNDRED
        )

        status = (
            AllocationStatus.COMPLETE.value
            if total_unallocated <= self._tolerance
            else AllocationStatus.PARTIAL.value
        )

        # Build provenance hash
        hash_payload = (
            f"summary|{total_allocated}|{total_unallocated}|{len(results)}"
        )
        for t in sorted(by_type):
            hash_payload += f"|{t}={by_type[t]}"
        provenance = _compute_hash(hash_payload)

        summary = AllocationSummary(
            total_source_emissions=source_total,
            total_allocated=total_allocated,
            total_unallocated=total_unallocated,
            allocation_count=len(results),
            completeness_pct=completeness,
            by_type=by_type,
            by_site=by_site,
            status=status,
            provenance_hash=provenance,
        )

        logger.info(
            "Allocation summary: total=%s unalloc=%s completeness=%s%% status=%s",
            total_allocated, total_unallocated, completeness, status,
        )
        return summary

    # ----------------------------------------------- completeness check
    def verify_allocation_completeness(
        self,
        source_total: Decimal,
        results: List[AllocationResult],
    ) -> CompletenessCheck:
        """
        Verify that the sum of allocated amounts matches the source total.

        Args:
            source_total: Expected total emissions that should be allocated (tCO2e).
            results: List of AllocationResult to check.

        Returns:
            CompletenessCheck with PASS/FAIL status and difference details.
        """
        logger.info(
            "Verifying allocation completeness: source=%s results=%d",
            source_total, len(results),
        )
        allocated_total = sum(r.total_allocated for r in results)
        difference = _quantise(source_total - allocated_total, self._precision)
        difference_abs = abs(difference)

        if source_total > _ZERO:
            difference_pct = _quantise(
                difference_abs / source_total * _HUNDRED, self._precision
            )
        else:
            difference_pct = _ZERO

        within_tolerance = difference_abs <= self._tolerance
        status = "PASS" if within_tolerance else "FAIL"

        check = CompletenessCheck(
            source_total=source_total,
            allocated_total=allocated_total,
            difference=difference,
            difference_pct=difference_pct,
            within_tolerance=within_tolerance,
            tolerance_used=self._tolerance,
            status=status,
        )

        if not within_tolerance:
            logger.warning(
                "Allocation completeness FAIL: source=%s allocated=%s diff=%s (%s%%)",
                source_total, allocated_total, difference, difference_pct,
            )
        else:
            logger.info(
                "Allocation completeness PASS: diff=%s (within tolerance %s)",
                difference, self._tolerance,
            )
        return check

    # ----------------------------------------------- batch allocation
    def allocate_batch(
        self,
        configs: List[AllocationConfig],
        source_emissions_map: Dict[str, Decimal],
    ) -> List[AllocationResult]:
        """
        Run shared-service allocation for a batch of configurations.

        Args:
            configs: List of AllocationConfig objects.
            source_emissions_map: Mapping of config_id -> source emissions.

        Returns:
            List of AllocationResult for all configurations.

        Raises:
            ValueError: If a config has no matching source emissions.
        """
        logger.info("Batch allocation: %d configs", len(configs))
        results: List[AllocationResult] = []

        for config in configs:
            source = source_emissions_map.get(config.config_id)
            if source is None:
                raise ValueError(
                    f"No source emissions for config_id={config.config_id}"
                )
            result = self.allocate_shared_services(config, source)
            results.append(result)

        logger.info("Batch allocation complete: %d results", len(results))
        return results

    # ----------------------------------------------- multi-tenant building
    def allocate_multi_tenant_building(
        self,
        site_id: str,
        whole_building_emissions: Decimal,
        tenants: Dict[str, Decimal],
        total_floor_area: Decimal,
        common_area_pct: Decimal = _ZERO,
    ) -> List[LandlordTenantSplit]:
        """
        Allocate a multi-tenant building's emissions across all tenants.

        Each tenant receives a proportional share including their portion
        of common areas.  The landlord retains the remainder.

        Args:
            site_id: Building identifier.
            whole_building_emissions: Total building emissions (tCO2e).
            tenants: Mapping of tenant_id -> occupied floor area (m2).
            total_floor_area: Total building gross internal area (m2).
            common_area_pct: Common area as a percentage (0-100).

        Returns:
            List of LandlordTenantSplit, one per tenant plus one for landlord remainder.
        """
        logger.info(
            "Multi-tenant allocation: site=%s tenants=%d emissions=%s",
            site_id, len(tenants), whole_building_emissions,
        )
        if not tenants:
            raise ValueError("tenants dict must not be empty")

        total_tenant_area = sum(tenants.values())
        if total_tenant_area > total_floor_area:
            raise ValueError(
                f"Total tenant area ({total_tenant_area}) exceeds "
                f"total floor area ({total_floor_area})"
            )

        splits: List[LandlordTenantSplit] = []
        cumulative_tenant_emissions = _ZERO

        sorted_tenants = sorted(tenants.keys())
        for tenant_id in sorted_tenants:
            tenant_area = tenants[tenant_id]
            split = self.calculate_landlord_tenant_split(
                site_id=f"{site_id}__tenant_{tenant_id}",
                whole_building_emissions=whole_building_emissions,
                tenant_floor_area=tenant_area,
                total_floor_area=total_floor_area,
                common_area_pct=common_area_pct,
            )
            splits.append(split)
            cumulative_tenant_emissions += split.tenant_emissions

        # Landlord remainder
        landlord_remainder = _quantise(
            whole_building_emissions - cumulative_tenant_emissions, self._precision
        )
        landlord_split = LandlordTenantSplit(
            site_id=f"{site_id}__landlord",
            whole_building_emissions=whole_building_emissions,
            tenant_floor_area=_ZERO,
            total_floor_area=total_floor_area,
            common_area_pct=common_area_pct,
            tenant_emissions=_ZERO,
            landlord_emissions=landlord_remainder,
        )
        splits.append(landlord_split)

        logger.info(
            "Multi-tenant result: %d tenants, landlord_remainder=%s",
            len(tenants), landlord_remainder,
        )
        return splits

    # ----------------------------------------------- rebalance keys
    def rebalance_allocation_keys(
        self,
        allocation_keys: Dict[str, Decimal],
    ) -> Dict[str, Decimal]:
        """
        Normalise allocation keys so they sum to exactly Decimal('1').

        Args:
            allocation_keys: Raw allocation keys per site.

        Returns:
            Normalised keys summing to Decimal('1').

        Raises:
            ValueError: If all keys are zero.
        """
        total = sum(allocation_keys.values())
        if total == _ZERO:
            raise ValueError("Cannot rebalance when all keys are zero")

        normalised: Dict[str, Decimal] = {}
        running_sum = _ZERO
        sorted_keys = sorted(allocation_keys.keys())

        for idx, site_id in enumerate(sorted_keys):
            if idx == len(sorted_keys) - 1:
                normalised[site_id] = _quantise(_ONE - running_sum, self._precision)
            else:
                val = _quantise(allocation_keys[site_id] / total, self._precision)
                normalised[site_id] = val
                running_sum += val

        return normalised

    # ----------------------------------------------- variance check
    def check_key_variance(
        self,
        current_keys: Dict[str, Decimal],
        prior_keys: Dict[str, Decimal],
        threshold_pct: Decimal = Decimal("10"),
    ) -> List[Dict[str, Any]]:
        """
        Check allocation key variance between current and prior period.

        Returns a list of sites where the key changed by more than
        threshold_pct percent, which may indicate boundary changes or
        data quality issues.

        Args:
            current_keys: Current period allocation keys.
            prior_keys: Prior period allocation keys.
            threshold_pct: Percentage change threshold for flagging.

        Returns:
            List of dicts with site_id, current, prior, change_pct, flagged fields.
        """
        logger.info(
            "Checking key variance: %d current vs %d prior, threshold=%s%%",
            len(current_keys), len(prior_keys), threshold_pct,
        )
        all_sites = sorted(set(current_keys.keys()) | set(prior_keys.keys()))
        variances: List[Dict[str, Any]] = []

        for site_id in all_sites:
            current_val = current_keys.get(site_id, _ZERO)
            prior_val = prior_keys.get(site_id, _ZERO)

            if prior_val == _ZERO:
                change_pct = Decimal("100") if current_val > _ZERO else _ZERO
            else:
                change_pct = _quantise(
                    abs(current_val - prior_val) / prior_val * _HUNDRED,
                    self._precision,
                )

            flagged = change_pct > threshold_pct
            variances.append({
                "site_id": site_id,
                "current_key": current_val,
                "prior_key": prior_val,
                "change_pct": change_pct,
                "flagged": flagged,
            })

            if flagged:
                logger.warning(
                    "Key variance flagged: site=%s change=%s%% (threshold=%s%%)",
                    site_id, change_pct, threshold_pct,
                )

        return variances

    # ----------------------------------------------- helper: build config
    @staticmethod
    def build_config(
        allocation_type: str,
        method: str,
        target_site_ids: List[str],
        allocation_keys: Dict[str, Decimal],
        source_site_id: Optional[str] = None,
    ) -> AllocationConfig:
        """
        Factory helper to build an AllocationConfig.

        Args:
            allocation_type: Type of allocation.
            method: Allocation method.
            target_site_ids: Target sites.
            allocation_keys: Keys per site.
            source_site_id: Optional source site.

        Returns:
            Configured AllocationConfig instance.
        """
        return AllocationConfig(
            allocation_type=allocation_type,
            method=method,
            source_site_id=source_site_id,
            target_site_ids=target_site_ids,
            allocation_keys=allocation_keys,
        )

# ---------------------------------------------------------------------------
# Pydantic v2 model rebuild (required with `from __future__ import annotations`)
# ---------------------------------------------------------------------------

AllocationConfig.model_rebuild()
AllocationResult.model_rebuild()
LandlordTenantSplit.model_rebuild()
CogenerationAllocation.model_rebuild()
VPPACertificate.model_rebuild()
DistrictConsumption.model_rebuild()
AllocationSummary.model_rebuild()
CompletenessCheck.model_rebuild()
