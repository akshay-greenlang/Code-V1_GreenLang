# -*- coding: utf-8 -*-
"""
Allocation Workflow
====================================

4-phase workflow for shared service and common-area GHG emission allocation
covering shared service identification, allocation method selection,
calculation, and verification within PACK-049 Multi-Site Management.

Phases:
    1. SharedServiceId          -- Identify shared services, landlord-tenant
                                   arrangements, cogeneration, and common
                                   infrastructure across the site portfolio.
    2. AllocationMethodSelect   -- Select allocation method per service type
                                   (floor area, headcount, revenue, energy
                                   metered, production output, custom).
    3. Calculate                -- Execute allocation calculations using
                                   Decimal arithmetic with ROUND_HALF_UP.
    4. Verify                   -- Verify completeness (allocations sum to
                                   100%), cross-check, generate provenance.

Regulatory Basis:
    GHG Protocol Corporate Standard (Ch. 8) -- Allocation
    ISO 14064-1:2018 (Cl. 5.2.4) -- Allocation methods
    CSRD / ESRS E1 -- Shared emissions reporting

Author: GreenLang Team
Version: 49.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AllocationPhase(str, Enum):
    SHARED_SERVICE_ID = "shared_service_id"
    ALLOCATION_METHOD_SELECT = "allocation_method_select"
    CALCULATE = "calculate"
    VERIFY = "verify"


class SharedServiceType(str, Enum):
    LANDLORD_TENANT = "landlord_tenant"
    COGENERATION = "cogeneration"
    SHARED_FLEET = "shared_fleet"
    CENTRAL_HEATING_COOLING = "central_heating_cooling"
    COMMON_AREA = "common_area"
    SHARED_DATA_CENTER = "shared_data_center"
    CORPORATE_OVERHEAD = "corporate_overhead"
    OTHER = "other"


class AllocationMethod(str, Enum):
    FLOOR_AREA = "floor_area"
    HEADCOUNT = "headcount"
    REVENUE = "revenue"
    ENERGY_METERED = "energy_metered"
    PRODUCTION_OUTPUT = "production_output"
    EQUAL_SPLIT = "equal_split"
    CUSTOM = "custom"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


# =============================================================================
# REFERENCE DATA
# =============================================================================

DEFAULT_METHOD_MAP: Dict[str, str] = {
    "landlord_tenant": "floor_area",
    "cogeneration": "energy_metered",
    "shared_fleet": "headcount",
    "central_heating_cooling": "floor_area",
    "common_area": "floor_area",
    "shared_data_center": "energy_metered",
    "corporate_overhead": "revenue",
    "other": "equal_split",
}

ALLOCATION_TOLERANCE_PCT = Decimal("0.01")  # 0.01% tolerance for 100% check


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    phase_name: str = Field(...)
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class SharedService(BaseModel):
    """A shared service or common facility to be allocated."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    service_id: str = Field(default_factory=_new_uuid)
    service_name: str = Field(...)
    service_type: SharedServiceType = Field(SharedServiceType.OTHER)
    total_emissions_tco2e: Decimal = Field(Decimal("0"))
    source_site_id: str = Field("", description="Site where emissions originate")
    benefiting_site_ids: List[str] = Field(default_factory=list)
    description: str = Field("")
    scope: str = Field("scope_2")


class SiteAllocationDriver(BaseModel):
    """Allocation driver values for a site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field(...)
    site_name: str = Field("")
    floor_area_sqm: Decimal = Field(Decimal("0"))
    headcount: Decimal = Field(Decimal("0"))
    revenue: Decimal = Field(Decimal("0"))
    energy_kwh: Decimal = Field(Decimal("0"))
    production_output: Decimal = Field(Decimal("0"))
    custom_driver: Decimal = Field(Decimal("0"))


class AllocationLineItem(BaseModel):
    """Result of allocating a shared service to one site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    allocation_id: str = Field(default_factory=_new_uuid)
    service_id: str = Field(...)
    service_name: str = Field("")
    site_id: str = Field(...)
    site_name: str = Field("")
    method: AllocationMethod = Field(AllocationMethod.FLOOR_AREA)
    driver_value: Decimal = Field(Decimal("0"))
    driver_total: Decimal = Field(Decimal("0"))
    allocation_pct: Decimal = Field(Decimal("0"))
    allocated_tco2e: Decimal = Field(Decimal("0"))
    provenance_hash: str = Field("")


class VerificationCheck(BaseModel):
    """Single verification check result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    check_name: str = Field(...)
    status: VerificationStatus = Field(VerificationStatus.PASSED)
    message: str = Field("")
    expected_value: str = Field("")
    actual_value: str = Field("")


class AllocationInput(BaseModel):
    """Input for the allocation workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    organisation_id: str = Field(...)
    reporting_year: int = Field(...)
    shared_services: List[Dict[str, Any]] = Field(default_factory=list)
    site_drivers: List[Dict[str, Any]] = Field(default_factory=list)
    method_overrides: Dict[str, str] = Field(
        default_factory=dict,
        description="service_id -> method override"
    )
    skip_phases: List[str] = Field(default_factory=list)


class AllocationResult(BaseModel):
    """Output from the allocation workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    workflow_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    allocation_items: List[AllocationLineItem] = Field(default_factory=list)
    verification_checks: List[VerificationCheck] = Field(default_factory=list)
    total_allocated_tco2e: Decimal = Field(Decimal("0"))
    services_count: int = Field(0)
    sites_count: int = Field(0)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(0.0)
    provenance_hash: str = Field("")
    started_at: str = Field("")
    completed_at: str = Field("")


# =============================================================================
# WORKFLOW CLASS
# =============================================================================


class AllocationWorkflow:
    """
    4-phase allocation workflow for shared GHG emissions.

    Identifies shared services, selects allocation methods, computes
    allocations using Decimal arithmetic, and verifies completeness.

    Example:
        >>> wf = AllocationWorkflow()
        >>> inp = AllocationInput(
        ...     organisation_id="ORG-001", reporting_year=2025,
        ...     shared_services=[{
        ...         "service_name": "Central Boiler",
        ...         "service_type": "central_heating_cooling",
        ...         "total_emissions_tco2e": "500",
        ...         "benefiting_site_ids": ["S1", "S2"],
        ...     }],
        ...     site_drivers=[
        ...         {"site_id": "S1", "floor_area_sqm": "5000"},
        ...         {"site_id": "S2", "floor_area_sqm": "3000"},
        ...     ],
        ... )
        >>> result = wf.execute(inp)
    """

    PHASE_ORDER: List[AllocationPhase] = [
        AllocationPhase.SHARED_SERVICE_ID,
        AllocationPhase.ALLOCATION_METHOD_SELECT,
        AllocationPhase.CALCULATE,
        AllocationPhase.VERIFY,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._services: List[SharedService] = []
        self._drivers: Dict[str, SiteAllocationDriver] = {}
        self._methods: Dict[str, AllocationMethod] = {}
        self._items: List[AllocationLineItem] = []

    def execute(self, input_data: AllocationInput) -> AllocationResult:
        """Execute the allocation workflow."""
        start = _utcnow()
        result = AllocationResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            status=WorkflowStatus.RUNNING,
            started_at=start.isoformat(),
        )

        phase_methods = {
            AllocationPhase.SHARED_SERVICE_ID: self._phase_shared_service_id,
            AllocationPhase.ALLOCATION_METHOD_SELECT: self._phase_method_select,
            AllocationPhase.CALCULATE: self._phase_calculate,
            AllocationPhase.VERIFY: self._phase_verify,
        }

        for idx, phase in enumerate(self.PHASE_ORDER, 1):
            if phase.value in input_data.skip_phases:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx, status=PhaseStatus.SKIPPED,
                ))
                continue
            phase_start = _utcnow()
            try:
                phase_out = phase_methods[phase](input_data, result)
                elapsed = (_utcnow() - phase_start).total_seconds()
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
                    outputs=phase_out, provenance_hash=_compute_hash(str(phase_out)),
                ))
            except Exception as exc:
                elapsed = (_utcnow() - phase_start).total_seconds()
                logger.error("Phase %s failed: %s", phase.value, exc, exc_info=True)
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.FAILED, duration_seconds=elapsed, errors=[str(exc)],
                ))
                result.status = WorkflowStatus.FAILED
                result.errors.append(f"Phase {phase.value}: {exc}")
                break

        if result.status != WorkflowStatus.FAILED:
            result.status = WorkflowStatus.COMPLETED

        end = _utcnow()
        result.completed_at = end.isoformat()
        result.duration_seconds = (end - start).total_seconds()
        result.provenance_hash = _compute_hash(
            f"{result.workflow_id}|{result.organisation_id}|"
            f"{float(result.total_allocated_tco2e)}|{result.completed_at}"
        )
        return result

    # -----------------------------------------------------------------
    # PHASE 1 -- SHARED SERVICE IDENTIFICATION
    # -----------------------------------------------------------------

    def _phase_shared_service_id(
        self, input_data: AllocationInput, result: AllocationResult,
    ) -> Dict[str, Any]:
        """Identify shared services requiring allocation."""
        logger.info("Phase 1 -- Shared Service ID: %d services", len(input_data.shared_services))
        services: List[SharedService] = []

        for raw in input_data.shared_services:
            try:
                stype = SharedServiceType(raw.get("service_type", "other"))
            except ValueError:
                stype = SharedServiceType.OTHER

            svc = SharedService(
                service_id=raw.get("service_id", _new_uuid()),
                service_name=raw.get("service_name", "Unknown"),
                service_type=stype,
                total_emissions_tco2e=self._dec(raw.get("total_emissions_tco2e", "0")),
                source_site_id=raw.get("source_site_id", ""),
                benefiting_site_ids=raw.get("benefiting_site_ids", []),
                description=raw.get("description", ""),
                scope=raw.get("scope", "scope_2"),
            )
            services.append(svc)

        self._services = services
        result.services_count = len(services)

        # Parse site drivers
        drivers: Dict[str, SiteAllocationDriver] = {}
        for raw in input_data.site_drivers:
            sid = raw.get("site_id", "")
            if not sid:
                continue
            drivers[sid] = SiteAllocationDriver(
                site_id=sid,
                site_name=raw.get("site_name", ""),
                floor_area_sqm=self._dec(raw.get("floor_area_sqm", "0")),
                headcount=self._dec(raw.get("headcount", "0")),
                revenue=self._dec(raw.get("revenue", "0")),
                energy_kwh=self._dec(raw.get("energy_kwh", "0")),
                production_output=self._dec(raw.get("production_output", "0")),
                custom_driver=self._dec(raw.get("custom_driver", "0")),
            )
        self._drivers = drivers
        result.sites_count = len(drivers)

        total_emissions = sum(s.total_emissions_tco2e for s in services)
        type_dist: Dict[str, int] = {}
        for s in services:
            k = s.service_type.value
            type_dist[k] = type_dist.get(k, 0) + 1

        logger.info("Identified %d services, %.2f tCO2e to allocate",
                     len(services), float(total_emissions))
        return {
            "services_identified": len(services),
            "total_to_allocate_tco2e": float(total_emissions),
            "type_distribution": type_dist,
            "sites_with_drivers": len(drivers),
        }

    # -----------------------------------------------------------------
    # PHASE 2 -- ALLOCATION METHOD SELECT
    # -----------------------------------------------------------------

    def _phase_method_select(
        self, input_data: AllocationInput, result: AllocationResult,
    ) -> Dict[str, Any]:
        """Select allocation method per shared service."""
        logger.info("Phase 2 -- Allocation Method Select")
        methods: Dict[str, AllocationMethod] = {}

        for svc in self._services:
            override_str = input_data.method_overrides.get(svc.service_id)
            if override_str:
                try:
                    methods[svc.service_id] = AllocationMethod(override_str)
                    continue
                except ValueError:
                    result.warnings.append(
                        f"Invalid method override '{override_str}' for {svc.service_name}"
                    )

            default_str = DEFAULT_METHOD_MAP.get(svc.service_type.value, "equal_split")
            try:
                methods[svc.service_id] = AllocationMethod(default_str)
            except ValueError:
                methods[svc.service_id] = AllocationMethod.EQUAL_SPLIT

        self._methods = methods

        method_dist: Dict[str, int] = {}
        for m in methods.values():
            method_dist[m.value] = method_dist.get(m.value, 0) + 1

        logger.info("Methods selected: %s", method_dist)
        return {
            "methods_assigned": len(methods),
            "method_distribution": method_dist,
        }

    # -----------------------------------------------------------------
    # PHASE 3 -- CALCULATE
    # -----------------------------------------------------------------

    def _phase_calculate(
        self, input_data: AllocationInput, result: AllocationResult,
    ) -> Dict[str, Any]:
        """Execute allocation calculations."""
        logger.info("Phase 3 -- Calculate")
        items: List[AllocationLineItem] = []
        total_allocated = Decimal("0")

        for svc in self._services:
            method = self._methods.get(svc.service_id, AllocationMethod.EQUAL_SPLIT)
            benefiting = svc.benefiting_site_ids
            if not benefiting:
                result.warnings.append(f"No benefiting sites for {svc.service_name}")
                continue

            # Compute driver total for benefiting sites
            driver_values: Dict[str, Decimal] = {}
            for sid in benefiting:
                drv = self._drivers.get(sid)
                if drv is None:
                    driver_values[sid] = Decimal("1") if method == AllocationMethod.EQUAL_SPLIT else Decimal("0")
                else:
                    driver_values[sid] = self._get_driver_value(drv, method)

            driver_total = sum(driver_values.values())
            if driver_total <= Decimal("0"):
                result.warnings.append(
                    f"Zero driver total for {svc.service_name} ({method.value}), using equal split"
                )
                driver_values = {sid: Decimal("1") for sid in benefiting}
                driver_total = Decimal(str(len(benefiting)))

            # Allocate
            for sid in benefiting:
                dv = driver_values.get(sid, Decimal("0"))
                pct = (dv / driver_total * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                allocated = (svc.total_emissions_tco2e * dv / driver_total).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

                drv_obj = self._drivers.get(sid)
                prov = _compute_hash(
                    f"{svc.service_id}|{sid}|{float(allocated)}|{method.value}"
                )

                item = AllocationLineItem(
                    service_id=svc.service_id,
                    service_name=svc.service_name,
                    site_id=sid,
                    site_name=drv_obj.site_name if drv_obj else sid,
                    method=method,
                    driver_value=dv,
                    driver_total=driver_total,
                    allocation_pct=pct,
                    allocated_tco2e=allocated,
                    provenance_hash=prov,
                )
                items.append(item)
                total_allocated += allocated

        self._items = items
        result.allocation_items = items
        result.total_allocated_tco2e = total_allocated

        logger.info("Allocated %.2f tCO2e across %d line items",
                     float(total_allocated), len(items))
        return {
            "line_items": len(items),
            "total_allocated_tco2e": float(total_allocated),
        }

    def _get_driver_value(self, driver: SiteAllocationDriver, method: AllocationMethod) -> Decimal:
        """Get the driver value based on the selected method."""
        mapping = {
            AllocationMethod.FLOOR_AREA: driver.floor_area_sqm,
            AllocationMethod.HEADCOUNT: driver.headcount,
            AllocationMethod.REVENUE: driver.revenue,
            AllocationMethod.ENERGY_METERED: driver.energy_kwh,
            AllocationMethod.PRODUCTION_OUTPUT: driver.production_output,
            AllocationMethod.CUSTOM: driver.custom_driver,
            AllocationMethod.EQUAL_SPLIT: Decimal("1"),
        }
        return mapping.get(method, Decimal("1"))

    # -----------------------------------------------------------------
    # PHASE 4 -- VERIFY
    # -----------------------------------------------------------------

    def _phase_verify(
        self, input_data: AllocationInput, result: AllocationResult,
    ) -> Dict[str, Any]:
        """Verify allocation completeness and integrity."""
        logger.info("Phase 4 -- Verify")
        checks: List[VerificationCheck] = []

        # Check 1: Each service sums to 100%
        for svc in self._services:
            svc_items = [i for i in self._items if i.service_id == svc.service_id]
            if not svc_items:
                checks.append(VerificationCheck(
                    check_name=f"completeness_{svc.service_id}",
                    status=VerificationStatus.WARNING,
                    message=f"No allocations for service {svc.service_name}",
                ))
                continue

            total_pct = sum(i.allocation_pct for i in svc_items)
            total_amt = sum(i.allocated_tco2e for i in svc_items)
            diff_pct = abs(total_pct - Decimal("100"))
            diff_amt = abs(total_amt - svc.total_emissions_tco2e)

            if diff_pct <= ALLOCATION_TOLERANCE_PCT:
                checks.append(VerificationCheck(
                    check_name=f"pct_sum_{svc.service_id}",
                    status=VerificationStatus.PASSED,
                    message=f"Allocation % sums to {total_pct}% (within tolerance)",
                    expected_value="100.00%",
                    actual_value=f"{total_pct}%",
                ))
            else:
                checks.append(VerificationCheck(
                    check_name=f"pct_sum_{svc.service_id}",
                    status=VerificationStatus.FAILED,
                    message=f"Allocation % sums to {total_pct}% (off by {diff_pct}%)",
                    expected_value="100.00%",
                    actual_value=f"{total_pct}%",
                ))

            # Amount check
            amt_tolerance = svc.total_emissions_tco2e * Decimal("0.01")  # 1%
            if diff_amt <= amt_tolerance:
                checks.append(VerificationCheck(
                    check_name=f"amt_sum_{svc.service_id}",
                    status=VerificationStatus.PASSED,
                    message=f"Allocated amount matches total (diff {diff_amt} tCO2e)",
                    expected_value=str(svc.total_emissions_tco2e),
                    actual_value=str(total_amt),
                ))
            else:
                checks.append(VerificationCheck(
                    check_name=f"amt_sum_{svc.service_id}",
                    status=VerificationStatus.FAILED,
                    message=f"Amount mismatch: expected {svc.total_emissions_tco2e}, got {total_amt}",
                    expected_value=str(svc.total_emissions_tco2e),
                    actual_value=str(total_amt),
                ))

        # Check 2: No negative allocations
        negatives = [i for i in self._items if i.allocated_tco2e < Decimal("0")]
        checks.append(VerificationCheck(
            check_name="no_negatives",
            status=VerificationStatus.PASSED if not negatives else VerificationStatus.FAILED,
            message=f"{len(negatives)} negative allocations found" if negatives
                    else "No negative allocations",
        ))

        # Check 3: All benefiting sites have drivers
        missing_drivers = set()
        for svc in self._services:
            for sid in svc.benefiting_site_ids:
                if sid not in self._drivers:
                    missing_drivers.add(sid)
        checks.append(VerificationCheck(
            check_name="driver_coverage",
            status=VerificationStatus.PASSED if not missing_drivers else VerificationStatus.WARNING,
            message=f"{len(missing_drivers)} sites missing driver data" if missing_drivers
                    else "All benefiting sites have driver data",
        ))

        result.verification_checks = checks
        passed = sum(1 for c in checks if c.status == VerificationStatus.PASSED)
        failed = sum(1 for c in checks if c.status == VerificationStatus.FAILED)

        logger.info("Verification: %d passed, %d failed", passed, failed)
        return {
            "checks_total": len(checks),
            "passed": passed,
            "failed": failed,
            "warnings": sum(1 for c in checks if c.status == VerificationStatus.WARNING),
        }

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    def _dec(self, value: Any) -> Decimal:
        if value is None:
            return Decimal("0")
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal("0")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "AllocationWorkflow",
    "AllocationInput",
    "AllocationResult",
    "AllocationPhase",
    "SharedServiceType",
    "AllocationMethod",
    "VerificationStatus",
    "SharedService",
    "SiteAllocationDriver",
    "AllocationLineItem",
    "VerificationCheck",
    "PhaseResult",
    "PhaseStatus",
    "WorkflowStatus",
]
