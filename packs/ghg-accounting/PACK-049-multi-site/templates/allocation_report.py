# -*- coding: utf-8 -*-
"""
AllocationReport - Shared services allocation report for PACK-049.

Sections: allocation_summary, shared_services, landlord_tenant,
          cogeneration, completeness, provenance.

Author: GreenLang Team
Version: 49.0.0
"""

from __future__ import annotations

import hashlib, logging, uuid, json, time
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


class AllocationLine(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    service_name: str = Field("")
    service_type: str = Field("")
    site_name: str = Field("")
    method: str = Field("")
    driver_value: Decimal = Field(Decimal("0"))
    allocation_pct: Decimal = Field(Decimal("0"))
    allocated_tco2e: Decimal = Field(Decimal("0"))

class ServiceSummary(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    service_name: str = Field("")
    service_type: str = Field("")
    total_tco2e: Decimal = Field(Decimal("0"))
    benefiting_sites: int = Field(0)
    method: str = Field("")
    verified: bool = Field(False)

class CompletenessCheck(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    service_name: str = Field("")
    allocation_sum_pct: Decimal = Field(Decimal("0"))
    amount_allocated: Decimal = Field(Decimal("0"))
    amount_expected: Decimal = Field(Decimal("0"))
    is_complete: bool = Field(True)

class AllocationReportInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    allocation_lines: List[Dict[str, Any]] = Field(default_factory=list)
    services: List[Dict[str, Any]] = Field(default_factory=list)
    completeness_checks: List[Dict[str, Any]] = Field(default_factory=list)
    total_allocated_tco2e: Decimal = Field(Decimal("0"))

class AllocationReportOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    total_allocated_tco2e: Decimal = Field(Decimal("0"))
    services_count: int = Field(0)
    allocation_lines: List[AllocationLine] = Field(default_factory=list)
    service_summaries: List[ServiceSummary] = Field(default_factory=list)
    landlord_tenant_items: List[AllocationLine] = Field(default_factory=list)
    cogeneration_items: List[AllocationLine] = Field(default_factory=list)
    completeness: List[CompletenessCheck] = Field(default_factory=list)
    all_complete: bool = Field(True)
    provenance_hash: str = Field("")


class AllocationReport:
    """Shared services allocation report template."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def render(self, data: Dict[str, Any]) -> AllocationReportOutput:
        start = time.monotonic()
        self.generated_at = _utcnow()
        inp = AllocationReportInput(**data) if isinstance(data, dict) else data

        lines = [AllocationLine(**l) if isinstance(l, dict) else l for l in inp.allocation_lines]
        services = [ServiceSummary(**s) if isinstance(s, dict) else s for s in inp.services]
        checks = [CompletenessCheck(**c) if isinstance(c, dict) else c for c in inp.completeness_checks]

        lt_items = [l for l in lines if l.service_type == "landlord_tenant"]
        cg_items = [l for l in lines if l.service_type == "cogeneration"]
        all_complete = all(c.is_complete for c in checks) if checks else True

        total = inp.total_allocated_tco2e or sum(l.allocated_tco2e for l in lines)
        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = AllocationReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            total_allocated_tco2e=total,
            services_count=len(services),
            allocation_lines=lines,
            service_summaries=services,
            landlord_tenant_items=lt_items,
            cogeneration_items=cg_items,
            completeness=checks,
            all_complete=all_complete,
            provenance_hash=prov,
        )
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return output

    def render_markdown(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_markdown(r)
    def render_html(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_html(r)
    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_json(r)

    def export_markdown(self, r: AllocationReportOutput) -> str:
        lines: List[str] = []
        lines.append(f"# Allocation Report - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Total Allocated:** {r.total_allocated_tco2e:,.2f} tCO2e | **Services:** {r.services_count}")
        lines.append("")
        lines.append("## Allocation Summary")
        lines.append("| Service | Type | Site | Method | Driver | % | tCO2e |")
        lines.append("|---------|------|------|--------|--------|---|-------|")
        for l in r.allocation_lines:
            lines.append(f"| {l.service_name} | {l.service_type} | {l.site_name} | {l.method} | {l.driver_value} | {l.allocation_pct}% | {l.allocated_tco2e:,.2f} |")
        lines.append("")
        if r.service_summaries:
            lines.append("## Shared Services")
            lines.append("| Service | Type | Total tCO2e | Sites | Method | Verified |")
            lines.append("|---------|------|-------------|-------|--------|----------|")
            for s in r.service_summaries:
                lines.append(f"| {s.service_name} | {s.service_type} | {s.total_tco2e:,.2f} | {s.benefiting_sites} | {s.method} | {'Yes' if s.verified else 'No'} |")
            lines.append("")
        if r.landlord_tenant_items:
            lines.append(f"## Landlord-Tenant ({len(r.landlord_tenant_items)} items)")
            for l in r.landlord_tenant_items:
                lines.append(f"- {l.site_name}: {l.allocated_tco2e:,.2f} tCO2e ({l.allocation_pct}%)")
            lines.append("")
        if r.cogeneration_items:
            lines.append(f"## Cogeneration ({len(r.cogeneration_items)} items)")
            for l in r.cogeneration_items:
                lines.append(f"- {l.site_name}: {l.allocated_tco2e:,.2f} tCO2e ({l.allocation_pct}%)")
            lines.append("")
        lines.append("## Completeness")
        status = "PASS" if r.all_complete else "FAIL"
        lines.append(f"Overall: **{status}**")
        for c in r.completeness:
            icon = "PASS" if c.is_complete else "FAIL"
            lines.append(f"- {c.service_name}: {c.allocation_sum_pct}% allocated [{icon}]")
        lines.append("")
        lines.append(f"---\n*Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: AllocationReportOutput) -> str:
        md = self.export_markdown(r)
        return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Allocation Report</title></head><body><pre>{md}</pre></body></html>"

    def export_json(self, r: AllocationReportOutput) -> Dict[str, Any]:
        return json.loads(r.model_dump_json())

    def export_csv(self, r: AllocationReportOutput) -> str:
        lines_out = ["service_name,service_type,site_name,method,allocation_pct,allocated_tco2e"]
        for l in r.allocation_lines:
            lines_out.append(f"{l.service_name},{l.service_type},{l.site_name},{l.method},{l.allocation_pct},{l.allocated_tco2e}")
        return "\n".join(lines_out)


__all__ = ["AllocationReport", "AllocationReportInput", "AllocationReportOutput"]
