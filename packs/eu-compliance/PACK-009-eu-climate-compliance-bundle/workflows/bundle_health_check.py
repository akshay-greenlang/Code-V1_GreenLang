# -*- coding: utf-8 -*-
"""
Bundle Health Check Workflow
=================================

Three-phase workflow that runs health checks across all four constituent
regulation packs (CSRD, CBAM, EU Taxonomy, EUDR), verifies cross-pack
bridges and data flows, and produces a bundle-level health status report.

Phases:
    1. PackLevelChecks - Run health checks in all 4 packs (20 categories each)
    2. IntegrationChecks - Verify cross-pack bridges and data flows
    3. BundleStatus - Produce bundle-level health status

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class RegulationPack(str, Enum):
    """Constituent regulation packs in the bundle."""
    CSRD = "CSRD"
    CBAM = "CBAM"
    EU_TAXONOMY = "EU_TAXONOMY"
    EUDR = "EUDR"


class HealthStatus(str, Enum):
    """Health status for a check."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


class CheckCategory(str, Enum):
    """Category of health check."""
    DATA_AVAILABILITY = "DATA_AVAILABILITY"
    DATA_QUALITY = "DATA_QUALITY"
    DATA_FRESHNESS = "DATA_FRESHNESS"
    PROCESS_STATUS = "PROCESS_STATUS"
    INTEGRATION_STATUS = "INTEGRATION_STATUS"
    SYSTEM_STATUS = "SYSTEM_STATUS"
    COMPLIANCE_STATUS = "COMPLIANCE_STATUS"
    CONFIGURATION = "CONFIGURATION"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"


# =============================================================================
# HEALTH CHECK DEFINITIONS
# =============================================================================


PACK_HEALTH_CHECKS: Dict[str, List[Dict[str, Any]]] = {
    RegulationPack.CSRD.value: [
        {"check_id": "CSRD-HC01", "name": "Scope 1 emissions data available", "category": "DATA_AVAILABILITY", "critical": True},
        {"check_id": "CSRD-HC02", "name": "Scope 2 emissions data available", "category": "DATA_AVAILABILITY", "critical": True},
        {"check_id": "CSRD-HC03", "name": "Scope 3 emissions data available", "category": "DATA_AVAILABILITY", "critical": True},
        {"check_id": "CSRD-HC04", "name": "Energy consumption data quality", "category": "DATA_QUALITY", "critical": True},
        {"check_id": "CSRD-HC05", "name": "Double materiality assessment current", "category": "DATA_FRESHNESS", "critical": True},
        {"check_id": "CSRD-HC06", "name": "Transition plan documented", "category": "PROCESS_STATUS", "critical": True},
        {"check_id": "CSRD-HC07", "name": "ESRS mapping complete", "category": "CONFIGURATION", "critical": True},
        {"check_id": "CSRD-HC08", "name": "Assurance provider connected", "category": "INTEGRATION_STATUS", "critical": False},
        {"check_id": "CSRD-HC09", "name": "Value chain data pipeline active", "category": "SYSTEM_STATUS", "critical": False},
        {"check_id": "CSRD-HC10", "name": "Governance structure configured", "category": "CONFIGURATION", "critical": True},
        {"check_id": "CSRD-HC11", "name": "Social metrics data quality", "category": "DATA_QUALITY", "critical": False},
        {"check_id": "CSRD-HC12", "name": "Biodiversity data available", "category": "DATA_AVAILABILITY", "critical": False},
        {"check_id": "CSRD-HC13", "name": "Water consumption data available", "category": "DATA_AVAILABILITY", "critical": False},
        {"check_id": "CSRD-HC14", "name": "Waste management data available", "category": "DATA_AVAILABILITY", "critical": False},
        {"check_id": "CSRD-HC15", "name": "iXBRL tagging system operational", "category": "SYSTEM_STATUS", "critical": True},
        {"check_id": "CSRD-HC16", "name": "Reporting period configured", "category": "CONFIGURATION", "critical": True},
        {"check_id": "CSRD-HC17", "name": "Data access controls in place", "category": "SECURITY", "critical": True},
        {"check_id": "CSRD-HC18", "name": "Audit trail active", "category": "COMPLIANCE_STATUS", "critical": True},
        {"check_id": "CSRD-HC19", "name": "Report generation latency", "category": "PERFORMANCE", "critical": False},
        {"check_id": "CSRD-HC20", "name": "Data refresh within SLA", "category": "DATA_FRESHNESS", "critical": False},
    ],
    RegulationPack.CBAM.value: [
        {"check_id": "CBAM-HC01", "name": "Import data ingestion active", "category": "DATA_AVAILABILITY", "critical": True},
        {"check_id": "CBAM-HC02", "name": "Embedded emissions calculated", "category": "DATA_QUALITY", "critical": True},
        {"check_id": "CBAM-HC03", "name": "Supplier data current", "category": "DATA_FRESHNESS", "critical": True},
        {"check_id": "CBAM-HC04", "name": "CN code classification accurate", "category": "DATA_QUALITY", "critical": True},
        {"check_id": "CBAM-HC05", "name": "Certificate management operational", "category": "PROCESS_STATUS", "critical": True},
        {"check_id": "CBAM-HC06", "name": "CBAM registry connection active", "category": "INTEGRATION_STATUS", "critical": True},
        {"check_id": "CBAM-HC07", "name": "Quarterly report generator ready", "category": "SYSTEM_STATUS", "critical": True},
        {"check_id": "CBAM-HC08", "name": "Annual declaration pipeline ready", "category": "SYSTEM_STATUS", "critical": True},
        {"check_id": "CBAM-HC09", "name": "Carbon price feed active", "category": "DATA_FRESHNESS", "critical": False},
        {"check_id": "CBAM-HC10", "name": "Verification process documented", "category": "PROCESS_STATUS", "critical": True},
        {"check_id": "CBAM-HC11", "name": "Country of origin data complete", "category": "DATA_AVAILABILITY", "critical": True},
        {"check_id": "CBAM-HC12", "name": "Installation data validated", "category": "DATA_QUALITY", "critical": True},
        {"check_id": "CBAM-HC13", "name": "Customs integration configured", "category": "INTEGRATION_STATUS", "critical": False},
        {"check_id": "CBAM-HC14", "name": "Declarant authorization valid", "category": "COMPLIANCE_STATUS", "critical": True},
        {"check_id": "CBAM-HC15", "name": "Free allocation tracking", "category": "PROCESS_STATUS", "critical": False},
        {"check_id": "CBAM-HC16", "name": "Emission factor database current", "category": "DATA_FRESHNESS", "critical": True},
        {"check_id": "CBAM-HC17", "name": "CBAM data encryption active", "category": "SECURITY", "critical": True},
        {"check_id": "CBAM-HC18", "name": "Audit log retention policy", "category": "COMPLIANCE_STATUS", "critical": True},
        {"check_id": "CBAM-HC19", "name": "Calculation engine performance", "category": "PERFORMANCE", "critical": False},
        {"check_id": "CBAM-HC20", "name": "Report generation within SLA", "category": "PERFORMANCE", "critical": False},
    ],
    RegulationPack.EU_TAXONOMY.value: [
        {"check_id": "TAX-HC01", "name": "Activity eligibility data loaded", "category": "DATA_AVAILABILITY", "critical": True},
        {"check_id": "TAX-HC02", "name": "NACE code mapping complete", "category": "CONFIGURATION", "critical": True},
        {"check_id": "TAX-HC03", "name": "Revenue KPI data available", "category": "DATA_AVAILABILITY", "critical": True},
        {"check_id": "TAX-HC04", "name": "CapEx KPI data available", "category": "DATA_AVAILABILITY", "critical": True},
        {"check_id": "TAX-HC05", "name": "OpEx KPI data available", "category": "DATA_AVAILABILITY", "critical": False},
        {"check_id": "TAX-HC06", "name": "DNSH criteria configured", "category": "CONFIGURATION", "critical": True},
        {"check_id": "TAX-HC07", "name": "Substantial contribution assessed", "category": "PROCESS_STATUS", "critical": True},
        {"check_id": "TAX-HC08", "name": "Minimum safeguards evaluated", "category": "PROCESS_STATUS", "critical": True},
        {"check_id": "TAX-HC09", "name": "Technical screening criteria loaded", "category": "DATA_AVAILABILITY", "critical": True},
        {"check_id": "TAX-HC10", "name": "Delegated act version current", "category": "DATA_FRESHNESS", "critical": True},
        {"check_id": "TAX-HC11", "name": "KPI calculation engine operational", "category": "SYSTEM_STATUS", "critical": True},
        {"check_id": "TAX-HC12", "name": "Activity classification quality", "category": "DATA_QUALITY", "critical": True},
        {"check_id": "TAX-HC13", "name": "Financial data integration active", "category": "INTEGRATION_STATUS", "critical": True},
        {"check_id": "TAX-HC14", "name": "Environmental data pipeline active", "category": "INTEGRATION_STATUS", "critical": True},
        {"check_id": "TAX-HC15", "name": "Taxonomy template version current", "category": "CONFIGURATION", "critical": False},
        {"check_id": "TAX-HC16", "name": "Third-party verification setup", "category": "PROCESS_STATUS", "critical": False},
        {"check_id": "TAX-HC17", "name": "Data access controls active", "category": "SECURITY", "critical": True},
        {"check_id": "TAX-HC18", "name": "Compliance audit trail active", "category": "COMPLIANCE_STATUS", "critical": True},
        {"check_id": "TAX-HC19", "name": "Dashboard rendering performance", "category": "PERFORMANCE", "critical": False},
        {"check_id": "TAX-HC20", "name": "Data synchronization within SLA", "category": "PERFORMANCE", "critical": False},
    ],
    RegulationPack.EUDR.value: [
        {"check_id": "EUDR-HC01", "name": "Commodity data loaded", "category": "DATA_AVAILABILITY", "critical": True},
        {"check_id": "EUDR-HC02", "name": "Geolocation data available", "category": "DATA_AVAILABILITY", "critical": True},
        {"check_id": "EUDR-HC03", "name": "Supply chain map current", "category": "DATA_FRESHNESS", "critical": True},
        {"check_id": "EUDR-HC04", "name": "Risk assessment model active", "category": "PROCESS_STATUS", "critical": True},
        {"check_id": "EUDR-HC05", "name": "Satellite monitoring feed active", "category": "INTEGRATION_STATUS", "critical": True},
        {"check_id": "EUDR-HC06", "name": "Due diligence workflow configured", "category": "CONFIGURATION", "critical": True},
        {"check_id": "EUDR-HC07", "name": "Deforestation cutoff date set", "category": "CONFIGURATION", "critical": True},
        {"check_id": "EUDR-HC08", "name": "Country risk database current", "category": "DATA_FRESHNESS", "critical": True},
        {"check_id": "EUDR-HC09", "name": "Supplier verification active", "category": "PROCESS_STATUS", "critical": True},
        {"check_id": "EUDR-HC10", "name": "DD statement generator ready", "category": "SYSTEM_STATUS", "critical": True},
        {"check_id": "EUDR-HC11", "name": "EU Information System connected", "category": "INTEGRATION_STATUS", "critical": True},
        {"check_id": "EUDR-HC12", "name": "Legality documentation complete", "category": "DATA_AVAILABILITY", "critical": True},
        {"check_id": "EUDR-HC13", "name": "GIS mapping engine operational", "category": "SYSTEM_STATUS", "critical": False},
        {"check_id": "EUDR-HC14", "name": "Commodity tracing accuracy", "category": "DATA_QUALITY", "critical": True},
        {"check_id": "EUDR-HC15", "name": "Monitoring alert system active", "category": "PROCESS_STATUS", "critical": False},
        {"check_id": "EUDR-HC16", "name": "Training compliance tracked", "category": "COMPLIANCE_STATUS", "critical": False},
        {"check_id": "EUDR-HC17", "name": "Geolocation data encryption", "category": "SECURITY", "critical": True},
        {"check_id": "EUDR-HC18", "name": "Audit trail for DD statements", "category": "COMPLIANCE_STATUS", "critical": True},
        {"check_id": "EUDR-HC19", "name": "Risk scoring engine performance", "category": "PERFORMANCE", "critical": False},
        {"check_id": "EUDR-HC20", "name": "Satellite data processing latency", "category": "PERFORMANCE", "critical": False},
    ],
}

CROSS_PACK_BRIDGES: List[Dict[str, Any]] = [
    {"bridge_id": "BRG-001", "name": "GHG emissions data flow", "from_pack": "CSRD", "to_pack": "CBAM", "data_field": "scope1_scope2_emissions", "critical": True},
    {"bridge_id": "BRG-002", "name": "GHG emissions to Taxonomy DNSH", "from_pack": "CSRD", "to_pack": "EU_TAXONOMY", "data_field": "ghg_emissions", "critical": True},
    {"bridge_id": "BRG-003", "name": "CBAM embedded emissions to CSRD E1", "from_pack": "CBAM", "to_pack": "CSRD", "data_field": "embedded_emissions", "critical": True},
    {"bridge_id": "BRG-004", "name": "Supplier data CBAM to EUDR", "from_pack": "CBAM", "to_pack": "EUDR", "data_field": "supplier_info", "critical": True},
    {"bridge_id": "BRG-005", "name": "Country of origin CBAM to EUDR", "from_pack": "CBAM", "to_pack": "EUDR", "data_field": "country_origin", "critical": False},
    {"bridge_id": "BRG-006", "name": "Taxonomy KPIs to CSRD disclosures", "from_pack": "EU_TAXONOMY", "to_pack": "CSRD", "data_field": "taxonomy_kpis", "critical": True},
    {"bridge_id": "BRG-007", "name": "CSRD governance to Taxonomy safeguards", "from_pack": "CSRD", "to_pack": "EU_TAXONOMY", "data_field": "governance_data", "critical": False},
    {"bridge_id": "BRG-008", "name": "EUDR biodiversity to CSRD E4", "from_pack": "EUDR", "to_pack": "CSRD", "data_field": "biodiversity_data", "critical": False},
    {"bridge_id": "BRG-009", "name": "CBAM carbon price to CSRD E1-8", "from_pack": "CBAM", "to_pack": "CSRD", "data_field": "carbon_price", "critical": False},
    {"bridge_id": "BRG-010", "name": "EUDR risk data to CSRD value chain", "from_pack": "EUDR", "to_pack": "CSRD", "data_field": "risk_assessment", "critical": False},
    {"bridge_id": "BRG-011", "name": "CSRD energy data to Taxonomy SC", "from_pack": "CSRD", "to_pack": "EU_TAXONOMY", "data_field": "energy_data", "critical": True},
    {"bridge_id": "BRG-012", "name": "EUDR supply chain to CBAM verification", "from_pack": "EUDR", "to_pack": "CBAM", "data_field": "supply_chain", "critical": False},
]


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(...)
    status: WorkflowStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class WorkflowConfig(BaseModel):
    """Configuration for bundle health check workflow."""
    organization_id: str = Field(...)
    reporting_year: int = Field(..., ge=2024, le=2050)
    target_packs: List[RegulationPack] = Field(
        default_factory=lambda: list(RegulationPack)
    )
    system_status: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current system status indicators"
    )
    data_availability: Dict[str, bool] = Field(
        default_factory=dict,
        description="Data field availability flags"
    )
    last_health_check: Optional[str] = Field(
        None,
        description="ISO timestamp of last health check"
    )
    skip_phases: List[str] = Field(default_factory=list)


class BundleHealthCheckResult(WorkflowResult):
    """Result from bundle health check workflow."""
    total_checks: int = Field(default=0)
    healthy_checks: int = Field(default=0)
    degraded_checks: int = Field(default=0)
    unhealthy_checks: int = Field(default=0)
    bundle_health: str = Field(default="UNKNOWN")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class BundleHealthCheckWorkflow:
    """
    Three-phase bundle health check workflow.

    Runs pack-level checks, verifies integration bridges,
    and produces bundle-level health status.

    Example:
        >>> wf = BundleHealthCheckWorkflow()
        >>> config = WorkflowConfig(
        ...     organization_id="org-123",
        ...     reporting_year=2026,
        ... )
        >>> result = wf.execute(config)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "bundle_health_check"

    PHASE_ORDER = [
        "pack_level_checks",
        "integration_checks",
        "bundle_status",
    ]

    def __init__(self) -> None:
        """Initialize the bundle health check workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._phase_outputs: Dict[str, Dict[str, Any]] = {}

    def execute(self, config: WorkflowConfig) -> BundleHealthCheckResult:
        """
        Execute the three-phase bundle health check workflow.

        Args:
            config: Validated workflow configuration.

        Returns:
            BundleHealthCheckResult with health check outcomes.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting bundle health check %s for org=%s year=%d",
            self.workflow_id, config.organization_id, config.reporting_year,
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING
        phase_methods = {
            "pack_level_checks": self._phase_pack_level_checks,
            "integration_checks": self._phase_integration_checks,
            "bundle_status": self._phase_bundle_status,
        }

        for phase_name in self.PHASE_ORDER:
            if phase_name in config.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                continue

            try:
                phase_result = phase_methods[phase_name](config)
                completed_phases.append(phase_result)
                if phase_result.status == PhaseStatus.COMPLETED:
                    self._phase_outputs[phase_name] = phase_result.outputs
                elif phase_result.status == PhaseStatus.FAILED:
                    overall_status = WorkflowStatus.FAILED
                    break
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = datetime.utcnow()
        summary = self._build_summary()
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        return BundleHealthCheckResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=(completed_at - started_at).total_seconds(),
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            total_checks=summary.get("total_checks", 0),
            healthy_checks=summary.get("healthy_checks", 0),
            degraded_checks=summary.get("degraded_checks", 0),
            unhealthy_checks=summary.get("unhealthy_checks", 0),
            bundle_health=summary.get("bundle_health", "UNKNOWN"),
        )

    # -------------------------------------------------------------------------
    # Phase 1: Pack-Level Checks
    # -------------------------------------------------------------------------

    def _phase_pack_level_checks(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 1: Run health checks in all 4 packs.

        Executes 20 health checks per pack covering data availability,
        quality, freshness, process status, and more.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            pack_results: Dict[str, Dict[str, Any]] = {}
            total_checks = 0
            total_healthy = 0
            total_degraded = 0
            total_unhealthy = 0

            for pack in config.target_packs:
                pack_name = pack.value
                checks = PACK_HEALTH_CHECKS.get(pack_name, [])
                check_results: List[Dict[str, Any]] = []

                for check_def in checks:
                    result = self._run_health_check(
                        check_def, config.system_status, config.data_availability
                    )
                    check_results.append(result)

                    if result["status"] == HealthStatus.HEALTHY.value:
                        total_healthy += 1
                    elif result["status"] == HealthStatus.DEGRADED.value:
                        total_degraded += 1
                    elif result["status"] == HealthStatus.UNHEALTHY.value:
                        total_unhealthy += 1

                        if result["critical"]:
                            errors.append(
                                f"[{pack_name}] Critical check UNHEALTHY: {result['name']}"
                            )

                    total_checks += 1

                pack_healthy = sum(1 for r in check_results if r["status"] == HealthStatus.HEALTHY.value)
                pack_total = len(check_results)
                pack_health_pct = (pack_healthy / max(pack_total, 1)) * 100

                critical_unhealthy = sum(
                    1 for r in check_results
                    if r["status"] == HealthStatus.UNHEALTHY.value and r["critical"]
                )

                if critical_unhealthy > 0:
                    pack_health = HealthStatus.UNHEALTHY.value
                elif pack_health_pct >= 80:
                    pack_health = HealthStatus.HEALTHY.value
                elif pack_health_pct >= 50:
                    pack_health = HealthStatus.DEGRADED.value
                else:
                    pack_health = HealthStatus.UNHEALTHY.value

                pack_results[pack_name] = {
                    "pack": pack_name,
                    "check_results": check_results,
                    "total_checks": pack_total,
                    "healthy": pack_healthy,
                    "degraded": sum(1 for r in check_results if r["status"] == HealthStatus.DEGRADED.value),
                    "unhealthy": sum(1 for r in check_results if r["status"] == HealthStatus.UNHEALTHY.value),
                    "health_percentage": round(pack_health_pct, 2),
                    "pack_health": pack_health,
                    "critical_failures": critical_unhealthy,
                    "checked_at": datetime.utcnow().isoformat(),
                }

                by_category: Dict[str, Dict[str, int]] = {}
                for r in check_results:
                    cat = r["category"]
                    if cat not in by_category:
                        by_category[cat] = {"healthy": 0, "degraded": 0, "unhealthy": 0}
                    by_category[cat][r["status"].lower()] = by_category[cat].get(r["status"].lower(), 0) + 1
                pack_results[pack_name]["by_category"] = by_category

            outputs["pack_results"] = pack_results
            outputs["total_checks"] = total_checks
            outputs["total_healthy"] = total_healthy
            outputs["total_degraded"] = total_degraded
            outputs["total_unhealthy"] = total_unhealthy
            outputs["overall_health_pct"] = round(
                (total_healthy / max(total_checks, 1)) * 100, 2
            )

            logger.info(
                "Pack-level checks complete: %d total, %d healthy, %d degraded, %d unhealthy",
                total_checks, total_healthy, total_degraded, total_unhealthy,
            )

            status = PhaseStatus.COMPLETED
            records = total_checks

        except Exception as exc:
            logger.error("Pack-level checks failed: %s", exc, exc_info=True)
            errors.append(f"Pack-level checks failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="pack_level_checks",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _run_health_check(
        self,
        check_def: Dict[str, Any],
        system_status: Dict[str, Any],
        data_availability: Dict[str, bool],
    ) -> Dict[str, Any]:
        """
        Run a single health check (simulated).

        Determines health based on system_status and data_availability
        configuration provided by the caller.
        """
        check_id = check_def["check_id"]
        name = check_def["name"]
        category = check_def["category"]
        critical = check_def["critical"]

        check_key = check_id.lower().replace("-", "_")

        if check_key in system_status:
            raw_status = system_status[check_key]
            if isinstance(raw_status, str):
                health = raw_status.upper()
            elif isinstance(raw_status, bool):
                health = HealthStatus.HEALTHY.value if raw_status else HealthStatus.UNHEALTHY.value
            elif isinstance(raw_status, (int, float)):
                if raw_status >= 0.8:
                    health = HealthStatus.HEALTHY.value
                elif raw_status >= 0.5:
                    health = HealthStatus.DEGRADED.value
                else:
                    health = HealthStatus.UNHEALTHY.value
            else:
                health = HealthStatus.UNKNOWN.value
        else:
            check_hash = hashlib.md5(check_id.encode()).hexdigest()
            hash_val = int(check_hash[:4], 16) % 100
            if hash_val < 65:
                health = HealthStatus.HEALTHY.value
            elif hash_val < 85:
                health = HealthStatus.DEGRADED.value
            else:
                health = HealthStatus.UNHEALTHY.value

        return {
            "check_id": check_id,
            "name": name,
            "category": category,
            "critical": critical,
            "status": health,
            "message": f"{name}: {health}",
            "checked_at": datetime.utcnow().isoformat(),
        }

    # -------------------------------------------------------------------------
    # Phase 2: Integration Checks
    # -------------------------------------------------------------------------

    def _phase_integration_checks(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 2: Verify cross-pack bridges and data flows.

        Checks that data flows between constituent packs are
        operational and data is flowing correctly.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            pack_out = self._phase_outputs.get("pack_level_checks", {})
            pack_results = pack_out.get("pack_results", {})
            active_packs = {p.value for p in config.target_packs}

            bridge_results: List[Dict[str, Any]] = []
            bridges_healthy = 0
            bridges_degraded = 0
            bridges_unhealthy = 0

            for bridge in CROSS_PACK_BRIDGES:
                from_pack = bridge["from_pack"]
                to_pack = bridge["to_pack"]

                if from_pack not in active_packs or to_pack not in active_packs:
                    continue

                from_health = pack_results.get(from_pack, {}).get("pack_health", "UNKNOWN")
                to_health = pack_results.get(to_pack, {}).get("pack_health", "UNKNOWN")

                if from_health == HealthStatus.HEALTHY.value and to_health == HealthStatus.HEALTHY.value:
                    bridge_status = HealthStatus.HEALTHY.value
                    bridges_healthy += 1
                elif from_health == HealthStatus.UNHEALTHY.value or to_health == HealthStatus.UNHEALTHY.value:
                    bridge_status = HealthStatus.UNHEALTHY.value
                    bridges_unhealthy += 1
                    if bridge["critical"]:
                        errors.append(
                            f"Critical bridge UNHEALTHY: {bridge['name']} "
                            f"({from_pack}->{to_pack})"
                        )
                else:
                    bridge_status = HealthStatus.DEGRADED.value
                    bridges_degraded += 1

                data_field = bridge["data_field"]
                data_available = config.data_availability.get(data_field, True)

                if not data_available and bridge_status != HealthStatus.UNHEALTHY.value:
                    bridge_status = HealthStatus.DEGRADED.value
                    if bridge["critical"]:
                        warnings.append(
                            f"Bridge data field '{data_field}' unavailable: {bridge['name']}"
                        )

                bridge_results.append({
                    "bridge_id": bridge["bridge_id"],
                    "name": bridge["name"],
                    "from_pack": from_pack,
                    "to_pack": to_pack,
                    "data_field": data_field,
                    "critical": bridge["critical"],
                    "status": bridge_status,
                    "from_pack_health": from_health,
                    "to_pack_health": to_health,
                    "data_available": data_available,
                    "checked_at": datetime.utcnow().isoformat(),
                })

            outputs["bridge_results"] = bridge_results
            outputs["total_bridges"] = len(bridge_results)
            outputs["bridges_healthy"] = bridges_healthy
            outputs["bridges_degraded"] = bridges_degraded
            outputs["bridges_unhealthy"] = bridges_unhealthy
            outputs["integration_health_pct"] = round(
                (bridges_healthy / max(len(bridge_results), 1)) * 100, 2
            )

            flow_matrix: Dict[str, Dict[str, str]] = {}
            for br in bridge_results:
                key = f"{br['from_pack']}->{br['to_pack']}"
                if key not in flow_matrix:
                    flow_matrix[key] = {"from": br["from_pack"], "to": br["to_pack"], "status": br["status"]}
                else:
                    current = flow_matrix[key]["status"]
                    if br["status"] == HealthStatus.UNHEALTHY.value or current == HealthStatus.UNHEALTHY.value:
                        flow_matrix[key]["status"] = HealthStatus.UNHEALTHY.value
                    elif br["status"] == HealthStatus.DEGRADED.value or current == HealthStatus.DEGRADED.value:
                        flow_matrix[key]["status"] = HealthStatus.DEGRADED.value
            outputs["flow_matrix"] = flow_matrix

            logger.info(
                "Integration checks complete: %d bridges, %d healthy, %d degraded, %d unhealthy",
                len(bridge_results), bridges_healthy, bridges_degraded, bridges_unhealthy,
            )

            status = PhaseStatus.COMPLETED
            records = len(bridge_results)

        except Exception as exc:
            logger.error("Integration checks failed: %s", exc, exc_info=True)
            errors.append(f"Integration checks failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="integration_checks",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Bundle Status
    # -------------------------------------------------------------------------

    def _phase_bundle_status(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 3: Produce bundle-level health status.

        Aggregates pack-level and integration check results into
        a single bundle health assessment.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            pack_out = self._phase_outputs.get("pack_level_checks", {})
            integ_out = self._phase_outputs.get("integration_checks", {})
            pack_results = pack_out.get("pack_results", {})
            overall_health_pct = pack_out.get("overall_health_pct", 0.0)
            integ_health_pct = integ_out.get("integration_health_pct", 0.0)

            bundle_health_pct = (overall_health_pct * 0.7) + (integ_health_pct * 0.3)
            bundle_health_pct = round(bundle_health_pct, 2)

            critical_pack_failures = sum(
                pr.get("critical_failures", 0) for pr in pack_results.values()
            )
            critical_bridge_failures = integ_out.get("bridges_unhealthy", 0)

            if critical_pack_failures > 0 or critical_bridge_failures > 0:
                bundle_health = HealthStatus.UNHEALTHY.value
            elif bundle_health_pct >= 80:
                bundle_health = HealthStatus.HEALTHY.value
            elif bundle_health_pct >= 50:
                bundle_health = HealthStatus.DEGRADED.value
            else:
                bundle_health = HealthStatus.UNHEALTHY.value

            pack_health_summary: Dict[str, str] = {
                pack: pr.get("pack_health", "UNKNOWN")
                for pack, pr in pack_results.items()
            }

            attention_items: List[Dict[str, Any]] = []
            for pack, pr in pack_results.items():
                for check in pr.get("check_results", []):
                    if check["status"] == HealthStatus.UNHEALTHY.value:
                        attention_items.append({
                            "pack": pack,
                            "check_id": check["check_id"],
                            "name": check["name"],
                            "critical": check["critical"],
                            "priority": "CRITICAL" if check["critical"] else "HIGH",
                        })
                    elif check["status"] == HealthStatus.DEGRADED.value and check["critical"]:
                        attention_items.append({
                            "pack": pack,
                            "check_id": check["check_id"],
                            "name": check["name"],
                            "critical": check["critical"],
                            "priority": "MEDIUM",
                        })

            attention_items.sort(
                key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x["priority"], 3)
            )

            outputs["bundle_health"] = bundle_health
            outputs["bundle_health_pct"] = bundle_health_pct
            outputs["pack_health_pct"] = overall_health_pct
            outputs["integration_health_pct"] = integ_health_pct
            outputs["pack_health_summary"] = pack_health_summary
            outputs["critical_pack_failures"] = critical_pack_failures
            outputs["critical_bridge_failures"] = critical_bridge_failures
            outputs["attention_items"] = attention_items
            outputs["attention_count"] = len(attention_items)
            outputs["checked_at"] = datetime.utcnow().isoformat()

            outputs["health_dashboard"] = {
                "organization_id": config.organization_id,
                "reporting_year": config.reporting_year,
                "bundle_health": bundle_health,
                "bundle_score": bundle_health_pct,
                "packs": pack_health_summary,
                "critical_issues": critical_pack_failures + critical_bridge_failures,
                "needs_attention": len(attention_items),
                "last_check": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Bundle status complete: health=%s score=%.2f%% critical_issues=%d",
                bundle_health, bundle_health_pct,
                critical_pack_failures + critical_bridge_failures,
            )

            status = PhaseStatus.COMPLETED
            records = len(pack_results)

        except Exception as exc:
            logger.error("Bundle status failed: %s", exc, exc_info=True)
            errors.append(f"Bundle status failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="bundle_status",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def _build_summary(self) -> Dict[str, Any]:
        """Build workflow summary from all phase outputs."""
        pack_out = self._phase_outputs.get("pack_level_checks", {})
        integ_out = self._phase_outputs.get("integration_checks", {})
        bundle_out = self._phase_outputs.get("bundle_status", {})

        return {
            "total_checks": pack_out.get("total_checks", 0) + integ_out.get("total_bridges", 0),
            "healthy_checks": pack_out.get("total_healthy", 0) + integ_out.get("bridges_healthy", 0),
            "degraded_checks": pack_out.get("total_degraded", 0) + integ_out.get("bridges_degraded", 0),
            "unhealthy_checks": pack_out.get("total_unhealthy", 0) + integ_out.get("bridges_unhealthy", 0),
            "bundle_health": bundle_out.get("bundle_health", "UNKNOWN"),
            "bundle_health_pct": bundle_out.get("bundle_health_pct", 0.0),
            "attention_items": bundle_out.get("attention_count", 0),
        }


# =============================================================================
# UTILITIES
# =============================================================================


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    serialized = str(data).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
