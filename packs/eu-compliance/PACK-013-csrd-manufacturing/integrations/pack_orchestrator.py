"""
PACK-013 CSRD Manufacturing Pack - Integration Orchestrator.

Manages the 11-phase manufacturing compliance pipeline from data intake
through final CSRD/ESRS reporting. Coordinates agents across MRV, DATA,
and FOUND layers with provenance tracking at every phase boundary.
"""

import hashlib
import importlib
import time
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PhaseStatus(str, Enum):
    """Status of a pipeline phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class PipelinePhase(str, Enum):
    """The 11 phases of the manufacturing compliance pipeline."""
    INITIALIZATION = "initialization"
    DATA_INTAKE = "data_intake"
    QUALITY_ASSURANCE = "quality_assurance"
    PROCESS_EMISSIONS = "process_emissions"
    ENERGY_ANALYSIS = "energy_analysis"
    PRODUCT_PCF = "product_pcf"
    CIRCULAR_ECONOMY = "circular_economy"
    WATER_POLLUTION = "water_pollution"
    BAT_COMPLIANCE = "bat_compliance"
    SUPPLY_CHAIN = "supply_chain"
    REPORTING = "reporting"


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------

class RetryPolicy(BaseModel):
    """Retry policy for failed phases."""
    max_retries: int = Field(default=3, ge=0, le=10)
    backoff_factor: float = Field(default=1.5, ge=1.0)
    retry_on_timeout: bool = Field(default=True)
    retry_on_error: bool = Field(default=True)


class OrchestratorConfig(BaseModel):
    """Configuration for the manufacturing pipeline orchestrator."""
    phases_enabled: Dict[str, bool] = Field(
        default_factory=lambda: {p.value: True for p in PipelinePhase}
    )
    parallel_execution: bool = Field(default=False)
    timeout_per_phase: int = Field(
        default=300_000, description="Phase timeout in milliseconds"
    )
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    provenance_enabled: bool = Field(default=True)
    dry_run: bool = Field(default=False)
    facility_id: Optional[str] = Field(default=None)
    reporting_year: int = Field(default=2025)
    sub_sector: Optional[str] = Field(default=None)
    log_level: str = Field(default="INFO")


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class PhaseProvenance(BaseModel):
    """Provenance record for a single phase execution."""
    phase_name: str
    started_at: float
    completed_at: float
    input_hash: str
    output_hash: str
    agent_versions: Dict[str, str] = Field(default_factory=dict)


class PhaseResult(BaseModel):
    """Result of a single pipeline phase execution."""
    phase_name: str
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    duration_ms: int = Field(default=0, ge=0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    records_processed: int = Field(default=0, ge=0)
    provenance_hash: str = Field(default="")
    provenance: Optional[PhaseProvenance] = Field(default=None)
    retries_used: int = Field(default=0, ge=0)


class PipelineResult(BaseModel):
    """Aggregate result of the full manufacturing pipeline."""
    total_phases: int = Field(default=0, ge=0)
    completed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    results: Dict[str, PhaseResult] = Field(default_factory=dict)
    total_duration_ms: int = Field(default=0, ge=0)
    provenance_hash: str = Field(default="")
    pipeline_status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    facility_id: Optional[str] = Field(default=None)
    reporting_year: int = Field(default=2025)


# ---------------------------------------------------------------------------
# Agent stub -- graceful fallback when agents are unavailable
# ---------------------------------------------------------------------------

class _AgentStub:
    """
    Placeholder agent used when a real agent module cannot be imported.

    Every method call returns a dict with status=unavailable so the
    pipeline degrades gracefully rather than crashing.
    """

    def __init__(self, agent_id: str, reason: str = "not installed") -> None:
        self._agent_id = agent_id
        self._reason = reason
        logger.warning(
            "Agent %s unavailable: %s  -- using stub", agent_id, reason
        )

    def __getattr__(self, name: str) -> Callable[..., Dict[str, Any]]:
        def _stub_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            logger.debug("Stub call: %s.%s", self._agent_id, name)
            return {
                "status": "unavailable",
                "agent_id": self._agent_id,
                "method": name,
                "reason": self._reason,
            }
        return _stub_method

    def __repr__(self) -> str:
        return f"_AgentStub(agent_id={self._agent_id!r}, reason={self._reason!r})"


# ---------------------------------------------------------------------------
# Phase dependency graph
# ---------------------------------------------------------------------------

PHASE_DEPENDENCIES: Dict[str, List[str]] = {
    PipelinePhase.INITIALIZATION.value: [],
    PipelinePhase.DATA_INTAKE.value: [
        PipelinePhase.INITIALIZATION.value,
    ],
    PipelinePhase.QUALITY_ASSURANCE.value: [
        PipelinePhase.DATA_INTAKE.value,
    ],
    PipelinePhase.PROCESS_EMISSIONS.value: [
        PipelinePhase.QUALITY_ASSURANCE.value,
    ],
    PipelinePhase.ENERGY_ANALYSIS.value: [
        PipelinePhase.QUALITY_ASSURANCE.value,
    ],
    PipelinePhase.PRODUCT_PCF.value: [
        PipelinePhase.PROCESS_EMISSIONS.value,
        PipelinePhase.ENERGY_ANALYSIS.value,
    ],
    PipelinePhase.CIRCULAR_ECONOMY.value: [
        PipelinePhase.QUALITY_ASSURANCE.value,
    ],
    PipelinePhase.WATER_POLLUTION.value: [
        PipelinePhase.QUALITY_ASSURANCE.value,
    ],
    PipelinePhase.BAT_COMPLIANCE.value: [
        PipelinePhase.PROCESS_EMISSIONS.value,
        PipelinePhase.ENERGY_ANALYSIS.value,
        PipelinePhase.WATER_POLLUTION.value,
    ],
    PipelinePhase.SUPPLY_CHAIN.value: [
        PipelinePhase.PRODUCT_PCF.value,
    ],
    PipelinePhase.REPORTING.value: [
        PipelinePhase.PROCESS_EMISSIONS.value,
        PipelinePhase.ENERGY_ANALYSIS.value,
        PipelinePhase.PRODUCT_PCF.value,
        PipelinePhase.CIRCULAR_ECONOMY.value,
        PipelinePhase.WATER_POLLUTION.value,
        PipelinePhase.BAT_COMPLIANCE.value,
        PipelinePhase.SUPPLY_CHAIN.value,
    ],
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class CSRDManufacturingOrchestrator:
    """
    Orchestrate the 11-phase CSRD manufacturing compliance pipeline.

    Each phase delegates to specialised agents (MRV, DATA, FOUND) and
    records provenance hashes so the entire pipeline is auditable.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.config = config or OrchestratorConfig()
        self._agents: Dict[str, Any] = {}
        self._phase_handlers: Dict[str, Callable] = {
            PipelinePhase.INITIALIZATION.value: self._phase_initialization,
            PipelinePhase.DATA_INTAKE.value: self._phase_data_intake,
            PipelinePhase.QUALITY_ASSURANCE.value: self._phase_quality_assurance,
            PipelinePhase.PROCESS_EMISSIONS.value: self._phase_process_emissions,
            PipelinePhase.ENERGY_ANALYSIS.value: self._phase_energy_analysis,
            PipelinePhase.PRODUCT_PCF.value: self._phase_product_pcf,
            PipelinePhase.CIRCULAR_ECONOMY.value: self._phase_circular_economy,
            PipelinePhase.WATER_POLLUTION.value: self._phase_water_pollution,
            PipelinePhase.BAT_COMPLIANCE.value: self._phase_bat_compliance,
            PipelinePhase.SUPPLY_CHAIN.value: self._phase_supply_chain,
            PipelinePhase.REPORTING.value: self._phase_reporting,
        }
        self._results_cache: Dict[str, PhaseResult] = {}
        self._load_agents()

    # -- agent loading -------------------------------------------------------

    def _load_agents(self) -> None:
        """Attempt to import real agents; fall back to stubs."""
        agent_map: Dict[str, str] = {
            "mrv_stationary": (
                "greenlang.agents.mrv.agent_001_stationary_combustion"
            ),
            "mrv_process": (
                "greenlang.agents.mrv.agent_004_process_emissions"
            ),
            "mrv_fugitive": (
                "greenlang.agents.mrv.agent_005_fugitive_emissions"
            ),
            "mrv_scope2_loc": (
                "greenlang.agents.mrv.agent_009_scope2_location"
            ),
            "mrv_scope2_mkt": (
                "greenlang.agents.mrv.agent_010_scope2_market"
            ),
            "mrv_steam": (
                "greenlang.agents.mrv.agent_011_steam_heat"
            ),
            "mrv_pgs": (
                "greenlang.agents.mrv.agent_014_purchased_goods"
            ),
            "mrv_upstream_transport": (
                "greenlang.agents.mrv.agent_017_upstream_transport"
            ),
            "mrv_waste": (
                "greenlang.agents.mrv.agent_018_waste_generated"
            ),
            "data_erp": (
                "greenlang.agents.data.agent_003_erp_connector"
            ),
            "data_excel": (
                "greenlang.agents.data.agent_002_excel_csv"
            ),
            "data_quality": (
                "greenlang.agents.data.agent_010_data_quality_profiler"
            ),
            "data_lineage": (
                "greenlang.agents.data.agent_018_data_lineage"
            ),
            "data_validation": (
                "greenlang.agents.data.agent_019_validation_rule_engine"
            ),
            "found_schema": (
                "greenlang.agents.found.agent_002_schema_compiler"
            ),
            "found_orchestrator": (
                "greenlang.agents.found.agent_001_orchestrator"
            ),
        }
        for key, module_path in agent_map.items():
            try:
                mod = importlib.import_module(module_path)
                self._agents[key] = mod
                logger.info("Loaded agent: %s from %s", key, module_path)
            except ImportError as exc:
                self._agents[key] = _AgentStub(key, str(exc))

    def _get_agent(self, key: str) -> Any:
        """Return agent or stub for the given key."""
        return self._agents.get(key, _AgentStub(key, "not registered"))

    # -- provenance ----------------------------------------------------------

    @staticmethod
    def _compute_hash(data: Any) -> str:
        """SHA-256 truncated to 16 hex chars for provenance tracking."""
        raw = str(data).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _build_provenance(
        self,
        phase_name: str,
        started: float,
        ended: float,
        input_data: Any,
        output_data: Any,
    ) -> PhaseProvenance:
        return PhaseProvenance(
            phase_name=phase_name,
            started_at=started,
            completed_at=ended,
            input_hash=self._compute_hash(input_data),
            output_hash=self._compute_hash(output_data),
        )

    # -- dependency resolution -----------------------------------------------

    def get_phase_dependencies(self) -> Dict[str, List[str]]:
        """Return the full phase dependency graph."""
        return dict(PHASE_DEPENDENCIES)

    def _dependencies_met(self, phase_name: str) -> bool:
        """Check whether all prerequisite phases are satisfied."""
        deps = PHASE_DEPENDENCIES.get(phase_name, [])
        for dep in deps:
            cached = self._results_cache.get(dep)
            if cached is None:
                if not self.config.phases_enabled.get(dep, True):
                    continue
                return False
            if cached.status not in (
                PhaseStatus.COMPLETED,
                PhaseStatus.SKIPPED,
            ):
                return False
        return True

    # -- output validation ---------------------------------------------------

    REQUIRED_OUTPUTS: Dict[str, List[str]] = {
        PipelinePhase.INITIALIZATION.value: [
            "facility_profile", "config_validated",
        ],
        PipelinePhase.DATA_INTAKE.value: [
            "records_loaded", "source_summary",
        ],
        PipelinePhase.QUALITY_ASSURANCE.value: [
            "quality_score", "validated_records",
        ],
        PipelinePhase.PROCESS_EMISSIONS.value: [
            "scope1_total", "by_source",
        ],
        PipelinePhase.ENERGY_ANALYSIS.value: [
            "total_consumption_mwh", "scope2_total",
        ],
        PipelinePhase.PRODUCT_PCF.value: [
            "pcf_per_product",
        ],
        PipelinePhase.CIRCULAR_ECONOMY.value: [
            "recycling_rate", "waste_summary",
        ],
        PipelinePhase.WATER_POLLUTION.value: [
            "water_withdrawal_m3", "pollutant_loads",
        ],
        PipelinePhase.BAT_COMPLIANCE.value: [
            "bat_status", "gaps",
        ],
        PipelinePhase.SUPPLY_CHAIN.value: [
            "scope3_total", "hotspots",
        ],
        PipelinePhase.REPORTING.value: [
            "esrs_disclosures", "report_id",
        ],
    }

    def validate_phase_outputs(
        self, phase: str, outputs: Dict[str, Any]
    ) -> bool:
        """Return True when the phase produced every required output key."""
        expected = self.REQUIRED_OUTPUTS.get(phase, [])
        return all(k in outputs for k in expected)

    # -- single phase execution ----------------------------------------------

    def run_phase(
        self,
        phase_name: str,
        input_data: Dict[str, Any],
    ) -> PhaseResult:
        """Execute a single pipeline phase with retry logic."""
        # Skip disabled phases
        if not self.config.phases_enabled.get(phase_name, True):
            result = PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.SKIPPED,
                provenance_hash=self._compute_hash("skipped"),
            )
            self._results_cache[phase_name] = result
            return result

        # Check dependencies
        if not self._dependencies_met(phase_name):
            deps = PHASE_DEPENDENCIES.get(phase_name, [])
            result = PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                errors=[f"Dependencies not met: {deps}"],
                provenance_hash=self._compute_hash("dep_fail"),
            )
            self._results_cache[phase_name] = result
            return result

        handler = self._phase_handlers.get(phase_name)
        if handler is None:
            result = PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                errors=[f"No handler registered for phase: {phase_name}"],
                provenance_hash=self._compute_hash("no_handler"),
            )
            self._results_cache[phase_name] = result
            return result

        retries = 0
        max_retries = self.config.retry_policy.max_retries
        last_error = ""

        while retries <= max_retries:
            start_time = time.monotonic()
            try:
                outputs = handler(input_data)
                elapsed_ms = int(
                    (time.monotonic() - start_time) * 1000
                )

                prov = self._build_provenance(
                    phase_name,
                    start_time,
                    time.monotonic(),
                    input_data,
                    outputs,
                )

                result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.COMPLETED,
                    duration_ms=elapsed_ms,
                    outputs=outputs,
                    records_processed=outputs.get(
                        "_records_processed", 0
                    ),
                    provenance_hash=prov.output_hash,
                    provenance=prov,
                    retries_used=retries,
                )
                self._results_cache[phase_name] = result
                logger.info(
                    "Phase %s completed in %d ms",
                    phase_name, elapsed_ms,
                )
                return result

            except Exception as exc:
                retries += 1
                last_error = str(exc)
                logger.warning(
                    "Phase %s attempt %d/%d failed: %s",
                    phase_name, retries, max_retries + 1, exc,
                )
                if retries > max_retries:
                    elapsed_ms = int(
                        (time.monotonic() - start_time) * 1000
                    )
                    result = PhaseResult(
                        phase_name=phase_name,
                        status=PhaseStatus.FAILED,
                        duration_ms=elapsed_ms,
                        errors=[last_error],
                        retries_used=retries - 1,
                        provenance_hash=self._compute_hash(last_error),
                    )
                    self._results_cache[phase_name] = result
                    return result
                backoff = self.config.retry_policy.backoff_factor * retries
                time.sleep(backoff)

        # Unreachable but satisfies type checker
        return PhaseResult(
            phase_name=phase_name, status=PhaseStatus.FAILED
        )

    # -- full pipeline -------------------------------------------------------

    def _topological_order(self) -> List[str]:
        """Return phases in execution order honouring dependencies."""
        visited: set = set()
        order: List[str] = []

        def _visit(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            for dep in PHASE_DEPENDENCIES.get(node, []):
                _visit(dep)
            order.append(node)

        for phase in PipelinePhase:
            _visit(phase.value)
        return order

    def run_pipeline(
        self, input_data: Dict[str, Any]
    ) -> PipelineResult:
        """
        Execute the full 11-phase manufacturing pipeline.

        Phases execute in topological order.  Disabled phases are skipped.
        Failed phases are retried according to the retry policy.  Each
        phase's outputs are merged into the cumulative input for subsequent
        phases so downstream phases have access to upstream results.
        """
        self._results_cache.clear()
        pipeline_start = time.monotonic()

        execution_order = self._topological_order()
        enabled_phases = [
            p for p in execution_order
            if self.config.phases_enabled.get(p, True)
        ]

        completed = 0
        failed = 0
        skipped = 0

        for phase_name in execution_order:
            if phase_name not in enabled_phases:
                self._results_cache[phase_name] = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=self._compute_hash("skipped"),
                )
                skipped += 1
                continue

            result = self.run_phase(phase_name, input_data)
            if result.status == PhaseStatus.COMPLETED:
                completed += 1
                # Merge outputs into cumulative context
                input_data = {**input_data, **result.outputs}
            elif result.status == PhaseStatus.SKIPPED:
                skipped += 1
            else:
                failed += 1

        total_ms = int((time.monotonic() - pipeline_start) * 1000)
        pipeline_hash = self._compute_hash(
            {k: v.provenance_hash for k, v in self._results_cache.items()}
        )
        overall_status = (
            PhaseStatus.COMPLETED if failed == 0 else PhaseStatus.FAILED
        )

        return PipelineResult(
            total_phases=len(execution_order),
            completed=completed,
            failed=failed,
            skipped=skipped,
            results=dict(self._results_cache),
            total_duration_ms=total_ms,
            provenance_hash=pipeline_hash,
            pipeline_status=overall_status,
            facility_id=self.config.facility_id,
            reporting_year=self.config.reporting_year,
        )

    # -----------------------------------------------------------------------
    # Phase implementations
    # -----------------------------------------------------------------------

    def _phase_initialization(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 1: Validate config, load facility profile, check agents."""
        facility_id = input_data.get(
            "facility_id", self.config.facility_id
        )
        sub_sector = input_data.get(
            "sub_sector", self.config.sub_sector
        )
        agents_available = {
            k: not isinstance(v, _AgentStub)
            for k, v in self._agents.items()
        }
        return {
            "facility_profile": {
                "facility_id": facility_id,
                "sub_sector": sub_sector,
                "reporting_year": self.config.reporting_year,
            },
            "config_validated": True,
            "agents_available": agents_available,
            "_records_processed": 0,
        }

    def _phase_data_intake(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 2: Ingest ERP exports, production logs, utility bills."""
        erp_agent = self._get_agent("data_erp")
        excel_agent = self._get_agent("data_excel")

        erp_result = erp_agent.ingest(
            input_data.get("erp_export", {})
        )
        excel_result = excel_agent.ingest(
            input_data.get("production_files", [])
        )

        record_count = (
            erp_result.get("record_count", 0)
            + excel_result.get("record_count", 0)
        )
        return {
            "records_loaded": record_count,
            "source_summary": {
                "erp": erp_result,
                "production_files": excel_result,
            },
            "raw_data": {
                **erp_result.get("data", {}),
                **excel_result.get("data", {}),
            },
            "_records_processed": record_count,
        }

    def _phase_quality_assurance(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 3: Profile, validate, and score data quality."""
        quality_agent = self._get_agent("data_quality")
        validation_agent = self._get_agent("data_validation")

        raw_data = input_data.get("raw_data", {})
        profile = quality_agent.profile(raw_data)
        validation = validation_agent.validate(
            raw_data, input_data.get("validation_rules", {})
        )

        quality_score = profile.get("quality_score", 0.0)
        validated_records = validation.get("valid_records", [])

        return {
            "quality_score": quality_score,
            "validated_records": validated_records,
            "quality_profile": profile,
            "validation_report": validation,
            "_records_processed": len(validated_records),
        }

    def _phase_process_emissions(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 4: Calculate Scope 1 process emissions."""
        stationary_agent = self._get_agent("mrv_stationary")
        process_agent = self._get_agent("mrv_process")
        fugitive_agent = self._get_agent("mrv_fugitive")

        validated = input_data.get("validated_records", [])
        facility = input_data.get("facility_profile", {})

        combustion = stationary_agent.calculate(validated, facility)
        process = process_agent.calculate(validated, facility)
        fugitive = fugitive_agent.calculate(validated, facility)

        scope1_sources = {
            "stationary_combustion": combustion.get("total_tco2e", 0.0),
            "process_emissions": process.get("total_tco2e", 0.0),
            "fugitive_emissions": fugitive.get("total_tco2e", 0.0),
        }
        scope1_total = sum(scope1_sources.values())

        return {
            "scope1_total": scope1_total,
            "by_source": scope1_sources,
            "combustion_detail": combustion,
            "process_detail": process,
            "fugitive_detail": fugitive,
            "_records_processed": len(validated),
        }

    def _phase_energy_analysis(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 5: Energy consumption and Scope 2 emissions."""
        scope2_loc = self._get_agent("mrv_scope2_loc")
        scope2_mkt = self._get_agent("mrv_scope2_mkt")
        steam_agent = self._get_agent("mrv_steam")

        validated = input_data.get("validated_records", [])
        facility = input_data.get("facility_profile", {})

        loc_result = scope2_loc.calculate(validated, facility)
        mkt_result = scope2_mkt.calculate(validated, facility)
        steam_result = steam_agent.calculate(validated, facility)

        total_mwh = (
            loc_result.get("total_consumption_mwh", 0.0)
            + steam_result.get("total_consumption_mwh", 0.0)
        )
        scope2_total = (
            loc_result.get("total_tco2e", 0.0)
            + steam_result.get("total_tco2e", 0.0)
        )
        production_volume = max(
            input_data.get("production_volume", 1), 1
        )

        return {
            "total_consumption_mwh": total_mwh,
            "scope2_total": scope2_total,
            "scope2_location": loc_result,
            "scope2_market": mkt_result,
            "steam_heat": steam_result,
            "energy_intensity": total_mwh / production_volume,
            "_records_processed": len(validated),
        }

    def _phase_product_pcf(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 6: Calculate Product Carbon Footprints."""
        scope1 = input_data.get("scope1_total", 0.0)
        scope2 = input_data.get("scope2_total", 0.0)
        production_volume = max(
            input_data.get("production_volume", 1), 1
        )
        products = input_data.get(
            "product_list", ["default_product"]
        )

        pcf_per_product: Dict[str, float] = {}
        for product in products:
            allocation = input_data.get(
                "allocation_factors", {}
            ).get(product, 1.0 / max(len(products), 1))
            pcf_per_product[product] = (
                (scope1 + scope2) * allocation / production_volume
            )

        return {
            "pcf_per_product": pcf_per_product,
            "total_embedded_emissions": scope1 + scope2,
            "allocation_method": input_data.get(
                "allocation_method", "mass"
            ),
            "_records_processed": len(products),
        }

    def _phase_circular_economy(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 7: Circular economy metrics (ESRS E5)."""
        validated = input_data.get("validated_records", [])

        waste_total = sum(
            r.get("waste_kg", 0)
            for r in validated if isinstance(r, dict)
        )
        waste_recycled = sum(
            r.get("waste_recycled_kg", 0)
            for r in validated if isinstance(r, dict)
        )
        recycling_rate = (
            (waste_recycled / waste_total * 100)
            if waste_total > 0 else 0.0
        )

        return {
            "recycling_rate": round(recycling_rate, 2),
            "waste_summary": {
                "total_waste_kg": waste_total,
                "recycled_kg": waste_recycled,
                "landfill_kg": waste_total - waste_recycled,
                "recycling_rate_pct": round(recycling_rate, 2),
            },
            "circular_material_use_rate": round(
                recycling_rate * 0.85, 2
            ),
            "_records_processed": len(validated),
        }

    def _phase_water_pollution(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 8: Water withdrawal, discharge, pollutant loads (ESRS E3)."""
        validated = input_data.get("validated_records", [])

        water_in = sum(
            r.get("water_withdrawal_m3", 0)
            for r in validated if isinstance(r, dict)
        )
        water_out = sum(
            r.get("water_discharge_m3", 0)
            for r in validated if isinstance(r, dict)
        )

        pollutant_loads: Dict[str, float] = {}
        for rec in validated:
            if isinstance(rec, dict):
                for pollutant, load in rec.get("pollutants", {}).items():
                    pollutant_loads[pollutant] = (
                        pollutant_loads.get(pollutant, 0) + load
                    )

        production_volume = max(
            input_data.get("production_volume", 1), 1
        )
        return {
            "water_withdrawal_m3": water_in,
            "water_discharge_m3": water_out,
            "water_consumption_m3": water_in - water_out,
            "pollutant_loads": pollutant_loads,
            "water_intensity": water_in / production_volume,
            "_records_processed": len(validated),
        }

    def _phase_bat_compliance(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 9: Best Available Techniques compliance check."""
        energy_intensity = input_data.get("energy_intensity", 0.0)
        water_intensity = input_data.get("water_intensity", 0.0)
        sub_sector = input_data.get(
            "facility_profile", {}
        ).get("sub_sector", "general")

        bat_benchmarks = self._get_bat_benchmarks(sub_sector)
        gaps: List[Dict[str, Any]] = []

        upper_energy = bat_benchmarks.get(
            "energy_intensity_upper", float("inf")
        )
        if energy_intensity > upper_energy:
            gaps.append({
                "metric": "energy_intensity",
                "actual": energy_intensity,
                "benchmark": upper_energy,
                "severity": "high",
            })

        upper_water = bat_benchmarks.get(
            "water_intensity_upper", float("inf")
        )
        if water_intensity > upper_water:
            gaps.append({
                "metric": "water_intensity",
                "actual": water_intensity,
                "benchmark": upper_water,
                "severity": "medium",
            })

        bat_status = (
            "compliant" if len(gaps) == 0 else "non_compliant"
        )

        return {
            "bat_status": bat_status,
            "gaps": gaps,
            "benchmarks_used": bat_benchmarks,
            "sub_sector": sub_sector,
            "_records_processed": 1,
        }

    @staticmethod
    def _get_bat_benchmarks(sub_sector: str) -> Dict[str, float]:
        """BAT-AEL benchmarks by manufacturing sub-sector."""
        benchmarks: Dict[str, Dict[str, float]] = {
            "cement": {
                "energy_intensity_upper": 4.5,
                "water_intensity_upper": 0.5,
            },
            "steel": {
                "energy_intensity_upper": 6.0,
                "water_intensity_upper": 3.0,
            },
            "chemicals": {
                "energy_intensity_upper": 3.0,
                "water_intensity_upper": 8.0,
            },
            "paper_pulp": {
                "energy_intensity_upper": 5.0,
                "water_intensity_upper": 50.0,
            },
            "glass": {
                "energy_intensity_upper": 4.0,
                "water_intensity_upper": 1.0,
            },
            "ceramics": {
                "energy_intensity_upper": 3.5,
                "water_intensity_upper": 1.5,
            },
            "food_beverage": {
                "energy_intensity_upper": 2.0,
                "water_intensity_upper": 10.0,
            },
            "automotive": {
                "energy_intensity_upper": 2.5,
                "water_intensity_upper": 5.0,
            },
            "electronics": {
                "energy_intensity_upper": 1.5,
                "water_intensity_upper": 4.0,
            },
            "textiles": {
                "energy_intensity_upper": 2.0,
                "water_intensity_upper": 100.0,
            },
            "general": {
                "energy_intensity_upper": 5.0,
                "water_intensity_upper": 10.0,
            },
        }
        return benchmarks.get(sub_sector, benchmarks["general"])

    def _phase_supply_chain(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 10: Scope 3 supply chain analysis."""
        pgs_agent = self._get_agent("mrv_pgs")
        transport_agent = self._get_agent("mrv_upstream_transport")
        waste_agent = self._get_agent("mrv_waste")

        supply_data = input_data.get("supply_chain_data", {})
        facility = input_data.get("facility_profile", {})

        pgs_result = pgs_agent.calculate(supply_data, facility)
        transport_result = transport_agent.calculate(
            supply_data, facility
        )
        waste_result = waste_agent.calculate(supply_data, facility)

        scope3_categories = {
            "cat1_purchased_goods": pgs_result.get(
                "total_tco2e", 0.0
            ),
            "cat4_upstream_transport": transport_result.get(
                "total_tco2e", 0.0
            ),
            "cat5_waste": waste_result.get("total_tco2e", 0.0),
        }
        scope3_total = sum(scope3_categories.values())

        hotspots = sorted(
            scope3_categories.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "scope3_total": scope3_total,
            "scope3_categories": scope3_categories,
            "hotspots": [
                {"category": cat, "tco2e": val}
                for cat, val in hotspots[:5]
            ],
            "_records_processed": 1,
        }

    def _phase_reporting(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phase 11: Compile ESRS disclosures and generate report."""
        fac_id = self.config.facility_id or "UNKNOWN"
        report_id = f"MFG-{fac_id}-{self.config.reporting_year}"

        esrs_disclosures = {
            "E1_climate": {
                "scope1_total": input_data.get("scope1_total", 0.0),
                "scope2_total": input_data.get("scope2_total", 0.0),
                "scope3_total": input_data.get("scope3_total", 0.0),
                "energy_consumption_mwh": input_data.get(
                    "total_consumption_mwh", 0.0
                ),
                "energy_intensity": input_data.get(
                    "energy_intensity", 0.0
                ),
                "pcf_per_product": input_data.get(
                    "pcf_per_product", {}
                ),
            },
            "E3_water": {
                "water_withdrawal_m3": input_data.get(
                    "water_withdrawal_m3", 0.0
                ),
                "water_consumption_m3": input_data.get(
                    "water_consumption_m3", 0.0
                ),
                "pollutant_loads": input_data.get(
                    "pollutant_loads", {}
                ),
            },
            "E5_circular_economy": {
                "recycling_rate": input_data.get(
                    "recycling_rate", 0.0
                ),
                "waste_summary": input_data.get("waste_summary", {}),
            },
            "bat_compliance": {
                "bat_status": input_data.get("bat_status", "unknown"),
                "gaps": input_data.get("gaps", []),
            },
        }

        return {
            "esrs_disclosures": esrs_disclosures,
            "report_id": report_id,
            "reporting_year": self.config.reporting_year,
            "data_quality_score": input_data.get(
                "quality_score", 0.0
            ),
            "_records_processed": 1,
        }
