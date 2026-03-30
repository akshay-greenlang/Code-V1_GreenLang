"""
PACK-013 CSRD Manufacturing Pack - Health Check.

22-category health verification covering every component required for a
fully functional CSRD manufacturing compliance deployment.  Each check
category validates module imports, configuration, data availability,
and connectivity.
"""

import hashlib
import importlib
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from greenlang.schemas.enums import HealthStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HealthCategory(str, Enum):
    """The 22 health check categories."""
    ENGINES = "engines"
    WORKFLOWS = "workflows"
    TEMPLATES = "templates"
    CONFIG = "config"
    PRESETS = "presets"
    INTEGRATIONS = "integrations"
    MRV_AGENTS = "mrv_agents"
    DATA_AGENTS = "data_agents"
    FOUND_AGENTS = "found_agents"
    INDUSTRIAL_AGENTS = "industrial_agents"
    CBAM = "cbam"
    ETS = "ets"
    TAXONOMY = "taxonomy"
    SUPPLY_CHAIN = "supply_chain"
    PROCESS_EMISSIONS = "process_emissions"
    ENERGY = "energy"
    PCF = "pcf"
    CIRCULAR = "circular"
    WATER = "water"
    BAT = "bat"
    DEMO = "demo"
    PROVENANCE = "provenance"


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class CheckDetail(BaseModel):
    """A single check result within a category."""
    name: str
    passed: bool = Field(default=False)
    message: str = Field(default="")
    duration_ms: int = Field(default=0, ge=0)


class CategoryResult(BaseModel):
    """Result of checking a single category."""
    category: str
    status: HealthStatus = Field(default=HealthStatus.UNKNOWN)
    checks_passed: int = Field(default=0, ge=0)
    checks_failed: int = Field(default=0, ge=0)
    checks_total: int = Field(default=0, ge=0)
    details: List[CheckDetail] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class HealthCheckResult(BaseModel):
    """Full health check result across all 22 categories."""
    overall_status: HealthStatus = Field(default=HealthStatus.UNKNOWN)
    categories: Dict[str, CategoryResult] = Field(default_factory=dict)
    total_checks: int = Field(default=0, ge=0)
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    warnings: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)
    provenance_hash: str = Field(default="")
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class ManufacturingHealthCheck:
    """
    Run 22-category health verification for the CSRD Manufacturing Pack.

    Validates that all engines, agents, integrations, templates, and
    configuration are present and functional.
    """

    def __init__(self) -> None:
        self._category_methods: Dict[str, Any] = {
            HealthCategory.ENGINES: self.check_engines,
            HealthCategory.WORKFLOWS: self.check_workflows,
            HealthCategory.TEMPLATES: self.check_templates,
            HealthCategory.CONFIG: self.check_config,
            HealthCategory.PRESETS: self.check_presets,
            HealthCategory.INTEGRATIONS: self.check_integrations,
            HealthCategory.MRV_AGENTS: self.check_mrv_agents,
            HealthCategory.DATA_AGENTS: self.check_data_agents,
            HealthCategory.FOUND_AGENTS: self.check_found_agents,
            HealthCategory.INDUSTRIAL_AGENTS: self.check_industrial_agents,
            HealthCategory.CBAM: self.check_cbam,
            HealthCategory.ETS: self.check_ets,
            HealthCategory.TAXONOMY: self.check_taxonomy,
            HealthCategory.SUPPLY_CHAIN: self.check_supply_chain,
            HealthCategory.PROCESS_EMISSIONS: self.check_process_emissions,
            HealthCategory.ENERGY: self.check_energy,
            HealthCategory.PCF: self.check_pcf,
            HealthCategory.CIRCULAR: self.check_circular,
            HealthCategory.WATER: self.check_water,
            HealthCategory.BAT: self.check_bat,
            HealthCategory.DEMO: self.check_demo,
            HealthCategory.PROVENANCE: self.check_provenance,
        }

    @staticmethod
    def _compute_hash(data: Any) -> str:
        raw = str(data).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _try_import(module_path: str) -> Tuple[bool, str]:
        """Try to import a module; return (success, message)."""
        try:
            importlib.import_module(module_path)
            return True, f"{module_path} imported successfully"
        except ImportError as exc:
            return False, f"{module_path} import failed: {exc}"
        except Exception as exc:
            return False, f"{module_path} error: {exc}"

    def _run_checks(
        self, category: str, checks: List[Tuple[str, bool, str]]
    ) -> CategoryResult:
        """Build a CategoryResult from a list of (name, passed, msg) tuples."""
        details: List[CheckDetail] = []
        warnings: List[str] = []
        for name, passed, msg in checks:
            details.append(CheckDetail(
                name=name, passed=passed, message=msg
            ))
            if not passed:
                warnings.append(f"{category}/{name}: {msg}")

        checks_passed = sum(1 for d in details if d.passed)
        checks_failed = sum(1 for d in details if not d.passed)

        if checks_failed == 0:
            status = HealthStatus.HEALTHY
        elif checks_passed > checks_failed:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        return CategoryResult(
            category=category,
            status=status,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            checks_total=len(details),
            details=details,
            warnings=warnings,
        )

    # -- public API ----------------------------------------------------------

    def run_full_check(self) -> HealthCheckResult:
        """Execute all 22 category checks and return aggregate result."""
        start = time.monotonic()
        categories: Dict[str, CategoryResult] = {}
        total_passed = 0
        total_failed = 0
        total_warnings = 0

        for cat, method in self._category_methods.items():
            cat_start = time.monotonic()
            result = method()
            cat_ms = int((time.monotonic() - cat_start) * 1000)
            # Attach timing to each detail (approximate)
            for d in result.details:
                d.duration_ms = cat_ms // max(len(result.details), 1)

            categories[cat.value] = result
            total_passed += result.checks_passed
            total_failed += result.checks_failed
            total_warnings += len(result.warnings)

        total_ms = int((time.monotonic() - start) * 1000)

        if total_failed == 0:
            overall = HealthStatus.HEALTHY
        elif total_passed > total_failed:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.UNHEALTHY

        data = {
            "passed": total_passed,
            "failed": total_failed,
            "ts": time.time(),
        }

        return HealthCheckResult(
            overall_status=overall,
            categories=categories,
            total_checks=total_passed + total_failed,
            passed=total_passed,
            failed=total_failed,
            warnings=total_warnings,
            duration_ms=total_ms,
            provenance_hash=self._compute_hash(data),
        )

    def check_category(
        self, category: HealthCategory
    ) -> CategoryResult:
        """Run checks for a single category."""
        method = self._category_methods.get(category)
        if method is None:
            return CategoryResult(
                category=category.value,
                status=HealthStatus.UNKNOWN,
            )
        return method()

    # -----------------------------------------------------------------------
    # Category check implementations
    # -----------------------------------------------------------------------

    def check_engines(self) -> CategoryResult:
        """Verify calculation engines are importable."""
        checks: List[Tuple[str, bool, str]] = []
        engine_modules = [
            ("process_emission_engine", "packs.eu_compliance.PACK_013_csrd_manufacturing.engines.process_emission_engine"),
            ("energy_analysis_engine", "packs.eu_compliance.PACK_013_csrd_manufacturing.engines.energy_analysis_engine"),
            ("pcf_engine", "packs.eu_compliance.PACK_013_csrd_manufacturing.engines.pcf_engine"),
            ("water_engine", "packs.eu_compliance.PACK_013_csrd_manufacturing.engines.water_engine"),
            ("circular_economy_engine", "packs.eu_compliance.PACK_013_csrd_manufacturing.engines.circular_economy_engine"),
            ("bat_engine", "packs.eu_compliance.PACK_013_csrd_manufacturing.engines.bat_engine"),
            ("supply_chain_engine", "packs.eu_compliance.PACK_013_csrd_manufacturing.engines.supply_chain_engine"),
        ]
        for name, mod in engine_modules:
            ok, msg = self._try_import(mod)
            checks.append((name, ok, msg))
        return self._run_checks(HealthCategory.ENGINES.value, checks)

    def check_workflows(self) -> CategoryResult:
        """Verify workflow definitions exist."""
        checks: List[Tuple[str, bool, str]] = []
        workflows = [
            "annual_reporting", "quarterly_review", "data_collection",
            "target_setting", "bat_assessment",
        ]
        for wf in workflows:
            mod = f"packs.eu_compliance.PACK_013_csrd_manufacturing.workflows.{wf}"
            ok, msg = self._try_import(mod)
            checks.append((wf, ok, msg))
        return self._run_checks(HealthCategory.WORKFLOWS.value, checks)

    def check_templates(self) -> CategoryResult:
        """Verify ESRS disclosure templates."""
        checks: List[Tuple[str, bool, str]] = []
        templates = ["e1_climate", "e2_pollution", "e3_water", "e5_circular"]
        for tpl in templates:
            mod = f"packs.eu_compliance.PACK_013_csrd_manufacturing.templates.{tpl}"
            ok, msg = self._try_import(mod)
            checks.append((tpl, ok, msg))
        return self._run_checks(HealthCategory.TEMPLATES.value, checks)

    def check_config(self) -> CategoryResult:
        """Verify configuration modules."""
        checks: List[Tuple[str, bool, str]] = []
        config_modules = [
            "pack_config", "sub_sector_config", "emission_factors",
        ]
        for cfg in config_modules:
            mod = f"packs.eu_compliance.PACK_013_csrd_manufacturing.config.{cfg}"
            ok, msg = self._try_import(mod)
            checks.append((cfg, ok, msg))
        return self._run_checks(HealthCategory.CONFIG.value, checks)

    def check_presets(self) -> CategoryResult:
        """Verify sub-sector presets."""
        checks: List[Tuple[str, bool, str]] = []
        presets = [
            "cement", "steel", "chemicals", "automotive",
            "food_beverage", "electronics", "paper_pulp",
        ]
        for preset in presets:
            mod = f"packs.eu_compliance.PACK_013_csrd_manufacturing.presets.{preset}"
            ok, msg = self._try_import(mod)
            checks.append((preset, ok, msg))
        return self._run_checks(HealthCategory.PRESETS.value, checks)

    def check_integrations(self) -> CategoryResult:
        """Verify this integrations package itself."""
        base = "packs.eu_compliance.PACK_013_csrd_manufacturing.integrations"
        checks: List[Tuple[str, bool, str]] = []
        modules = [
            "pack_orchestrator", "csrd_pack_bridge", "cbam_pack_bridge",
            "mrv_industrial_bridge", "data_manufacturing_bridge",
            "eu_ets_bridge", "taxonomy_bridge", "health_check",
            "setup_wizard",
        ]
        for mod_name in modules:
            ok, msg = self._try_import(f"{base}.{mod_name}")
            checks.append((mod_name, ok, msg))
        return self._run_checks(HealthCategory.INTEGRATIONS.value, checks)

    def check_mrv_agents(self) -> CategoryResult:
        """Verify MRV agent availability."""
        checks: List[Tuple[str, bool, str]] = []
        for i in range(1, 31):
            agent_id = f"MRV-{i:03d}"
            mod = f"greenlang.agents.mrv.agent_{i:03d}"
            ok, msg = self._try_import(mod)
            checks.append((agent_id, ok, msg))
        return self._run_checks(HealthCategory.MRV_AGENTS.value, checks)

    def check_data_agents(self) -> CategoryResult:
        """Verify DATA agent availability."""
        checks: List[Tuple[str, bool, str]] = []
        for i in range(1, 21):
            agent_id = f"DATA-{i:03d}"
            mod = f"greenlang.agents.data.agent_{i:03d}"
            ok, msg = self._try_import(mod)
            checks.append((agent_id, ok, msg))
        return self._run_checks(HealthCategory.DATA_AGENTS.value, checks)

    def check_found_agents(self) -> CategoryResult:
        """Verify FOUND agent availability."""
        checks: List[Tuple[str, bool, str]] = []
        for i in range(1, 11):
            agent_id = f"FOUND-{i:03d}"
            mod = f"greenlang.agents.found.agent_{i:03d}"
            ok, msg = self._try_import(mod)
            checks.append((agent_id, ok, msg))
        return self._run_checks(HealthCategory.FOUND_AGENTS.value, checks)

    def check_industrial_agents(self) -> CategoryResult:
        """Verify manufacturing-specific agent modules."""
        checks: List[Tuple[str, bool, str]] = []
        industrial = [
            ("process_emissions", "greenlang.agents.mrv.agent_004_process_emissions"),
            ("fugitive_emissions", "greenlang.agents.mrv.agent_005_fugitive_emissions"),
            ("stationary_combustion", "greenlang.agents.mrv.agent_001_stationary_combustion"),
            ("refrigerants", "greenlang.agents.mrv.agent_002_refrigerants_fgas"),
        ]
        for name, mod in industrial:
            ok, msg = self._try_import(mod)
            checks.append((name, ok, msg))
        return self._run_checks(
            HealthCategory.INDUSTRIAL_AGENTS.value, checks
        )

    def check_cbam(self) -> CategoryResult:
        """Verify CBAM pack bridge and dependencies."""
        checks: List[Tuple[str, bool, str]] = []

        ok, msg = self._try_import(
            "packs.eu_compliance.PACK_013_csrd_manufacturing."
            "integrations.cbam_pack_bridge"
        )
        checks.append(("cbam_bridge_import", ok, msg))

        try:
            from packs.eu_compliance.PACK_013_csrd_manufacturing.integrations.cbam_pack_bridge import (
                CBAMPackBridge,
                CBAMBridgeConfig,
            )
            bridge = CBAMPackBridge(CBAMBridgeConfig())
            checks.append(("cbam_bridge_init", True, "CBAMPackBridge initialised"))
        except Exception as exc:
            checks.append(("cbam_bridge_init", False, str(exc)))

        pack_ok, pack_msg = self._try_import("packs.eu_compliance.PACK_004_cbam_readiness")
        checks.append(("cbam_readiness_pack", pack_ok, pack_msg))

        pack_ok2, pack_msg2 = self._try_import("packs.eu_compliance.PACK_005_cbam_complete")
        checks.append(("cbam_complete_pack", pack_ok2, pack_msg2))

        return self._run_checks(HealthCategory.CBAM.value, checks)

    def check_ets(self) -> CategoryResult:
        """Verify EU ETS bridge."""
        checks: List[Tuple[str, bool, str]] = []
        ok, msg = self._try_import(
            "packs.eu_compliance.PACK_013_csrd_manufacturing."
            "integrations.eu_ets_bridge"
        )
        checks.append(("ets_bridge_import", ok, msg))

        try:
            from packs.eu_compliance.PACK_013_csrd_manufacturing.integrations.eu_ets_bridge import (
                EUETSBridge,
                ETSBridgeConfig,
            )
            bridge = EUETSBridge(ETSBridgeConfig())
            result = bridge.calculate_ets_obligation(5000.0)
            checks.append((
                "ets_calculation",
                result.verified_emissions == 5000.0,
                f"Verified: {result.verified_emissions}",
            ))
        except Exception as exc:
            checks.append(("ets_calculation", False, str(exc)))

        return self._run_checks(HealthCategory.ETS.value, checks)

    def check_taxonomy(self) -> CategoryResult:
        """Verify EU Taxonomy bridge."""
        checks: List[Tuple[str, bool, str]] = []
        ok, msg = self._try_import(
            "packs.eu_compliance.PACK_013_csrd_manufacturing."
            "integrations.taxonomy_bridge"
        )
        checks.append(("taxonomy_bridge_import", ok, msg))

        try:
            from packs.eu_compliance.PACK_013_csrd_manufacturing.integrations.taxonomy_bridge import (
                TaxonomyBridge,
                TaxonomyBridgeConfig,
            )
            bridge = TaxonomyBridge(TaxonomyBridgeConfig())
            result = bridge.assess_alignment([])
            checks.append((
                "taxonomy_assessment",
                result.eligible_pct == 0.0,
                "Empty assessment returned correctly",
            ))
        except Exception as exc:
            checks.append(("taxonomy_assessment", False, str(exc)))

        return self._run_checks(HealthCategory.TAXONOMY.value, checks)

    def check_supply_chain(self) -> CategoryResult:
        """Verify supply chain Scope 3 components."""
        checks: List[Tuple[str, bool, str]] = []
        scope3_agents = [
            ("cat1_pgs", "greenlang.agents.mrv.agent_014_purchased_goods"),
            ("cat4_transport", "greenlang.agents.mrv.agent_017_upstream_transport"),
            ("cat5_waste", "greenlang.agents.mrv.agent_018_waste_generated"),
            ("cat10_processing", "greenlang.agents.mrv.agent_023_processing_sold"),
            ("cat12_eol", "greenlang.agents.mrv.agent_025_end_of_life"),
        ]
        for name, mod in scope3_agents:
            ok, msg = self._try_import(mod)
            checks.append((name, ok, msg))
        return self._run_checks(
            HealthCategory.SUPPLY_CHAIN.value, checks
        )

    def check_process_emissions(self) -> CategoryResult:
        """Verify process emissions calculation capability."""
        checks: List[Tuple[str, bool, str]] = []
        modules = [
            ("stationary", "greenlang.agents.mrv.agent_001_stationary_combustion"),
            ("process", "greenlang.agents.mrv.agent_004_process_emissions"),
            ("fugitive", "greenlang.agents.mrv.agent_005_fugitive_emissions"),
        ]
        for name, mod in modules:
            ok, msg = self._try_import(mod)
            checks.append((name, ok, msg))
        return self._run_checks(
            HealthCategory.PROCESS_EMISSIONS.value, checks
        )

    def check_energy(self) -> CategoryResult:
        """Verify energy analysis components."""
        checks: List[Tuple[str, bool, str]] = []
        modules = [
            ("scope2_location", "greenlang.agents.mrv.agent_009_scope2_location"),
            ("scope2_market", "greenlang.agents.mrv.agent_010_scope2_market"),
            ("steam_heat", "greenlang.agents.mrv.agent_011_steam_heat"),
            ("cooling", "greenlang.agents.mrv.agent_012_cooling"),
        ]
        for name, mod in modules:
            ok, msg = self._try_import(mod)
            checks.append((name, ok, msg))
        return self._run_checks(HealthCategory.ENERGY.value, checks)

    def check_pcf(self) -> CategoryResult:
        """Verify Product Carbon Footprint capability."""
        checks: List[Tuple[str, bool, str]] = []
        modules = [
            ("pcf_engine", "packs.eu_compliance.PACK_013_csrd_manufacturing.engines.pcf_engine"),
            ("allocation_model", "packs.eu_compliance.PACK_013_csrd_manufacturing.engines.allocation_model"),
        ]
        for name, mod in modules:
            ok, msg = self._try_import(mod)
            checks.append((name, ok, msg))
        return self._run_checks(HealthCategory.PCF.value, checks)

    def check_circular(self) -> CategoryResult:
        """Verify circular economy / waste management components."""
        checks: List[Tuple[str, bool, str]] = []
        modules = [
            ("waste_treatment", "greenlang.agents.mrv.agent_007_waste_treatment"),
            ("circular_engine", "packs.eu_compliance.PACK_013_csrd_manufacturing.engines.circular_economy_engine"),
        ]
        for name, mod in modules:
            ok, msg = self._try_import(mod)
            checks.append((name, ok, msg))
        return self._run_checks(HealthCategory.CIRCULAR.value, checks)

    def check_water(self) -> CategoryResult:
        """Verify water and pollution components."""
        checks: List[Tuple[str, bool, str]] = []
        modules = [
            ("water_engine", "packs.eu_compliance.PACK_013_csrd_manufacturing.engines.water_engine"),
            ("pollutant_registry", "packs.eu_compliance.PACK_013_csrd_manufacturing.config.pollutant_registry"),
        ]
        for name, mod in modules:
            ok, msg = self._try_import(mod)
            checks.append((name, ok, msg))
        return self._run_checks(HealthCategory.WATER.value, checks)

    def check_bat(self) -> CategoryResult:
        """Verify BAT compliance components."""
        checks: List[Tuple[str, bool, str]] = []
        modules = [
            ("bat_engine", "packs.eu_compliance.PACK_013_csrd_manufacturing.engines.bat_engine"),
            ("bat_benchmarks", "packs.eu_compliance.PACK_013_csrd_manufacturing.config.bat_benchmarks"),
        ]
        for name, mod in modules:
            ok, msg = self._try_import(mod)
            checks.append((name, ok, msg))
        return self._run_checks(HealthCategory.BAT.value, checks)

    def check_demo(self) -> CategoryResult:
        """Verify demo / sample data availability."""
        checks: List[Tuple[str, bool, str]] = []
        demo_items = [
            ("demo_facility", "packs.eu_compliance.PACK_013_csrd_manufacturing.demo.sample_facility"),
            ("demo_data", "packs.eu_compliance.PACK_013_csrd_manufacturing.demo.sample_data"),
        ]
        for name, mod in demo_items:
            ok, msg = self._try_import(mod)
            checks.append((name, ok, msg))
        return self._run_checks(HealthCategory.DEMO.value, checks)

    def check_provenance(self) -> CategoryResult:
        """Verify provenance tracking infrastructure."""
        checks: List[Tuple[str, bool, str]] = []

        # Check hashlib availability
        try:
            h = hashlib.sha256(b"test").hexdigest()
            checks.append(("sha256", True, f"SHA-256 available: {h[:8]}..."))
        except Exception as exc:
            checks.append(("sha256", False, str(exc)))

        # Check orchestrator provenance
        try:
            from packs.eu_compliance.PACK_013_csrd_manufacturing.integrations.pack_orchestrator import (
                CSRDManufacturingOrchestrator,
            )
            orch = CSRDManufacturingOrchestrator()
            test_hash = orch._compute_hash({"test": True})
            checks.append((
                "orchestrator_hash",
                len(test_hash) == 16,
                f"Hash: {test_hash}",
            ))
        except Exception as exc:
            checks.append(("orchestrator_hash", False, str(exc)))

        return self._run_checks(HealthCategory.PROVENANCE.value, checks)
