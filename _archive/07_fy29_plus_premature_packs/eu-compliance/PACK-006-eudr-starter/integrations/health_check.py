# -*- coding: utf-8 -*-
"""
EUDRStarterHealthCheck - 14-Category Health Verification for EUDR Starter Pack
================================================================================

This module implements a comprehensive 14-category health check system that
validates the operational readiness of the EUDR Starter Pack. Each category
verifies a specific subsystem (engines, bridges, configuration) and returns
a detailed health status with remediation guidance.

Check Categories:
    1. Configuration Validity
    2. DDS Assembly Engine
    3. Geolocation Engine
    4. Risk Scoring Engine
    5. Commodity Classification Engine
    6. Supplier Compliance Engine
    7. Cutoff Date Engine
    8. Policy Compliance Engine
    9. EUDR App Bridge
    10. Traceability Bridge
    11. Satellite Bridge
    12. GIS Bridge
    13. EU IS Bridge
    14. Demo Data Availability

Example:
    >>> health = EUDRStarterHealthCheck()
    >>> result = await health.check_all()
    >>> for cat in result.category_results:
    ...     print(f"{cat.category}: {cat.status}")

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import importlib
import logging
import os
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field
from greenlang.schemas.enums import HealthStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class HealthCategory(str, Enum):
    """The 14 health check categories for EUDR Starter Pack."""
    CONFIGURATION_VALIDITY = "configuration_validity"
    DDS_ASSEMBLY_ENGINE = "dds_assembly_engine"
    GEOLOCATION_ENGINE = "geolocation_engine"
    RISK_SCORING_ENGINE = "risk_scoring_engine"
    COMMODITY_CLASSIFICATION_ENGINE = "commodity_classification_engine"
    SUPPLIER_COMPLIANCE_ENGINE = "supplier_compliance_engine"
    CUTOFF_DATE_ENGINE = "cutoff_date_engine"
    POLICY_COMPLIANCE_ENGINE = "policy_compliance_engine"
    EUDR_APP_BRIDGE = "eudr_app_bridge"
    TRACEABILITY_BRIDGE = "traceability_bridge"
    SATELLITE_BRIDGE = "satellite_bridge"
    GIS_BRIDGE = "gis_bridge"
    EU_IS_BRIDGE = "eu_is_bridge"
    DEMO_DATA_AVAILABILITY = "demo_data_availability"


class RemediationSeverity(str, Enum):
    """Severity level for remediation suggestions."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


# =============================================================================
# Data Models
# =============================================================================


class HealthCheckConfig(BaseModel):
    """Configuration for the health check system."""
    project_root: str = Field(
        default="", description="Root directory of the GreenLang project"
    )
    check_engines: bool = Field(default=True, description="Check engine availability")
    check_bridges: bool = Field(default=True, description="Check bridge connectivity")
    check_configuration: bool = Field(default=True, description="Check configuration files")
    check_demo_data: bool = Field(default=True, description="Check demo data availability")
    timeout_per_check_seconds: int = Field(
        default=30, description="Timeout per individual check in seconds"
    )


class RemediationSuggestion(BaseModel):
    """A remediation suggestion for a health check finding."""
    finding: str = Field(..., description="What was found")
    suggestion: str = Field(..., description="What to do about it")
    severity: RemediationSeverity = Field(..., description="Severity of the finding")
    category: HealthCategory = Field(..., description="Which category this belongs to")
    auto_fixable: bool = Field(default=False, description="Whether this can be auto-fixed")


class CategoryHealthResult(BaseModel):
    """Health result for a single category."""
    category: HealthCategory = Field(..., description="Health check category")
    status: HealthStatus = Field(..., description="Category health status")
    message: str = Field(default="", description="Status message")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed check results"
    )
    latency_ms: float = Field(default=0.0, description="Check execution time in ms")
    checks_passed: int = Field(default=0, description="Number of sub-checks passed")
    checks_failed: int = Field(default=0, description="Number of sub-checks failed")
    remediations: List[RemediationSuggestion] = Field(
        default_factory=list, description="Remediation suggestions"
    )


class HealthCheckResult(BaseModel):
    """Complete health check result across all 14 categories."""
    check_id: str = Field(default="", description="Unique health check run ID")
    overall_status: HealthStatus = Field(..., description="Overall health status")
    check_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the check was run"
    )
    total_categories: int = Field(default=14, description="Total categories checked")
    healthy_count: int = Field(default=0, description="Number of healthy categories")
    degraded_count: int = Field(default=0, description="Number of degraded categories")
    unhealthy_count: int = Field(default=0, description="Number of unhealthy categories")
    category_results: List[CategoryHealthResult] = Field(
        default_factory=list, description="Per-category health results"
    )
    critical_issues: List[RemediationSuggestion] = Field(
        default_factory=list, description="Critical issues requiring immediate attention"
    )
    warnings: List[RemediationSuggestion] = Field(
        default_factory=list, description="Non-critical warnings"
    )
    total_execution_time_ms: float = Field(
        default=0.0, description="Total health check time in ms"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash of the health check result"
    )


# =============================================================================
# Engine & Bridge module paths for import checks
# =============================================================================


ENGINE_MODULES: Dict[HealthCategory, Dict[str, str]] = {
    HealthCategory.DDS_ASSEMBLY_ENGINE: {
        "module_hint": "packs.eu_compliance.PACK_006_eudr_starter.engines.dds_assembly_engine",
        "class_name": "DDSAssemblyEngine",
        "description": "DDS Assembly Engine for Annex II statements",
    },
    HealthCategory.GEOLOCATION_ENGINE: {
        "module_hint": "packs.eu_compliance.PACK_006_eudr_starter.engines.geolocation_engine",
        "class_name": "GeolocationEngine",
        "description": "Geolocation validation engine for coordinates and polygons",
    },
    HealthCategory.RISK_SCORING_ENGINE: {
        "module_hint": "packs.eu_compliance.PACK_006_eudr_starter.engines.risk_scoring_engine",
        "class_name": "RiskScoringEngine",
        "description": "Multi-source risk scoring engine",
    },
    HealthCategory.COMMODITY_CLASSIFICATION_ENGINE: {
        "module_hint": "packs.eu_compliance.PACK_006_eudr_starter.engines.commodity_classification_engine",
        "class_name": "CommodityClassificationEngine",
        "description": "EUDR commodity and CN code classification",
    },
    HealthCategory.SUPPLIER_COMPLIANCE_ENGINE: {
        "module_hint": "packs.eu_compliance.PACK_006_eudr_starter.engines.supplier_compliance_engine",
        "class_name": "SupplierComplianceEngine",
        "description": "Supplier compliance status management",
    },
    HealthCategory.CUTOFF_DATE_ENGINE: {
        "module_hint": "packs.eu_compliance.PACK_006_eudr_starter.engines.cutoff_date_engine",
        "class_name": "CutoffDateEngine",
        "description": "EUDR cutoff date (31 Dec 2020) verification",
    },
    HealthCategory.POLICY_COMPLIANCE_ENGINE: {
        "module_hint": "packs.eu_compliance.PACK_006_eudr_starter.engines.policy_compliance_engine",
        "class_name": "PolicyComplianceEngine",
        "description": "45-rule EUDR policy compliance engine",
    },
}

BRIDGE_MODULES: Dict[HealthCategory, Dict[str, str]] = {
    HealthCategory.EUDR_APP_BRIDGE: {
        "module_hint": "packs.eu_compliance.PACK_006_eudr_starter.integrations.eudr_app_bridge",
        "class_name": "EUDRAppBridge",
        "description": "Bridge to GL-EUDR-APP v1.0 services",
    },
    HealthCategory.TRACEABILITY_BRIDGE: {
        "module_hint": "packs.eu_compliance.PACK_006_eudr_starter.integrations.traceability_bridge",
        "class_name": "TraceabilityBridge",
        "description": "Bridge to EUDR Traceability Connector",
    },
    HealthCategory.SATELLITE_BRIDGE: {
        "module_hint": "packs.eu_compliance.PACK_006_eudr_starter.integrations.satellite_bridge",
        "class_name": "SatelliteBridge",
        "description": "Bridge to Deforestation Satellite Connector",
    },
    HealthCategory.GIS_BRIDGE: {
        "module_hint": "packs.eu_compliance.PACK_006_eudr_starter.integrations.gis_bridge",
        "class_name": "GISBridge",
        "description": "Bridge to GIS/Mapping Connector",
    },
    HealthCategory.EU_IS_BRIDGE: {
        "module_hint": "packs.eu_compliance.PACK_006_eudr_starter.integrations.eu_information_system_bridge",
        "class_name": "EUInformationSystemBridge",
        "description": "Bridge to EU EUDR Information System",
    },
}

# Required pack configuration directories and files
REQUIRED_PACK_DIRS: List[str] = [
    "config", "engines", "integrations", "templates", "tests", "workflows",
]

REQUIRED_CONFIG_FILES: List[str] = [
    "config/__init__.py",
    "config/pack_config.py",
    "engines/__init__.py",
    "integrations/__init__.py",
    "templates/__init__.py",
    "workflows/__init__.py",
    "pack.yaml",
]

# Demo data file patterns
DEMO_DATA_INDICATORS: List[str] = [
    "config/demo",
    "config/presets",
    "config/sectors",
]


# =============================================================================
# Health Check Implementation
# =============================================================================


class EUDRStarterHealthCheck:
    """14-category health verification for EUDR Starter Pack.

    Validates operational readiness across engines, bridges, configuration,
    and demo data. Each category returns a status (HEALTHY/DEGRADED/UNHEALTHY),
    a human-readable message, detailed findings, and latency information.

    Attributes:
        config: Health check configuration
        _results: Accumulated category health results

    Example:
        >>> health = EUDRStarterHealthCheck()
        >>> result = await health.check_all()
        >>> print(result.overall_status)
    """

    def __init__(self, config: Optional[HealthCheckConfig] = None) -> None:
        """Initialize the health check system.

        Args:
            config: Health check configuration. Uses defaults if not provided.
        """
        self.config = config or HealthCheckConfig()
        self._results: List[CategoryHealthResult] = []

        if not self.config.project_root:
            self.config.project_root = str(
                Path(__file__).resolve().parents[4]
            )

        logger.info(
            "EUDRStarterHealthCheck initialized with project_root=%s",
            self.config.project_root,
        )

    # -------------------------------------------------------------------------
    # Main Check Entry Point
    # -------------------------------------------------------------------------

    async def check_all(self) -> HealthCheckResult:
        """Run all 14 health check categories and return a comprehensive result.

        Returns:
            HealthCheckResult with per-category details and overall status.
        """
        start_time = time.monotonic()
        self._results = []
        check_id = str(uuid4())[:12]

        logger.info("Starting 14-category health check (id=%s)", check_id)

        check_methods = [
            (self.config.check_configuration, HealthCategory.CONFIGURATION_VALIDITY,
             self._check_configuration_validity),
            (self.config.check_engines, HealthCategory.DDS_ASSEMBLY_ENGINE,
             self._check_engine, HealthCategory.DDS_ASSEMBLY_ENGINE),
            (self.config.check_engines, HealthCategory.GEOLOCATION_ENGINE,
             self._check_engine, HealthCategory.GEOLOCATION_ENGINE),
            (self.config.check_engines, HealthCategory.RISK_SCORING_ENGINE,
             self._check_engine, HealthCategory.RISK_SCORING_ENGINE),
            (self.config.check_engines, HealthCategory.COMMODITY_CLASSIFICATION_ENGINE,
             self._check_engine, HealthCategory.COMMODITY_CLASSIFICATION_ENGINE),
            (self.config.check_engines, HealthCategory.SUPPLIER_COMPLIANCE_ENGINE,
             self._check_engine, HealthCategory.SUPPLIER_COMPLIANCE_ENGINE),
            (self.config.check_engines, HealthCategory.CUTOFF_DATE_ENGINE,
             self._check_engine, HealthCategory.CUTOFF_DATE_ENGINE),
            (self.config.check_engines, HealthCategory.POLICY_COMPLIANCE_ENGINE,
             self._check_engine, HealthCategory.POLICY_COMPLIANCE_ENGINE),
            (self.config.check_bridges, HealthCategory.EUDR_APP_BRIDGE,
             self._check_bridge, HealthCategory.EUDR_APP_BRIDGE),
            (self.config.check_bridges, HealthCategory.TRACEABILITY_BRIDGE,
             self._check_bridge, HealthCategory.TRACEABILITY_BRIDGE),
            (self.config.check_bridges, HealthCategory.SATELLITE_BRIDGE,
             self._check_bridge, HealthCategory.SATELLITE_BRIDGE),
            (self.config.check_bridges, HealthCategory.GIS_BRIDGE,
             self._check_bridge, HealthCategory.GIS_BRIDGE),
            (self.config.check_bridges, HealthCategory.EU_IS_BRIDGE,
             self._check_bridge, HealthCategory.EU_IS_BRIDGE),
            (self.config.check_demo_data, HealthCategory.DEMO_DATA_AVAILABILITY,
             self._check_demo_data_availability),
        ]

        for entry in check_methods:
            enabled = entry[0]
            category = entry[1]
            method = entry[2]

            if not enabled:
                self._results.append(CategoryHealthResult(
                    category=category,
                    status=HealthStatus.HEALTHY,
                    message="Check skipped (disabled in configuration)",
                ))
                continue

            try:
                if len(entry) > 3:
                    cat_arg = entry[3]
                    result = await method(cat_arg)
                else:
                    result = await method()
                self._results.append(result)
            except Exception as exc:
                logger.error(
                    "Health check for %s failed: %s", category.value, exc, exc_info=True
                )
                self._results.append(CategoryHealthResult(
                    category=category,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check raised exception: {exc}",
                ))

        total_elapsed = (time.monotonic() - start_time) * 1000
        return self._build_result(check_id, total_elapsed)

    async def check_category(self, category: HealthCategory) -> CategoryHealthResult:
        """Run a single health check category.

        Args:
            category: The category to check.

        Returns:
            CategoryHealthResult for the specified category.
        """
        if category == HealthCategory.CONFIGURATION_VALIDITY:
            return await self._check_configuration_validity()
        elif category == HealthCategory.DEMO_DATA_AVAILABILITY:
            return await self._check_demo_data_availability()
        elif category in ENGINE_MODULES:
            return await self._check_engine(category)
        elif category in BRIDGE_MODULES:
            return await self._check_bridge(category)
        else:
            return CategoryHealthResult(
                category=category,
                status=HealthStatus.UNHEALTHY,
                message=f"Unknown category: {category.value}",
            )

    # -------------------------------------------------------------------------
    # Category 1: Configuration Validity
    # -------------------------------------------------------------------------

    async def _check_configuration_validity(self) -> CategoryHealthResult:
        """Check that pack configuration files and directories are valid.

        Returns:
            CategoryHealthResult for configuration validity.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {"directories": {}, "files": {}}

        pack_root = os.path.join(
            self.config.project_root,
            "packs", "eu-compliance", "PACK-006-eudr-starter",
        )

        # Check required directories
        for dir_name in REQUIRED_PACK_DIRS:
            dir_path = os.path.join(pack_root, dir_name)
            if os.path.isdir(dir_path):
                details["directories"][dir_name] = "present"
                passed += 1
            else:
                details["directories"][dir_name] = "missing"
                failed += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Required directory '{dir_name}' not found",
                    suggestion=f"Create directory: {dir_path}",
                    severity=RemediationSeverity.CRITICAL,
                    category=HealthCategory.CONFIGURATION_VALIDITY,
                    auto_fixable=True,
                ))

        # Check required files
        for rel_path in REQUIRED_CONFIG_FILES:
            file_path = os.path.join(pack_root, rel_path)
            if os.path.isfile(file_path):
                details["files"][rel_path] = "present"
                passed += 1
            else:
                details["files"][rel_path] = "missing"
                failed += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Required file '{rel_path}' not found",
                    suggestion=f"Create file: {file_path}",
                    severity=RemediationSeverity.CRITICAL,
                    category=HealthCategory.CONFIGURATION_VALIDITY,
                ))

        # Check pack.yaml parsability
        pack_yaml_path = os.path.join(pack_root, "pack.yaml")
        if os.path.isfile(pack_yaml_path):
            try:
                with open(pack_yaml_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if len(content) > 100:
                    details["pack_yaml_size"] = len(content)
                    passed += 1
                else:
                    details["pack_yaml_size"] = "too_small"
                    failed += 1
                    remediations.append(RemediationSuggestion(
                        finding="pack.yaml appears incomplete (less than 100 chars)",
                        suggestion="Regenerate pack.yaml with full manifest content",
                        severity=RemediationSeverity.WARNING,
                        category=HealthCategory.CONFIGURATION_VALIDITY,
                    ))
            except Exception as exc:
                details["pack_yaml_read"] = f"error: {exc}"
                failed += 1

        # Check preset files
        presets_dir = os.path.join(pack_root, "config", "presets")
        if os.path.isdir(presets_dir):
            preset_files = [f for f in os.listdir(presets_dir) if f.endswith(".yaml")]
            details["preset_files"] = preset_files
            if len(preset_files) >= 2:
                passed += 1
            else:
                failed += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Only {len(preset_files)} preset files found (expected >= 2)",
                    suggestion="Create SME, mid-market, and large enterprise preset files",
                    severity=RemediationSeverity.WARNING,
                    category=HealthCategory.CONFIGURATION_VALIDITY,
                ))
        else:
            details["presets_dir"] = "missing"

        latency_ms = (time.monotonic() - start_time) * 1000
        status = self._determine_status(passed, failed)

        return CategoryHealthResult(
            category=HealthCategory.CONFIGURATION_VALIDITY,
            status=status,
            message=f"{passed} configuration checks passed, {failed} failed",
            details=details,
            latency_ms=round(latency_ms, 2),
            checks_passed=passed,
            checks_failed=failed,
            remediations=remediations,
        )

    # -------------------------------------------------------------------------
    # Categories 2-8: Engine Checks
    # -------------------------------------------------------------------------

    async def _check_engine(self, category: HealthCategory) -> CategoryHealthResult:
        """Check that a specific engine module is importable and functional.

        Args:
            category: The engine category to check.

        Returns:
            CategoryHealthResult for the specified engine.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {}

        engine_info = ENGINE_MODULES.get(category)
        if engine_info is None:
            return CategoryHealthResult(
                category=category,
                status=HealthStatus.UNHEALTHY,
                message=f"No engine info for category {category.value}",
            )

        module_hint = engine_info["module_hint"]
        class_name = engine_info["class_name"]
        description = engine_info["description"]

        # Check module file exists on disk
        module_path_parts = module_hint.replace(".", os.sep) + ".py"
        module_file = os.path.join(self.config.project_root, module_path_parts)
        if os.path.isfile(module_file):
            details["file_exists"] = True
            passed += 1
        else:
            details["file_exists"] = False
            failed += 1
            remediations.append(RemediationSuggestion(
                finding=f"Engine module file not found: {module_file}",
                suggestion=f"Create the {class_name} module at: {module_file}",
                severity=RemediationSeverity.CRITICAL,
                category=category,
            ))

        # Check module importability
        try:
            mod = importlib.import_module(module_hint)
            details["importable"] = True
            passed += 1

            # Check class exists
            if hasattr(mod, class_name):
                details["class_found"] = True
                passed += 1
            else:
                details["class_found"] = False
                failed += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Class '{class_name}' not found in module",
                    suggestion=f"Ensure {class_name} is defined and exported in {module_hint}",
                    severity=RemediationSeverity.CRITICAL,
                    category=category,
                ))
        except ImportError as exc:
            details["importable"] = False
            details["import_error"] = str(exc)
            failed += 1
            remediations.append(RemediationSuggestion(
                finding=f"Engine module '{module_hint}' could not be imported: {exc}",
                suggestion="Verify module path and all dependencies are installed",
                severity=RemediationSeverity.WARNING,
                category=category,
            ))
        except Exception as exc:
            details["importable"] = False
            details["unexpected_error"] = str(exc)
            failed += 1

        # Check engine description
        details["description"] = description

        latency_ms = (time.monotonic() - start_time) * 1000
        status = self._determine_status(passed, failed)

        return CategoryHealthResult(
            category=category,
            status=status,
            message=f"{description}: {status.value}",
            details=details,
            latency_ms=round(latency_ms, 2),
            checks_passed=passed,
            checks_failed=failed,
            remediations=remediations,
        )

    # -------------------------------------------------------------------------
    # Categories 9-13: Bridge Checks
    # -------------------------------------------------------------------------

    async def _check_bridge(self, category: HealthCategory) -> CategoryHealthResult:
        """Check that a specific bridge module is importable and functional.

        Args:
            category: The bridge category to check.

        Returns:
            CategoryHealthResult for the specified bridge.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {}

        bridge_info = BRIDGE_MODULES.get(category)
        if bridge_info is None:
            return CategoryHealthResult(
                category=category,
                status=HealthStatus.UNHEALTHY,
                message=f"No bridge info for category {category.value}",
            )

        module_hint = bridge_info["module_hint"]
        class_name = bridge_info["class_name"]
        description = bridge_info["description"]

        # Check module file on disk
        module_path_parts = module_hint.replace(".", os.sep) + ".py"
        module_file = os.path.join(self.config.project_root, module_path_parts)
        if os.path.isfile(module_file):
            details["file_exists"] = True
            passed += 1
        else:
            details["file_exists"] = False
            failed += 1
            remediations.append(RemediationSuggestion(
                finding=f"Bridge module file not found: {module_file}",
                suggestion=f"Create the {class_name} module at: {module_file}",
                severity=RemediationSeverity.CRITICAL,
                category=category,
            ))

        # Check module importability
        try:
            mod = importlib.import_module(module_hint)
            details["importable"] = True
            passed += 1

            if hasattr(mod, class_name):
                details["class_found"] = True
                passed += 1

                # Try instantiation with defaults
                try:
                    bridge_cls = getattr(mod, class_name)
                    instance = bridge_cls()
                    details["instantiable"] = True
                    passed += 1

                    # Check for required proxy methods
                    expected_methods = _get_expected_bridge_methods(category)
                    for method_name in expected_methods:
                        if hasattr(instance, method_name):
                            details[f"method_{method_name}"] = "present"
                            passed += 1
                        else:
                            details[f"method_{method_name}"] = "missing"
                            failed += 1
                            remediations.append(RemediationSuggestion(
                                finding=f"Method '{method_name}' not found on {class_name}",
                                suggestion=f"Implement {method_name}() on {class_name}",
                                severity=RemediationSeverity.WARNING,
                                category=category,
                            ))
                except Exception as exc:
                    details["instantiable"] = False
                    details["init_error"] = str(exc)
                    failed += 1
            else:
                details["class_found"] = False
                failed += 1
        except ImportError as exc:
            details["importable"] = False
            details["import_error"] = str(exc)
            failed += 1
            remediations.append(RemediationSuggestion(
                finding=f"Bridge module '{module_hint}' could not be imported: {exc}",
                suggestion="Verify module path and dependencies",
                severity=RemediationSeverity.WARNING,
                category=category,
            ))
        except Exception as exc:
            details["importable"] = False
            details["unexpected_error"] = str(exc)
            failed += 1

        details["description"] = description

        latency_ms = (time.monotonic() - start_time) * 1000
        status = self._determine_status(passed, failed)

        return CategoryHealthResult(
            category=category,
            status=status,
            message=f"{description}: {status.value}",
            details=details,
            latency_ms=round(latency_ms, 2),
            checks_passed=passed,
            checks_failed=failed,
            remediations=remediations,
        )

    # -------------------------------------------------------------------------
    # Category 14: Demo Data Availability
    # -------------------------------------------------------------------------

    async def _check_demo_data_availability(self) -> CategoryHealthResult:
        """Check that demo data files and directories are present.

        Returns:
            CategoryHealthResult for demo data availability.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {}

        pack_root = os.path.join(
            self.config.project_root,
            "packs", "eu-compliance", "PACK-006-eudr-starter",
        )

        # Check demo data directories
        for rel_path in DEMO_DATA_INDICATORS:
            full_path = os.path.join(pack_root, rel_path)
            if os.path.isdir(full_path):
                details[rel_path] = "present"
                passed += 1

                # Count files in directory
                try:
                    file_count = len(os.listdir(full_path))
                    details[f"{rel_path}_file_count"] = file_count
                    if file_count > 0:
                        passed += 1
                    else:
                        failed += 1
                        remediations.append(RemediationSuggestion(
                            finding=f"Directory '{rel_path}' is empty",
                            suggestion=f"Add demo/config files to {full_path}",
                            severity=RemediationSeverity.WARNING,
                            category=HealthCategory.DEMO_DATA_AVAILABILITY,
                        ))
                except OSError:
                    details[f"{rel_path}_file_count"] = "error"
            else:
                details[rel_path] = "missing"
                failed += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Demo data directory '{rel_path}' not found",
                    suggestion=f"Create directory: {full_path}",
                    severity=RemediationSeverity.WARNING,
                    category=HealthCategory.DEMO_DATA_AVAILABILITY,
                    auto_fixable=True,
                ))

        # Check for demo config YAML
        demo_dir = os.path.join(pack_root, "config", "demo")
        if os.path.isdir(demo_dir):
            yaml_files = [f for f in os.listdir(demo_dir) if f.endswith((".yaml", ".yml"))]
            json_files = [f for f in os.listdir(demo_dir) if f.endswith(".json")]
            csv_files = [f for f in os.listdir(demo_dir) if f.endswith(".csv")]
            details["demo_yaml_files"] = yaml_files
            details["demo_json_files"] = json_files
            details["demo_csv_files"] = csv_files

            if yaml_files or json_files or csv_files:
                passed += 1
            else:
                failed += 1
                remediations.append(RemediationSuggestion(
                    finding="No demo data files found in config/demo",
                    suggestion="Add demo YAML/JSON/CSV files with sample supplier and plot data",
                    severity=RemediationSeverity.WARNING,
                    category=HealthCategory.DEMO_DATA_AVAILABILITY,
                ))

        # Check for EUDR agent modules (downstream dependencies)
        eudr_agent_base = os.path.join(
            self.config.project_root, "greenlang", "agents", "eudr"
        )
        if os.path.isdir(eudr_agent_base):
            details["eudr_agents_dir"] = "present"
            passed += 1
        else:
            details["eudr_agents_dir"] = "missing"
            failed += 1
            remediations.append(RemediationSuggestion(
                finding="EUDR agents directory not found",
                suggestion=f"Ensure EUDR agents exist at: {eudr_agent_base}",
                severity=RemediationSeverity.CRITICAL,
                category=HealthCategory.DEMO_DATA_AVAILABILITY,
            ))

        latency_ms = (time.monotonic() - start_time) * 1000
        status = self._determine_status(passed, failed)

        return CategoryHealthResult(
            category=HealthCategory.DEMO_DATA_AVAILABILITY,
            status=status,
            message=f"Demo data: {passed} items present, {failed} missing",
            details=details,
            latency_ms=round(latency_ms, 2),
            checks_passed=passed,
            checks_failed=failed,
            remediations=remediations,
        )

    # -------------------------------------------------------------------------
    # Result Assembly
    # -------------------------------------------------------------------------

    def _build_result(self, check_id: str, total_elapsed_ms: float) -> HealthCheckResult:
        """Build the final HealthCheckResult from accumulated category results.

        Args:
            check_id: Unique check run identifier.
            total_elapsed_ms: Total elapsed time for all checks.

        Returns:
            Complete HealthCheckResult.
        """
        healthy = sum(1 for r in self._results if r.status == HealthStatus.HEALTHY)
        degraded = sum(1 for r in self._results if r.status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for r in self._results if r.status == HealthStatus.UNHEALTHY)

        critical_issues: List[RemediationSuggestion] = []
        warnings: List[RemediationSuggestion] = []

        for cat_result in self._results:
            for rem in cat_result.remediations:
                if rem.severity == RemediationSeverity.CRITICAL:
                    critical_issues.append(rem)
                else:
                    warnings.append(rem)

        # Determine overall status
        if unhealthy == 0 and len(critical_issues) == 0:
            overall = HealthStatus.HEALTHY
        elif unhealthy <= 3 and len(critical_issues) <= 2:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.UNHEALTHY

        provenance = _compute_hash(
            f"eudr-health:{check_id}:{datetime.utcnow().isoformat()}"
            f":{healthy}:{degraded}:{unhealthy}"
        )

        result = HealthCheckResult(
            check_id=check_id,
            overall_status=overall,
            total_categories=len(self._results),
            healthy_count=healthy,
            degraded_count=degraded,
            unhealthy_count=unhealthy,
            category_results=list(self._results),
            critical_issues=critical_issues,
            warnings=warnings,
            total_execution_time_ms=round(total_elapsed_ms, 2),
            provenance_hash=provenance,
        )

        logger.info(
            "Health check complete: %s (healthy=%d, degraded=%d, unhealthy=%d, "
            "critical=%d, warnings=%d) in %.1fms",
            overall.value, healthy, degraded, unhealthy,
            len(critical_issues), len(warnings), total_elapsed_ms,
        )

        return result

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _determine_status(passed: int, failed: int) -> HealthStatus:
        """Determine health status from pass/fail counts.

        Args:
            passed: Number of checks passed.
            failed: Number of checks failed.

        Returns:
            HealthStatus based on the ratio.
        """
        if failed == 0:
            return HealthStatus.HEALTHY
        total = passed + failed
        if total == 0:
            return HealthStatus.UNHEALTHY
        pass_rate = passed / total
        if pass_rate >= 0.7:
            return HealthStatus.DEGRADED
        return HealthStatus.UNHEALTHY

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the most recent health check.

        Returns:
            Dictionary with category names and their statuses.
        """
        return {
            r.category.value: {
                "status": r.status.value,
                "message": r.message,
                "latency_ms": r.latency_ms,
                "passed": r.checks_passed,
                "failed": r.checks_failed,
            }
            for r in self._results
        }


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string.

    Args:
        data: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _get_expected_bridge_methods(category: HealthCategory) -> List[str]:
    """Return the expected public methods for a bridge class.

    Args:
        category: The bridge category.

    Returns:
        List of expected method names.
    """
    method_map: Dict[HealthCategory, List[str]] = {
        HealthCategory.EUDR_APP_BRIDGE: [
            "get_supplier_service", "get_plot_service", "get_dds_service",
            "get_pipeline_service", "get_risk_service", "get_dashboard_service",
            "get_document_service", "get_settings_service",
        ],
        HealthCategory.TRACEABILITY_BRIDGE: [
            "get_plot_registry", "get_chain_of_custody", "get_commodity_classifier",
            "get_compliance_verifier", "get_due_diligence", "get_risk_assessment",
            "get_supply_chain_mapper",
        ],
        HealthCategory.SATELLITE_BRIDGE: [
            "get_satellite_data", "check_forest_change", "get_deforestation_alerts",
            "assess_baseline", "run_monitoring_pipeline", "get_alert_aggregation",
            "generate_compliance_report",
        ],
        HealthCategory.GIS_BRIDGE: [
            "transform_coordinates", "resolve_boundaries", "analyze_spatial",
            "classify_land_cover", "geocode_address", "reverse_geocode",
            "parse_geospatial_format", "validate_topology",
        ],
        HealthCategory.EU_IS_BRIDGE: [
            "submit_dds", "check_submission_status", "amend_dds",
            "retrieve_dds", "validate_dds_format", "register_operator",
            "get_operator_status", "search_reference_numbers",
            "get_country_benchmarks",
        ],
    }
    return method_map.get(category, [])
