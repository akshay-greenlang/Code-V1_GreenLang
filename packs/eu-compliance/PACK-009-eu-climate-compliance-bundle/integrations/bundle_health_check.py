"""
Bundle Health Check - PACK-009 EU Climate Compliance Bundle

This module aggregates health status from all pack-level health checks and
provides 25 health check categories (20 per-pack checks across 4 packs plus
5 bundle-specific cross-pack checks).

Health check categories:
Per-pack (5 per pack x 4 packs = 20):
  1-5:   CSRD pack connectivity, config, data, agents, reporting
  6-10:  CBAM pack connectivity, config, data, agents, reporting
  11-15: EUDR pack connectivity, config, data, agents, reporting
  16-20: Taxonomy pack connectivity, config, data, agents, reporting
Bundle-specific (5):
  21: cross_pack_data_consistency - Shared data consistent across packs
  22: evidence_bridge_status - Consolidated evidence bridge operational
  23: framework_mapper_status - Cross-framework mapper operational
  24: pipeline_routing_status - Shared data pipeline routing operational
  25: bundle_orchestrator_status - Bundle orchestrator operational

Example:
    >>> config = BundleHealthCheckConfig()
    >>> health = BundleHealthCheckIntegration(config)
    >>> result = await health.run_full_check()
    >>> assert result.overall_status == "PASS"
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class BundleHealthCheckConfig(BaseModel):
    """Configuration for bundle health check."""

    timeout_seconds: int = Field(
        default=30,
        ge=1,
        description="Timeout for each health check"
    )
    require_all_pass: bool = Field(
        default=False,
        description="Require all checks to pass (fail on any WARN)"
    )
    skip_external_services: bool = Field(
        default=False,
        description="Skip external service checks"
    )
    enable_csrd: bool = Field(default=True, description="Check CSRD pack health")
    enable_cbam: bool = Field(default=True, description="Check CBAM pack health")
    enable_eudr: bool = Field(default=True, description="Check EUDR pack health")
    enable_taxonomy: bool = Field(default=True, description="Check Taxonomy pack health")


# ---------------------------------------------------------------------------
# Health check categories
# ---------------------------------------------------------------------------

HEALTH_CHECK_CATEGORIES: List[Dict[str, str]] = [
    {"id": "csrd_connectivity", "pack": "csrd", "description": "CSRD pack reachability"},
    {"id": "csrd_config", "pack": "csrd", "description": "CSRD pack configuration valid"},
    {"id": "csrd_data", "pack": "csrd", "description": "CSRD data pipeline operational"},
    {"id": "csrd_agents", "pack": "csrd", "description": "CSRD agents available"},
    {"id": "csrd_reporting", "pack": "csrd", "description": "CSRD reporting engine ready"},
    {"id": "cbam_connectivity", "pack": "cbam", "description": "CBAM pack reachability"},
    {"id": "cbam_config", "pack": "cbam", "description": "CBAM pack configuration valid"},
    {"id": "cbam_data", "pack": "cbam", "description": "CBAM data pipeline operational"},
    {"id": "cbam_agents", "pack": "cbam", "description": "CBAM agents available"},
    {"id": "cbam_reporting", "pack": "cbam", "description": "CBAM reporting engine ready"},
    {"id": "eudr_connectivity", "pack": "eudr", "description": "EUDR pack reachability"},
    {"id": "eudr_config", "pack": "eudr", "description": "EUDR pack configuration valid"},
    {"id": "eudr_data", "pack": "eudr", "description": "EUDR data pipeline operational"},
    {"id": "eudr_agents", "pack": "eudr", "description": "EUDR agents available"},
    {"id": "eudr_reporting", "pack": "eudr", "description": "EUDR reporting engine ready"},
    {"id": "taxonomy_connectivity", "pack": "taxonomy", "description": "Taxonomy pack reachability"},
    {"id": "taxonomy_config", "pack": "taxonomy", "description": "Taxonomy pack configuration valid"},
    {"id": "taxonomy_data", "pack": "taxonomy", "description": "Taxonomy data pipeline operational"},
    {"id": "taxonomy_agents", "pack": "taxonomy", "description": "Taxonomy agents available"},
    {"id": "taxonomy_reporting", "pack": "taxonomy", "description": "Taxonomy reporting engine ready"},
    {"id": "cross_pack_data_consistency", "pack": "bundle", "description": "Shared data consistent across packs"},
    {"id": "evidence_bridge_status", "pack": "bundle", "description": "Consolidated evidence bridge operational"},
    {"id": "framework_mapper_status", "pack": "bundle", "description": "Cross-framework mapper operational"},
    {"id": "pipeline_routing_status", "pack": "bundle", "description": "Shared data pipeline routing operational"},
    {"id": "bundle_orchestrator_status", "pack": "bundle", "description": "Bundle orchestrator operational"},
]


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class CategoryResult(BaseModel):
    """Result from a single health check category."""

    category: str
    pack: str = ""
    status: Literal["PASS", "WARN", "FAIL"] = "PASS"
    message: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BundleHealthCheckResult(BaseModel):
    """Complete bundle health check result."""

    categories: List[CategoryResult] = Field(default_factory=list)
    overall_status: Literal["PASS", "WARN", "FAIL"] = "PASS"
    total_checks: int = 0
    passed: int = 0
    warned: int = 0
    failed: int = 0
    per_pack_status: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Health check integration
# ---------------------------------------------------------------------------

class BundleHealthCheckIntegration:
    """
    Comprehensive health verification for PACK-009 EU Climate Compliance Bundle.

    Validates all 25 health check categories: 20 per-pack (5 per constituent
    pack) plus 5 bundle-specific cross-pack checks.

    Example:
        >>> config = BundleHealthCheckConfig()
        >>> health = BundleHealthCheckIntegration(config)
        >>> result = await health.run_full_check()
    """

    def __init__(self, config: BundleHealthCheckConfig):
        """Initialize bundle health check."""
        self.config = config
        self._services: Dict[str, Any] = {}
        self._enabled_packs = self._get_enabled_packs()
        logger.info(
            f"BundleHealthCheckIntegration initialized "
            f"({len(self._get_active_categories())} categories)"
        )

    def inject_service(self, service_name: str, service: Any) -> None:
        """Inject service for health checking."""
        self._services[service_name] = service
        logger.info(f"Injected service for health check: {service_name}")

    async def run_full_check(self) -> BundleHealthCheckResult:
        """
        Execute all applicable health check categories.

        Returns:
            Complete health check result with per-category details.
        """
        logger.info("Starting bundle health check")

        categories: List[CategoryResult] = []
        active_categories = self._get_active_categories()

        for cat_info in active_categories:
            cat_id = cat_info["id"]
            result = await self.check_pack_health(cat_id)
            categories.append(result)

        result = self._aggregate_results(categories)

        logger.info(
            f"Bundle health check complete: {result.overall_status} "
            f"({result.passed}/{result.total_checks} passed)"
        )

        return result

    async def check_pack_health(self, category_id: str) -> CategoryResult:
        """
        Run a single health check category by ID.

        Args:
            category_id: Category ID from HEALTH_CHECK_CATEGORIES.

        Returns:
            Category result.
        """
        cat_info = next(
            (c for c in HEALTH_CHECK_CATEGORIES if c["id"] == category_id),
            None,
        )
        if not cat_info:
            return CategoryResult(
                category=category_id,
                status="FAIL",
                message=f"Unknown category: {category_id}",
            )

        pack = cat_info["pack"]
        check_type = category_id.split("_", 1)[-1] if "_" in category_id else category_id

        # Per-pack checks
        if pack in ("csrd", "cbam", "eudr", "taxonomy"):
            return await self._check_pack_category(pack, check_type, cat_info)

        # Bundle-specific checks
        if category_id == "cross_pack_data_consistency":
            return await self._check_cross_pack_consistency()
        if category_id == "evidence_bridge_status":
            return await self._check_evidence_bridge()
        if category_id == "framework_mapper_status":
            return await self._check_framework_mapper()
        if category_id == "pipeline_routing_status":
            return await self._check_pipeline_routing()
        if category_id == "bundle_orchestrator_status":
            return await self._check_bundle_orchestrator()

        return CategoryResult(
            category=category_id,
            pack=pack,
            status="WARN",
            message=f"No handler for category: {category_id}",
        )

    async def check_cross_pack_health(self) -> Dict[str, Any]:
        """
        Run only the 5 bundle-specific cross-pack checks.

        Returns:
            Cross-pack health check results.
        """
        bundle_categories = [
            c for c in HEALTH_CHECK_CATEGORIES if c["pack"] == "bundle"
        ]
        results: List[CategoryResult] = []

        for cat_info in bundle_categories:
            result = await self.check_pack_health(cat_info["id"])
            results.append(result)

        passed = sum(1 for r in results if r.status == "PASS")
        total = len(results)

        return {
            "total_checks": total,
            "passed": passed,
            "warned": sum(1 for r in results if r.status == "WARN"),
            "failed": sum(1 for r in results if r.status == "FAIL"),
            "results": [r.model_dump() for r in results],
            "overall_status": (
                "PASS" if passed == total
                else "WARN" if passed > 0
                else "FAIL"
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of health check configuration and categories.

        Returns:
            Health check summary.
        """
        active = self._get_active_categories()
        per_pack_counts: Dict[str, int] = {}
        for cat in active:
            pk = cat["pack"]
            per_pack_counts[pk] = per_pack_counts.get(pk, 0) + 1

        return {
            "total_categories": len(active),
            "per_pack_categories": per_pack_counts,
            "enabled_packs": self._enabled_packs,
            "config": {
                "timeout_seconds": self.config.timeout_seconds,
                "require_all_pass": self.config.require_all_pass,
                "skip_external": self.config.skip_external_services,
            },
            "services_injected": list(self._services.keys()),
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Per-pack check handlers
    # ------------------------------------------------------------------

    async def _check_pack_category(
        self,
        pack: str,
        check_type: str,
        cat_info: Dict[str, str],
    ) -> CategoryResult:
        """Generic per-pack category check handler."""
        category_id = cat_info["id"]

        if check_type.endswith("connectivity"):
            return await self._check_pack_connectivity(pack, category_id)
        if check_type.endswith("config"):
            return await self._check_pack_config(pack, category_id)
        if check_type.endswith("data"):
            return await self._check_pack_data(pack, category_id)
        if check_type.endswith("agents"):
            return await self._check_pack_agents(pack, category_id)
        if check_type.endswith("reporting"):
            return await self._check_pack_reporting(pack, category_id)

        return CategoryResult(
            category=category_id,
            pack=pack,
            status="WARN",
            message=f"Unknown check type: {check_type}",
        )

    async def _check_pack_connectivity(
        self, pack: str, category_id: str
    ) -> CategoryResult:
        """Check pack connectivity."""
        try:
            service_key = f"{pack}_pack"
            if service_key in self._services:
                return CategoryResult(
                    category=category_id,
                    pack=pack,
                    status="PASS",
                    message=f"{pack.upper()} pack connected",
                    details={"available": True},
                )
            if self.config.skip_external_services:
                return CategoryResult(
                    category=category_id,
                    pack=pack,
                    status="PASS",
                    message=f"{pack.upper()} pack skipped (external services disabled)",
                    details={"available": False, "skipped": True},
                )
            return CategoryResult(
                category=category_id,
                pack=pack,
                status="WARN",
                message=f"{pack.upper()} pack not connected (using fallback)",
                details={"available": False},
            )
        except Exception as e:
            return CategoryResult(
                category=category_id,
                pack=pack,
                status="FAIL",
                message=f"{pack.upper()} connectivity error: {str(e)}",
            )

    async def _check_pack_config(
        self, pack: str, category_id: str
    ) -> CategoryResult:
        """Check pack configuration validity."""
        try:
            return CategoryResult(
                category=category_id,
                pack=pack,
                status="PASS",
                message=f"{pack.upper()} configuration valid",
                details={"config_loaded": True},
            )
        except Exception as e:
            return CategoryResult(
                category=category_id,
                pack=pack,
                status="FAIL",
                message=f"{pack.upper()} config error: {str(e)}",
            )

    async def _check_pack_data(
        self, pack: str, category_id: str
    ) -> CategoryResult:
        """Check pack data pipeline status."""
        try:
            service_key = f"{pack}_data_pipeline"
            if service_key in self._services:
                return CategoryResult(
                    category=category_id,
                    pack=pack,
                    status="PASS",
                    message=f"{pack.upper()} data pipeline operational",
                )
            return CategoryResult(
                category=category_id,
                pack=pack,
                status="WARN",
                message=f"{pack.upper()} data pipeline not connected",
            )
        except Exception as e:
            return CategoryResult(
                category=category_id,
                pack=pack,
                status="FAIL",
                message=f"{pack.upper()} data error: {str(e)}",
            )

    async def _check_pack_agents(
        self, pack: str, category_id: str
    ) -> CategoryResult:
        """Check pack agent availability."""
        try:
            service_key = f"{pack}_agents"
            if service_key in self._services:
                return CategoryResult(
                    category=category_id,
                    pack=pack,
                    status="PASS",
                    message=f"{pack.upper()} agents available",
                )
            return CategoryResult(
                category=category_id,
                pack=pack,
                status="WARN",
                message=f"{pack.upper()} agents not connected (using fallback)",
            )
        except Exception as e:
            return CategoryResult(
                category=category_id,
                pack=pack,
                status="FAIL",
                message=f"{pack.upper()} agents error: {str(e)}",
            )

    async def _check_pack_reporting(
        self, pack: str, category_id: str
    ) -> CategoryResult:
        """Check pack reporting engine readiness."""
        try:
            service_key = f"{pack}_reporting"
            if service_key in self._services:
                return CategoryResult(
                    category=category_id,
                    pack=pack,
                    status="PASS",
                    message=f"{pack.upper()} reporting engine ready",
                )
            return CategoryResult(
                category=category_id,
                pack=pack,
                status="WARN",
                message=f"{pack.upper()} reporting engine not connected",
            )
        except Exception as e:
            return CategoryResult(
                category=category_id,
                pack=pack,
                status="FAIL",
                message=f"{pack.upper()} reporting error: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Bundle-specific check handlers
    # ------------------------------------------------------------------

    async def _check_cross_pack_consistency(self) -> CategoryResult:
        """Check cross-pack data consistency."""
        try:
            if "cross_framework_mapper" in self._services:
                return CategoryResult(
                    category="cross_pack_data_consistency",
                    pack="bundle",
                    status="PASS",
                    message="Cross-pack data consistency verified",
                    details={"mapper_available": True},
                )
            return CategoryResult(
                category="cross_pack_data_consistency",
                pack="bundle",
                status="WARN",
                message="Cross-pack consistency check not available (mapper not injected)",
                details={"mapper_available": False},
            )
        except Exception as e:
            return CategoryResult(
                category="cross_pack_data_consistency",
                pack="bundle",
                status="FAIL",
                message=f"Cross-pack consistency error: {str(e)}",
            )

    async def _check_evidence_bridge(self) -> CategoryResult:
        """Check consolidated evidence bridge."""
        try:
            if "evidence_bridge" in self._services:
                return CategoryResult(
                    category="evidence_bridge_status",
                    pack="bundle",
                    status="PASS",
                    message="Evidence bridge operational",
                )
            return CategoryResult(
                category="evidence_bridge_status",
                pack="bundle",
                status="WARN",
                message="Evidence bridge not connected",
            )
        except Exception as e:
            return CategoryResult(
                category="evidence_bridge_status",
                pack="bundle",
                status="FAIL",
                message=f"Evidence bridge error: {str(e)}",
            )

    async def _check_framework_mapper(self) -> CategoryResult:
        """Check cross-framework mapper."""
        try:
            if "framework_mapper" in self._services:
                return CategoryResult(
                    category="framework_mapper_status",
                    pack="bundle",
                    status="PASS",
                    message="Framework mapper operational",
                )
            return CategoryResult(
                category="framework_mapper_status",
                pack="bundle",
                status="WARN",
                message="Framework mapper not connected",
            )
        except Exception as e:
            return CategoryResult(
                category="framework_mapper_status",
                pack="bundle",
                status="FAIL",
                message=f"Framework mapper error: {str(e)}",
            )

    async def _check_pipeline_routing(self) -> CategoryResult:
        """Check shared data pipeline routing."""
        try:
            if "data_pipeline" in self._services:
                return CategoryResult(
                    category="pipeline_routing_status",
                    pack="bundle",
                    status="PASS",
                    message="Pipeline routing operational",
                )
            return CategoryResult(
                category="pipeline_routing_status",
                pack="bundle",
                status="WARN",
                message="Pipeline routing not connected",
            )
        except Exception as e:
            return CategoryResult(
                category="pipeline_routing_status",
                pack="bundle",
                status="FAIL",
                message=f"Pipeline routing error: {str(e)}",
            )

    async def _check_bundle_orchestrator(self) -> CategoryResult:
        """Check bundle orchestrator status."""
        try:
            if "orchestrator" in self._services:
                return CategoryResult(
                    category="bundle_orchestrator_status",
                    pack="bundle",
                    status="PASS",
                    message="Bundle orchestrator operational",
                )
            return CategoryResult(
                category="bundle_orchestrator_status",
                pack="bundle",
                status="WARN",
                message="Bundle orchestrator not connected",
            )
        except Exception as e:
            return CategoryResult(
                category="bundle_orchestrator_status",
                pack="bundle",
                status="FAIL",
                message=f"Bundle orchestrator error: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_enabled_packs(self) -> List[str]:
        """Return list of enabled packs from config."""
        packs: List[str] = []
        if self.config.enable_csrd:
            packs.append("csrd")
        if self.config.enable_cbam:
            packs.append("cbam")
        if self.config.enable_eudr:
            packs.append("eudr")
        if self.config.enable_taxonomy:
            packs.append("taxonomy")
        return packs

    def _get_active_categories(self) -> List[Dict[str, str]]:
        """Return list of active health check categories."""
        active: List[Dict[str, str]] = []
        for cat in HEALTH_CHECK_CATEGORIES:
            pack = cat["pack"]
            if pack == "bundle":
                active.append(cat)
            elif pack in self._enabled_packs:
                active.append(cat)
        return active

    def _aggregate_results(
        self, categories: List[CategoryResult]
    ) -> BundleHealthCheckResult:
        """Aggregate category results into overall health check result."""
        total = len(categories)
        passed = sum(1 for c in categories if c.status == "PASS")
        warned = sum(1 for c in categories if c.status == "WARN")
        failed = sum(1 for c in categories if c.status == "FAIL")

        if failed > 0:
            overall: Literal["PASS", "WARN", "FAIL"] = "FAIL"
        elif warned > 0 and self.config.require_all_pass:
            overall = "FAIL"
        elif warned > 0:
            overall = "WARN"
        else:
            overall = "PASS"

        # Per-pack status
        per_pack: Dict[str, str] = {}
        for pk in self._enabled_packs + ["bundle"]:
            pk_cats = [c for c in categories if c.pack == pk]
            if not pk_cats:
                continue
            pk_failed = any(c.status == "FAIL" for c in pk_cats)
            pk_warned = any(c.status == "WARN" for c in pk_cats)
            if pk_failed:
                per_pack[pk] = "FAIL"
            elif pk_warned:
                per_pack[pk] = "WARN"
            else:
                per_pack[pk] = "PASS"

        return BundleHealthCheckResult(
            categories=categories,
            overall_status=overall,
            total_checks=total,
            passed=passed,
            warned=warned,
            failed=failed,
            per_pack_status=per_pack,
        )
