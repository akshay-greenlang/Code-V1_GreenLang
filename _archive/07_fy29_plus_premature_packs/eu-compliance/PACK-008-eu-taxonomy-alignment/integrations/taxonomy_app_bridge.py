"""
Taxonomy App Bridge - PACK-008 EU Taxonomy Alignment

This module bridges to GL-Taxonomy-APP v1.0 (APP-010), wiring its 10 engines
into pack workflows with unified configuration and service injection.

GL-Taxonomy-APP engines:
1. Eligibility Engine - NACE activity screening
2. Substantial Contribution Engine - TSC evaluation
3. DNSH Engine - 6-objective DNSH matrix
4. Minimum Safeguards Engine - 4-topic MS verification
5. KPI Engine - Turnover/CapEx/OpEx calculation
6. GAR Engine - Green Asset Ratio (financial institutions)
7. TSC Engine - Technical Screening Criteria lookup
8. Transition Engine - Article 10(2) assessment
9. Enabling Engine - Article 16 classification
10. Reporting Engine - Article 8 / EBA Pillar 3 templates

Example:
    >>> config = TaxonomyAppBridgeConfig(
    ...     app_base_url="https://api.greenlang.com/taxonomy/v1",
    ...     timeout_seconds=30
    ... )
    >>> bridge = TaxonomyAppBridge(config)
    >>> result = await bridge.screen_activities(activities)
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
import hashlib
import logging
import asyncio

logger = logging.getLogger(__name__)


class TaxonomyAppBridgeConfig(BaseModel):
    """Configuration for Taxonomy App Bridge."""

    app_base_url: HttpUrl = Field(
        default="https://api.greenlang.com/taxonomy/v1",
        description="GL-Taxonomy-APP base API URL"
    )
    api_version: str = Field(
        default="v1",
        description="API version string"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API authentication key"
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        description="API request timeout"
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable response caching for repeated lookups"
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        description="Number of retry attempts on failure"
    )
    da_version: str = Field(
        default="2023",
        description="Active Delegated Act version"
    )


class EngineProxy:
    """
    Proxy for a GL-Taxonomy-APP engine.

    Provides fallback when real engine service is not injected.
    """

    def __init__(self, engine_name: str, config: TaxonomyAppBridgeConfig):
        self.engine_name = engine_name
        self.config = config
        self._service: Any = None
        logger.debug(f"EngineProxy created for {engine_name}")

    def inject(self, service: Any) -> None:
        """Inject real engine service instance."""
        self._service = service
        logger.info(f"Injected service into {self.engine_name} proxy")

    async def call(self, method_name: str, **kwargs) -> Dict[str, Any]:
        """Call engine method with fallback."""
        if self._service and hasattr(self._service, method_name):
            method = getattr(self._service, method_name)
            if asyncio.iscoroutinefunction(method):
                return await method(**kwargs)
            return method(**kwargs)

        logger.warning(
            f"Engine {self.engine_name} not available, using fallback for {method_name}"
        )
        return {
            "status": "fallback",
            "engine": self.engine_name,
            "method": method_name,
            "message": f"Executed fallback for {self.engine_name}.{method_name}",
            "timestamp": datetime.utcnow().isoformat()
        }


class TaxonomyAppBridge:
    """
    Bridge to GL-Taxonomy-APP v1.0 with 10 engine proxies.

    Maps pack engine calls to GL-Taxonomy-APP engine calls with unified config,
    graceful degradation, and service injection support.

    Example:
        >>> config = TaxonomyAppBridgeConfig()
        >>> bridge = TaxonomyAppBridge(config)
        >>> bridge.inject_service("eligibility", eligibility_engine)
        >>> result = await bridge.screen_activities(activities)
    """

    ENGINE_NAMES: List[str] = [
        "eligibility", "substantial_contribution", "dnsh",
        "minimum_safeguards", "kpi", "gar", "tsc",
        "transition", "enabling", "reporting"
    ]

    def __init__(self, config: TaxonomyAppBridgeConfig):
        """Initialize bridge with engine proxies."""
        self.config = config
        self._engines: Dict[str, EngineProxy] = {}
        self._services: Dict[str, Any] = {}

        for name in self.ENGINE_NAMES:
            self._engines[name] = EngineProxy(name, config)

        logger.info(
            f"TaxonomyAppBridge initialized with {len(self._engines)} engine proxies"
        )

    def inject_service(self, engine_name: str, service: Any) -> None:
        """Inject real service into engine proxy."""
        if engine_name in self._engines:
            self._engines[engine_name].inject(service)
            self._services[engine_name] = service
        else:
            logger.warning(f"Unknown engine name: {engine_name}")

    async def health_check(self) -> Dict[str, Any]:
        """Verify bridge connectivity and engine availability."""
        try:
            engine_status = {}
            for name, proxy in self._engines.items():
                result = await proxy.call("health_check")
                engine_status[name] = result.get("status", "unknown")

            return {
                "status": "healthy",
                "app_url": str(self.config.app_base_url),
                "api_version": self.config.api_version,
                "engines": engine_status,
                "services_injected": list(self._services.keys()),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def screen_activities(
        self,
        activities: List[Dict[str, Any]],
        objectives: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Screen economic activities for taxonomy eligibility.

        Args:
            activities: List of activity records with NACE codes
            objectives: Environmental objectives to screen against

        Returns:
            Eligibility screening results per activity
        """
        try:
            result = await self._engines["eligibility"].call(
                "screen_activities",
                activities=activities,
                objectives=objectives or ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"]
            )

            return {
                "total_screened": len(activities),
                "eligible_count": result.get("eligible_count", 0),
                "results": result.get("results", []),
                "provenance_hash": self._calculate_hash(result),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Activity screening failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "total_screened": len(activities),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def assess_alignment(
        self,
        activity_id: str,
        activity_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform full 4-condition alignment assessment for a single activity.

        Evaluates: SC + DNSH + MS + TSC compliance.

        Args:
            activity_id: Taxonomy activity identifier
            activity_data: Optional activity-specific data

        Returns:
            Complete alignment assessment
        """
        try:
            # SC assessment
            sc_result = await self._engines["substantial_contribution"].call(
                "assess",
                activity_id=activity_id,
                data=activity_data or {}
            )

            # DNSH assessment
            dnsh_result = await self._engines["dnsh"].call(
                "assess",
                activity_id=activity_id,
                data=activity_data or {}
            )

            # MS verification
            ms_result = await self._engines["minimum_safeguards"].call(
                "verify",
                activity_id=activity_id,
                data=activity_data or {}
            )

            # TSC evaluation
            tsc_result = await self._engines["tsc"].call(
                "evaluate",
                activity_id=activity_id,
                da_version=self.config.da_version,
                data=activity_data or {}
            )

            sc_pass = sc_result.get("pass", False)
            dnsh_pass = dnsh_result.get("pass", False)
            ms_pass = ms_result.get("pass", False)
            tsc_pass = tsc_result.get("pass", False)

            aligned = sc_pass and dnsh_pass and ms_pass and tsc_pass

            alignment_result = {
                "activity_id": activity_id,
                "aligned": aligned,
                "conditions": {
                    "substantial_contribution": sc_pass,
                    "dnsh": dnsh_pass,
                    "minimum_safeguards": ms_pass,
                    "tsc_compliance": tsc_pass
                },
                "details": {
                    "sc": sc_result,
                    "dnsh": dnsh_result,
                    "ms": ms_result,
                    "tsc": tsc_result
                },
                "provenance_hash": self._calculate_hash({
                    "sc": sc_result, "dnsh": dnsh_result,
                    "ms": ms_result, "tsc": tsc_result
                }),
                "timestamp": datetime.utcnow().isoformat()
            }

            return alignment_result

        except Exception as e:
            logger.error(f"Alignment assessment failed: {str(e)}")
            return {
                "activity_id": activity_id,
                "aligned": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def calculate_kpis(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate taxonomy KPIs (Turnover, CapEx, OpEx alignment ratios).

        Args:
            data: Financial data with activity mapping

        Returns:
            KPI calculation results
        """
        try:
            result = await self._engines["kpi"].call(
                "calculate",
                financial_data=data.get("financial_data", {}),
                activity_mapping=data.get("activity_mapping", {}),
                prevent_double_counting=True
            )

            return {
                "turnover_ratio": result.get("turnover_ratio", 0.0),
                "capex_ratio": result.get("capex_ratio", 0.0),
                "opex_ratio": result.get("opex_ratio", 0.0),
                "eligible_turnover": result.get("eligible_turnover", 0.0),
                "eligible_capex": result.get("eligible_capex", 0.0),
                "eligible_opex": result.get("eligible_opex", 0.0),
                "by_activity": result.get("by_activity", {}),
                "provenance_hash": self._calculate_hash(result),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"KPI calculation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def calculate_gar(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate Green Asset Ratio for financial institutions.

        Args:
            data: Exposure data with counterparty taxonomy information

        Returns:
            GAR/BTAR calculation results
        """
        try:
            result = await self._engines["gar"].call(
                "calculate",
                exposures=data.get("exposures", {}),
                counterparty_data=data.get("counterparty_data", {}),
                include_btar=True
            )

            return {
                "gar_stock": result.get("gar_stock", 0.0),
                "gar_flow": result.get("gar_flow", 0.0),
                "btar": result.get("btar", 0.0),
                "exposure_breakdown": result.get("exposure_breakdown", {}),
                "provenance_hash": self._calculate_hash(result),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"GAR calculation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def generate_disclosures(
        self,
        data: Dict[str, Any],
        disclosure_format: Literal["article_8", "eba_pillar_3", "both"] = "article_8"
    ) -> Dict[str, Any]:
        """
        Generate taxonomy disclosure templates.

        Args:
            data: Assessment and KPI data for disclosure population
            disclosure_format: Target disclosure format

        Returns:
            Generated disclosure templates
        """
        try:
            result = await self._engines["reporting"].call(
                "generate",
                assessment_data=data,
                format=disclosure_format,
                include_nuclear_gas=False
            )

            return {
                "format": disclosure_format,
                "templates": result.get("templates", []),
                "template_count": result.get("template_count", 0),
                "xbrl_tags": result.get("xbrl_tags", []),
                "provenance_hash": self._calculate_hash(result),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Disclosure generation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def classify_activity_type(
        self,
        activity_id: str,
        activity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify activity as enabling, transitional, or standard.

        Args:
            activity_id: Taxonomy activity identifier
            activity_data: Activity context data

        Returns:
            Activity classification result
        """
        try:
            # Check enabling classification (Article 16)
            enabling_result = await self._engines["enabling"].call(
                "classify",
                activity_id=activity_id,
                data=activity_data
            )

            # Check transitional classification (Article 10(2))
            transition_result = await self._engines["transition"].call(
                "classify",
                activity_id=activity_id,
                data=activity_data
            )

            is_enabling = enabling_result.get("is_enabling", False)
            is_transitional = transition_result.get("is_transitional", False)

            if is_enabling:
                classification = "enabling"
            elif is_transitional:
                classification = "transitional"
            else:
                classification = "standard"

            return {
                "activity_id": activity_id,
                "classification": classification,
                "enabling_details": enabling_result,
                "transition_details": transition_result,
                "provenance_hash": self._calculate_hash({
                    "enabling": enabling_result,
                    "transition": transition_result
                }),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Activity classification failed: {str(e)}")
            return {
                "activity_id": activity_id,
                "classification": "unknown",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        import json
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
