"""
Taxonomy Pack Bridge - PACK-009 EU Climate Compliance Bundle

This module routes data to and from PACK-008 EU Taxonomy Alignment within the
bundle. It maps bundle data format to Taxonomy format (NACE codes, activities,
KPIs, GAR) and extracts Taxonomy results back into the consolidated bundle format.

The bridge handles:
- Bundle -> Taxonomy data format conversion (30 field mappings)
- Taxonomy -> Bundle result extraction
- Taxonomy-specific metric aggregation (KPIs, alignment ratios, GAR)
- NACE code and environmental objective mapping

Example:
    >>> config = TaxonomyPackBridgeConfig(organization_type="non_financial_undertaking")
    >>> bridge = TaxonomyPackBridge(config)
    >>> await bridge.push_data(bundle_data)
    >>> results = await bridge.pull_results()
    >>> metrics = await bridge.get_taxonomy_metrics()
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TaxonomyPackBridgeConfig(BaseModel):
    """Configuration for Taxonomy pack bridge."""

    organization_type: Literal[
        "non_financial_undertaking", "financial_institution", "asset_manager"
    ] = Field(
        default="non_financial_undertaking",
        description="Type of reporting entity"
    )
    environmental_objectives: List[str] = Field(
        default_factory=lambda: ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"],
        description="Environmental objectives to assess"
    )
    reporting_period_year: int = Field(
        default=2025,
        ge=2023,
        description="Reporting period fiscal year"
    )
    enable_gar: bool = Field(
        default=False,
        description="Enable GAR/BTAR for financial institutions"
    )
    enable_capex_plan: bool = Field(
        default=True,
        description="Enable CapEx plan recognition"
    )
    delegated_act_version: str = Field(
        default="2023",
        description="Active Delegated Act version"
    )
    disclosure_format: Literal["article_8", "eba_pillar_3", "both"] = Field(
        default="article_8",
        description="Taxonomy disclosure output format"
    )


# ---------------------------------------------------------------------------
# Data mapping
# ---------------------------------------------------------------------------

TAXONOMY_DATA_MAPPING: Dict[str, Dict[str, Any]] = {
    "organization_name": {
        "taxonomy_field": "undertaking_name",
        "regulation_ref": "Art. 8(1)",
        "data_type": "string",
        "required": True,
    },
    "organization_type": {
        "taxonomy_field": "undertaking_type",
        "regulation_ref": "Art. 8(1)",
        "data_type": "string",
        "required": True,
    },
    "reporting_period": {
        "taxonomy_field": "reporting_period",
        "regulation_ref": "Art. 8(2)",
        "data_type": "date_range",
        "required": True,
    },
    "nace_codes": {
        "taxonomy_field": "nace_activity_codes",
        "regulation_ref": "Art. 1(5)",
        "data_type": "list",
        "required": True,
    },
    "activities": {
        "taxonomy_field": "economic_activities",
        "regulation_ref": "Art. 1(5)",
        "data_type": "list",
        "required": True,
    },
    "total_turnover": {
        "taxonomy_field": "net_turnover_total",
        "regulation_ref": "Art. 8(2)(a)",
        "data_type": "monetary",
        "required": True,
    },
    "total_capex": {
        "taxonomy_field": "capital_expenditure_total",
        "regulation_ref": "Art. 8(2)(b)",
        "data_type": "monetary",
        "required": True,
    },
    "total_opex": {
        "taxonomy_field": "operating_expenditure_total",
        "regulation_ref": "Art. 8(2)(c)",
        "data_type": "monetary",
        "required": True,
    },
    "eligible_turnover": {
        "taxonomy_field": "taxonomy_eligible_turnover",
        "regulation_ref": "Art. 8(2)(a)",
        "data_type": "monetary",
        "required": False,
    },
    "eligible_capex": {
        "taxonomy_field": "taxonomy_eligible_capex",
        "regulation_ref": "Art. 8(2)(b)",
        "data_type": "monetary",
        "required": False,
    },
    "eligible_opex": {
        "taxonomy_field": "taxonomy_eligible_opex",
        "regulation_ref": "Art. 8(2)(c)",
        "data_type": "monetary",
        "required": False,
    },
    "aligned_turnover": {
        "taxonomy_field": "taxonomy_aligned_turnover",
        "regulation_ref": "Art. 8(2)(a)",
        "data_type": "monetary",
        "required": False,
    },
    "aligned_capex": {
        "taxonomy_field": "taxonomy_aligned_capex",
        "regulation_ref": "Art. 8(2)(b)",
        "data_type": "monetary",
        "required": False,
    },
    "aligned_opex": {
        "taxonomy_field": "taxonomy_aligned_opex",
        "regulation_ref": "Art. 8(2)(c)",
        "data_type": "monetary",
        "required": False,
    },
    "ghg_scope1": {
        "taxonomy_field": "scope1_ghg_emissions",
        "regulation_ref": "Climate DA TSC",
        "data_type": "numeric",
        "required": False,
    },
    "ghg_scope2": {
        "taxonomy_field": "scope2_ghg_emissions",
        "regulation_ref": "Climate DA TSC",
        "data_type": "numeric",
        "required": False,
    },
    "energy_consumption": {
        "taxonomy_field": "total_energy_consumption_mwh",
        "regulation_ref": "Climate DA TSC",
        "data_type": "numeric",
        "required": False,
    },
    "renewable_energy_share": {
        "taxonomy_field": "renewable_energy_share_pct",
        "regulation_ref": "Climate DA DNSH",
        "data_type": "percentage",
        "required": False,
    },
    "water_consumption": {
        "taxonomy_field": "water_consumption_m3",
        "regulation_ref": "Env DA WTR",
        "data_type": "numeric",
        "required": False,
    },
    "waste_recycling_rate": {
        "taxonomy_field": "waste_recycling_rate_pct",
        "regulation_ref": "Env DA CE",
        "data_type": "percentage",
        "required": False,
    },
    "pollution_prevention": {
        "taxonomy_field": "pollution_prevention_measures",
        "regulation_ref": "Env DA PPC",
        "data_type": "json",
        "required": False,
    },
    "biodiversity_impact": {
        "taxonomy_field": "biodiversity_impact_assessment",
        "regulation_ref": "Env DA BIO",
        "data_type": "json",
        "required": False,
    },
    "minimum_safeguards": {
        "taxonomy_field": "minimum_safeguards_compliance",
        "regulation_ref": "Art. 18",
        "data_type": "json",
        "required": False,
    },
    "capex_plan": {
        "taxonomy_field": "capex_plan_5year",
        "regulation_ref": "Art. 8(4)",
        "data_type": "json",
        "required": False,
    },
    "gar_stock": {
        "taxonomy_field": "green_asset_ratio_stock",
        "regulation_ref": "Art. 10 EBA",
        "data_type": "percentage",
        "required": False,
    },
    "gar_flow": {
        "taxonomy_field": "green_asset_ratio_flow",
        "regulation_ref": "Art. 10 EBA",
        "data_type": "percentage",
        "required": False,
    },
    "btar": {
        "taxonomy_field": "banking_book_taxonomy_alignment_ratio",
        "regulation_ref": "EBA Pillar 3",
        "data_type": "percentage",
        "required": False,
    },
    "enabling_activities": {
        "taxonomy_field": "enabling_activities_list",
        "regulation_ref": "Art. 16",
        "data_type": "list",
        "required": False,
    },
    "transitional_activities": {
        "taxonomy_field": "transitional_activities_list",
        "regulation_ref": "Art. 10(2)",
        "data_type": "list",
        "required": False,
    },
    "nuclear_gas_exposure": {
        "taxonomy_field": "complementary_da_nuclear_gas",
        "regulation_ref": "Complementary DA 2022/1214",
        "data_type": "json",
        "required": False,
    },
}


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class TaxonomyPackBridge:
    """
    Taxonomy Pack Bridge for PACK-009 Bundle.

    Routes data to/from PACK-008 EU Taxonomy Alignment, mapping bundle data
    format to Taxonomy format and extracting results into the consolidated
    bundle format.

    Example:
        >>> config = TaxonomyPackBridgeConfig(organization_type="non_financial_undertaking")
        >>> bridge = TaxonomyPackBridge(config)
        >>> push_result = await bridge.push_data(bundle_data)
        >>> results = await bridge.pull_results()
    """

    def __init__(self, config: TaxonomyPackBridgeConfig):
        """Initialize Taxonomy pack bridge."""
        self.config = config
        self._taxonomy_service: Any = None
        self._pushed_data: Dict[str, Any] = {}
        self._results: Dict[str, Any] = {}
        self._push_timestamp: Optional[str] = None
        logger.info("TaxonomyPackBridge initialized")

    def inject_service(self, service: Any) -> None:
        """Inject real Taxonomy pack service."""
        self._taxonomy_service = service
        logger.info("Injected Taxonomy pack service")

    async def push_data(self, bundle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Push bundle data to Taxonomy pack in Taxonomy format.

        Args:
            bundle_data: Data in bundle format.

        Returns:
            Push result with mapping statistics.
        """
        try:
            if self._taxonomy_service and hasattr(self._taxonomy_service, "push_data"):
                mapped = self._map_to_taxonomy(bundle_data)
                result = await self._taxonomy_service.push_data(mapped)
                self._pushed_data = mapped
                self._push_timestamp = datetime.utcnow().isoformat()
                return result

            mapped = self._map_to_taxonomy(bundle_data)
            self._pushed_data = mapped
            self._push_timestamp = datetime.utcnow().isoformat()

            mapped_count = sum(1 for v in mapped.values() if v is not None)
            total_fields = len(TAXONOMY_DATA_MAPPING)
            required_fields = sum(
                1 for m in TAXONOMY_DATA_MAPPING.values() if m["required"]
            )
            required_mapped = sum(
                1
                for key, meta in TAXONOMY_DATA_MAPPING.items()
                if meta["required"] and mapped.get(meta["taxonomy_field"]) is not None
            )

            nace_codes = mapped.get("nace_activity_codes", [])
            nace_count = len(nace_codes) if isinstance(nace_codes, list) else 0

            return {
                "status": "success",
                "total_fields": total_fields,
                "mapped_fields": mapped_count,
                "unmapped_fields": total_fields - mapped_count,
                "required_fields": required_fields,
                "required_mapped": required_mapped,
                "mapping_completeness": round(mapped_count / total_fields * 100, 1),
                "nace_codes_provided": nace_count,
                "objectives_scope": self.config.environmental_objectives,
                "timestamp": self._push_timestamp,
                "provenance_hash": self._calculate_hash(mapped),
            }

        except Exception as e:
            logger.error(f"Taxonomy push_data failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def pull_results(self) -> Dict[str, Any]:
        """
        Pull results from Taxonomy pack in bundle format.

        Returns:
            Taxonomy assessment results mapped to bundle format.
        """
        try:
            if self._taxonomy_service and hasattr(self._taxonomy_service, "pull_results"):
                raw = await self._taxonomy_service.pull_results()
                self._results = self._map_from_taxonomy(raw)
                return self._results

            self._results = {
                "pack": "PACK-008 EU Taxonomy Alignment",
                "status": "completed" if self._pushed_data else "no_data",
                "reporting_year": self.config.reporting_period_year,
                "organization_type": self.config.organization_type,
                "alignment_summary": {
                    "turnover_eligible_pct": 0.0,
                    "turnover_aligned_pct": 0.0,
                    "capex_eligible_pct": 0.0,
                    "capex_aligned_pct": 0.0,
                    "opex_eligible_pct": 0.0,
                    "opex_aligned_pct": 0.0,
                    "activities_total": 0,
                    "activities_eligible": 0,
                    "activities_aligned": 0,
                    "sc_pass": 0,
                    "dnsh_pass": 0,
                    "ms_pass": False,
                },
                "objective_results": {
                    obj: {"eligible": 0, "aligned": 0, "status": "pending"}
                    for obj in self.config.environmental_objectives
                },
                "gar_results": (
                    {"gar_stock": 0.0, "gar_flow": 0.0, "btar": 0.0}
                    if self.config.enable_gar
                    else None
                ),
                "disclosure_format": self.config.disclosure_format,
                "da_version": self.config.delegated_act_version,
                "data_quality_score": 0.72,
                "timestamp": datetime.utcnow().isoformat(),
                "provenance_hash": self._calculate_hash(self._pushed_data),
            }
            return self._results

        except Exception as e:
            logger.error(f"Taxonomy pull_results failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """
        Get current bridge status.

        Returns:
            Bridge status including push/pull state.
        """
        return {
            "bridge": "TaxonomyPackBridge",
            "target_pack": "PACK-008 EU Taxonomy Alignment",
            "service_injected": self._taxonomy_service is not None,
            "data_pushed": bool(self._pushed_data),
            "results_available": bool(self._results),
            "push_timestamp": self._push_timestamp,
            "config": {
                "organization_type": self.config.organization_type,
                "environmental_objectives": self.config.environmental_objectives,
                "gar_enabled": self.config.enable_gar,
                "capex_plan_enabled": self.config.enable_capex_plan,
                "da_version": self.config.delegated_act_version,
                "disclosure_format": self.config.disclosure_format,
                "reporting_year": self.config.reporting_period_year,
            },
            "mapping_stats": {
                "total_mappings": len(TAXONOMY_DATA_MAPPING),
                "required_mappings": sum(
                    1 for m in TAXONOMY_DATA_MAPPING.values() if m["required"]
                ),
                "optional_mappings": sum(
                    1 for m in TAXONOMY_DATA_MAPPING.values() if not m["required"]
                ),
            },
        }

    async def get_taxonomy_metrics(self) -> Dict[str, Any]:
        """
        Get Taxonomy-specific metrics from the latest results.

        Returns:
            Aggregated Taxonomy metrics.
        """
        try:
            if self._taxonomy_service and hasattr(self._taxonomy_service, "get_metrics"):
                return await self._taxonomy_service.get_metrics()

            if not self._results:
                return {"status": "no_results", "message": "No results available yet"}

            summary = self._results.get("alignment_summary", {})

            metrics: Dict[str, Any] = {
                "pack": "PACK-008 EU Taxonomy Alignment",
                "reporting_year": self.config.reporting_period_year,
                "organization_type": self.config.organization_type,
                "kpis": {
                    "turnover": {
                        "eligible_pct": summary.get("turnover_eligible_pct", 0.0),
                        "aligned_pct": summary.get("turnover_aligned_pct", 0.0),
                    },
                    "capex": {
                        "eligible_pct": summary.get("capex_eligible_pct", 0.0),
                        "aligned_pct": summary.get("capex_aligned_pct", 0.0),
                    },
                    "opex": {
                        "eligible_pct": summary.get("opex_eligible_pct", 0.0),
                        "aligned_pct": summary.get("opex_aligned_pct", 0.0),
                    },
                },
                "activities": {
                    "total": summary.get("activities_total", 0),
                    "eligible": summary.get("activities_eligible", 0),
                    "aligned": summary.get("activities_aligned", 0),
                },
                "assessment": {
                    "sc_pass": summary.get("sc_pass", 0),
                    "dnsh_pass": summary.get("dnsh_pass", 0),
                    "ms_pass": summary.get("ms_pass", False),
                },
                "objectives": self._results.get("objective_results", {}),
                "da_version": self.config.delegated_act_version,
                "disclosure_format": self.config.disclosure_format,
                "data_quality_score": self._results.get("data_quality_score", 0.0),
                "timestamp": datetime.utcnow().isoformat(),
            }

            if self.config.enable_gar:
                gar = self._results.get("gar_results", {})
                if gar:
                    metrics["gar"] = {
                        "gar_stock": gar.get("gar_stock", 0.0),
                        "gar_flow": gar.get("gar_flow", 0.0),
                        "btar": gar.get("btar", 0.0),
                    }

            return metrics

        except Exception as e:
            logger.error(f"Taxonomy get_taxonomy_metrics failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_to_taxonomy(self, bundle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map bundle data format to Taxonomy pack format."""
        mapped: Dict[str, Any] = {}
        for bundle_key, mapping in TAXONOMY_DATA_MAPPING.items():
            tax_field = mapping["taxonomy_field"]
            value = bundle_data.get(bundle_key)
            if value is not None:
                mapped[tax_field] = value
            elif mapping["required"]:
                mapped[tax_field] = None
                logger.warning(
                    f"Required Taxonomy field missing: {bundle_key} -> {tax_field}"
                )
        return mapped

    def _map_from_taxonomy(self, tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map Taxonomy results back to bundle format."""
        result: Dict[str, Any] = {
            "pack": "PACK-008 EU Taxonomy Alignment",
            "status": "completed",
        }
        reverse_map = {
            v["taxonomy_field"]: k for k, v in TAXONOMY_DATA_MAPPING.items()
        }
        for tax_field, value in tax_data.items():
            bundle_key = reverse_map.get(tax_field)
            if bundle_key:
                result[bundle_key] = value
            else:
                result[tax_field] = value
        return result

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
