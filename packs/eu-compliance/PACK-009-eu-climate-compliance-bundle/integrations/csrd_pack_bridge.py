"""
CSRD Pack Bridge - PACK-009 EU Climate Compliance Bundle

This module routes data to and from PACK-001 CSRD Starter within the bundle.
It maps bundle data format to CSRD pack format and extracts CSRD results
back into the consolidated bundle format.

The bridge handles:
- Bundle -> CSRD data format conversion (30 field mappings)
- CSRD -> Bundle result extraction
- CSRD-specific metric aggregation
- Materiality and ESRS topic mapping

Example:
    >>> config = CSRDPackBridgeConfig(esrs_version="2023")
    >>> bridge = CSRDPackBridge(config)
    >>> await bridge.push_data(bundle_data)
    >>> results = await bridge.pull_results()
    >>> metrics = await bridge.get_csrd_metrics()
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

class CSRDPackBridgeConfig(BaseModel):
    """Configuration for CSRD pack bridge."""

    esrs_version: str = Field(
        default="2023",
        description="ESRS standard version"
    )
    enable_materiality: bool = Field(
        default=True,
        description="Enable double materiality assessment mapping"
    )
    enable_value_chain: bool = Field(
        default=True,
        description="Enable value chain data mapping"
    )
    reporting_period_year: int = Field(
        default=2025,
        ge=2023,
        description="Reporting period fiscal year"
    )
    include_sector_specific: bool = Field(
        default=False,
        description="Include sector-specific ESRS standards"
    )
    data_quality_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum data quality score for mapping"
    )


# ---------------------------------------------------------------------------
# Data mapping
# ---------------------------------------------------------------------------

CSRD_DATA_MAPPING: Dict[str, Dict[str, Any]] = {
    "organization_name": {
        "csrd_field": "undertaking_name",
        "esrs_ref": "ESRS 2 BP-1",
        "data_type": "string",
        "required": True,
    },
    "organization_type": {
        "csrd_field": "undertaking_type",
        "esrs_ref": "ESRS 2 BP-1",
        "data_type": "string",
        "required": True,
    },
    "reporting_period": {
        "csrd_field": "reporting_period",
        "esrs_ref": "ESRS 2 BP-1",
        "data_type": "date_range",
        "required": True,
    },
    "total_revenue": {
        "csrd_field": "net_turnover",
        "esrs_ref": "ESRS 2 BP-2",
        "data_type": "monetary",
        "required": True,
    },
    "employee_count": {
        "csrd_field": "average_number_of_employees",
        "esrs_ref": "ESRS 2 BP-2",
        "data_type": "integer",
        "required": True,
    },
    "total_assets": {
        "csrd_field": "balance_sheet_total",
        "esrs_ref": "ESRS 2 BP-2",
        "data_type": "monetary",
        "required": True,
    },
    "ghg_scope1": {
        "csrd_field": "gross_scope1_ghg_emissions",
        "esrs_ref": "E1-6.44",
        "data_type": "numeric",
        "required": True,
    },
    "ghg_scope2_location": {
        "csrd_field": "gross_scope2_location_ghg_emissions",
        "esrs_ref": "E1-6.46",
        "data_type": "numeric",
        "required": True,
    },
    "ghg_scope2_market": {
        "csrd_field": "gross_scope2_market_ghg_emissions",
        "esrs_ref": "E1-6.47",
        "data_type": "numeric",
        "required": True,
    },
    "ghg_scope3": {
        "csrd_field": "gross_scope3_ghg_emissions",
        "esrs_ref": "E1-6.51",
        "data_type": "numeric",
        "required": False,
    },
    "ghg_intensity": {
        "csrd_field": "ghg_intensity_per_net_revenue",
        "esrs_ref": "E1-6.53",
        "data_type": "numeric",
        "required": False,
    },
    "energy_consumption": {
        "csrd_field": "total_energy_consumption",
        "esrs_ref": "E1-5.37",
        "data_type": "numeric",
        "required": True,
    },
    "renewable_energy_share": {
        "csrd_field": "share_of_renewable_energy",
        "esrs_ref": "E1-5.38",
        "data_type": "percentage",
        "required": False,
    },
    "water_consumption": {
        "csrd_field": "total_water_consumption",
        "esrs_ref": "E3-4",
        "data_type": "numeric",
        "required": False,
    },
    "waste_generated": {
        "csrd_field": "total_waste_generated",
        "esrs_ref": "E5-5",
        "data_type": "numeric",
        "required": False,
    },
    "biodiversity_sites": {
        "csrd_field": "sites_near_biodiversity_areas",
        "esrs_ref": "E4-5",
        "data_type": "integer",
        "required": False,
    },
    "workforce_total": {
        "csrd_field": "total_workforce",
        "esrs_ref": "S1-6",
        "data_type": "integer",
        "required": True,
    },
    "gender_pay_gap": {
        "csrd_field": "gender_pay_gap_percentage",
        "esrs_ref": "S1-16",
        "data_type": "percentage",
        "required": False,
    },
    "board_diversity": {
        "csrd_field": "board_gender_diversity",
        "esrs_ref": "G1-5",
        "data_type": "percentage",
        "required": False,
    },
    "anti_corruption_training": {
        "csrd_field": "anti_corruption_training_coverage",
        "esrs_ref": "G1-3",
        "data_type": "percentage",
        "required": False,
    },
    "supply_chain_due_diligence": {
        "csrd_field": "supply_chain_due_diligence_process",
        "esrs_ref": "G1-2",
        "data_type": "boolean",
        "required": False,
    },
    "transition_plan": {
        "csrd_field": "climate_transition_plan",
        "esrs_ref": "E1-1",
        "data_type": "document",
        "required": False,
    },
    "taxonomy_turnover_ratio": {
        "csrd_field": "taxonomy_aligned_turnover_percentage",
        "esrs_ref": "E1-6.69",
        "data_type": "percentage",
        "required": False,
    },
    "taxonomy_capex_ratio": {
        "csrd_field": "taxonomy_aligned_capex_percentage",
        "esrs_ref": "E1-6.69",
        "data_type": "percentage",
        "required": False,
    },
    "taxonomy_opex_ratio": {
        "csrd_field": "taxonomy_aligned_opex_percentage",
        "esrs_ref": "E1-6.69",
        "data_type": "percentage",
        "required": False,
    },
    "climate_risk_physical": {
        "csrd_field": "physical_climate_risks",
        "esrs_ref": "E1-9",
        "data_type": "json",
        "required": False,
    },
    "climate_risk_transition": {
        "csrd_field": "transition_climate_risks",
        "esrs_ref": "E1-9",
        "data_type": "json",
        "required": False,
    },
    "ghg_reduction_targets": {
        "csrd_field": "ghg_emission_reduction_targets",
        "esrs_ref": "E1-4",
        "data_type": "json",
        "required": False,
    },
    "sbti_validation": {
        "csrd_field": "sbti_target_validation_status",
        "esrs_ref": "E1-4",
        "data_type": "string",
        "required": False,
    },
    "materiality_assessment": {
        "csrd_field": "double_materiality_assessment",
        "esrs_ref": "ESRS 2 IRO-1",
        "data_type": "document",
        "required": True,
    },
}


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class CSRDPackBridge:
    """
    CSRD Pack Bridge for PACK-009 Bundle.

    Routes data to/from PACK-001 CSRD Starter, mapping bundle data format
    to CSRD pack format and extracting CSRD results into the consolidated
    bundle format.

    Example:
        >>> config = CSRDPackBridgeConfig()
        >>> bridge = CSRDPackBridge(config)
        >>> push_result = await bridge.push_data(bundle_data)
        >>> results = await bridge.pull_results()
    """

    def __init__(self, config: CSRDPackBridgeConfig):
        """Initialize CSRD pack bridge."""
        self.config = config
        self._csrd_service: Any = None
        self._pushed_data: Dict[str, Any] = {}
        self._results: Dict[str, Any] = {}
        self._push_timestamp: Optional[str] = None
        logger.info("CSRDPackBridge initialized")

    def inject_service(self, service: Any) -> None:
        """Inject real CSRD pack service."""
        self._csrd_service = service
        logger.info("Injected CSRD pack service")

    async def push_data(self, bundle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Push bundle data to CSRD pack in CSRD format.

        Args:
            bundle_data: Data in bundle format.

        Returns:
            Push result with mapping statistics.
        """
        try:
            if self._csrd_service and hasattr(self._csrd_service, "push_data"):
                mapped = self._map_to_csrd(bundle_data)
                result = await self._csrd_service.push_data(mapped)
                self._pushed_data = mapped
                self._push_timestamp = datetime.utcnow().isoformat()
                return result

            mapped = self._map_to_csrd(bundle_data)
            self._pushed_data = mapped
            self._push_timestamp = datetime.utcnow().isoformat()

            mapped_count = sum(
                1 for v in mapped.values() if v is not None
            )
            total_fields = len(CSRD_DATA_MAPPING)
            required_fields = sum(
                1
                for m in CSRD_DATA_MAPPING.values()
                if m["required"]
            )
            required_mapped = sum(
                1
                for key, meta in CSRD_DATA_MAPPING.items()
                if meta["required"] and mapped.get(meta["csrd_field"]) is not None
            )

            return {
                "status": "success",
                "total_fields": total_fields,
                "mapped_fields": mapped_count,
                "unmapped_fields": total_fields - mapped_count,
                "required_fields": required_fields,
                "required_mapped": required_mapped,
                "mapping_completeness": round(mapped_count / total_fields * 100, 1),
                "timestamp": self._push_timestamp,
                "provenance_hash": self._calculate_hash(mapped),
            }

        except Exception as e:
            logger.error(f"CSRD push_data failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def pull_results(self) -> Dict[str, Any]:
        """
        Pull results from CSRD pack in bundle format.

        Returns:
            CSRD assessment results mapped to bundle format.
        """
        try:
            if self._csrd_service and hasattr(self._csrd_service, "pull_results"):
                raw_results = await self._csrd_service.pull_results()
                self._results = self._map_from_csrd(raw_results)
                return self._results

            # Fallback: return simulated results based on pushed data
            self._results = {
                "pack": "PACK-001 CSRD Starter",
                "status": "completed" if self._pushed_data else "no_data",
                "reporting_year": self.config.reporting_period_year,
                "esrs_version": self.config.esrs_version,
                "compliance_summary": {
                    "e1_climate": "assessed",
                    "e2_pollution": "not_assessed",
                    "e3_water": "partial",
                    "e4_biodiversity": "not_assessed",
                    "e5_circular_economy": "partial",
                    "s1_own_workforce": "assessed",
                    "s2_value_chain_workers": "not_assessed",
                    "s3_affected_communities": "not_assessed",
                    "s4_consumers": "not_assessed",
                    "g1_governance": "assessed",
                },
                "materiality_topics": self._pushed_data.get(
                    "double_materiality_assessment", {}
                ),
                "data_quality_score": 0.75,
                "timestamp": datetime.utcnow().isoformat(),
                "provenance_hash": self._calculate_hash(self._pushed_data),
            }
            return self._results

        except Exception as e:
            logger.error(f"CSRD pull_results failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """
        Get current bridge status.

        Returns:
            Bridge status including push/pull state.
        """
        return {
            "bridge": "CSRDPackBridge",
            "target_pack": "PACK-001 CSRD Starter",
            "service_injected": self._csrd_service is not None,
            "data_pushed": bool(self._pushed_data),
            "results_available": bool(self._results),
            "push_timestamp": self._push_timestamp,
            "config": {
                "esrs_version": self.config.esrs_version,
                "materiality_enabled": self.config.enable_materiality,
                "value_chain_enabled": self.config.enable_value_chain,
                "reporting_year": self.config.reporting_period_year,
            },
            "mapping_stats": {
                "total_mappings": len(CSRD_DATA_MAPPING),
                "required_mappings": sum(
                    1 for m in CSRD_DATA_MAPPING.values() if m["required"]
                ),
                "optional_mappings": sum(
                    1 for m in CSRD_DATA_MAPPING.values() if not m["required"]
                ),
            },
        }

    async def get_csrd_metrics(self) -> Dict[str, Any]:
        """
        Get CSRD-specific metrics from the latest results.

        Returns:
            Aggregated CSRD metrics.
        """
        try:
            if self._csrd_service and hasattr(self._csrd_service, "get_metrics"):
                return await self._csrd_service.get_metrics()

            if not self._results:
                return {"status": "no_results", "message": "No results available yet"}

            compliance = self._results.get("compliance_summary", {})
            assessed = sum(
                1 for v in compliance.values() if v == "assessed"
            )
            partial = sum(
                1 for v in compliance.values() if v == "partial"
            )
            not_assessed = sum(
                1 for v in compliance.values() if v == "not_assessed"
            )

            return {
                "pack": "PACK-001 CSRD Starter",
                "esrs_version": self.config.esrs_version,
                "reporting_year": self.config.reporting_period_year,
                "topics_assessed": assessed,
                "topics_partial": partial,
                "topics_not_assessed": not_assessed,
                "total_topics": assessed + partial + not_assessed,
                "assessment_coverage": (
                    round(
                        (assessed + partial * 0.5)
                        / (assessed + partial + not_assessed)
                        * 100,
                        1,
                    )
                    if (assessed + partial + not_assessed) > 0
                    else 0.0
                ),
                "data_quality_score": self._results.get("data_quality_score", 0.0),
                "ghg_emissions": {
                    "scope1": self._pushed_data.get("gross_scope1_ghg_emissions"),
                    "scope2_location": self._pushed_data.get(
                        "gross_scope2_location_ghg_emissions"
                    ),
                    "scope2_market": self._pushed_data.get(
                        "gross_scope2_market_ghg_emissions"
                    ),
                    "scope3": self._pushed_data.get("gross_scope3_ghg_emissions"),
                },
                "taxonomy_kpis": {
                    "turnover_ratio": self._pushed_data.get(
                        "taxonomy_aligned_turnover_percentage"
                    ),
                    "capex_ratio": self._pushed_data.get(
                        "taxonomy_aligned_capex_percentage"
                    ),
                    "opex_ratio": self._pushed_data.get(
                        "taxonomy_aligned_opex_percentage"
                    ),
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"CSRD get_csrd_metrics failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_to_csrd(self, bundle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map bundle data format to CSRD pack format using CSRD_DATA_MAPPING."""
        mapped: Dict[str, Any] = {}
        for bundle_key, mapping in CSRD_DATA_MAPPING.items():
            csrd_field = mapping["csrd_field"]
            value = bundle_data.get(bundle_key)
            if value is not None:
                mapped[csrd_field] = value
            elif mapping["required"]:
                mapped[csrd_field] = None
                logger.warning(
                    f"Required CSRD field missing: {bundle_key} -> {csrd_field}"
                )
        return mapped

    def _map_from_csrd(self, csrd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map CSRD results back to bundle format."""
        result: Dict[str, Any] = {
            "pack": "PACK-001 CSRD Starter",
            "status": "completed",
        }
        reverse_map = {
            v["csrd_field"]: k for k, v in CSRD_DATA_MAPPING.items()
        }
        for csrd_field, value in csrd_data.items():
            bundle_key = reverse_map.get(csrd_field)
            if bundle_key:
                result[bundle_key] = value
            else:
                result[csrd_field] = value
        return result

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
