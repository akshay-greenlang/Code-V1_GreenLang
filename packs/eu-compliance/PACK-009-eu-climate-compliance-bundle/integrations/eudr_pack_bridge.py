"""
EUDR Pack Bridge - PACK-009 EU Climate Compliance Bundle

This module routes data to and from PACK-006 EUDR Starter within the bundle.
It maps bundle data format to EUDR format (traceability, due diligence
statements, geolocation) and extracts EUDR results back into the
consolidated bundle format.

The bridge handles:
- Bundle -> EUDR data format conversion (25 field mappings)
- EUDR -> Bundle result extraction
- EUDR-specific metric aggregation (commodities, traceability, risk)
- Geolocation and supply chain mapping

Example:
    >>> config = EUDRPackBridgeConfig(commodities=["soy", "palm_oil"])
    >>> bridge = EUDRPackBridge(config)
    >>> await bridge.push_data(bundle_data)
    >>> results = await bridge.pull_results()
    >>> metrics = await bridge.get_eudr_metrics()
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

class EUDRPackBridgeConfig(BaseModel):
    """Configuration for EUDR pack bridge."""

    commodities: List[str] = Field(
        default_factory=lambda: [
            "soy", "palm_oil", "wood", "cocoa", "coffee", "rubber", "cattle"
        ],
        description="EUDR regulated commodities to assess"
    )
    enable_geolocation: bool = Field(
        default=True,
        description="Enable geolocation data mapping"
    )
    enable_risk_assessment: bool = Field(
        default=True,
        description="Enable country/region risk assessment"
    )
    reporting_period_year: int = Field(
        default=2025,
        ge=2023,
        description="Reporting period fiscal year"
    )
    deforestation_cutoff_date: str = Field(
        default="2020-12-31",
        description="Deforestation-free cutoff date per EUDR"
    )
    operator_type: Literal["operator", "trader", "sme_trader"] = Field(
        default="operator",
        description="Type of entity under EUDR"
    )


# ---------------------------------------------------------------------------
# Data mapping
# ---------------------------------------------------------------------------

EUDR_DATA_MAPPING: Dict[str, Dict[str, Any]] = {
    "operator_name": {
        "eudr_field": "operator_legal_name",
        "regulation_ref": "Art. 4(1)",
        "data_type": "string",
        "required": True,
    },
    "operator_country": {
        "eudr_field": "operator_country_of_establishment",
        "regulation_ref": "Art. 4(1)",
        "data_type": "string",
        "required": True,
    },
    "operator_registration": {
        "eudr_field": "operator_registration_number",
        "regulation_ref": "Art. 4(2)",
        "data_type": "string",
        "required": True,
    },
    "commodity_type": {
        "eudr_field": "relevant_commodity",
        "regulation_ref": "Art. 1",
        "data_type": "string",
        "required": True,
    },
    "product_description": {
        "eudr_field": "product_description",
        "regulation_ref": "Art. 4(2)(b)",
        "data_type": "string",
        "required": True,
    },
    "product_hs_code": {
        "eudr_field": "harmonised_system_code",
        "regulation_ref": "Annex I",
        "data_type": "string",
        "required": True,
    },
    "quantity": {
        "eudr_field": "quantity_of_product",
        "regulation_ref": "Art. 4(2)(c)",
        "data_type": "numeric",
        "required": True,
    },
    "quantity_unit": {
        "eudr_field": "quantity_unit_of_measurement",
        "regulation_ref": "Art. 4(2)(c)",
        "data_type": "string",
        "required": True,
    },
    "supplier_name": {
        "eudr_field": "supplier_legal_name",
        "regulation_ref": "Art. 4(2)(d)",
        "data_type": "string",
        "required": True,
    },
    "supplier_country": {
        "eudr_field": "supplier_country_of_origin",
        "regulation_ref": "Art. 4(2)(d)",
        "data_type": "string",
        "required": True,
    },
    "production_country": {
        "eudr_field": "country_of_production",
        "regulation_ref": "Art. 4(2)(e)",
        "data_type": "string",
        "required": True,
    },
    "geolocation_data": {
        "eudr_field": "geolocation_coordinates",
        "regulation_ref": "Art. 4(2)(f)",
        "data_type": "geojson",
        "required": True,
    },
    "plot_of_land": {
        "eudr_field": "plot_of_land_identification",
        "regulation_ref": "Art. 4(2)(f)",
        "data_type": "json",
        "required": False,
    },
    "production_date": {
        "eudr_field": "date_of_production",
        "regulation_ref": "Art. 4(2)(g)",
        "data_type": "date",
        "required": True,
    },
    "deforestation_free": {
        "eudr_field": "deforestation_free_declaration",
        "regulation_ref": "Art. 3(a)",
        "data_type": "boolean",
        "required": True,
    },
    "legal_compliance": {
        "eudr_field": "legal_compliance_declaration",
        "regulation_ref": "Art. 3(b)",
        "data_type": "boolean",
        "required": True,
    },
    "due_diligence_statement": {
        "eudr_field": "due_diligence_statement_reference",
        "regulation_ref": "Art. 4(1)",
        "data_type": "string",
        "required": True,
    },
    "risk_assessment_result": {
        "eudr_field": "risk_assessment_outcome",
        "regulation_ref": "Art. 10",
        "data_type": "string",
        "required": False,
    },
    "risk_level": {
        "eudr_field": "country_risk_classification",
        "regulation_ref": "Art. 29",
        "data_type": "string",
        "required": False,
    },
    "mitigation_measures": {
        "eudr_field": "risk_mitigation_measures_applied",
        "regulation_ref": "Art. 11",
        "data_type": "list",
        "required": False,
    },
    "satellite_verification": {
        "eudr_field": "satellite_monitoring_data",
        "regulation_ref": "Art. 10(2)",
        "data_type": "json",
        "required": False,
    },
    "certification_scheme": {
        "eudr_field": "voluntary_certification_scheme",
        "regulation_ref": "Art. 12",
        "data_type": "string",
        "required": False,
    },
    "supply_chain_map": {
        "eudr_field": "supply_chain_traceability_map",
        "regulation_ref": "Art. 9(1)(d)",
        "data_type": "json",
        "required": False,
    },
    "forest_degradation_free": {
        "eudr_field": "forest_degradation_free_declaration",
        "regulation_ref": "Art. 3(a)",
        "data_type": "boolean",
        "required": True,
    },
    "third_party_verification": {
        "eudr_field": "independent_verification_report",
        "regulation_ref": "Art. 10(6)",
        "data_type": "document",
        "required": False,
    },
}


# ---------------------------------------------------------------------------
# EUDR commodities and HS codes reference
# ---------------------------------------------------------------------------

EUDR_COMMODITIES_HS: Dict[str, List[str]] = {
    "soy": ["1201", "1208 10", "1507", "2304"],
    "palm_oil": ["1511", "1513 21", "1513 29"],
    "wood": ["4401", "4403", "4407", "4408", "4409", "4410", "4411", "4412", "9401"],
    "cocoa": ["1801 00 00", "1802 00 00", "1803", "1804 00 00", "1805 00 00", "1806"],
    "coffee": ["0901", "2101 11"],
    "rubber": ["4001", "4005", "4011", "4012", "4013"],
    "cattle": ["0102", "0201", "0202", "4101", "4104", "4107", "4113", "4114"],
}


# ---------------------------------------------------------------------------
# Country risk classifications (simplified reference)
# ---------------------------------------------------------------------------

EUDR_RISK_CLASSIFICATIONS: Dict[str, str] = {
    "high": "Enhanced due diligence required; 15% supply chain checks",
    "standard": "Standard due diligence required; 3% supply chain checks",
    "low": "Simplified due diligence; 1% supply chain checks",
}


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class EUDRPackBridge:
    """
    EUDR Pack Bridge for PACK-009 Bundle.

    Routes data to/from PACK-006 EUDR Starter, mapping bundle data format
    to EUDR format and extracting EUDR results into the consolidated
    bundle format.

    Example:
        >>> config = EUDRPackBridgeConfig(commodities=["soy", "palm_oil"])
        >>> bridge = EUDRPackBridge(config)
        >>> push_result = await bridge.push_data(bundle_data)
        >>> results = await bridge.pull_results()
    """

    def __init__(self, config: EUDRPackBridgeConfig):
        """Initialize EUDR pack bridge."""
        self.config = config
        self._eudr_service: Any = None
        self._pushed_data: Dict[str, Any] = {}
        self._results: Dict[str, Any] = {}
        self._push_timestamp: Optional[str] = None
        logger.info("EUDRPackBridge initialized")

    def inject_service(self, service: Any) -> None:
        """Inject real EUDR pack service."""
        self._eudr_service = service
        logger.info("Injected EUDR pack service")

    async def push_data(self, bundle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Push bundle data to EUDR pack in EUDR format.

        Args:
            bundle_data: Data in bundle format.

        Returns:
            Push result with mapping statistics.
        """
        try:
            if self._eudr_service and hasattr(self._eudr_service, "push_data"):
                mapped = self._map_to_eudr(bundle_data)
                result = await self._eudr_service.push_data(mapped)
                self._pushed_data = mapped
                self._push_timestamp = datetime.utcnow().isoformat()
                return result

            mapped = self._map_to_eudr(bundle_data)
            self._pushed_data = mapped
            self._push_timestamp = datetime.utcnow().isoformat()

            mapped_count = sum(1 for v in mapped.values() if v is not None)
            total_fields = len(EUDR_DATA_MAPPING)
            required_fields = sum(
                1 for m in EUDR_DATA_MAPPING.values() if m["required"]
            )
            required_mapped = sum(
                1
                for key, meta in EUDR_DATA_MAPPING.items()
                if meta["required"] and mapped.get(meta["eudr_field"]) is not None
            )

            commodities_found = self._identify_commodities(mapped)

            return {
                "status": "success",
                "total_fields": total_fields,
                "mapped_fields": mapped_count,
                "unmapped_fields": total_fields - mapped_count,
                "required_fields": required_fields,
                "required_mapped": required_mapped,
                "mapping_completeness": round(mapped_count / total_fields * 100, 1),
                "commodities_identified": commodities_found,
                "operator_type": self.config.operator_type,
                "timestamp": self._push_timestamp,
                "provenance_hash": self._calculate_hash(mapped),
            }

        except Exception as e:
            logger.error(f"EUDR push_data failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def pull_results(self) -> Dict[str, Any]:
        """
        Pull results from EUDR pack in bundle format.

        Returns:
            EUDR assessment results mapped to bundle format.
        """
        try:
            if self._eudr_service and hasattr(self._eudr_service, "pull_results"):
                raw = await self._eudr_service.pull_results()
                self._results = self._map_from_eudr(raw)
                return self._results

            self._results = {
                "pack": "PACK-006 EUDR Starter",
                "status": "completed" if self._pushed_data else "no_data",
                "reporting_year": self.config.reporting_period_year,
                "operator_type": self.config.operator_type,
                "assessment_summary": {
                    "commodities_assessed": len(self.config.commodities),
                    "supply_chains_mapped": 0,
                    "geolocation_plots": 0,
                    "due_diligence_statements": 0,
                    "deforestation_free_verified": False,
                    "legal_compliance_verified": False,
                    "risk_level": "standard",
                },
                "commodity_details": {
                    c: {"status": "pending", "risk": "standard"}
                    for c in self.config.commodities
                },
                "traceability_score": 0.0,
                "data_quality_score": 0.65,
                "timestamp": datetime.utcnow().isoformat(),
                "provenance_hash": self._calculate_hash(self._pushed_data),
            }
            return self._results

        except Exception as e:
            logger.error(f"EUDR pull_results failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """
        Get current bridge status.

        Returns:
            Bridge status including push/pull state.
        """
        return {
            "bridge": "EUDRPackBridge",
            "target_pack": "PACK-006 EUDR Starter",
            "service_injected": self._eudr_service is not None,
            "data_pushed": bool(self._pushed_data),
            "results_available": bool(self._results),
            "push_timestamp": self._push_timestamp,
            "config": {
                "commodities": self.config.commodities,
                "geolocation_enabled": self.config.enable_geolocation,
                "risk_assessment_enabled": self.config.enable_risk_assessment,
                "operator_type": self.config.operator_type,
                "cutoff_date": self.config.deforestation_cutoff_date,
                "reporting_year": self.config.reporting_period_year,
            },
            "mapping_stats": {
                "total_mappings": len(EUDR_DATA_MAPPING),
                "required_mappings": sum(
                    1 for m in EUDR_DATA_MAPPING.values() if m["required"]
                ),
                "optional_mappings": sum(
                    1 for m in EUDR_DATA_MAPPING.values() if not m["required"]
                ),
            },
        }

    async def get_eudr_metrics(self) -> Dict[str, Any]:
        """
        Get EUDR-specific metrics from the latest results.

        Returns:
            Aggregated EUDR metrics.
        """
        try:
            if self._eudr_service and hasattr(self._eudr_service, "get_metrics"):
                return await self._eudr_service.get_metrics()

            if not self._results:
                return {"status": "no_results", "message": "No results available yet"}

            summary = self._results.get("assessment_summary", {})

            return {
                "pack": "PACK-006 EUDR Starter",
                "reporting_year": self.config.reporting_period_year,
                "operator_type": self.config.operator_type,
                "commodities": {
                    "total": len(self.config.commodities),
                    "assessed": summary.get("commodities_assessed", 0),
                    "list": self.config.commodities,
                    "details": self._results.get("commodity_details", {}),
                },
                "traceability": {
                    "supply_chains_mapped": summary.get("supply_chains_mapped", 0),
                    "geolocation_plots": summary.get("geolocation_plots", 0),
                    "traceability_score": self._results.get("traceability_score", 0.0),
                },
                "due_diligence": {
                    "statements_filed": summary.get("due_diligence_statements", 0),
                    "deforestation_free": summary.get(
                        "deforestation_free_verified", False
                    ),
                    "legal_compliance": summary.get(
                        "legal_compliance_verified", False
                    ),
                },
                "risk": {
                    "overall_level": summary.get("risk_level", "standard"),
                    "classifications": EUDR_RISK_CLASSIFICATIONS,
                },
                "data_quality_score": self._results.get("data_quality_score", 0.0),
                "cutoff_date": self.config.deforestation_cutoff_date,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"EUDR get_eudr_metrics failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_to_eudr(self, bundle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map bundle data format to EUDR pack format."""
        mapped: Dict[str, Any] = {}
        for bundle_key, mapping in EUDR_DATA_MAPPING.items():
            eudr_field = mapping["eudr_field"]
            value = bundle_data.get(bundle_key)
            if value is not None:
                mapped[eudr_field] = value
            elif mapping["required"]:
                mapped[eudr_field] = None
                logger.warning(
                    f"Required EUDR field missing: {bundle_key} -> {eudr_field}"
                )
        return mapped

    def _map_from_eudr(self, eudr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map EUDR results back to bundle format."""
        result: Dict[str, Any] = {
            "pack": "PACK-006 EUDR Starter",
            "status": "completed",
        }
        reverse_map = {
            v["eudr_field"]: k for k, v in EUDR_DATA_MAPPING.items()
        }
        for eudr_field, value in eudr_data.items():
            bundle_key = reverse_map.get(eudr_field)
            if bundle_key:
                result[bundle_key] = value
            else:
                result[eudr_field] = value
        return result

    def _identify_commodities(self, mapped_data: Dict[str, Any]) -> List[str]:
        """Identify EUDR commodities from mapped data."""
        commodity = mapped_data.get("relevant_commodity")
        if isinstance(commodity, str) and commodity in EUDR_COMMODITIES_HS:
            return [commodity]

        hs_code = mapped_data.get("harmonised_system_code", "")
        found: List[str] = []
        if isinstance(hs_code, str):
            for comm, codes in EUDR_COMMODITIES_HS.items():
                for code in codes:
                    code_stripped = code.replace(" ", "")
                    if hs_code.replace(" ", "").startswith(code_stripped):
                        if comm not in found:
                            found.append(comm)
        return found if found else list(self.config.commodities)

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
