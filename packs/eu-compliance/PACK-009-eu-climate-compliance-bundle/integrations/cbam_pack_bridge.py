"""
CBAM Pack Bridge - PACK-009 EU Climate Compliance Bundle

This module routes data to and from PACK-004 CBAM Readiness within the bundle.
It maps bundle data format to CBAM format (installations, embedded emissions,
CN codes) and extracts CBAM results back into the consolidated bundle format.

The bridge handles:
- Bundle -> CBAM data format conversion (25 field mappings)
- CBAM -> Bundle result extraction
- CBAM-specific metric aggregation (installations, CN codes, embedded emissions)
- Quarterly reporting period mapping

Example:
    >>> config = CBAMPackBridgeConfig(transitional_period=True)
    >>> bridge = CBAMPackBridge(config)
    >>> await bridge.push_data(bundle_data)
    >>> results = await bridge.pull_results()
    >>> metrics = await bridge.get_cbam_metrics()
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

class CBAMPackBridgeConfig(BaseModel):
    """Configuration for CBAM pack bridge."""

    transitional_period: bool = Field(
        default=True,
        description="Whether in CBAM transitional period (before 2026)"
    )
    quarterly_reporting: bool = Field(
        default=True,
        description="Enable quarterly CBAM report generation"
    )
    reporting_period_year: int = Field(
        default=2025,
        ge=2023,
        description="Reporting period fiscal year"
    )
    default_calculation_method: Literal[
        "actual", "default_values", "eu_etd"
    ] = Field(
        default="actual",
        description="Default emission calculation method"
    )
    include_indirect_emissions: bool = Field(
        default=True,
        description="Include indirect (Scope 2) embedded emissions"
    )
    currency: str = Field(
        default="EUR",
        description="Currency for CBAM certificate pricing"
    )


# ---------------------------------------------------------------------------
# Data mapping
# ---------------------------------------------------------------------------

CBAM_DATA_MAPPING: Dict[str, Dict[str, Any]] = {
    "importer_name": {
        "cbam_field": "authorized_declarant_name",
        "regulation_ref": "Art. 5",
        "data_type": "string",
        "required": True,
    },
    "importer_eori": {
        "cbam_field": "declarant_eori_number",
        "regulation_ref": "Art. 5",
        "data_type": "string",
        "required": True,
    },
    "import_goods": {
        "cbam_field": "goods_imported",
        "regulation_ref": "Art. 7",
        "data_type": "list",
        "required": True,
    },
    "cn_codes": {
        "cbam_field": "combined_nomenclature_codes",
        "regulation_ref": "Annex I",
        "data_type": "list",
        "required": True,
    },
    "origin_country": {
        "cbam_field": "country_of_origin",
        "regulation_ref": "Art. 7",
        "data_type": "string",
        "required": True,
    },
    "installation_name": {
        "cbam_field": "installation_name",
        "regulation_ref": "Art. 10",
        "data_type": "string",
        "required": True,
    },
    "installation_country": {
        "cbam_field": "installation_country",
        "regulation_ref": "Art. 10",
        "data_type": "string",
        "required": True,
    },
    "installation_id": {
        "cbam_field": "installation_identification",
        "regulation_ref": "Art. 10",
        "data_type": "string",
        "required": False,
    },
    "embedded_emissions_direct": {
        "cbam_field": "specific_direct_embedded_emissions",
        "regulation_ref": "Art. 7(2)",
        "data_type": "numeric",
        "required": True,
    },
    "embedded_emissions_indirect": {
        "cbam_field": "specific_indirect_embedded_emissions",
        "regulation_ref": "Art. 7(3)",
        "data_type": "numeric",
        "required": False,
    },
    "total_embedded_emissions": {
        "cbam_field": "total_embedded_emissions",
        "regulation_ref": "Art. 7",
        "data_type": "numeric",
        "required": True,
    },
    "quantity_imported": {
        "cbam_field": "quantity_of_goods_imported",
        "regulation_ref": "Art. 7",
        "data_type": "numeric",
        "required": True,
    },
    "quantity_unit": {
        "cbam_field": "unit_of_measurement",
        "regulation_ref": "Art. 7",
        "data_type": "string",
        "required": True,
    },
    "carbon_price_paid": {
        "cbam_field": "carbon_price_paid_abroad",
        "regulation_ref": "Art. 9",
        "data_type": "monetary",
        "required": False,
    },
    "carbon_price_currency": {
        "cbam_field": "carbon_price_currency",
        "regulation_ref": "Art. 9",
        "data_type": "string",
        "required": False,
    },
    "ets_benchmark": {
        "cbam_field": "eu_ets_benchmark_value",
        "regulation_ref": "Art. 7",
        "data_type": "numeric",
        "required": False,
    },
    "calculation_method": {
        "cbam_field": "emission_calculation_methodology",
        "regulation_ref": "Art. 7(7)",
        "data_type": "string",
        "required": True,
    },
    "verification_status": {
        "cbam_field": "verification_statement_status",
        "regulation_ref": "Art. 8",
        "data_type": "string",
        "required": False,
    },
    "verifier_name": {
        "cbam_field": "accredited_verifier_name",
        "regulation_ref": "Art. 8",
        "data_type": "string",
        "required": False,
    },
    "reporting_quarter": {
        "cbam_field": "cbam_reporting_quarter",
        "regulation_ref": "Art. 35",
        "data_type": "string",
        "required": True,
    },
    "product_category": {
        "cbam_field": "cbam_product_category",
        "regulation_ref": "Annex I",
        "data_type": "string",
        "required": True,
    },
    "production_process": {
        "cbam_field": "production_process_description",
        "regulation_ref": "Art. 10",
        "data_type": "string",
        "required": False,
    },
    "precursors": {
        "cbam_field": "precursor_goods",
        "regulation_ref": "Art. 7(4)",
        "data_type": "list",
        "required": False,
    },
    "free_allocation_deduction": {
        "cbam_field": "eu_ets_free_allocation_deduction",
        "regulation_ref": "Art. 31",
        "data_type": "numeric",
        "required": False,
    },
    "cbam_certificates_required": {
        "cbam_field": "cbam_certificates_to_surrender",
        "regulation_ref": "Art. 22",
        "data_type": "numeric",
        "required": False,
    },
}


# ---------------------------------------------------------------------------
# CBAM product categories
# ---------------------------------------------------------------------------

CBAM_PRODUCT_CATEGORIES: Dict[str, List[str]] = {
    "cement": ["2523 10", "2523 21", "2523 29", "2523 30", "2523 90"],
    "iron_and_steel": [
        "7201", "7202", "7203", "7204", "7205", "7206", "7207",
        "7208", "7209", "7210", "7211", "7212", "7213",
        "7214", "7215", "7216", "7217", "7218", "7219",
        "7220", "7221", "7222", "7223", "7224", "7225",
        "7226", "7227", "7228", "7229", "7301", "7302",
        "7303", "7304", "7305", "7306", "7307",
    ],
    "aluminium": [
        "7601", "7602", "7603", "7604", "7605", "7606",
        "7607", "7608", "7609",
    ],
    "fertilizers": [
        "2808 00 00", "3102", "3105",
    ],
    "electricity": ["2716 00 00"],
    "hydrogen": ["2804 10 00"],
}


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class CBAMPackBridge:
    """
    CBAM Pack Bridge for PACK-009 Bundle.

    Routes data to/from PACK-004 CBAM Readiness, mapping bundle data format
    to CBAM format and extracting CBAM results into the consolidated bundle format.

    Example:
        >>> config = CBAMPackBridgeConfig(transitional_period=True)
        >>> bridge = CBAMPackBridge(config)
        >>> push_result = await bridge.push_data(bundle_data)
        >>> results = await bridge.pull_results()
    """

    def __init__(self, config: CBAMPackBridgeConfig):
        """Initialize CBAM pack bridge."""
        self.config = config
        self._cbam_service: Any = None
        self._pushed_data: Dict[str, Any] = {}
        self._results: Dict[str, Any] = {}
        self._push_timestamp: Optional[str] = None
        logger.info("CBAMPackBridge initialized")

    def inject_service(self, service: Any) -> None:
        """Inject real CBAM pack service."""
        self._cbam_service = service
        logger.info("Injected CBAM pack service")

    async def push_data(self, bundle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Push bundle data to CBAM pack in CBAM format.

        Args:
            bundle_data: Data in bundle format.

        Returns:
            Push result with mapping statistics.
        """
        try:
            if self._cbam_service and hasattr(self._cbam_service, "push_data"):
                mapped = self._map_to_cbam(bundle_data)
                result = await self._cbam_service.push_data(mapped)
                self._pushed_data = mapped
                self._push_timestamp = datetime.utcnow().isoformat()
                return result

            mapped = self._map_to_cbam(bundle_data)
            self._pushed_data = mapped
            self._push_timestamp = datetime.utcnow().isoformat()

            mapped_count = sum(1 for v in mapped.values() if v is not None)
            total_fields = len(CBAM_DATA_MAPPING)
            required_fields = sum(
                1 for m in CBAM_DATA_MAPPING.values() if m["required"]
            )
            required_mapped = sum(
                1
                for key, meta in CBAM_DATA_MAPPING.items()
                if meta["required"] and mapped.get(meta["cbam_field"]) is not None
            )

            # Categorize imported goods
            cn_codes = mapped.get("combined_nomenclature_codes", [])
            categories_found = self._categorize_cn_codes(
                cn_codes if isinstance(cn_codes, list) else []
            )

            return {
                "status": "success",
                "total_fields": total_fields,
                "mapped_fields": mapped_count,
                "unmapped_fields": total_fields - mapped_count,
                "required_fields": required_fields,
                "required_mapped": required_mapped,
                "mapping_completeness": round(mapped_count / total_fields * 100, 1),
                "product_categories": categories_found,
                "timestamp": self._push_timestamp,
                "provenance_hash": self._calculate_hash(mapped),
            }

        except Exception as e:
            logger.error(f"CBAM push_data failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def pull_results(self) -> Dict[str, Any]:
        """
        Pull results from CBAM pack in bundle format.

        Returns:
            CBAM assessment results mapped to bundle format.
        """
        try:
            if self._cbam_service and hasattr(self._cbam_service, "pull_results"):
                raw = await self._cbam_service.pull_results()
                self._results = self._map_from_cbam(raw)
                return self._results

            self._results = {
                "pack": "PACK-004 CBAM Readiness",
                "status": "completed" if self._pushed_data else "no_data",
                "reporting_year": self.config.reporting_period_year,
                "transitional_period": self.config.transitional_period,
                "assessment_summary": {
                    "installations_assessed": 0,
                    "goods_categories": 0,
                    "total_embedded_emissions_tco2": 0.0,
                    "direct_emissions_tco2": 0.0,
                    "indirect_emissions_tco2": 0.0,
                    "carbon_price_credit_eur": 0.0,
                    "certificates_required": 0,
                },
                "quarterly_status": self._generate_quarterly_status(),
                "verification_status": self._pushed_data.get(
                    "verification_statement_status", "not_started"
                ),
                "data_quality_score": 0.70,
                "timestamp": datetime.utcnow().isoformat(),
                "provenance_hash": self._calculate_hash(self._pushed_data),
            }
            return self._results

        except Exception as e:
            logger.error(f"CBAM pull_results failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """
        Get current bridge status.

        Returns:
            Bridge status including push/pull state.
        """
        return {
            "bridge": "CBAMPackBridge",
            "target_pack": "PACK-004 CBAM Readiness",
            "service_injected": self._cbam_service is not None,
            "data_pushed": bool(self._pushed_data),
            "results_available": bool(self._results),
            "push_timestamp": self._push_timestamp,
            "config": {
                "transitional_period": self.config.transitional_period,
                "quarterly_reporting": self.config.quarterly_reporting,
                "calculation_method": self.config.default_calculation_method,
                "include_indirect": self.config.include_indirect_emissions,
                "reporting_year": self.config.reporting_period_year,
            },
            "mapping_stats": {
                "total_mappings": len(CBAM_DATA_MAPPING),
                "required_mappings": sum(
                    1 for m in CBAM_DATA_MAPPING.values() if m["required"]
                ),
                "optional_mappings": sum(
                    1 for m in CBAM_DATA_MAPPING.values() if not m["required"]
                ),
            },
        }

    async def get_cbam_metrics(self) -> Dict[str, Any]:
        """
        Get CBAM-specific metrics from the latest results.

        Returns:
            Aggregated CBAM metrics.
        """
        try:
            if self._cbam_service and hasattr(self._cbam_service, "get_metrics"):
                return await self._cbam_service.get_metrics()

            if not self._results:
                return {"status": "no_results", "message": "No results available yet"}

            summary = self._results.get("assessment_summary", {})

            return {
                "pack": "PACK-004 CBAM Readiness",
                "reporting_year": self.config.reporting_period_year,
                "transitional_period": self.config.transitional_period,
                "installations": summary.get("installations_assessed", 0),
                "goods_categories": summary.get("goods_categories", 0),
                "emissions": {
                    "total_embedded_tco2": summary.get(
                        "total_embedded_emissions_tco2", 0.0
                    ),
                    "direct_tco2": summary.get("direct_emissions_tco2", 0.0),
                    "indirect_tco2": summary.get("indirect_emissions_tco2", 0.0),
                },
                "financial": {
                    "carbon_price_credit_eur": summary.get(
                        "carbon_price_credit_eur", 0.0
                    ),
                    "certificates_required": summary.get("certificates_required", 0),
                    "currency": self.config.currency,
                },
                "quarterly_status": self._results.get("quarterly_status", {}),
                "verification_status": self._results.get(
                    "verification_status", "unknown"
                ),
                "data_quality_score": self._results.get("data_quality_score", 0.0),
                "product_categories": list(CBAM_PRODUCT_CATEGORIES.keys()),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"CBAM get_cbam_metrics failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_to_cbam(self, bundle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map bundle data format to CBAM pack format."""
        mapped: Dict[str, Any] = {}
        for bundle_key, mapping in CBAM_DATA_MAPPING.items():
            cbam_field = mapping["cbam_field"]
            value = bundle_data.get(bundle_key)
            if value is not None:
                mapped[cbam_field] = value
            elif mapping["required"]:
                mapped[cbam_field] = None
                logger.warning(
                    f"Required CBAM field missing: {bundle_key} -> {cbam_field}"
                )
        return mapped

    def _map_from_cbam(self, cbam_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map CBAM results back to bundle format."""
        result: Dict[str, Any] = {
            "pack": "PACK-004 CBAM Readiness",
            "status": "completed",
        }
        reverse_map = {
            v["cbam_field"]: k for k, v in CBAM_DATA_MAPPING.items()
        }
        for cbam_field, value in cbam_data.items():
            bundle_key = reverse_map.get(cbam_field)
            if bundle_key:
                result[bundle_key] = value
            else:
                result[cbam_field] = value
        return result

    def _categorize_cn_codes(self, cn_codes: List[str]) -> List[str]:
        """Categorize CN codes into CBAM product categories."""
        found_categories: List[str] = []
        for category, codes in CBAM_PRODUCT_CATEGORIES.items():
            for cn in cn_codes:
                cn_stripped = cn.replace(" ", "")
                for ref_code in codes:
                    ref_stripped = ref_code.replace(" ", "")
                    if cn_stripped.startswith(ref_stripped):
                        if category not in found_categories:
                            found_categories.append(category)
                        break
        return found_categories

    def _generate_quarterly_status(self) -> Dict[str, str]:
        """Generate quarterly reporting status."""
        year = self.config.reporting_period_year
        return {
            f"Q1_{year}": "pending",
            f"Q2_{year}": "pending",
            f"Q3_{year}": "pending",
            f"Q4_{year}": "pending",
        }

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
