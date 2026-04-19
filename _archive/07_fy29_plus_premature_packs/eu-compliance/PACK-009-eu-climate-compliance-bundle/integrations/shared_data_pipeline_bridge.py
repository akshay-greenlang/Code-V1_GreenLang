"""
Shared Data Pipeline Bridge - PACK-009 EU Climate Compliance Bundle

This module routes deduplicated data to appropriate pack data pipelines and
manages data flow splitting and routing across the 4 constituent packs.
It ensures each pack receives only the data it needs, reducing redundant
processing and improving efficiency.

The bridge handles:
- Data routing to appropriate pack pipelines
- Data flow splitting based on field-to-pack mapping
- Deduplication of shared data collection
- Pipeline status tracking

Example:
    >>> config = SharedDataPipelineConfig()
    >>> bridge = SharedDataPipelineBridge(config)
    >>> route_result = await bridge.route_data(collected_data)
    >>> status = await bridge.get_pipeline_status()
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

class SharedDataPipelineConfig(BaseModel):
    """Configuration for shared data pipeline bridge."""

    enable_deduplication: bool = Field(
        default=True,
        description="Enable data deduplication before routing"
    )
    enable_validation: bool = Field(
        default=True,
        description="Enable data validation before routing"
    )
    batch_size: int = Field(
        default=500,
        ge=1,
        description="Batch size for data routing"
    )
    reporting_period_year: int = Field(
        default=2025,
        ge=2023,
        description="Reporting period fiscal year"
    )
    enable_csrd: bool = Field(default=True, description="Route data to CSRD pack")
    enable_cbam: bool = Field(default=True, description="Route data to CBAM pack")
    enable_eudr: bool = Field(default=True, description="Route data to EUDR pack")
    enable_taxonomy: bool = Field(default=True, description="Route data to Taxonomy pack")


# ---------------------------------------------------------------------------
# Routing table
# ---------------------------------------------------------------------------

ROUTING_TABLE: Dict[str, Dict[str, Any]] = {
    # --- Organization data -> multiple packs ---
    "organization": {
        "target_packs": ["csrd", "cbam", "eudr", "taxonomy"],
        "priority": "high",
        "description": "Organization identity and structure data",
        "fields": [
            "organization_name", "organization_type", "organization_country",
            "reporting_period", "employee_count", "total_assets",
        ],
    },
    # --- GHG / Emissions data ---
    "ghg_emissions": {
        "target_packs": ["csrd", "taxonomy"],
        "priority": "high",
        "description": "GHG emissions data (Scope 1/2/3)",
        "fields": [
            "ghg_scope1", "ghg_scope2_location", "ghg_scope2_market",
            "ghg_scope3", "ghg_intensity",
        ],
    },
    "embedded_emissions": {
        "target_packs": ["cbam"],
        "priority": "high",
        "description": "CBAM embedded emissions data",
        "fields": [
            "embedded_emissions_direct", "embedded_emissions_indirect",
            "total_embedded_emissions", "calculation_method",
        ],
    },
    # --- Financial data ---
    "financial": {
        "target_packs": ["csrd", "taxonomy"],
        "priority": "high",
        "description": "Financial data (revenue, CapEx, OpEx)",
        "fields": [
            "total_revenue", "total_capex", "total_opex", "total_assets",
        ],
    },
    "carbon_pricing": {
        "target_packs": ["cbam"],
        "priority": "medium",
        "description": "Carbon pricing and certificate data",
        "fields": [
            "carbon_price_paid", "carbon_price_currency",
            "ets_benchmark", "free_allocation_deduction",
            "cbam_certificates_required",
        ],
    },
    # --- Energy data ---
    "energy": {
        "target_packs": ["csrd", "taxonomy"],
        "priority": "medium",
        "description": "Energy consumption and mix data",
        "fields": [
            "energy_consumption", "renewable_energy_share",
        ],
    },
    # --- Environmental data ---
    "water": {
        "target_packs": ["csrd", "taxonomy"],
        "priority": "medium",
        "description": "Water consumption data",
        "fields": ["water_consumption"],
    },
    "waste": {
        "target_packs": ["csrd", "taxonomy"],
        "priority": "medium",
        "description": "Waste and circular economy data",
        "fields": ["waste_generated", "waste_recycling_rate"],
    },
    "biodiversity": {
        "target_packs": ["csrd", "taxonomy", "eudr"],
        "priority": "medium",
        "description": "Biodiversity and land use data",
        "fields": [
            "biodiversity_sites", "biodiversity_impact",
            "deforestation_free", "forest_degradation_free",
        ],
    },
    # --- Supply chain data ---
    "supply_chain": {
        "target_packs": ["csrd", "eudr"],
        "priority": "high",
        "description": "Supply chain and due diligence data",
        "fields": [
            "supply_chain_due_diligence", "supplier_name",
            "supplier_country", "supply_chain_map",
        ],
    },
    # --- Trade / Import data ---
    "imports": {
        "target_packs": ["cbam"],
        "priority": "high",
        "description": "Import goods and installation data",
        "fields": [
            "import_goods", "cn_codes", "origin_country",
            "installation_name", "installation_country",
            "quantity_imported", "quantity_unit",
            "product_category", "production_process", "precursors",
        ],
    },
    # --- Commodity / Traceability data ---
    "commodities": {
        "target_packs": ["eudr"],
        "priority": "high",
        "description": "EUDR commodity and traceability data",
        "fields": [
            "commodity_type", "product_description", "product_hs_code",
            "quantity", "production_country", "production_date",
            "geolocation_data", "plot_of_land",
        ],
    },
    # --- Taxonomy-specific data ---
    "taxonomy_activities": {
        "target_packs": ["taxonomy"],
        "priority": "high",
        "description": "NACE activities and taxonomy KPIs",
        "fields": [
            "nace_codes", "activities", "eligible_turnover",
            "eligible_capex", "eligible_opex",
            "aligned_turnover", "aligned_capex", "aligned_opex",
            "enabling_activities", "transitional_activities",
        ],
    },
    "taxonomy_kpis": {
        "target_packs": ["csrd", "taxonomy"],
        "priority": "medium",
        "description": "Taxonomy alignment KPI ratios",
        "fields": [
            "taxonomy_turnover_ratio", "taxonomy_capex_ratio",
            "taxonomy_opex_ratio",
        ],
    },
    "taxonomy_financial_institution": {
        "target_packs": ["taxonomy"],
        "priority": "low",
        "description": "GAR/BTAR data for financial institutions",
        "fields": ["gar_stock", "gar_flow", "btar"],
    },
    # --- Governance data ---
    "governance": {
        "target_packs": ["csrd"],
        "priority": "medium",
        "description": "Governance and social data",
        "fields": [
            "board_diversity", "anti_corruption_training",
            "gender_pay_gap", "workforce_total",
        ],
    },
    # --- Climate risk & strategy data ---
    "climate_strategy": {
        "target_packs": ["csrd", "taxonomy"],
        "priority": "medium",
        "description": "Climate risk and transition planning data",
        "fields": [
            "transition_plan", "climate_risk_physical",
            "climate_risk_transition", "ghg_reduction_targets",
            "sbti_validation", "minimum_safeguards",
        ],
    },
    # --- Verification data ---
    "verification": {
        "target_packs": ["cbam", "eudr"],
        "priority": "medium",
        "description": "Verification and certification data",
        "fields": [
            "verification_status", "verifier_name",
            "certification_scheme", "third_party_verification",
            "satellite_verification",
        ],
    },
    # --- Risk assessment data ---
    "risk_assessment": {
        "target_packs": ["eudr"],
        "priority": "medium",
        "description": "Risk assessment and mitigation data",
        "fields": [
            "risk_assessment_result", "risk_level",
            "mitigation_measures",
        ],
    },
}


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class SharedDataPipelineBridge:
    """
    Shared Data Pipeline Bridge for PACK-009 Bundle.

    Routes deduplicated data to appropriate pack data pipelines,
    manages data flow splitting and routing across 4 constituent packs.

    Example:
        >>> config = SharedDataPipelineConfig()
        >>> bridge = SharedDataPipelineBridge(config)
        >>> route_result = await bridge.route_data(collected_data)
        >>> table = bridge.get_routing_table()
    """

    def __init__(self, config: SharedDataPipelineConfig):
        """Initialize shared data pipeline bridge."""
        self.config = config
        self._routed_data: Dict[str, Dict[str, Any]] = {}
        self._routing_stats: Dict[str, Any] = {}
        self._last_route_timestamp: Optional[str] = None
        logger.info("SharedDataPipelineBridge initialized")

    async def route_data(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route collected data to appropriate pack pipelines.

        Args:
            collected_data: Deduplicated data from collection phase.

        Returns:
            Routing result with per-pack data splits.
        """
        try:
            start_time = datetime.utcnow()

            # Initialize per-pack data buckets
            active_packs = self._get_active_packs()
            pack_data: Dict[str, Dict[str, Any]] = {
                pk: {} for pk in active_packs
            }

            fields_routed = 0
            fields_skipped = 0
            categories_processed = 0

            for category, route_config in ROUTING_TABLE.items():
                target_packs = [
                    pk for pk in route_config["target_packs"]
                    if pk in active_packs
                ]
                if not target_packs:
                    continue

                categories_processed += 1
                category_fields = route_config["fields"]

                for field in category_fields:
                    value = collected_data.get(field)
                    if value is None:
                        fields_skipped += 1
                        continue

                    if self.config.enable_validation:
                        if not self._validate_field(field, value):
                            fields_skipped += 1
                            logger.warning(
                                f"Field '{field}' failed validation, skipped"
                            )
                            continue

                    for pk in target_packs:
                        pack_data[pk][field] = value
                    fields_routed += 1

            self._routed_data = pack_data
            self._last_route_timestamp = datetime.utcnow().isoformat()

            duration = (datetime.utcnow() - start_time).total_seconds()

            self._routing_stats = {
                "total_input_fields": len(collected_data),
                "fields_routed": fields_routed,
                "fields_skipped": fields_skipped,
                "categories_processed": categories_processed,
                "total_categories": len(ROUTING_TABLE),
                "per_pack_fields": {
                    pk: len(data) for pk, data in pack_data.items()
                },
                "duration_seconds": duration,
            }

            return {
                "status": "success",
                "routing_stats": self._routing_stats,
                "pack_data": {
                    pk: {"fields": len(data), "keys": list(data.keys())}
                    for pk, data in pack_data.items()
                },
                "timestamp": self._last_route_timestamp,
                "provenance_hash": self._calculate_hash(self._routing_stats),
            }

        except Exception as e:
            logger.error(f"Data routing failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def get_routing_table(self) -> Dict[str, Any]:
        """
        Return the routing table with active pack filtering.

        Returns:
            Filtered routing table with descriptions and field counts.
        """
        active_packs = self._get_active_packs()
        filtered: Dict[str, Any] = {}

        for category, route_config in ROUTING_TABLE.items():
            targets = [
                pk for pk in route_config["target_packs"]
                if pk in active_packs
            ]
            if targets:
                filtered[category] = {
                    "target_packs": targets,
                    "priority": route_config["priority"],
                    "description": route_config["description"],
                    "field_count": len(route_config["fields"]),
                    "fields": route_config["fields"],
                }

        return {
            "total_categories": len(filtered),
            "active_packs": active_packs,
            "categories": filtered,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def validate_data_flow(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate data flow integrity before routing.

        Args:
            data: Data to validate.

        Returns:
            Validation result with field-level status.
        """
        try:
            issues: List[Dict[str, str]] = []
            valid_fields = 0
            invalid_fields = 0

            all_expected_fields: List[str] = []
            for route_config in ROUTING_TABLE.values():
                all_expected_fields.extend(route_config["fields"])
            unique_fields = list(set(all_expected_fields))

            for field in unique_fields:
                value = data.get(field)
                if value is not None:
                    if self._validate_field(field, value):
                        valid_fields += 1
                    else:
                        invalid_fields += 1
                        issues.append({
                            "field": field,
                            "issue": "validation_failed",
                            "message": f"Field '{field}' failed type validation",
                        })

            present_fields = sum(1 for f in unique_fields if data.get(f) is not None)
            missing_fields = len(unique_fields) - present_fields

            return {
                "status": "valid" if not issues else "issues_found",
                "total_expected_fields": len(unique_fields),
                "present_fields": present_fields,
                "missing_fields": missing_fields,
                "valid_fields": valid_fields,
                "invalid_fields": invalid_fields,
                "coverage_pct": (
                    round(present_fields / len(unique_fields) * 100, 1)
                    if unique_fields
                    else 0.0
                ),
                "issues": issues,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Data flow validation failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status including routing statistics.

        Returns:
            Pipeline status with per-pack details.
        """
        return {
            "bridge": "SharedDataPipelineBridge",
            "data_routed": bool(self._routed_data),
            "last_route_timestamp": self._last_route_timestamp,
            "routing_stats": self._routing_stats,
            "config": {
                "deduplication_enabled": self.config.enable_deduplication,
                "validation_enabled": self.config.enable_validation,
                "batch_size": self.config.batch_size,
                "reporting_year": self.config.reporting_period_year,
            },
            "active_packs": self._get_active_packs(),
            "routing_table_categories": len(ROUTING_TABLE),
            "per_pack_routed_fields": (
                {pk: len(data) for pk, data in self._routed_data.items()}
                if self._routed_data
                else {}
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_pack_data(self, pack_key: str) -> Dict[str, Any]:
        """
        Get the routed data for a specific pack.

        Args:
            pack_key: Pack identifier (csrd, cbam, eudr, taxonomy).

        Returns:
            Data routed to the specified pack.
        """
        if pack_key not in self._routed_data:
            return {
                "pack": pack_key,
                "status": "no_data",
                "message": f"No data routed to pack '{pack_key}' yet",
            }

        data = self._routed_data[pack_key]
        return {
            "pack": pack_key,
            "status": "available",
            "field_count": len(data),
            "fields": list(data.keys()),
            "data": data,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_active_packs(self) -> List[str]:
        """Return list of active packs from config."""
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

    def _validate_field(self, field: str, value: Any) -> bool:
        """Validate a field value (basic type checking)."""
        if value is None:
            return False
        if isinstance(value, str) and len(value) == 0:
            return False
        if isinstance(value, list) and len(value) == 0:
            return False
        return True

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
