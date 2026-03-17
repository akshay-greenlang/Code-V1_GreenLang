"""
Cross-Framework Mapper Bridge - PACK-009 EU Climate Compliance Bundle

This module consolidates mapping tables from all 4 constituent pack bridges
and provides unified field lookup across all EU regulations (CSRD, CBAM,
EUDR, EU Taxonomy). It enables cross-framework data reuse and identifies
overlapping disclosure requirements.

The bridge handles:
- Unified field lookup across all 4 regulations
- Batch field mapping across frameworks
- Overlap identification (shared data requirements)
- Cross-framework mapping report generation

Example:
    >>> config = CrossFrameworkMapperConfig()
    >>> bridge = CrossFrameworkMapperBridge(config)
    >>> result = bridge.map_field("ghg_scope1")
    >>> overlap = bridge.get_overlap_report()
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

class CrossFrameworkMapperConfig(BaseModel):
    """Configuration for cross-framework mapper bridge."""

    enable_csrd: bool = Field(default=True, description="Include CSRD mappings")
    enable_cbam: bool = Field(default=True, description="Include CBAM mappings")
    enable_eudr: bool = Field(default=True, description="Include EUDR mappings")
    enable_taxonomy: bool = Field(default=True, description="Include Taxonomy mappings")
    detect_overlaps: bool = Field(
        default=True,
        description="Detect overlapping data requirements across frameworks"
    )
    reporting_period_year: int = Field(
        default=2025, ge=2023,
        description="Reporting period fiscal year"
    )


# ---------------------------------------------------------------------------
# Unified mapping table
# ---------------------------------------------------------------------------

UNIFIED_MAPPING_TABLE: Dict[str, Dict[str, Any]] = {
    # --- Organization / Entity ---
    "organization_name": {
        "csrd": {"field": "undertaking_name", "ref": "ESRS 2 BP-1", "required": True},
        "cbam": {"field": "authorized_declarant_name", "ref": "Art. 5", "required": True},
        "eudr": {"field": "operator_legal_name", "ref": "Art. 4(1)", "required": True},
        "taxonomy": {"field": "undertaking_name", "ref": "Art. 8(1)", "required": True},
        "data_type": "string",
        "overlap": True,
    },
    "organization_type": {
        "csrd": {"field": "undertaking_type", "ref": "ESRS 2 BP-1", "required": True},
        "taxonomy": {"field": "undertaking_type", "ref": "Art. 8(1)", "required": True},
        "data_type": "string",
        "overlap": True,
    },
    "reporting_period": {
        "csrd": {"field": "reporting_period", "ref": "ESRS 2 BP-1", "required": True},
        "taxonomy": {"field": "reporting_period", "ref": "Art. 8(2)", "required": True},
        "data_type": "date_range",
        "overlap": True,
    },
    "organization_country": {
        "cbam": {"field": "declarant_country", "ref": "Art. 5", "required": False},
        "eudr": {"field": "operator_country_of_establishment", "ref": "Art. 4(1)", "required": True},
        "data_type": "string",
        "overlap": True,
    },
    # --- GHG Emissions ---
    "ghg_scope1": {
        "csrd": {"field": "gross_scope1_ghg_emissions", "ref": "E1-6.44", "required": True},
        "taxonomy": {"field": "scope1_ghg_emissions", "ref": "Climate DA TSC", "required": False},
        "data_type": "numeric",
        "overlap": True,
    },
    "ghg_scope2_location": {
        "csrd": {"field": "gross_scope2_location_ghg_emissions", "ref": "E1-6.46", "required": True},
        "data_type": "numeric",
        "overlap": False,
    },
    "ghg_scope2_market": {
        "csrd": {"field": "gross_scope2_market_ghg_emissions", "ref": "E1-6.47", "required": True},
        "data_type": "numeric",
        "overlap": False,
    },
    "ghg_scope2": {
        "taxonomy": {"field": "scope2_ghg_emissions", "ref": "Climate DA TSC", "required": False},
        "data_type": "numeric",
        "overlap": False,
    },
    "ghg_scope3": {
        "csrd": {"field": "gross_scope3_ghg_emissions", "ref": "E1-6.51", "required": False},
        "data_type": "numeric",
        "overlap": False,
    },
    "ghg_intensity": {
        "csrd": {"field": "ghg_intensity_per_net_revenue", "ref": "E1-6.53", "required": False},
        "data_type": "numeric",
        "overlap": False,
    },
    "embedded_emissions_direct": {
        "cbam": {"field": "specific_direct_embedded_emissions", "ref": "Art. 7(2)", "required": True},
        "data_type": "numeric",
        "overlap": False,
    },
    "embedded_emissions_indirect": {
        "cbam": {"field": "specific_indirect_embedded_emissions", "ref": "Art. 7(3)", "required": False},
        "data_type": "numeric",
        "overlap": False,
    },
    "total_embedded_emissions": {
        "cbam": {"field": "total_embedded_emissions", "ref": "Art. 7", "required": True},
        "data_type": "numeric",
        "overlap": False,
    },
    # --- Energy ---
    "energy_consumption": {
        "csrd": {"field": "total_energy_consumption", "ref": "E1-5.37", "required": True},
        "taxonomy": {"field": "total_energy_consumption_mwh", "ref": "Climate DA TSC", "required": False},
        "data_type": "numeric",
        "overlap": True,
    },
    "renewable_energy_share": {
        "csrd": {"field": "share_of_renewable_energy", "ref": "E1-5.38", "required": False},
        "taxonomy": {"field": "renewable_energy_share_pct", "ref": "Climate DA DNSH", "required": False},
        "data_type": "percentage",
        "overlap": True,
    },
    # --- Financial ---
    "total_revenue": {
        "csrd": {"field": "net_turnover", "ref": "ESRS 2 BP-2", "required": True},
        "taxonomy": {"field": "net_turnover_total", "ref": "Art. 8(2)(a)", "required": True},
        "data_type": "monetary",
        "overlap": True,
    },
    "total_capex": {
        "taxonomy": {"field": "capital_expenditure_total", "ref": "Art. 8(2)(b)", "required": True},
        "data_type": "monetary",
        "overlap": False,
    },
    "total_opex": {
        "taxonomy": {"field": "operating_expenditure_total", "ref": "Art. 8(2)(c)", "required": True},
        "data_type": "monetary",
        "overlap": False,
    },
    "total_assets": {
        "csrd": {"field": "balance_sheet_total", "ref": "ESRS 2 BP-2", "required": True},
        "data_type": "monetary",
        "overlap": False,
    },
    "employee_count": {
        "csrd": {"field": "average_number_of_employees", "ref": "ESRS 2 BP-2", "required": True},
        "data_type": "integer",
        "overlap": False,
    },
    "carbon_price_paid": {
        "cbam": {"field": "carbon_price_paid_abroad", "ref": "Art. 9", "required": False},
        "data_type": "monetary",
        "overlap": False,
    },
    # --- Taxonomy KPIs ---
    "taxonomy_turnover_ratio": {
        "csrd": {"field": "taxonomy_aligned_turnover_percentage", "ref": "E1-6.69", "required": False},
        "taxonomy": {"field": "taxonomy_aligned_turnover", "ref": "Art. 8(2)(a)", "required": False},
        "data_type": "percentage",
        "overlap": True,
    },
    "taxonomy_capex_ratio": {
        "csrd": {"field": "taxonomy_aligned_capex_percentage", "ref": "E1-6.69", "required": False},
        "taxonomy": {"field": "taxonomy_aligned_capex", "ref": "Art. 8(2)(b)", "required": False},
        "data_type": "percentage",
        "overlap": True,
    },
    "taxonomy_opex_ratio": {
        "csrd": {"field": "taxonomy_aligned_opex_percentage", "ref": "E1-6.69", "required": False},
        "taxonomy": {"field": "taxonomy_aligned_opex", "ref": "Art. 8(2)(c)", "required": False},
        "data_type": "percentage",
        "overlap": True,
    },
    # --- Environment ---
    "water_consumption": {
        "csrd": {"field": "total_water_consumption", "ref": "E3-4", "required": False},
        "taxonomy": {"field": "water_consumption_m3", "ref": "Env DA WTR", "required": False},
        "data_type": "numeric",
        "overlap": True,
    },
    "waste_generated": {
        "csrd": {"field": "total_waste_generated", "ref": "E5-5", "required": False},
        "data_type": "numeric",
        "overlap": False,
    },
    "waste_recycling_rate": {
        "taxonomy": {"field": "waste_recycling_rate_pct", "ref": "Env DA CE", "required": False},
        "data_type": "percentage",
        "overlap": False,
    },
    "biodiversity_sites": {
        "csrd": {"field": "sites_near_biodiversity_areas", "ref": "E4-5", "required": False},
        "data_type": "integer",
        "overlap": False,
    },
    "biodiversity_impact": {
        "taxonomy": {"field": "biodiversity_impact_assessment", "ref": "Env DA BIO", "required": False},
        "data_type": "json",
        "overlap": False,
    },
    # --- Supply Chain / Traceability ---
    "supply_chain_due_diligence": {
        "csrd": {"field": "supply_chain_due_diligence_process", "ref": "G1-2", "required": False},
        "eudr": {"field": "due_diligence_statement_reference", "ref": "Art. 4(1)", "required": True},
        "data_type": "string",
        "overlap": True,
    },
    "supplier_name": {
        "eudr": {"field": "supplier_legal_name", "ref": "Art. 4(2)(d)", "required": True},
        "data_type": "string",
        "overlap": False,
    },
    "supplier_country": {
        "eudr": {"field": "supplier_country_of_origin", "ref": "Art. 4(2)(d)", "required": True},
        "data_type": "string",
        "overlap": False,
    },
    "deforestation_free": {
        "eudr": {"field": "deforestation_free_declaration", "ref": "Art. 3(a)", "required": True},
        "data_type": "boolean",
        "overlap": False,
    },
    "geolocation_data": {
        "eudr": {"field": "geolocation_coordinates", "ref": "Art. 4(2)(f)", "required": True},
        "data_type": "geojson",
        "overlap": False,
    },
    # --- CBAM specific ---
    "cn_codes": {
        "cbam": {"field": "combined_nomenclature_codes", "ref": "Annex I", "required": True},
        "data_type": "list",
        "overlap": False,
    },
    "installation_name": {
        "cbam": {"field": "installation_name", "ref": "Art. 10", "required": True},
        "data_type": "string",
        "overlap": False,
    },
    "quantity_imported": {
        "cbam": {"field": "quantity_of_goods_imported", "ref": "Art. 7", "required": True},
        "data_type": "numeric",
        "overlap": False,
    },
    # --- Governance ---
    "transition_plan": {
        "csrd": {"field": "climate_transition_plan", "ref": "E1-1", "required": False},
        "data_type": "document",
        "overlap": False,
    },
    "minimum_safeguards": {
        "taxonomy": {"field": "minimum_safeguards_compliance", "ref": "Art. 18", "required": False},
        "data_type": "json",
        "overlap": False,
    },
    "anti_corruption_training": {
        "csrd": {"field": "anti_corruption_training_coverage", "ref": "G1-3", "required": False},
        "data_type": "percentage",
        "overlap": False,
    },
    "board_diversity": {
        "csrd": {"field": "board_gender_diversity", "ref": "G1-5", "required": False},
        "data_type": "percentage",
        "overlap": False,
    },
    # --- EUDR specifics ---
    "commodity_type": {
        "eudr": {"field": "relevant_commodity", "ref": "Art. 1", "required": True},
        "data_type": "string",
        "overlap": False,
    },
    "risk_level": {
        "eudr": {"field": "country_risk_classification", "ref": "Art. 29", "required": False},
        "data_type": "string",
        "overlap": False,
    },
    # --- Taxonomy specifics ---
    "nace_codes": {
        "taxonomy": {"field": "nace_activity_codes", "ref": "Art. 1(5)", "required": True},
        "data_type": "list",
        "overlap": False,
    },
    "enabling_activities": {
        "taxonomy": {"field": "enabling_activities_list", "ref": "Art. 16", "required": False},
        "data_type": "list",
        "overlap": False,
    },
    "gar_stock": {
        "taxonomy": {"field": "green_asset_ratio_stock", "ref": "Art. 10 EBA", "required": False},
        "data_type": "percentage",
        "overlap": False,
    },
}


# ---------------------------------------------------------------------------
# Frameworks constant
# ---------------------------------------------------------------------------

ALL_FRAMEWORKS = ["csrd", "cbam", "eudr", "taxonomy"]


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class CrossFrameworkMapperBridge:
    """
    Cross-Framework Mapper Bridge for PACK-009 Bundle.

    Consolidates mapping tables from all 4 constituent pack bridges and
    provides unified field lookup across CSRD, CBAM, EUDR, and EU Taxonomy.

    Example:
        >>> config = CrossFrameworkMapperConfig()
        >>> bridge = CrossFrameworkMapperBridge(config)
        >>> result = bridge.map_field("ghg_scope1")
        >>> overlap = bridge.get_overlap_report()
    """

    def __init__(self, config: CrossFrameworkMapperConfig):
        """Initialize cross-framework mapper bridge."""
        self.config = config
        self._active_frameworks = self._get_active_frameworks()
        self._cached_overlaps: Optional[Dict[str, Any]] = None
        logger.info(
            f"CrossFrameworkMapperBridge initialized with {len(self._active_frameworks)} frameworks"
        )

    def map_field(self, bundle_field: str) -> Dict[str, Any]:
        """
        Map a single bundle field to all applicable framework fields.

        Args:
            bundle_field: Field name in bundle format.

        Returns:
            Mapping result with per-framework field references.
        """
        if bundle_field not in UNIFIED_MAPPING_TABLE:
            return {
                "bundle_field": bundle_field,
                "found": False,
                "message": f"Field '{bundle_field}' not found in unified mapping table",
                "frameworks": {},
            }

        entry = UNIFIED_MAPPING_TABLE[bundle_field]
        frameworks: Dict[str, Any] = {}

        for fw in self._active_frameworks:
            if fw in entry:
                frameworks[fw] = entry[fw]

        return {
            "bundle_field": bundle_field,
            "found": True,
            "data_type": entry.get("data_type", "unknown"),
            "is_overlap": entry.get("overlap", False),
            "frameworks": frameworks,
            "framework_count": len(frameworks),
        }

    def map_batch(self, bundle_fields: List[str]) -> Dict[str, Any]:
        """
        Map multiple bundle fields to framework fields in batch.

        Args:
            bundle_fields: List of field names in bundle format.

        Returns:
            Batch mapping results.
        """
        results: Dict[str, Any] = {}
        found_count = 0
        overlap_count = 0

        for field in bundle_fields:
            mapping = self.map_field(field)
            results[field] = mapping
            if mapping.get("found"):
                found_count += 1
            if mapping.get("is_overlap"):
                overlap_count += 1

        return {
            "total_fields": len(bundle_fields),
            "found": found_count,
            "not_found": len(bundle_fields) - found_count,
            "overlapping": overlap_count,
            "mappings": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_all_mappings(self) -> Dict[str, Any]:
        """
        Return the complete unified mapping table filtered to active frameworks.

        Returns:
            Filtered mapping table with statistics.
        """
        filtered: Dict[str, Any] = {}
        for bundle_field, entry in UNIFIED_MAPPING_TABLE.items():
            fw_entries = {}
            for fw in self._active_frameworks:
                if fw in entry:
                    fw_entries[fw] = entry[fw]
            if fw_entries:
                filtered[bundle_field] = {
                    "data_type": entry.get("data_type", "unknown"),
                    "overlap": entry.get("overlap", False),
                    "frameworks": fw_entries,
                }

        total = len(filtered)
        overlaps = sum(1 for v in filtered.values() if v.get("overlap"))

        return {
            "total_mappings": total,
            "overlap_mappings": overlaps,
            "active_frameworks": self._active_frameworks,
            "mappings": filtered,
            "timestamp": datetime.utcnow().isoformat(),
            "provenance_hash": self._calculate_hash(filtered),
        }

    def get_overlap_report(self) -> Dict[str, Any]:
        """
        Generate a report of overlapping data requirements across frameworks.

        Returns:
            Overlap report with shared fields and framework pairs.
        """
        if self._cached_overlaps is not None and self.config.detect_overlaps:
            return self._cached_overlaps

        overlaps: List[Dict[str, Any]] = []
        framework_pairs: Dict[str, int] = {}

        for bundle_field, entry in UNIFIED_MAPPING_TABLE.items():
            if not entry.get("overlap", False):
                continue

            participating_fws = [
                fw for fw in self._active_frameworks if fw in entry
            ]
            if len(participating_fws) < 2:
                continue

            overlap_entry = {
                "bundle_field": bundle_field,
                "data_type": entry.get("data_type", "unknown"),
                "frameworks": {
                    fw: entry[fw] for fw in participating_fws
                },
                "framework_count": len(participating_fws),
            }
            overlaps.append(overlap_entry)

            for i, fw1 in enumerate(participating_fws):
                for fw2 in participating_fws[i + 1:]:
                    pair_key = f"{fw1}+{fw2}"
                    framework_pairs[pair_key] = framework_pairs.get(pair_key, 0) + 1

        report = {
            "total_overlapping_fields": len(overlaps),
            "total_unified_fields": len(UNIFIED_MAPPING_TABLE),
            "overlap_percentage": (
                round(len(overlaps) / len(UNIFIED_MAPPING_TABLE) * 100, 1)
                if UNIFIED_MAPPING_TABLE
                else 0.0
            ),
            "framework_pair_overlaps": framework_pairs,
            "overlapping_fields": overlaps,
            "data_reuse_potential": self._estimate_reuse_potential(overlaps),
            "active_frameworks": self._active_frameworks,
            "timestamp": datetime.utcnow().isoformat(),
            "provenance_hash": self._calculate_hash(overlaps),
        }

        if self.config.detect_overlaps:
            self._cached_overlaps = report

        return report

    def get_framework_summary(self, framework: str) -> Dict[str, Any]:
        """
        Get mapping summary for a specific framework.

        Args:
            framework: Framework key (csrd, cbam, eudr, taxonomy).

        Returns:
            Summary of all fields mapped to this framework.
        """
        if framework not in ALL_FRAMEWORKS:
            return {
                "framework": framework,
                "error": f"Unknown framework: {framework}",
                "valid_frameworks": ALL_FRAMEWORKS,
            }

        fields: List[Dict[str, Any]] = []
        required_count = 0
        optional_count = 0

        for bundle_field, entry in UNIFIED_MAPPING_TABLE.items():
            if framework in entry:
                fw_entry = entry[framework]
                fields.append({
                    "bundle_field": bundle_field,
                    "framework_field": fw_entry["field"],
                    "regulation_ref": fw_entry["ref"],
                    "required": fw_entry["required"],
                    "data_type": entry.get("data_type", "unknown"),
                    "overlap": entry.get("overlap", False),
                })
                if fw_entry["required"]:
                    required_count += 1
                else:
                    optional_count += 1

        return {
            "framework": framework,
            "total_fields": len(fields),
            "required_fields": required_count,
            "optional_fields": optional_count,
            "fields": fields,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_active_frameworks(self) -> List[str]:
        """Return list of active frameworks based on config."""
        active: List[str] = []
        if self.config.enable_csrd:
            active.append("csrd")
        if self.config.enable_cbam:
            active.append("cbam")
        if self.config.enable_eudr:
            active.append("eudr")
        if self.config.enable_taxonomy:
            active.append("taxonomy")
        return active

    def _estimate_reuse_potential(
        self, overlaps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Estimate data reuse potential from overlapping fields."""
        if not overlaps:
            return {
                "reuse_score": 0.0,
                "fields_reusable": 0,
                "estimated_effort_reduction_pct": 0.0,
            }

        total_fields = len(UNIFIED_MAPPING_TABLE)
        reusable = len(overlaps)
        total_framework_fields = sum(
            o.get("framework_count", 0) for o in overlaps
        )
        deduplicated = reusable

        effort_reduction = (
            round((total_framework_fields - deduplicated) / total_framework_fields * 100, 1)
            if total_framework_fields > 0
            else 0.0
        )

        return {
            "reuse_score": round(reusable / total_fields * 100, 1) if total_fields > 0 else 0.0,
            "fields_reusable": reusable,
            "total_framework_field_refs": total_framework_fields,
            "deduplicated_collections": deduplicated,
            "estimated_effort_reduction_pct": effort_reduction,
        }

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
