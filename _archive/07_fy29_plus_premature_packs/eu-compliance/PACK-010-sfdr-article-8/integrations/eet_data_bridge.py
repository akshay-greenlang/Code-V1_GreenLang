# -*- coding: utf-8 -*-
"""
EETDataBridge - European ESG Template Import/Export
====================================================

This module provides bidirectional European ESG Template (EET) integration
for SFDR Article 8 products. It imports EET data from data providers and
exports EET fields for distribution to distributors and fund platforms.

The EET is the industry-standard template for exchanging ESG-related data
between product manufacturers and distributors to support MiFID II
suitability assessments and SFDR disclosures.

Architecture:
    Data Providers --> EETDataBridge (import) --> SFDR Engine Data
    SFDR Engine Data --> EETDataBridge (export) --> Distributors

Example:
    >>> config = EETDataBridgeConfig()
    >>> bridge = EETDataBridge(config)
    >>> eet = bridge.import_eet(raw_eet_data)
    >>> exported = bridge.export_eet(sfdr_data)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow
from greenlang.schemas.enums import ReportFormat

logger = logging.getLogger(__name__)

# =============================================================================
# Utility Helpers
# =============================================================================

def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

# =============================================================================
# Enums
# =============================================================================

class EETVersion(str, Enum):
    """EET specification version."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"

class EETSection(str, Enum):
    """EET field section."""
    PRODUCT_INFO = "product_info"
    SFDR_CLASSIFICATION = "sfdr_classification"
    PAI = "pai"
    TAXONOMY = "taxonomy"
    SUSTAINABILITY = "sustainability"
    COSTS = "costs"
    TARGET_MARKET = "target_market"

# =============================================================================
# Data Models
# =============================================================================

class EETDataBridgeConfig(BaseModel):
    """Configuration for the EET Data Bridge."""
    eet_version: EETVersion = Field(
        default=EETVersion.V2_0,
        description="EET specification version",
    )
    export_format: ReportFormat = Field(
        default=ReportFormat.CSV,
        description="Default export format",
    )
    include_optional_fields: bool = Field(
        default=True,
        description="Include optional EET fields in export",
    )
    validate_on_import: bool = Field(
        default=True,
        description="Validate EET data on import",
    )
    validate_on_export: bool = Field(
        default=True,
        description="Validate EET data on export",
    )
    strict_mode: bool = Field(
        default=False,
        description="Reject records with any validation errors",
    )

class EETField(BaseModel):
    """A single EET field definition."""
    field_id: str = Field(default="", description="EET field identifier")
    field_name: str = Field(default="", description="Human-readable name")
    section: EETSection = Field(
        default=EETSection.PRODUCT_INFO, description="EET section"
    )
    data_type: str = Field(default="string", description="Data type")
    required: bool = Field(default=False, description="Whether required")
    sfdr_relevant: bool = Field(default=True, description="SFDR-specific field")
    value: Any = Field(default=None, description="Field value")
    validation_status: str = Field(default="", description="Validation status")

class EETImportResult(BaseModel):
    """Result of an EET import."""
    total_fields: int = Field(default=0, description="Total EET fields processed")
    populated_fields: int = Field(default=0, description="Fields with values")
    empty_fields: int = Field(default=0, description="Fields without values")
    sfdr_fields_populated: int = Field(
        default=0, description="SFDR-specific fields populated"
    )
    validation_passed: bool = Field(default=True, description="Validation result")
    validation_errors: List[str] = Field(
        default_factory=list, description="Validation errors"
    )
    validation_warnings: List[str] = Field(
        default_factory=list, description="Validation warnings"
    )
    fields: Dict[str, Any] = Field(
        default_factory=dict, description="Parsed EET field values"
    )
    sections: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Fields organized by section"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    imported_at: str = Field(default="", description="Import timestamp")

class EETExportResult(BaseModel):
    """Result of an EET export."""
    total_fields: int = Field(default=0, description="Total fields exported")
    format: str = Field(default="csv", description="Export format")
    eet_version: str = Field(default="2.0", description="EET version")
    content: str = Field(default="", description="Serialized EET content")
    validation_passed: bool = Field(default=True, description="Validation result")
    validation_errors: List[str] = Field(
        default_factory=list, description="Validation errors"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    exported_at: str = Field(default="", description="Export timestamp")

# =============================================================================
# EET SFDR Field Mappings (80+ fields)
# =============================================================================

EET_SFDR_FIELDS: Dict[str, Dict[str, Any]] = {
    # Product Info
    "EET_00010": {"name": "Product ISIN", "section": "product_info", "required": True, "type": "string"},
    "EET_00020": {"name": "Product name", "section": "product_info", "required": True, "type": "string"},
    "EET_00030": {"name": "Product domicile", "section": "product_info", "required": True, "type": "string"},
    "EET_00040": {"name": "Product currency", "section": "product_info", "required": True, "type": "string"},
    "EET_00050": {"name": "Management company", "section": "product_info", "required": True, "type": "string"},
    "EET_00060": {"name": "LEI code", "section": "product_info", "required": False, "type": "string"},
    "EET_00070": {"name": "Fund type", "section": "product_info", "required": True, "type": "string"},
    "EET_00080": {"name": "Reference date", "section": "product_info", "required": True, "type": "date"},

    # SFDR Classification
    "EET_10010": {"name": "SFDR classification", "section": "sfdr_classification", "required": True, "type": "enum"},
    "EET_10020": {"name": "Sustainable investment objective (Y/N)", "section": "sfdr_classification", "required": True, "type": "boolean"},
    "EET_10030": {"name": "Promotes E/S characteristics (Y/N)", "section": "sfdr_classification", "required": True, "type": "boolean"},
    "EET_10040": {"name": "Environmental characteristics", "section": "sfdr_classification", "required": False, "type": "list"},
    "EET_10050": {"name": "Social characteristics", "section": "sfdr_classification", "required": False, "type": "list"},
    "EET_10060": {"name": "Sustainable investment minimum %", "section": "sfdr_classification", "required": False, "type": "number"},
    "EET_10070": {"name": "Sustainable investment with env objective %", "section": "sfdr_classification", "required": False, "type": "number"},
    "EET_10080": {"name": "Sustainable investment with soc objective %", "section": "sfdr_classification", "required": False, "type": "number"},

    # PAI Indicators
    "EET_20010": {"name": "PAI considered (Y/N)", "section": "pai", "required": True, "type": "boolean"},
    "EET_20020": {"name": "PAI 1: GHG emissions Scope 1", "section": "pai", "required": True, "type": "number"},
    "EET_20030": {"name": "PAI 1: GHG emissions Scope 2", "section": "pai", "required": True, "type": "number"},
    "EET_20040": {"name": "PAI 1: GHG emissions Scope 3", "section": "pai", "required": True, "type": "number"},
    "EET_20050": {"name": "PAI 1: GHG emissions total", "section": "pai", "required": True, "type": "number"},
    "EET_20060": {"name": "PAI 2: Carbon footprint", "section": "pai", "required": True, "type": "number"},
    "EET_20070": {"name": "PAI 3: GHG intensity", "section": "pai", "required": True, "type": "number"},
    "EET_20080": {"name": "PAI 4: Fossil fuel exposure %", "section": "pai", "required": True, "type": "number"},
    "EET_20090": {"name": "PAI 5: Non-renewable energy share %", "section": "pai", "required": True, "type": "number"},
    "EET_20100": {"name": "PAI 6: Energy intensity", "section": "pai", "required": True, "type": "number"},
    "EET_20110": {"name": "PAI 7: Biodiversity-sensitive areas", "section": "pai", "required": True, "type": "number"},
    "EET_20120": {"name": "PAI 8: Emissions to water", "section": "pai", "required": True, "type": "number"},
    "EET_20130": {"name": "PAI 9: Hazardous waste ratio", "section": "pai", "required": True, "type": "number"},
    "EET_20140": {"name": "PAI 10: UNGC/OECD violations", "section": "pai", "required": True, "type": "number"},
    "EET_20150": {"name": "PAI 11: UNGC/OECD processes %", "section": "pai", "required": True, "type": "number"},
    "EET_20160": {"name": "PAI 12: Gender pay gap", "section": "pai", "required": True, "type": "number"},
    "EET_20170": {"name": "PAI 13: Board gender diversity %", "section": "pai", "required": True, "type": "number"},
    "EET_20180": {"name": "PAI 14: Controversial weapons %", "section": "pai", "required": True, "type": "number"},
    "EET_20190": {"name": "PAI 1-14 data coverage %", "section": "pai", "required": False, "type": "number"},

    # Taxonomy
    "EET_30010": {"name": "Taxonomy alignment turnover %", "section": "taxonomy", "required": True, "type": "number"},
    "EET_30020": {"name": "Taxonomy alignment CapEx %", "section": "taxonomy", "required": True, "type": "number"},
    "EET_30030": {"name": "Taxonomy alignment OpEx %", "section": "taxonomy", "required": False, "type": "number"},
    "EET_30040": {"name": "Taxonomy eligible turnover %", "section": "taxonomy", "required": True, "type": "number"},
    "EET_30050": {"name": "Taxonomy eligible CapEx %", "section": "taxonomy", "required": True, "type": "number"},
    "EET_30060": {"name": "Taxonomy eligible OpEx %", "section": "taxonomy", "required": False, "type": "number"},
    "EET_30070": {"name": "Climate mitigation aligned %", "section": "taxonomy", "required": False, "type": "number"},
    "EET_30080": {"name": "Climate adaptation aligned %", "section": "taxonomy", "required": False, "type": "number"},
    "EET_30090": {"name": "Water aligned %", "section": "taxonomy", "required": False, "type": "number"},
    "EET_30100": {"name": "Circular economy aligned %", "section": "taxonomy", "required": False, "type": "number"},
    "EET_30110": {"name": "Pollution prevention aligned %", "section": "taxonomy", "required": False, "type": "number"},
    "EET_30120": {"name": "Biodiversity aligned %", "section": "taxonomy", "required": False, "type": "number"},
    "EET_30130": {"name": "Fossil gas aligned turnover %", "section": "taxonomy", "required": False, "type": "number"},
    "EET_30140": {"name": "Nuclear aligned turnover %", "section": "taxonomy", "required": False, "type": "number"},
    "EET_30150": {"name": "Transitional activities %", "section": "taxonomy", "required": False, "type": "number"},
    "EET_30160": {"name": "Enabling activities %", "section": "taxonomy", "required": False, "type": "number"},

    # Sustainability
    "EET_40010": {"name": "DNSH assessment (Y/N)", "section": "sustainability", "required": True, "type": "boolean"},
    "EET_40020": {"name": "Good governance check (Y/N)", "section": "sustainability", "required": True, "type": "boolean"},
    "EET_40030": {"name": "Minimum safeguards (Y/N)", "section": "sustainability", "required": True, "type": "boolean"},
    "EET_40040": {"name": "Exclusion policy applied (Y/N)", "section": "sustainability", "required": True, "type": "boolean"},
    "EET_40050": {"name": "ESG integration approach", "section": "sustainability", "required": False, "type": "string"},
    "EET_40060": {"name": "Engagement policy (Y/N)", "section": "sustainability", "required": False, "type": "boolean"},
    "EET_40070": {"name": "Proxy voting policy (Y/N)", "section": "sustainability", "required": False, "type": "boolean"},
    "EET_40080": {"name": "Sustainability risk integration", "section": "sustainability", "required": True, "type": "string"},
    "EET_40090": {"name": "Sustainability risk impact (positive/negative/neutral)", "section": "sustainability", "required": False, "type": "enum"},
    "EET_40100": {"name": "Data quality score", "section": "sustainability", "required": False, "type": "number"},

    # Costs
    "EET_50010": {"name": "Ongoing charges %", "section": "costs", "required": False, "type": "number"},
    "EET_50020": {"name": "Transaction costs %", "section": "costs", "required": False, "type": "number"},

    # Target Market
    "EET_60010": {"name": "Sustainability preferences: ESG approach", "section": "target_market", "required": False, "type": "string"},
    "EET_60020": {"name": "Sustainability preferences: minimum taxonomy %", "section": "target_market", "required": False, "type": "number"},
    "EET_60030": {"name": "Sustainability preferences: minimum sustainable %", "section": "target_market", "required": False, "type": "number"},
    "EET_60040": {"name": "Sustainability preferences: PAI considered", "section": "target_market", "required": False, "type": "boolean"},
    "EET_60050": {"name": "MiFID II sustainability preferences alignment", "section": "target_market", "required": False, "type": "string"},

    # Additional SFDR fields
    "EET_70010": {"name": "Website disclosure URL", "section": "sfdr_classification", "required": False, "type": "string"},
    "EET_70020": {"name": "Pre-contractual annex reference", "section": "sfdr_classification", "required": False, "type": "string"},
    "EET_70030": {"name": "Periodic disclosure annex reference", "section": "sfdr_classification", "required": False, "type": "string"},
    "EET_70040": {"name": "Index designation (Y/N)", "section": "sfdr_classification", "required": False, "type": "boolean"},
    "EET_70050": {"name": "Designated index name", "section": "sfdr_classification", "required": False, "type": "string"},
    "EET_70060": {"name": "Benchmark alignment description", "section": "sfdr_classification", "required": False, "type": "string"},
    "EET_70070": {"name": "Binding elements description", "section": "sustainability", "required": False, "type": "string"},
    "EET_70080": {"name": "Investment strategy description", "section": "sustainability", "required": False, "type": "string"},
    "EET_70090": {"name": "Asset allocation description", "section": "sustainability", "required": False, "type": "string"},
    "EET_70100": {"name": "Monitoring methodology", "section": "sustainability", "required": False, "type": "string"},
}

# =============================================================================
# EET Data Bridge
# =============================================================================

class EETDataBridge:
    """Bridge for importing and exporting European ESG Template data.

    Provides bidirectional EET integration: imports EET data from data
    providers for SFDR processing, and exports SFDR engine outputs as
    EET fields for distribution.

    Attributes:
        config: Bridge configuration.

    Example:
        >>> bridge = EETDataBridge(EETDataBridgeConfig())
        >>> imported = bridge.import_eet({"EET_10010": "article_8"})
        >>> exported = bridge.export_eet(sfdr_data)
    """

    def __init__(self, config: Optional[EETDataBridgeConfig] = None) -> None:
        """Initialize the EET Data Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or EETDataBridgeConfig()
        self.logger = logger

        self.logger.info(
            "EETDataBridge initialized: version=%s, format=%s",
            self.config.eet_version.value,
            self.config.export_format.value,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def import_eet(
        self,
        raw_data: Dict[str, Any],
    ) -> EETImportResult:
        """Import EET data from a data provider.

        Parses raw EET field-value pairs, validates against the EET
        specification, and organizes fields by section.

        Args:
            raw_data: Dictionary of EET field IDs to values.

        Returns:
            EETImportResult with parsed and validated fields.
        """
        errors: List[str] = []
        warnings: List[str] = []
        parsed_fields: Dict[str, Any] = {}
        sections: Dict[str, Dict[str, Any]] = {}
        populated = 0
        empty = 0
        sfdr_populated = 0

        for field_id, field_def in EET_SFDR_FIELDS.items():
            value = raw_data.get(field_id)
            section = field_def["section"]

            if section not in sections:
                sections[section] = {}

            if value is not None and value != "":
                parsed_fields[field_id] = value
                sections[section][field_id] = {
                    "name": field_def["name"],
                    "value": value,
                    "type": field_def["type"],
                }
                populated += 1
                sfdr_populated += 1
            else:
                empty += 1
                if field_def["required"] and self.config.validate_on_import:
                    if self.config.strict_mode:
                        errors.append(
                            f"Required field {field_id} ({field_def['name']}) is missing"
                        )
                    else:
                        warnings.append(
                            f"Required field {field_id} ({field_def['name']}) is missing"
                        )

        # Type validation
        if self.config.validate_on_import:
            type_errors = self._validate_field_types(parsed_fields)
            errors.extend(type_errors)

        total = populated + empty
        validation_passed = len(errors) == 0

        result = EETImportResult(
            total_fields=total,
            populated_fields=populated,
            empty_fields=empty,
            sfdr_fields_populated=sfdr_populated,
            validation_passed=validation_passed,
            validation_errors=errors,
            validation_warnings=warnings,
            fields=parsed_fields,
            sections=sections,
            imported_at=utcnow().isoformat(),
        )
        result.provenance_hash = _hash_data({
            "total": total, "populated": populated,
            "sfdr": sfdr_populated, "valid": validation_passed,
        })

        self.logger.info(
            "EET imported: %d/%d fields populated, %d SFDR fields, valid=%s",
            populated, total, sfdr_populated, validation_passed,
        )
        return result

    def export_eet(
        self,
        sfdr_data: Dict[str, Any],
        export_format: Optional[ReportFormat] = None,
    ) -> EETExportResult:
        """Export SFDR engine data as EET fields.

        Maps SFDR engine outputs to EET field identifiers and serializes
        the result in the configured format.

        Args:
            sfdr_data: SFDR engine output data to export.
            export_format: Override export format.

        Returns:
            EETExportResult with serialized EET content.
        """
        fmt = export_format or self.config.export_format
        errors: List[str] = []
        eet_fields: Dict[str, Any] = {}

        # Map SFDR data to EET fields
        eet_fields.update(self._map_product_info(sfdr_data))
        eet_fields.update(self._map_classification(sfdr_data))
        eet_fields.update(self._map_pai_indicators(sfdr_data))
        eet_fields.update(self._map_taxonomy(sfdr_data))
        eet_fields.update(self._map_sustainability(sfdr_data))

        # Validate export
        if self.config.validate_on_export:
            for field_id, field_def in EET_SFDR_FIELDS.items():
                if field_def["required"] and field_id not in eet_fields:
                    errors.append(
                        f"Required export field {field_id} ({field_def['name']}) "
                        "not populated"
                    )

        # Serialize
        content = self._serialize_eet(eet_fields, fmt)

        result = EETExportResult(
            total_fields=len(eet_fields),
            format=fmt.value,
            eet_version=self.config.eet_version.value,
            content=content,
            validation_passed=len(errors) == 0,
            validation_errors=errors,
            exported_at=utcnow().isoformat(),
        )
        result.provenance_hash = _hash_data({
            "fields": len(eet_fields), "format": fmt.value,
        })

        self.logger.info(
            "EET exported: %d fields, format=%s, valid=%s",
            len(eet_fields), fmt.value, result.validation_passed,
        )
        return result

    def map_to_eet_fields(
        self,
        sfdr_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map SFDR data to EET field identifiers without serialization.

        Args:
            sfdr_data: SFDR engine output data.

        Returns:
            Dictionary of EET field IDs to values.
        """
        eet_fields: Dict[str, Any] = {}
        eet_fields.update(self._map_product_info(sfdr_data))
        eet_fields.update(self._map_classification(sfdr_data))
        eet_fields.update(self._map_pai_indicators(sfdr_data))
        eet_fields.update(self._map_taxonomy(sfdr_data))
        eet_fields.update(self._map_sustainability(sfdr_data))
        return eet_fields

    def validate_eet(
        self,
        eet_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate an EET dataset against the specification.

        Args:
            eet_data: EET field-value pairs to validate.

        Returns:
            Validation report with errors, warnings, and coverage.
        """
        errors: List[str] = []
        warnings: List[str] = []
        populated = 0
        missing_required = 0

        for field_id, field_def in EET_SFDR_FIELDS.items():
            value = eet_data.get(field_id)
            if value is not None and value != "":
                populated += 1
            elif field_def["required"]:
                missing_required += 1
                errors.append(
                    f"Required: {field_id} ({field_def['name']})"
                )

        type_errors = self._validate_field_types(eet_data)
        errors.extend(type_errors)

        total = len(EET_SFDR_FIELDS)
        coverage = round((populated / max(total, 1)) * 100, 1)

        return {
            "valid": len(errors) == 0,
            "total_fields": total,
            "populated": populated,
            "missing_required": missing_required,
            "coverage_pct": coverage,
            "errors": errors,
            "warnings": warnings,
            "provenance_hash": _hash_data({
                "valid": len(errors) == 0, "coverage": coverage,
            }),
        }

    def get_sfdr_section(
        self,
        section: str,
    ) -> List[Dict[str, Any]]:
        """Get all EET fields belonging to an SFDR section.

        Args:
            section: Section name (product_info, sfdr_classification, pai, etc.).

        Returns:
            List of field definitions in the section.
        """
        result: List[Dict[str, Any]] = []
        for field_id, field_def in EET_SFDR_FIELDS.items():
            if field_def["section"] == section:
                result.append({
                    "field_id": field_id,
                    "name": field_def["name"],
                    "type": field_def["type"],
                    "required": field_def["required"],
                })
        return result

    # -------------------------------------------------------------------------
    # Internal Mapping Methods
    # -------------------------------------------------------------------------

    def _map_product_info(self, sfdr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map product info fields to EET."""
        return {
            "EET_00010": sfdr_data.get("product_isin", ""),
            "EET_00020": sfdr_data.get("product_name", ""),
            "EET_00030": sfdr_data.get("product_domicile", ""),
            "EET_00040": sfdr_data.get("reporting_currency", "EUR"),
            "EET_00050": sfdr_data.get("management_company", ""),
            "EET_00060": sfdr_data.get("lei_code", ""),
            "EET_00070": sfdr_data.get("fund_type", "UCITS"),
            "EET_00080": sfdr_data.get("reference_date", utcnow().isoformat()[:10]),
        }

    def _map_classification(self, sfdr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map SFDR classification fields to EET."""
        classification = sfdr_data.get("sfdr_classification", "article_8")
        return {
            "EET_10010": classification,
            "EET_10020": "N" if classification in ("article_8", "article_6") else "Y",
            "EET_10030": "Y" if "article_8" in classification else "N",
            "EET_10040": sfdr_data.get("environmental_characteristics", []),
            "EET_10050": sfdr_data.get("social_characteristics", []),
            "EET_10060": sfdr_data.get("sustainable_investment_min_pct", 0.0),
            "EET_10070": sfdr_data.get("sustainable_env_pct", 0.0),
            "EET_10080": sfdr_data.get("sustainable_soc_pct", 0.0),
        }

    def _map_pai_indicators(self, sfdr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map PAI indicator values to EET fields."""
        pai = sfdr_data.get("pai_indicators", {})
        return {
            "EET_20010": "Y" if pai else "N",
            "EET_20020": pai.get("scope_1_tco2e", 0.0),
            "EET_20030": pai.get("scope_2_tco2e", 0.0),
            "EET_20040": pai.get("scope_3_tco2e", 0.0),
            "EET_20050": pai.get("total_tco2e", 0.0),
            "EET_20060": pai.get("carbon_footprint", 0.0),
            "EET_20070": pai.get("ghg_intensity", 0.0),
            "EET_20080": pai.get("fossil_fuel_exposure_pct", 0.0),
            "EET_20090": pai.get("non_renewable_energy_pct", 0.0),
            "EET_20100": pai.get("energy_intensity", 0.0),
            "EET_20110": pai.get("biodiversity_count", 0),
            "EET_20120": pai.get("emissions_to_water", 0.0),
            "EET_20130": pai.get("hazardous_waste_ratio", 0.0),
            "EET_20140": pai.get("ungc_violations", 0),
            "EET_20150": pai.get("ungc_processes_pct", 0.0),
            "EET_20160": pai.get("gender_pay_gap", 0.0),
            "EET_20170": pai.get("board_gender_diversity_pct", 0.0),
            "EET_20180": pai.get("controversial_weapons_pct", 0.0),
            "EET_20190": pai.get("data_coverage_pct", 0.0),
        }

    def _map_taxonomy(self, sfdr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map taxonomy alignment data to EET fields."""
        tax = sfdr_data.get("taxonomy", {})
        return {
            "EET_30010": tax.get("aligned_turnover_pct", 0.0),
            "EET_30020": tax.get("aligned_capex_pct", 0.0),
            "EET_30030": tax.get("aligned_opex_pct", 0.0),
            "EET_30040": tax.get("eligible_turnover_pct", 0.0),
            "EET_30050": tax.get("eligible_capex_pct", 0.0),
            "EET_30060": tax.get("eligible_opex_pct", 0.0),
            "EET_30070": tax.get("climate_mitigation_pct", 0.0),
            "EET_30080": tax.get("climate_adaptation_pct", 0.0),
            "EET_30090": tax.get("water_pct", 0.0),
            "EET_30100": tax.get("circular_economy_pct", 0.0),
            "EET_30110": tax.get("pollution_pct", 0.0),
            "EET_30120": tax.get("biodiversity_pct", 0.0),
            "EET_30130": tax.get("fossil_gas_pct", 0.0),
            "EET_30140": tax.get("nuclear_pct", 0.0),
            "EET_30150": tax.get("transitional_pct", 0.0),
            "EET_30160": tax.get("enabling_pct", 0.0),
        }

    def _map_sustainability(self, sfdr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map sustainability fields to EET."""
        sus = sfdr_data.get("sustainability", {})
        return {
            "EET_40010": "Y" if sus.get("dnsh_assessment", False) else "N",
            "EET_40020": "Y" if sus.get("good_governance", False) else "N",
            "EET_40030": "Y" if sus.get("minimum_safeguards", False) else "N",
            "EET_40040": "Y" if sus.get("exclusion_policy", False) else "N",
            "EET_40050": sus.get("esg_integration_approach", ""),
            "EET_40060": "Y" if sus.get("engagement_policy", False) else "N",
            "EET_40070": "Y" if sus.get("proxy_voting", False) else "N",
            "EET_40080": sus.get("sustainability_risk_integration", ""),
            "EET_40090": sus.get("sustainability_risk_impact", "neutral"),
            "EET_40100": sus.get("data_quality_score", 0.0),
        }

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def _serialize_eet(
        self, fields: Dict[str, Any], fmt: ReportFormat
    ) -> str:
        """Serialize EET fields to the specified format.

        Args:
            fields: EET field-value pairs.
            fmt: Target format.

        Returns:
            Serialized string.
        """
        if fmt == ReportFormat.JSON:
            return json.dumps(fields, indent=2, default=str)

        elif fmt == ReportFormat.CSV:
            lines = ["field_id,field_name,value"]
            for field_id, value in sorted(fields.items()):
                field_def = EET_SFDR_FIELDS.get(field_id, {})
                name = field_def.get("name", field_id)
                val_str = str(value).replace('"', '""')
                lines.append(f'{field_id},"{name}","{val_str}"')
            return "\n".join(lines)

        elif fmt == ReportFormat.XML:
            parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<EET>"]
            for field_id, value in sorted(fields.items()):
                val_str = str(value)
                parts.append(f"  <{field_id}>{val_str}</{field_id}>")
            parts.append("</EET>")
            return "\n".join(parts)

        return json.dumps(fields, default=str)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_field_types(
        self, fields: Dict[str, Any]
    ) -> List[str]:
        """Validate EET field values against their declared types.

        Args:
            fields: EET field-value pairs.

        Returns:
            List of type validation errors.
        """
        errors: List[str] = []

        for field_id, value in fields.items():
            field_def = EET_SFDR_FIELDS.get(field_id)
            if field_def is None:
                continue

            expected_type = field_def["type"]

            if expected_type == "number":
                try:
                    float(value)
                except (ValueError, TypeError):
                    errors.append(
                        f"{field_id}: expected number, got '{type(value).__name__}'"
                    )

            elif expected_type == "boolean":
                if value not in (True, False, "Y", "N", "y", "n", 1, 0):
                    errors.append(
                        f"{field_id}: expected boolean, got '{value}'"
                    )

        return errors
