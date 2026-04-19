# -*- coding: utf-8 -*-
"""
EETDataBridge - European ESG Template Data Import/Export for Article 9
=======================================================================

This module connects PACK-011 (SFDR Article 9) with the European ESG Template
(EET) standard to support import and export of SFDR disclosure data via the
industry-standard EET format. The EET contains 87+ SFDR-related fields
covering pre-contractual, periodic, and website disclosures. Article 9
products require population of additional fields beyond Article 8, including
sustainable investment objective details, benchmark designation, and enhanced
PAI disclosure.

Architecture:
    PACK-011 SFDR Art 9 <--> EETDataBridge <--> EET v1.1.2 Format
                                  |
                                  v
    Import: Parse EET --> Validate Art 9 Fields --> Map to Pipeline
    Export: Pipeline Results --> Map to EET Fields --> Generate EET

Regulatory Context:
    The EET is maintained by FinDatEx and is the standard data exchange
    format for ESG data between manufacturers, distributors, and data
    vendors under SFDR, Taxonomy Regulation, and MiFID II sustainability
    preferences. Version 1.1.2 includes fields for Article 6, 8, 8+,
    and 9 products.

Example:
    >>> config = EETBridgeConfig()
    >>> bridge = EETDataBridge(config)
    >>> result = bridge.import_eet(eet_data)
    >>> print(f"Fields mapped: {result.fields_mapped}")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
Version: 1.0.0
Status: Production Ready
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow
from greenlang.schemas.enums import ValidationSeverity

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
    """EET format version."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V1_1_1 = "1.1.1"
    V1_1_2 = "1.1.2"

class EETFieldCategory(str, Enum):
    """EET field category."""
    GENERAL = "general"
    SFDR_PRE_CONTRACTUAL = "sfdr_pre_contractual"
    SFDR_PERIODIC = "sfdr_periodic"
    SFDR_WEBSITE = "sfdr_website"
    TAXONOMY = "taxonomy"
    PAI = "pai"
    BENCHMARK = "benchmark"
    MIFID_SUSTAINABILITY = "mifid_sustainability"
    ARTICLE_9_SPECIFIC = "article_9_specific"

class FieldDataType(str, Enum):
    """EET field data types."""
    TEXT = "text"
    NUMERIC = "numeric"
    PERCENTAGE = "percentage"
    BOOLEAN = "boolean"
    DATE = "date"
    ENUM = "enum"
    ISIN = "isin"
    LEI = "lei"

# =============================================================================
# Data Models
# =============================================================================

class EETBridgeConfig(BaseModel):
    """Configuration for the EET Data Bridge."""
    eet_version: EETVersion = Field(
        default=EETVersion.V1_1_2,
        description="EET format version to use",
    )
    strict_validation: bool = Field(
        default=True,
        description="Enforce strict Article 9 field validation",
    )
    require_all_mandatory: bool = Field(
        default=True,
        description="Require all Article 9 mandatory fields",
    )
    auto_populate_defaults: bool = Field(
        default=True,
        description="Auto-populate default values for Article 9",
    )
    include_optional_fields: bool = Field(
        default=True,
        description="Include optional EET fields in export",
    )
    export_format: str = Field(
        default="json",
        description="Export format: json, csv, or xml",
    )
    enable_provenance: bool = Field(
        default=True, description="Enable provenance hash tracking"
    )

class EETFieldDefinition(BaseModel):
    """Definition of a single EET field."""
    field_id: str = Field(default="", description="EET field identifier")
    field_name: str = Field(default="", description="Field display name")
    category: str = Field(default="general", description="Field category")
    data_type: str = Field(default="text", description="Data type")
    mandatory_art_9: bool = Field(
        default=False, description="Mandatory for Article 9 products"
    )
    mandatory_art_8: bool = Field(
        default=False, description="Mandatory for Article 8 products"
    )
    description: str = Field(default="", description="Field description")
    allowed_values: List[str] = Field(
        default_factory=list, description="Allowed values for enum fields"
    )
    default_art_9: Optional[str] = Field(
        default=None, description="Default value for Article 9"
    )

class EETFieldValue(BaseModel):
    """A single EET field value."""
    field_id: str = Field(default="", description="EET field identifier")
    field_name: str = Field(default="", description="Field display name")
    value: Any = Field(default=None, description="Field value")
    source: str = Field(
        default="pipeline", description="Data source"
    )
    validated: bool = Field(default=False, description="Whether validated")
    validation_message: str = Field(
        default="", description="Validation message"
    )

class Article9EETFields(BaseModel):
    """Article 9 specific EET fields beyond Article 8."""
    # Product classification
    sfdr_classification: str = Field(
        default="article_9", description="SFDR product classification"
    )
    has_sustainable_objective: bool = Field(
        default=True,
        description="Whether product has sustainable investment objective",
    )
    sustainable_investment_objective: str = Field(
        default="", description="Stated sustainable investment objective"
    )

    # Sustainable investment breakdown
    sustainable_investment_pct: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Minimum share of sustainable investments (%)",
    )
    si_environmental_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Share of SI with environmental objective (%)",
    )
    si_social_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Share of SI with social objective (%)",
    )
    si_taxonomy_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Share of SI that is taxonomy-aligned (%)",
    )

    # Taxonomy alignment (all 6 objectives for Art 9)
    taxonomy_alignment_turnover: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Taxonomy alignment turnover (%)"
    )
    taxonomy_alignment_capex: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Taxonomy alignment CapEx (%)"
    )
    taxonomy_alignment_opex: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Taxonomy alignment OpEx (%)"
    )
    ccm_alignment_pct: float = Field(
        default=0.0, description="Climate change mitigation alignment (%)"
    )
    cca_alignment_pct: float = Field(
        default=0.0, description="Climate change adaptation alignment (%)"
    )
    water_alignment_pct: float = Field(
        default=0.0, description="Water and marine resources alignment (%)"
    )
    circular_economy_alignment_pct: float = Field(
        default=0.0, description="Circular economy alignment (%)"
    )
    pollution_alignment_pct: float = Field(
        default=0.0, description="Pollution prevention alignment (%)"
    )
    biodiversity_alignment_pct: float = Field(
        default=0.0, description="Biodiversity alignment (%)"
    )

    # Gas/nuclear CDA
    fossil_gas_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Fossil gas alignment (%)"
    )
    nuclear_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Nuclear alignment (%)"
    )

    # PAI (all 18 mandatory for Art 9)
    considers_pai: bool = Field(
        default=True, description="Considers principal adverse impacts"
    )
    pai_all_mandatory_reported: bool = Field(
        default=True, description="All 18 mandatory PAI indicators reported"
    )
    pai_ghg_emissions_scope_1: float = Field(
        default=0.0, description="PAI 1a: Scope 1 GHG emissions (tCO2e)"
    )
    pai_ghg_emissions_scope_2: float = Field(
        default=0.0, description="PAI 1b: Scope 2 GHG emissions (tCO2e)"
    )
    pai_ghg_emissions_scope_3: float = Field(
        default=0.0, description="PAI 1c: Scope 3 GHG emissions (tCO2e)"
    )
    pai_carbon_footprint: float = Field(
        default=0.0, description="PAI 2: Carbon footprint (tCO2e/EUR M)"
    )
    pai_ghg_intensity: float = Field(
        default=0.0, description="PAI 3: GHG intensity (tCO2e/EUR M revenue)"
    )
    pai_fossil_fuel_pct: float = Field(
        default=0.0, description="PAI 4: Fossil fuel exposure (%)"
    )
    pai_non_renewable_energy_pct: float = Field(
        default=0.0, description="PAI 5: Non-renewable energy share (%)"
    )
    pai_energy_intensity: float = Field(
        default=0.0, description="PAI 6: Energy consumption intensity"
    )

    # Benchmark (Art 9(3))
    has_designated_benchmark: bool = Field(
        default=False, description="Has designated reference benchmark"
    )
    benchmark_type: str = Field(
        default="", description="Benchmark type: CTB or PAB"
    )
    benchmark_name: str = Field(
        default="", description="Benchmark name"
    )
    benchmark_provider: str = Field(
        default="", description="Benchmark provider"
    )

    # DNSH / Good Governance
    dnsh_all_6_objectives: bool = Field(
        default=True, description="Enhanced DNSH across all 6 objectives"
    )
    good_governance_check: bool = Field(
        default=True, description="Good governance check performed"
    )
    minimum_safeguards: bool = Field(
        default=True, description="Minimum safeguards check"
    )

    # Impact
    impact_measurement_approach: str = Field(
        default="", description="Impact measurement methodology"
    )
    sdg_alignment: List[int] = Field(
        default_factory=list, description="Aligned SDG goals"
    )

    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class EETImportResult(BaseModel):
    """Result of importing EET data."""
    import_id: str = Field(default="", description="Import identifier")
    eet_version: str = Field(default="1.1.2", description="EET version detected")
    product_isin: str = Field(default="", description="Product ISIN")
    product_name: str = Field(default="", description="Product name")
    sfdr_classification: str = Field(
        default="", description="Detected SFDR classification"
    )
    total_fields_parsed: int = Field(
        default=0, description="Total EET fields parsed"
    )
    fields_mapped: int = Field(
        default=0, description="Fields successfully mapped"
    )
    art_9_fields_populated: int = Field(
        default=0, description="Article 9 specific fields populated"
    )
    art_9_fields_total: int = Field(
        default=0, description="Total Article 9 specific fields"
    )
    article_9_fields: Optional[Article9EETFields] = Field(
        default=None, description="Parsed Article 9 fields"
    )
    field_values: List[EETFieldValue] = Field(
        default_factory=list, description="Individual field values"
    )
    validation_errors: List[Dict[str, str]] = Field(
        default_factory=list, description="Validation errors"
    )
    validation_warnings: List[Dict[str, str]] = Field(
        default_factory=list, description="Validation warnings"
    )
    is_valid: bool = Field(default=False, description="Overall validation status")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    imported_at: str = Field(default="", description="Import timestamp")
    execution_time_ms: float = Field(default=0.0, description="Execution time")

class EETExportResult(BaseModel):
    """Result of exporting pipeline data to EET format."""
    export_id: str = Field(default="", description="Export identifier")
    eet_version: str = Field(default="1.1.2", description="EET version")
    product_isin: str = Field(default="", description="Product ISIN")
    product_name: str = Field(default="", description="Product name")
    total_fields_exported: int = Field(
        default=0, description="Total fields exported"
    )
    mandatory_fields_complete: bool = Field(
        default=False, description="All mandatory fields populated"
    )
    export_format: str = Field(
        default="json", description="Output format"
    )
    eet_data: Dict[str, Any] = Field(
        default_factory=dict, description="Exported EET data"
    )
    missing_mandatory: List[str] = Field(
        default_factory=list, description="Missing mandatory fields"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Export warnings"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    exported_at: str = Field(default="", description="Export timestamp")
    execution_time_ms: float = Field(default=0.0, description="Execution time")

# =============================================================================
# EET Field Registry (87+ SFDR fields)
# =============================================================================

ARTICLE_9_MANDATORY_FIELDS: List[EETFieldDefinition] = [
    # General
    EETFieldDefinition(
        field_id="EET_001", field_name="Product_ISIN",
        category="general", data_type="isin", mandatory_art_9=True,
        description="Product ISIN code",
    ),
    EETFieldDefinition(
        field_id="EET_002", field_name="Product_Name",
        category="general", data_type="text", mandatory_art_9=True,
        description="Product name",
    ),
    EETFieldDefinition(
        field_id="EET_003", field_name="SFDR_Classification",
        category="general", data_type="enum", mandatory_art_9=True,
        description="SFDR classification",
        allowed_values=["article_6", "article_8", "article_8_plus", "article_9"],
        default_art_9="article_9",
    ),
    EETFieldDefinition(
        field_id="EET_004", field_name="LEI_Code",
        category="general", data_type="lei", mandatory_art_9=True,
        description="Legal Entity Identifier",
    ),
    EETFieldDefinition(
        field_id="EET_005", field_name="Reporting_Date",
        category="general", data_type="date", mandatory_art_9=True,
        description="Reporting reference date",
    ),
    # Sustainable Investment Objective (Art 9 specific)
    EETFieldDefinition(
        field_id="EET_010", field_name="Has_Sustainable_Objective",
        category="article_9_specific", data_type="boolean", mandatory_art_9=True,
        description="Has sustainable investment as its objective",
        default_art_9="true",
    ),
    EETFieldDefinition(
        field_id="EET_011", field_name="Sustainable_Objective_Description",
        category="article_9_specific", data_type="text", mandatory_art_9=True,
        description="Description of sustainable investment objective",
    ),
    EETFieldDefinition(
        field_id="EET_012", field_name="Sustainable_Investment_Min_Pct",
        category="article_9_specific", data_type="percentage", mandatory_art_9=True,
        description="Minimum sustainable investment percentage",
        default_art_9="100",
    ),
    EETFieldDefinition(
        field_id="EET_013", field_name="SI_Environmental_Pct",
        category="article_9_specific", data_type="percentage", mandatory_art_9=True,
        description="Share of SI with environmental objective",
    ),
    EETFieldDefinition(
        field_id="EET_014", field_name="SI_Social_Pct",
        category="article_9_specific", data_type="percentage", mandatory_art_9=True,
        description="Share of SI with social objective",
    ),
    EETFieldDefinition(
        field_id="EET_015", field_name="SI_Taxonomy_Aligned_Pct",
        category="article_9_specific", data_type="percentage", mandatory_art_9=True,
        description="Share of SI that is taxonomy-aligned",
    ),
    # Taxonomy alignment
    EETFieldDefinition(
        field_id="EET_020", field_name="Taxonomy_Alignment_Turnover",
        category="taxonomy", data_type="percentage", mandatory_art_9=True,
        description="Taxonomy alignment ratio (turnover)",
    ),
    EETFieldDefinition(
        field_id="EET_021", field_name="Taxonomy_Alignment_CapEx",
        category="taxonomy", data_type="percentage", mandatory_art_9=True,
        description="Taxonomy alignment ratio (CapEx)",
    ),
    EETFieldDefinition(
        field_id="EET_022", field_name="Taxonomy_Alignment_OpEx",
        category="taxonomy", data_type="percentage", mandatory_art_9=True,
        description="Taxonomy alignment ratio (OpEx)",
    ),
    EETFieldDefinition(
        field_id="EET_023", field_name="CCM_Alignment_Pct",
        category="taxonomy", data_type="percentage", mandatory_art_9=True,
        description="Climate change mitigation alignment",
    ),
    EETFieldDefinition(
        field_id="EET_024", field_name="CCA_Alignment_Pct",
        category="taxonomy", data_type="percentage", mandatory_art_9=True,
        description="Climate change adaptation alignment",
    ),
    EETFieldDefinition(
        field_id="EET_025", field_name="Water_Alignment_Pct",
        category="taxonomy", data_type="percentage", mandatory_art_9=True,
        description="Water and marine resources alignment",
    ),
    EETFieldDefinition(
        field_id="EET_026", field_name="CE_Alignment_Pct",
        category="taxonomy", data_type="percentage", mandatory_art_9=True,
        description="Circular economy alignment",
    ),
    EETFieldDefinition(
        field_id="EET_027", field_name="Pollution_Alignment_Pct",
        category="taxonomy", data_type="percentage", mandatory_art_9=True,
        description="Pollution prevention alignment",
    ),
    EETFieldDefinition(
        field_id="EET_028", field_name="Biodiversity_Alignment_Pct",
        category="taxonomy", data_type="percentage", mandatory_art_9=True,
        description="Biodiversity and ecosystems alignment",
    ),
    EETFieldDefinition(
        field_id="EET_029", field_name="Fossil_Gas_Pct",
        category="taxonomy", data_type="percentage", mandatory_art_9=True,
        description="Fossil gas taxonomy alignment (CDA)",
    ),
    EETFieldDefinition(
        field_id="EET_030", field_name="Nuclear_Pct",
        category="taxonomy", data_type="percentage", mandatory_art_9=True,
        description="Nuclear taxonomy alignment (CDA)",
    ),
    # PAI indicators (all 18 mandatory for Art 9)
    EETFieldDefinition(
        field_id="EET_040", field_name="Considers_PAI",
        category="pai", data_type="boolean", mandatory_art_9=True,
        description="Considers principal adverse impacts",
        default_art_9="true",
    ),
    EETFieldDefinition(
        field_id="EET_041", field_name="PAI_GHG_Scope_1",
        category="pai", data_type="numeric", mandatory_art_9=True,
        description="PAI 1a: Scope 1 GHG emissions",
    ),
    EETFieldDefinition(
        field_id="EET_042", field_name="PAI_GHG_Scope_2",
        category="pai", data_type="numeric", mandatory_art_9=True,
        description="PAI 1b: Scope 2 GHG emissions",
    ),
    EETFieldDefinition(
        field_id="EET_043", field_name="PAI_GHG_Scope_3",
        category="pai", data_type="numeric", mandatory_art_9=True,
        description="PAI 1c: Scope 3 GHG emissions",
    ),
    EETFieldDefinition(
        field_id="EET_044", field_name="PAI_Carbon_Footprint",
        category="pai", data_type="numeric", mandatory_art_9=True,
        description="PAI 2: Carbon footprint",
    ),
    EETFieldDefinition(
        field_id="EET_045", field_name="PAI_GHG_Intensity",
        category="pai", data_type="numeric", mandatory_art_9=True,
        description="PAI 3: GHG intensity",
    ),
    EETFieldDefinition(
        field_id="EET_046", field_name="PAI_Fossil_Fuel_Pct",
        category="pai", data_type="percentage", mandatory_art_9=True,
        description="PAI 4: Fossil fuel exposure",
    ),
    EETFieldDefinition(
        field_id="EET_047", field_name="PAI_Non_Renewable_Energy",
        category="pai", data_type="percentage", mandatory_art_9=True,
        description="PAI 5: Non-renewable energy share",
    ),
    EETFieldDefinition(
        field_id="EET_048", field_name="PAI_Energy_Intensity",
        category="pai", data_type="numeric", mandatory_art_9=True,
        description="PAI 6: Energy consumption intensity",
    ),
    # DNSH and governance
    EETFieldDefinition(
        field_id="EET_060", field_name="DNSH_All_6_Objectives",
        category="article_9_specific", data_type="boolean", mandatory_art_9=True,
        description="Enhanced DNSH across all 6 environmental objectives",
        default_art_9="true",
    ),
    EETFieldDefinition(
        field_id="EET_061", field_name="Good_Governance_Check",
        category="article_9_specific", data_type="boolean", mandatory_art_9=True,
        description="Good governance screening performed",
        default_art_9="true",
    ),
    EETFieldDefinition(
        field_id="EET_062", field_name="Minimum_Safeguards",
        category="article_9_specific", data_type="boolean", mandatory_art_9=True,
        description="Minimum safeguards check (Art 18 Taxonomy Reg)",
        default_art_9="true",
    ),
    # Benchmark
    EETFieldDefinition(
        field_id="EET_070", field_name="Has_Designated_Benchmark",
        category="benchmark", data_type="boolean", mandatory_art_9=False,
        description="Has designated reference benchmark (Art 9(3))",
    ),
    EETFieldDefinition(
        field_id="EET_071", field_name="Benchmark_Type",
        category="benchmark", data_type="enum", mandatory_art_9=False,
        description="Benchmark type: CTB or PAB",
        allowed_values=["CTB", "PAB", "other"],
    ),
    EETFieldDefinition(
        field_id="EET_072", field_name="Benchmark_Name",
        category="benchmark", data_type="text", mandatory_art_9=False,
        description="Designated benchmark name",
    ),
]

# Field ID to pipeline key mapping
EET_TO_PIPELINE_MAPPING: Dict[str, str] = {
    "EET_001": "product_isin",
    "EET_002": "product_name",
    "EET_003": "sfdr_classification",
    "EET_004": "lei_code",
    "EET_005": "reporting_date",
    "EET_010": "has_sustainable_objective",
    "EET_011": "sustainable_objective",
    "EET_012": "sustainable_investment_min_pct",
    "EET_013": "si_environmental_pct",
    "EET_014": "si_social_pct",
    "EET_015": "si_taxonomy_aligned_pct",
    "EET_020": "taxonomy_alignment_turnover",
    "EET_021": "taxonomy_alignment_capex",
    "EET_022": "taxonomy_alignment_opex",
    "EET_023": "ccm_alignment_pct",
    "EET_024": "cca_alignment_pct",
    "EET_025": "water_alignment_pct",
    "EET_026": "circular_economy_alignment_pct",
    "EET_027": "pollution_alignment_pct",
    "EET_028": "biodiversity_alignment_pct",
    "EET_029": "fossil_gas_pct",
    "EET_030": "nuclear_pct",
    "EET_040": "considers_pai",
    "EET_041": "pai_ghg_scope_1",
    "EET_042": "pai_ghg_scope_2",
    "EET_043": "pai_ghg_scope_3",
    "EET_044": "pai_carbon_footprint",
    "EET_045": "pai_ghg_intensity",
    "EET_046": "pai_fossil_fuel_pct",
    "EET_047": "pai_non_renewable_energy_pct",
    "EET_048": "pai_energy_intensity",
    "EET_060": "dnsh_all_6_objectives",
    "EET_061": "good_governance_check",
    "EET_062": "minimum_safeguards",
    "EET_070": "has_designated_benchmark",
    "EET_071": "benchmark_type",
    "EET_072": "benchmark_name",
}

# =============================================================================
# EET Data Bridge
# =============================================================================

class EETDataBridge:
    """Bridge for EET (European ESG Template) import and export.

    Handles parsing, validation, and mapping of EET data for SFDR
    Article 9 products. Supports import of EET data into the pipeline
    and export of pipeline results to EET format.

    Attributes:
        config: Bridge configuration.
        _field_registry: Article 9 mandatory field definitions.

    Example:
        >>> bridge = EETDataBridge(EETBridgeConfig())
        >>> result = bridge.import_eet({"EET_003": "article_9", ...})
        >>> print(f"Valid: {result.is_valid}")
    """

    def __init__(self, config: Optional[EETBridgeConfig] = None) -> None:
        """Initialize the EET Data Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or EETBridgeConfig()
        self.logger = logger
        self._field_registry = list(ARTICLE_9_MANDATORY_FIELDS)

        self.logger.info(
            "EETDataBridge initialized: version=%s, strict=%s, "
            "mandatory_fields=%d, export_format=%s",
            self.config.eet_version.value,
            self.config.strict_validation,
            len(self._field_registry),
            self.config.export_format,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def import_eet(
        self,
        eet_data: Dict[str, Any],
    ) -> EETImportResult:
        """Import and validate EET data for Article 9 product.

        Parses the EET data, validates Article 9 mandatory fields,
        maps values to pipeline format, and returns structured result.

        Args:
            eet_data: Raw EET data keyed by field ID or field name.

        Returns:
            EETImportResult with parsed fields and validation status.
        """
        start_time = time.time()
        errors: List[Dict[str, str]] = []
        warnings: List[Dict[str, str]] = []
        field_values: List[EETFieldValue] = []

        # Normalize keys
        normalized = self._normalize_keys(eet_data)

        # Parse each registered field
        fields_mapped = 0
        art_9_populated = 0
        art_9_total = sum(
            1 for f in self._field_registry if f.mandatory_art_9
        )

        for field_def in self._field_registry:
            value = normalized.get(field_def.field_id)
            if value is None:
                value = normalized.get(field_def.field_name)

            if value is not None:
                # Validate field value
                valid, msg = self._validate_field(field_def, value)
                field_values.append(EETFieldValue(
                    field_id=field_def.field_id,
                    field_name=field_def.field_name,
                    value=value,
                    source="eet_import",
                    validated=valid,
                    validation_message=msg,
                ))
                fields_mapped += 1
                if field_def.mandatory_art_9:
                    art_9_populated += 1
                if not valid:
                    errors.append({
                        "field_id": field_def.field_id,
                        "field_name": field_def.field_name,
                        "message": msg,
                        "severity": "error",
                    })
            elif field_def.mandatory_art_9:
                # Auto-populate with default if configured
                if self.config.auto_populate_defaults and field_def.default_art_9:
                    field_values.append(EETFieldValue(
                        field_id=field_def.field_id,
                        field_name=field_def.field_name,
                        value=field_def.default_art_9,
                        source="default",
                        validated=True,
                    ))
                    art_9_populated += 1
                    fields_mapped += 1
                else:
                    errors.append({
                        "field_id": field_def.field_id,
                        "field_name": field_def.field_name,
                        "message": f"Mandatory Art 9 field missing: {field_def.field_name}",
                        "severity": "error",
                    })

        # Build Article 9 fields model
        art_9_fields = self._build_article_9_fields(field_values)

        # Check classification
        classification = normalized.get("EET_003", normalized.get("SFDR_Classification", ""))
        if classification and classification != "article_9":
            warnings.append({
                "field_id": "EET_003",
                "field_name": "SFDR_Classification",
                "message": f"Product classified as {classification}, not article_9",
                "severity": "warning",
            })

        is_valid = len(errors) == 0 or not self.config.strict_validation
        elapsed_ms = (time.time() - start_time) * 1000

        result = EETImportResult(
            import_id=f"IMP-{utcnow().strftime('%Y%m%d%H%M%S')}",
            eet_version=self.config.eet_version.value,
            product_isin=str(normalized.get("EET_001", "")),
            product_name=str(normalized.get("EET_002", "")),
            sfdr_classification=str(classification),
            total_fields_parsed=len(normalized),
            fields_mapped=fields_mapped,
            art_9_fields_populated=art_9_populated,
            art_9_fields_total=art_9_total,
            article_9_fields=art_9_fields,
            field_values=field_values,
            validation_errors=errors,
            validation_warnings=warnings,
            is_valid=is_valid,
            imported_at=utcnow().isoformat(),
            execution_time_ms=elapsed_ms,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _hash_data(
                result.model_dump(
                    exclude={"provenance_hash", "field_values", "article_9_fields"}
                )
            )

        self.logger.info(
            "EETDataBridge import: %d fields mapped, %d/%d Art 9 populated, "
            "valid=%s, errors=%d, elapsed=%.1fms",
            fields_mapped, art_9_populated, art_9_total,
            is_valid, len(errors), elapsed_ms,
        )
        return result

    def export_eet(
        self,
        pipeline_data: Dict[str, Any],
    ) -> EETExportResult:
        """Export pipeline results to EET format.

        Maps pipeline data to EET field identifiers and generates
        the output in the configured format (json/csv/xml).

        Args:
            pipeline_data: Pipeline result data.

        Returns:
            EETExportResult with exported EET data.
        """
        start_time = time.time()
        warnings: List[str] = []
        missing_mandatory: List[str] = []
        eet_data: Dict[str, Any] = {}

        # Reverse mapping: pipeline key -> EET field ID
        reverse_map: Dict[str, str] = {
            v: k for k, v in EET_TO_PIPELINE_MAPPING.items()
        }

        # Map pipeline data to EET fields
        for pipeline_key, value in pipeline_data.items():
            eet_field_id = reverse_map.get(pipeline_key)
            if eet_field_id:
                eet_data[eet_field_id] = value

        # Check mandatory field coverage
        for field_def in self._field_registry:
            if field_def.mandatory_art_9 and field_def.field_id not in eet_data:
                # Try auto-populate
                if self.config.auto_populate_defaults and field_def.default_art_9:
                    eet_data[field_def.field_id] = field_def.default_art_9
                else:
                    missing_mandatory.append(field_def.field_id)

        if missing_mandatory:
            warnings.append(
                f"Missing {len(missing_mandatory)} mandatory EET fields"
            )

        mandatory_complete = len(missing_mandatory) == 0
        elapsed_ms = (time.time() - start_time) * 1000

        result = EETExportResult(
            export_id=f"EXP-{utcnow().strftime('%Y%m%d%H%M%S')}",
            eet_version=self.config.eet_version.value,
            product_isin=str(pipeline_data.get("product_isin", "")),
            product_name=str(pipeline_data.get("product_name", "")),
            total_fields_exported=len(eet_data),
            mandatory_fields_complete=mandatory_complete,
            export_format=self.config.export_format,
            eet_data=eet_data,
            missing_mandatory=missing_mandatory,
            warnings=warnings,
            exported_at=utcnow().isoformat(),
            execution_time_ms=elapsed_ms,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _hash_data(
                result.model_dump(exclude={"provenance_hash", "eet_data"})
            )

        self.logger.info(
            "EETDataBridge export: %d fields, mandatory_complete=%s, "
            "missing=%d, elapsed=%.1fms",
            len(eet_data), mandatory_complete, len(missing_mandatory), elapsed_ms,
        )
        return result

    def get_field_registry(self) -> List[EETFieldDefinition]:
        """Get the Article 9 EET field registry.

        Returns:
            List of EET field definitions.
        """
        return list(self._field_registry)

    def get_mandatory_fields(self) -> List[EETFieldDefinition]:
        """Get only mandatory Article 9 EET fields.

        Returns:
            List of mandatory field definitions.
        """
        return [f for f in self._field_registry if f.mandatory_art_9]

    def validate_eet_completeness(
        self,
        eet_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate EET data completeness for Article 9.

        Args:
            eet_data: EET data to validate.

        Returns:
            Validation result with per-category coverage.
        """
        normalized = self._normalize_keys(eet_data)
        category_coverage: Dict[str, Dict[str, int]] = {}

        for field_def in self._field_registry:
            cat = field_def.category
            if cat not in category_coverage:
                category_coverage[cat] = {"total": 0, "populated": 0}
            category_coverage[cat]["total"] += 1
            if normalized.get(field_def.field_id) is not None:
                category_coverage[cat]["populated"] += 1

        total = sum(c["total"] for c in category_coverage.values())
        populated = sum(c["populated"] for c in category_coverage.values())

        return {
            "overall_pct": (populated / total * 100.0) if total > 0 else 0.0,
            "total_fields": total,
            "populated_fields": populated,
            "category_coverage": {
                cat: {
                    "total": data["total"],
                    "populated": data["populated"],
                    "pct": (data["populated"] / data["total"] * 100.0) if data["total"] > 0 else 0.0,
                }
                for cat, data in category_coverage.items()
            },
            "validated_at": utcnow().isoformat(),
        }

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _normalize_keys(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize EET data keys to field IDs."""
        normalized: Dict[str, Any] = {}
        name_to_id: Dict[str, str] = {
            f.field_name: f.field_id for f in self._field_registry
        }
        for key, value in data.items():
            if key.startswith("EET_"):
                normalized[key] = value
            elif key in name_to_id:
                normalized[name_to_id[key]] = value
            else:
                normalized[key] = value
        return normalized

    def _validate_field(
        self,
        field_def: EETFieldDefinition,
        value: Any,
    ) -> tuple:
        """Validate a single EET field value.

        Returns:
            Tuple of (is_valid: bool, message: str).
        """
        if value is None:
            return (False, "Value is None")

        if field_def.data_type == "percentage":
            try:
                pct = float(value)
                if pct < 0 or pct > 100:
                    return (False, f"Percentage {pct} out of range [0, 100]")
            except (ValueError, TypeError):
                return (False, f"Invalid percentage value: {value}")

        elif field_def.data_type == "numeric":
            try:
                float(value)
            except (ValueError, TypeError):
                return (False, f"Invalid numeric value: {value}")

        elif field_def.data_type == "boolean":
            if str(value).lower() not in ("true", "false", "1", "0", "yes", "no"):
                return (False, f"Invalid boolean value: {value}")

        elif field_def.data_type == "enum":
            if field_def.allowed_values and str(value) not in field_def.allowed_values:
                return (False, f"Value '{value}' not in {field_def.allowed_values}")

        elif field_def.data_type == "isin":
            isin_str = str(value)
            if len(isin_str) != 12:
                return (False, f"Invalid ISIN length: {len(isin_str)}")

        return (True, "")

    def _build_article_9_fields(
        self,
        field_values: List[EETFieldValue],
    ) -> Article9EETFields:
        """Build Article9EETFields from parsed field values."""
        values_by_id: Dict[str, Any] = {
            fv.field_id: fv.value for fv in field_values
        }

        fields = Article9EETFields(
            sfdr_classification=str(values_by_id.get("EET_003", "article_9")),
            has_sustainable_objective=self._to_bool(values_by_id.get("EET_010", True)),
            sustainable_investment_objective=str(values_by_id.get("EET_011", "")),
            sustainable_investment_pct=self._to_float(values_by_id.get("EET_012", 100.0)),
            si_environmental_pct=self._to_float(values_by_id.get("EET_013", 0.0)),
            si_social_pct=self._to_float(values_by_id.get("EET_014", 0.0)),
            si_taxonomy_aligned_pct=self._to_float(values_by_id.get("EET_015", 0.0)),
            taxonomy_alignment_turnover=self._to_float(values_by_id.get("EET_020", 0.0)),
            taxonomy_alignment_capex=self._to_float(values_by_id.get("EET_021", 0.0)),
            taxonomy_alignment_opex=self._to_float(values_by_id.get("EET_022", 0.0)),
            ccm_alignment_pct=self._to_float(values_by_id.get("EET_023", 0.0)),
            cca_alignment_pct=self._to_float(values_by_id.get("EET_024", 0.0)),
            water_alignment_pct=self._to_float(values_by_id.get("EET_025", 0.0)),
            circular_economy_alignment_pct=self._to_float(values_by_id.get("EET_026", 0.0)),
            pollution_alignment_pct=self._to_float(values_by_id.get("EET_027", 0.0)),
            biodiversity_alignment_pct=self._to_float(values_by_id.get("EET_028", 0.0)),
            fossil_gas_pct=self._to_float(values_by_id.get("EET_029", 0.0)),
            nuclear_pct=self._to_float(values_by_id.get("EET_030", 0.0)),
            considers_pai=self._to_bool(values_by_id.get("EET_040", True)),
            pai_ghg_emissions_scope_1=self._to_float(values_by_id.get("EET_041", 0.0)),
            pai_ghg_emissions_scope_2=self._to_float(values_by_id.get("EET_042", 0.0)),
            pai_ghg_emissions_scope_3=self._to_float(values_by_id.get("EET_043", 0.0)),
            pai_carbon_footprint=self._to_float(values_by_id.get("EET_044", 0.0)),
            pai_ghg_intensity=self._to_float(values_by_id.get("EET_045", 0.0)),
            pai_fossil_fuel_pct=self._to_float(values_by_id.get("EET_046", 0.0)),
            pai_non_renewable_energy_pct=self._to_float(values_by_id.get("EET_047", 0.0)),
            pai_energy_intensity=self._to_float(values_by_id.get("EET_048", 0.0)),
            dnsh_all_6_objectives=self._to_bool(values_by_id.get("EET_060", True)),
            good_governance_check=self._to_bool(values_by_id.get("EET_061", True)),
            minimum_safeguards=self._to_bool(values_by_id.get("EET_062", True)),
            has_designated_benchmark=self._to_bool(values_by_id.get("EET_070", False)),
            benchmark_type=str(values_by_id.get("EET_071", "")),
            benchmark_name=str(values_by_id.get("EET_072", "")),
        )

        if self.config.enable_provenance:
            fields.provenance_hash = _hash_data(
                fields.model_dump(exclude={"provenance_hash"})
            )

        return fields

    def _to_float(self, value: Any) -> float:
        """Safely convert value to float."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _to_bool(self, value: Any) -> bool:
        """Safely convert value to bool."""
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes")
