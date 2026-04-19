"""
DEFRA Emission Factor Pipeline
==============================

ETL pipeline for UK Government Conversion Factors (DEFRA).
Handles Excel parsing, data transformation, and loading.

Source: https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
import logging
import uuid
import re
import hashlib

from pydantic import BaseModel, Field

from greenlang.data_engineering.etl.base_pipeline import (
    BasePipeline,
    PipelineConfig,
    PipelineStage,
    LoadMode,
)
from greenlang.data_engineering.schemas.emission_factor_schema import (
    EmissionFactorSchema,
    EmissionFactorSource,
    EmissionFactorQuality,
    EmissionFactorVersion,
    DataSourceType,
    GHGType,
    ScopeType,
    IndustryCategory,
    GeographicRegion,
    QualityTier,
    VersionStatus,
)

logger = logging.getLogger(__name__)


class DEFRARecord(BaseModel):
    """Raw DEFRA record from Excel."""
    sheet_name: str
    activity: str
    fuel_type: Optional[str] = None
    unit: str
    kg_co2: Optional[float] = None
    kg_ch4: Optional[float] = None
    kg_n2o: Optional[float] = None
    kg_co2e: float
    scope: str
    year: int
    category: Optional[str] = None
    subcategory: Optional[str] = None
    uncertainty_percent: Optional[float] = None


class DEFRAPipelineConfig(PipelineConfig):
    """DEFRA-specific pipeline configuration."""
    defra_file_path: str = Field(..., description="Path to DEFRA Excel file")
    defra_year: int = Field(default=2024, description="DEFRA publication year")
    sheets_to_process: List[str] = Field(
        default_factory=lambda: [
            "Fuels",
            "UK electricity",
            "Heat and steam",
            "UK & overseas elec",
            "WTT- fuels",
            "WTT- UK & overseas elec",
            "Transmission and distribution",
            "Business travel- air",
            "Business travel- sea",
            "Business travel- land",
            "Freighting goods",
            "Managed assets- vehicles",
            "Material use",
            "Waste disposal",
            "Water supply",
            "Water treatment",
        ]
    )


class DEFRAPipeline(BasePipeline[DEFRARecord]):
    """
    DEFRA Emission Factor ETL Pipeline.

    Processes UK Government Conversion Factors Excel workbook:
    - Extracts factors from multiple sheets (Scope 1, 2, 3, WTT)
    - Handles merged cells and complex formatting
    - Transforms to standard emission factor schema
    - Loads to database with quality scoring

    DEFRA Categories Covered:
    - Fuels (natural gas, LPG, coal, etc.)
    - Electricity (UK grid, overseas)
    - Business travel (air, road, rail, sea)
    - Freight transport
    - Waste disposal
    - Water supply/treatment
    - Materials
    """

    def __init__(self, config: DEFRAPipelineConfig):
        """Initialize DEFRA pipeline."""
        super().__init__(config)
        self.defra_config = config
        self.file_path = Path(config.defra_file_path)

        # DEFRA scope mappings
        self.scope_mapping = {
            "Scope 1": ScopeType.SCOPE_1,
            "Scope 2": ScopeType.SCOPE_2_LOCATION,
            "Scope 2 (Location-based)": ScopeType.SCOPE_2_LOCATION,
            "Scope 2 (Market-based)": ScopeType.SCOPE_2_MARKET,
            "Scope 3": ScopeType.SCOPE_3,
            "WTT": ScopeType.WTT,
            "Well-to-tank": ScopeType.WTT,
        }

        # Industry category mappings
        self.industry_mapping = {
            "Fuels": IndustryCategory.GENERAL,
            "UK electricity": IndustryCategory.ELECTRICITY,
            "Heat and steam": IndustryCategory.GENERAL,
            "Business travel- air": IndustryCategory.AVIATION,
            "Business travel- sea": IndustryCategory.SHIPPING,
            "Business travel- land": IndustryCategory.AUTOMOTIVE,
            "Freighting goods": IndustryCategory.ROAD_FREIGHT,
            "Waste disposal": IndustryCategory.WASTE,
            "Material use": IndustryCategory.GENERAL,
            "Water supply": IndustryCategory.GENERAL,
            "Water treatment": IndustryCategory.WASTE,
        }

    async def extract(self) -> List[DEFRARecord]:
        """
        Extract emission factors from DEFRA Excel workbook.

        Returns:
            List of raw DEFRA records
        """
        records = []

        try:
            import pandas as pd

            if not self.file_path.exists():
                raise FileNotFoundError(f"DEFRA file not found: {self.file_path}")

            logger.info(f"Reading DEFRA file: {self.file_path}")

            # Read Excel file
            xlsx = pd.ExcelFile(self.file_path)

            for sheet_name in self.defra_config.sheets_to_process:
                if sheet_name not in xlsx.sheet_names:
                    logger.warning(f"Sheet '{sheet_name}' not found in DEFRA file")
                    continue

                logger.debug(f"Processing sheet: {sheet_name}")

                try:
                    # Read sheet with header row detection
                    df = pd.read_excel(
                        xlsx,
                        sheet_name=sheet_name,
                        header=None,  # We'll find header ourselves
                    )

                    # Parse sheet and extract records
                    sheet_records = self._parse_defra_sheet(df, sheet_name)
                    records.extend(sheet_records)
                    logger.info(f"Extracted {len(sheet_records)} records from '{sheet_name}'")

                except Exception as e:
                    logger.error(f"Error processing sheet '{sheet_name}': {e}")
                    self.metrics.warnings.append(f"Sheet '{sheet_name}' processing error: {e}")

            logger.info(f"Total records extracted from DEFRA: {len(records)}")
            return records

        except ImportError:
            raise ImportError("pandas and openpyxl required for DEFRA pipeline")

    def _parse_defra_sheet(self, df, sheet_name: str) -> List[DEFRARecord]:
        """
        Parse a DEFRA Excel sheet.

        DEFRA sheets typically have:
        - Header rows with category info
        - Activity column
        - Fuel/type column
        - Unit column
        - kg CO2e per unit column
        - Sometimes individual GHG columns (CO2, CH4, N2O)
        """
        records = []

        # Find header row (look for "Activity" or "Fuel" columns)
        header_row = None
        for idx, row in df.iterrows():
            row_str = ' '.join([str(v).lower() for v in row.values if pd.notna(v)])
            if 'activity' in row_str or 'fuel' in row_str or 'kg co2e' in row_str:
                header_row = idx
                break

        if header_row is None:
            logger.warning(f"Could not find header row in sheet '{sheet_name}'")
            return records

        # Set headers and process data rows
        headers = df.iloc[header_row].tolist()
        data_df = df.iloc[header_row + 1:].copy()
        data_df.columns = headers

        # Map common column names
        col_mapping = self._get_column_mapping(headers)

        if not col_mapping.get('activity') or not col_mapping.get('co2e'):
            logger.warning(f"Missing required columns in sheet '{sheet_name}'")
            return records

        # Determine scope from sheet name
        scope = self._determine_scope(sheet_name)

        # Process each row
        for idx, row in data_df.iterrows():
            try:
                activity = self._get_cell_value(row, col_mapping.get('activity'))
                if not activity or pd.isna(activity):
                    continue

                co2e_value = self._get_cell_value(row, col_mapping.get('co2e'))
                if co2e_value is None or pd.isna(co2e_value):
                    continue

                # Convert to float
                try:
                    co2e_float = float(co2e_value)
                except (ValueError, TypeError):
                    continue

                if co2e_float <= 0:
                    continue

                # Create record
                record = DEFRARecord(
                    sheet_name=sheet_name,
                    activity=str(activity).strip(),
                    fuel_type=self._get_cell_value(row, col_mapping.get('fuel')),
                    unit=self._get_cell_value(row, col_mapping.get('unit')) or 'unit',
                    kg_co2=self._safe_float(self._get_cell_value(row, col_mapping.get('co2'))),
                    kg_ch4=self._safe_float(self._get_cell_value(row, col_mapping.get('ch4'))),
                    kg_n2o=self._safe_float(self._get_cell_value(row, col_mapping.get('n2o'))),
                    kg_co2e=co2e_float,
                    scope=scope,
                    year=self.defra_config.defra_year,
                    category=self._extract_category(sheet_name),
                    subcategory=self._get_cell_value(row, col_mapping.get('subcategory')),
                )
                records.append(record)

            except Exception as e:
                logger.debug(f"Error parsing row {idx}: {e}")
                continue

        return records

    def _get_column_mapping(self, headers: List) -> Dict[str, Optional[str]]:
        """Map standard column names to actual headers."""
        mapping = {
            'activity': None,
            'fuel': None,
            'unit': None,
            'co2': None,
            'ch4': None,
            'n2o': None,
            'co2e': None,
            'subcategory': None,
        }

        for header in headers:
            if header is None or pd.isna(header):
                continue
            header_lower = str(header).lower().strip()

            if 'activity' in header_lower:
                mapping['activity'] = header
            elif header_lower in ['fuel', 'fuel type', 'type']:
                mapping['fuel'] = header
            elif 'unit' in header_lower:
                mapping['unit'] = header
            elif 'kg co2e' in header_lower or 'co2e' in header_lower:
                mapping['co2e'] = header
            elif 'kg co2' in header_lower and 'co2e' not in header_lower:
                mapping['co2'] = header
            elif 'kg ch4' in header_lower:
                mapping['ch4'] = header
            elif 'kg n2o' in header_lower:
                mapping['n2o'] = header

        return mapping

    def _get_cell_value(self, row, column: Optional[str]) -> Optional[str]:
        """Safely get cell value from row."""
        if column is None:
            return None
        try:
            value = row.get(column)
            if pd.isna(value):
                return None
            return str(value).strip()
        except:
            return None

    def _safe_float(self, value: Optional[str]) -> Optional[float]:
        """Safely convert to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _determine_scope(self, sheet_name: str) -> str:
        """Determine emission scope from sheet name."""
        sheet_lower = sheet_name.lower()

        if 'wtt' in sheet_lower:
            return "WTT"
        elif 'electricity' in sheet_lower:
            return "Scope 2"
        elif 'fuel' in sheet_lower:
            return "Scope 1"
        elif 'travel' in sheet_lower or 'freight' in sheet_lower:
            return "Scope 3"
        else:
            return "Scope 3"

    def _extract_category(self, sheet_name: str) -> str:
        """Extract category from sheet name."""
        # Remove common prefixes/suffixes
        category = sheet_name
        for prefix in ['WTT-', 'WTT- ', 'UK & overseas ']:
            if category.startswith(prefix):
                category = category[len(prefix):]
        return category.strip()

    def _is_valid_source_record(self, record: DEFRARecord) -> bool:
        """Validate DEFRA source record."""
        if not record.activity:
            return False
        if record.kg_co2e is None or record.kg_co2e <= 0:
            return False
        return True

    async def transform(self, data: List[DEFRARecord]) -> List[Dict[str, Any]]:
        """
        Transform DEFRA records to emission factor schema.

        Args:
            data: List of validated DEFRA records

        Returns:
            List of transformed emission factor dictionaries
        """
        transformed = []

        for record in data:
            try:
                # Generate factor ID
                factor_id = str(uuid.uuid4())

                # Map scope
                scope_type = self.scope_mapping.get(record.scope, ScopeType.SCOPE_3)

                # Map industry
                industry = self.industry_mapping.get(
                    record.sheet_name,
                    IndustryCategory.GENERAL
                )

                # Build product name
                product_name = record.activity
                if record.fuel_type:
                    product_name = f"{record.activity} - {record.fuel_type}"

                # Create emission factor
                factor_dict = {
                    "factor_id": factor_id,
                    "factor_hash": self._calculate_hash(record),
                    "industry": industry.value,
                    "product_code": None,  # DEFRA doesn't use standard codes
                    "product_name": product_name,
                    "product_subcategory": record.subcategory,
                    "production_route": None,
                    "region": GeographicRegion.UK.value,
                    "country_code": "GBR",
                    "state_province": None,
                    "ghg_type": GHGType.CO2E.value,
                    "scope_type": scope_type.value,
                    "factor_value": Decimal(str(record.kg_co2e)),
                    "factor_unit": f"kgCO2e/{record.unit}",
                    "input_unit": record.unit,
                    "output_unit": "kgCO2e",
                    "gwp_source": "IPCC AR6",
                    "gwp_timeframe": 100,
                    "reference_year": record.year,
                    "valid_from": date(record.year, 1, 1).isoformat(),
                    "valid_to": None,
                    "source": {
                        "source_type": DataSourceType.DEFRA.value,
                        "source_name": f"UK Government GHG Conversion Factors {record.year}",
                        "source_url": "https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
                        "publication_date": f"{record.year}-06-01",
                        "version": str(record.year),
                    },
                    "quality": {
                        "quality_tier": QualityTier.TIER_2.value,
                        "reliability_score": 2,  # Verified data from government
                        "completeness_score": 2,
                        "temporal_score": 1,  # Current year
                        "geographic_score": 2,  # UK-specific
                        "technology_score": 3,  # Generic technology
                    },
                    "version": {
                        "version_id": "1.0.0",
                        "status": VersionStatus.ACTIVE.value,
                        "effective_from": date(record.year, 1, 1).isoformat(),
                    },
                    "cbam_eligible": False,
                    "csrd_compliant": True,
                    "ghg_protocol_compliant": True,
                    "tags": [
                        "defra",
                        str(record.year),
                        record.sheet_name.lower().replace(" ", "_"),
                    ],
                    "created_at": datetime.utcnow().isoformat(),
                }

                # Add individual GHG values if available
                if record.kg_co2 or record.kg_ch4 or record.kg_n2o:
                    factor_dict["metadata"] = {
                        "kg_co2": record.kg_co2,
                        "kg_ch4": record.kg_ch4,
                        "kg_n2o": record.kg_n2o,
                    }

                transformed.append(factor_dict)

            except Exception as e:
                logger.warning(f"Error transforming record: {e}")
                self.metrics.warnings.append(f"Transform error: {e}")
                continue

        return transformed

    def _calculate_hash(self, record: DEFRARecord) -> str:
        """Calculate unique hash for deduplication."""
        content = f"defra:{record.year}:{record.sheet_name}:{record.activity}:{record.fuel_type}:{record.unit}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def load_production(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Load transformed data to production database.

        Args:
            data: Transformed emission factor records

        Returns:
            Load statistics
        """
        inserted = 0
        updated = 0
        errors = 0

        # This would be replaced with actual database operations
        # For now, we'll simulate the load process
        for record in data:
            try:
                # Check if record exists (by hash)
                # If exists, update; otherwise insert
                # Simulated logic:
                inserted += 1

            except Exception as e:
                logger.error(f"Error loading record: {e}")
                errors += 1

        return {
            "inserted": inserted,
            "updated": updated,
            "errors": errors,
        }


# Import pandas conditionally
try:
    import pandas as pd
except ImportError:
    pd = None
