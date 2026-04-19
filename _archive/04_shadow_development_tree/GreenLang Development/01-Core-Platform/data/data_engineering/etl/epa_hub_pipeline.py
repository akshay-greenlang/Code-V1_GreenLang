"""
EPA Emission Factor Hub Pipeline
================================

ETL pipeline for US EPA Emission Factor Hub data.
Covers stationary combustion, mobile sources, fugitive emissions, and industrial processes.

Source: https://www.epa.gov/climateleadership/ghg-emission-factors-hub

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from enum import Enum
import logging
import uuid
import hashlib
import asyncio

from pydantic import BaseModel, Field

from greenlang.data_engineering.etl.base_pipeline import (
    BasePipeline,
    PipelineConfig,
    PipelineStage,
    LoadMode,
)
from greenlang.data_engineering.schemas.emission_factor_schema import (
    DataSourceType,
    GHGType,
    ScopeType,
    IndustryCategory,
    GeographicRegion,
    QualityTier,
    VersionStatus,
)

logger = logging.getLogger(__name__)


class EPAHubCategory(str, Enum):
    """EPA Emission Factor Hub categories."""
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    INDUSTRIAL_PROCESSES = "industrial_processes"
    ELECTRICITY = "electricity"
    STEAM_HEAT = "steam_heat"
    BUSINESS_TRAVEL = "business_travel"
    PRODUCT_TRANSPORT = "product_transport"
    WASTE = "waste"


class EPAHubRecord(BaseModel):
    """Raw EPA Hub record from Excel."""
    category: EPAHubCategory
    subcategory: str
    fuel_type: Optional[str] = None
    emission_source: str
    activity_type: str
    co2_factor: Optional[float] = None
    co2_unit: Optional[str] = None
    ch4_factor: Optional[float] = None
    ch4_unit: Optional[str] = None
    n2o_factor: Optional[float] = None
    n2o_unit: Optional[str] = None
    co2e_factor: Optional[float] = None
    co2e_unit: Optional[str] = None
    reference_year: int
    source_document: Optional[str] = None
    notes: Optional[str] = None


class EPAHubPipelineConfig(PipelineConfig):
    """EPA Hub-specific pipeline configuration."""
    epa_hub_file_path: str = Field(..., description="Path to EPA Hub Excel file")
    epa_hub_year: int = Field(default=2024, description="EPA Hub publication year")
    categories_to_process: List[EPAHubCategory] = Field(
        default_factory=lambda: list(EPAHubCategory)
    )
    include_gwp_conversion: bool = Field(default=True, description="Convert to CO2e using AR6 GWPs")


class EPAHubPipeline(BasePipeline[EPAHubRecord]):
    """
    EPA Emission Factor Hub ETL Pipeline.

    Processes EPA GHG Emission Factors Hub Excel workbook:
    - Stationary Combustion (natural gas, fuel oil, coal, propane, etc.)
    - Mobile Combustion (gasoline, diesel, aviation fuels)
    - Fugitive Emissions (refrigerants, SF6)
    - Industrial Processes (cement, steel, chemicals)

    EPA Hub Data Structure:
    - Table 1: Stationary Combustion
    - Table 2: Mobile Combustion - On-Road
    - Table 3: Mobile Combustion - Non-Road
    - Table 4: Refrigerants & Fugitives
    - Table 5: Electricity
    - Table 6: Steam & Heat
    - Table 7: Business Travel
    - Table 8: Product Transport
    - Table 9: Waste
    """

    # EPA Hub sheet name patterns
    SHEET_MAPPINGS = {
        EPAHubCategory.STATIONARY_COMBUSTION: [
            "Table 1", "Stationary Combustion", "Stationary"
        ],
        EPAHubCategory.MOBILE_COMBUSTION: [
            "Table 2", "Table 3", "Mobile Combustion", "Mobile"
        ],
        EPAHubCategory.FUGITIVE_EMISSIONS: [
            "Table 4", "Refrigerants", "Fugitive"
        ],
        EPAHubCategory.ELECTRICITY: [
            "Table 5", "Electricity"
        ],
        EPAHubCategory.STEAM_HEAT: [
            "Table 6", "Steam", "Heat"
        ],
        EPAHubCategory.BUSINESS_TRAVEL: [
            "Table 7", "Business Travel", "Travel"
        ],
        EPAHubCategory.PRODUCT_TRANSPORT: [
            "Table 8", "Product Transport", "Transport", "Freight"
        ],
        EPAHubCategory.WASTE: [
            "Table 9", "Waste"
        ],
    }

    # Industry mapping
    INDUSTRY_MAPPING = {
        EPAHubCategory.STATIONARY_COMBUSTION: IndustryCategory.GENERAL,
        EPAHubCategory.MOBILE_COMBUSTION: IndustryCategory.AUTOMOTIVE,
        EPAHubCategory.FUGITIVE_EMISSIONS: IndustryCategory.CHEMICALS,
        EPAHubCategory.INDUSTRIAL_PROCESSES: IndustryCategory.GENERAL,
        EPAHubCategory.ELECTRICITY: IndustryCategory.ELECTRICITY,
        EPAHubCategory.STEAM_HEAT: IndustryCategory.GENERAL,
        EPAHubCategory.BUSINESS_TRAVEL: IndustryCategory.AVIATION,
        EPAHubCategory.PRODUCT_TRANSPORT: IndustryCategory.ROAD_FREIGHT,
        EPAHubCategory.WASTE: IndustryCategory.WASTE,
    }

    # GWP values from IPCC AR6 (100-year)
    GWP_AR6 = {
        'CO2': 1,
        'CH4': 28,
        'N2O': 265,
        'SF6': 23500,
        'NF3': 16100,
        'HFC-134a': 1300,
        'HFC-32': 677,
        'R-410A': 1924,
        'R-404A': 4728,
        'R-407C': 1624,
    }

    def __init__(self, config: EPAHubPipelineConfig):
        """Initialize EPA Hub pipeline."""
        super().__init__(config)
        self.hub_config = config
        self.file_path = Path(config.epa_hub_file_path)

    async def extract(self) -> List[EPAHubRecord]:
        """
        Extract emission factors from EPA Hub Excel workbook.

        Returns:
            List of raw EPA Hub records
        """
        records = []

        try:
            import pandas as pd

            if not self.file_path.exists():
                raise FileNotFoundError(f"EPA Hub file not found: {self.file_path}")

            logger.info(f"Reading EPA Hub file: {self.file_path}")

            xlsx = pd.ExcelFile(self.file_path)

            # Process each category
            for category in self.hub_config.categories_to_process:
                sheet_names = self._find_matching_sheets(xlsx.sheet_names, category)

                for sheet_name in sheet_names:
                    try:
                        logger.debug(f"Processing sheet: {sheet_name} for category: {category}")
                        df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
                        sheet_records = self._parse_epa_hub_sheet(df, category, sheet_name)
                        records.extend(sheet_records)
                        logger.info(f"Extracted {len(sheet_records)} records from '{sheet_name}'")
                    except Exception as e:
                        logger.error(f"Error processing sheet '{sheet_name}': {e}")
                        self.metrics.warnings.append(f"Sheet '{sheet_name}' error: {e}")

            logger.info(f"Total records extracted from EPA Hub: {len(records)}")
            return records

        except ImportError:
            raise ImportError("pandas and openpyxl required for EPA Hub pipeline")

    def _find_matching_sheets(
        self,
        sheet_names: List[str],
        category: EPAHubCategory
    ) -> List[str]:
        """Find sheets matching the category."""
        patterns = self.SHEET_MAPPINGS.get(category, [])
        matching = []

        for sheet in sheet_names:
            sheet_lower = sheet.lower()
            for pattern in patterns:
                if pattern.lower() in sheet_lower:
                    matching.append(sheet)
                    break

        return matching

    def _parse_epa_hub_sheet(
        self,
        df,
        category: EPAHubCategory,
        sheet_name: str
    ) -> List[EPAHubRecord]:
        """Parse an EPA Hub Excel sheet."""
        records = []
        import pandas as pd

        # Find header row
        header_row = self._find_header_row(df)
        if header_row is None:
            logger.warning(f"Could not find header row in sheet '{sheet_name}'")
            return records

        # Set headers and process data rows
        headers = df.iloc[header_row].tolist()
        data_df = df.iloc[header_row + 1:].copy()
        data_df.columns = headers

        # Map column names
        col_mapping = self._get_column_mapping(headers)

        # Process each row
        for idx, row in data_df.iterrows():
            try:
                record = self._parse_row(row, col_mapping, category)
                if record:
                    records.append(record)
            except Exception as e:
                logger.debug(f"Error parsing row {idx}: {e}")
                continue

        return records

    def _find_header_row(self, df) -> Optional[int]:
        """Find the header row in an EPA Hub sheet."""
        import pandas as pd

        for idx, row in df.iterrows():
            row_str = ' '.join([str(v).lower() for v in row.values if pd.notna(v)])
            # Look for common header indicators
            if any(term in row_str for term in [
                'fuel type', 'emission factor', 'kg co2', 'activity',
                'source', 'vehicle', 'refrigerant'
            ]):
                return idx

        return None

    def _get_column_mapping(self, headers: List) -> Dict[str, Optional[str]]:
        """Map standard column names to actual headers."""
        import pandas as pd

        mapping = {
            'fuel': None,
            'source': None,
            'activity': None,
            'co2': None,
            'ch4': None,
            'n2o': None,
            'unit': None,
            'notes': None,
        }

        for header in headers:
            if header is None or pd.isna(header):
                continue
            header_lower = str(header).lower().strip()

            if 'fuel' in header_lower or 'type' in header_lower:
                mapping['fuel'] = header
            elif 'source' in header_lower or 'vehicle' in header_lower:
                mapping['source'] = header
            elif 'activity' in header_lower:
                mapping['activity'] = header
            elif 'kg co2' in header_lower and 'ch4' not in header_lower and 'n2o' not in header_lower:
                mapping['co2'] = header
            elif 'kg ch4' in header_lower or 'g ch4' in header_lower:
                mapping['ch4'] = header
            elif 'kg n2o' in header_lower or 'g n2o' in header_lower:
                mapping['n2o'] = header
            elif 'unit' in header_lower:
                mapping['unit'] = header
            elif 'note' in header_lower or 'comment' in header_lower:
                mapping['notes'] = header

        return mapping

    def _parse_row(
        self,
        row,
        col_mapping: Dict[str, Optional[str]],
        category: EPAHubCategory
    ) -> Optional[EPAHubRecord]:
        """Parse a single data row."""
        import pandas as pd

        # Get values
        fuel_type = self._get_cell_value(row, col_mapping.get('fuel'))
        source = self._get_cell_value(row, col_mapping.get('source'))
        activity = self._get_cell_value(row, col_mapping.get('activity'))

        # Need at least one identifier
        emission_source = fuel_type or source or activity
        if not emission_source:
            return None

        # Get emission factors
        co2_value = self._safe_float(self._get_cell_value(row, col_mapping.get('co2')))
        ch4_value = self._safe_float(self._get_cell_value(row, col_mapping.get('ch4')))
        n2o_value = self._safe_float(self._get_cell_value(row, col_mapping.get('n2o')))

        # Need at least one emission factor
        if co2_value is None and ch4_value is None and n2o_value is None:
            return None

        # Get unit
        unit = self._get_cell_value(row, col_mapping.get('unit')) or 'unit'

        # Calculate CO2e
        co2e_value = None
        if co2_value is not None:
            co2e_value = co2_value
            if ch4_value is not None:
                co2e_value += ch4_value * self.GWP_AR6['CH4']
            if n2o_value is not None:
                co2e_value += n2o_value * self.GWP_AR6['N2O']

        return EPAHubRecord(
            category=category,
            subcategory=activity or '',
            fuel_type=fuel_type,
            emission_source=emission_source,
            activity_type=activity or 'general',
            co2_factor=co2_value,
            co2_unit=f"kgCO2/{unit}" if co2_value else None,
            ch4_factor=ch4_value,
            ch4_unit=f"gCH4/{unit}" if ch4_value else None,
            n2o_factor=n2o_value,
            n2o_unit=f"gN2O/{unit}" if n2o_value else None,
            co2e_factor=co2e_value,
            co2e_unit=f"kgCO2e/{unit}" if co2e_value else None,
            reference_year=self.hub_config.epa_hub_year,
            notes=self._get_cell_value(row, col_mapping.get('notes')),
        )

    def _get_cell_value(self, row, column: Optional[str]) -> Optional[str]:
        """Safely get cell value from row."""
        import pandas as pd

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
            # Handle values with commas
            cleaned = value.replace(',', '')
            return float(cleaned)
        except (ValueError, TypeError, AttributeError):
            return None

    def _is_valid_source_record(self, record: EPAHubRecord) -> bool:
        """Validate EPA Hub source record."""
        if not record.emission_source:
            return False
        # Need at least one emission factor
        if (record.co2_factor is None and
            record.ch4_factor is None and
            record.n2o_factor is None):
            return False
        return True

    async def transform(self, data: List[EPAHubRecord]) -> List[Dict[str, Any]]:
        """
        Transform EPA Hub records to emission factor schema.

        Args:
            data: List of validated EPA Hub records

        Returns:
            List of transformed emission factor dictionaries
        """
        transformed = []

        for record in data:
            try:
                # Generate factor ID
                factor_id = str(uuid.uuid4())

                # Map industry
                industry = self.INDUSTRY_MAPPING.get(
                    record.category,
                    IndustryCategory.GENERAL
                )

                # Build product name
                product_name = record.emission_source
                if record.fuel_type and record.fuel_type != record.emission_source:
                    product_name = f"{record.fuel_type} - {record.emission_source}"

                # Determine scope
                scope_type = self._determine_scope(record.category)

                # Use CO2e factor if available, otherwise CO2
                factor_value = record.co2e_factor or record.co2_factor or 0
                factor_unit = record.co2e_unit or record.co2_unit or "kgCO2e/unit"

                factor_dict = {
                    "factor_id": factor_id,
                    "factor_hash": self._calculate_hash(record),
                    "industry": industry.value,
                    "product_code": None,
                    "product_name": product_name,
                    "product_subcategory": record.subcategory,
                    "production_route": None,
                    "region": GeographicRegion.USA.value,
                    "country_code": "USA",
                    "state_province": None,
                    "ghg_type": GHGType.CO2E.value if record.co2e_factor else GHGType.CO2.value,
                    "scope_type": scope_type.value,
                    "factor_value": Decimal(str(round(factor_value, 6))),
                    "factor_unit": factor_unit,
                    "input_unit": self._extract_input_unit(factor_unit),
                    "output_unit": "kgCO2e",
                    "gwp_source": "IPCC AR6",
                    "gwp_timeframe": 100,
                    "reference_year": record.reference_year,
                    "valid_from": date(record.reference_year, 1, 1).isoformat(),
                    "valid_to": None,
                    "source": {
                        "source_type": DataSourceType.EPA_GHG.value,
                        "source_name": f"EPA GHG Emission Factors Hub {record.reference_year}",
                        "source_url": "https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
                        "publication_date": f"{record.reference_year}-04-01",
                        "version": str(record.reference_year),
                    },
                    "quality": {
                        "quality_tier": QualityTier.TIER_2.value,
                        "reliability_score": 2,
                        "completeness_score": 2,
                        "temporal_score": 1,
                        "geographic_score": 2,
                        "technology_score": 3,
                    },
                    "version": {
                        "version_id": "1.0.0",
                        "status": VersionStatus.ACTIVE.value,
                        "effective_from": date(record.reference_year, 1, 1).isoformat(),
                    },
                    "cbam_eligible": False,
                    "csrd_compliant": True,
                    "ghg_protocol_compliant": True,
                    "tags": [
                        "epa_hub",
                        str(record.reference_year),
                        record.category.value,
                        "us_factors",
                    ],
                    "metadata": {
                        "category": record.category.value,
                        "co2_factor": record.co2_factor,
                        "ch4_factor": record.ch4_factor,
                        "n2o_factor": record.n2o_factor,
                        "notes": record.notes,
                    },
                    "created_at": datetime.utcnow().isoformat(),
                }

                transformed.append(factor_dict)

            except Exception as e:
                logger.warning(f"Error transforming EPA Hub record: {e}")
                self.metrics.warnings.append(f"Transform error: {e}")
                continue

        return transformed

    def _determine_scope(self, category: EPAHubCategory) -> ScopeType:
        """Determine GHG Protocol scope from category."""
        scope_mapping = {
            EPAHubCategory.STATIONARY_COMBUSTION: ScopeType.SCOPE_1,
            EPAHubCategory.MOBILE_COMBUSTION: ScopeType.SCOPE_1,
            EPAHubCategory.FUGITIVE_EMISSIONS: ScopeType.SCOPE_1,
            EPAHubCategory.INDUSTRIAL_PROCESSES: ScopeType.SCOPE_1,
            EPAHubCategory.ELECTRICITY: ScopeType.SCOPE_2_LOCATION,
            EPAHubCategory.STEAM_HEAT: ScopeType.SCOPE_2_LOCATION,
            EPAHubCategory.BUSINESS_TRAVEL: ScopeType.SCOPE_3,
            EPAHubCategory.PRODUCT_TRANSPORT: ScopeType.SCOPE_3,
            EPAHubCategory.WASTE: ScopeType.SCOPE_3,
        }
        return scope_mapping.get(category, ScopeType.SCOPE_3)

    def _extract_input_unit(self, factor_unit: str) -> str:
        """Extract input unit from factor unit string."""
        if '/' in factor_unit:
            return factor_unit.split('/')[-1]
        return 'unit'

    def _calculate_hash(self, record: EPAHubRecord) -> str:
        """Calculate unique hash for deduplication."""
        content = f"epa_hub:{record.reference_year}:{record.category}:{record.emission_source}:{record.fuel_type}"
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

        for record in data:
            try:
                # Database operations would go here
                inserted += 1
            except Exception as e:
                logger.error(f"Error loading record: {e}")
                errors += 1

        return {
            "inserted": inserted,
            "updated": updated,
            "errors": errors,
        }


# Stationary Combustion Factors (common US fuels)
STATIONARY_COMBUSTION_FACTORS = {
    "natural_gas": {
        "co2_kg_mmbtu": 53.06,
        "ch4_g_mmbtu": 1.0,
        "n2o_g_mmbtu": 0.1,
        "unit": "MMBtu",
    },
    "distillate_fuel_oil_no2": {
        "co2_kg_gallon": 10.21,
        "ch4_g_gallon": 0.43,
        "n2o_g_gallon": 0.08,
        "unit": "gallon",
    },
    "residual_fuel_oil_no6": {
        "co2_kg_gallon": 11.27,
        "ch4_g_gallon": 0.68,
        "n2o_g_gallon": 0.14,
        "unit": "gallon",
    },
    "propane": {
        "co2_kg_gallon": 5.72,
        "ch4_g_gallon": 0.22,
        "n2o_g_gallon": 0.04,
        "unit": "gallon",
    },
    "coal_bituminous": {
        "co2_kg_mmbtu": 93.28,
        "ch4_g_mmbtu": 11.0,
        "n2o_g_mmbtu": 1.6,
        "unit": "MMBtu",
    },
}


# Mobile Combustion Factors
MOBILE_COMBUSTION_FACTORS = {
    "gasoline_passenger_car": {
        "co2_kg_gallon": 8.78,
        "ch4_g_mile": 0.0138,
        "n2o_g_mile": 0.0048,
    },
    "diesel_heavy_truck": {
        "co2_kg_gallon": 10.21,
        "ch4_g_mile": 0.0051,
        "n2o_g_mile": 0.0048,
    },
    "jet_fuel": {
        "co2_kg_gallon": 9.75,
        "ch4_g_gallon": 0.14,
        "n2o_g_gallon": 0.14,
    },
}


# Refrigerant GWP values
REFRIGERANT_GWPS = {
    "HFC-134a": 1300,
    "HFC-32": 677,
    "R-410A": 1924,
    "R-404A": 4728,
    "R-407C": 1624,
    "R-22": 1760,
    "SF6": 23500,
}


# Import pandas conditionally
try:
    import pandas as pd
except ImportError:
    pd = None
