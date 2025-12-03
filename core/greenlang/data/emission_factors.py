# -*- coding: utf-8 -*-
"""
GreenLang Emission Factor Loader

Loads and manages emission factors from multiple authoritative sources:
- DEFRA 2024 (UK Government)
- EPA eGRID 2023 (US Electricity)
- IPCC 2021 (Intergovernmental Panel on Climate Change)

Stores in PostgreSQL for fast lookup and caching.
"""

from typing import Dict, List, Optional, Any, Literal
from decimal import Decimal
from datetime import datetime, date
from pathlib import Path
import csv
import logging
from pydantic import BaseModel, Field, condecimal
from uuid import UUID, uuid4

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

logger = logging.getLogger(__name__)


# ============================================================================
# EMISSION FACTOR MODELS
# ============================================================================

class EmissionFactor(BaseModel):
    """
    Emission Factor Data Model

    Represents a single emission factor with complete metadata.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")

    # Source Information
    source: str = Field(
        ...,
        description="Source database (DEFRA, EPA, IPCC, etc.)"
    )

    source_year: int = Field(
        ...,
        description="Year of emission factor dataset"
    )

    source_version: str = Field(
        ...,
        description="Version of source dataset"
    )

    # Category & Description
    category: str = Field(
        ...,
        description="Top-level category (Fuels, Electricity, etc.)"
    )

    subcategory: Optional[str] = Field(
        None,
        description="Subcategory for detailed classification"
    )

    activity_name: str = Field(
        ...,
        description="Name of activity (e.g., 'Natural gas', 'Grid electricity')"
    )

    description: Optional[str] = Field(
        None,
        description="Detailed description"
    )

    # Geographic Scope
    geographic_scope: Literal["global", "country", "region", "grid"] = Field(
        ...,
        description="Geographic applicability"
    )

    country_code: Optional[str] = Field(
        None,
        description="ISO 3166-1 alpha-2 country code if country-specific"
    )

    region_code: Optional[str] = Field(
        None,
        description="Region/state code if region-specific"
    )

    grid_region: Optional[str] = Field(
        None,
        description="Electricity grid region (e.g., 'WECC', 'ERCOT')"
    )

    # Emission Factor Values
    co2_factor: condecimal(ge=Decimal(0), max_digits=12, decimal_places=6) = Field(
        default=Decimal(0),
        description="CO2 emission factor"
    )

    ch4_factor: condecimal(ge=Decimal(0), max_digits=12, decimal_places=9) = Field(
        default=Decimal(0),
        description="CH4 emission factor"
    )

    n2o_factor: condecimal(ge=Decimal(0), max_digits=12, decimal_places=9) = Field(
        default=Decimal(0),
        description="N2O emission factor"
    )

    co2e_factor: condecimal(ge=Decimal(0), max_digits=12, decimal_places=6) = Field(
        ...,
        description="Total CO2e emission factor (includes all GHGs)"
    )

    # Units
    unit_numerator: str = Field(
        ...,
        description="Numerator unit (kgCO2e, tCO2e, etc.)"
    )

    unit_denominator: str = Field(
        ...,
        description="Denominator unit (kWh, liter, kg, tonne, etc.)"
    )

    # GHG Protocol Scope
    ghg_scope: Optional[str] = Field(
        None,
        description="GHG Protocol scope if applicable"
    )

    # Quality & Metadata
    quality_rating: Literal["high", "medium", "low"] = Field(
        ...,
        description="Data quality rating from source"
    )

    uncertainty_percentage: Optional[condecimal(ge=Decimal(0), le=Decimal(100))] = Field(
        None,
        description="Uncertainty in emission factor (%)"
    )

    reference_url: Optional[str] = Field(
        None,
        description="URL to source documentation"
    )

    notes: Optional[str] = Field(
        None,
        description="Additional notes"
    )

    # Temporal Validity
    valid_from: date = Field(
        ...,
        description="Start date of validity"
    )

    valid_to: Optional[date] = Field(
        None,
        description="End date of validity (None = currently valid)"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "source": "DEFRA",
                "source_year": 2024,
                "source_version": "v1.0",
                "category": "Fuels",
                "subcategory": "Gaseous fuels",
                "activity_name": "Natural gas",
                "geographic_scope": "country",
                "country_code": "GB",
                "co2_factor": "0.184",
                "ch4_factor": "0.0000037",
                "n2o_factor": "0.0000004",
                "co2e_factor": "0.18443",
                "unit_numerator": "kgCO2e",
                "unit_denominator": "kWh",
                "quality_rating": "high",
                "valid_from": "2024-01-01"
            }
        }


# ============================================================================
# EMISSION FACTOR LOADER
# ============================================================================

class EmissionFactorLoader:
    """
    Loads emission factors from multiple sources into PostgreSQL.

    Supports:
    - DEFRA 2024 (UK Government GHG Conversion Factors)
    - EPA eGRID 2023 (US Electricity Grid)
    - IPCC 2021 (Global factors)
    - Custom CSV uploads
    """

    def __init__(self, db_connection_string: str):
        """
        Initialize loader.

        Args:
            db_connection_string: PostgreSQL connection string

        Raises:
            ImportError: If asyncpg is not installed
        """
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is required for EmissionFactorLoader. "
                "Install it with: pip install asyncpg"
            )
        self.db_connection_string = db_connection_string
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(self.db_connection_string)
        await self._create_tables()
        logger.info("EmissionFactorLoader initialized")

    async def _create_tables(self):
        """Create emission_factors table if not exists."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS emission_factors (
            id UUID PRIMARY KEY,
            source VARCHAR(100) NOT NULL,
            source_year INTEGER NOT NULL,
            source_version VARCHAR(50) NOT NULL,
            category VARCHAR(200) NOT NULL,
            subcategory VARCHAR(200),
            activity_name VARCHAR(500) NOT NULL,
            description TEXT,
            geographic_scope VARCHAR(50) NOT NULL,
            country_code VARCHAR(2),
            region_code VARCHAR(50),
            grid_region VARCHAR(100),
            co2_factor NUMERIC(12, 6) DEFAULT 0,
            ch4_factor NUMERIC(12, 9) DEFAULT 0,
            n2o_factor NUMERIC(12, 9) DEFAULT 0,
            co2e_factor NUMERIC(12, 6) NOT NULL,
            unit_numerator VARCHAR(50) NOT NULL,
            unit_denominator VARCHAR(50) NOT NULL,
            ghg_scope VARCHAR(50),
            quality_rating VARCHAR(20) NOT NULL,
            uncertainty_percentage NUMERIC(5, 2),
            reference_url TEXT,
            notes TEXT,
            valid_from DATE NOT NULL,
            valid_to DATE,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'::jsonb
        );

        CREATE INDEX IF NOT EXISTS idx_emission_factors_source ON emission_factors(source, source_year);
        CREATE INDEX IF NOT EXISTS idx_emission_factors_category ON emission_factors(category, subcategory);
        CREATE INDEX IF NOT EXISTS idx_emission_factors_activity ON emission_factors(activity_name);
        CREATE INDEX IF NOT EXISTS idx_emission_factors_country ON emission_factors(country_code);
        CREATE INDEX IF NOT EXISTS idx_emission_factors_grid ON emission_factors(grid_region);
        CREATE INDEX IF NOT EXISTS idx_emission_factors_validity ON emission_factors(valid_from, valid_to);
        """

        async with self.pool.acquire() as conn:
            await conn.execute(create_table_sql)
        logger.info("Emission factors table created/verified")

    async def load_defra_2024(self, csv_path: Path) -> int:
        """
        Load DEFRA 2024 GHG Conversion Factors.

        DEFRA CSV format:
        Level 1, Level 2, Level 3, Level 4, GHG, UOM, Year, kgCO2e

        Args:
            csv_path: Path to DEFRA CSV file

        Returns:
            Number of emission factors loaded
        """
        logger.info(f"Loading DEFRA 2024 factors from {csv_path}")

        if not csv_path.exists():
            raise FileNotFoundError(f"DEFRA CSV not found: {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        # Expected columns
        required_cols = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'GHG', 'UOM', 'Year', 'kgCO2e']

        # Parse and create emission factors
        emission_factors = []

        for _, row in df.iterrows():
            # Build activity name from hierarchy
            activity_parts = [
                str(row.get('Level 1', '')),
                str(row.get('Level 2', '')),
                str(row.get('Level 3', '')),
                str(row.get('Level 4', ''))
            ]
            activity_name = ' - '.join([p.strip() for p in activity_parts if p.strip() and p.strip() != 'nan'])

            # Determine which GHG
            ghg_type = str(row.get('GHG', 'CO2e')).upper()

            # Parse emission factor value
            try:
                factor_value = Decimal(str(row.get('kgCO2e', 0)))
            except:
                logger.warning(f"Invalid factor value for {activity_name}, skipping")
                continue

            # Build emission factor
            ef = EmissionFactor(
                source="DEFRA",
                source_year=int(row.get('Year', 2024)),
                source_version="2024 v1.0",
                category=str(row.get('Level 1', 'Unknown')),
                subcategory=str(row.get('Level 2', None)) if pd.notna(row.get('Level 2')) else None,
                activity_name=activity_name,
                geographic_scope="country",
                country_code="GB",
                co2_factor=factor_value if ghg_type == 'CO2' else Decimal(0),
                ch4_factor=factor_value if ghg_type == 'CH4' else Decimal(0),
                n2o_factor=factor_value if ghg_type == 'N2O' else Decimal(0),
                co2e_factor=factor_value if ghg_type == 'CO2E' else factor_value,
                unit_numerator="kgCO2e",
                unit_denominator=str(row.get('UOM', 'unit')),
                quality_rating="high",
                valid_from=date(2024, 1, 1),
                reference_url="https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024"
            )

            emission_factors.append(ef)

        # Insert into database
        count = await self._bulk_insert(emission_factors)
        logger.info(f"Loaded {count} DEFRA emission factors")
        return count

    async def load_epa_egrid_2023(self, excel_path: Path) -> int:
        """
        Load EPA eGRID 2023 Electricity Emission Factors.

        EPA eGRID provides CO2, CH4, N2O factors for US electricity grids.

        Args:
            excel_path: Path to EPA eGRID Excel file

        Returns:
            Number of emission factors loaded
        """
        logger.info(f"Loading EPA eGRID 2023 factors from {excel_path}")

        if not excel_path.exists():
            raise FileNotFoundError(f"EPA eGRID file not found: {excel_path}")

        # Read eGRID data (typically in 'PLNT' or 'NRL' sheet)
        df = pd.read_excel(excel_path, sheet_name='NRL22')  # Non-baseload rates

        emission_factors = []

        # eGRID columns: PSTATABB, SUBRGN, NRLCO2, NRLCH4, NRLN2O
        for _, row in df.iterrows():
            subregion = str(row.get('SUBRGN', ''))
            if not subregion or subregion == 'nan':
                continue

            # CO2 (lb/MWh -> kgCO2e/kWh)
            co2_lb_mwh = float(row.get('NRLCO2', 0))
            co2_kg_kwh = Decimal(co2_lb_mwh * 0.453592 / 1000)  # lb to kg, MWh to kWh

            # CH4 (lb/MWh -> kgCH4/kWh)
            ch4_lb_mwh = float(row.get('NRLCH4', 0))
            ch4_kg_kwh = Decimal(ch4_lb_mwh * 0.453592 / 1000 * 28)  # Convert to CO2e (GWP=28)

            # N2O (lb/MWh -> kgN2O/kWh)
            n2o_lb_mwh = float(row.get('NRLN2O', 0))
            n2o_kg_kwh = Decimal(n2o_lb_mwh * 0.453592 / 1000 * 265)  # Convert to CO2e (GWP=265)

            # Total CO2e
            co2e_factor = co2_kg_kwh + ch4_kg_kwh + n2o_kg_kwh

            ef = EmissionFactor(
                source="EPA_eGRID",
                source_year=2023,
                source_version="2023 v1.0",
                category="Electricity",
                subcategory="Grid electricity",
                activity_name=f"US Grid - {subregion}",
                description=f"Non-baseload emission rate for {subregion} grid region",
                geographic_scope="grid",
                country_code="US",
                grid_region=subregion,
                co2_factor=co2_kg_kwh,
                ch4_factor=Decimal(ch4_lb_mwh * 0.453592 / 1000),
                n2o_factor=Decimal(n2o_lb_mwh * 0.453592 / 1000),
                co2e_factor=co2e_factor,
                unit_numerator="kgCO2e",
                unit_denominator="kWh",
                ghg_scope="scope_2",
                quality_rating="high",
                valid_from=date(2023, 1, 1),
                reference_url="https://www.epa.gov/egrid"
            )

            emission_factors.append(ef)

        # Insert into database
        count = await self._bulk_insert(emission_factors)
        logger.info(f"Loaded {count} EPA eGRID emission factors")
        return count

    async def load_custom_csv(
        self,
        csv_path: Path,
        source_name: str,
        source_year: int,
        column_mapping: Dict[str, str]
    ) -> int:
        """
        Load emission factors from custom CSV.

        Args:
            csv_path: Path to CSV file
            source_name: Name of data source
            source_year: Year of data
            column_mapping: Map CSV columns to EmissionFactor fields

        Returns:
            Number of factors loaded
        """
        logger.info(f"Loading custom CSV from {csv_path}")

        df = pd.read_csv(csv_path)

        emission_factors = []

        for _, row in df.iterrows():
            # Build EmissionFactor from mapping
            ef_data = {
                'source': source_name,
                'source_year': source_year,
                'source_version': f"{source_year} custom",
                'quality_rating': 'medium',
                'valid_from': date(source_year, 1, 1)
            }

            # Map columns
            for csv_col, ef_field in column_mapping.items():
                if csv_col in df.columns:
                    ef_data[ef_field] = row[csv_col]

            # Set defaults
            ef_data.setdefault('geographic_scope', 'global')
            ef_data.setdefault('unit_numerator', 'kgCO2e')
            ef_data.setdefault('unit_denominator', 'unit')

            try:
                ef = EmissionFactor(**ef_data)
                emission_factors.append(ef)
            except Exception as e:
                logger.warning(f"Failed to create emission factor: {e}")
                continue

        count = await self._bulk_insert(emission_factors)
        logger.info(f"Loaded {count} custom emission factors")
        return count

    async def _bulk_insert(self, emission_factors: List[EmissionFactor]) -> int:
        """Insert emission factors in bulk."""
        if not emission_factors:
            return 0

        insert_sql = """
        INSERT INTO emission_factors (
            id, source, source_year, source_version, category, subcategory,
            activity_name, description, geographic_scope, country_code, region_code,
            grid_region, co2_factor, ch4_factor, n2o_factor, co2e_factor,
            unit_numerator, unit_denominator, ghg_scope, quality_rating,
            uncertainty_percentage, reference_url, notes, valid_from, valid_to, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                  $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26)
        ON CONFLICT (id) DO UPDATE SET
            updated_at = NOW()
        """

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for ef in emission_factors:
                    await conn.execute(
                        insert_sql,
                        ef.id, ef.source, ef.source_year, ef.source_version,
                        ef.category, ef.subcategory, ef.activity_name, ef.description,
                        ef.geographic_scope, ef.country_code, ef.region_code,
                        ef.grid_region, float(ef.co2_factor), float(ef.ch4_factor),
                        float(ef.n2o_factor), float(ef.co2e_factor), ef.unit_numerator,
                        ef.unit_denominator, ef.ghg_scope, ef.quality_rating,
                        float(ef.uncertainty_percentage) if ef.uncertainty_percentage else None,
                        ef.reference_url, ef.notes, ef.valid_from, ef.valid_to,
                        ef.metadata
                    )

        return len(emission_factors)

    async def search_factors(
        self,
        activity_name: Optional[str] = None,
        category: Optional[str] = None,
        country_code: Optional[str] = None,
        grid_region: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[EmissionFactor]:
        """
        Search emission factors.

        Args:
            activity_name: Filter by activity name (partial match)
            category: Filter by category
            country_code: Filter by country
            grid_region: Filter by grid region
            source: Filter by source
            limit: Maximum results

        Returns:
            List of matching emission factors
        """
        query = "SELECT * FROM emission_factors WHERE 1=1"
        params = []
        param_num = 1

        if activity_name:
            query += f" AND activity_name ILIKE ${param_num}"
            params.append(f"%{activity_name}%")
            param_num += 1

        if category:
            query += f" AND category = ${param_num}"
            params.append(category)
            param_num += 1

        if country_code:
            query += f" AND country_code = ${param_num}"
            params.append(country_code)
            param_num += 1

        if grid_region:
            query += f" AND grid_region = ${param_num}"
            params.append(grid_region)
            param_num += 1

        if source:
            query += f" AND source = ${param_num}"
            params.append(source)
            param_num += 1

        query += f" ORDER BY co2e_factor DESC LIMIT ${param_num}"
        params.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        factors = []
        for row in rows:
            ef_dict = dict(row)
            # Convert Decimal fields
            for field in ['co2_factor', 'ch4_factor', 'n2o_factor', 'co2e_factor', 'uncertainty_percentage']:
                if ef_dict.get(field) is not None:
                    ef_dict[field] = Decimal(str(ef_dict[field]))
            factors.append(EmissionFactor(**ef_dict))

        return factors

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
        logger.info("EmissionFactorLoader closed")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def load_defra_factors(db_connection_string: str, csv_path: Path) -> int:
    """
    Convenience function to load DEFRA factors.

    Args:
        db_connection_string: PostgreSQL connection string
        csv_path: Path to DEFRA CSV

    Returns:
        Number of factors loaded
    """
    loader = EmissionFactorLoader(db_connection_string)
    await loader.initialize()
    count = await loader.load_defra_2024(csv_path)
    await loader.close()
    return count


async def load_epa_egrid_factors(db_connection_string: str, excel_path: Path) -> int:
    """
    Convenience function to load EPA eGRID factors.

    Args:
        db_connection_string: PostgreSQL connection string
        excel_path: Path to eGRID Excel file

    Returns:
        Number of factors loaded
    """
    loader = EmissionFactorLoader(db_connection_string)
    await loader.initialize()
    count = await loader.load_epa_egrid_2023(excel_path)
    await loader.close()
    return count
