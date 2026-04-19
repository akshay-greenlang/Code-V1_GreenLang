"""
Emission Factor Library

Manages loading and retrieval of CBAM default emission factors.
Supports versioning and provenance tracking for audit purposes.
"""

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Optional

from cbam_pack.errors import MissingEmissionFactorError, ExpiredEmissionFactorError


@dataclass
class EmissionFactor:
    """An emission factor with full provenance."""
    factor_id: str
    product_type: str  # "steel" or "aluminum"
    country: str
    country_name: str
    direct_emissions_factor: Decimal
    indirect_emissions_factor: Decimal
    unit: str
    source: str
    version: str
    effective_date: date
    expiry_date: date

    def is_valid(self, reference_date: Optional[date] = None) -> bool:
        """Check if the factor is valid for a given date."""
        check_date = reference_date or date.today()
        return self.effective_date <= check_date <= self.expiry_date


class EmissionFactorLibrary:
    """
    Library for managing CBAM emission factors.

    Loads factors from JSON files and provides lookup by CN code and country.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the factor library.

        Args:
            data_dir: Directory containing factor JSON files.
                     Defaults to package data directory.
        """
        if data_dir is None:
            # Default to package data directory
            data_dir = Path(__file__).parent.parent.parent.parent / "data" / "emission_factors"

        self.data_dir = data_dir
        self._factors: dict = {}
        self._metadata: dict = {}
        self._loaded = False

    def load(self) -> None:
        """Load all factor files from the data directory."""
        factor_file = self.data_dir / "cbam_defaults_2024.json"

        if not factor_file.exists():
            raise FileNotFoundError(f"Factor file not found: {factor_file}")

        with open(factor_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._metadata = data.get("metadata", {})
        self._factors = data.get("factors", {})
        self._loaded = True

    def _ensure_loaded(self) -> None:
        """Ensure factors are loaded."""
        if not self._loaded:
            self.load()

    @property
    def version(self) -> str:
        """Get the factor library version."""
        self._ensure_loaded()
        return self._metadata.get("version", "unknown")

    @property
    def source(self) -> str:
        """Get the factor library source."""
        self._ensure_loaded()
        return self._metadata.get("source", "unknown")

    def get_product_type(self, cn_code: str) -> str:
        """
        Determine product type from CN code.

        Args:
            cn_code: 8-digit CN code

        Returns:
            "steel" or "aluminum"

        Raises:
            ValueError: If CN code is not supported
        """
        prefix = cn_code[:2]
        if prefix in ("72", "73"):
            return "steel"
        elif prefix == "76":
            return "aluminum"
        else:
            raise ValueError(f"Unsupported CN code prefix: {prefix}")

    def get_factor(
        self,
        cn_code: str,
        country: str,
        reference_date: Optional[date] = None,
    ) -> EmissionFactor:
        """
        Get emission factor for a CN code and country.

        Args:
            cn_code: 8-digit CN code
            country: ISO 3166-1 alpha-2 country code
            reference_date: Date to check factor validity (defaults to today)

        Returns:
            EmissionFactor with all provenance information

        Raises:
            MissingEmissionFactorError: If no factor found
            ExpiredEmissionFactorError: If factor has expired
        """
        self._ensure_loaded()

        try:
            product_type = self.get_product_type(cn_code)
        except ValueError:
            raise MissingEmissionFactorError(cn_code, country)

        product_factors = self._factors.get(product_type, {})
        country_factors = product_factors.get("by_country", {})

        # Look up country-specific factor, fall back to DEFAULT
        country_upper = country.upper()
        factor_data = country_factors.get(country_upper)

        if factor_data is None:
            factor_data = country_factors.get("DEFAULT")
            if factor_data is None:
                raise MissingEmissionFactorError(cn_code, country)
            country_upper = "DEFAULT"

        # Build factor ID
        factor_id = f"EF-{product_type.upper()}-{country_upper}-{self.version}"

        # Parse dates
        effective_date = date.fromisoformat(self._metadata.get("effective_date", "2024-01-01"))
        expiry_date = date.fromisoformat(self._metadata.get("expiry_date", "2025-12-31"))

        factor = EmissionFactor(
            factor_id=factor_id,
            product_type=product_type,
            country=country_upper,
            country_name=factor_data.get("country_name", country_upper),
            direct_emissions_factor=Decimal(str(factor_data["direct_emissions_factor"])),
            indirect_emissions_factor=Decimal(str(factor_data["indirect_emissions_factor"])),
            unit=factor_data.get("unit", "tCO2e/tonne"),
            source=self.source,
            version=self.version,
            effective_date=effective_date,
            expiry_date=expiry_date,
        )

        # Check validity
        if not factor.is_valid(reference_date):
            raise ExpiredEmissionFactorError(
                factor_id=factor.factor_id,
                expiry_date=factor.expiry_date.isoformat(),
            )

        return factor

    def list_supported_countries(self, product_type: str) -> list[str]:
        """
        List countries with specific factors for a product type.

        Args:
            product_type: "steel" or "aluminum"

        Returns:
            List of country codes (excluding DEFAULT)
        """
        self._ensure_loaded()

        product_factors = self._factors.get(product_type, {})
        country_factors = product_factors.get("by_country", {})

        return [c for c in country_factors.keys() if c != "DEFAULT"]
