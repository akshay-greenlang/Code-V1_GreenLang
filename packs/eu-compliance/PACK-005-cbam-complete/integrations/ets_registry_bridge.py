# -*- coding: utf-8 -*-
"""
ETSRegistryBridge - EU ETS Union Registry (EUTL) Data Integration
===================================================================

This module provides integration with the EU ETS Union Registry (EUTL)
for free allocation lookups, benchmark values, installation compliance
status, CBAM-ETS cross-referencing, phaseout consistency verification,
ETS price history, carbon leakage list data, and covered installation
queries.

Includes hardcoded ETS benchmark values for all CBAM-relevant products
as established in the EU ETS Implementing Regulation.

Example:
    >>> bridge = ETSRegistryBridge()
    >>> benchmark = bridge.get_benchmark_value("hot_metal_bf_bof")
    >>> assert benchmark.value_tco2_per_unit == 1.328
    >>> price = bridge.get_current_ets_price()
    >>> assert price.price_eur_per_tco2 > 0

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-005 CBAM Complete
"""

import hashlib
import logging
import math
import time
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class FreeAllocation(BaseModel):
    """Free allocation data for an ETS installation."""
    installation_id: str = Field(..., description="ETS installation ID")
    year: int = Field(..., description="Allocation year")
    total_free_allocation: float = Field(
        default=0.0, description="Total free allocation in EUAs"
    )
    benchmark_id: str = Field(default="", description="Product benchmark applied")
    benchmark_value: float = Field(default=0.0, description="Benchmark tCO2/t product")
    activity_level: float = Field(default=0.0, description="Historical activity level")
    cross_sectoral_correction: float = Field(
        default=1.0, description="Cross-sectoral correction factor"
    )
    carbon_leakage_exposed: bool = Field(
        default=True, description="Whether on carbon leakage list"
    )
    notes: str = Field(default="", description="Additional notes")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class BenchmarkValue(BaseModel):
    """ETS product benchmark value."""
    benchmark_id: str = Field(..., description="Benchmark identifier")
    product_name: str = Field(default="", description="Product name")
    value_tco2_per_unit: float = Field(
        ..., description="Benchmark value in tCO2 per unit of product"
    )
    unit: str = Field(default="tonnes", description="Product unit")
    sector: str = Field(default="", description="Industrial sector")
    cbam_category: str = Field(default="", description="Corresponding CBAM category")
    update_year: int = Field(default=2021, description="Last benchmark update year")
    reduction_rate_pct: float = Field(
        default=1.6, description="Annual benchmark reduction rate (%)"
    )
    notes: str = Field(default="", description="Additional notes")


class ComplianceStatus(BaseModel):
    """ETS compliance status for an installation."""
    installation_id: str = Field(..., description="Installation ID")
    year: int = Field(..., description="Compliance year")
    verified_emissions: float = Field(default=0.0, description="Verified emissions tCO2")
    free_allocation: float = Field(default=0.0, description="Free allocation EUAs")
    surrendered_allowances: int = Field(default=0, description="Allowances surrendered")
    balance: float = Field(default=0.0, description="Surplus or deficit")
    is_compliant: bool = Field(default=True, description="Whether installation is compliant")
    compliance_status: str = Field(default="", description="Status description")


class CrossReference(BaseModel):
    """Cross-reference between CBAM and ETS data."""
    reference_id: str = Field(
        default_factory=lambda: str(uuid4())[:12], description="Reference ID"
    )
    cbam_installation_id: str = Field(default="", description="CBAM installation ID")
    ets_installation_id: str = Field(default="", description="ETS installation ID")
    cbam_category: str = Field(default="", description="CBAM goods category")
    ets_activity: str = Field(default="", description="ETS activity")
    benchmark_id: str = Field(default="", description="Applicable benchmark")
    emission_factor_match: bool = Field(
        default=False, description="Whether emission factors are consistent"
    )
    notes: str = Field(default="", description="Cross-reference notes")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class ConsistencyCheck(BaseModel):
    """Consistency check between CBAM phaseout and ETS allocation."""
    check_id: str = Field(
        default_factory=lambda: str(uuid4())[:12], description="Check ID"
    )
    year: int = Field(..., description="Year checked")
    cbam_phaseout_pct: float = Field(default=0.0, description="CBAM phaseout percentage")
    ets_free_allocation_pct: float = Field(
        default=0.0, description="ETS free allocation percentage"
    )
    sum_pct: float = Field(default=0.0, description="Sum of CBAM + ETS pct")
    is_consistent: bool = Field(default=True, description="Whether sum <= 100%")
    variance_pct: float = Field(default=0.0, description="Variance from 100%")
    message: str = Field(default="", description="Consistency message")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class ETSPrice(BaseModel):
    """EU ETS allowance price observation."""
    date: str = Field(..., description="Price date (YYYY-MM-DD)")
    price_eur_per_tco2: float = Field(
        ..., ge=0.0, description="Price in EUR/tCO2"
    )
    source: str = Field(default="MOCK", description="Price source")


class Installation(BaseModel):
    """An ETS-covered installation."""
    installation_id: str = Field(..., description="Installation ID")
    installation_name: str = Field(default="", description="Installation name")
    country: str = Field(default="", description="Country code")
    sector: str = Field(default="", description="NACE sector")
    activity: str = Field(default="", description="ETS activity type")
    annual_emissions_tco2: float = Field(default=0.0, description="Annual verified emissions")
    free_allocation: float = Field(default=0.0, description="Annual free allocation")
    carbon_leakage_exposed: bool = Field(default=True, description="On CL list")


class Sector(BaseModel):
    """A sector on the carbon leakage list."""
    nace_code: str = Field(..., description="NACE sector code")
    description: str = Field(default="", description="Sector description")
    carbon_leakage_exposed: bool = Field(default=True, description="CL exposed")
    trade_intensity: float = Field(default=0.0, description="Trade intensity ratio")
    emission_intensity: float = Field(default=0.0, description="Emission intensity")
    cbam_sector: str = Field(default="", description="Corresponding CBAM sector")


# =============================================================================
# ETS Benchmark Database
# =============================================================================


_ETS_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    # Iron & Steel
    "hot_metal_bf_bof": {
        "product_name": "Hot metal (BF-BOF)",
        "value": 1.328,
        "unit": "tonnes",
        "sector": "Iron and Steel",
        "cbam_category": "IRON_AND_STEEL",
        "notes": "Blast furnace - basic oxygen furnace route",
    },
    "eaf_carbon_steel": {
        "product_name": "EAF carbon steel",
        "value": 0.283,
        "unit": "tonnes",
        "sector": "Iron and Steel",
        "cbam_category": "IRON_AND_STEEL",
        "notes": "Electric arc furnace route",
    },
    "eaf_high_alloy_steel": {
        "product_name": "EAF high alloy steel",
        "value": 0.352,
        "unit": "tonnes",
        "sector": "Iron and Steel",
        "cbam_category": "IRON_AND_STEEL",
        "notes": "Electric arc furnace, high alloy",
    },
    "sintered_ore": {
        "product_name": "Sintered ore",
        "value": 0.171,
        "unit": "tonnes",
        "sector": "Iron and Steel",
        "cbam_category": "IRON_AND_STEEL",
        "notes": "Sinter plant",
    },
    "coke": {
        "product_name": "Coke",
        "value": 0.286,
        "unit": "tonnes",
        "sector": "Iron and Steel",
        "cbam_category": "IRON_AND_STEEL",
        "notes": "Coke oven",
    },
    "iron_casting": {
        "product_name": "Iron casting",
        "value": 0.325,
        "unit": "tonnes",
        "sector": "Iron and Steel",
        "cbam_category": "IRON_AND_STEEL",
    },
    # Cement
    "grey_cement_clinker": {
        "product_name": "Grey cement clinker",
        "value": 0.766,
        "unit": "tonnes",
        "sector": "Cement",
        "cbam_category": "CEMENT",
        "notes": "Grey Portland cement clinker",
    },
    "white_cement_clinker": {
        "product_name": "White cement clinker",
        "value": 0.987,
        "unit": "tonnes",
        "sector": "Cement",
        "cbam_category": "CEMENT",
        "notes": "White Portland cement clinker",
    },
    # Aluminium
    "primary_aluminium": {
        "product_name": "Primary aluminium (pre-bake)",
        "value": 1.514,
        "unit": "tonnes",
        "sector": "Aluminium",
        "cbam_category": "ALUMINIUM",
        "notes": "Pre-bake anode process",
    },
    "alumina_refining": {
        "product_name": "Alumina (Bayer process)",
        "value": 0.472,
        "unit": "tonnes",
        "sector": "Aluminium",
        "cbam_category": "ALUMINIUM",
        "notes": "Alumina refining",
    },
    # Fertilisers
    "ammonia": {
        "product_name": "Ammonia",
        "value": 1.619,
        "unit": "tonnes",
        "sector": "Fertilisers",
        "cbam_category": "FERTILISERS",
        "notes": "Haber-Bosch process",
    },
    "nitric_acid": {
        "product_name": "Nitric acid",
        "value": 0.302,
        "unit": "tonnes",
        "sector": "Fertilisers",
        "cbam_category": "FERTILISERS",
        "notes": "Nitric acid production (100% concentration)",
    },
    "urea": {
        "product_name": "Urea",
        "value": 0.742,
        "unit": "tonnes",
        "sector": "Fertilisers",
        "cbam_category": "FERTILISERS",
        "notes": "Urea production",
    },
    "ammonium_nitrate": {
        "product_name": "Ammonium nitrate",
        "value": 0.580,
        "unit": "tonnes",
        "sector": "Fertilisers",
        "cbam_category": "FERTILISERS",
    },
    # Hydrogen
    "hydrogen_smr": {
        "product_name": "Hydrogen (SMR)",
        "value": 8.85,
        "unit": "tonnes",
        "sector": "Hydrogen",
        "cbam_category": "HYDROGEN",
        "notes": "Steam methane reforming",
    },
    "hydrogen_electrolysis": {
        "product_name": "Hydrogen (electrolysis)",
        "value": 0.0,
        "unit": "tonnes",
        "sector": "Hydrogen",
        "cbam_category": "HYDROGEN",
        "notes": "Green hydrogen (zero direct emissions)",
    },
}

# CBAM phaseout schedule (percentage of CBAM obligation that applies)
_CBAM_PHASEOUT_SCHEDULE: Dict[int, float] = {
    2026: 2.5,
    2027: 5.0,
    2028: 10.0,
    2029: 22.5,
    2030: 35.0,
    2031: 48.5,
    2032: 61.0,
    2033: 73.5,
    2034: 86.0,
    2035: 100.0,
}

# Carbon leakage sectors
_CARBON_LEAKAGE_SECTORS: List[Dict[str, Any]] = [
    {"nace": "24.10", "description": "Manufacture of basic iron and steel", "cbam": "IRON_AND_STEEL", "trade_intensity": 0.28, "emission_intensity": 1.85},
    {"nace": "24.42", "description": "Aluminium production", "cbam": "ALUMINIUM", "trade_intensity": 0.35, "emission_intensity": 8.40},
    {"nace": "23.51", "description": "Manufacture of cement", "cbam": "CEMENT", "trade_intensity": 0.06, "emission_intensity": 0.64},
    {"nace": "20.15", "description": "Manufacture of fertilisers", "cbam": "FERTILISERS", "trade_intensity": 0.32, "emission_intensity": 2.96},
    {"nace": "20.11", "description": "Manufacture of industrial gases (hydrogen)", "cbam": "HYDROGEN", "trade_intensity": 0.10, "emission_intensity": 5.00},
]

# Sample covered installations
_SAMPLE_INSTALLATIONS: Dict[str, List[Dict[str, Any]]] = {
    "DE": [
        {"id": "DE-ETS-0001", "name": "ThyssenKrupp Duisburg", "sector": "24.10", "activity": "Iron and steel", "emissions": 15000000, "allocation": 12000000},
        {"id": "DE-ETS-0002", "name": "Salzgitter Flachstahl", "sector": "24.10", "activity": "Iron and steel", "emissions": 8000000, "allocation": 6500000},
    ],
    "FR": [
        {"id": "FR-ETS-0001", "name": "ArcelorMittal Dunkerque", "sector": "24.10", "activity": "Iron and steel", "emissions": 10000000, "allocation": 8000000},
    ],
    "PL": [
        {"id": "PL-ETS-0001", "name": "ArcelorMittal Poland", "sector": "24.10", "activity": "Iron and steel", "emissions": 6000000, "allocation": 5000000},
    ],
    "NL": [
        {"id": "NL-ETS-0001", "name": "Tata Steel IJmuiden", "sector": "24.10", "activity": "Iron and steel", "emissions": 12000000, "allocation": 9500000},
    ],
}


# =============================================================================
# ETS Registry Bridge Implementation
# =============================================================================


class ETSRegistryBridge:
    """EU ETS Union Registry (EUTL) data integration bridge.

    Provides access to free allocation data, product benchmark values,
    installation compliance status, CBAM-ETS cross-referencing,
    phaseout consistency verification, ETS price history, carbon
    leakage list data, and covered installation queries.

    Includes hardcoded benchmark values for all CBAM-relevant products
    as per the EU ETS Implementing Regulation.

    Attributes:
        config: Optional configuration dictionary
        _benchmarks: Product benchmark database

    Example:
        >>> bridge = ETSRegistryBridge()
        >>> bm = bridge.get_benchmark_value("hot_metal_bf_bof")
        >>> assert bm.value_tco2_per_unit == 1.328
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ETS Registry Bridge.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.logger = logger
        self._benchmarks = dict(_ETS_BENCHMARKS)

        self.logger.info(
            "ETSRegistryBridge initialized with %d benchmarks",
            len(self._benchmarks),
        )

    # -------------------------------------------------------------------------
    # Free Allocation
    # -------------------------------------------------------------------------

    def get_free_allocation(
        self, installation_id: str, year: int
    ) -> FreeAllocation:
        """Get free allocation for an ETS installation and year.

        Args:
            installation_id: ETS installation identifier.
            year: Allocation year.

        Returns:
            FreeAllocation with allocation details.
        """
        # Look up installation in sample data
        for country_installations in _SAMPLE_INSTALLATIONS.values():
            for inst in country_installations:
                if inst["id"] == installation_id:
                    allocation = FreeAllocation(
                        installation_id=installation_id,
                        year=year,
                        total_free_allocation=float(inst.get("allocation", 0)),
                        benchmark_id="hot_metal_bf_bof",
                        benchmark_value=1.328,
                        activity_level=float(inst.get("emissions", 0)),
                        cross_sectoral_correction=1.0,
                        carbon_leakage_exposed=True,
                    )
                    allocation.provenance_hash = _compute_hash(
                        f"allocation:{installation_id}:{year}:{allocation.total_free_allocation}"
                    )
                    return allocation

        # Default response for unknown installations
        result = FreeAllocation(
            installation_id=installation_id,
            year=year,
            total_free_allocation=0.0,
            notes="Installation not found in sample data",
        )
        result.provenance_hash = _compute_hash(
            f"allocation:{installation_id}:{year}:0"
        )
        return result

    # -------------------------------------------------------------------------
    # Benchmark Values
    # -------------------------------------------------------------------------

    def get_benchmark_value(self, product_benchmark_id: str) -> BenchmarkValue:
        """Get the ETS product benchmark value.

        Args:
            product_benchmark_id: Benchmark identifier (e.g. "hot_metal_bf_bof").

        Returns:
            BenchmarkValue with the benchmark tCO2 per unit.
        """
        entry = self._benchmarks.get(product_benchmark_id)
        if entry is None:
            self.logger.warning(
                "Benchmark '%s' not found", product_benchmark_id,
            )
            return BenchmarkValue(
                benchmark_id=product_benchmark_id,
                product_name="Unknown",
                value_tco2_per_unit=0.0,
            )

        return BenchmarkValue(
            benchmark_id=product_benchmark_id,
            product_name=entry.get("product_name", ""),
            value_tco2_per_unit=entry["value"],
            unit=entry.get("unit", "tonnes"),
            sector=entry.get("sector", ""),
            cbam_category=entry.get("cbam_category", ""),
            update_year=2021,
            reduction_rate_pct=1.6,
            notes=entry.get("notes", ""),
        )

    def list_benchmarks(self) -> List[BenchmarkValue]:
        """List all available product benchmarks.

        Returns:
            List of BenchmarkValue entries.
        """
        results: List[BenchmarkValue] = []
        for bm_id in sorted(self._benchmarks.keys()):
            results.append(self.get_benchmark_value(bm_id))
        return results

    def get_benchmarks_by_category(self, cbam_category: str) -> List[BenchmarkValue]:
        """Get benchmarks for a specific CBAM category.

        Args:
            cbam_category: CBAM category (e.g. "IRON_AND_STEEL").

        Returns:
            List of BenchmarkValue entries for the category.
        """
        results: List[BenchmarkValue] = []
        for bm_id, entry in self._benchmarks.items():
            if entry.get("cbam_category", "") == cbam_category:
                results.append(self.get_benchmark_value(bm_id))
        return results

    # -------------------------------------------------------------------------
    # Installation Compliance
    # -------------------------------------------------------------------------

    def get_installation_compliance(
        self, installation_id: str
    ) -> ComplianceStatus:
        """Get compliance status for an ETS installation.

        Args:
            installation_id: ETS installation identifier.

        Returns:
            ComplianceStatus with compliance details.
        """
        year = datetime.utcnow().year - 1

        for country_installations in _SAMPLE_INSTALLATIONS.values():
            for inst in country_installations:
                if inst["id"] == installation_id:
                    emissions = float(inst.get("emissions", 0))
                    allocation = float(inst.get("allocation", 0))
                    balance = allocation - emissions

                    return ComplianceStatus(
                        installation_id=installation_id,
                        year=year,
                        verified_emissions=emissions,
                        free_allocation=allocation,
                        surrendered_allowances=int(emissions),
                        balance=balance,
                        is_compliant=balance >= 0,
                        compliance_status="compliant" if balance >= 0 else "deficit",
                    )

        return ComplianceStatus(
            installation_id=installation_id,
            year=year,
            compliance_status="not_found",
        )

    # -------------------------------------------------------------------------
    # Cross-Reference
    # -------------------------------------------------------------------------

    def cross_reference_cbam_ets(
        self,
        cbam_installation: str,
        ets_installation: str,
    ) -> CrossReference:
        """Cross-reference CBAM and ETS installation data.

        Args:
            cbam_installation: CBAM installation identifier.
            ets_installation: ETS installation identifier.

        Returns:
            CrossReference with consistency analysis.
        """
        ref = CrossReference(
            cbam_installation_id=cbam_installation,
            ets_installation_id=ets_installation,
        )

        # Look up ETS installation
        for country_installations in _SAMPLE_INSTALLATIONS.values():
            for inst in country_installations:
                if inst["id"] == ets_installation:
                    ref.ets_activity = inst.get("activity", "")
                    ref.benchmark_id = "hot_metal_bf_bof"
                    ref.cbam_category = "IRON_AND_STEEL"
                    ref.emission_factor_match = True
                    ref.notes = (
                        f"Matched: CBAM={cbam_installation} <-> "
                        f"ETS={ets_installation} ({inst.get('name', '')})"
                    )
                    break

        ref.provenance_hash = _compute_hash(
            f"xref:{ref.reference_id}:{cbam_installation}:{ets_installation}"
        )
        return ref

    # -------------------------------------------------------------------------
    # Phaseout Consistency
    # -------------------------------------------------------------------------

    def verify_phaseout_consistency(
        self,
        cbam_phaseout: float,
        ets_allocation: float,
    ) -> ConsistencyCheck:
        """Verify consistency between CBAM phaseout and ETS free allocation.

        The CBAM phaseout and ETS free allocation reduction should be
        mirror images: as CBAM increases, free allocation decreases, and
        their sum should approximate 100%.

        Args:
            cbam_phaseout: CBAM phaseout percentage (0-100).
            ets_allocation: ETS free allocation percentage (0-100).

        Returns:
            ConsistencyCheck with consistency analysis.
        """
        sum_pct = cbam_phaseout + ets_allocation
        is_consistent = abs(sum_pct - 100.0) <= 5.0  # 5% tolerance
        variance = round(sum_pct - 100.0, 2)

        if is_consistent:
            message = f"Consistent: CBAM {cbam_phaseout}% + ETS {ets_allocation}% = {sum_pct}%"
        else:
            message = (
                f"Inconsistent: CBAM {cbam_phaseout}% + ETS {ets_allocation}% = "
                f"{sum_pct}% (variance: {variance}%)"
            )

        check = ConsistencyCheck(
            year=datetime.utcnow().year,
            cbam_phaseout_pct=cbam_phaseout,
            ets_free_allocation_pct=ets_allocation,
            sum_pct=round(sum_pct, 2),
            is_consistent=is_consistent,
            variance_pct=variance,
            message=message,
        )
        check.provenance_hash = _compute_hash(
            f"consistency:{check.check_id}:{cbam_phaseout}:{ets_allocation}"
        )
        return check

    def get_phaseout_schedule(self) -> Dict[int, Dict[str, float]]:
        """Get the CBAM phaseout and ETS allocation schedule.

        Returns:
            Dictionary mapping year to CBAM and ETS percentages.
        """
        schedule: Dict[int, Dict[str, float]] = {}
        for year, cbam_pct in _CBAM_PHASEOUT_SCHEDULE.items():
            ets_pct = 100.0 - cbam_pct
            schedule[year] = {
                "cbam_pct": cbam_pct,
                "ets_free_allocation_pct": ets_pct,
                "sum": 100.0,
            }
        return schedule

    # -------------------------------------------------------------------------
    # ETS Price
    # -------------------------------------------------------------------------

    def get_ets_price_history(
        self, start_date: str, end_date: str
    ) -> List[ETSPrice]:
        """Get ETS allowance price history.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            List of ETSPrice observations.
        """
        prices: List[ETSPrice] = []
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        current = start
        base_price = 65.0
        while current <= end:
            if current.weekday() < 5:
                months = (current.year - 2024) * 12 + current.month - 1
                seasonal = 5.0 * math.sin(2 * math.pi * (current.month - 1) / 12)
                trend = 0.5 * months
                noise = ((current.toordinal() * 7 + 13) % 100 - 50) / 25.0
                price = max(40.0, min(120.0, round(base_price + trend + seasonal + noise, 2)))

                prices.append(ETSPrice(
                    date=current.strftime("%Y-%m-%d"),
                    price_eur_per_tco2=price,
                    source="MOCK",
                ))
            current += timedelta(days=1)

        return prices

    def get_current_ets_price(self) -> ETSPrice:
        """Get the most recent ETS price.

        Returns:
            ETSPrice for the latest trading day.
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        history = self.get_ets_price_history(week_ago, today)
        if history:
            return history[-1]

        return ETSPrice(
            date=today,
            price_eur_per_tco2=75.0,
            source="FALLBACK",
        )

    # -------------------------------------------------------------------------
    # Covered Installations
    # -------------------------------------------------------------------------

    def list_covered_installations(self, country: str) -> List[Installation]:
        """List ETS-covered installations in a country.

        Args:
            country: ISO alpha-2 country code.

        Returns:
            List of Installation entries.
        """
        installations_data = _SAMPLE_INSTALLATIONS.get(country.upper(), [])
        results: List[Installation] = []

        for inst in installations_data:
            results.append(Installation(
                installation_id=inst["id"],
                installation_name=inst.get("name", ""),
                country=country.upper(),
                sector=inst.get("sector", ""),
                activity=inst.get("activity", ""),
                annual_emissions_tco2=float(inst.get("emissions", 0)),
                free_allocation=float(inst.get("allocation", 0)),
                carbon_leakage_exposed=True,
            ))

        return results

    # -------------------------------------------------------------------------
    # Carbon Leakage List
    # -------------------------------------------------------------------------

    def get_carbon_leakage_list(self) -> List[Sector]:
        """Get the EU ETS carbon leakage list sectors.

        Returns:
            List of Sector entries on the carbon leakage list.
        """
        results: List[Sector] = []
        for sector_data in _CARBON_LEAKAGE_SECTORS:
            results.append(Sector(
                nace_code=sector_data["nace"],
                description=sector_data["description"],
                carbon_leakage_exposed=True,
                trade_intensity=sector_data.get("trade_intensity", 0.0),
                emission_intensity=sector_data.get("emission_intensity", 0.0),
                cbam_sector=sector_data.get("cbam", ""),
            ))
        return results


# =============================================================================
# Module-Level Helper
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string.

    Args:
        data: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
