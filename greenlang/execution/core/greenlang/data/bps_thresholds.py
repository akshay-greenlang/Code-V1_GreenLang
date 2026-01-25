"""
Building Performance Standards (BPS) Threshold Database

This module provides BPS thresholds for different building types and climate zones.
Used to determine compliance with urban building decarbonization requirements.

Sources:
- ASHRAE Standard 90.1 (Energy Standard for Buildings)
- NYC Local Law 97 (Building Emissions Law)
- Washington State Building Performance Standard
- ENERGY STAR Portfolio Manager benchmarks
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class BuildingType(str, Enum):
    """Building types for BPS classification."""
    OFFICE = "office"
    RESIDENTIAL_MULTIFAMILY = "residential"
    RETAIL = "retail"
    INDUSTRIAL = "industrial"
    WAREHOUSE = "warehouse"
    HOTEL = "hotel"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    RESTAURANT = "restaurant"


class ClimateZone(str, Enum):
    """ASHRAE climate zones (US-based)."""
    ZONE_1A = "1A"  # Very hot, humid (Miami)
    ZONE_2A = "2A"  # Hot, humid (Houston)
    ZONE_3A = "3A"  # Warm, humid (Atlanta)
    ZONE_4A = "4A"  # Mixed, humid (NYC, Boston)
    ZONE_5A = "5A"  # Cool, humid (Chicago)
    ZONE_6A = "6A"  # Cold, humid (Minneapolis)
    ZONE_7 = "7"    # Very cold (Duluth)


@dataclass
class BPSThreshold:
    """BPS threshold for a building type and climate zone."""
    building_type: str
    climate_zone: str
    threshold_kwh_per_sqm: float  # kWh/m²/year
    ghg_threshold_kgco2e_per_sqm: Optional[float]  # kgCO2e/m²/year
    source: str
    jurisdiction: str
    effective_date: str
    notes: str


class BPSThresholdDatabase:
    """
    Database of Building Performance Standard thresholds.

    Thresholds vary by:
    - Building type (office, residential, etc.)
    - Climate zone (heating/cooling degree days)
    - Jurisdiction (NYC, Washington, California)
    """

    def __init__(self):
        self.thresholds: Dict[str, BPSThreshold] = {}
        self._load_thresholds()

    def _load_thresholds(self):
        """Load BPS thresholds from various jurisdictions."""

        # Office buildings
        self.thresholds["office_4A"] = BPSThreshold(
            building_type="office",
            climate_zone="4A",
            threshold_kwh_per_sqm=80.0,
            ghg_threshold_kgco2e_per_sqm=5.4,
            source="NYC Local Law 97 2024-2029",
            jurisdiction="NYC",
            effective_date="2024-01-01",
            notes="Office buildings in NYC Climate Zone 4A (mixed humid)"
        )

        self.thresholds["office_5A"] = BPSThreshold(
            building_type="office",
            climate_zone="5A",
            threshold_kwh_per_sqm=85.0,
            ghg_threshold_kgco2e_per_sqm=5.7,
            source="ENERGY STAR median EUI",
            jurisdiction="US National",
            effective_date="2024-01-01",
            notes="Office buildings in Climate Zone 5A (cool humid)"
        )

        self.thresholds["office_default"] = BPSThreshold(
            building_type="office",
            climate_zone="default",
            threshold_kwh_per_sqm=80.0,
            ghg_threshold_kgco2e_per_sqm=5.4,
            source="ASHRAE 90.1-2019",
            jurisdiction="US National",
            effective_date="2024-01-01",
            notes="Default office building threshold"
        )

        # Residential multifamily
        self.thresholds["residential_4A"] = BPSThreshold(
            building_type="residential",
            climate_zone="4A",
            threshold_kwh_per_sqm=100.0,
            ghg_threshold_kgco2e_per_sqm=6.7,
            source="NYC Local Law 97 2024-2029",
            jurisdiction="NYC",
            effective_date="2024-01-01",
            notes="Multifamily residential in NYC"
        )

        self.thresholds["residential_5A"] = BPSThreshold(
            building_type="residential",
            climate_zone="5A",
            threshold_kwh_per_sqm=110.0,
            ghg_threshold_kgco2e_per_sqm=7.3,
            source="ENERGY STAR median EUI",
            jurisdiction="US National",
            effective_date="2024-01-01",
            notes="Multifamily residential in cool climate"
        )

        self.thresholds["residential_default"] = BPSThreshold(
            building_type="residential",
            climate_zone="default",
            threshold_kwh_per_sqm=100.0,
            ghg_threshold_kgco2e_per_sqm=6.7,
            source="ENERGY STAR",
            jurisdiction="US National",
            effective_date="2024-01-01",
            notes="Default residential threshold"
        )

        # Retail
        self.thresholds["retail_4A"] = BPSThreshold(
            building_type="retail",
            climate_zone="4A",
            threshold_kwh_per_sqm=120.0,
            ghg_threshold_kgco2e_per_sqm=8.0,
            source="NYC Local Law 97",
            jurisdiction="NYC",
            effective_date="2024-01-01",
            notes="Retail buildings (higher EUI due to lighting, HVAC)"
        )

        self.thresholds["retail_default"] = BPSThreshold(
            building_type="retail",
            climate_zone="default",
            threshold_kwh_per_sqm=120.0,
            ghg_threshold_kgco2e_per_sqm=8.0,
            source="ENERGY STAR",
            jurisdiction="US National",
            effective_date="2024-01-01",
            notes="Default retail threshold"
        )

        # Industrial
        self.thresholds["industrial_default"] = BPSThreshold(
            building_type="industrial",
            climate_zone="default",
            threshold_kwh_per_sqm=200.0,
            ghg_threshold_kgco2e_per_sqm=13.4,
            source="DOE Manufacturing Energy Consumption Survey",
            jurisdiction="US National",
            effective_date="2024-01-01",
            notes="Industrial facilities (highly variable by process)"
        )

        # Warehouse
        self.thresholds["warehouse_default"] = BPSThreshold(
            building_type="warehouse",
            climate_zone="default",
            threshold_kwh_per_sqm=50.0,
            ghg_threshold_kgco2e_per_sqm=3.4,
            source="ENERGY STAR",
            jurisdiction="US National",
            effective_date="2024-01-01",
            notes="Warehouses (low EUI due to minimal HVAC)"
        )

        # Hotel
        self.thresholds["hotel_4A"] = BPSThreshold(
            building_type="hotel",
            climate_zone="4A",
            threshold_kwh_per_sqm=140.0,
            ghg_threshold_kgco2e_per_sqm=9.4,
            source="ENERGY STAR",
            jurisdiction="US National",
            effective_date="2024-01-01",
            notes="Hotels (high EUI due to 24/7 operation, hot water)"
        )

        # Hospital
        self.thresholds["hospital_default"] = BPSThreshold(
            building_type="hospital",
            climate_zone="default",
            threshold_kwh_per_sqm=250.0,
            ghg_threshold_kgco2e_per_sqm=16.8,
            source="ENERGY STAR",
            jurisdiction="US National",
            effective_date="2024-01-01",
            notes="Hospitals (highest EUI, critical operations)"
        )

        # School
        self.thresholds["school_4A"] = BPSThreshold(
            building_type="school",
            climate_zone="4A",
            threshold_kwh_per_sqm=90.0,
            ghg_threshold_kgco2e_per_sqm=6.0,
            source="ENERGY STAR",
            jurisdiction="US National",
            effective_date="2024-01-01",
            notes="K-12 schools"
        )

    def lookup(
        self,
        building_type: str,
        climate_zone: Optional[str] = None
    ) -> Optional[BPSThreshold]:
        """
        Look up BPS threshold for building type and climate zone.

        Args:
            building_type: Building type (office, residential, etc.)
            climate_zone: ASHRAE climate zone (optional)

        Returns:
            BPSThreshold if found, None otherwise
        """
        # Try specific climate zone first
        if climate_zone:
            key = f"{building_type}_{climate_zone}"
            if key in self.thresholds:
                return self.thresholds[key]

        # Fall back to default for building type
        default_key = f"{building_type}_default"
        if default_key in self.thresholds:
            return self.thresholds[default_key]

        # Try without climate zone suffix
        if building_type in self.thresholds:
            return self.thresholds[building_type]

        return None

    def list_building_types(self) -> list:
        """List all supported building types."""
        types = set()
        for key in self.thresholds.keys():
            building_type = key.split('_')[0]
            types.add(building_type)
        return sorted(types)

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "total_thresholds": len(self.thresholds),
            "building_types": len(self.list_building_types()),
            "avg_eui": sum(t.threshold_kwh_per_sqm for t in self.thresholds.values()) / len(self.thresholds),
            "max_eui": max(t.threshold_kwh_per_sqm for t in self.thresholds.values()),
            "min_eui": min(t.threshold_kwh_per_sqm for t in self.thresholds.values()),
        }


# Global instance
_bps_db: Optional[BPSThresholdDatabase] = None


def get_bps_database() -> BPSThresholdDatabase:
    """Get global BPS threshold database instance."""
    global _bps_db
    if _bps_db is None:
        _bps_db = BPSThresholdDatabase()
    return _bps_db
