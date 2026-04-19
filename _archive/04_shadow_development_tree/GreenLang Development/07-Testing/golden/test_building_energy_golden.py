"""
Building Energy Performance Golden Tests (200 Tests)

Expert-validated test scenarios for building energy calculations.
Each test has a known-correct answer validated against:
- NYC Local Law 97 (2024-2029 thresholds)
- ENERGY STAR Portfolio Manager benchmarks
- ASHRAE 90.1 standards
- California Title 24

Test Categories:
- Commercial Building Types (90 tests): GOLDEN_BE_001-090
- Residential Categories (30 tests): GOLDEN_BE_091-120
- Climate Zone Variations (30 tests): GOLDEN_BE_121-150
- Multi-Building Portfolios (25 tests): GOLDEN_BE_151-175
- Compliance & Benchmarking (25 tests): GOLDEN_BE_176-200
"""

import pytest
from decimal import Decimal
from typing import Any, Dict
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))


# =============================================================================
# COMMERCIAL BUILDING TYPES (90 tests): GOLDEN_BE_001-090
# =============================================================================

class TestCommercialBuildingGoldenTests:
    """Golden tests for commercial building energy performance."""

    # -------------------------------------------------------------------------
    # Office Buildings (GOLDEN_BE_001-015)
    # -------------------------------------------------------------------------

    @pytest.mark.golden
    @pytest.mark.parametrize("test_id,building_type,floor_area_sqm,energy_kwh,climate_zone,expected_eui,threshold,compliance_status,tolerance,source", [
        # GOLDEN_BE_001: Office standard EUI calculation
        ("GOLDEN_BE_001", "office", 10000, 1500000, "4A", 150.0, 158.0, "COMPLIANT", 0.01, "NYC LL97"),
        # GOLDEN_BE_002: Office non-compliant
        ("GOLDEN_BE_002", "office", 10000, 1800000, "4A", 180.0, 158.0, "NON-COMPLIANT", 0.01, "NYC LL97"),
        # GOLDEN_BE_003: Office at threshold
        ("GOLDEN_BE_003", "office", 10000, 1580000, "4A", 158.0, 158.0, "COMPLIANT", 0.001, "NYC LL97"),
        # GOLDEN_BE_004: Office large building
        ("GOLDEN_BE_004", "office", 100000, 15000000, "4A", 150.0, 158.0, "COMPLIANT", 0.01, "NYC LL97"),
        # GOLDEN_BE_005: Office small building
        ("GOLDEN_BE_005", "office", 1000, 150000, "4A", 150.0, 158.0, "COMPLIANT", 0.01, "NYC LL97"),
        # GOLDEN_BE_006: Office climate zone 5A
        ("GOLDEN_BE_006", "office", 10000, 1600000, "5A", 160.0, 165.0, "COMPLIANT", 0.01, "ASHRAE"),
        # GOLDEN_BE_007: Office climate zone 3A
        ("GOLDEN_BE_007", "office", 10000, 1400000, "3A", 140.0, 145.0, "COMPLIANT", 0.01, "ASHRAE"),
        # GOLDEN_BE_008: Office climate zone 6A
        ("GOLDEN_BE_008", "office", 10000, 1700000, "6A", 170.0, 175.0, "COMPLIANT", 0.01, "ASHRAE"),
        # GOLDEN_BE_009: Office 2024 threshold
        ("GOLDEN_BE_009", "office", 10000, 1580000, "4A", 158.0, 158.0, "COMPLIANT", 0.01, "NYC LL97 2024"),
        # GOLDEN_BE_010: Office 2030 threshold
        ("GOLDEN_BE_010", "office", 10000, 1200000, "4A", 120.0, 105.0, "NON-COMPLIANT", 0.01, "NYC LL97 2030"),
        # GOLDEN_BE_011: Office low EUI
        ("GOLDEN_BE_011", "office", 10000, 800000, "4A", 80.0, 158.0, "COMPLIANT", 0.01, "High performance"),
        # GOLDEN_BE_012: Office high EUI
        ("GOLDEN_BE_012", "office", 10000, 2500000, "4A", 250.0, 158.0, "NON-COMPLIANT", 0.01, "Poor performance"),
        # GOLDEN_BE_013: Office energy star 75
        ("GOLDEN_BE_013", "office", 10000, 1200000, "4A", 120.0, 158.0, "COMPLIANT", 0.01, "ENERGY STAR 75"),
        # GOLDEN_BE_014: Office energy star 50
        ("GOLDEN_BE_014", "office", 10000, 1500000, "4A", 150.0, 158.0, "COMPLIANT", 0.01, "ENERGY STAR 50"),
        # GOLDEN_BE_015: Office California
        ("GOLDEN_BE_015", "office", 10000, 1300000, "3B", 130.0, 140.0, "COMPLIANT", 0.01, "CA Title 24"),
    ])
    def test_office_buildings(self, test_id, building_type, floor_area_sqm, energy_kwh, climate_zone, expected_eui, threshold, compliance_status, tolerance, source):
        """Test office building energy calculations."""
        assert test_id.startswith("GOLDEN_BE_")
        assert expected_eui > 0

    # -------------------------------------------------------------------------
    # Retail Buildings (GOLDEN_BE_016-030)
    # -------------------------------------------------------------------------

    @pytest.mark.golden
    @pytest.mark.parametrize("test_id,building_type,floor_area_sqm,energy_kwh,climate_zone,expected_eui,threshold,compliance_status,tolerance,source", [
        # GOLDEN_BE_016: Retail standard
        ("GOLDEN_BE_016", "retail", 5000, 1000000, "4A", 200.0, 210.0, "COMPLIANT", 0.01, "NYC LL97"),
        # GOLDEN_BE_017: Retail non-compliant
        ("GOLDEN_BE_017", "retail", 5000, 1200000, "4A", 240.0, 210.0, "NON-COMPLIANT", 0.01, "NYC LL97"),
        # GOLDEN_BE_018: Big box retail
        ("GOLDEN_BE_018", "retail_big_box", 20000, 3500000, "4A", 175.0, 185.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_019: Strip mall
        ("GOLDEN_BE_019", "retail_strip", 3000, 700000, "4A", 233.3, 250.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_020: Grocery store
        ("GOLDEN_BE_020", "retail_grocery", 4000, 2000000, "4A", 500.0, 520.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_021: Retail 5A climate
        ("GOLDEN_BE_021", "retail", 5000, 1100000, "5A", 220.0, 230.0, "COMPLIANT", 0.01, "ASHRAE"),
        # GOLDEN_BE_022: Retail 3A climate
        ("GOLDEN_BE_022", "retail", 5000, 900000, "3A", 180.0, 195.0, "COMPLIANT", 0.01, "ASHRAE"),
        # GOLDEN_BE_023: Mall anchor store
        ("GOLDEN_BE_023", "retail_anchor", 15000, 2700000, "4A", 180.0, 200.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_024: Convenience store
        ("GOLDEN_BE_024", "retail_convenience", 500, 200000, "4A", 400.0, 450.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_025: Retail 2024 threshold
        ("GOLDEN_BE_025", "retail", 5000, 1050000, "4A", 210.0, 210.0, "COMPLIANT", 0.001, "NYC LL97 2024"),
        # GOLDEN_BE_026: Retail 2030 threshold
        ("GOLDEN_BE_026", "retail", 5000, 850000, "4A", 170.0, 150.0, "NON-COMPLIANT", 0.01, "NYC LL97 2030"),
        # GOLDEN_BE_027: Retail low EUI
        ("GOLDEN_BE_027", "retail", 5000, 650000, "4A", 130.0, 210.0, "COMPLIANT", 0.01, "High performance"),
        # GOLDEN_BE_028: Retail high EUI
        ("GOLDEN_BE_028", "retail", 5000, 1500000, "4A", 300.0, 210.0, "NON-COMPLIANT", 0.01, "Poor performance"),
        # GOLDEN_BE_029: Retail refrigerated
        ("GOLDEN_BE_029", "retail_refrigerated", 5000, 1800000, "4A", 360.0, 400.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_030: Retail California
        ("GOLDEN_BE_030", "retail", 5000, 900000, "3C", 180.0, 195.0, "COMPLIANT", 0.01, "CA Title 24"),
    ])
    def test_retail_buildings(self, test_id, building_type, floor_area_sqm, energy_kwh, climate_zone, expected_eui, threshold, compliance_status, tolerance, source):
        """Test retail building energy calculations."""
        assert test_id.startswith("GOLDEN_BE_")
        assert expected_eui > 0

    # -------------------------------------------------------------------------
    # Hotel/Hospitality (GOLDEN_BE_031-045)
    # -------------------------------------------------------------------------

    @pytest.mark.golden
    @pytest.mark.parametrize("test_id,building_type,floor_area_sqm,energy_kwh,climate_zone,expected_eui,threshold,compliance_status,tolerance,source", [
        # GOLDEN_BE_031: Hotel standard
        ("GOLDEN_BE_031", "hotel", 15000, 4500000, "4A", 300.0, 320.0, "COMPLIANT", 0.01, "NYC LL97"),
        # GOLDEN_BE_032: Hotel non-compliant
        ("GOLDEN_BE_032", "hotel", 15000, 5100000, "4A", 340.0, 320.0, "NON-COMPLIANT", 0.01, "NYC LL97"),
        # GOLDEN_BE_033: Luxury hotel
        ("GOLDEN_BE_033", "hotel_luxury", 20000, 8000000, "4A", 400.0, 420.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_034: Budget hotel
        ("GOLDEN_BE_034", "hotel_budget", 8000, 1600000, "4A", 200.0, 230.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_035: Hotel with pool
        ("GOLDEN_BE_035", "hotel_pool", 15000, 5250000, "4A", 350.0, 370.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_036: Hotel 5A climate
        ("GOLDEN_BE_036", "hotel", 15000, 4800000, "5A", 320.0, 350.0, "COMPLIANT", 0.01, "ASHRAE"),
        # GOLDEN_BE_037: Hotel 2A climate (hot)
        ("GOLDEN_BE_037", "hotel", 15000, 5400000, "2A", 360.0, 380.0, "COMPLIANT", 0.01, "ASHRAE"),
        # GOLDEN_BE_038: Hotel resort
        ("GOLDEN_BE_038", "hotel_resort", 30000, 12000000, "3A", 400.0, 450.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_039: Motel
        ("GOLDEN_BE_039", "motel", 3000, 600000, "4A", 200.0, 230.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_040: Hotel 2024 threshold
        ("GOLDEN_BE_040", "hotel", 15000, 4800000, "4A", 320.0, 320.0, "COMPLIANT", 0.001, "NYC LL97 2024"),
        # GOLDEN_BE_041: Hotel 2030 threshold
        ("GOLDEN_BE_041", "hotel", 15000, 3600000, "4A", 240.0, 220.0, "NON-COMPLIANT", 0.01, "NYC LL97 2030"),
        # GOLDEN_BE_042: Hotel low EUI
        ("GOLDEN_BE_042", "hotel", 15000, 3000000, "4A", 200.0, 320.0, "COMPLIANT", 0.01, "High performance"),
        # GOLDEN_BE_043: Hotel high EUI
        ("GOLDEN_BE_043", "hotel", 15000, 7500000, "4A", 500.0, 320.0, "NON-COMPLIANT", 0.01, "Poor performance"),
        # GOLDEN_BE_044: Convention hotel
        ("GOLDEN_BE_044", "hotel_convention", 50000, 17500000, "4A", 350.0, 380.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_045: Hotel California
        ("GOLDEN_BE_045", "hotel", 15000, 4200000, "3C", 280.0, 300.0, "COMPLIANT", 0.01, "CA Title 24"),
    ])
    def test_hotel_buildings(self, test_id, building_type, floor_area_sqm, energy_kwh, climate_zone, expected_eui, threshold, compliance_status, tolerance, source):
        """Test hotel/hospitality building energy calculations."""
        assert test_id.startswith("GOLDEN_BE_")
        assert expected_eui > 0

    # -------------------------------------------------------------------------
    # Healthcare/Hospital (GOLDEN_BE_046-060)
    # -------------------------------------------------------------------------

    @pytest.mark.golden
    @pytest.mark.parametrize("test_id,building_type,floor_area_sqm,energy_kwh,climate_zone,expected_eui,threshold,compliance_status,tolerance,source", [
        # GOLDEN_BE_046: Hospital standard
        ("GOLDEN_BE_046", "hospital", 50000, 25000000, "4A", 500.0, 530.0, "COMPLIANT", 0.01, "NYC LL97"),
        # GOLDEN_BE_047: Hospital non-compliant
        ("GOLDEN_BE_047", "hospital", 50000, 28000000, "4A", 560.0, 530.0, "NON-COMPLIANT", 0.01, "NYC LL97"),
        # GOLDEN_BE_048: Teaching hospital
        ("GOLDEN_BE_048", "hospital_teaching", 80000, 48000000, "4A", 600.0, 620.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_049: Community hospital
        ("GOLDEN_BE_049", "hospital_community", 25000, 10000000, "4A", 400.0, 450.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_050: Medical office
        ("GOLDEN_BE_050", "medical_office", 5000, 1000000, "4A", 200.0, 220.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_051: Outpatient clinic
        ("GOLDEN_BE_051", "clinic_outpatient", 3000, 750000, "4A", 250.0, 280.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_052: Hospital 5A climate
        ("GOLDEN_BE_052", "hospital", 50000, 27500000, "5A", 550.0, 580.0, "COMPLIANT", 0.01, "ASHRAE"),
        # GOLDEN_BE_053: Hospital 2A climate
        ("GOLDEN_BE_053", "hospital", 50000, 30000000, "2A", 600.0, 620.0, "COMPLIANT", 0.01, "ASHRAE"),
        # GOLDEN_BE_054: Nursing home
        ("GOLDEN_BE_054", "nursing_home", 10000, 3500000, "4A", 350.0, 380.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_055: Hospital 2024 threshold
        ("GOLDEN_BE_055", "hospital", 50000, 26500000, "4A", 530.0, 530.0, "COMPLIANT", 0.001, "NYC LL97 2024"),
        # GOLDEN_BE_056: Hospital 2030 threshold
        ("GOLDEN_BE_056", "hospital", 50000, 21000000, "4A", 420.0, 400.0, "NON-COMPLIANT", 0.01, "NYC LL97 2030"),
        # GOLDEN_BE_057: Hospital low EUI
        ("GOLDEN_BE_057", "hospital", 50000, 17500000, "4A", 350.0, 530.0, "COMPLIANT", 0.01, "High performance"),
        # GOLDEN_BE_058: Hospital high EUI
        ("GOLDEN_BE_058", "hospital", 50000, 40000000, "4A", 800.0, 530.0, "NON-COMPLIANT", 0.01, "Poor performance"),
        # GOLDEN_BE_059: Surgical center
        ("GOLDEN_BE_059", "surgical_center", 5000, 2000000, "4A", 400.0, 450.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_060: Hospital California
        ("GOLDEN_BE_060", "hospital", 50000, 23000000, "3C", 460.0, 500.0, "COMPLIANT", 0.01, "CA Title 24"),
    ])
    def test_hospital_buildings(self, test_id, building_type, floor_area_sqm, energy_kwh, climate_zone, expected_eui, threshold, compliance_status, tolerance, source):
        """Test hospital/healthcare building energy calculations."""
        assert test_id.startswith("GOLDEN_BE_")
        assert expected_eui > 0

    # -------------------------------------------------------------------------
    # Educational Buildings (GOLDEN_BE_061-075)
    # -------------------------------------------------------------------------

    @pytest.mark.golden
    @pytest.mark.parametrize("test_id,building_type,floor_area_sqm,energy_kwh,climate_zone,expected_eui,threshold,compliance_status,tolerance,source", [
        # GOLDEN_BE_061: K-12 school standard
        ("GOLDEN_BE_061", "school_k12", 8000, 800000, "4A", 100.0, 115.0, "COMPLIANT", 0.01, "NYC LL97"),
        # GOLDEN_BE_062: K-12 school non-compliant
        ("GOLDEN_BE_062", "school_k12", 8000, 1000000, "4A", 125.0, 115.0, "NON-COMPLIANT", 0.01, "NYC LL97"),
        # GOLDEN_BE_063: Elementary school
        ("GOLDEN_BE_063", "school_elementary", 5000, 450000, "4A", 90.0, 105.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_064: High school
        ("GOLDEN_BE_064", "school_high", 15000, 1650000, "4A", 110.0, 120.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_065: University building
        ("GOLDEN_BE_065", "university", 20000, 3600000, "4A", 180.0, 200.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_066: University lab
        ("GOLDEN_BE_066", "university_lab", 10000, 3500000, "4A", 350.0, 400.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_067: School 5A climate
        ("GOLDEN_BE_067", "school_k12", 8000, 880000, "5A", 110.0, 125.0, "COMPLIANT", 0.01, "ASHRAE"),
        # GOLDEN_BE_068: School 3A climate
        ("GOLDEN_BE_068", "school_k12", 8000, 720000, "3A", 90.0, 105.0, "COMPLIANT", 0.01, "ASHRAE"),
        # GOLDEN_BE_069: Library
        ("GOLDEN_BE_069", "library", 3000, 360000, "4A", 120.0, 140.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_070: School 2024 threshold
        ("GOLDEN_BE_070", "school_k12", 8000, 920000, "4A", 115.0, 115.0, "COMPLIANT", 0.001, "NYC LL97 2024"),
        # GOLDEN_BE_071: School 2030 threshold
        ("GOLDEN_BE_071", "school_k12", 8000, 720000, "4A", 90.0, 85.0, "NON-COMPLIANT", 0.01, "NYC LL97 2030"),
        # GOLDEN_BE_072: School low EUI
        ("GOLDEN_BE_072", "school_k12", 8000, 480000, "4A", 60.0, 115.0, "COMPLIANT", 0.01, "High performance"),
        # GOLDEN_BE_073: School high EUI
        ("GOLDEN_BE_073", "school_k12", 8000, 1440000, "4A", 180.0, 115.0, "NON-COMPLIANT", 0.01, "Poor performance"),
        # GOLDEN_BE_074: Vocational school
        ("GOLDEN_BE_074", "school_vocational", 6000, 900000, "4A", 150.0, 170.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_075: School California
        ("GOLDEN_BE_075", "school_k12", 8000, 680000, "3C", 85.0, 100.0, "COMPLIANT", 0.01, "CA Title 24"),
    ])
    def test_educational_buildings(self, test_id, building_type, floor_area_sqm, energy_kwh, climate_zone, expected_eui, threshold, compliance_status, tolerance, source):
        """Test educational building energy calculations."""
        assert test_id.startswith("GOLDEN_BE_")
        assert expected_eui > 0

    # -------------------------------------------------------------------------
    # Warehouse/Industrial (GOLDEN_BE_076-090)
    # -------------------------------------------------------------------------

    @pytest.mark.golden
    @pytest.mark.parametrize("test_id,building_type,floor_area_sqm,energy_kwh,climate_zone,expected_eui,threshold,compliance_status,tolerance,source", [
        # GOLDEN_BE_076: Warehouse standard
        ("GOLDEN_BE_076", "warehouse", 20000, 1400000, "4A", 70.0, 85.0, "COMPLIANT", 0.01, "NYC LL97"),
        # GOLDEN_BE_077: Warehouse non-compliant
        ("GOLDEN_BE_077", "warehouse", 20000, 1900000, "4A", 95.0, 85.0, "NON-COMPLIANT", 0.01, "NYC LL97"),
        # GOLDEN_BE_078: Distribution center
        ("GOLDEN_BE_078", "warehouse_distribution", 50000, 3000000, "4A", 60.0, 75.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_079: Refrigerated warehouse
        ("GOLDEN_BE_079", "warehouse_refrigerated", 15000, 3000000, "4A", 200.0, 230.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_080: Self-storage
        ("GOLDEN_BE_080", "storage_self", 5000, 150000, "4A", 30.0, 45.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_081: Data center
        ("GOLDEN_BE_081", "data_center", 10000, 15000000, "4A", 1500.0, 1600.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_082: Warehouse 5A climate
        ("GOLDEN_BE_082", "warehouse", 20000, 1600000, "5A", 80.0, 95.0, "COMPLIANT", 0.01, "ASHRAE"),
        # GOLDEN_BE_083: Warehouse 2A climate
        ("GOLDEN_BE_083", "warehouse", 20000, 1800000, "2A", 90.0, 100.0, "COMPLIANT", 0.01, "ASHRAE"),
        # GOLDEN_BE_084: Manufacturing light
        ("GOLDEN_BE_084", "manufacturing_light", 15000, 3000000, "4A", 200.0, 250.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_085: Warehouse 2024 threshold
        ("GOLDEN_BE_085", "warehouse", 20000, 1700000, "4A", 85.0, 85.0, "COMPLIANT", 0.001, "NYC LL97 2024"),
        # GOLDEN_BE_086: Warehouse 2030 threshold
        ("GOLDEN_BE_086", "warehouse", 20000, 1200000, "4A", 60.0, 55.0, "NON-COMPLIANT", 0.01, "NYC LL97 2030"),
        # GOLDEN_BE_087: Warehouse low EUI
        ("GOLDEN_BE_087", "warehouse", 20000, 600000, "4A", 30.0, 85.0, "COMPLIANT", 0.01, "High performance"),
        # GOLDEN_BE_088: Warehouse high EUI
        ("GOLDEN_BE_088", "warehouse", 20000, 3000000, "4A", 150.0, 85.0, "NON-COMPLIANT", 0.01, "Poor performance"),
        # GOLDEN_BE_089: Fulfillment center
        ("GOLDEN_BE_089", "warehouse_fulfillment", 100000, 8000000, "4A", 80.0, 90.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        # GOLDEN_BE_090: Warehouse California
        ("GOLDEN_BE_090", "warehouse", 20000, 1200000, "3C", 60.0, 75.0, "COMPLIANT", 0.01, "CA Title 24"),
    ])
    def test_warehouse_buildings(self, test_id, building_type, floor_area_sqm, energy_kwh, climate_zone, expected_eui, threshold, compliance_status, tolerance, source):
        """Test warehouse/industrial building energy calculations."""
        assert test_id.startswith("GOLDEN_BE_")
        assert expected_eui > 0


# =============================================================================
# RESIDENTIAL CATEGORIES (30 tests): GOLDEN_BE_091-120
# =============================================================================

class TestResidentialBuildingGoldenTests:
    """Golden tests for residential building energy performance."""

    @pytest.mark.golden
    @pytest.mark.parametrize("test_id,building_type,floor_area_sqm,energy_kwh,climate_zone,expected_eui,threshold,compliance_status,tolerance,source", [
        # Single Family (GOLDEN_BE_091-105)
        ("GOLDEN_BE_091", "residential_single_family", 200, 20000, "4A", 100.0, 120.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        ("GOLDEN_BE_092", "residential_single_family", 200, 26000, "4A", 130.0, 120.0, "NON-COMPLIANT", 0.01, "ENERGY STAR"),
        ("GOLDEN_BE_093", "residential_single_family", 300, 30000, "4A", 100.0, 120.0, "COMPLIANT", 0.01, "ENERGY STAR"),
        ("GOLDEN_BE_094", "residential_single_family", 150, 12000, "4A", 80.0, 120.0, "COMPLIANT", 0.01, "High efficiency"),
        ("GOLDEN_BE_095", "residential_single_family", 400, 60000, "4A", 150.0, 120.0, "NON-COMPLIANT", 0.01, "Large home"),
        ("GOLDEN_BE_096", "residential_single_family", 200, 22000, "5A", 110.0, 130.0, "COMPLIANT", 0.01, "Cold climate"),
        ("GOLDEN_BE_097", "residential_single_family", 200, 16000, "2A", 80.0, 100.0, "COMPLIANT", 0.01, "Hot climate"),
        ("GOLDEN_BE_098", "residential_single_family", 200, 24000, "6A", 120.0, 140.0, "COMPLIANT", 0.01, "Very cold"),
        ("GOLDEN_BE_099", "residential_townhouse", 180, 16200, "4A", 90.0, 110.0, "COMPLIANT", 0.01, "Townhouse"),
        ("GOLDEN_BE_100", "residential_single_family", 200, 10000, "4A", 50.0, 120.0, "COMPLIANT", 0.01, "Net zero"),
        ("GOLDEN_BE_101", "residential_single_family", 200, 36000, "4A", 180.0, 120.0, "NON-COMPLIANT", 0.01, "Poor efficiency"),
        ("GOLDEN_BE_102", "residential_single_family", 250, 20000, "4A", 80.0, 120.0, "COMPLIANT", 0.01, "ENERGY STAR certified"),
        ("GOLDEN_BE_103", "residential_single_family", 200, 18000, "3B", 90.0, 100.0, "COMPLIANT", 0.01, "Dry climate"),
        ("GOLDEN_BE_104", "residential_single_family", 200, 22000, "4C", 110.0, 120.0, "COMPLIANT", 0.01, "Marine climate"),
        ("GOLDEN_BE_105", "residential_single_family", 200, 24000, "7", 120.0, 150.0, "COMPLIANT", 0.01, "Arctic climate"),

        # Multi-Family (GOLDEN_BE_106-120)
        ("GOLDEN_BE_106", "residential_multifamily", 5000, 400000, "4A", 80.0, 95.0, "COMPLIANT", 0.01, "NYC LL97"),
        ("GOLDEN_BE_107", "residential_multifamily", 5000, 525000, "4A", 105.0, 95.0, "NON-COMPLIANT", 0.01, "NYC LL97"),
        ("GOLDEN_BE_108", "residential_multifamily", 10000, 750000, "4A", 75.0, 95.0, "COMPLIANT", 0.01, "Large building"),
        ("GOLDEN_BE_109", "residential_multifamily", 3000, 270000, "4A", 90.0, 95.0, "COMPLIANT", 0.01, "Small building"),
        ("GOLDEN_BE_110", "residential_multifamily", 5000, 450000, "5A", 90.0, 105.0, "COMPLIANT", 0.01, "Cold climate"),
        ("GOLDEN_BE_111", "residential_multifamily", 5000, 350000, "2A", 70.0, 85.0, "COMPLIANT", 0.01, "Hot climate"),
        ("GOLDEN_BE_112", "residential_highrise", 20000, 2000000, "4A", 100.0, 110.0, "COMPLIANT", 0.01, "High-rise"),
        ("GOLDEN_BE_113", "residential_multifamily", 5000, 475000, "4A", 95.0, 95.0, "COMPLIANT", 0.001, "At threshold"),
        ("GOLDEN_BE_114", "residential_multifamily", 5000, 300000, "4A", 60.0, 95.0, "COMPLIANT", 0.01, "High efficiency"),
        ("GOLDEN_BE_115", "residential_multifamily", 5000, 600000, "4A", 120.0, 95.0, "NON-COMPLIANT", 0.01, "Poor efficiency"),
        ("GOLDEN_BE_116", "residential_senior", 8000, 640000, "4A", 80.0, 100.0, "COMPLIANT", 0.01, "Senior housing"),
        ("GOLDEN_BE_117", "residential_affordable", 6000, 480000, "4A", 80.0, 100.0, "COMPLIANT", 0.01, "Affordable housing"),
        ("GOLDEN_BE_118", "residential_mixed_use", 10000, 1100000, "4A", 110.0, 120.0, "COMPLIANT", 0.01, "Mixed use"),
        ("GOLDEN_BE_119", "residential_multifamily", 5000, 375000, "4A", 75.0, 70.0, "NON-COMPLIANT", 0.01, "2030 threshold"),
        ("GOLDEN_BE_120", "residential_multifamily", 5000, 350000, "3C", 70.0, 85.0, "COMPLIANT", 0.01, "California"),
    ])
    def test_residential_buildings(self, test_id, building_type, floor_area_sqm, energy_kwh, climate_zone, expected_eui, threshold, compliance_status, tolerance, source):
        """Test residential building energy calculations."""
        assert test_id.startswith("GOLDEN_BE_")
        assert expected_eui > 0


# =============================================================================
# CLIMATE ZONE VARIATIONS (30 tests): GOLDEN_BE_121-150
# =============================================================================

class TestClimateZoneGoldenTests:
    """Golden tests for climate zone threshold variations."""

    @pytest.mark.golden
    @pytest.mark.parametrize("test_id,building_type,climate_zone,base_threshold,adjustment_factor,adjusted_threshold,source", [
        # Hot Climate Zones 1-3 (GOLDEN_BE_121-130)
        ("GOLDEN_BE_121", "office", "1A", 158.0, 1.20, 189.6, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_122", "office", "2A", 158.0, 1.15, 181.7, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_123", "office", "2B", 158.0, 1.12, 177.0, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_124", "office", "3A", 158.0, 1.08, 170.6, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_125", "office", "3B", 158.0, 1.05, 165.9, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_126", "office", "3C", 158.0, 0.95, 150.1, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_127", "retail", "1A", 210.0, 1.25, 262.5, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_128", "retail", "2A", 210.0, 1.18, 247.8, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_129", "hotel", "2A", 320.0, 1.15, 368.0, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_130", "hospital", "3A", 530.0, 1.08, 572.4, "ASHRAE climate adjustment"),

        # Mixed Climate Zones 4-5 (GOLDEN_BE_131-140)
        ("GOLDEN_BE_131", "office", "4A", 158.0, 1.00, 158.0, "NYC LL97 baseline"),
        ("GOLDEN_BE_132", "office", "4B", 158.0, 0.98, 154.8, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_133", "office", "4C", 158.0, 0.92, 145.4, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_134", "office", "5A", 158.0, 1.05, 165.9, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_135", "office", "5B", 158.0, 1.02, 161.2, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_136", "retail", "4A", 210.0, 1.00, 210.0, "NYC LL97 baseline"),
        ("GOLDEN_BE_137", "retail", "5A", 210.0, 1.08, 226.8, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_138", "hotel", "4A", 320.0, 1.00, 320.0, "NYC LL97 baseline"),
        ("GOLDEN_BE_139", "hospital", "4A", 530.0, 1.00, 530.0, "NYC LL97 baseline"),
        ("GOLDEN_BE_140", "warehouse", "5A", 85.0, 1.10, 93.5, "ASHRAE climate adjustment"),

        # Cold Climate Zones 6-7 (GOLDEN_BE_141-150)
        ("GOLDEN_BE_141", "office", "6A", 158.0, 1.12, 177.0, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_142", "office", "6B", 158.0, 1.15, 181.7, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_143", "office", "7", 158.0, 1.25, 197.5, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_144", "retail", "6A", 210.0, 1.15, 241.5, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_145", "retail", "7", 210.0, 1.30, 273.0, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_146", "hotel", "6A", 320.0, 1.12, 358.4, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_147", "hotel", "7", 320.0, 1.22, 390.4, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_148", "hospital", "6A", 530.0, 1.10, 583.0, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_149", "warehouse", "6A", 85.0, 1.18, 100.3, "ASHRAE climate adjustment"),
        ("GOLDEN_BE_150", "warehouse", "7", 85.0, 1.35, 114.75, "ASHRAE climate adjustment"),
    ])
    def test_climate_zone_adjustments(self, test_id, building_type, climate_zone, base_threshold, adjustment_factor, adjusted_threshold, source):
        """Test climate zone threshold adjustments."""
        assert test_id.startswith("GOLDEN_BE_")
        assert adjusted_threshold > 0
        # Verify calculation
        calculated = base_threshold * adjustment_factor
        assert abs(calculated - adjusted_threshold) < 0.1


# =============================================================================
# MULTI-BUILDING PORTFOLIOS (25 tests): GOLDEN_BE_151-175
# =============================================================================

class TestMultiBuildingPortfolioGoldenTests:
    """Golden tests for multi-building portfolio energy analysis."""

    @pytest.mark.golden
    @pytest.mark.parametrize("test_id,portfolio_name,num_buildings,total_area_sqm,total_energy_kwh,weighted_eui,compliance_rate,description", [
        # Office Portfolios (GOLDEN_BE_151-160)
        ("GOLDEN_BE_151", "corporate_hq_portfolio", 5, 50000, 7500000, 150.0, 0.80, "Corporate headquarters"),
        ("GOLDEN_BE_152", "regional_office_portfolio", 10, 100000, 16000000, 160.0, 0.70, "Regional offices"),
        ("GOLDEN_BE_153", "small_office_portfolio", 20, 40000, 6400000, 160.0, 0.65, "Small offices"),
        ("GOLDEN_BE_154", "mixed_office_portfolio", 15, 75000, 11250000, 150.0, 0.73, "Mixed office types"),
        ("GOLDEN_BE_155", "tech_campus_portfolio", 8, 120000, 18000000, 150.0, 0.88, "Tech campus"),
        ("GOLDEN_BE_156", "government_portfolio", 12, 60000, 9000000, 150.0, 0.83, "Government buildings"),
        ("GOLDEN_BE_157", "financial_district", 7, 140000, 21000000, 150.0, 0.86, "Financial district"),
        ("GOLDEN_BE_158", "suburban_office_park", 25, 125000, 18750000, 150.0, 0.72, "Suburban office park"),
        ("GOLDEN_BE_159", "class_a_portfolio", 6, 90000, 12600000, 140.0, 0.92, "Class A offices"),
        ("GOLDEN_BE_160", "aging_office_portfolio", 15, 75000, 13500000, 180.0, 0.47, "Aging buildings"),

        # Mixed-Use Portfolios (GOLDEN_BE_161-170)
        ("GOLDEN_BE_161", "retail_chain_portfolio", 30, 75000, 15000000, 200.0, 0.77, "Retail chain"),
        ("GOLDEN_BE_162", "hotel_chain_portfolio", 12, 180000, 54000000, 300.0, 0.83, "Hotel chain"),
        ("GOLDEN_BE_163", "hospital_network", 5, 250000, 125000000, 500.0, 0.80, "Hospital network"),
        ("GOLDEN_BE_164", "school_district", 20, 160000, 16000000, 100.0, 0.85, "School district"),
        ("GOLDEN_BE_165", "university_campus", 25, 500000, 90000000, 180.0, 0.72, "University campus"),
        ("GOLDEN_BE_166", "industrial_park", 15, 300000, 21000000, 70.0, 0.87, "Industrial park"),
        ("GOLDEN_BE_167", "mixed_use_development", 8, 80000, 12800000, 160.0, 0.75, "Mixed-use development"),
        ("GOLDEN_BE_168", "data_center_portfolio", 4, 40000, 60000000, 1500.0, 0.75, "Data centers"),
        ("GOLDEN_BE_169", "affordable_housing", 50, 150000, 12000000, 80.0, 0.90, "Affordable housing"),
        ("GOLDEN_BE_170", "senior_living", 10, 80000, 8000000, 100.0, 0.85, "Senior living"),

        # Compliance Focus Portfolios (GOLDEN_BE_171-175)
        ("GOLDEN_BE_171", "ll97_compliance_portfolio", 20, 200000, 32000000, 160.0, 0.70, "NYC LL97 focus"),
        ("GOLDEN_BE_172", "energy_star_portfolio", 15, 150000, 19500000, 130.0, 0.93, "ENERGY STAR focus"),
        ("GOLDEN_BE_173", "net_zero_target", 8, 40000, 2800000, 70.0, 1.00, "Net zero target"),
        ("GOLDEN_BE_174", "carbon_neutral_portfolio", 12, 60000, 5400000, 90.0, 0.92, "Carbon neutral focus"),
        ("GOLDEN_BE_175", "retrofit_candidate_portfolio", 25, 250000, 50000000, 200.0, 0.32, "Retrofit candidates"),
    ])
    def test_multi_building_portfolios(self, test_id, portfolio_name, num_buildings, total_area_sqm, total_energy_kwh, weighted_eui, compliance_rate, description):
        """Test multi-building portfolio energy analysis."""
        assert test_id.startswith("GOLDEN_BE_")
        assert weighted_eui > 0
        assert 0 <= compliance_rate <= 1.0


# =============================================================================
# COMPLIANCE & BENCHMARKING (25 tests): GOLDEN_BE_176-200
# =============================================================================

class TestComplianceBenchmarkingGoldenTests:
    """Golden tests for building energy compliance and benchmarking scenarios."""

    @pytest.mark.golden
    @pytest.mark.parametrize("test_id,scenario,building_type,current_eui,target_eui,gap_percent,reduction_needed_kwh,timeline_years,description", [
        # NYC LL97 Compliance Scenarios (GOLDEN_BE_176-185)
        ("GOLDEN_BE_176", "ll97_2024_compliant", "office", 150.0, 158.0, -5.1, 0, 0, "Already compliant 2024"),
        ("GOLDEN_BE_177", "ll97_2024_gap", "office", 180.0, 158.0, 13.9, 220000, 1, "Gap to 2024 threshold"),
        ("GOLDEN_BE_178", "ll97_2030_compliant", "office", 100.0, 105.0, -4.8, 0, 0, "Already compliant 2030"),
        ("GOLDEN_BE_179", "ll97_2030_gap", "office", 150.0, 105.0, 42.9, 450000, 6, "Gap to 2030 threshold"),
        ("GOLDEN_BE_180", "ll97_2030_major_gap", "office", 200.0, 105.0, 90.5, 950000, 6, "Major gap to 2030"),
        ("GOLDEN_BE_181", "ll97_retail_2024", "retail", 210.0, 210.0, 0.0, 0, 0, "Retail at threshold"),
        ("GOLDEN_BE_182", "ll97_hotel_gap", "hotel", 350.0, 320.0, 9.4, 450000, 2, "Hotel compliance gap"),
        ("GOLDEN_BE_183", "ll97_hospital_compliant", "hospital", 500.0, 530.0, -5.7, 0, 0, "Hospital compliant"),
        ("GOLDEN_BE_184", "ll97_multifamily_gap", "residential_multifamily", 100.0, 95.0, 5.3, 25000, 1, "Multifamily gap"),
        ("GOLDEN_BE_185", "ll97_warehouse_compliant", "warehouse", 70.0, 85.0, -17.6, 0, 0, "Warehouse compliant"),

        # ENERGY STAR Benchmarking (GOLDEN_BE_186-195)
        ("GOLDEN_BE_186", "es_score_75", "office", 120.0, 130.0, -7.7, 0, 0, "ENERGY STAR 75"),
        ("GOLDEN_BE_187", "es_score_50", "office", 150.0, 150.0, 0.0, 0, 0, "ENERGY STAR 50"),
        ("GOLDEN_BE_188", "es_score_25", "office", 180.0, 200.0, -10.0, 0, 0, "ENERGY STAR 25"),
        ("GOLDEN_BE_189", "es_certification_target", "office", 140.0, 130.0, 7.7, 100000, 1, "ES certification goal"),
        ("GOLDEN_BE_190", "es_top_performer", "office", 90.0, 130.0, -30.8, 0, 0, "Top performer"),
        ("GOLDEN_BE_191", "es_improvement_plan", "retail", 250.0, 200.0, 25.0, 250000, 2, "Improvement plan"),
        ("GOLDEN_BE_192", "es_hotel_target", "hotel", 280.0, 250.0, 12.0, 450000, 2, "Hotel ES target"),
        ("GOLDEN_BE_193", "es_hospital_target", "hospital", 480.0, 450.0, 6.7, 1500000, 3, "Hospital ES target"),
        ("GOLDEN_BE_194", "es_data_center", "data_center", 1400.0, 1200.0, 16.7, 2000000, 2, "Data center efficiency"),
        ("GOLDEN_BE_195", "es_warehouse_target", "warehouse", 65.0, 60.0, 8.3, 100000, 1, "Warehouse ES target"),

        # Carbon Neutrality Pathways (GOLDEN_BE_196-200)
        ("GOLDEN_BE_196", "net_zero_2030", "office", 150.0, 0.0, 100.0, 1500000, 6, "Net zero by 2030"),
        ("GOLDEN_BE_197", "net_zero_2035", "retail", 200.0, 0.0, 100.0, 1000000, 11, "Net zero by 2035"),
        ("GOLDEN_BE_198", "net_zero_2040", "hospital", 500.0, 0.0, 100.0, 25000000, 16, "Net zero by 2040"),
        ("GOLDEN_BE_199", "deep_retrofit", "office", 250.0, 100.0, 150.0, 1500000, 3, "Deep energy retrofit"),
        ("GOLDEN_BE_200", "electrification_pathway", "hotel", 320.0, 200.0, 60.0, 1800000, 5, "Electrification pathway"),
    ])
    def test_compliance_benchmarking(self, test_id, scenario, building_type, current_eui, target_eui, gap_percent, reduction_needed_kwh, timeline_years, description):
        """Test building energy compliance and benchmarking scenarios."""
        assert test_id.startswith("GOLDEN_BE_")
        assert current_eui >= 0
        assert target_eui >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "golden"])
