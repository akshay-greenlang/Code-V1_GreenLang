# -*- coding: utf-8 -*-
"""
Unit tests for PACK-042 Compliance Reference Calculations
============================================================

Validates reference emission calculations for Scope 3 categories:
spend-based, average-data, transport, business travel, commuting, waste,
double-counting adjustments, uncertainty, DQR, and footprint percentage.

Coverage target: 85%+
Total tests: ~45
"""

from decimal import Decimal
import math

import pytest

from tests.conftest import (
    SCOPE3_CATEGORIES,
    UPSTREAM_CATEGORIES,
    DOWNSTREAM_CATEGORIES,
    compute_provenance_hash,
)


# =============================================================================
# Spend-Based Calculation Tests
# =============================================================================


class TestSpendBasedCalculation:
    """Test spend-based emission calculations (Tier 1)."""

    def test_office_supplies_spend_based(self):
        """$1M office supplies x EEIO factor = expected tCO2e."""
        spend_eur = Decimal("1000000")
        eeio_factor_kgco2e_per_eur = Decimal("0.25")  # wholesale_trade
        emissions_kg = spend_eur * eeio_factor_kgco2e_per_eur
        emissions_tco2e = emissions_kg / Decimal("1000")
        assert emissions_tco2e == Decimal("250.0")

    def test_chemicals_spend_based(self):
        """EUR 500k chemicals x EEIO factor."""
        spend_eur = Decimal("500000")
        eeio_factor = Decimal("1.55")  # chemicals_pharmaceuticals
        emissions_tco2e = spend_eur * eeio_factor / Decimal("1000")
        assert emissions_tco2e == Decimal("775.0")

    def test_it_services_spend_based(self):
        """EUR 200k IT services x EEIO factor."""
        spend_eur = Decimal("200000")
        eeio_factor = Decimal("0.18")  # it_services
        emissions_tco2e = spend_eur * eeio_factor / Decimal("1000")
        assert emissions_tco2e == Decimal("36.0")

    def test_metals_spend_based(self):
        """EUR 2.5M basic metals x EEIO factor."""
        spend_eur = Decimal("2500000")
        eeio_factor = Decimal("2.95")  # basic_metals
        emissions_tco2e = spend_eur * eeio_factor / Decimal("1000")
        assert emissions_tco2e == Decimal("7375.0")

    def test_spend_formula_is_activity_times_factor(self):
        """Core formula: E = Spend (EUR) * EF (kgCO2e/EUR) / 1000."""
        spend = Decimal("1000000")
        ef = Decimal("0.50")
        result = spend * ef / Decimal("1000")
        assert result == Decimal("500.0")


# =============================================================================
# Average-Data Calculation Tests
# =============================================================================


class TestAverageDataCalculation:
    """Test average-data emission calculations (Tier 2)."""

    def test_steel_average_data(self):
        """1000 tonnes steel x DEFRA factor."""
        quantity_tonnes = Decimal("1000")
        ef_tco2e_per_tonne = Decimal("1.85")  # Approximate DEFRA steel
        emissions_tco2e = quantity_tonnes * ef_tco2e_per_tonne
        assert emissions_tco2e == Decimal("1850.0")

    def test_cement_average_data(self):
        """500 tonnes cement x emission factor."""
        quantity_tonnes = Decimal("500")
        ef_tco2e_per_tonne = Decimal("0.62")
        emissions_tco2e = quantity_tonnes * ef_tco2e_per_tonne
        assert emissions_tco2e == Decimal("310.0")

    def test_average_formula_is_quantity_times_factor(self):
        """Core formula: E = Quantity (units) * EF (tCO2e/unit)."""
        quantity = Decimal("5000")
        ef = Decimal("0.5")
        result = quantity * ef
        assert result == Decimal("2500.0")


# =============================================================================
# Transport Calculation Tests
# =============================================================================


class TestTransportCalculation:
    """Test transport emission calculations (Cat 4/Cat 9)."""

    def test_road_freight_distance_based(self):
        """50,000 tonne-km road freight x GLEC factor."""
        tonne_km = Decimal("50000")
        ef_kgco2e_per_tkm = Decimal("0.0621")  # Road freight average
        emissions_kg = tonne_km * ef_kgco2e_per_tkm
        emissions_tco2e = emissions_kg / Decimal("1000")
        assert emissions_tco2e == Decimal("3.105")

    def test_ocean_freight(self):
        """100,000 tonne-km ocean freight x factor."""
        tonne_km = Decimal("100000")
        ef_kgco2e_per_tkm = Decimal("0.0160")  # Container ship
        emissions_tco2e = tonne_km * ef_kgco2e_per_tkm / Decimal("1000")
        assert emissions_tco2e == Decimal("1.600")

    def test_air_freight(self):
        """10,000 tonne-km air freight x factor."""
        tonne_km = Decimal("10000")
        ef_kgco2e_per_tkm = Decimal("0.6023")  # Air cargo
        emissions_tco2e = tonne_km * ef_kgco2e_per_tkm / Decimal("1000")
        expected = Decimal("6.023")
        assert emissions_tco2e == expected

    def test_transport_formula(self):
        """E = distance (tonne-km) * EF (kgCO2e/tonne-km) / 1000."""
        distance = Decimal("250000")
        ef = Decimal("0.0621")
        result = distance * ef / Decimal("1000")
        assert result > 0


# =============================================================================
# Business Travel Calculation Tests
# =============================================================================


class TestBusinessTravelCalculation:
    """Test business travel emission calculations (Cat 6)."""

    def test_air_travel_short_haul(self):
        """100,000 passenger-km short-haul x ICAO factor."""
        pkm = Decimal("100000")
        ef_kgco2e_per_pkm = Decimal("0.1557")  # Short-haul average with RF
        emissions_tco2e = pkm * ef_kgco2e_per_pkm / Decimal("1000")
        assert emissions_tco2e == Decimal("15.570")

    def test_air_travel_long_haul(self):
        """200,000 passenger-km long-haul x factor."""
        pkm = Decimal("200000")
        ef_kgco2e_per_pkm = Decimal("0.1024")  # Long-haul economy
        emissions_tco2e = pkm * ef_kgco2e_per_pkm / Decimal("1000")
        assert emissions_tco2e == Decimal("20.480")

    def test_rail_travel(self):
        """50,000 passenger-km rail x factor."""
        pkm = Decimal("50000")
        ef_kgco2e_per_pkm = Decimal("0.0064")  # European rail average
        emissions_tco2e = pkm * ef_kgco2e_per_pkm / Decimal("1000")
        assert emissions_tco2e == Decimal("0.320")

    def test_hotel_stays(self):
        """500 hotel nights x factor."""
        nights = Decimal("500")
        ef_kgco2e_per_night = Decimal("20.6")  # Average hotel
        emissions_tco2e = nights * ef_kgco2e_per_night / Decimal("1000")
        assert emissions_tco2e == Decimal("10.300")


# =============================================================================
# Employee Commuting Calculation Tests
# =============================================================================


class TestEmployeeCommuting:
    """Test employee commuting emission calculations (Cat 7)."""

    def test_commuting_average_method(self):
        """500 employees x average commute factors."""
        employees = Decimal("500")
        avg_commute_km_per_day = Decimal("30")
        working_days = Decimal("230")
        car_share_pct = Decimal("0.60")
        ef_car_kgco2e_per_km = Decimal("0.171")

        total_km = employees * avg_commute_km_per_day * working_days * car_share_pct
        emissions_tco2e = total_km * ef_car_kgco2e_per_km / Decimal("1000")
        assert emissions_tco2e > 0
        assert emissions_tco2e < Decimal("500")

    def test_remote_work_reduction(self):
        """Remote work reduces commuting emissions."""
        base_emissions = Decimal("350")
        remote_pct = Decimal("0.40")
        reduced = base_emissions * (1 - remote_pct)
        assert reduced == Decimal("210.0")


# =============================================================================
# Waste Calculation Tests
# =============================================================================


class TestWasteCalculation:
    """Test waste emission calculations (Cat 5)."""

    def test_landfill_waste(self):
        """100 tonnes landfill x DEFRA factor."""
        tonnes = Decimal("100")
        ef_tco2e_per_tonne = Decimal("0.586")  # DEFRA mixed waste landfill
        emissions_tco2e = tonnes * ef_tco2e_per_tonne
        assert emissions_tco2e == Decimal("58.6")

    def test_recycling_lower_than_landfill(self):
        ef_landfill = Decimal("0.586")
        ef_recycling = Decimal("0.021")
        assert ef_recycling < ef_landfill

    def test_incineration_waste(self):
        """50 tonnes incineration x factor."""
        tonnes = Decimal("50")
        ef = Decimal("0.919")  # DEFRA incineration
        result = tonnes * ef
        assert result == Decimal("45.950")


# =============================================================================
# Double-Counting Adjustment Tests
# =============================================================================


class TestDoubleCountingAdjustment:
    """Test double-counting adjustment calculations."""

    def test_cat1_cat4_overlap_adjustment(self):
        cat1_total = Decimal("28500")
        cat4_total = Decimal("5100")
        overlap = Decimal("450")
        adjusted_cat4 = cat4_total - overlap
        adjusted_total = cat1_total + adjusted_cat4
        assert adjusted_total == cat1_total + cat4_total - overlap

    def test_proportional_allocation(self):
        overlap = Decimal("600")
        cat1_share_pct = Decimal("0.65")
        cat4_share_pct = Decimal("0.35")
        cat1_adjustment = overlap * cat1_share_pct
        cat4_adjustment = overlap * cat4_share_pct
        assert cat1_adjustment + cat4_adjustment == overlap

    def test_net_effect_on_total(self):
        original_total = Decimal("61430")
        adjustment = Decimal("-570")
        adjusted_total = original_total + adjustment
        assert adjusted_total == Decimal("60860")


# =============================================================================
# Uncertainty Calculation Tests
# =============================================================================


class TestUncertaintyCalculation:
    """Test uncertainty range calculations."""

    def test_spend_based_uncertainty_range(self):
        """Spend-based: point estimate +/- 100%."""
        point = Decimal("28500")
        uncertainty_pct = Decimal("100")
        lower = point * (1 - uncertainty_pct / 100)
        upper = point * (1 + uncertainty_pct / 100)
        assert lower == Decimal("0")
        assert upper == Decimal("57000")

    def test_supplier_specific_uncertainty_range(self):
        """Supplier-specific: point estimate +/- 10%."""
        point = Decimal("28500")
        uncertainty_pct = Decimal("10")
        lower = point * (1 - uncertainty_pct / 100)
        upper = point * (1 + uncertainty_pct / 100)
        assert lower == Decimal("25650.0")
        assert upper == Decimal("31350.0")

    def test_quadrature_aggregation(self):
        """Root-sum-square aggregation of independent uncertainties."""
        # Two categories, independent
        e1, u1 = 28500.0, 35.0  # Cat 1: 35% uncertainty
        e2, u2 = 5100.0, 45.0   # Cat 4: 45% uncertainty
        combined_variance = (e1 * u1/100)**2 + (e2 * u2/100)**2
        combined_std = math.sqrt(combined_variance)
        total_emissions = e1 + e2
        combined_uncertainty = combined_std / total_emissions * 100
        assert combined_uncertainty < max(u1, u2)


# =============================================================================
# DQR Calculation Tests
# =============================================================================


class TestDQRCalculation:
    """Test DQR calculation from 5 DQIs."""

    def test_weighted_average_dqr(self):
        """DQR = weighted average of 5 DQIs."""
        dqi = {
            "technological": Decimal("3.0"),
            "temporal": Decimal("2.5"),
            "geographical": Decimal("3.0"),
            "completeness": Decimal("3.5"),
            "reliability": Decimal("4.0"),
        }
        weights = [Decimal("0.30"), Decimal("0.20"), Decimal("0.20"), Decimal("0.15"), Decimal("0.15")]
        dqr = sum(w * v for w, v in zip(weights, dqi.values()))
        assert Decimal("2.5") < dqr < Decimal("4.0")

    def test_dqr_equal_weights(self):
        """Equal weights: simple average."""
        scores = [Decimal("3"), Decimal("3"), Decimal("3"), Decimal("3"), Decimal("3")]
        avg = sum(scores) / len(scores)
        assert avg == Decimal("3")


# =============================================================================
# Scope 3 Percentage of Total Footprint Tests
# =============================================================================


class TestScope3PercentOfTotal:
    """Test Scope 3 as % of total footprint calculation."""

    def test_scope3_pct_calculation(self):
        scope1 = Decimal("12000")
        scope2 = Decimal("5200")
        scope3 = Decimal("61430")
        total = scope1 + scope2 + scope3
        s3_pct = scope3 / total * Decimal("100")
        assert s3_pct > Decimal("75"), "Scope 3 should be > 75% for manufacturing"

    def test_scope3_multiples_of_scope12(self):
        scope12 = Decimal("17200")
        scope3 = Decimal("61430")
        multiple = scope3 / scope12
        assert multiple > Decimal("3"), "Scope 3 should be > 3x Scope 1+2"

    def test_footprint_sum(self):
        scope1 = Decimal("12000")
        scope2 = Decimal("5200")
        scope3 = Decimal("61430")
        total = scope1 + scope2 + scope3
        assert total == Decimal("78630")
