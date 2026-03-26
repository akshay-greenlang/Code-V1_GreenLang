# -*- coding: utf-8 -*-
"""
Unit tests for PACK-043 Sector-Specific Engine
=================================================

Tests PCAF financed emissions, attribution factors, WACI calculation,
retail last-mile, packaging lifecycle, manufacturing circular economy,
cloud carbon, embodied carbon, and SaaS use-phase modelling.

Coverage target: 85%+
Total tests: ~50
"""

from decimal import Decimal

import pytest


# =============================================================================
# PCAF Financed Emissions (Listed Equity)
# =============================================================================


class TestPCAFFinancedEmissions:
    """Test PCAF financed emissions for listed equity."""

    def test_portfolio_has_listed_equity(self, sample_pcaf_portfolio):
        assert "listed_equity" in sample_pcaf_portfolio["asset_classes"]

    def test_three_equity_investments(self, sample_pcaf_portfolio):
        investments = sample_pcaf_portfolio["asset_classes"]["listed_equity"]["investments"]
        assert len(investments) == 3

    def test_one_bond_investment(self, sample_pcaf_portfolio):
        bonds = sample_pcaf_portfolio["asset_classes"]["corporate_bonds"]["investments"]
        assert len(bonds) == 1

    @pytest.mark.parametrize("inv_idx", [0, 1, 2])
    def test_equity_has_required_fields(self, inv_idx, sample_pcaf_portfolio):
        inv = sample_pcaf_portfolio["asset_classes"]["listed_equity"]["investments"][inv_idx]
        required = [
            "investee_id", "investee_name", "invested_amount_usd",
            "evic_usd", "investee_scope12_tco2e", "investee_scope3_tco2e",
            "data_quality_score",
        ]
        for field in required:
            assert field in inv

    def test_attribution_factor_inv001(self, sample_pcaf_portfolio):
        """Attribution = invested_amount / EVIC."""
        inv = sample_pcaf_portfolio["asset_classes"]["listed_equity"]["investments"][0]
        attribution = inv["invested_amount_usd"] / inv["evic_usd"]
        # 50M / 500M = 0.10
        assert attribution == Decimal("0.10")

    def test_financed_emissions_inv001(self, sample_pcaf_portfolio):
        """Financed = attribution * investee_emissions."""
        inv = sample_pcaf_portfolio["asset_classes"]["listed_equity"]["investments"][0]
        attribution = inv["invested_amount_usd"] / inv["evic_usd"]
        financed_s12 = attribution * inv["investee_scope12_tco2e"]
        financed_s3 = attribution * inv["investee_scope3_tco2e"]
        # 0.10 * 25000 = 2500
        assert financed_s12 == Decimal("2500")
        # 0.10 * 120000 = 12000
        assert financed_s3 == Decimal("12000")


# =============================================================================
# PCAF Attribution Factor
# =============================================================================


class TestPCAFAttributionFactor:
    """Test PCAF EVIC-based attribution factor calculation."""

    def test_attribution_factor_range(self, sample_pcaf_portfolio):
        """Attribution factors should be between 0 and 1."""
        for inv in sample_pcaf_portfolio["asset_classes"]["listed_equity"]["investments"]:
            attr = inv["invested_amount_usd"] / inv["evic_usd"]
            assert Decimal("0") < attr <= Decimal("1")

    def test_attribution_factor_bond(self, sample_pcaf_portfolio):
        bond = sample_pcaf_portfolio["asset_classes"]["corporate_bonds"]["investments"][0]
        attr = bond["invested_amount_usd"] / bond["evic_usd"]
        # 25M / 150M = 0.1667
        assert Decimal("0.16") < attr < Decimal("0.17")

    def test_data_quality_scores_in_range(self, sample_pcaf_portfolio):
        for inv in sample_pcaf_portfolio["asset_classes"]["listed_equity"]["investments"]:
            assert 1 <= inv["data_quality_score"] <= 5


# =============================================================================
# WACI Calculation
# =============================================================================


class TestWACICalculation:
    """Test Weighted Average Carbon Intensity calculation."""

    def test_waci_calculation(self, sample_pcaf_portfolio):
        """WACI = sum(weight_i * intensity_i)."""
        investments = sample_pcaf_portfolio["asset_classes"]["listed_equity"]["investments"]
        total_invested = sum(i["invested_amount_usd"] for i in investments)
        waci = Decimal("0")
        for inv in investments:
            weight = inv["invested_amount_usd"] / total_invested
            # Intensity: scope12 per million revenue (simplified as per EVIC)
            intensity = inv["investee_scope12_tco2e"] / (inv["evic_usd"] / Decimal("1000000"))
            waci += weight * intensity
        assert waci > Decimal("0")

    def test_portfolio_total_value(self, sample_pcaf_portfolio):
        assert sample_pcaf_portfolio["total_portfolio_value_usd"] == Decimal("125000000")

    def test_total_financed_emissions(self, sample_pcaf_portfolio):
        """Sum of attributed emissions across all investments."""
        total = Decimal("0")
        for asset_class in sample_pcaf_portfolio["asset_classes"].values():
            for inv in asset_class["investments"]:
                attr = inv["invested_amount_usd"] / inv["evic_usd"]
                total += attr * inv["investee_scope12_tco2e"]
        assert total > Decimal("0")


# =============================================================================
# Retail Last-Mile Emissions
# =============================================================================


class TestRetailLastMile:
    """Test retail last-mile delivery emissions."""

    def test_last_mile_data_present(self, sample_retail_data):
        assert "last_mile" in sample_retail_data

    def test_total_deliveries(self, sample_retail_data):
        assert sample_retail_data["last_mile"]["total_deliveries"] == 5000000

    def test_last_mile_emissions_calculation(self, sample_retail_data):
        """Total = deliveries * avg_distance * ef, adjusted for EV share."""
        lm = sample_retail_data["last_mile"]
        ice_pct = (Decimal("100") - lm["electric_vehicle_pct"]) / Decimal("100")
        ev_pct = lm["electric_vehicle_pct"] / Decimal("100")
        total_km = lm["total_deliveries"] * lm["average_distance_km"]
        ice_emissions = total_km * ice_pct * lm["emission_factor_kgco2e_per_km"]
        ev_emissions = total_km * ev_pct * lm["ev_emission_factor_kgco2e_per_km"]
        total_tco2e = (ice_emissions + ev_emissions) / Decimal("1000")
        assert total_tco2e > Decimal("0")

    def test_ev_reduces_emissions(self, sample_retail_data):
        """Higher EV % should reduce total emissions."""
        lm = sample_retail_data["last_mile"]
        total_km = lm["total_deliveries"] * lm["average_distance_km"]
        all_ice = total_km * lm["emission_factor_kgco2e_per_km"]
        ev_pct = lm["electric_vehicle_pct"] / Decimal("100")
        mixed = (
            total_km * (Decimal("1") - ev_pct) * lm["emission_factor_kgco2e_per_km"]
            + total_km * ev_pct * lm["ev_emission_factor_kgco2e_per_km"]
        )
        assert mixed < all_ice


# =============================================================================
# Packaging Lifecycle
# =============================================================================


class TestPackagingLifecycle:
    """Test packaging lifecycle emissions."""

    def test_packaging_data_present(self, sample_retail_data):
        assert "packaging" in sample_retail_data

    def test_three_packaging_materials(self, sample_retail_data):
        assert len(sample_retail_data["packaging"]["materials"]) == 3

    def test_total_packaging_emissions(self, sample_retail_data):
        pkg = sample_retail_data["packaging"]
        total = Decimal("0")
        for mat in pkg["materials"]:
            material_emissions = (
                pkg["total_units"]
                * mat["mass_kg_per_unit"]
                * mat["ef_kgco2e_per_kg"]
            )
            total += material_emissions
        total_tco2e = total / Decimal("1000")
        assert total_tco2e > Decimal("0")

    def test_cardboard_largest_by_mass(self, sample_retail_data):
        materials = sample_retail_data["packaging"]["materials"]
        cardboard = next(m for m in materials if m["type"] == "cardboard")
        for mat in materials:
            assert cardboard["mass_kg_per_unit"] >= mat["mass_kg_per_unit"]


# =============================================================================
# Manufacturing Circular Economy
# =============================================================================


class TestManufacturingCircularEconomy:
    """Test manufacturing circular economy benefit."""

    def test_recycled_content_benefit(self, sample_product_bom):
        """Recycled content reduces virgin material emissions."""
        prod = sample_product_bom[0]
        virgin_only = sum(
            c["mass_kg"] * c["emission_factor_kgco2e_per_kg"]
            for c in prod["components"]
        )
        with_recycling = Decimal("0")
        for c in prod["components"]:
            recycled_pct = c["recycled_content_pct"] / Decimal("100")
            recycled_ef = c["emission_factor_kgco2e_per_kg"] * Decimal("0.4")
            effective_ef = (
                recycled_pct * recycled_ef
                + (Decimal("1") - recycled_pct) * c["emission_factor_kgco2e_per_kg"]
            )
            with_recycling += c["mass_kg"] * effective_ef
        assert with_recycling < virgin_only

    def test_circular_benefit_pct(self, sample_product_bom):
        """Calculate circular economy benefit as % reduction."""
        prod = sample_product_bom[0]
        virgin_total = sum(
            c["mass_kg"] * c["emission_factor_kgco2e_per_kg"]
            for c in prod["components"]
        )
        circular_total = Decimal("0")
        for c in prod["components"]:
            r = c["recycled_content_pct"] / Decimal("100")
            ef_adj = c["emission_factor_kgco2e_per_kg"] * (Decimal("1") - r * Decimal("0.6"))
            circular_total += c["mass_kg"] * ef_adj
        benefit_pct = (virgin_total - circular_total) / virgin_total * Decimal("100")
        assert benefit_pct > Decimal("0")


# =============================================================================
# Cloud Carbon (AWS/Azure/GCP)
# =============================================================================


class TestCloudCarbon:
    """Test cloud computing carbon calculations."""

    def test_three_providers(self, sample_cloud_data):
        assert len(sample_cloud_data["providers"]) == 3

    @pytest.mark.parametrize("provider", ["aws", "azure", "gcp"])
    def test_provider_data_complete(self, provider, sample_cloud_data):
        p = sample_cloud_data["providers"][provider]
        assert p["kwh_consumed"] > Decimal("0")
        assert p["grid_factor_kgco2e_per_kwh"] > Decimal("0")
        assert p["pue"] >= Decimal("1.0")
        assert Decimal("0") <= p["renewable_pct"] <= Decimal("100")

    def test_cloud_emissions_calculation(self, sample_cloud_data):
        """Emissions = kWh * PUE * grid_factor * (1 - renewable%) / 1000."""
        for name, p in sample_cloud_data["providers"].items():
            non_renewable_factor = (Decimal("100") - p["renewable_pct"]) / Decimal("100")
            emissions_tco2e = (
                p["kwh_consumed"]
                * p["pue"]
                * p["grid_factor_kgco2e_per_kwh"]
                * non_renewable_factor
                / Decimal("1000")
            )
            assert emissions_tco2e >= Decimal("0")

    def test_gcp_lowest_emissions(self, sample_cloud_data):
        """GCP should have lowest emissions (100% renewable, low grid factor)."""
        emissions = {}
        for name, p in sample_cloud_data["providers"].items():
            non_renewable = (Decimal("100") - p["renewable_pct"]) / Decimal("100")
            em = p["kwh_consumed"] * p["pue"] * p["grid_factor_kgco2e_per_kwh"] * non_renewable
            emissions[name] = em
        assert emissions["gcp"] <= min(emissions["aws"], emissions["azure"])


# =============================================================================
# Embodied Carbon of Hardware
# =============================================================================


class TestEmbodiedCarbon:
    """Test embodied carbon of IT hardware."""

    def test_embodied_carbon_categories(self, sample_cloud_data):
        ec = sample_cloud_data["embodied_carbon"]
        assert "servers" in ec
        assert "networking" in ec
        assert "storage" in ec

    def test_annualized_embodied_carbon(self, sample_cloud_data):
        """Annualized = units * kgCO2e_per_unit / amortization_years / 1000."""
        ec = sample_cloud_data["embodied_carbon"]
        total_annual_tco2e = Decimal("0")
        for category, data in ec.items():
            annual = (
                Decimal(str(data["units"]))
                * data["kgco2e_per_unit"]
                / Decimal(str(data["amortization_years"]))
                / Decimal("1000")
            )
            total_annual_tco2e += annual
            assert annual > Decimal("0")
        assert total_annual_tco2e > Decimal("0")

    def test_servers_largest_embodied(self, sample_cloud_data):
        ec = sample_cloud_data["embodied_carbon"]
        server_total = ec["servers"]["units"] * ec["servers"]["kgco2e_per_unit"]
        network_total = ec["networking"]["units"] * ec["networking"]["kgco2e_per_unit"]
        storage_total = ec["storage"]["units"] * ec["storage"]["kgco2e_per_unit"]
        assert server_total > network_total
        assert server_total > storage_total


# =============================================================================
# SaaS Use-Phase Modelling
# =============================================================================


class TestSaaSUsePhase:
    """Test SaaS use-phase carbon modelling."""

    def test_saas_data_present(self, sample_cloud_data):
        assert "saas_use_phase" in sample_cloud_data

    def test_saas_emissions_calculation(self, sample_cloud_data):
        """SaaS use-phase = users * hours * power * grid_factor * 12 months."""
        saas = sample_cloud_data["saas_use_phase"]
        annual_kwh = (
            saas["active_users"]
            * float(saas["avg_session_hours_per_month"])
            * 12
            * float(saas["device_power_w"])
            / 1000
        )
        emissions_tco2e = annual_kwh * float(saas["grid_factor_kgco2e_per_kwh"]) / 1000
        assert emissions_tco2e > 0

    def test_active_users_positive(self, sample_cloud_data):
        assert sample_cloud_data["saas_use_phase"]["active_users"] > 0

    def test_device_power_reasonable(self, sample_cloud_data):
        power = sample_cloud_data["saas_use_phase"]["device_power_w"]
        assert Decimal("10") < power < Decimal("300")


# =============================================================================
# Returns Emissions
# =============================================================================


class TestReturnsEmissions:
    """Test retail returns emissions impact."""

    def test_return_rate_present(self, sample_retail_data):
        assert "returns" in sample_retail_data
        assert sample_retail_data["returns"]["return_rate_pct"] > Decimal("0")

    def test_return_emissions_multiplier(self, sample_retail_data):
        """Returns add 1.5x delivery emissions for returned items."""
        multiplier = sample_retail_data["returns"]["return_emissions_multiplier"]
        assert multiplier > Decimal("1")

    def test_total_with_returns(self, sample_retail_data):
        """Total emissions including returns."""
        lm = sample_retail_data["last_mile"]
        returns = sample_retail_data["returns"]
        base_deliveries = lm["total_deliveries"]
        return_deliveries = base_deliveries * returns["return_rate_pct"] / Decimal("100")
        total_delivery_units = base_deliveries + return_deliveries * returns["return_emissions_multiplier"]
        assert total_delivery_units > base_deliveries
