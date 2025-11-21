# -*- coding: utf-8 -*-
"""
tests/agents/test_wtt_boundary.py

Well-to-Tank (WTT) Boundary Support Tests

Tests WTT and WTW boundary calculations:
- WTT (Well-to-Tank): Upstream emissions only (extraction, refining, transport)
- WTW (Well-to-Wheel): Full lifecycle = WTT + combustion
- Validation against GREET 2024 and UK BEIS 2024 reference data

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
from greenlang.agents.fuel_agent_ai_v2 import FuelAgentAI_v2
from greenlang.data.emission_factor_database import EmissionFactorDatabase
from greenlang.data import wtt_emission_factors


class TestWTTBoundarySupport:
    """Test suite for WTT (Well-to-Tank) boundary support"""

    def setup_method(self):
        """Setup test fixtures."""
        self.agent = FuelAgentAI_v2(enable_fast_path=True)
        self.db = EmissionFactorDatabase()

    # ==================== WTT FACTOR RETRIEVAL ====================

    def test_wtt_factor_diesel_us(self):
        """Test WTT factor retrieval for diesel (US)."""
        wtt_factor, source = wtt_emission_factors.get_wtt_factor(
            "diesel", "gallons", "US"
        )

        assert wtt_factor == 2.04, "WTT factor should be 2.04 kgCO2e/gallon"
        assert "GREET 2024" in source, "Source should reference GREET 2024"

    def test_wtt_factor_natural_gas_us(self):
        """Test WTT factor retrieval for natural gas (US)."""
        wtt_factor, source = wtt_emission_factors.get_wtt_factor(
            "natural_gas", "therms", "US"
        )

        assert wtt_factor == 0.95, "WTT factor should be 0.95 kgCO2e/therm"
        assert "methane leakage" in source.lower(), "Source should mention methane leakage"

    def test_wtt_factor_electricity_us(self):
        """Test WTT factor retrieval for electricity (US - T&D losses)."""
        wtt_factor, source = wtt_emission_factors.get_wtt_factor(
            "electricity", "kWh", "US"
        )

        assert wtt_factor == 0.029, "WTT factor should be 0.029 kgCO2e/kWh (T&D losses)"
        assert "EPA" in source, "Source should reference EPA"

    def test_wtt_factor_gasoline_uk(self):
        """Test WTT factor retrieval for gasoline (UK)."""
        wtt_factor, source = wtt_emission_factors.get_wtt_factor(
            "gasoline", "liters", "UK"
        )

        assert wtt_factor == 0.42, "WTT factor should be 0.42 kgCO2e/liter"
        assert "UK BEIS" in source, "Source should reference UK BEIS"

    def test_wtt_factor_not_found(self):
        """Test WTT factor retrieval for unavailable fuel."""
        wtt_factor, source = wtt_emission_factors.get_wtt_factor(
            "unknown_fuel", "units", "US"
        )

        assert wtt_factor == 0.0, "Should return 0.0 for unavailable factor"
        assert "No WTT data" in source, "Source should indicate no data"

    # ==================== WTT RATIO VALIDATION ====================

    def test_typical_wtt_ratio_diesel(self):
        """Test typical WTT ratio for diesel (~20% of combustion)."""
        ratio = wtt_emission_factors.TYPICAL_WTT_RATIOS["diesel"]
        assert ratio == 0.20, "Diesel WTT should be ~20% of combustion"

    def test_typical_wtt_ratio_gasoline(self):
        """Test typical WTT ratio for gasoline (~18% of combustion)."""
        ratio = wtt_emission_factors.TYPICAL_WTT_RATIOS["gasoline"]
        assert ratio == 0.18, "Gasoline WTT should be ~18% of combustion"

    def test_typical_wtt_ratio_natural_gas(self):
        """Test typical WTT ratio for natural gas (~18% of combustion)."""
        ratio = wtt_emission_factors.TYPICAL_WTT_RATIOS["natural_gas"]
        assert ratio == 0.18, "Natural gas WTT should be ~18% of combustion"

    def test_typical_wtt_ratio_coal(self):
        """Test typical WTT ratio for coal (~8% of combustion)."""
        ratio = wtt_emission_factors.TYPICAL_WTT_RATIOS["coal"]
        assert ratio == 0.08, "Coal WTT should be ~8% of combustion"

    # ==================== WTT ESTIMATION ====================

    def test_estimate_wtt_factor_diesel(self):
        """Test WTT estimation for diesel using typical ratio."""
        combustion_factor = 10.21  # kgCO2e/gallon (US diesel)
        estimated_wtt = wtt_emission_factors.estimate_wtt_factor(
            "diesel", combustion_factor
        )

        expected = combustion_factor * 0.20
        assert abs(estimated_wtt - expected) < 0.01, \
            f"Estimated WTT should be ~{expected:.2f} kgCO2e/gallon"

    def test_estimate_wtt_factor_unknown_fuel(self):
        """Test WTT estimation for unknown fuel (default 15% ratio)."""
        combustion_factor = 100.0
        estimated_wtt = wtt_emission_factors.estimate_wtt_factor(
            "unknown_fuel", combustion_factor
        )

        expected = combustion_factor * 0.15  # Default ratio
        assert estimated_wtt == expected, \
            "Unknown fuel should use 15% default ratio"

    # ==================== WTW CALCULATION ====================

    def test_wtw_calculation(self):
        """Test WTW (Well-to-Wheel) = WTT + combustion."""
        combustion = 10.21
        wtt = 2.04
        wtw = wtt_emission_factors.calculate_wtw_factor(combustion, wtt)

        assert wtw == 12.25, "WTW should equal combustion + WTT"

    def test_wtt_percentage_diesel(self):
        """Test WTT as percentage of total lifecycle for diesel."""
        combustion = 10.21
        wtt = 2.04
        pct = wtt_emission_factors.get_wtt_percentage(combustion, wtt)

        expected = (wtt / (combustion + wtt)) * 100
        assert abs(pct - expected) < 0.1, \
            f"WTT should be ~{expected:.1f}% of total lifecycle"

    # ==================== DATABASE INTEGRATION ====================

    def test_database_wtt_boundary_diesel(self):
        """Test database retrieval for WTT boundary (diesel)."""
        factor = self.db.get_factor_record(
            fuel_type="diesel",
            unit="gallons",
            geography="US",
            scope="1",
            boundary="WTT",
            gwp_set="IPCC_AR6_100",
        )

        assert factor is not None, "WTT factor should be computed"
        assert factor.boundary.value == "WTT", "Boundary should be WTT"

        # WTT should be ~2.04 kgCO2e/gallon
        wtt_co2e = factor.get_co2e("100yr")
        assert 2.0 <= wtt_co2e <= 2.1, \
            f"WTT CO2e should be ~2.04 kgCO2e/gallon, got {wtt_co2e}"

    def test_database_wtw_boundary_diesel(self):
        """Test database retrieval for WTW boundary (diesel)."""
        factor = self.db.get_factor_record(
            fuel_type="diesel",
            unit="gallons",
            geography="US",
            scope="1",
            boundary="WTW",
            gwp_set="IPCC_AR6_100",
        )

        assert factor is not None, "WTW factor should be computed"
        assert factor.boundary.value == "WTW", "Boundary should be WTW"

        # WTW should be ~12.25 kgCO2e/gallon (10.21 combustion + 2.04 WTT)
        wtw_co2e = factor.get_co2e("100yr")
        assert 12.0 <= wtw_co2e <= 12.5, \
            f"WTW CO2e should be ~12.25 kgCO2e/gallon, got {wtw_co2e}"

    def test_database_wtw_vs_combustion_diesel(self):
        """Test WTW = combustion + WTT for diesel."""
        combustion_factor = self.db.get_factor_record(
            "diesel", "gallons", "US", "1", "combustion", "IPCC_AR6_100"
        )
        wtw_factor = self.db.get_factor_record(
            "diesel", "gallons", "US", "1", "WTW", "IPCC_AR6_100"
        )

        combustion_co2e = combustion_factor.get_co2e("100yr")
        wtw_co2e = wtw_factor.get_co2e("100yr")

        # WTW should be ~20% higher than combustion (diesel WTT ratio)
        ratio = (wtw_co2e - combustion_co2e) / combustion_co2e
        assert 0.18 <= ratio <= 0.22, \
            f"WTW should be ~20% higher than combustion, got {ratio*100:.1f}%"

    # ==================== AGENT INTEGRATION ====================

    def test_agent_wtt_boundary_calculation(self):
        """Test FuelAgentAI v2 with WTT boundary."""
        payload = {
            "fuel_type": "diesel",
            "amount": 100,
            "unit": "gallons",
            "country": "US",
            "boundary": "WTT",
            "response_format": "enhanced",
        }

        result = self.agent.run(payload)

        assert result["success"], f"Request should succeed: {result.get('error')}"

        data = result["data"]
        assert data["boundary"] == "WTT", "Boundary should be WTT"

        # 100 gallons × 2.04 kgCO2e/gallon = 204 kgCO2e
        wtt_emissions = data["co2e_emissions_kg"]
        assert 200 <= wtt_emissions <= 210, \
            f"WTT emissions should be ~204 kgCO2e, got {wtt_emissions}"

    def test_agent_wtw_boundary_calculation(self):
        """Test FuelAgentAI v2 with WTW boundary."""
        payload = {
            "fuel_type": "diesel",
            "amount": 100,
            "unit": "gallons",
            "country": "US",
            "boundary": "WTW",
            "response_format": "enhanced",
        }

        result = self.agent.run(payload)

        assert result["success"], f"Request should succeed: {result.get('error')}"

        data = result["data"]
        assert data["boundary"] == "WTW", "Boundary should be WTW"

        # 100 gallons × 12.25 kgCO2e/gallon = 1225 kgCO2e
        wtw_emissions = data["co2e_emissions_kg"]
        assert 1200 <= wtw_emissions <= 1250, \
            f"WTW emissions should be ~1225 kgCO2e, got {wtw_emissions}"

    def test_agent_combustion_vs_wtw_comparison(self):
        """Test combustion vs WTW boundary comparison."""
        base_payload = {
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms",
            "country": "US",
            "response_format": "compact",
        }

        # Run combustion
        combustion_payload = {**base_payload, "boundary": "combustion"}
        combustion_result = self.agent.run(combustion_payload)

        # Run WTW
        wtw_payload = {**base_payload, "boundary": "WTW"}
        wtw_result = self.agent.run(wtw_payload)

        assert combustion_result["success"], "Combustion calculation should succeed"
        assert wtw_result["success"], "WTW calculation should succeed"

        combustion_emissions = combustion_result["data"]["co2e_emissions_kg"]
        wtw_emissions = wtw_result["data"]["co2e_emissions_kg"]

        # WTW should be ~18% higher than combustion for natural gas
        ratio = (wtw_emissions - combustion_emissions) / combustion_emissions
        assert 0.16 <= ratio <= 0.20, \
            f"WTW should be ~18% higher than combustion for natural gas, got {ratio*100:.1f}%"

    # ==================== MULTI-GAS BREAKDOWN ====================

    def test_wtt_multigas_breakdown_natural_gas(self):
        """Test multi-gas breakdown for WTT (natural gas includes CH4 leakage)."""
        factor = self.db.get_factor_record(
            "natural_gas", "therms", "US", "1", "WTT", "IPCC_AR6_100"
        )

        assert factor is not None, "WTT factor should be computed"

        # Natural gas WTT should include CH4 leakage
        assert factor.vectors.CH4 > 0, "WTT for natural gas should include CH4 leakage"
        assert factor.vectors.CO2 > 0, "WTT should include CO2 emissions"

    def test_wtw_multigas_breakdown_diesel(self):
        """Test multi-gas breakdown for WTW (diesel)."""
        wtw_factor = self.db.get_factor_record(
            "diesel", "gallons", "US", "1", "WTW", "IPCC_AR6_100"
        )
        combustion_factor = self.db.get_factor_record(
            "diesel", "gallons", "US", "1", "combustion", "IPCC_AR6_100"
        )

        # WTW CO2 should be higher than combustion CO2
        assert wtw_factor.vectors.CO2 > combustion_factor.vectors.CO2, \
            "WTW CO2 should include upstream emissions"

        # CH4 and N2O should be similar (minimal in upstream for diesel)
        assert abs(wtw_factor.vectors.CH4 - combustion_factor.vectors.CH4) < 0.001, \
            "WTW CH4 should be similar to combustion (minimal upstream CH4 for diesel)"

    # ==================== PROVENANCE TRACKING ====================

    def test_wtt_provenance_tracking(self):
        """Test provenance tracking for WTT boundary."""
        payload = {
            "fuel_type": "diesel",
            "amount": 100,
            "unit": "gallons",
            "country": "US",
            "boundary": "WTT",
            "response_format": "enhanced",
        }

        result = self.agent.run(payload)
        data = result["data"]

        # Check provenance
        assert "provenance" in data, "WTT result should include provenance"
        prov = data["provenance"]

        assert prov["source_org"] == "GreenLang", "WTT factors computed by GreenLang"
        assert "GREET" in prov["citation"], "Citation should reference GREET for WTT data"

    def test_wtw_provenance_tracking(self):
        """Test provenance tracking for WTW boundary."""
        payload = {
            "fuel_type": "diesel",
            "amount": 100,
            "unit": "gallons",
            "country": "US",
            "boundary": "WTW",
            "response_format": "enhanced",
        }

        result = self.agent.run(payload)
        data = result["data"]

        # Check provenance
        assert "provenance" in data, "WTW result should include provenance"
        prov = data["provenance"]

        assert prov["source_org"] == "GreenLang", "WTW factors computed by GreenLang"
        assert "EPA" in prov["citation"], "Citation should reference EPA for combustion"
        assert "GREET" in prov["citation"], "Citation should reference GREET for WTT"

    # ==================== EDGE CASES ====================

    def test_wtt_boundary_electricity(self):
        """Test WTT for electricity (transmission & distribution losses)."""
        payload = {
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "country": "US",
            "boundary": "WTT",
            "response_format": "compact",
        }

        result = self.agent.run(payload)
        assert result["success"], "Electricity WTT calculation should succeed"

        # WTT for electricity is T&D losses (~8%)
        wtt_emissions = result["data"]["co2e_emissions_kg"]
        assert 25 <= wtt_emissions <= 35, \
            f"Electricity WTT should be ~29 kgCO2e (1000 kWh × 0.029), got {wtt_emissions}"

    def test_wtw_boundary_coal(self):
        """Test WTW for coal (low upstream ratio ~8%)."""
        payload = {
            "fuel_type": "coal",
            "amount": 1,
            "unit": "tons",
            "country": "US",
            "boundary": "WTW",
            "response_format": "compact",
        }

        result = self.agent.run(payload)
        assert result["success"], "Coal WTW calculation should succeed"

        # Coal WTW should be ~8% higher than combustion
        # Combustion: ~2089 kgCO2e/ton, WTW: ~2269 kgCO2e/ton
        wtw_emissions = result["data"]["co2e_emissions_kg"]
        assert 2250 <= wtw_emissions <= 2350, \
            f"Coal WTW should be ~2269 kgCO2e/ton, got {wtw_emissions}"

    def test_wtt_with_renewable_offset(self):
        """Test WTT with renewable offset (should still apply offset)."""
        payload = {
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "country": "US",
            "boundary": "WTT",
            "renewable_percentage": 50,
            "response_format": "compact",
        }

        result = self.agent.run(payload)
        assert result["success"], "WTT with renewable offset should succeed"

        # 50% renewable offset should reduce emissions by half
        wtt_emissions = result["data"]["co2e_emissions_kg"]
        assert 12 <= wtt_emissions <= 17, \
            f"50% renewable offset should reduce WTT by ~50%, got {wtt_emissions} kgCO2e"

    # ==================== GREET 2024 COMPLIANCE ====================

    def test_greet_compliance_diesel_wtt(self):
        """Test compliance with GREET 2024 diesel WTT factor."""
        # GREET 2024: Diesel WTT = 2.04 kgCO2e/gallon
        wtt_factor, _ = wtt_emission_factors.get_wtt_factor(
            "diesel", "gallons", "US"
        )

        greet_reference = 2.04
        assert abs(wtt_factor - greet_reference) < 0.01, \
            f"Diesel WTT should match GREET 2024 ({greet_reference} kgCO2e/gallon)"

    def test_greet_compliance_gasoline_wtt(self):
        """Test compliance with GREET 2024 gasoline WTT factor."""
        # GREET 2024: Gasoline WTT = 1.58 kgCO2e/gallon
        wtt_factor, _ = wtt_emission_factors.get_wtt_factor(
            "gasoline", "gallons", "US"
        )

        greet_reference = 1.58
        assert abs(wtt_factor - greet_reference) < 0.01, \
            f"Gasoline WTT should match GREET 2024 ({greet_reference} kgCO2e/gallon)"

    def test_greet_compliance_natural_gas_wtt(self):
        """Test compliance with GREET 2024 natural gas WTT factor."""
        # GREET 2024: Natural gas WTT = 0.95 kgCO2e/therm (includes methane leakage)
        wtt_factor, _ = wtt_emission_factors.get_wtt_factor(
            "natural_gas", "therms", "US"
        )

        greet_reference = 0.95
        assert abs(wtt_factor - greet_reference) < 0.01, \
            f"Natural gas WTT should match GREET 2024 ({greet_reference} kgCO2e/therm)"

    # ==================== UK BEIS 2024 COMPLIANCE ====================

    def test_beis_compliance_diesel_wtt_uk(self):
        """Test compliance with UK BEIS 2024 diesel WTT factor."""
        # UK BEIS 2024: Diesel WTT = 0.54 kgCO2e/liter
        wtt_factor, _ = wtt_emission_factors.get_wtt_factor(
            "diesel", "liters", "UK"
        )

        beis_reference = 0.54
        assert abs(wtt_factor - beis_reference) < 0.01, \
            f"Diesel WTT (UK) should match BEIS 2024 ({beis_reference} kgCO2e/liter)"

    def test_beis_compliance_gasoline_wtt_uk(self):
        """Test compliance with UK BEIS 2024 gasoline WTT factor."""
        # UK BEIS 2024: Gasoline WTT = 0.42 kgCO2e/liter
        wtt_factor, _ = wtt_emission_factors.get_wtt_factor(
            "gasoline", "liters", "UK"
        )

        beis_reference = 0.42
        assert abs(wtt_factor - beis_reference) < 0.01, \
            f"Gasoline WTT (UK) should match BEIS 2024 ({beis_reference} kgCO2e/liter)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
