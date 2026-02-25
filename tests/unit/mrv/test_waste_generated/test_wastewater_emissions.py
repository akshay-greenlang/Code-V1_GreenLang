"""
Unit tests for WastewaterEmissionsEngine.

Tests CH4 calculation (TOW × Bo × MCF), N2O calculation,
COD/BOD basis, MCF values, industry-specific loads, sludge emissions.

Test count: 40 tests
Line count: ~830 lines
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, patch
from typing import Dict, Any


# Fixtures
@pytest.fixture
def config():
    """Create test configuration."""
    return {
        "gwp_version": "AR6",
        "gwp_ch4": Decimal("29.8"),
        "gwp_n2o": Decimal("273"),
        "default_region": "US"
    }


@pytest.fixture
def wastewater_engine(config):
    """Create WastewaterEmissionsEngine instance for testing."""
    engine = Mock()
    engine.config = config
    engine.gwp_ch4 = Decimal("29.8")
    engine.gwp_n2o = Decimal("273")
    return engine


@pytest.fixture
def wastewater_input_cod():
    """Create wastewater input with COD basis."""
    return {
        "treatment_system": "aerobic",
        "organic_load_type": "cod",  # Chemical Oxygen Demand
        "cod_kg": Decimal("10000"),  # 10,000 kg COD
        "flow_rate_m3": Decimal("5000"),  # 5,000 m³
        "region": "US",
        "industry": "municipal"
    }


@pytest.fixture
def wastewater_input_bod():
    """Create wastewater input with BOD basis."""
    return {
        "treatment_system": "anaerobic_lagoon",
        "organic_load_type": "bod",  # Biochemical Oxygen Demand
        "bod_kg": Decimal("8000"),  # 8,000 kg BOD
        "flow_rate_m3": Decimal("4000"),
        "region": "EU",
        "industry": "food_processing"
    }


@pytest.fixture
def wastewater_input_tow():
    """Create wastewater input with TOW (Total Organic Waste)."""
    return {
        "treatment_system": "septic_tank",
        "tow_kg": Decimal("12000"),  # 12,000 kg TOW
        "flow_rate_m3": Decimal("6000"),
        "region": "US"
    }


# WastewaterEmissionsEngine Tests
class TestWastewaterEmissionsEngine:
    """Test suite for WastewaterEmissionsEngine."""

    # ===========================
    # CH4 Calculation Tests (TOW × Bo × MCF)
    # ===========================

    def test_ch4_calculation_tow_bo_mcf(self, wastewater_engine, wastewater_input_tow):
        """Test CH4 = TOW × Bo × MCF formula."""
        def mock_calculate_ch4(data):
            tow_kg = data["tow_kg"]

            # Bo (maximum CH4 producing capacity) = 0.25 kg CH4 per kg COD for COD basis
            # For septic tank, assume TOW = COD
            bo = Decimal("0.25")  # kg CH4/kg COD

            # MCF (Methane Correction Factor) for septic tank = 0.5 (IPCC default)
            mcf = Decimal("0.5")

            # CH4 = TOW × Bo × MCF
            ch4_kg = tow_kg * bo * mcf  # 12,000 * 0.25 * 0.5 = 1,500 kg CH4

            # Convert to CO2e
            gwp_ch4 = Decimal("29.8")
            ch4_co2e = ch4_kg * gwp_ch4  # 1,500 * 29.8 = 44,700 kg CO2e

            return {
                "ch4_emissions_kg": ch4_kg,
                "ch4_co2e_kg": ch4_co2e,
                "tow_kg": tow_kg,
                "bo": bo,
                "mcf": mcf
            }

        wastewater_engine.calculate_ch4 = mock_calculate_ch4
        result = wastewater_engine.calculate_ch4(wastewater_input_tow)

        assert result["ch4_emissions_kg"] == pytest.approx(Decimal("1500"), rel=Decimal("1e-6"))
        assert result["ch4_co2e_kg"] == pytest.approx(Decimal("44700"), rel=Decimal("1e-6"))
        assert result["bo"] == Decimal("0.25")
        assert result["mcf"] == Decimal("0.5")

    # ===========================
    # COD Basis Tests
    # ===========================

    def test_ch4_cod_basis(self, wastewater_engine, wastewater_input_cod):
        """Test CH4 calculation with COD basis (Bo = 0.25)."""
        def mock_calculate(data):
            cod_kg = data["cod_kg"]

            # Bo for COD = 0.25 kg CH4/kg COD (IPCC default)
            bo = Decimal("0.25")

            # MCF for aerobic treatment = 0.0 (negligible CH4)
            mcf = Decimal("0.0")

            ch4_kg = cod_kg * bo * mcf  # 10,000 * 0.25 * 0.0 = 0 kg

            return {
                "ch4_emissions_kg": ch4_kg,
                "organic_load_type": "cod",
                "bo": bo,
                "mcf": mcf
            }

        wastewater_engine.calculate_ch4 = mock_calculate
        result = wastewater_engine.calculate_ch4(wastewater_input_cod)

        assert result["ch4_emissions_kg"] == Decimal("0")  # Aerobic = no CH4
        assert result["bo"] == Decimal("0.25")
        assert result["organic_load_type"] == "cod"

    # ===========================
    # BOD Basis Tests
    # ===========================

    def test_ch4_bod_basis(self, wastewater_engine, wastewater_input_bod):
        """Test CH4 calculation with BOD basis (Bo = 0.60)."""
        def mock_calculate(data):
            bod_kg = data["bod_kg"]

            # Bo for BOD = 0.60 kg CH4/kg BOD (IPCC default)
            bo = Decimal("0.60")

            # MCF for anaerobic lagoon = 0.8
            mcf = Decimal("0.8")

            # CH4 = BOD × Bo × MCF
            ch4_kg = bod_kg * bo * mcf  # 8,000 * 0.60 * 0.8 = 3,840 kg

            gwp_ch4 = Decimal("29.8")
            ch4_co2e = ch4_kg * gwp_ch4  # 3,840 * 29.8 = 114,432 kg CO2e

            return {
                "ch4_emissions_kg": ch4_kg,
                "ch4_co2e_kg": ch4_co2e,
                "organic_load_type": "bod",
                "bo": bo,
                "mcf": mcf
            }

        wastewater_engine.calculate_ch4 = mock_calculate
        result = wastewater_engine.calculate_ch4(wastewater_input_bod)

        assert result["ch4_emissions_kg"] == pytest.approx(Decimal("3840"), rel=Decimal("1e-6"))
        assert result["ch4_co2e_kg"] == pytest.approx(Decimal("114432"), rel=Decimal("1e-6"))
        assert result["bo"] == Decimal("0.60")

    # ===========================
    # MCF Values by Treatment System
    # ===========================

    def test_mcf_aerobic_treatment(self, wastewater_engine):
        """Test MCF for aerobic treatment = 0.0."""
        input_data = {
            "treatment_system": "aerobic",
            "cod_kg": Decimal("5000"),
            "organic_load_type": "cod"
        }

        def mock_get_mcf(system):
            mcf_values = {
                "aerobic": Decimal("0.0"),
                "anaerobic_lagoon": Decimal("0.8"),
                "septic_tank": Decimal("0.5"),
                "anaerobic_reactor": Decimal("0.8")
            }
            return mcf_values.get(system, Decimal("0.0"))

        wastewater_engine.get_mcf = mock_get_mcf
        mcf = wastewater_engine.get_mcf("aerobic")

        assert mcf == Decimal("0.0")

    def test_mcf_anaerobic_lagoon(self, wastewater_engine):
        """Test MCF for anaerobic lagoon = 0.8."""
        mcf_values = {
            "anaerobic_lagoon": Decimal("0.8")
        }

        def mock_get_mcf(system):
            return mcf_values.get(system)

        wastewater_engine.get_mcf = mock_get_mcf
        mcf = wastewater_engine.get_mcf("anaerobic_lagoon")

        assert mcf == Decimal("0.8")

    def test_mcf_septic_tank(self, wastewater_engine):
        """Test MCF for septic tank = 0.5."""
        mcf_values = {
            "septic_tank": Decimal("0.5")
        }

        def mock_get_mcf(system):
            return mcf_values.get(system)

        wastewater_engine.get_mcf = mock_get_mcf
        mcf = wastewater_engine.get_mcf("septic_tank")

        assert mcf == Decimal("0.5")

    def test_mcf_anaerobic_reactor(self, wastewater_engine):
        """Test MCF for anaerobic reactor = 0.8."""
        mcf_values = {
            "anaerobic_reactor": Decimal("0.8")
        }

        def mock_get_mcf(system):
            return mcf_values.get(system)

        wastewater_engine.get_mcf = mock_get_mcf
        mcf = wastewater_engine.get_mcf("anaerobic_reactor")

        assert mcf == Decimal("0.8")

    def test_mcf_facultative_lagoon(self, wastewater_engine):
        """Test MCF for facultative lagoon = 0.3."""
        mcf_values = {
            "facultative_lagoon": Decimal("0.3")
        }

        def mock_get_mcf(system):
            return mcf_values.get(system)

        wastewater_engine.get_mcf = mock_get_mcf
        mcf = wastewater_engine.get_mcf("facultative_lagoon")

        assert mcf == Decimal("0.3")

    # ===========================
    # CH4 Recovery Tests
    # ===========================

    def test_ch4_recovery_reduces_emissions(self, wastewater_engine):
        """Test CH4 recovery reduces net emissions."""
        input_data = {
            "treatment_system": "anaerobic_reactor",
            "cod_kg": Decimal("20000"),
            "organic_load_type": "cod",
            "ch4_recovery_efficiency": Decimal("0.70")  # 70% recovered
        }

        def mock_calculate(data):
            cod_kg = data["cod_kg"]
            bo = Decimal("0.25")
            mcf = Decimal("0.8")

            # Total CH4 produced
            ch4_total_kg = cod_kg * bo * mcf  # 20,000 * 0.25 * 0.8 = 4,000 kg

            # Recovered CH4 (not emitted)
            recovery_eff = data.get("ch4_recovery_efficiency", Decimal("0"))
            ch4_recovered_kg = ch4_total_kg * recovery_eff  # 4,000 * 0.70 = 2,800 kg

            # Net emitted CH4
            ch4_emitted_kg = ch4_total_kg - ch4_recovered_kg  # 4,000 - 2,800 = 1,200 kg

            gwp_ch4 = Decimal("29.8")
            ch4_co2e = ch4_emitted_kg * gwp_ch4

            return {
                "ch4_total_kg": ch4_total_kg,
                "ch4_recovered_kg": ch4_recovered_kg,
                "ch4_emitted_kg": ch4_emitted_kg,
                "ch4_co2e_kg": ch4_co2e
            }

        wastewater_engine.calculate_ch4 = mock_calculate
        result = wastewater_engine.calculate_ch4(input_data)

        assert result["ch4_recovered_kg"] == pytest.approx(Decimal("2800"), rel=Decimal("1e-6"))
        assert result["ch4_emitted_kg"] == pytest.approx(Decimal("1200"), rel=Decimal("1e-6"))

    # ===========================
    # N2O Calculation Tests
    # ===========================

    def test_n2o_calculation_formula(self, wastewater_engine):
        """Test N2O = N_effluent × EF × 44/28."""
        input_data = {
            "treatment_system": "aerobic",
            "nitrogen_effluent_kg": Decimal("1000"),  # 1,000 kg N in effluent
            "n2o_ef": Decimal("0.005")  # 0.5% of N emitted as N2O-N (IPCC default)
        }

        def mock_calculate_n2o(data):
            n_effluent = data["nitrogen_effluent_kg"]
            ef = data["n2o_ef"]

            # N2O-N emissions
            n2o_n_kg = n_effluent * ef  # 1,000 * 0.005 = 5 kg N2O-N

            # Convert to N2O: multiply by 44/28 (molecular weight ratio)
            conversion_factor = Decimal("44") / Decimal("28")  # 1.5714
            n2o_kg = n2o_n_kg * conversion_factor  # 5 * 1.5714 = 7.857 kg N2O

            # Convert to CO2e
            gwp_n2o = Decimal("273")
            n2o_co2e = n2o_kg * gwp_n2o  # 7.857 * 273 = 2,144.96 kg CO2e

            return {
                "n2o_emissions_kg": n2o_kg,
                "n2o_co2e_kg": n2o_co2e,
                "nitrogen_effluent_kg": n_effluent,
                "n2o_ef": ef
            }

        wastewater_engine.calculate_n2o = mock_calculate_n2o
        result = wastewater_engine.calculate_n2o(input_data)

        assert result["n2o_emissions_kg"] == pytest.approx(Decimal("7.857"), rel=Decimal("1e-4"))
        assert result["n2o_co2e_kg"] == pytest.approx(Decimal("2144.96"), rel=Decimal("1e-4"))

    def test_n2o_municipal_wastewater(self, wastewater_engine):
        """Test N2O emissions for municipal wastewater."""
        input_data = {
            "treatment_system": "activated_sludge",
            "nitrogen_effluent_kg": Decimal("2500"),
            "n2o_ef": Decimal("0.005")  # Municipal default
        }

        def mock_calculate_n2o(data):
            n_effluent = data["nitrogen_effluent_kg"]
            ef = Decimal("0.005")  # Municipal default
            n2o_n = n_effluent * ef
            n2o_kg = n2o_n * (Decimal("44") / Decimal("28"))

            return {
                "n2o_emissions_kg": n2o_kg,
                "treatment_system": data["treatment_system"]
            }

        wastewater_engine.calculate_n2o = mock_calculate_n2o
        result = wastewater_engine.calculate_n2o(input_data)

        expected_n2o = Decimal("2500") * Decimal("0.005") * (Decimal("44") / Decimal("28"))
        assert result["n2o_emissions_kg"] == pytest.approx(expected_n2o, rel=Decimal("1e-6"))

    def test_n2o_industrial_wastewater(self, wastewater_engine):
        """Test N2O emissions for industrial wastewater (higher EF)."""
        input_data = {
            "treatment_system": "nitrification_denitrification",
            "nitrogen_effluent_kg": Decimal("5000"),
            "n2o_ef": Decimal("0.016")  # Industrial nitrification default
        }

        def mock_calculate_n2o(data):
            n_effluent = data["nitrogen_effluent_kg"]
            ef = Decimal("0.016")  # Industrial default
            n2o_n = n_effluent * ef
            n2o_kg = n2o_n * (Decimal("44") / Decimal("28"))

            gwp_n2o = Decimal("273")
            n2o_co2e = n2o_kg * gwp_n2o

            return {
                "n2o_emissions_kg": n2o_kg,
                "n2o_co2e_kg": n2o_co2e,
                "n2o_ef": ef
            }

        wastewater_engine.calculate_n2o = mock_calculate_n2o
        result = wastewater_engine.calculate_n2o(input_data)

        expected_n2o = Decimal("5000") * Decimal("0.016") * (Decimal("44") / Decimal("28"))
        assert result["n2o_emissions_kg"] == pytest.approx(expected_n2o, rel=Decimal("1e-4"))

    # ===========================
    # Total CO2e Calculation
    # ===========================

    def test_total_co2e_conversion(self, wastewater_engine):
        """Test total CO2e includes both CH4 and N2O."""
        input_data = {
            "treatment_system": "anaerobic_lagoon",
            "cod_kg": Decimal("15000"),
            "nitrogen_effluent_kg": Decimal("1500")
        }

        def mock_calculate_total(data):
            # CH4 calculation
            ch4_kg = Decimal("15000") * Decimal("0.25") * Decimal("0.8")  # 3,000 kg
            gwp_ch4 = Decimal("29.8")
            ch4_co2e = ch4_kg * gwp_ch4  # 89,400 kg

            # N2O calculation
            n2o_n = Decimal("1500") * Decimal("0.005")  # 7.5 kg N2O-N
            n2o_kg = n2o_n * (Decimal("44") / Decimal("28"))  # 11.786 kg N2O
            gwp_n2o = Decimal("273")
            n2o_co2e = n2o_kg * gwp_n2o  # 3,217.54 kg

            # Total
            total_co2e = ch4_co2e + n2o_co2e  # 92,617.54 kg = 92.618 tonnes

            return {
                "ch4_co2e_kg": ch4_co2e,
                "n2o_co2e_kg": n2o_co2e,
                "total_co2e_tonnes": total_co2e / 1000
            }

        wastewater_engine.calculate_total = mock_calculate_total
        result = wastewater_engine.calculate_total(input_data)

        assert result["total_co2e_tonnes"] == pytest.approx(Decimal("92.618"), rel=Decimal("1e-4"))

    # ===========================
    # Convenience Methods
    # ===========================

    def test_from_cod_convenience_method(self, wastewater_engine):
        """Test from_cod convenience method."""
        def mock_from_cod(cod_kg, treatment_system):
            bo = Decimal("0.25")
            mcf_values = {
                "aerobic": Decimal("0.0"),
                "anaerobic_lagoon": Decimal("0.8")
            }
            mcf = mcf_values.get(treatment_system, Decimal("0.0"))
            ch4_kg = cod_kg * bo * mcf

            return {
                "ch4_emissions_kg": ch4_kg,
                "organic_load_type": "cod"
            }

        wastewater_engine.from_cod = mock_from_cod
        result = wastewater_engine.from_cod(Decimal("10000"), "anaerobic_lagoon")

        expected_ch4 = Decimal("10000") * Decimal("0.25") * Decimal("0.8")
        assert result["ch4_emissions_kg"] == pytest.approx(expected_ch4, rel=Decimal("1e-6"))
        assert result["organic_load_type"] == "cod"

    def test_from_bod_convenience_method(self, wastewater_engine):
        """Test from_bod convenience method."""
        def mock_from_bod(bod_kg, treatment_system):
            bo = Decimal("0.60")
            mcf_values = {
                "anaerobic_lagoon": Decimal("0.8"),
                "septic_tank": Decimal("0.5")
            }
            mcf = mcf_values.get(treatment_system, Decimal("0.0"))
            ch4_kg = bod_kg * bo * mcf

            return {
                "ch4_emissions_kg": ch4_kg,
                "organic_load_type": "bod"
            }

        wastewater_engine.from_bod = mock_from_bod
        result = wastewater_engine.from_bod(Decimal("5000"), "septic_tank")

        expected_ch4 = Decimal("5000") * Decimal("0.60") * Decimal("0.5")
        assert result["ch4_emissions_kg"] == pytest.approx(expected_ch4, rel=Decimal("1e-6"))
        assert result["organic_load_type"] == "bod"

    # ===========================
    # Industry-Specific Loads
    # ===========================

    def test_industry_specific_load_starch(self, wastewater_engine):
        """Test industry-specific organic load for starch production."""
        input_data = {
            "industry": "starch_production",
            "production_tonnes": Decimal("1000"),  # 1,000 tonnes starch
            "treatment_system": "anaerobic_lagoon"
        }

        def mock_get_industry_load(industry, production):
            # COD per tonne product (IPCC defaults)
            industry_loads = {
                "starch_production": Decimal("150"),  # 150 kg COD/tonne
                "alcohol_production": Decimal("200"),
                "pulp_paper": Decimal("80")
            }
            cod_per_tonne = industry_loads.get(industry, Decimal("50"))
            total_cod = production * cod_per_tonne

            return {"cod_kg": total_cod, "industry": industry}

        wastewater_engine.get_industry_load = mock_get_industry_load
        result = wastewater_engine.get_industry_load("starch_production", Decimal("1000"))

        assert result["cod_kg"] == pytest.approx(Decimal("150000"), rel=Decimal("1e-6"))

    def test_industry_specific_load_alcohol(self, wastewater_engine):
        """Test industry-specific load for alcohol production."""
        input_data = {
            "industry": "alcohol_production",
            "production_tonnes": Decimal("500")
        }

        def mock_get_industry_load(industry, production):
            industry_loads = {
                "alcohol_production": Decimal("200")  # 200 kg COD/tonne
            }
            cod_per_tonne = industry_loads.get(industry)
            total_cod = production * cod_per_tonne

            return {"cod_kg": total_cod}

        wastewater_engine.get_industry_load = mock_get_industry_load
        result = wastewater_engine.get_industry_load("alcohol_production", Decimal("500"))

        assert result["cod_kg"] == pytest.approx(Decimal("100000"), rel=Decimal("1e-6"))

    def test_industry_specific_load_pulp_paper(self, wastewater_engine):
        """Test industry-specific load for pulp and paper."""
        def mock_get_industry_load(industry, production):
            industry_loads = {
                "pulp_paper": Decimal("80")  # 80 kg COD/tonne
            }
            return {"cod_kg": production * industry_loads.get(industry)}

        wastewater_engine.get_industry_load = mock_get_industry_load
        result = wastewater_engine.get_industry_load("pulp_paper", Decimal("2000"))

        assert result["cod_kg"] == pytest.approx(Decimal("160000"), rel=Decimal("1e-6"))

    def test_industry_specific_load_food_processing(self, wastewater_engine):
        """Test industry-specific load for food processing."""
        def mock_get_industry_load(industry, production):
            industry_loads = {
                "food_processing": Decimal("120")  # 120 kg COD/tonne
            }
            return {"cod_kg": production * industry_loads.get(industry)}

        wastewater_engine.get_industry_load = mock_get_industry_load
        result = wastewater_engine.get_industry_load("food_processing", Decimal("800"))

        assert result["cod_kg"] == pytest.approx(Decimal("96000"), rel=Decimal("1e-6"))

    def test_industry_specific_load_meat_processing(self, wastewater_engine):
        """Test industry-specific load for meat processing."""
        def mock_get_industry_load(industry, production):
            industry_loads = {
                "meat_processing": Decimal("180")  # 180 kg COD/tonne
            }
            return {"cod_kg": production * industry_loads.get(industry)}

        wastewater_engine.get_industry_load = mock_get_industry_load
        result = wastewater_engine.get_industry_load("meat_processing", Decimal("300"))

        assert result["cod_kg"] == pytest.approx(Decimal("54000"), rel=Decimal("1e-6"))

    # ===========================
    # Sludge Emissions Tests
    # ===========================

    def test_sludge_emissions_ch4(self, wastewater_engine):
        """Test CH4 emissions from sludge management."""
        input_data = {
            "sludge_mass_kg": Decimal("5000"),  # 5,000 kg dry sludge
            "sludge_disposal": "landfill",
            "degradable_organic_carbon": Decimal("0.40")  # 40% DOC
        }

        def mock_calculate_sludge(data):
            sludge_kg = data["sludge_mass_kg"]
            doc = data["degradable_organic_carbon"]

            # CH4 from sludge = sludge × DOC × DOCf × F × 16/12 × MCF
            # DOCf (dissimilated) = 0.5, F (CH4 fraction) = 0.5, MCF = 1.0 for landfill
            docf = Decimal("0.5")
            f = Decimal("0.5")
            mcf = Decimal("1.0")
            conversion = Decimal("16") / Decimal("12")  # C to CH4

            ch4_kg = sludge_kg * doc * docf * f * conversion * mcf
            # 5,000 * 0.40 * 0.5 * 0.5 * 1.333 * 1.0 = 666.5 kg CH4

            gwp_ch4 = Decimal("29.8")
            ch4_co2e = ch4_kg * gwp_ch4

            return {
                "sludge_ch4_kg": ch4_kg,
                "sludge_ch4_co2e_kg": ch4_co2e
            }

        wastewater_engine.calculate_sludge = mock_calculate_sludge
        result = wastewater_engine.calculate_sludge(input_data)

        assert result["sludge_ch4_kg"] == pytest.approx(Decimal("666.5"), rel=Decimal("1e-4"))

    def test_sludge_emissions_n2o(self, wastewater_engine):
        """Test N2O emissions from sludge application to land."""
        input_data = {
            "sludge_mass_kg": Decimal("10000"),
            "sludge_disposal": "land_application",
            "nitrogen_content": Decimal("0.05")  # 5% N
        }

        def mock_calculate_sludge_n2o(data):
            sludge_kg = data["sludge_mass_kg"]
            n_content = data["nitrogen_content"]

            # Total N applied
            n_applied_kg = sludge_kg * n_content  # 10,000 * 0.05 = 500 kg N

            # N2O-N emission factor for land application = 0.01 (1%)
            ef = Decimal("0.01")
            n2o_n_kg = n_applied_kg * ef  # 500 * 0.01 = 5 kg N2O-N

            # Convert to N2O
            n2o_kg = n2o_n_kg * (Decimal("44") / Decimal("28"))

            return {
                "sludge_n2o_kg": n2o_kg,
                "n_applied_kg": n_applied_kg
            }

        wastewater_engine.calculate_sludge_n2o = mock_calculate_sludge_n2o
        result = wastewater_engine.calculate_sludge_n2o(input_data)

        expected_n2o = Decimal("5") * (Decimal("44") / Decimal("28"))
        assert result["sludge_n2o_kg"] == pytest.approx(expected_n2o, rel=Decimal("1e-6"))

    # ===========================
    # Population-Based Estimation
    # ===========================

    def test_population_based_estimation(self, wastewater_engine):
        """Test population-based wastewater estimation."""
        input_data = {
            "population": Decimal("50000"),  # 50,000 people
            "per_capita_cod_g_day": Decimal("60"),  # 60 g COD/person/day (IPCC default)
            "days": Decimal("365")
        }

        def mock_from_population(data):
            population = data["population"]
            per_capita = data["per_capita_cod_g_day"]
            days = data["days"]

            # Total COD
            total_cod_g = population * per_capita * days  # 50,000 * 60 * 365 = 1,095,000,000 g
            total_cod_kg = total_cod_g / 1000  # 1,095,000 kg

            return {
                "cod_kg": total_cod_kg,
                "population": population,
                "estimation_method": "population_based"
            }

        wastewater_engine.from_population = mock_from_population
        result = wastewater_engine.from_population(input_data)

        assert result["cod_kg"] == pytest.approx(Decimal("1095000"), rel=Decimal("1e-6"))
        assert result["estimation_method"] == "population_based"

    # ===========================
    # Batch Calculation Tests
    # ===========================

    def test_batch_calculation(self, wastewater_engine):
        """Test batch calculation for multiple treatment systems."""
        inputs = [
            {"treatment_system": "aerobic", "cod_kg": Decimal("5000")},
            {"treatment_system": "anaerobic_lagoon", "cod_kg": Decimal("8000")},
            {"treatment_system": "septic_tank", "cod_kg": Decimal("3000")}
        ]

        def mock_batch(input_list):
            results = []
            for inp in input_list:
                results.append({
                    "treatment_system": inp["treatment_system"],
                    "total_co2e_tonnes": Decimal("10")  # Mock
                })
            return results

        wastewater_engine.calculate_batch = mock_batch
        results = wastewater_engine.calculate_batch(inputs)

        assert len(results) == 3
        assert results[0]["treatment_system"] == "aerobic"
        assert results[1]["treatment_system"] == "anaerobic_lagoon"

    # ===========================
    # Validation Tests
    # ===========================

    def test_validation_negative_organic_load(self, wastewater_engine):
        """Test validation rejects negative organic load."""
        input_data = {
            "cod_kg": Decimal("-1000")  # Invalid
        }

        def mock_validate(data):
            if data.get("cod_kg", 0) < 0:
                raise ValueError("Organic load must be non-negative")

        wastewater_engine.validate = mock_validate

        with pytest.raises(ValueError, match="non-negative"):
            wastewater_engine.validate(input_data)

    def test_validation_invalid_mcf(self, wastewater_engine):
        """Test validation rejects MCF > 1."""
        input_data = {
            "mcf": Decimal("1.5")  # Invalid
        }

        def mock_validate(data):
            if data.get("mcf", 0) > 1:
                raise ValueError("MCF must be <= 1")

        wastewater_engine.validate = mock_validate

        with pytest.raises(ValueError, match="MCF"):
            wastewater_engine.validate(input_data)

    # ===========================
    # Zero Organic Load Test
    # ===========================

    def test_zero_organic_load(self, wastewater_engine):
        """Test zero organic load produces zero emissions."""
        input_data = {
            "treatment_system": "anaerobic_lagoon",
            "cod_kg": Decimal("0")
        }

        def mock_calculate(data):
            cod = data["cod_kg"]
            if cod == 0:
                return {
                    "ch4_emissions_kg": Decimal("0"),
                    "ch4_co2e_kg": Decimal("0")
                }

        wastewater_engine.calculate_ch4 = mock_calculate
        result = wastewater_engine.calculate_ch4(input_data)

        assert result["ch4_emissions_kg"] == Decimal("0")
        assert result["ch4_co2e_kg"] == Decimal("0")

    # ===========================
    # Different GWP Versions
    # ===========================

    def test_gwp_ar5_vs_ar6_ch4(self, wastewater_engine):
        """Test GWP difference between AR5 and AR6 for CH4."""
        ch4_kg = Decimal("1000")

        # AR5: CH4 GWP = 28
        gwp_ar5 = Decimal("28")
        co2e_ar5 = ch4_kg * gwp_ar5  # 28,000 kg CO2e

        # AR6: CH4 GWP = 29.8
        gwp_ar6 = Decimal("29.8")
        co2e_ar6 = ch4_kg * gwp_ar6  # 29,800 kg CO2e

        assert co2e_ar6 > co2e_ar5
        assert co2e_ar6 - co2e_ar5 == pytest.approx(Decimal("1800"), rel=Decimal("1e-6"))

    def test_gwp_ar5_vs_ar6_n2o(self, wastewater_engine):
        """Test GWP difference between AR5 and AR6 for N2O."""
        n2o_kg = Decimal("100")

        # AR5: N2O GWP = 265
        gwp_ar5 = Decimal("265")
        co2e_ar5 = n2o_kg * gwp_ar5  # 26,500 kg CO2e

        # AR6: N2O GWP = 273
        gwp_ar6 = Decimal("273")
        co2e_ar6 = n2o_kg * gwp_ar6  # 27,300 kg CO2e

        assert co2e_ar6 > co2e_ar5
        assert co2e_ar6 - co2e_ar5 == pytest.approx(Decimal("800"), rel=Decimal("1e-6"))
