"""
Standards Generators Tests
GL-VCCI Scope 3 Platform
"""

import pytest
from datetime import datetime
from services.agents.reporting.standards import ESRSE1Generator, CDPGenerator, IFRSS2Generator, ISO14083Generator
from services.agents.reporting.models import CompanyInfo, EmissionsData, EnergyData, TransportData


@pytest.fixture
def sample_company():
    return CompanyInfo(name="Test Corp", reporting_year=2024)


@pytest.fixture
def sample_emissions():
    return EmissionsData(
        scope1_tco2e=1000.0,
        scope2_location_tco2e=2000.0,
        scope2_market_tco2e=1800.0,
        scope3_tco2e=15000.0,
        scope3_categories={1: 10000.0, 4: 3000.0, 6: 2000.0},
        avg_dqi_score=85.0,
        reporting_period_start=datetime(2024, 1, 1),
        reporting_period_end=datetime(2024, 12, 31),
    )


# ESRS E1 Tests
def test_esrs_e1_generator(sample_company, sample_emissions):
    """Test ESRS E1 generator."""
    generator = ESRSE1Generator()
    content = generator.generate_report_content(sample_company, sample_emissions)

    assert content["standard"] == "ESRS E1"
    assert "disclosures" in content
    assert len(content["disclosures"]) > 0


# CDP Tests
def test_cdp_generator(sample_company, sample_emissions):
    """Test CDP generator."""
    generator = CDPGenerator()
    content = generator.generate_report_content(sample_company, sample_emissions)

    assert "C6" in content
    assert content["C6"]["C6.1"]["scope1_tco2e"] == 1000.0


# IFRS S2 Tests
def test_ifrs_s2_generator(sample_company, sample_emissions):
    """Test IFRS S2 generator."""
    generator = IFRSS2Generator()
    content = generator.generate_report_content(sample_company, sample_emissions)

    assert content["standard"] == "IFRS S2"
    assert "pillars" in content


# ISO 14083 Tests
def test_iso_14083_generator():
    """Test ISO 14083 generator."""
    generator = ISO14083Generator()
    transport_data = {
        "transport_by_mode": {"road": {}, "sea": {}},
        "total_emissions_tco2e": 3000.0,
        "methodology": "ISO 14083:2023",
        "emission_factors_used": [],
        "data_quality_score": 85.0,
    }

    certificate = generator.generate_certificate(transport_data, {})

    assert "certificate_id" in certificate
    assert certificate["standard"] == "ISO 14083:2023"
    assert certificate["conformance_level"] == "Full"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
