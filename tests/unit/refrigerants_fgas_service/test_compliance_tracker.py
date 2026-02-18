# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceTrackerEngine - AGENT-MRV-002 Engine 6

Tests regulatory compliance checking across nine frameworks, phase-down
schedule lookups, equipment ban enforcement, leak check requirements,
quota management, compliance reporting, and provenance tracking.

Target: 75+ tests, 800+ lines.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pytest

from greenlang.refrigerants_fgas.compliance_tracker import (
    ComplianceTrackerEngine,
    ComplianceRecord,
    ComplianceStatus,
    QuotaRecord,
    RegulatoryFramework,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> ComplianceTrackerEngine:
    """Create a fresh ComplianceTrackerEngine."""
    eng = ComplianceTrackerEngine()
    yield eng
    eng.clear()


@pytest.fixture
def standard_emissions() -> Decimal:
    """Standard emissions value for tests (50,000 kg CO2e = 50 tCO2e)."""
    return Decimal("50000")


@pytest.fixture
def high_emissions() -> Decimal:
    """High emissions value exceeding EPA thresholds (30,000,000 kg CO2e)."""
    return Decimal("30000000")


# ===========================================================================
# Test: Initialization
# ===========================================================================


class TestComplianceTrackerInit:
    """Tests for engine initialization."""

    def test_creation(self, engine: ComplianceTrackerEngine):
        """Engine initializes successfully."""
        assert engine is not None

    def test_repr(self, engine: ComplianceTrackerEngine):
        """Engine has a human-readable repr."""
        r = repr(engine)
        assert "ComplianceTrackerEngine" in r
        assert "frameworks=" in r
        assert "checks=" in r
        assert "quotas=" in r

    def test_len_initially_zero(self, engine: ComplianceTrackerEngine):
        """Engine starts with zero compliance checks."""
        assert len(engine) == 0

    def test_stats_initial(self, engine: ComplianceTrackerEngine):
        """get_stats returns correct initial state."""
        stats = engine.get_stats()
        assert stats["total_checks"] == 0
        assert stats["active_quotas"] == 0
        assert stats["frameworks_supported"] == 10

    def test_clear(self, engine: ComplianceTrackerEngine):
        """clear() resets compliance history and quotas."""
        engine.check_compliance(
            emissions_co2e=Decimal("1000"),
            framework="GHG_PROTOCOL",
        )
        engine.register_quota("org_test", Decimal("100000"), 2026)
        assert len(engine) == 1
        engine.clear()
        assert len(engine) == 0


# ===========================================================================
# Test: Compliance Checking per Framework
# ===========================================================================


class TestCheckComplianceByFramework:
    """Tests for check_compliance across all supported frameworks."""

    def test_check_compliance_ghg_protocol(
        self, engine: ComplianceTrackerEngine, standard_emissions: Decimal
    ):
        """GHG Protocol compliance check returns valid ComplianceRecord."""
        record = engine.check_compliance(
            emissions_co2e=standard_emissions,
            refrigerant_type="R-410A",
            framework="GHG_PROTOCOL",
        )
        assert isinstance(record, ComplianceRecord)
        assert record.framework == "GHG_PROTOCOL"
        assert record.record_id.startswith("cc_")
        assert len(record.findings) > 0
        # Should report F-gas emissions included and refrigerant type disclosed
        finding_reqs = [f["requirement"] for f in record.findings]
        assert "F-gas emissions included in Scope 1 inventory" in finding_reqs
        assert "Refrigerant type disclosure" in finding_reqs

    def test_check_compliance_iso_14064(
        self, engine: ComplianceTrackerEngine, standard_emissions: Decimal
    ):
        """ISO 14064 compliance check returns valid record."""
        record = engine.check_compliance(
            emissions_co2e=standard_emissions,
            refrigerant_type="R-134a",
            framework="ISO_14064",
        )
        assert record.framework == "ISO_14064"
        assert len(record.findings) > 0
        finding_reqs = [f["requirement"] for f in record.findings]
        assert "F-gas emissions included in Scope 1 inventory" in finding_reqs

    def test_check_compliance_csrd(
        self, engine: ComplianceTrackerEngine, standard_emissions: Decimal
    ):
        """CSRD/ESRS E1 compliance check returns valid record."""
        record = engine.check_compliance(
            emissions_co2e=standard_emissions,
            refrigerant_type="R-410A",
            framework="CSRD_ESRS_E1",
        )
        assert record.framework == "CSRD_ESRS_E1"
        assert len(record.findings) > 0

    def test_check_compliance_eu_fgas(
        self, engine: ComplianceTrackerEngine, standard_emissions: Decimal
    ):
        """EU F-Gas 2024 compliance check includes phase-down target."""
        record = engine.check_compliance(
            emissions_co2e=standard_emissions,
            refrigerant_type="R-410A",
            equipment_type="COMMERCIAL_AC_PACKAGED",
            framework="EU_FGAS_2024",
            gwp=Decimal("2088"),
            charge_co2e=Decimal("100"),
            year=2026,
        )
        assert record.framework == "EU_FGAS_2024"
        assert record.phase_down_target_pct is not None
        assert record.phase_down_target_pct == Decimal("45")
        assert record.phase_down_year == 2026

    def test_check_compliance_kigali_non_a5(
        self, engine: ComplianceTrackerEngine, standard_emissions: Decimal
    ):
        """Kigali Non-A5 compliance check includes phase-down target."""
        record = engine.check_compliance(
            emissions_co2e=standard_emissions,
            framework="KIGALI_NON_A5",
            year=2029,
        )
        assert record.framework == "KIGALI_NON_A5"
        assert record.phase_down_target_pct == Decimal("30")

    def test_check_compliance_kigali_a5(
        self, engine: ComplianceTrackerEngine, standard_emissions: Decimal
    ):
        """Kigali A5 Group 1 compliance check includes phase-down target."""
        record = engine.check_compliance(
            emissions_co2e=standard_emissions,
            framework="KIGALI_A5_G1",
            year=2032,
        )
        assert record.framework == "KIGALI_A5_G1"
        assert record.phase_down_target_pct == Decimal("70")

    def test_check_compliance_epa_dd(
        self, engine: ComplianceTrackerEngine, standard_emissions: Decimal
    ):
        """EPA Subpart DD compliance check identifies threshold."""
        # With charge_co2e above threshold
        record = engine.check_compliance(
            emissions_co2e=standard_emissions,
            framework="EPA_SUBPART_DD",
            charge_co2e=Decimal("10000"),  # above 8083 kg CO2e threshold
        )
        assert record.framework == "EPA_SUBPART_DD"
        threshold_finding = [
            f for f in record.findings
            if "Subpart DD" in f["requirement"]
        ]
        assert len(threshold_finding) == 1
        assert threshold_finding[0]["status"] == "TRIGGERED"

    def test_check_compliance_epa_oo(
        self, engine: ComplianceTrackerEngine, high_emissions: Decimal
    ):
        """EPA Subpart OO compliance check with high emissions triggers reporting."""
        record = engine.check_compliance(
            emissions_co2e=high_emissions,
            framework="EPA_SUBPART_OO",
        )
        assert record.framework == "EPA_SUBPART_OO"
        oo_finding = [
            f for f in record.findings
            if "Subpart OO" in f["requirement"]
        ]
        assert len(oo_finding) == 1
        assert oo_finding[0]["status"] == "TRIGGERED"

    def test_check_compliance_epa_l(
        self, engine: ComplianceTrackerEngine, high_emissions: Decimal
    ):
        """EPA Subpart L compliance check with high emissions triggers reporting."""
        record = engine.check_compliance(
            emissions_co2e=high_emissions,
            framework="EPA_SUBPART_L",
        )
        assert record.framework == "EPA_SUBPART_L"
        l_finding = [
            f for f in record.findings
            if "Subpart L" in f["requirement"]
        ]
        assert len(l_finding) == 1
        assert l_finding[0]["status"] == "TRIGGERED"

    def test_check_compliance_uk_fgas(
        self, engine: ComplianceTrackerEngine, standard_emissions: Decimal
    ):
        """UK F-Gas compliance check includes phase-down target."""
        record = engine.check_compliance(
            emissions_co2e=standard_emissions,
            framework="UK_FGAS",
            year=2027,
            charge_co2e=Decimal("100"),
        )
        assert record.framework == "UK_FGAS"
        assert record.phase_down_target_pct == Decimal("31")

    def test_check_compliance_invalid_framework(
        self, engine: ComplianceTrackerEngine
    ):
        """Invalid framework raises ValueError."""
        with pytest.raises(ValueError, match="Unknown framework"):
            engine.check_compliance(
                emissions_co2e=Decimal("1000"),
                framework="NONEXISTENT_FRAMEWORK",
            )

    def test_check_compliance_negative_emissions(
        self, engine: ComplianceTrackerEngine
    ):
        """Negative emissions raises ValueError."""
        with pytest.raises(ValueError, match="emissions_co2e must be >= 0"):
            engine.check_compliance(
                emissions_co2e=Decimal("-1"),
                framework="GHG_PROTOCOL",
            )


# ===========================================================================
# Test: Check All Frameworks
# ===========================================================================


class TestCheckAllFrameworks:
    """Tests for check_all_frameworks."""

    def test_check_all_frameworks(
        self, engine: ComplianceTrackerEngine, standard_emissions: Decimal
    ):
        """check_all_frameworks returns one record per framework."""
        records = engine.check_all_frameworks(
            emissions_co2e=standard_emissions,
            refrigerant_type="R-410A",
        )
        assert len(records) == 10
        frameworks_seen = {r.framework for r in records}
        assert frameworks_seen == {f.value for f in RegulatoryFramework}

    def test_check_all_frameworks_subset(
        self, engine: ComplianceTrackerEngine, standard_emissions: Decimal
    ):
        """check_all_frameworks with subset returns only specified frameworks."""
        selected = ["GHG_PROTOCOL", "ISO_14064"]
        records = engine.check_all_frameworks(
            emissions_co2e=standard_emissions,
            frameworks=selected,
        )
        assert len(records) == 2
        assert {r.framework for r in records} == set(selected)


# ===========================================================================
# Test: EU F-Gas Phase-Down Schedule
# ===========================================================================


class TestEUFGasPhaseDown:
    """Tests for EU F-Gas Regulation 2024/573 phase-down targets."""

    @pytest.mark.parametrize(
        "year,expected_pct",
        [
            (2015, Decimal("100")),
            (2016, Decimal("93")),
            (2018, Decimal("63")),
            (2021, Decimal("45")),
            (2024, Decimal("45")),
            (2027, Decimal("31")),
            (2030, Decimal("24")),
            (2033, Decimal("15")),
            (2036, Decimal("15")),
        ],
    )
    def test_eu_phase_down_target(
        self, engine: ComplianceTrackerEngine, year: int, expected_pct: Decimal
    ):
        """EU F-Gas phase-down target matches regulation schedule."""
        target = engine.get_phase_down_target("EU_FGAS_2024", year)
        assert target == expected_pct

    def test_eu_phase_down_before_schedule(
        self, engine: ComplianceTrackerEngine
    ):
        """Year before 2015 returns 100% (baseline)."""
        target = engine.get_phase_down_target("EU_FGAS_2024", 2010)
        assert target == Decimal("100")

    def test_eu_phase_down_after_terminal(
        self, engine: ComplianceTrackerEngine
    ):
        """Year after 2036 returns 15% (terminal target)."""
        target = engine.get_phase_down_target("EU_FGAS_2024", 2040)
        assert target == Decimal("15")


# ===========================================================================
# Test: Kigali Non-Article 5 Phase-Down
# ===========================================================================


class TestKigaliNonA5PhaseDown:
    """Tests for Kigali Amendment Non-A5 (developed countries) phase-down."""

    @pytest.mark.parametrize(
        "year,expected_pct",
        [
            (2019, Decimal("100")),
            (2021, Decimal("90")),
            (2024, Decimal("60")),
            (2029, Decimal("30")),
            (2034, Decimal("20")),
            (2036, Decimal("15")),
        ],
    )
    def test_kigali_non_a5_target(
        self, engine: ComplianceTrackerEngine, year: int, expected_pct: Decimal
    ):
        """Kigali Non-A5 phase-down target matches schedule."""
        target = engine.get_phase_down_target("KIGALI_NON_A5", year)
        assert target == expected_pct

    def test_kigali_non_a5_after_terminal(
        self, engine: ComplianceTrackerEngine
    ):
        """Year after 2036 returns 15% terminal target."""
        target = engine.get_phase_down_target("KIGALI_NON_A5", 2040)
        assert target == Decimal("15")


# ===========================================================================
# Test: Kigali Article 5 Group 1 Phase-Down
# ===========================================================================


class TestKigaliA5PhaseDown:
    """Tests for Kigali Amendment A5 Group 1 (developing countries) phase-down."""

    @pytest.mark.parametrize(
        "year,expected_pct",
        [
            (2024, Decimal("100")),
            (2029, Decimal("90")),
            (2032, Decimal("70")),
            (2034, Decimal("60")),
            (2036, Decimal("50")),
            (2040, Decimal("20")),
            (2045, Decimal("15")),
        ],
    )
    def test_kigali_a5_target(
        self, engine: ComplianceTrackerEngine, year: int, expected_pct: Decimal
    ):
        """Kigali A5 G1 phase-down target matches schedule."""
        target = engine.get_phase_down_target("KIGALI_A5_G1", year)
        assert target == expected_pct

    def test_kigali_a5_before_schedule(
        self, engine: ComplianceTrackerEngine
    ):
        """Year before 2024 returns 100% for A5 parties."""
        target = engine.get_phase_down_target("KIGALI_A5_G1", 2020)
        assert target == Decimal("100")

    def test_kigali_a5_after_terminal(
        self, engine: ComplianceTrackerEngine
    ):
        """Year after 2045 returns 15% terminal target."""
        target = engine.get_phase_down_target("KIGALI_A5_G1", 2050)
        assert target == Decimal("15")


# ===========================================================================
# Test: UK F-Gas Phase-Down
# ===========================================================================


class TestUKFGasPhaseDown:
    """Tests for UK F-Gas Regulations phase-down targets."""

    @pytest.mark.parametrize(
        "year,expected_pct",
        [
            (2015, Decimal("100")),
            (2018, Decimal("63")),
            (2021, Decimal("45")),
            (2024, Decimal("45")),
            (2027, Decimal("31")),
            (2030, Decimal("24")),
            (2033, Decimal("15")),
            (2036, Decimal("15")),
        ],
    )
    def test_uk_fgas_target(
        self, engine: ComplianceTrackerEngine, year: int, expected_pct: Decimal
    ):
        """UK F-Gas phase-down target matches schedule."""
        target = engine.get_phase_down_target("UK_FGAS", year)
        assert target == expected_pct


# ===========================================================================
# Test: Phase-Down Full Schedule
# ===========================================================================


class TestPhaseDownFullSchedule:
    """Tests for get_phase_down_schedule method."""

    def test_get_full_schedule_eu_fgas(self, engine: ComplianceTrackerEngine):
        """Full EU F-Gas schedule is a dict of year -> pct strings."""
        schedule = engine.get_phase_down_schedule("EU_FGAS_2024")
        assert isinstance(schedule, dict)
        assert 2015 in schedule
        assert schedule[2015] == "100%"
        assert schedule[2036] == "15%"

    def test_get_full_schedule_invalid_framework(
        self, engine: ComplianceTrackerEngine
    ):
        """Framework without phase-down schedule raises ValueError."""
        with pytest.raises(ValueError, match="does not have a phase-down schedule"):
            engine.get_phase_down_schedule("GHG_PROTOCOL")


# ===========================================================================
# Test: Equipment Ban Checking
# ===========================================================================


class TestEquipmentBanChecking:
    """Tests for EU F-Gas equipment ban enforcement."""

    def test_check_equipment_ban_high_gwp(
        self, engine: ComplianceTrackerEngine
    ):
        """High-GWP refrigerant in commercial AC is banned after 2025."""
        banned = engine.check_equipment_ban(
            refrigerant_type="R-410A",
            equipment_type="COMMERCIAL_AC_PACKAGED",
            gwp=Decimal("2088"),
            check_date=date(2025, 6, 1),
        )
        assert banned is True

    def test_check_equipment_ban_allowed(
        self, engine: ComplianceTrackerEngine
    ):
        """Low-GWP refrigerant is not banned."""
        banned = engine.check_equipment_ban(
            refrigerant_type="R-32",
            equipment_type="COMMERCIAL_AC_PACKAGED",
            gwp=Decimal("675"),
            check_date=date(2025, 6, 1),
        )
        assert banned is False

    def test_check_equipment_ban_before_effective_date(
        self, engine: ComplianceTrackerEngine
    ):
        """Check before ban effective date returns False."""
        banned = engine.check_equipment_ban(
            refrigerant_type="R-410A",
            equipment_type="COMMERCIAL_AC_PACKAGED",
            gwp=Decimal("2088"),
            check_date=date(2024, 1, 1),
        )
        assert banned is False

    def test_check_equipment_ban_no_gwp(
        self, engine: ComplianceTrackerEngine
    ):
        """No GWP provided returns False (cannot determine ban)."""
        banned = engine.check_equipment_ban(
            refrigerant_type="R-410A",
            equipment_type="COMMERCIAL_AC_PACKAGED",
            gwp=None,
        )
        assert banned is False

    def test_check_equipment_ban_no_equipment_type(
        self, engine: ComplianceTrackerEngine
    ):
        """No equipment type returns False."""
        banned = engine.check_equipment_ban(
            refrigerant_type="R-410A",
            equipment_type=None,
            gwp=Decimal("2088"),
        )
        assert banned is False

    def test_check_equipment_ban_centralized_commercial(
        self, engine: ComplianceTrackerEngine
    ):
        """Centralized commercial refrigeration with GWP >= 2500 banned from 2020."""
        banned = engine.check_equipment_ban(
            refrigerant_type="R-404A",
            equipment_type="COMMERCIAL_REFRIGERATION_CENTRALIZED",
            gwp=Decimal("3922"),
            check_date=date(2021, 1, 1),
        )
        assert banned is True

    def test_check_equipment_ban_industrial_2032(
        self, engine: ComplianceTrackerEngine
    ):
        """Industrial refrigeration with GWP >= 150 banned from 2032."""
        banned = engine.check_equipment_ban(
            equipment_type="INDUSTRIAL_REFRIGERATION",
            gwp=Decimal("2088"),
            check_date=date(2032, 6, 1),
        )
        assert banned is True

    def test_get_equipment_bans_all(
        self, engine: ComplianceTrackerEngine
    ):
        """get_equipment_bans returns the full list."""
        bans = engine.get_equipment_bans()
        assert len(bans) > 0
        assert all("equipment_category" in b for b in bans)

    def test_get_equipment_bans_filtered_by_type(
        self, engine: ComplianceTrackerEngine
    ):
        """get_equipment_bans can filter by equipment type."""
        bans = engine.get_equipment_bans(
            equipment_type="COMMERCIAL_AC_PACKAGED"
        )
        assert len(bans) >= 1
        assert all(b["equipment_category"] == "COMMERCIAL_AC_PACKAGED" for b in bans)

    def test_get_equipment_bans_filtered_by_date(
        self, engine: ComplianceTrackerEngine
    ):
        """get_equipment_bans can filter by as_of_date."""
        bans_2020 = engine.get_equipment_bans(as_of_date=date(2020, 12, 31))
        bans_2030 = engine.get_equipment_bans(as_of_date=date(2030, 12, 31))
        assert len(bans_2030) >= len(bans_2020)


# ===========================================================================
# Test: Leak Check Requirements
# ===========================================================================


class TestLeakCheckRequirements:
    """Tests for EU F-Gas leak check requirement determination."""

    def test_check_leak_check_requirement_large_system(
        self, engine: ComplianceTrackerEngine
    ):
        """System >= 500 tCO2e requires quarterly checks + automatic detection."""
        result = engine.check_leak_check_requirement(Decimal("600"))
        assert result["required"] is True
        assert result["frequency_months"] == 3
        assert result["automatic_detection_required"] is True

    def test_check_leak_check_requirement_medium_system(
        self, engine: ComplianceTrackerEngine
    ):
        """System >= 50 and < 500 tCO2e requires semi-annual checks."""
        result = engine.check_leak_check_requirement(Decimal("100"))
        assert result["required"] is True
        assert result["frequency_months"] == 6
        assert result["automatic_detection_required"] is False

    def test_check_leak_check_requirement_small_system(
        self, engine: ComplianceTrackerEngine
    ):
        """System >= 5 and < 50 tCO2e requires annual checks."""
        result = engine.check_leak_check_requirement(Decimal("20"))
        assert result["required"] is True
        assert result["frequency_months"] == 12

    def test_check_leak_check_below_threshold(
        self, engine: ComplianceTrackerEngine
    ):
        """System < 5 tCO2e does not require leak checks."""
        result = engine.check_leak_check_requirement(Decimal("3"))
        assert result["required"] is False
        assert result["frequency_months"] is None

    def test_check_leak_check_exact_threshold_5(
        self, engine: ComplianceTrackerEngine
    ):
        """System at exactly 5 tCO2e requires annual checks."""
        result = engine.check_leak_check_requirement(Decimal("5"))
        assert result["required"] is True
        assert result["frequency_months"] == 12

    def test_check_leak_check_exact_threshold_50(
        self, engine: ComplianceTrackerEngine
    ):
        """System at exactly 50 tCO2e requires semi-annual checks."""
        result = engine.check_leak_check_requirement(Decimal("50"))
        assert result["required"] is True
        assert result["frequency_months"] == 6

    def test_check_leak_check_exact_threshold_500(
        self, engine: ComplianceTrackerEngine
    ):
        """System at exactly 500 tCO2e requires quarterly checks."""
        result = engine.check_leak_check_requirement(Decimal("500"))
        assert result["required"] is True
        assert result["frequency_months"] == 3
        assert result["automatic_detection_required"] is True


# ===========================================================================
# Test: Quota Management
# ===========================================================================


class TestQuotaManagement:
    """Tests for organization quota registration and tracking."""

    def test_register_quota(self, engine: ComplianceTrackerEngine):
        """register_quota creates a new quota record."""
        key = engine.register_quota(
            organization_id="org_test",
            quota_co2e=Decimal("1000000"),
            year=2026,
            framework="EU_FGAS_2024",
        )
        assert key == "org_test:2026:EU_FGAS_2024"

    def test_register_quota_empty_org(self, engine: ComplianceTrackerEngine):
        """Empty organization_id raises ValueError."""
        with pytest.raises(ValueError, match="organization_id must not be empty"):
            engine.register_quota("", Decimal("1000"), 2026)

    def test_register_quota_zero_amount(self, engine: ComplianceTrackerEngine):
        """Zero quota amount raises ValueError."""
        with pytest.raises(ValueError, match="quota_co2e must be > 0"):
            engine.register_quota("org_test", Decimal("0"), 2026)

    def test_register_quota_negative_amount(self, engine: ComplianceTrackerEngine):
        """Negative quota amount raises ValueError."""
        with pytest.raises(ValueError, match="quota_co2e must be > 0"):
            engine.register_quota("org_test", Decimal("-500"), 2026)

    def test_update_quota_usage(self, engine: ComplianceTrackerEngine):
        """update_quota_usage adds to tracked usage."""
        engine.register_quota("org_test", Decimal("1000000"), 2026)
        result = engine.update_quota_usage(
            organization_id="org_test",
            usage_co2e=Decimal("250000"),
            year=2026,
            framework="EU_FGAS_2024",
        )
        assert Decimal(result["used_co2e"]) == Decimal("250000")
        assert Decimal(result["remaining_co2e"]) == Decimal("750000")
        assert Decimal(result["usage_pct"]) == Decimal("25.00")

    def test_update_quota_usage_unregistered(
        self, engine: ComplianceTrackerEngine
    ):
        """Updating unregistered quota raises ValueError."""
        with pytest.raises(ValueError, match="No quota registered"):
            engine.update_quota_usage("no_such_org", Decimal("100"), 2026)

    def test_update_quota_usage_negative(
        self, engine: ComplianceTrackerEngine
    ):
        """Negative usage amount raises ValueError."""
        engine.register_quota("org_test", Decimal("1000000"), 2026)
        with pytest.raises(ValueError, match="usage_co2e must be >= 0"):
            engine.update_quota_usage("org_test", Decimal("-100"), 2026)

    def test_quota_status_compliant(self, engine: ComplianceTrackerEngine):
        """Usage under 75% is 'under_quota' status."""
        engine.register_quota("org_test", Decimal("1000000"), 2026)
        engine.update_quota_usage("org_test", Decimal("500000"), 2026)
        status = engine.get_quota_status("org_test", 2026)
        assert status["status"] == "under_quota"

    def test_quota_status_elevated(self, engine: ComplianceTrackerEngine):
        """Usage between 75% and 90% is 'elevated' status."""
        engine.register_quota("org_test", Decimal("1000000"), 2026)
        engine.update_quota_usage("org_test", Decimal("800000"), 2026)
        status = engine.get_quota_status("org_test", 2026)
        assert status["status"] == "elevated"

    def test_quota_status_warning(self, engine: ComplianceTrackerEngine):
        """Usage between 90% and 100% is 'approaching_limit' status."""
        engine.register_quota("org_test", Decimal("1000000"), 2026)
        engine.update_quota_usage("org_test", Decimal("950000"), 2026)
        status = engine.get_quota_status("org_test", 2026)
        assert status["status"] == "approaching_limit"

    def test_quota_status_exceeded(self, engine: ComplianceTrackerEngine):
        """Usage exceeding 100% is 'exceeded' status."""
        engine.register_quota("org_test", Decimal("1000000"), 2026)
        engine.update_quota_usage("org_test", Decimal("1100000"), 2026)
        status = engine.get_quota_status("org_test", 2026)
        assert status["status"] == "exceeded"
        assert Decimal(status["usage_pct"]) > Decimal("100")

    def test_quota_remaining_floors_at_zero(
        self, engine: ComplianceTrackerEngine
    ):
        """Remaining quota cannot go below 0."""
        engine.register_quota("org_test", Decimal("1000000"), 2026)
        engine.update_quota_usage("org_test", Decimal("2000000"), 2026)
        status = engine.get_quota_status("org_test", 2026)
        assert Decimal(status["remaining_co2e"]) == Decimal("0")

    def test_get_quota_status_unregistered(
        self, engine: ComplianceTrackerEngine
    ):
        """Querying unregistered quota raises ValueError."""
        with pytest.raises(ValueError, match="No quota registered"):
            engine.get_quota_status("no_such_org", 2026)

    def test_cumulative_usage_updates(self, engine: ComplianceTrackerEngine):
        """Multiple usage updates accumulate."""
        engine.register_quota("org_test", Decimal("1000000"), 2026)
        engine.update_quota_usage("org_test", Decimal("200000"), 2026)
        engine.update_quota_usage("org_test", Decimal("300000"), 2026)
        status = engine.get_quota_status("org_test", 2026)
        assert Decimal(status["used_co2e"]) == Decimal("500000")


# ===========================================================================
# Test: Compliance Reporting
# ===========================================================================


class TestComplianceReporting:
    """Tests for get_compliance_report."""

    def test_get_compliance_report(self, engine: ComplianceTrackerEngine):
        """Compliance report includes summary and by-framework breakdown."""
        engine.check_compliance(Decimal("1000"), framework="GHG_PROTOCOL")
        engine.check_compliance(Decimal("1000"), framework="ISO_14064")
        report = engine.get_compliance_report()
        assert "summary" in report
        assert "by_framework" in report
        assert "recommendations" in report
        assert report["summary"]["total_checks"] == 2

    def test_get_compliance_report_empty(
        self, engine: ComplianceTrackerEngine
    ):
        """Report with no checks returns zero counts."""
        report = engine.get_compliance_report()
        assert report["summary"]["total_checks"] == 0
        assert report["summary"]["compliance_rate_pct"] == "N/A"

    def test_get_compliance_report_with_quota(
        self, engine: ComplianceTrackerEngine
    ):
        """Report includes quota summary."""
        engine.register_quota("org_test", Decimal("1000000"), 2026)
        engine.update_quota_usage("org_test", Decimal("500000"), 2026)
        report = engine.get_compliance_report(
            organization_id="org_test", year=2026
        )
        assert "quota_summary" in report
        assert len(report["quota_summary"]) > 0

    def test_get_compliance_report_framework_filter(
        self, engine: ComplianceTrackerEngine
    ):
        """Report can filter by frameworks."""
        engine.check_compliance(Decimal("1000"), framework="GHG_PROTOCOL")
        engine.check_compliance(Decimal("1000"), framework="ISO_14064")
        engine.check_compliance(Decimal("1000"), framework="CSRD_ESRS_E1")
        report = engine.get_compliance_report(
            frameworks=["GHG_PROTOCOL", "ISO_14064"]
        )
        assert report["summary"]["total_checks"] == 2


# ===========================================================================
# Test: Regulatory Requirements Mapping
# ===========================================================================


class TestRegulatoryRequirementsMapping:
    """Tests for map_to_regulatory_requirements."""

    def test_map_to_regulatory_requirements_ghg(
        self, engine: ComplianceTrackerEngine
    ):
        """GHG Protocol requirements include Scope 1 and methodology."""
        reqs = engine.map_to_regulatory_requirements("GHG_PROTOCOL")
        assert isinstance(reqs, list)
        assert len(reqs) > 0
        req_texts = [r["requirement"] for r in reqs]
        assert any("Scope 1" in t for t in req_texts)

    def test_map_to_regulatory_requirements_eu_fgas(
        self, engine: ComplianceTrackerEngine
    ):
        """EU F-Gas requirements include leak checks and phase-down."""
        reqs = engine.map_to_regulatory_requirements("EU_FGAS_2024")
        req_texts = [r["requirement"] for r in reqs]
        assert any("Leak" in t or "leak" in t for t in req_texts)
        assert any("Phase-down" in t or "phase-down" in t.lower() for t in req_texts)

    def test_map_to_regulatory_requirements_iso(
        self, engine: ComplianceTrackerEngine
    ):
        """ISO 14064 requirements include quantification and verification."""
        reqs = engine.map_to_regulatory_requirements("ISO_14064")
        assert len(reqs) > 0
        req_texts = [r["requirement"] for r in reqs]
        assert any("verification" in t.lower() for t in req_texts)

    def test_map_to_regulatory_requirements_csrd(
        self, engine: ComplianceTrackerEngine
    ):
        """CSRD/ESRS E1 requirements include climate risk reporting."""
        reqs = engine.map_to_regulatory_requirements("CSRD_ESRS_E1")
        assert len(reqs) > 0
        req_texts = [r["requirement"] for r in reqs]
        assert any("climate" in t.lower() for t in req_texts)

    def test_map_to_regulatory_requirements_invalid(
        self, engine: ComplianceTrackerEngine
    ):
        """Invalid framework raises ValueError."""
        with pytest.raises(ValueError, match="Unknown framework"):
            engine.map_to_regulatory_requirements("INVALID")

    def test_get_all_frameworks(self, engine: ComplianceTrackerEngine):
        """get_all_frameworks returns all 10 supported frameworks."""
        frameworks = engine.get_all_frameworks()
        assert len(frameworks) == 10
        assert all("framework" in f and "description" in f for f in frameworks)


# ===========================================================================
# Test: Upcoming Deadlines
# ===========================================================================


class TestUpcomingDeadlines:
    """Tests for get_upcoming_deadlines."""

    def test_get_upcoming_deadlines(self, engine: ComplianceTrackerEngine):
        """Returns deadlines sorted by date."""
        deadlines = engine.get_upcoming_deadlines(
            from_date=date(2025, 1, 1),
            within_years=10,
        )
        assert len(deadlines) > 0
        # Verify sorted by date
        dates = [d["date"] for d in deadlines]
        assert dates == sorted(dates)

    def test_get_upcoming_deadlines_includes_phase_down(
        self, engine: ComplianceTrackerEngine
    ):
        """Deadlines include phase-down step changes."""
        deadlines = engine.get_upcoming_deadlines(
            from_date=date(2026, 1, 1),
            within_years=5,
        )
        phase_down_events = [
            d for d in deadlines if d["type"] == "phase_down_step"
        ]
        assert len(phase_down_events) > 0

    def test_get_upcoming_deadlines_includes_equipment_bans(
        self, engine: ComplianceTrackerEngine
    ):
        """Deadlines include equipment ban effective dates."""
        deadlines = engine.get_upcoming_deadlines(
            from_date=date(2025, 1, 1),
            within_years=10,
        )
        ban_events = [d for d in deadlines if d["type"] == "equipment_ban"]
        assert len(ban_events) > 0

    def test_get_upcoming_deadlines_framework_filter(
        self, engine: ComplianceTrackerEngine
    ):
        """Deadlines can be filtered by framework."""
        deadlines = engine.get_upcoming_deadlines(
            frameworks=["KIGALI_NON_A5"],
            from_date=date(2026, 1, 1),
            within_years=10,
        )
        for d in deadlines:
            assert d["framework"] == "KIGALI_NON_A5"


# ===========================================================================
# Test: Provenance Tracking
# ===========================================================================


class TestProvenanceTracking:
    """Tests for provenance hash generation and audit trail."""

    def test_provenance_hash_present(
        self, engine: ComplianceTrackerEngine
    ):
        """ComplianceRecord includes a 64-char SHA-256 provenance hash."""
        record = engine.check_compliance(
            emissions_co2e=Decimal("1000"),
            framework="GHG_PROTOCOL",
        )
        assert len(record.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in record.provenance_hash)

    def test_provenance_hash_deterministic(
        self, engine: ComplianceTrackerEngine
    ):
        """Same inputs produce the same provenance hash."""
        # Note: check_date defaults to today and timestamp varies,
        # but the provenance data dict is built from fixed params
        r1 = engine.check_compliance(
            emissions_co2e=Decimal("1000"),
            framework="GHG_PROTOCOL",
            refrigerant_type="R-410A",
            year=2026,
            check_date=date(2026, 1, 15),
        )
        r2 = engine.check_compliance(
            emissions_co2e=Decimal("1000"),
            framework="GHG_PROTOCOL",
            refrigerant_type="R-410A",
            year=2026,
            check_date=date(2026, 1, 15),
        )
        assert r1.provenance_hash == r2.provenance_hash

    def test_timestamp_present(self, engine: ComplianceTrackerEngine):
        """ComplianceRecord includes a timestamp string."""
        record = engine.check_compliance(
            emissions_co2e=Decimal("1000"),
            framework="GHG_PROTOCOL",
        )
        assert record.timestamp is not None
        assert len(record.timestamp) > 0

    def test_to_dict_serializable(self, engine: ComplianceTrackerEngine):
        """ComplianceRecord.to_dict() produces a JSON-serializable dict."""
        record = engine.check_compliance(
            emissions_co2e=Decimal("1000"),
            framework="GHG_PROTOCOL",
        )
        d = record.to_dict()
        assert isinstance(d, dict)
        assert d["framework"] == "GHG_PROTOCOL"
        assert d["provenance_hash"] is not None


# ===========================================================================
# Test: Compliance Status Values
# ===========================================================================


class TestComplianceStatusValues:
    """Tests for ComplianceStatus enum values."""

    def test_compliance_status_values(self):
        """All expected status values exist."""
        assert ComplianceStatus.COMPLIANT.value == "COMPLIANT"
        assert ComplianceStatus.NON_COMPLIANT.value == "NON_COMPLIANT"
        assert ComplianceStatus.PARTIALLY_COMPLIANT.value == "PARTIALLY_COMPLIANT"
        assert ComplianceStatus.NOT_APPLICABLE.value == "NOT_APPLICABLE"
        assert ComplianceStatus.PENDING_REVIEW.value == "PENDING_REVIEW"
        assert ComplianceStatus.WARNING.value == "WARNING"

    def test_regulatory_framework_values(self):
        """All 10 framework enum values are present."""
        expected = {
            "EU_FGAS_2024", "KIGALI_NON_A5", "KIGALI_A5_G1",
            "EPA_SUBPART_DD", "EPA_SUBPART_OO", "EPA_SUBPART_L",
            "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS_E1", "UK_FGAS",
        }
        actual = {f.value for f in RegulatoryFramework}
        assert actual == expected


# ===========================================================================
# Test: History
# ===========================================================================


class TestHistory:
    """Tests for compliance check history management."""

    def test_get_history_all(self, engine: ComplianceTrackerEngine):
        """get_history returns all entries."""
        engine.check_compliance(Decimal("1000"), framework="GHG_PROTOCOL")
        engine.check_compliance(Decimal("1000"), framework="ISO_14064")
        engine.check_compliance(Decimal("1000"), framework="CSRD_ESRS_E1")
        history = engine.get_history()
        assert len(history) == 3

    def test_get_history_by_framework(
        self, engine: ComplianceTrackerEngine
    ):
        """get_history can filter by framework."""
        engine.check_compliance(Decimal("1000"), framework="GHG_PROTOCOL")
        engine.check_compliance(Decimal("1000"), framework="ISO_14064")
        ghg_only = engine.get_history(framework="GHG_PROTOCOL")
        assert len(ghg_only) == 1
        assert ghg_only[0].framework == "GHG_PROTOCOL"

    def test_get_history_by_status(
        self, engine: ComplianceTrackerEngine
    ):
        """get_history can filter by compliance status."""
        engine.check_compliance(
            emissions_co2e=Decimal("1000"),
            refrigerant_type="R-410A",
            framework="GHG_PROTOCOL",
        )
        history = engine.get_history(status="COMPLIANT")
        # All GHG Protocol checks with valid emissions + refrigerant type should be COMPLIANT
        for r in history:
            assert r.status == "COMPLIANT"

    def test_get_history_with_limit(
        self, engine: ComplianceTrackerEngine
    ):
        """get_history respects limit parameter."""
        for _ in range(5):
            engine.check_compliance(Decimal("1000"), framework="GHG_PROTOCOL")
        limited = engine.get_history(limit=2)
        assert len(limited) == 2

    def test_stats_after_checks(self, engine: ComplianceTrackerEngine):
        """get_stats reflects checks performed."""
        engine.check_compliance(Decimal("1000"), framework="GHG_PROTOCOL")
        engine.check_compliance(Decimal("1000"), framework="ISO_14064")
        stats = engine.get_stats()
        assert stats["total_checks"] == 2
        assert "GHG_PROTOCOL" in stats["checks_by_framework"]
        assert "ISO_14064" in stats["checks_by_framework"]


# ===========================================================================
# Test: Compliance Check with Quota Integration
# ===========================================================================


class TestComplianceWithQuota:
    """Tests for compliance check with registered quota."""

    def test_compliance_check_quota_compliant(
        self, engine: ComplianceTrackerEngine
    ):
        """Compliance check shows compliant when under quota."""
        engine.register_quota("org_q", Decimal("1000000"), 2026, "EU_FGAS_2024")
        engine.update_quota_usage("org_q", Decimal("400000"), 2026, "EU_FGAS_2024")
        record = engine.check_compliance(
            emissions_co2e=Decimal("5000"),
            framework="EU_FGAS_2024",
            year=2026,
            organization_id="org_q",
        )
        quota_findings = [f for f in record.findings if "Quota" in f.get("requirement", "")]
        assert len(quota_findings) == 1
        assert quota_findings[0]["status"] == ComplianceStatus.COMPLIANT.value

    def test_compliance_check_quota_warning(
        self, engine: ComplianceTrackerEngine
    ):
        """Compliance check shows WARNING when quota usage > 90%."""
        engine.register_quota("org_w", Decimal("1000000"), 2026, "EU_FGAS_2024")
        engine.update_quota_usage("org_w", Decimal("920000"), 2026, "EU_FGAS_2024")
        record = engine.check_compliance(
            emissions_co2e=Decimal("5000"),
            framework="EU_FGAS_2024",
            year=2026,
            organization_id="org_w",
        )
        quota_findings = [f for f in record.findings if "Quota" in f.get("requirement", "")]
        assert len(quota_findings) == 1
        assert quota_findings[0]["status"] == ComplianceStatus.WARNING.value
        assert record.status in {ComplianceStatus.WARNING.value, ComplianceStatus.PENDING_REVIEW.value}

    def test_compliance_check_quota_exceeded(
        self, engine: ComplianceTrackerEngine
    ):
        """Compliance check shows NON_COMPLIANT when quota exceeded."""
        engine.register_quota("org_x", Decimal("1000000"), 2026, "EU_FGAS_2024")
        engine.update_quota_usage("org_x", Decimal("1100000"), 2026, "EU_FGAS_2024")
        record = engine.check_compliance(
            emissions_co2e=Decimal("5000"),
            framework="EU_FGAS_2024",
            year=2026,
            organization_id="org_x",
        )
        assert record.status == ComplianceStatus.NON_COMPLIANT.value
        quota_findings = [f for f in record.findings if "Quota" in f.get("requirement", "")]
        assert len(quota_findings) == 1
        assert quota_findings[0]["status"] == ComplianceStatus.NON_COMPLIANT.value


# ===========================================================================
# Test: EPA Below Threshold
# ===========================================================================


class TestEPABelowThreshold:
    """Tests for EPA framework checks when below reporting threshold."""

    def test_epa_dd_below_threshold(self, engine: ComplianceTrackerEngine):
        """EPA DD check shows NOT_APPLICABLE when below threshold."""
        record = engine.check_compliance(
            emissions_co2e=Decimal("1000"),
            framework="EPA_SUBPART_DD",
            charge_co2e=Decimal("1000"),  # below 8083 kg CO2e
        )
        dd_finding = [f for f in record.findings if "Subpart DD" in f["requirement"]]
        assert len(dd_finding) == 1
        assert dd_finding[0]["status"] == ComplianceStatus.NOT_APPLICABLE.value

    def test_epa_oo_below_threshold(self, engine: ComplianceTrackerEngine):
        """EPA OO check shows NOT_APPLICABLE when below threshold."""
        record = engine.check_compliance(
            emissions_co2e=Decimal("1000000"),  # 1000 tCO2e < 25000 threshold
            framework="EPA_SUBPART_OO",
        )
        oo_finding = [f for f in record.findings if "Subpart OO" in f["requirement"]]
        assert len(oo_finding) == 1
        assert oo_finding[0]["status"] == ComplianceStatus.NOT_APPLICABLE.value

    def test_epa_l_below_threshold(self, engine: ComplianceTrackerEngine):
        """EPA L check shows NOT_APPLICABLE when below threshold."""
        record = engine.check_compliance(
            emissions_co2e=Decimal("1000000"),  # 1000 tCO2e < 25000 threshold
            framework="EPA_SUBPART_L",
        )
        l_finding = [f for f in record.findings if "Subpart L" in f["requirement"]]
        assert len(l_finding) == 1
        assert l_finding[0]["status"] == ComplianceStatus.NOT_APPLICABLE.value


# ===========================================================================
# Test: Auto-Compute charge_co2e
# ===========================================================================


class TestAutoComputeChargeCO2e:
    """Tests for automatic charge_co2e computation from charge_kg and gwp."""

    def test_charge_co2e_auto_computed(
        self, engine: ComplianceTrackerEngine
    ):
        """charge_co2e is auto-computed when charge_kg and gwp are provided."""
        record = engine.check_compliance(
            emissions_co2e=Decimal("50000"),
            framework="EU_FGAS_2024",
            charge_kg=Decimal("20"),
            gwp=Decimal("2088"),
            year=2026,
        )
        # 20 kg * 2088 / 1000 = 41.76 tCO2e => requires annual leak checks
        assert record.leak_check_required is True

    def test_charge_co2e_explicit_overrides_auto(
        self, engine: ComplianceTrackerEngine
    ):
        """Explicit charge_co2e takes precedence over auto-computation."""
        record = engine.check_compliance(
            emissions_co2e=Decimal("50000"),
            framework="EU_FGAS_2024",
            charge_kg=Decimal("20"),
            gwp=Decimal("2088"),
            charge_co2e=Decimal("2"),  # Explicitly set below threshold
            year=2026,
        )
        # Should use the explicit 2 tCO2e (below 5 threshold)
        assert record.leak_check_required is False


# ===========================================================================
# Test: GHG/ISO/CSRD No Refrigerant Type
# ===========================================================================


class TestNoRefrigerantType:
    """Tests for compliance checks without refrigerant type disclosure."""

    def test_ghg_no_refrigerant_type(self, engine: ComplianceTrackerEngine):
        """GHG Protocol check without refrigerant type reports PENDING_REVIEW."""
        record = engine.check_compliance(
            emissions_co2e=Decimal("1000"),
            framework="GHG_PROTOCOL",
            refrigerant_type=None,
        )
        disclosure_findings = [
            f for f in record.findings
            if "Refrigerant type disclosure" in f.get("requirement", "")
        ]
        assert len(disclosure_findings) == 1
        assert disclosure_findings[0]["status"] == ComplianceStatus.PENDING_REVIEW.value
