# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Acquisition & Divestiture Engine Tests

Tests acquisition pro-rata calculation, divestiture pro-rata calculation,
base year restatement triggers, organic vs structural growth separation,
merger integration, JV formation/dissolution, structural change records,
and provenance tracking.

Target: 50-70 tests.
"""

import pytest
from decimal import Decimal
from datetime import date

from engines.acquisition_divestiture_engine import (
    AcquisitionDivestitureEngine,
    MnAEvent,
    ProRataCalculation,
    BaseYearRestatement,
    StructuralChangeRecord,
    OrganicGrowthAnalysis,
    MnAEventType,
    RestatementTrigger,
    _round2,
    _round4,
    _days_in_year,
)


@pytest.fixture
def engine():
    """Fresh AcquisitionDivestitureEngine."""
    return AcquisitionDivestitureEngine()


@pytest.fixture
def engine_low_threshold():
    """Engine with 1% significance threshold."""
    return AcquisitionDivestitureEngine(significance_threshold_pct=Decimal("1"))


@pytest.fixture
def acquisition_event_data():
    """Standard mid-year acquisition event data."""
    return {
        "event_type": "ACQUISITION",
        "entity_id": "ENT-ACQUIRED",
        "effective_date": date(2025, 7, 1),
        "reporting_year": 2025,
        "annual_emissions_tco2e": Decimal("10000"),
        "scope1_tco2e": Decimal("4000"),
        "scope2_location_tco2e": Decimal("3000"),
        "scope2_market_tco2e": Decimal("2800"),
        "scope3_tco2e": Decimal("3000"),
        "equity_pct": Decimal("100"),
    }


@pytest.fixture
def divestiture_event_data():
    """Standard mid-year divestiture event data."""
    return {
        "event_type": "DIVESTITURE",
        "entity_id": "ENT-DIVESTED",
        "effective_date": date(2025, 4, 1),
        "reporting_year": 2025,
        "annual_emissions_tco2e": Decimal("8000"),
        "scope1_tco2e": Decimal("3000"),
        "scope2_location_tco2e": Decimal("2500"),
        "scope2_market_tco2e": Decimal("2300"),
        "scope3_tco2e": Decimal("2500"),
        "equity_pct": Decimal("100"),
    }


class TestEventRegistration:
    """Test M&A event registration."""

    def test_register_acquisition(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        assert isinstance(event, MnAEvent)
        assert event.event_type == MnAEventType.ACQUISITION.value
        assert event.entity_id == "ENT-ACQUIRED"
        assert event.annual_emissions_tco2e == Decimal("10000")

    def test_register_divestiture(self, engine, divestiture_event_data):
        event = engine.register_event(divestiture_event_data)
        assert event.event_type == MnAEventType.DIVESTITURE.value

    def test_register_merger(self, engine):
        event = engine.register_event({
            "event_type": "MERGER",
            "entity_id": "ENT-MERGED",
            "effective_date": date(2025, 1, 1),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("5000"),
        })
        assert event.event_type == MnAEventType.MERGER.value

    def test_register_jv_formation(self, engine):
        event = engine.register_event({
            "event_type": "JV_FORMATION",
            "entity_id": "ENT-JV-NEW",
            "effective_date": date(2025, 6, 1),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("6000"),
            "equity_pct": Decimal("50"),
        })
        assert event.event_type == MnAEventType.JV_FORMATION.value
        assert event.equity_pct == Decimal("50")

    def test_register_jv_dissolution(self, engine):
        event = engine.register_event({
            "event_type": "JV_DISSOLUTION",
            "entity_id": "ENT-JV-OLD",
            "effective_date": date(2025, 9, 30),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("4000"),
            "equity_pct": Decimal("50"),
        })
        assert event.event_type == MnAEventType.JV_DISSOLUTION.value

    def test_register_outsourcing(self, engine):
        event = engine.register_event({
            "event_type": "OUTSOURCING",
            "entity_id": "ENT-OUTSOURCED",
            "effective_date": date(2025, 3, 1),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("2000"),
        })
        assert event.event_type == MnAEventType.OUTSOURCING.value

    def test_register_insourcing(self, engine):
        event = engine.register_event({
            "event_type": "INSOURCING",
            "entity_id": "ENT-INSOURCED",
            "effective_date": date(2025, 5, 1),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("3000"),
        })
        assert event.event_type == MnAEventType.INSOURCING.value

    def test_event_provenance_hash(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        assert len(event.provenance_hash) == 64

    def test_event_provenance_deterministic(self, engine):
        data1 = {
            "event_type": "ACQUISITION",
            "entity_id": "ENT-X",
            "effective_date": date(2025, 7, 1),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("10000"),
        }
        # Two separate engines produce consistent hashes for same data
        eng2 = AcquisitionDivestitureEngine()
        e1 = engine.register_event(dict(data1, event_id="EVT-001"))
        e2 = eng2.register_event(dict(data1, event_id="EVT-001"))
        assert e1.provenance_hash == e2.provenance_hash

    def test_invalid_event_type_raises(self, engine):
        with pytest.raises(ValueError, match="Invalid event_type"):
            engine.register_event({
                "event_type": "INVALID_TYPE",
                "entity_id": "ENT-X",
                "effective_date": date(2025, 1, 1),
                "reporting_year": 2025,
            })

    def test_string_date_accepted(self, engine):
        event = engine.register_event({
            "event_type": "ACQUISITION",
            "entity_id": "ENT-X",
            "effective_date": "2025-07-01",
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("1000"),
        })
        assert event.effective_date == date(2025, 7, 1)


class TestAcquisitionProRata:
    """Test acquisition pro-rata calculation."""

    def test_mid_year_acquisition_factor(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        calc = engine.calculate_prorate(event.event_id)
        assert isinstance(calc, ProRataCalculation)
        # Jul 1 to Dec 31 = 184 days out of 365
        assert calc.days_included == 184
        assert calc.total_days_in_period == 365
        expected_factor = _round4(Decimal("184") / Decimal("365"))
        assert calc.pro_rata_factor == expected_factor

    def test_mid_year_acquisition_prorated_total(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        calc = engine.calculate_prorate(event.event_id)
        expected_factor = _round4(Decimal("184") / Decimal("365"))
        expected = _round2(Decimal("10000") * expected_factor)
        assert calc.prorated_emissions_tco2e == expected

    def test_acquisition_scope1_prorated(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        calc = engine.calculate_prorate(event.event_id)
        expected_factor = _round4(Decimal("184") / Decimal("365"))
        expected_s1 = _round2(Decimal("4000") * expected_factor)
        assert calc.scope1_prorated == expected_s1

    def test_jan_1_acquisition_full_year(self, engine):
        event = engine.register_event({
            "event_type": "ACQUISITION",
            "entity_id": "ENT-A",
            "effective_date": date(2025, 1, 1),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("10000"),
        })
        calc = engine.calculate_prorate(event.event_id)
        assert calc.days_included == 365
        assert calc.pro_rata_factor == Decimal("1.0000")
        assert calc.prorated_emissions_tco2e == Decimal("10000.00")

    def test_dec_31_acquisition_one_day(self, engine):
        event = engine.register_event({
            "event_type": "ACQUISITION",
            "entity_id": "ENT-A",
            "effective_date": date(2025, 12, 31),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("10000"),
        })
        calc = engine.calculate_prorate(event.event_id)
        assert calc.days_included == 1
        expected_factor = _round4(Decimal("1") / Decimal("365"))
        assert calc.pro_rata_factor == expected_factor

    def test_partial_equity_acquisition(self, engine):
        event = engine.register_event({
            "event_type": "ACQUISITION",
            "entity_id": "ENT-PARTIAL",
            "effective_date": date(2025, 7, 1),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("10000"),
            "equity_pct": Decimal("60"),
        })
        calc = engine.calculate_prorate(event.event_id)
        factor = _round4(Decimal("184") / Decimal("365"))
        expected = _round2(Decimal("10000") * factor * Decimal("60") / Decimal("100"))
        assert calc.prorated_emissions_tco2e == expected
        assert calc.equity_pct == Decimal("60")

    def test_prorate_provenance_hash(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        calc = engine.calculate_prorate(event.event_id)
        assert len(calc.provenance_hash) == 64

    def test_prorate_unknown_event_raises(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.calculate_prorate("NONEXISTENT")


class TestDivestitureProRata:
    """Test divestiture pro-rata calculation."""

    def test_divestiture_includes_start_to_date(self, engine, divestiture_event_data):
        event = engine.register_event(divestiture_event_data)
        calc = engine.calculate_prorate(event.event_id)
        # Jan 1 to Apr 1 = 91 days
        assert calc.days_included == 91
        assert calc.total_days_in_period == 365

    def test_divestiture_prorated_total(self, engine, divestiture_event_data):
        event = engine.register_event(divestiture_event_data)
        calc = engine.calculate_prorate(event.event_id)
        factor = _round4(Decimal("91") / Decimal("365"))
        expected = _round2(Decimal("8000") * factor)
        assert calc.prorated_emissions_tco2e == expected

    def test_jan_1_divestiture_one_day(self, engine):
        event = engine.register_event({
            "event_type": "DIVESTITURE",
            "entity_id": "ENT-D",
            "effective_date": date(2025, 1, 1),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("10000"),
        })
        calc = engine.calculate_prorate(event.event_id)
        assert calc.days_included == 1

    def test_dec_31_divestiture_full_year(self, engine):
        event = engine.register_event({
            "event_type": "DIVESTITURE",
            "entity_id": "ENT-D",
            "effective_date": date(2025, 12, 31),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("10000"),
        })
        calc = engine.calculate_prorate(event.event_id)
        assert calc.days_included == 365

    def test_demerger_prorata_like_divestiture(self, engine):
        event = engine.register_event({
            "event_type": "DEMERGER",
            "entity_id": "ENT-DEMERGE",
            "effective_date": date(2025, 6, 30),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("5000"),
        })
        calc = engine.calculate_prorate(event.event_id)
        # Jan 1 to Jun 30 = 181 days
        assert calc.days_included == 181


class TestBaseYearRestatement:
    """Test base year restatement triggers."""

    def test_acquisition_restatement_adds(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        restatement = engine.trigger_base_year_restatement(
            event.event_id,
            base_year=2020,
            base_year_total_tco2e=Decimal("100000"),
        )
        assert isinstance(restatement, BaseYearRestatement)
        assert restatement.adjustment_tco2e == Decimal("10000.00")
        assert restatement.restated_total_tco2e == Decimal("110000.00")

    def test_divestiture_restatement_subtracts(self, engine, divestiture_event_data):
        event = engine.register_event(divestiture_event_data)
        restatement = engine.trigger_base_year_restatement(
            event.event_id,
            base_year=2020,
            base_year_total_tco2e=Decimal("100000"),
        )
        assert restatement.adjustment_tco2e == Decimal("-8000.00")
        assert restatement.restated_total_tco2e == Decimal("92000.00")

    def test_significance_above_threshold(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        restatement = engine.trigger_base_year_restatement(
            event.event_id,
            base_year=2020,
            base_year_total_tco2e=Decimal("100000"),
        )
        # 10000/100000 = 10% > 5% threshold
        assert restatement.is_significant is True
        assert restatement.significance_pct == Decimal("10.00")

    def test_significance_below_threshold(self, engine):
        event = engine.register_event({
            "event_type": "ACQUISITION",
            "entity_id": "ENT-SMALL",
            "effective_date": date(2025, 7, 1),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("100"),
        })
        restatement = engine.trigger_base_year_restatement(
            event.event_id,
            base_year=2020,
            base_year_total_tco2e=Decimal("100000"),
        )
        # 100/100000 = 0.10% < 5%
        assert restatement.is_significant is False

    def test_custom_threshold(self, engine_low_threshold):
        event = engine_low_threshold.register_event({
            "event_type": "ACQUISITION",
            "entity_id": "ENT-X",
            "effective_date": date(2025, 7, 1),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("1500"),
        })
        restatement = engine_low_threshold.trigger_base_year_restatement(
            event.event_id,
            base_year=2020,
            base_year_total_tco2e=Decimal("100000"),
        )
        # 1500/100000 = 1.5% > 1% threshold
        assert restatement.is_significant is True

    def test_restatement_scope_level_detail(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        restatement = engine.trigger_base_year_restatement(
            event.event_id,
            base_year=2020,
            base_year_total_tco2e=Decimal("100000"),
            base_year_scope1=Decimal("40000"),
            base_year_scope2_location=Decimal("30000"),
        )
        assert restatement.restated_scope1 == Decimal("44000.00")
        assert restatement.restated_scope2_location == Decimal("33000.00")

    def test_restatement_provenance_hash(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        restatement = engine.trigger_base_year_restatement(
            event.event_id,
            base_year=2020,
            base_year_total_tco2e=Decimal("100000"),
        )
        assert len(restatement.provenance_hash) == 64

    def test_restatement_trigger_structural(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        restatement = engine.trigger_base_year_restatement(
            event.event_id,
            base_year=2020,
            base_year_total_tco2e=Decimal("100000"),
        )
        assert restatement.trigger == RestatementTrigger.STRUCTURAL_CHANGE.value

    def test_restatement_unknown_event_raises(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.trigger_base_year_restatement(
                "NONEXISTENT",
                base_year=2020,
                base_year_total_tco2e=Decimal("100000"),
            )

    def test_partial_equity_restatement(self, engine):
        event = engine.register_event({
            "event_type": "ACQUISITION",
            "entity_id": "ENT-P",
            "effective_date": date(2025, 7, 1),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("10000"),
            "scope1_tco2e": Decimal("5000"),
            "equity_pct": Decimal("60"),
        })
        restatement = engine.trigger_base_year_restatement(
            event.event_id,
            base_year=2020,
            base_year_total_tco2e=Decimal("100000"),
            base_year_scope1=Decimal("50000"),
        )
        # 10000 * 60/100 = 6000
        assert restatement.adjustment_tco2e == Decimal("6000.00")
        assert restatement.restated_scope1 == Decimal("53000.00")


class TestOrganicVsStructuralGrowth:
    """Test organic vs structural growth separation."""

    def test_organic_growth_calculation(self, engine):
        analysis = engine.separate_organic_structural(
            reporting_year=2025,
            base_year=2020,
            base_year_original_tco2e=Decimal("100000"),
            base_year_adjusted_tco2e=Decimal("110000"),
            current_year_tco2e=Decimal("115000"),
        )
        assert isinstance(analysis, OrganicGrowthAnalysis)
        assert analysis.structural_change_tco2e == Decimal("10000.00")
        assert analysis.organic_change_tco2e == Decimal("5000.00")
        assert analysis.total_change_tco2e == Decimal("15000.00")

    def test_organic_growth_percentage(self, engine):
        analysis = engine.separate_organic_structural(
            reporting_year=2025,
            base_year=2020,
            base_year_original_tco2e=Decimal("100000"),
            base_year_adjusted_tco2e=Decimal("110000"),
            current_year_tco2e=Decimal("115000"),
        )
        # organic = 5000 / 110000 * 100 = 4.55%
        expected = _round2(Decimal("5000") / Decimal("110000") * Decimal("100"))
        assert analysis.organic_change_pct == expected

    def test_structural_growth_percentage(self, engine):
        analysis = engine.separate_organic_structural(
            reporting_year=2025,
            base_year=2020,
            base_year_original_tco2e=Decimal("100000"),
            base_year_adjusted_tco2e=Decimal("110000"),
            current_year_tco2e=Decimal("115000"),
        )
        # structural = 10000 / 100000 * 100 = 10.00%
        assert analysis.structural_change_pct == Decimal("10.00")

    def test_negative_organic_growth(self, engine):
        analysis = engine.separate_organic_structural(
            reporting_year=2025,
            base_year=2020,
            base_year_original_tco2e=Decimal("100000"),
            base_year_adjusted_tco2e=Decimal("110000"),
            current_year_tco2e=Decimal("105000"),
        )
        # organic = 105000 - 110000 = -5000
        assert analysis.organic_change_tco2e == Decimal("-5000.00")

    def test_zero_structural_change(self, engine):
        analysis = engine.separate_organic_structural(
            reporting_year=2025,
            base_year=2020,
            base_year_original_tco2e=Decimal("100000"),
            base_year_adjusted_tco2e=Decimal("100000"),
            current_year_tco2e=Decimal("95000"),
        )
        assert analysis.structural_change_tco2e == Decimal("0.00")
        assert analysis.organic_change_tco2e == Decimal("-5000.00")

    def test_growth_analysis_provenance_hash(self, engine):
        analysis = engine.separate_organic_structural(
            reporting_year=2025,
            base_year=2020,
            base_year_original_tco2e=Decimal("100000"),
            base_year_adjusted_tco2e=Decimal("110000"),
            current_year_tco2e=Decimal("115000"),
        )
        assert len(analysis.provenance_hash) == 64


class TestMnATimeline:
    """Test M&A timeline retrieval."""

    def test_timeline_sorted_by_date(self, engine):
        engine.register_event({
            "event_type": "ACQUISITION", "entity_id": "A",
            "effective_date": date(2025, 9, 1), "reporting_year": 2025,
        })
        engine.register_event({
            "event_type": "DIVESTITURE", "entity_id": "B",
            "effective_date": date(2025, 3, 1), "reporting_year": 2025,
        })
        engine.register_event({
            "event_type": "MERGER", "entity_id": "C",
            "effective_date": date(2025, 6, 1), "reporting_year": 2025,
        })
        timeline = engine.get_mna_timeline(reporting_year=2025)
        assert len(timeline) == 3
        assert timeline[0].effective_date < timeline[1].effective_date
        assert timeline[1].effective_date < timeline[2].effective_date

    def test_timeline_filter_by_year(self, engine):
        engine.register_event({
            "event_type": "ACQUISITION", "entity_id": "A",
            "effective_date": date(2025, 1, 1), "reporting_year": 2025,
        })
        engine.register_event({
            "event_type": "ACQUISITION", "entity_id": "B",
            "effective_date": date(2024, 6, 1), "reporting_year": 2024,
        })
        assert len(engine.get_mna_timeline(reporting_year=2025)) == 1
        assert len(engine.get_mna_timeline(reporting_year=2024)) == 1

    def test_timeline_filter_by_entity(self, engine):
        engine.register_event({
            "event_type": "ACQUISITION", "entity_id": "A",
            "effective_date": date(2025, 1, 1), "reporting_year": 2025,
        })
        engine.register_event({
            "event_type": "DIVESTITURE", "entity_id": "B",
            "effective_date": date(2025, 6, 1), "reporting_year": 2025,
        })
        assert len(engine.get_mna_timeline(entity_id="A")) == 1

    def test_empty_timeline(self, engine):
        assert len(engine.get_mna_timeline()) == 0


class TestStructuralChangeRecord:
    """Test structural change record generation."""

    def test_structural_record_counts(self, engine):
        engine.register_event({
            "event_type": "ACQUISITION", "entity_id": "A",
            "effective_date": date(2025, 3, 1), "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("5000"),
        })
        engine.register_event({
            "event_type": "DIVESTITURE", "entity_id": "B",
            "effective_date": date(2025, 6, 1), "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("3000"),
        })
        record = engine.get_structural_change_record(2025)
        assert isinstance(record, StructuralChangeRecord)
        assert record.total_events == 2
        assert record.acquisitions_count == 1
        assert record.divestitures_count == 1

    def test_structural_record_net_impact(self, engine):
        evt1 = engine.register_event({
            "event_type": "ACQUISITION", "entity_id": "A",
            "effective_date": date(2025, 7, 1), "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("10000"),
        })
        engine.calculate_prorate(evt1.event_id)

        evt2 = engine.register_event({
            "event_type": "DIVESTITURE", "entity_id": "B",
            "effective_date": date(2025, 7, 1), "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("6000"),
        })
        engine.calculate_prorate(evt2.event_id)

        record = engine.get_structural_change_record(2025)
        # acquisition adds prorated, divestiture subtracts prorated
        assert record.net_emission_impact_tco2e != Decimal("0")

    def test_structural_record_provenance_hash(self, engine):
        engine.register_event({
            "event_type": "ACQUISITION", "entity_id": "A",
            "effective_date": date(2025, 1, 1), "reporting_year": 2025,
        })
        record = engine.get_structural_change_record(2025)
        assert len(record.provenance_hash) == 64

    def test_empty_structural_record(self, engine):
        record = engine.get_structural_change_record(2025)
        assert record.total_events == 0
        assert record.net_emission_impact_tco2e == Decimal("0")


class TestAccessors:
    """Test accessor methods."""

    def test_get_event(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        retrieved = engine.get_event(event.event_id)
        assert retrieved.entity_id == "ENT-ACQUIRED"

    def test_get_event_not_found(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.get_event("NONEXISTENT")

    def test_get_all_events(self, engine):
        engine.register_event({
            "event_type": "ACQUISITION", "entity_id": "A",
            "effective_date": date(2025, 1, 1), "reporting_year": 2025,
        })
        engine.register_event({
            "event_type": "DIVESTITURE", "entity_id": "B",
            "effective_date": date(2025, 6, 1), "reporting_year": 2025,
        })
        assert len(engine.get_all_events()) == 2

    def test_get_prorate(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        calc = engine.calculate_prorate(event.event_id)
        retrieved = engine.get_prorate(calc.calculation_id)
        assert retrieved.entity_id == "ENT-ACQUIRED"

    def test_get_prorate_not_found(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.get_prorate("NONEXISTENT")

    def test_get_restatement(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        restatement = engine.trigger_base_year_restatement(
            event.event_id, base_year=2020,
            base_year_total_tco2e=Decimal("100000"),
        )
        retrieved = engine.get_restatement(restatement.restatement_id)
        assert retrieved.base_year == 2020

    def test_get_restatement_not_found(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.get_restatement("NONEXISTENT")

    def test_get_all_restatements(self, engine, acquisition_event_data):
        event = engine.register_event(acquisition_event_data)
        engine.trigger_base_year_restatement(
            event.event_id, base_year=2020,
            base_year_total_tco2e=Decimal("100000"),
        )
        assert len(engine.get_all_restatements()) == 1


class TestDaysInYear:
    """Test _days_in_year helper."""

    def test_normal_year(self):
        assert _days_in_year(2025) == 365

    def test_leap_year(self):
        assert _days_in_year(2024) == 366

    def test_century_non_leap(self):
        assert _days_in_year(1900) == 365

    def test_century_leap(self):
        assert _days_in_year(2000) == 366
