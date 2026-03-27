# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - End-to-End Tests

Tests full consolidation scenarios from entity registration through
to final report generation and assurance package. Validates the
complete pipeline across all 10 engines.

Target: 30-50 tests.
"""

import pytest
from decimal import Decimal
from datetime import date

from engines.entity_registry_engine import EntityRegistryEngine
from engines.ownership_structure_engine import (
    OwnershipStructureEngine,
    _round2,
    _round4,
)
from engines.intercompany_elimination_engine import IntercompanyEliminationEngine
from engines.acquisition_divestiture_engine import AcquisitionDivestitureEngine
from engines.consolidation_adjustment_engine import ConsolidationAdjustmentEngine
from engines.group_reporting_engine import GroupReportingEngine
from engines.consolidation_audit_engine import ConsolidationAuditEngine


@pytest.fixture
def entity_registry():
    """Populated entity registry with 6 entities."""
    engine = EntityRegistryEngine()
    entities = [
        {"entity_id": "PARENT", "entity_name": "Parent Corp HQ", "entity_type": "PARENT", "country": "CH"},
        {"entity_id": "SUB-DE", "entity_name": "Manufacturing DE", "entity_type": "SUBSIDIARY", "country": "DE"},
        {"entity_id": "SUB-GB", "entity_name": "Operations GB", "entity_type": "SUBSIDIARY", "country": "GB"},
        {"entity_id": "SUB-US", "entity_name": "US Division", "entity_type": "SUBSIDIARY", "country": "US"},
        {"entity_id": "JV-NL", "entity_name": "Joint Venture NL", "entity_type": "JOINT_VENTURE", "country": "NL"},
        {"entity_id": "ASSOC-NO", "entity_name": "Associate NO", "entity_type": "ASSOCIATE", "country": "NO"},
    ]
    for ent in entities:
        engine.register_entity(ent)
    return engine


@pytest.fixture
def ownership_engine():
    """Populated ownership engine."""
    engine = OwnershipStructureEngine()
    links = [
        {"owner_entity_id": "PARENT", "target_entity_id": "SUB-DE", "ownership_pct": Decimal("100"), "manages_operations": True},
        {"owner_entity_id": "PARENT", "target_entity_id": "SUB-GB", "ownership_pct": Decimal("80"), "manages_operations": True},
        {"owner_entity_id": "PARENT", "target_entity_id": "SUB-US", "ownership_pct": Decimal("60"), "manages_operations": True},
        {"owner_entity_id": "PARENT", "target_entity_id": "JV-NL", "ownership_pct": Decimal("50")},
        {"owner_entity_id": "PARENT", "target_entity_id": "ASSOC-NO", "ownership_pct": Decimal("30")},
    ]
    for link in links:
        engine.set_ownership(link)
    return engine


@pytest.fixture
def entity_emissions():
    """Emission data by entity (tCO2e)."""
    return {
        "PARENT": {"scope1": Decimal("500"), "scope2_loc": Decimal("300"), "scope2_mkt": Decimal("280"), "scope3": Decimal("200")},
        "SUB-DE": {"scope1": Decimal("15000"), "scope2_loc": Decimal("8000"), "scope2_mkt": Decimal("7500"), "scope3": Decimal("5000")},
        "SUB-GB": {"scope1": Decimal("3000"), "scope2_loc": Decimal("2000"), "scope2_mkt": Decimal("1800"), "scope3": Decimal("1500")},
        "SUB-US": {"scope1": Decimal("8000"), "scope2_loc": Decimal("4000"), "scope2_mkt": Decimal("3800"), "scope3": Decimal("3000")},
        "JV-NL": {"scope1": Decimal("6000"), "scope2_loc": Decimal("3000"), "scope2_mkt": Decimal("2800"), "scope3": Decimal("2000")},
        "ASSOC-NO": {"scope1": Decimal("1000"), "scope2_loc": Decimal("500"), "scope2_mkt": Decimal("450"), "scope3": Decimal("400")},
    }


class TestE2EEquityShareConsolidation:
    """End-to-end equity share consolidation."""

    def test_equity_share_scope1_total(self, ownership_engine, entity_emissions):
        total_s1 = Decimal("0")
        for eid, data in entity_emissions.items():
            if eid == "PARENT":
                pct = Decimal("100")
            else:
                chain = ownership_engine.resolve_equity_chain("PARENT", eid)
                pct = chain.effective_ownership_pct
            total_s1 += _round2(data["scope1"] * pct / Decimal("100"))

        # PARENT: 500*1 + SUB-DE: 15000*1 + SUB-GB: 3000*0.8 + SUB-US: 8000*0.6
        # + JV-NL: 6000*0.5 + ASSOC-NO: 1000*0.3
        expected = (
            Decimal("500") + Decimal("15000") + Decimal("2400")
            + Decimal("4800") + Decimal("3000") + Decimal("300")
        )
        assert total_s1 == expected

    def test_equity_share_all_scopes_consolidated(self, ownership_engine, entity_emissions):
        """Consolidate all scopes using equity share approach."""
        totals = {"scope1": Decimal("0"), "scope2_loc": Decimal("0"), "scope3": Decimal("0")}
        for eid, data in entity_emissions.items():
            pct = Decimal("100") if eid == "PARENT" else ownership_engine.resolve_equity_chain("PARENT", eid).effective_ownership_pct
            for scope in totals:
                totals[scope] += _round2(data[scope] * pct / Decimal("100"))

        assert totals["scope1"] > Decimal("0")
        assert totals["scope2_loc"] > Decimal("0")
        assert totals["scope3"] > Decimal("0")
        grand_total = sum(totals.values())
        assert grand_total > Decimal("40000")

    def test_equity_share_vs_operational_different_totals(self, ownership_engine, entity_emissions):
        """Equity share total != operational control total."""
        equity_total = Decimal("0")
        control_total = Decimal("0")
        for eid, data in entity_emissions.items():
            if eid == "PARENT":
                eq_pct = Decimal("100")
                ctrl_pct = Decimal("100")
            else:
                chain = ownership_engine.resolve_equity_chain("PARENT", eid)
                assessment = ownership_engine.assess_control("PARENT", eid)
                eq_pct = chain.effective_ownership_pct
                ctrl_pct = assessment.inclusion_pct_operational
            equity_total += _round2(data["scope1"] * eq_pct / Decimal("100"))
            control_total += _round2(data["scope1"] * ctrl_pct / Decimal("100"))

        assert equity_total != control_total


class TestE2EOperationalControlConsolidation:
    """End-to-end operational control consolidation."""

    def test_operational_control_100_0_inclusion(self, ownership_engine, entity_emissions):
        """Subs get 100%, JV and associate get 0%."""
        total = Decimal("0")
        for eid, data in entity_emissions.items():
            if eid == "PARENT":
                pct = Decimal("100")
            else:
                assessment = ownership_engine.assess_control("PARENT", eid)
                pct = assessment.inclusion_pct_operational
            total += _round2(data["scope1"] * pct / Decimal("100"))

        # PARENT: 500 + SUB-DE: 15000 + SUB-GB: 3000 + SUB-US: 8000
        # JV-NL: 0 + ASSOC-NO: 0
        expected = Decimal("500") + Decimal("15000") + Decimal("3000") + Decimal("8000")
        assert total == expected


class TestE2EWithEliminations:
    """End-to-end consolidation with intercompany eliminations."""

    def test_elimination_reduces_total(self, ownership_engine, entity_emissions):
        elim_engine = IntercompanyEliminationEngine()
        # SUB-DE sells electricity to SUB-GB (intra-group)
        elim_engine.register_transfer({
            "seller_entity_id": "SUB-DE",
            "buyer_entity_id": "SUB-GB",
            "transfer_type": "ELECTRICITY",
            "seller_emissions_tco2e": Decimal("500"),
            "buyer_emissions_tco2e": Decimal("500"),
            "intra_group_pct": Decimal("100"),
        })
        result = elim_engine.calculate_eliminations()
        total_eliminated = sum(e.elimination_amount_tco2e for e in result.eliminations)
        assert total_eliminated == Decimal("500.00")

        # Build consolidated total
        equity_total_s1 = Decimal("0")
        for eid, data in entity_emissions.items():
            pct = Decimal("100") if eid == "PARENT" else ownership_engine.resolve_equity_chain("PARENT", eid).effective_ownership_pct
            equity_total_s1 += _round2(data["scope1"] * pct / Decimal("100"))

        net_total = equity_total_s1 - total_eliminated
        assert net_total < equity_total_s1
        assert net_total > Decimal("0")


class TestE2EWithMnAEvents:
    """End-to-end consolidation with M&A events."""

    def test_acquisition_adds_prorated_emissions(self, ownership_engine, entity_emissions):
        mna_engine = AcquisitionDivestitureEngine()
        # Mid-year acquisition of new entity
        event = mna_engine.register_event({
            "event_type": "ACQUISITION",
            "entity_id": "NEW-ACQ",
            "effective_date": date(2025, 7, 1),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("12000"),
            "scope1_tco2e": Decimal("5000"),
            "scope2_location_tco2e": Decimal("4000"),
            "scope3_tco2e": Decimal("3000"),
        })
        prorate = mna_engine.calculate_prorate(event.event_id)
        assert prorate.prorated_emissions_tco2e < Decimal("12000")
        assert prorate.prorated_emissions_tco2e > Decimal("0")
        assert prorate.pro_rata_factor < Decimal("1")

    def test_divestiture_and_restatement(self):
        mna_engine = AcquisitionDivestitureEngine()
        event = mna_engine.register_event({
            "event_type": "DIVESTITURE",
            "entity_id": "OLD-DIV",
            "effective_date": date(2025, 4, 1),
            "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("8000"),
            "scope1_tco2e": Decimal("3000"),
        })
        prorate = mna_engine.calculate_prorate(event.event_id)
        restatement = mna_engine.trigger_base_year_restatement(
            event.event_id,
            base_year=2020,
            base_year_total_tco2e=Decimal("100000"),
            base_year_scope1=Decimal("40000"),
        )
        assert restatement.restated_total_tco2e < Decimal("100000")
        assert restatement.restated_scope1 < Decimal("40000")

        # Organic vs structural
        analysis = mna_engine.separate_organic_structural(
            reporting_year=2025,
            base_year=2020,
            base_year_original_tco2e=Decimal("100000"),
            base_year_adjusted_tco2e=restatement.restated_total_tco2e,
            current_year_tco2e=Decimal("95000"),
        )
        assert analysis.structural_change_tco2e < Decimal("0")


class TestE2EWithAdjustments:
    """End-to-end consolidation with manual adjustments."""

    def test_adjustment_through_full_workflow(self):
        adj_engine = ConsolidationAdjustmentEngine()
        adj = adj_engine.create_adjustment({
            "reporting_year": 2025,
            "entity_id": "SUB-DE",
            "category": "ERROR_CORRECTION",
            "scope_target": "SCOPE_1",
            "adjustment_amount_tco2e": Decimal("-500"),
            "before_value_tco2e": Decimal("15000"),
            "justification": "Gas meter calibration error",
        })
        adj_engine.submit_adjustment(adj.adjustment_id, "analyst@corp.com")
        adj_engine.review_adjustment(adj.adjustment_id, "reviewer@corp.com")
        adj_engine.approve_adjustment(adj.adjustment_id, "cfo@corp.com")

        impact = adj_engine.calculate_impact(2025, pre_adjustment_total=Decimal("50000"))
        assert impact.total_adjustment_tco2e == Decimal("-500.00")
        assert impact.post_adjustment_total == Decimal("49500.00")


class TestE2EGroupReportGeneration:
    """End-to-end group report generation."""

    def test_full_report_with_all_features(self, ownership_engine, entity_emissions):
        reporting_engine = GroupReportingEngine()

        # Build entity data with equity share adjustments
        entity_data = []
        for eid, data in entity_emissions.items():
            pct = Decimal("100") if eid == "PARENT" else ownership_engine.resolve_equity_chain("PARENT", eid).effective_ownership_pct
            entity_data.append({
                "entity_id": eid,
                "entity_name": eid,
                "scope1": str(_round2(data["scope1"] * pct / Decimal("100"))),
                "scope2_location": str(_round2(data["scope2_loc"] * pct / Decimal("100"))),
                "scope2_market": str(_round2(data["scope2_mkt"] * pct / Decimal("100"))),
                "scope3": str(_round2(data["scope3"] * pct / Decimal("100"))),
                "country": {"PARENT": "CH", "SUB-DE": "DE", "SUB-GB": "GB", "SUB-US": "US", "JV-NL": "NL", "ASSOC-NO": "NO"}.get(eid, "XX"),
                "region": "EUROPE" if eid not in ("SUB-US",) else "AMERICAS",
            })

        # Prior year for trends
        prior_data = [{
            "entity_id": "PRIOR",
            "scope1": "28000",
            "scope2_location": "14000",
            "scope3": "10000",
            "country": "CH",
        }]

        report = reporting_engine.generate_report(
            reporting_year=2025,
            entity_data=entity_data,
            organisation_name="Test Corp Group",
            consolidation_approach="EQUITY_SHARE",
            eliminations_tco2e=Decimal("500"),
            adjustments_tco2e=Decimal("-300"),
            prior_year_data=prior_data,
            intensity_denominators={"revenue_m": "2000", "employees": "25000"},
            sbti_targets={
                "base_year": 2020,
                "base_year_emissions": "60000",
                "target_year": 2030,
                "target_reduction_pct": "42",
            },
        )

        assert report.entity_count == 6
        assert report.scope_breakdown.total > Decimal("0")
        assert report.scope_breakdown.scope1 > Decimal("0")
        assert report.waterfall is not None
        assert report.geographic_breakdown is not None
        assert len(report.intensity_metrics) >= 2
        assert len(report.trends) == 2
        assert report.sbti_target_progress is not None
        assert report.variance_vs_prior is not None
        assert len(report.provenance_hash) == 64

    def test_framework_mappings_all_populated(self, ownership_engine, entity_emissions):
        reporting_engine = GroupReportingEngine()
        entity_data = []
        for eid, data in entity_emissions.items():
            entity_data.append({
                "entity_id": eid,
                "scope1": str(data["scope1"]),
                "scope2_location": str(data["scope2_loc"]),
                "scope3": str(data["scope3"]),
                "country": "US",
            })

        report = reporting_engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )

        for fw in ["CSRD_ESRS_E1", "CDP", "GRI_305", "TCFD", "SEC_CLIMATE", "SBTI", "IFRS_S2", "ISO_14064"]:
            mapping = reporting_engine.map_to_framework(report, fw)
            assert mapping.coverage_pct > Decimal("0"), f"Framework {fw} has 0% coverage"


class TestE2EAuditTrail:
    """End-to-end audit trail and assurance package."""

    def test_full_audit_trail_to_assurance(self, ownership_engine, entity_emissions):
        audit_engine = ConsolidationAuditEngine()

        # Record steps for each entity
        for eid in entity_emissions:
            audit_engine.record_step(
                2025, "DATA_RECEIPT",
                f"Received data from {eid}",
                entity_id=eid,
            )
            audit_engine.record_step(
                2025, "DATA_VALIDATION",
                f"Validated data for {eid}",
                entity_id=eid,
            )

        # Equity adjustments
        audit_engine.record_step(2025, "EQUITY_ADJUSTMENT", "Applied equity shares")

        # Elimination
        audit_engine.record_step(2025, "INTERCOMPANY_ELIMINATION", "Eliminated transfers", impact_tco2e="-500")

        # Reconcile
        recon = audit_engine.reconcile(2025, Decimal("50000"), Decimal("50200"))
        assert recon.status in ("RECONCILED", "PARTIALLY_RECONCILED")

        # Completeness
        entity_ids = list(entity_emissions.keys())
        check = audit_engine.check_completeness(2025, entity_ids, entity_ids)
        assert check.completeness_pct == Decimal("100.00")

        # Sign-offs
        for eid in entity_emissions:
            audit_engine.record_signoff(2025, "ENTITY", f"cfo_{eid}@corp.com", entity_id=eid)
        audit_engine.record_signoff(2025, "GROUP", "group_cfo@corp.com")

        # Generate assurance package
        package = audit_engine.generate_assurance_package(
            2025,
            organisation_name="Test Corp",
            consolidated_total_tco2e=Decimal("50000"),
        )
        assert package.is_assurance_ready is True
        assert package.total_audit_entries > 10
        assert package.entity_signoffs == 6
        assert package.group_signoffs == 1
        assert package.completeness_pct == Decimal("100.00")
        assert len(package.provenance_hash) == 64


class TestE2EFullPipeline:
    """End-to-end full consolidation pipeline."""

    def test_complete_pipeline_all_engines(self):
        """Run through all 10 engine types in sequence."""
        # 1. Entity Registry
        registry = EntityRegistryEngine()
        registry.register_entity({"entity_id": "HQ", "entity_name": "HQ", "entity_type": "PARENT", "country": "CH"})
        registry.register_entity({"entity_id": "SUB", "entity_name": "Sub", "entity_type": "SUBSIDIARY", "country": "DE"})
        assert len(registry.get_all_entities()) == 2

        # 2. Ownership
        ownership = OwnershipStructureEngine()
        ownership.set_ownership({"owner_entity_id": "HQ", "target_entity_id": "SUB", "ownership_pct": Decimal("80"), "manages_operations": True})

        # 3. Equity chain
        chain = ownership.resolve_equity_chain("HQ", "SUB")
        assert chain.effective_ownership_pct == Decimal("80")

        # 4. Control assessment
        control = ownership.assess_control("HQ", "SUB")
        assert control.inclusion_pct_operational == Decimal("100")
        assert control.inclusion_pct_equity == Decimal("80")

        # 5. Elimination
        elim = IntercompanyEliminationEngine()
        elim.register_transfer({
            "seller_entity_id": "HQ", "buyer_entity_id": "SUB",
            "transfer_type": "ELECTRICITY",
            "seller_emissions_tco2e": Decimal("100"),
            "buyer_emissions_tco2e": Decimal("100"),
            "intra_group_pct": Decimal("100"),
        })
        elim_result = elim.calculate_eliminations()
        assert len(elim_result.eliminations) == 1

        # 6. M&A
        mna = AcquisitionDivestitureEngine()
        event = mna.register_event({
            "event_type": "ACQUISITION", "entity_id": "NEW",
            "effective_date": date(2025, 10, 1), "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("5000"),
        })
        prorate = mna.calculate_prorate(event.event_id)
        assert prorate.pro_rata_factor < Decimal("1")

        # 7. Adjustments
        adj_eng = ConsolidationAdjustmentEngine()
        adj = adj_eng.create_adjustment({
            "reporting_year": 2025, "entity_id": "SUB",
            "category": "ERROR_CORRECTION",
            "adjustment_amount_tco2e": Decimal("-50"),
            "justification": "Meter fix",
        })
        adj_eng.submit_adjustment(adj.adjustment_id, "analyst")
        adj_eng.review_adjustment(adj.adjustment_id, "reviewer")
        adj_eng.approve_adjustment(adj.adjustment_id, "cfo")

        # 8. Group report
        reporting = GroupReportingEngine()
        report = reporting.generate_report(
            reporting_year=2025,
            entity_data=[
                {"entity_id": "HQ", "scope1": "500", "scope2_location": "300", "scope3": "200", "country": "CH"},
                {"entity_id": "SUB", "scope1": "12000", "scope2_location": "6000", "scope3": "4000", "country": "DE"},
            ],
        )
        assert report.scope_breakdown.total > Decimal("0")

        # 9. Audit trail
        audit = ConsolidationAuditEngine()
        audit.record_step(2025, "DATA_RECEIPT", "HQ data", entity_id="HQ")
        audit.record_step(2025, "DATA_RECEIPT", "SUB data", entity_id="SUB")
        audit.reconcile(2025, report.scope_breakdown.total, report.scope_breakdown.total)
        audit.check_completeness(2025, ["HQ", "SUB"], ["HQ", "SUB"])
        audit.record_signoff(2025, "ENTITY", "cfo_hq", entity_id="HQ")
        audit.record_signoff(2025, "ENTITY", "cfo_sub", entity_id="SUB")
        audit.record_signoff(2025, "GROUP", "group_cfo")

        # 10. Assurance package
        package = audit.generate_assurance_package(2025, consolidated_total_tco2e=report.scope_breakdown.total)
        assert package.is_assurance_ready is True
        assert len(package.provenance_hash) == 64

    def test_provenance_chain_integrity(self):
        """Verify SHA-256 hashes appear on every result object."""
        ownership = OwnershipStructureEngine()
        ownership.set_ownership({"owner_entity_id": "A", "target_entity_id": "B", "ownership_pct": Decimal("80"), "manages_operations": True})

        chain = ownership.resolve_equity_chain("A", "B")
        assert len(chain.provenance_hash) == 64

        assessment = ownership.assess_control("A", "B")
        assert len(assessment.provenance_hash) == 64

        mna = AcquisitionDivestitureEngine()
        event = mna.register_event({
            "event_type": "ACQUISITION", "entity_id": "C",
            "effective_date": date(2025, 7, 1), "reporting_year": 2025,
            "annual_emissions_tco2e": Decimal("10000"),
        })
        assert len(event.provenance_hash) == 64

        prorate = mna.calculate_prorate(event.event_id)
        assert len(prorate.provenance_hash) == 64

        restatement = mna.trigger_base_year_restatement(
            event.event_id, base_year=2020,
            base_year_total_tco2e=Decimal("100000"),
        )
        assert len(restatement.provenance_hash) == 64

        reporting = GroupReportingEngine()
        report = reporting.generate_report(
            reporting_year=2025,
            entity_data=[{"entity_id": "A", "scope1": "1000", "scope2_location": "500", "scope3": "300", "country": "US"}],
        )
        assert len(report.provenance_hash) == 64
        assert len(report.scope_breakdown.provenance_hash) == 64

    def test_decimal_precision_end_to_end(self):
        """Verify no floating-point errors throughout the pipeline."""
        ownership = OwnershipStructureEngine()
        ownership.set_ownership({"owner_entity_id": "A", "target_entity_id": "B", "ownership_pct": Decimal("33.33")})
        ownership.set_ownership({"owner_entity_id": "B", "target_entity_id": "C", "ownership_pct": Decimal("66.67")})

        chain = ownership.resolve_equity_chain("A", "C")
        assert isinstance(chain.effective_ownership_pct, Decimal)
        assert "E" not in str(chain.effective_ownership_pct)

        emissions = Decimal("10000.00")
        allocated = _round2(emissions * chain.effective_ownership_pct / Decimal("100"))
        assert isinstance(allocated, Decimal)
        assert "E" not in str(allocated)

        # Verify the calculation: 33.33% * 66.67% = 22.2211%
        expected_pct = _round4(Decimal("33.33") * Decimal("66.67") / Decimal("100"))
        assert chain.effective_ownership_pct == expected_pct
        expected_allocated = _round2(emissions * expected_pct / Decimal("100"))
        assert allocated == expected_allocated
