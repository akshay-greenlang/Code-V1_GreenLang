# -*- coding: utf-8 -*-
"""
End-to-End Integration Tests for PACK-045.

Tests full pipeline flows from data input through to reporting.
Target: ~15 tests.
"""

import pytest
from decimal import Decimal
from datetime import date
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from engines.base_year_selection_engine import (
    BaseYearSelectionEngine, CandidateYear, SelectionConfig, SectorType,
)
from engines.base_year_inventory_engine import (
    BaseYearInventoryEngine, SourceEmission, InventoryConfig,
    SourceCategory, GasType, InventoryStatus,
)
from engines.recalculation_policy_engine import (
    RecalculationPolicyEngine, PolicyType, ComplianceFramework,
)
from engines.recalculation_trigger_engine import (
    RecalculationTriggerEngine, EntityRegistryEntry, TriggerDetectionConfig,
    InventorySnapshot, ExternalEvent,
    TriggerType as TrigTriggerType,
)
from engines.significance_assessment_engine import (
    SignificanceAssessmentEngine, TriggerInput as SigTriggerInput,
    AssessmentPolicy, SignificanceMethod,
    TriggerType as SigTriggerType,
)
from engines.base_year_adjustment_engine import (
    BaseYearAdjustmentEngine, AdjustmentConfig, BaseYearInventory,
    TriggerInput as AdjTriggerInput,
    TriggerType as AdjTriggerType,
    Scope as AdjScope,
)
from engines.time_series_consistency_engine import (
    TimeSeriesConsistencyEngine, YearData, ConsistencyConfig, GWPVersion,
    ConsolidationApproach, ReportingFramework,
)
from engines.target_tracking_engine import (
    TargetTrackingEngine, EmissionsTarget, YearlyActual,
    TargetType, ScopeType as TTScopeType, SBTiAmbition,
)
from engines.base_year_audit_engine import (
    BaseYearAuditEngine, AuditEventType, AuditSeverity,
)
from engines.base_year_reporting_engine import (
    BaseYearReportingEngine, InventoryData, ReportConfig,
    ReportingFramework as RptFramework, OutputFormat,
)
from config.pack_config import load_preset, validate_config


class TestE2EBaseYearEstablishment:
    """End-to-end test: Full base year establishment pipeline."""

    def test_full_establishment_pipeline(self):
        """Test selecting a base year, establishing inventory, and validating."""
        # Step 1: Select base year
        sel_engine = BaseYearSelectionEngine()
        candidates = [
            CandidateYear(
                year=2020,
                scope1_tco2e=Decimal("5000"),
                scope2_tco2e=Decimal("3000"),
                total_tco2e=Decimal("8000"),
                data_quality_score=Decimal("80"),
                completeness_pct=Decimal("90"),
                methodology_tier=2,
                is_verified=False,
            ),
            CandidateYear(
                year=2021,
                scope1_tco2e=Decimal("4800"),
                scope2_tco2e=Decimal("2800"),
                total_tco2e=Decimal("7600"),
                data_quality_score=Decimal("88"),
                completeness_pct=Decimal("95"),
                methodology_tier=3,
                is_verified=True,
            ),
            CandidateYear(
                year=2022,
                scope1_tco2e=Decimal("4600"),
                scope2_tco2e=Decimal("2700"),
                total_tco2e=Decimal("7300"),
                data_quality_score=Decimal("92"),
                completeness_pct=Decimal("97"),
                methodology_tier=3,
                is_verified=True,
            ),
        ]
        selection = sel_engine.evaluate_candidates(candidates)
        assert selection.recommended_year in (2020, 2021, 2022)

        # Step 2: Establish inventory for selected year
        inv_engine = BaseYearInventoryEngine()
        sources = [
            SourceEmission(
                category=SourceCategory.STATIONARY_COMBUSTION,
                tco2e=Decimal("3000"),
                facility_id="FAC-001",
            ),
            SourceEmission(
                category=SourceCategory.MOBILE_COMBUSTION,
                tco2e=Decimal("1600"),
                facility_id="FAC-001",
            ),
            SourceEmission(
                category=SourceCategory.ELECTRICITY_LOCATION,
                tco2e=Decimal("2700"),
                facility_id="FAC-001",
            ),
        ]
        config = InventoryConfig(
            organization_id="ORG-E2E",
            base_year=selection.recommended_year,
        )
        inventory = inv_engine.establish_inventory(sources, config)
        assert inventory.grand_total_tco2e > Decimal("0")
        assert inventory.provenance_hash != ""

        # Step 3: Snapshot the inventory
        snapshot = inv_engine.snapshot_inventory(inventory)
        assert snapshot.status == InventoryStatus.FROZEN


class TestE2ERecalculationPipeline:
    """End-to-end test: Trigger detection through significance assessment."""

    def test_trigger_to_significance_pipeline(self):
        """Test detecting a trigger and assessing significance."""
        # Step 1: Detect triggers using inventory snapshots
        trig_engine = RecalculationTriggerEngine()
        current = InventorySnapshot(
            year=2024,
            scope1_total_tco2e=Decimal("55000"),
            entity_ids=["FAC-001", "FAC-002", "FAC-NEW"],
            by_facility={"FAC-001": Decimal("25000"), "FAC-002": Decimal("20000"),
                         "FAC-NEW": Decimal("10000")},
        )
        previous = InventorySnapshot(
            year=2023,
            scope1_total_tco2e=Decimal("45000"),
            entity_ids=["FAC-001", "FAC-002"],
            by_facility={"FAC-001": Decimal("25000"), "FAC-002": Decimal("20000")},
        )
        config = TriggerDetectionConfig(base_year_total_tco2e=Decimal("100000"))
        detected = trig_engine.detect_triggers(current, previous, [], config)
        assert detected.total_triggers_detected >= 1

        # Step 2: Assess significance
        sig_engine = SignificanceAssessmentEngine()
        sig_triggers = [
            SigTriggerInput(
                trigger_id=f"SIG-{i}",
                trigger_type=SigTriggerType.ACQUISITION,
                emission_impact_tco2e=Decimal("10000"),
                description="Acquired new facility",
            )
            for i in range(1)
        ]
        sig_result = sig_engine.assess_significance(
            sig_triggers, base_year_total_tco2e=Decimal("100000")
        )
        assert sig_result.provenance_hash != ""


class TestE2EAdjustmentPipeline:
    """End-to-end test: Creating and applying adjustments."""

    def test_adjustment_creation(self):
        """Test creating an adjustment package."""
        adj_engine = BaseYearAdjustmentEngine()
        inventory = BaseYearInventory(
            base_year=2022,
            scope1_tco2e=Decimal("50000"),
            scope2_location_tco2e=Decimal("30000"),
        )
        trigger = AdjTriggerInput(
            trigger_id="ADJ-E2E-1",
            trigger_type=AdjTriggerType.ACQUISITION,
            scope=AdjScope.SCOPE_1,
            entity_id="ENT-ACQ",
            entity_emissions_tco2e=Decimal("8000"),
            ownership_pct=Decimal("100"),
            effective_date=date(2024, 7, 1),
        )
        pkg = adj_engine.create_adjustment_package(inventory, [trigger])
        assert len(pkg.adjustment_lines) >= 1
        assert pkg.provenance_hash != ""


class TestE2ETargetTracking:
    """End-to-end test: Base year to target tracking."""

    def test_base_year_to_target_tracking(self):
        """Test establishing base year and tracking against targets."""
        target_engine = TargetTrackingEngine()
        target = EmissionsTarget(
            target_id="TGT-E2E",
            name="E2E Scope 1 Target",
            target_type=TargetType.ABSOLUTE,
            scopes=[TTScopeType.SCOPE_1],
            base_year=2019,
            base_year_tco2e=Decimal("100000"),
            target_year=2030,
            target_reduction_pct=Decimal("42.0"),
        )

        actuals = [
            YearlyActual(year=2019, actual_tco2e=Decimal("100000")),
            YearlyActual(year=2020, actual_tco2e=Decimal("95000")),
            YearlyActual(year=2021, actual_tco2e=Decimal("90000")),
            YearlyActual(year=2022, actual_tco2e=Decimal("85000")),
            YearlyActual(year=2023, actual_tco2e=Decimal("80000")),
        ]

        result = target_engine.track_progress(target, actuals)
        assert result.provenance_hash != ""
        assert len(result.progress_points) >= 1

        pathway = target_engine.calculate_linear_pathway(target)
        assert len(pathway) >= 2

        sbti_pathway = target_engine.calculate_sbti_pathway(target)
        assert len(sbti_pathway) >= 2


class TestE2ETimeSeriesConsistency:
    """End-to-end test: Time series consistency checking."""

    def test_time_series_consistency_check(self):
        engine = TimeSeriesConsistencyEngine()
        series = [
            YearData(
                year=2019 + i,
                total_tco2e=Decimal(str(10000 - 500 * i)),
                scope1_tco2e=Decimal(str(5000 - 250 * i)),
                scope2_location_tco2e=Decimal(str(3000 - 150 * i)),
                gwp_version=GWPVersion.AR5,
                consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
            )
            for i in range(5)
        ]
        result = engine.assess_consistency(series)
        assert result is not None

        trend = engine.calculate_trend(series)
        assert len(trend) >= 1


class TestE2EAuditTrail:
    """End-to-end test: Audit trail from start to verification."""

    def test_audit_trail_lifecycle(self):
        engine = BaseYearAuditEngine()

        engine.create_audit_entry(
            event_type=AuditEventType.BASE_YEAR_ESTABLISHED,
            actor="system@greenlang",
            organization_id="ORG-AUDIT-E2E",
            base_year=2022,
            description="Base year 2022 established",
            severity=AuditSeverity.INFO,
        )
        engine.create_audit_entry(
            event_type=AuditEventType.TRIGGER_DETECTED,
            actor="system@greenlang",
            organization_id="ORG-AUDIT-E2E",
            base_year=2022,
            description="Acquisition trigger detected",
            severity=AuditSeverity.MEDIUM,
        )

        trail = engine.get_audit_trail(organization_id="ORG-AUDIT-E2E", base_year=2022)
        assert len(trail.entries) >= 2

        count = engine.get_entry_count(organization_id="ORG-AUDIT-E2E", base_year=2022)
        assert count >= 2


class TestE2EReporting:
    """End-to-end test: Multi-framework reporting."""

    def test_multi_framework_report(self):
        engine = BaseYearReportingEngine()
        data = InventoryData(
            organization_name="E2E Test Corp",
            organization_id="ORG-RPT-E2E",
            base_year=2022,
            reporting_year=2026,
            base_year_scope1_tco2e=Decimal("5000"),
            base_year_scope2_location_tco2e=Decimal("3000"),
            base_year_scope2_market_tco2e=Decimal("2800"),
            base_year_scope3_tco2e=Decimal("15000"),
            base_year_total_tco2e=Decimal("25800"),
        )
        config = ReportConfig(
            output_format=OutputFormat.JSON,
        )
        report = engine.generate_multi_framework_report(
            data,
            [RptFramework.GHG_PROTOCOL, RptFramework.CDP, RptFramework.SBTI],
            config,
        )
        assert report is not None
        assert len(report.reports) >= 1


class TestE2EConfigPreset:
    """End-to-end test: Loading preset and validating."""

    def test_load_and_validate_preset(self):
        pc = load_preset("manufacturing", overrides={
            "company_name": "E2E Manufacturing Co",
            "reporting_year": 2026,
        })
        assert pc.pack.company_name == "E2E Manufacturing Co"
        warnings = validate_config(pc.pack)
        assert isinstance(warnings, list)

    def test_load_all_presets_valid(self):
        preset_names = [
            "corporate_office", "manufacturing", "energy_utility",
            "transport_logistics", "food_agriculture", "real_estate",
            "healthcare", "sme_simplified",
        ]
        for name in preset_names:
            pc = load_preset(name)
            assert pc.preset_name == name, f"Preset {name} loaded with wrong name"
            assert pc.pack is not None, f"Preset {name} has no pack config"


class TestE2EProvenanceChain:
    """End-to-end test: Provenance hash chain across engines."""

    def test_provenance_chain(self):
        # Selection
        sel_engine = BaseYearSelectionEngine()
        candidates = [
            CandidateYear(year=2021, total_tco2e=Decimal("8000"),
                          data_quality_score=Decimal("85"), completeness_pct=Decimal("90")),
            CandidateYear(year=2022, total_tco2e=Decimal("7500"),
                          data_quality_score=Decimal("90"), completeness_pct=Decimal("95")),
        ]
        sel_result = sel_engine.evaluate_candidates(candidates)
        assert len(sel_result.provenance_hash) == 64

        # Inventory
        inv_engine = BaseYearInventoryEngine()
        sources = [
            SourceEmission(category=SourceCategory.STATIONARY_COMBUSTION, tco2e=Decimal("5000")),
            SourceEmission(category=SourceCategory.ELECTRICITY_LOCATION, tco2e=Decimal("2500")),
        ]
        inv_config = InventoryConfig(
            organization_id="ORG-PROV", base_year=sel_result.recommended_year,
        )
        inv = inv_engine.establish_inventory(sources, inv_config)
        assert len(inv.provenance_hash) == 64

        # All hashes are unique
        assert sel_result.provenance_hash != inv.provenance_hash
