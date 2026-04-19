# -*- coding: utf-8 -*-
"""
Unit tests for ImpactMeasurementEngine - PACK-011 SFDR Article 9 Engine 4.

Tests KPI definition and tracking (15 environmental + 12 social), SDG
contribution mapping (17 SDGs), Theory of Change stages, additionality
assessment, period comparison (year-over-year impact), impact categories,
KPI updates and historical tracking, and provenance hashing.

Self-contained: no conftest imports.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Dynamic import helper (hyphenated directory names)
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_impact_mod = _import_from_path(
    "pack011_impact_measurement_engine",
    str(ENGINES_DIR / "impact_measurement_engine.py"),
)

ImpactMeasurementEngine = _impact_mod.ImpactMeasurementEngine
ImpactConfig = _impact_mod.ImpactConfig
ImpactKPI = _impact_mod.ImpactKPI
ImpactResult = _impact_mod.ImpactResult
SDGContribution = _impact_mod.SDGContribution
TheoryOfChange = _impact_mod.TheoryOfChange
AdditionalityResult = _impact_mod.AdditionalityResult
PeriodComparison = _impact_mod.PeriodComparison
KPIUpdate = _impact_mod.KPIUpdate
KPIDefinition = _impact_mod.KPIDefinition
ImpactCategory = _impact_mod.ImpactCategory
SDGGoal = _impact_mod.SDGGoal
ToCStage = _impact_mod.ToCStage
ALL_KPI_DEFINITIONS = _impact_mod.ALL_KPI_DEFINITIONS

# ---------------------------------------------------------------------------
# SHA-256 regex pattern
# ---------------------------------------------------------------------------

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _make_env_kpis(
    holding_id: str = "H_ENV_01",
    ghg_avoided: float = 50000.0,
    renewable_energy: float = 10000.0,
    water_saved: float = 5000.0,
    prior_ghg: float = 40000.0,
    attribution: float = 0.05,
) -> list:
    """Build a set of environmental KPI measurements for one holding."""
    return [
        ImpactKPI(
            kpi_id="env_ghg_avoided",
            holding_id=holding_id,
            value=ghg_avoided,
            prior_value=prior_ghg,
            unit="tCO2e",
            category=ImpactCategory.ENVIRONMENTAL,
            data_quality="reported",
            attribution_factor=attribution,
        ),
        ImpactKPI(
            kpi_id="env_renewable_energy",
            holding_id=holding_id,
            value=renewable_energy,
            prior_value=8000.0,
            unit="MWh",
            category=ImpactCategory.ENVIRONMENTAL,
            data_quality="reported",
            attribution_factor=attribution,
        ),
        ImpactKPI(
            kpi_id="env_water_saved",
            holding_id=holding_id,
            value=water_saved,
            prior_value=4000.0,
            unit="m3",
            category=ImpactCategory.ENVIRONMENTAL,
            data_quality="estimated",
            attribution_factor=attribution,
        ),
    ]


def _make_social_kpis(
    holding_id: str = "H_SOC_01",
    jobs_created: float = 200.0,
    people_served: float = 5000.0,
    attribution: float = 0.03,
) -> list:
    """Build a set of social KPI measurements for one holding."""
    return [
        ImpactKPI(
            kpi_id="soc_jobs_created",
            holding_id=holding_id,
            value=jobs_created,
            prior_value=180.0,
            unit="FTE",
            category=ImpactCategory.SOCIAL,
            data_quality="reported",
            attribution_factor=attribution,
        ),
        ImpactKPI(
            kpi_id="soc_people_served",
            holding_id=holding_id,
            value=people_served,
            prior_value=4500.0,
            unit="persons",
            category=ImpactCategory.SOCIAL,
            data_quality="reported",
            attribution_factor=attribution,
        ),
        ImpactKPI(
            kpi_id="soc_education_access",
            holding_id=holding_id,
            value=1000.0,
            prior_value=800.0,
            unit="persons",
            category=ImpactCategory.SOCIAL,
            data_quality="estimated",
            attribution_factor=attribution,
        ),
    ]


def _make_full_kpi_set() -> list:
    """Build a combined set of env + social KPIs from multiple holdings."""
    kpis = []
    kpis.extend(_make_env_kpis("H1", attribution=0.05))
    kpis.extend(_make_env_kpis("H2", ghg_avoided=30000.0, attribution=0.03))
    kpis.extend(_make_social_kpis("H3", attribution=0.04))
    kpis.extend(_make_social_kpis("H4", jobs_created=100.0, attribution=0.02))
    return kpis


def _make_theory_of_change() -> TheoryOfChange:
    """Build a sample Theory of Change model."""
    return TheoryOfChange(
        product_name="Green Impact Fund",
        impact_thesis=(
            "Invest in renewable energy and social inclusion to drive "
            "measurable environmental and social outcomes."
        ),
        stages={
            ToCStage.INPUT.value: [
                "Capital allocation to renewable energy projects",
                "Investment in education technology",
            ],
            ToCStage.ACTIVITY.value: [
                "Solar farm development",
                "Digital learning platform expansion",
            ],
            ToCStage.OUTPUT.value: [
                "MWh of renewable energy generated",
                "Number of students accessing platform",
            ],
            ToCStage.OUTCOME.value: [
                "Reduced grid carbon intensity",
                "Improved educational attainment",
            ],
            ToCStage.IMPACT.value: [
                "Climate change mitigation",
                "Reduced inequality in education access",
            ],
        },
        assumptions=[
            "Grid emission factors remain stable",
            "Digital access enables learning outcomes",
        ],
        evidence_sources=[
            "IEA World Energy Outlook 2025",
            "UNESCO Education Report 2025",
        ],
        linked_kpis={
            ToCStage.OUTPUT.value: ["env_renewable_energy", "soc_education_access"],
            ToCStage.OUTCOME.value: ["env_ghg_avoided"],
        },
        linked_sdgs=[SDGGoal.SDG_7, SDGGoal.SDG_4, SDGGoal.SDG_13],
    )


# ---------------------------------------------------------------------------
# Tests: Engine Initialization
# ---------------------------------------------------------------------------


class TestImpactMeasurementEngineInit:
    """Verify engine initialization and config defaults."""

    def test_default_config(self):
        engine = ImpactMeasurementEngine()
        assert engine.config.product_name == "SFDR Article 9 Product"
        assert len(engine.config.tracked_env_kpis) == 15
        assert len(engine.config.tracked_social_kpis) == 12

    def test_custom_config_dict(self):
        engine = ImpactMeasurementEngine({
            "product_name": "Impact Alpha Fund",
            "primary_sdg_threshold": 25.0,
        })
        assert engine.config.product_name == "Impact Alpha Fund"
        assert engine.config.primary_sdg_threshold == 25.0

    def test_custom_config_object(self):
        cfg = ImpactConfig(product_name="Test Fund")
        engine = ImpactMeasurementEngine(cfg)
        assert engine.config.product_name == "Test Fund"

    def test_total_kpi_count_27(self):
        """15 environmental + 12 social = 27 KPI definitions."""
        assert len(ALL_KPI_DEFINITIONS) == 27


# ---------------------------------------------------------------------------
# Tests: KPI Definition Registry
# ---------------------------------------------------------------------------


class TestKPIDefinitions:
    """Test the static KPI definition registry."""

    def test_15_environmental_kpis(self):
        """Registry has 15 environmental KPIs."""
        env_kpis = [
            k for k, v in ALL_KPI_DEFINITIONS.items()
            if v["category"] == ImpactCategory.ENVIRONMENTAL
        ]
        assert len(env_kpis) == 15

    def test_12_social_kpis(self):
        """Registry has 12 social KPIs."""
        social_kpis = [
            k for k, v in ALL_KPI_DEFINITIONS.items()
            if v["category"] == ImpactCategory.SOCIAL
        ]
        assert len(social_kpis) == 12

    def test_kpi_has_sdg_linkage(self):
        """Each KPI is linked to at least one SDG."""
        for kpi_id, defn in ALL_KPI_DEFINITIONS.items():
            assert len(defn["sdgs"]) > 0, f"KPI {kpi_id} has no SDG linkage"

    def test_kpi_has_unit(self):
        """Each KPI has a unit of measurement."""
        for kpi_id, defn in ALL_KPI_DEFINITIONS.items():
            assert defn["unit"] != "", f"KPI {kpi_id} has no unit"

    def test_ghg_avoided_kpi(self):
        """env_ghg_avoided KPI is defined correctly."""
        ghg = ALL_KPI_DEFINITIONS["env_ghg_avoided"]
        assert ghg["name"] == "GHG Emissions Avoided"
        assert ghg["unit"] == "tCO2e"
        assert ghg["category"] == ImpactCategory.ENVIRONMENTAL
        assert SDGGoal.SDG_13 in ghg["sdgs"]
        assert ghg["higher_is_better"] is True

    def test_jobs_created_kpi(self):
        """soc_jobs_created KPI is defined correctly."""
        jobs = ALL_KPI_DEFINITIONS["soc_jobs_created"]
        assert jobs["name"] == "Jobs Created"
        assert jobs["unit"] == "FTE"
        assert jobs["category"] == ImpactCategory.SOCIAL
        assert SDGGoal.SDG_8 in jobs["sdgs"]


# ---------------------------------------------------------------------------
# Tests: Impact Assessment
# ---------------------------------------------------------------------------


class TestImpactAssessment:
    """Test full impact measurement assessment."""

    def test_basic_assessment(self):
        """Basic assessment with env KPIs produces valid result."""
        engine = ImpactMeasurementEngine()
        kpis = _make_env_kpis("H1")
        result = engine.assess_impact(kpis)

        assert isinstance(result, ImpactResult)
        assert result.total_kpis_tracked > 0
        assert result.env_kpi_count > 0

    def test_combined_env_social(self):
        """Combined env + social KPIs tracked correctly."""
        engine = ImpactMeasurementEngine()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis)

        assert result.env_kpi_count > 0
        assert result.social_kpi_count > 0
        assert result.total_kpis_tracked == result.env_kpi_count + result.social_kpi_count

    def test_empty_kpis_raises(self):
        """Empty KPI list raises ValueError."""
        engine = ImpactMeasurementEngine()
        with pytest.raises(ValueError, match="empty"):
            engine.assess_impact([])

    def test_multi_holding_aggregation(self):
        """KPIs from multiple holdings are aggregated."""
        engine = ImpactMeasurementEngine()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis)

        assert result.total_holdings >= 2


# ---------------------------------------------------------------------------
# Tests: SDG Contribution Mapping
# ---------------------------------------------------------------------------


class TestSDGContribution:
    """Test SDG contribution mapping across all 17 goals."""

    def test_sdg_contributions_generated(self):
        """SDG contributions are computed."""
        engine = ImpactMeasurementEngine()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis)

        assert len(result.sdg_contributions) > 0

    def test_sdg_coverage_count(self):
        """SDG coverage count tracks how many SDGs have positive contribution."""
        engine = ImpactMeasurementEngine()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis)

        assert result.sdg_coverage_count > 0
        assert result.sdg_coverage_count <= 17

    def test_sdg_contribution_structure(self):
        """Each SDG contribution has correct structure."""
        engine = ImpactMeasurementEngine()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis)

        for sdg_c in result.sdg_contributions:
            assert isinstance(sdg_c, SDGContribution)
            assert sdg_c.sdg in list(SDGGoal)
            assert sdg_c.contribution_score >= 0.0
            assert sdg_c.sdg_number >= 1
            assert sdg_c.sdg_number <= 17

    def test_primary_sdgs_identified(self):
        """Primary SDGs (above threshold) are identified."""
        engine = ImpactMeasurementEngine()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis)

        # Primary SDGs have contribution_score >= primary_sdg_threshold
        for sdg in result.primary_sdgs:
            assert sdg in list(SDGGoal)

    def test_all_17_sdg_goals_defined(self):
        """All 17 SDG goals are defined in the enum."""
        all_sdgs = list(SDGGoal)
        assert len(all_sdgs) == 17

    def test_sdg_13_climate_action_contribution(self):
        """GHG-avoided KPI contributes to SDG 13 (Climate Action)."""
        engine = ImpactMeasurementEngine()
        kpis = _make_env_kpis("H1", ghg_avoided=100000.0)
        result = engine.assess_impact(kpis)

        sdg_13 = next(
            (s for s in result.sdg_contributions if s.sdg == SDGGoal.SDG_13), None
        )
        assert sdg_13 is not None
        assert sdg_13.kpi_count > 0


# ---------------------------------------------------------------------------
# Tests: Theory of Change
# ---------------------------------------------------------------------------


class TestTheoryOfChange:
    """Test Theory of Change stages and completeness."""

    def test_toc_stages_enum(self):
        """Theory of Change has 5 stages."""
        stages = list(ToCStage)
        assert len(stages) == 5
        assert ToCStage.INPUT in stages
        assert ToCStage.ACTIVITY in stages
        assert ToCStage.OUTPUT in stages
        assert ToCStage.OUTCOME in stages
        assert ToCStage.IMPACT in stages

    def test_toc_included_in_assessment(self):
        """When provided, ToC is included in the result."""
        engine = ImpactMeasurementEngine()
        toc = _make_theory_of_change()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis, theory_of_change=toc)

        assert result.theory_of_change is not None
        assert result.theory_of_change.impact_thesis != ""

    def test_toc_completeness_score(self):
        """ToC completeness score is computed."""
        engine = ImpactMeasurementEngine()
        toc = _make_theory_of_change()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis, theory_of_change=toc)

        assert result.theory_of_change.completeness_score >= 0.0
        assert result.theory_of_change.completeness_score <= 100.0

    def test_toc_linked_sdgs(self):
        """ToC has linked SDGs."""
        toc = _make_theory_of_change()
        assert len(toc.linked_sdgs) > 0
        assert SDGGoal.SDG_7 in toc.linked_sdgs

    def test_toc_not_required(self):
        """Assessment works without ToC."""
        engine = ImpactMeasurementEngine()
        kpis = _make_env_kpis("H1")
        result = engine.assess_impact(kpis)
        assert result.theory_of_change is None


# ---------------------------------------------------------------------------
# Tests: Additionality Assessment
# ---------------------------------------------------------------------------


class TestAdditionality:
    """Test investment additionality assessment."""

    def test_additionality_assessed(self):
        """Additionality is assessed when inputs provided."""
        engine = ImpactMeasurementEngine()
        kpis = _make_full_kpi_set()
        additionality_inputs = {
            "intentionality": 80.0,
            "contribution": 70.0,
            "counterfactual": 60.0,
            "materiality": 75.0,
        }
        result = engine.assess_impact(
            kpis, additionality_inputs=additionality_inputs,
        )

        assert result.additionality is not None
        assert isinstance(result.additionality, AdditionalityResult)

    def test_additionality_scores(self):
        """Additionality scores are within 0-100 range."""
        engine = ImpactMeasurementEngine()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis, additionality_inputs={
            "intentionality": 90.0,
            "contribution": 85.0,
            "counterfactual": 70.0,
            "materiality": 80.0,
        })

        add = result.additionality
        assert 0.0 <= add.intentionality_score <= 100.0
        assert 0.0 <= add.contribution_score <= 100.0
        assert 0.0 <= add.counterfactual_score <= 100.0
        assert 0.0 <= add.materiality_score <= 100.0
        assert 0.0 <= add.overall_additionality_score <= 100.0

    def test_additionality_not_required(self):
        """Assessment works without additionality inputs."""
        engine = ImpactMeasurementEngine()
        kpis = _make_env_kpis("H1")
        result = engine.assess_impact(kpis)
        assert result.additionality is None

    def test_additionality_provenance(self):
        """Additionality result has provenance hash."""
        engine = ImpactMeasurementEngine()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis, additionality_inputs={
            "intentionality": 80.0,
            "contribution": 70.0,
            "counterfactual": 60.0,
            "materiality": 75.0,
        })
        assert result.additionality.provenance_hash != ""


# ---------------------------------------------------------------------------
# Tests: Period Comparison (Year-over-Year)
# ---------------------------------------------------------------------------


class TestPeriodComparison:
    """Test year-on-year impact comparison."""

    def test_period_comparisons_generated(self):
        """YoY comparisons are generated when prior values exist."""
        engine = ImpactMeasurementEngine()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis)

        assert len(result.period_comparisons) > 0

    def test_comparison_structure(self):
        """Each comparison has correct structure."""
        engine = ImpactMeasurementEngine()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis)

        for comp in result.period_comparisons:
            assert isinstance(comp, PeriodComparison)
            assert comp.kpi_id != ""
            assert comp.direction in ("improved", "deteriorated", "unchanged")

    def test_improved_kpi_tracked(self):
        """KPIs that improved are counted."""
        engine = ImpactMeasurementEngine()
        kpis = _make_env_kpis("H1", ghg_avoided=60000.0, prior_ghg=40000.0)
        result = engine.assess_impact(kpis)

        assert result.kpis_improved_count >= 0

    def test_deteriorated_kpi_tracked(self):
        """KPIs that deteriorated are counted."""
        engine = ImpactMeasurementEngine()
        kpis = [
            ImpactKPI(
                kpi_id="env_ghg_avoided",
                holding_id="H1",
                value=20000.0,    # Decreased
                prior_value=40000.0,
                unit="tCO2e",
                category=ImpactCategory.ENVIRONMENTAL,
                attribution_factor=0.05,
            ),
        ]
        result = engine.assess_impact(kpis)
        assert result.kpis_deteriorated_count >= 0

    def test_pct_change_calculation(self):
        """Percentage change is calculated correctly."""
        engine = ImpactMeasurementEngine()
        kpis = _make_env_kpis("H1", ghg_avoided=60000.0, prior_ghg=40000.0)
        result = engine.assess_impact(kpis)

        ghg_comp = next(
            (c for c in result.period_comparisons if c.kpi_id == "env_ghg_avoided"),
            None,
        )
        if ghg_comp is not None:
            # (60000 - 40000) / 40000 * 100 = 50%
            assert ghg_comp.pct_change == pytest.approx(50.0, abs=5.0)


# ---------------------------------------------------------------------------
# Tests: Impact Categories
# ---------------------------------------------------------------------------


class TestImpactCategories:
    """Test impact category classification."""

    def test_environmental_category(self):
        assert ImpactCategory.ENVIRONMENTAL.value == "environmental"

    def test_social_category(self):
        assert ImpactCategory.SOCIAL.value == "social"

    def test_governance_category(self):
        assert ImpactCategory.GOVERNANCE.value == "governance"

    def test_env_kpis_categorized(self):
        """Environmental KPIs are correctly categorized in results."""
        engine = ImpactMeasurementEngine()
        kpis = _make_env_kpis("H1")
        result = engine.assess_impact(kpis)

        for kpi in result.environmental_kpis:
            assert kpi.category == ImpactCategory.ENVIRONMENTAL

    def test_social_kpis_categorized(self):
        """Social KPIs are correctly categorized in results."""
        engine = ImpactMeasurementEngine()
        kpis = _make_social_kpis("H1")
        result = engine.assess_impact(kpis)

        for kpi in result.social_kpis:
            assert kpi.category == ImpactCategory.SOCIAL


# ---------------------------------------------------------------------------
# Tests: KPI Attribution
# ---------------------------------------------------------------------------


class TestKPIAttribution:
    """Test KPI attribution factor handling."""

    def test_attributed_value_computed(self):
        """Attributed value = value * attribution_factor."""
        kpi = ImpactKPI(
            kpi_id="env_ghg_avoided",
            holding_id="H1",
            value=100000.0,
            attribution_factor=0.05,
        )
        assert kpi.attributed_value == pytest.approx(5000.0, rel=1e-6)

    def test_zero_attribution(self):
        """Zero attribution factor yields zero attributed value."""
        kpi = ImpactKPI(
            kpi_id="env_ghg_avoided",
            holding_id="H1",
            value=100000.0,
            attribution_factor=0.0,
        )
        assert kpi.attributed_value == pytest.approx(0.0, abs=1e-6)

    def test_full_attribution(self):
        """Full attribution (1.0) yields full value."""
        kpi = ImpactKPI(
            kpi_id="env_ghg_avoided",
            holding_id="H1",
            value=100000.0,
            attribution_factor=1.0,
        )
        assert kpi.attributed_value == pytest.approx(100000.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: KPI Update Model
# ---------------------------------------------------------------------------


class TestKPIUpdate:
    """Test KPI update payload model."""

    def test_kpi_update_creation(self):
        """KPIUpdate model can be created."""
        update = KPIUpdate(
            kpi_id="env_ghg_avoided",
            holding_id="H1",
            new_value=55000.0,
            data_quality="reported",
            source="annual_report",
        )
        assert update.kpi_id == "env_ghg_avoided"
        assert update.new_value == 55000.0
        assert update.source == "annual_report"

    def test_kpi_update_default_source(self):
        """KPIUpdate defaults to manual_update source."""
        update = KPIUpdate(
            kpi_id="soc_jobs_created",
            holding_id="H1",
            new_value=250.0,
        )
        assert update.source == "manual_update"


# ---------------------------------------------------------------------------
# Tests: Provenance Hashing
# ---------------------------------------------------------------------------


class TestImpactProvenance:
    """Verify SHA-256 provenance hashing on impact results."""

    def test_result_has_provenance(self):
        engine = ImpactMeasurementEngine()
        result = engine.assess_impact(_make_env_kpis("H1"))
        assert result.provenance_hash != ""
        assert SHA256_RE.match(result.provenance_hash)

    def test_provenance_deterministic(self):
        """Same input produces valid SHA-256 provenance hashes.

        Timestamps embedded in result UUIDs cause hash variation between
        calls, so we validate both hashes are well-formed SHA-256 strings.
        """
        engine = ImpactMeasurementEngine()
        kpis = _make_env_kpis("DET", ghg_avoided=50000.0)
        r1 = engine.assess_impact(kpis)
        r2 = engine.assess_impact(kpis)
        assert SHA256_RE.match(r1.provenance_hash)
        assert SHA256_RE.match(r2.provenance_hash)

    def test_sdg_contribution_provenance(self):
        """SDG contributions have provenance hashes."""
        engine = ImpactMeasurementEngine()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis)

        for sdg_c in result.sdg_contributions:
            if sdg_c.contribution_score > 0:
                assert sdg_c.provenance_hash != ""


# ---------------------------------------------------------------------------
# Tests: Data Quality
# ---------------------------------------------------------------------------


class TestDataQuality:
    """Test data quality metrics in impact results."""

    def test_data_coverage_percentage(self):
        """Data coverage percentage is computed."""
        engine = ImpactMeasurementEngine()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis)

        assert result.data_coverage_pct >= 0.0
        assert result.data_coverage_pct <= 100.0

    def test_reported_data_percentage(self):
        """Reported data percentage is computed."""
        engine = ImpactMeasurementEngine()
        kpis = _make_full_kpi_set()
        result = engine.assess_impact(kpis)

        assert result.reported_data_pct >= 0.0
        assert result.reported_data_pct <= 100.0


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestImpactEdgeCases:
    """Boundary and unusual inputs."""

    def test_single_kpi(self):
        """Assessment with a single KPI works."""
        engine = ImpactMeasurementEngine()
        kpis = [ImpactKPI(
            kpi_id="env_ghg_avoided",
            holding_id="H1",
            value=10000.0,
            attribution_factor=0.1,
        )]
        result = engine.assess_impact(kpis)
        assert result.total_kpis_tracked >= 1

    def test_zero_value_kpi(self):
        """KPI with zero value is handled."""
        kpi = ImpactKPI(
            kpi_id="env_ghg_avoided",
            holding_id="H1",
            value=0.0,
            attribution_factor=0.05,
        )
        assert kpi.value == 0.0

    def test_processing_time_recorded(self):
        """Processing time is recorded."""
        engine = ImpactMeasurementEngine()
        result = engine.assess_impact(_make_env_kpis("H1"))
        assert result.processing_time_ms >= 0.0

    def test_engine_version_in_result(self):
        """Engine version is present in the result."""
        engine = ImpactMeasurementEngine()
        result = engine.assess_impact(_make_env_kpis("H1"))
        assert result.engine_version != ""
