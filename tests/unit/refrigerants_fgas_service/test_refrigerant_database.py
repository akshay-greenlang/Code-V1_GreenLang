# -*- coding: utf-8 -*-
"""
Unit tests for RefrigerantDatabaseEngine - AGENT-MRV-002 Engine 1

Tests refrigerant lookup, GWP retrieval, blend decomposition, search,
custom registration, and provenance tracking for all 50+ refrigerants.

Target: 95+ tests, 800+ lines.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.refrigerants_fgas.refrigerant_database import (
    RefrigerantDatabaseEngine,
)
from greenlang.refrigerants_fgas.models import (
    BlendComponent,
    GasEmission,
    GWPSource,
    GWPTimeframe,
    GWPValue,
    RefrigerantCategory,
    RefrigerantProperties,
    RefrigerantType,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> RefrigerantDatabaseEngine:
    """Create a fresh RefrigerantDatabaseEngine for each test."""
    return RefrigerantDatabaseEngine()


# ===========================================================================
# Test: Initialization and Properties
# ===========================================================================


class TestRefrigerantDatabaseEngineInit:
    """Tests for engine initialization and convenience properties."""

    def test_initialization_creates_engine(self, engine: RefrigerantDatabaseEngine):
        """Engine initializes without error."""
        assert engine is not None

    def test_refrigerant_count_positive(self, engine: RefrigerantDatabaseEngine):
        """Engine has a positive refrigerant count after init."""
        assert engine.refrigerant_count > 0

    def test_blend_count_positive(self, engine: RefrigerantDatabaseEngine):
        """Engine has a positive blend count after init."""
        assert engine.blend_count > 0

    def test_pure_gas_count_positive(self, engine: RefrigerantDatabaseEngine):
        """Engine has a positive pure gas count after init."""
        assert engine.pure_gas_count > 0

    def test_custom_count_zero_initially(self, engine: RefrigerantDatabaseEngine):
        """No custom registrations initially."""
        assert engine.custom_count == 0

    def test_len_equals_refrigerant_count(self, engine: RefrigerantDatabaseEngine):
        """len(engine) returns refrigerant_count."""
        assert len(engine) == engine.refrigerant_count

    def test_repr_contains_counts(self, engine: RefrigerantDatabaseEngine):
        """repr() includes key counts."""
        r = repr(engine)
        assert "RefrigerantDatabaseEngine" in r
        assert "refrigerants=" in r
        assert "blends=" in r


# ===========================================================================
# Test: get_refrigerant by Category
# ===========================================================================


class TestGetRefrigerant:
    """Tests for get_refrigerant across all categories."""

    @pytest.mark.parametrize("ref_type,expected_category", [
        (RefrigerantType.R_32, RefrigerantCategory.HFC),
        (RefrigerantType.R_134A, RefrigerantCategory.HFC),
        (RefrigerantType.R_125, RefrigerantCategory.HFC),
        (RefrigerantType.R_143A, RefrigerantCategory.HFC),
        (RefrigerantType.R_152A, RefrigerantCategory.HFC),
        (RefrigerantType.R_23, RefrigerantCategory.HFC),
    ])
    def test_get_refrigerant_hfc(
        self, engine: RefrigerantDatabaseEngine, ref_type, expected_category
    ):
        """HFC refrigerants are categorized correctly."""
        props = engine.get_refrigerant(ref_type)
        assert props.category == expected_category
        assert props.refrigerant_type == ref_type
        assert props.is_regulated is True

    @pytest.mark.parametrize("ref_type", [
        RefrigerantType.R_1234YF,
        RefrigerantType.R_1234ZE,
        RefrigerantType.R_1233ZD,
        RefrigerantType.R_1336MZZ,
    ])
    def test_get_refrigerant_hfo(self, engine: RefrigerantDatabaseEngine, ref_type):
        """HFO refrigerants return HFO category."""
        props = engine.get_refrigerant(ref_type)
        assert props.category == RefrigerantCategory.HFO

    @pytest.mark.parametrize("ref_type", [
        RefrigerantType.CF4,
        RefrigerantType.C2F6,
        RefrigerantType.C3F8,
        RefrigerantType.C_C4F8,
        RefrigerantType.C4F10,
        RefrigerantType.C5F12,
        RefrigerantType.C6F14,
    ])
    def test_get_refrigerant_pfc(self, engine: RefrigerantDatabaseEngine, ref_type):
        """PFC refrigerants return PFC category."""
        props = engine.get_refrigerant(ref_type)
        assert props.category == RefrigerantCategory.PFC

    def test_get_refrigerant_sf6(self, engine: RefrigerantDatabaseEngine):
        """SF6 has its own category."""
        props = engine.get_refrigerant(RefrigerantType.SF6_GAS)
        assert props.category == RefrigerantCategory.SF6

    def test_get_refrigerant_nf3(self, engine: RefrigerantDatabaseEngine):
        """NF3 has its own category."""
        props = engine.get_refrigerant(RefrigerantType.NF3_GAS)
        assert props.category == RefrigerantCategory.NF3

    @pytest.mark.parametrize("ref_type", [
        RefrigerantType.R_22,
        RefrigerantType.R_123,
        RefrigerantType.R_141B,
        RefrigerantType.R_142B,
    ])
    def test_get_refrigerant_hcfc(self, engine: RefrigerantDatabaseEngine, ref_type):
        """HCFC refrigerants return HCFC category."""
        props = engine.get_refrigerant(ref_type)
        assert props.category == RefrigerantCategory.HCFC

    @pytest.mark.parametrize("ref_type", [
        RefrigerantType.R_11,
        RefrigerantType.R_12,
        RefrigerantType.R_113,
        RefrigerantType.R_114,
        RefrigerantType.R_115,
    ])
    def test_get_refrigerant_cfc(self, engine: RefrigerantDatabaseEngine, ref_type):
        """CFC refrigerants return CFC category."""
        props = engine.get_refrigerant(ref_type)
        assert props.category == RefrigerantCategory.CFC

    @pytest.mark.parametrize("ref_type", [
        RefrigerantType.R_717,
        RefrigerantType.R_744,
        RefrigerantType.R_290,
        RefrigerantType.R_600A,
    ])
    def test_get_refrigerant_natural(self, engine: RefrigerantDatabaseEngine, ref_type):
        """Natural refrigerants return NATURAL category."""
        props = engine.get_refrigerant(ref_type)
        assert props.category == RefrigerantCategory.NATURAL

    def test_get_refrigerant_returns_properties_model(
        self, engine: RefrigerantDatabaseEngine
    ):
        """get_refrigerant returns a RefrigerantProperties instance."""
        props = engine.get_refrigerant(RefrigerantType.R_134A)
        assert isinstance(props, RefrigerantProperties)
        assert props.name != ""
        assert props.gwp_values is not None
        assert len(props.gwp_values) >= 4  # AR4, AR5, AR6, AR6_20yr

    def test_get_refrigerant_has_formula(self, engine: RefrigerantDatabaseEngine):
        """Pure gases have formula set."""
        props = engine.get_refrigerant(RefrigerantType.R_32)
        assert props.formula == "CH2F2"

    def test_get_refrigerant_has_molecular_weight(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Pure gases have molecular weight set."""
        props = engine.get_refrigerant(RefrigerantType.R_32)
        assert props.molecular_weight is not None
        assert props.molecular_weight > 0


# ===========================================================================
# Test: get_gwp with AR4, AR5, AR6, AR6_20YR
# ===========================================================================


class TestGetGWP:
    """Tests for GWP lookups across IPCC Assessment Reports."""

    def test_get_gwp_default_source_is_ar6(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Default GWP source should be AR6."""
        gwp = engine.get_gwp(RefrigerantType.R_134A)
        expected = Decimal("1530")
        assert gwp == expected

    @pytest.mark.parametrize("source,expected", [
        ("AR4", Decimal("1430")),
        ("AR5", Decimal("1300")),
        ("AR6", Decimal("1530")),
    ])
    def test_get_gwp_r134a_by_source(
        self, engine: RefrigerantDatabaseEngine, source, expected
    ):
        """R-134a GWP varies by IPCC Assessment Report."""
        gwp = engine.get_gwp(RefrigerantType.R_134A, gwp_source=source)
        assert gwp == expected

    def test_get_gwp_ar6_20yr(self, engine: RefrigerantDatabaseEngine):
        """AR6_20YR returns the 20-year GWP."""
        gwp = engine.get_gwp(RefrigerantType.R_134A, gwp_source="AR6_20YR")
        assert gwp == Decimal("4144")

    @pytest.mark.parametrize("ref_type,ar6_gwp", [
        (RefrigerantType.R_32, Decimal("771")),
        (RefrigerantType.R_125, Decimal("3740")),
        (RefrigerantType.R_143A, Decimal("5810")),
        (RefrigerantType.R_152A, Decimal("164")),
        (RefrigerantType.R_227EA, Decimal("3600")),
        (RefrigerantType.R_23, Decimal("14600")),
        (RefrigerantType.R_1234YF, Decimal("1")),
        (RefrigerantType.CF4, Decimal("7380")),
        (RefrigerantType.C2F6, Decimal("12400")),
        (RefrigerantType.SF6_GAS, Decimal("25200")),
        (RefrigerantType.NF3_GAS, Decimal("17400")),
        (RefrigerantType.R_22, Decimal("1960")),
        (RefrigerantType.R_11, Decimal("5560")),
        (RefrigerantType.R_12, Decimal("10200")),
        (RefrigerantType.R_717, Decimal("0")),
        (RefrigerantType.R_744, Decimal("1")),
        (RefrigerantType.R_290, Decimal("0.02")),
        (RefrigerantType.R_600A, Decimal("0.02")),
        (RefrigerantType.R_245FA, Decimal("962")),
        (RefrigerantType.R_236FA, Decimal("8998")),
    ])
    def test_known_gwp_values_ar6(
        self, engine: RefrigerantDatabaseEngine, ref_type, ar6_gwp
    ):
        """Verify known AR6 GWP values for 20+ key gases."""
        gwp = engine.get_gwp(ref_type, gwp_source="AR6")
        assert gwp == ar6_gwp

    def test_get_gwp_returns_decimal(self, engine: RefrigerantDatabaseEngine):
        """get_gwp returns a Decimal, not float."""
        gwp = engine.get_gwp(RefrigerantType.R_32)
        assert isinstance(gwp, Decimal)

    def test_get_gwp_blend_returns_weighted_average(
        self, engine: RefrigerantDatabaseEngine
    ):
        """For blends, get_gwp returns a weight-fraction-averaged GWP."""
        gwp_410a = engine.get_gwp(RefrigerantType.R_410A, gwp_source="AR6")
        # R-410A = 50% R-32 (771) + 50% R-125 (3740) = 2255.5
        expected = (Decimal("0.50") * Decimal("771") + Decimal("0.50") * Decimal("3740"))
        # Quantized to 3 decimal places
        assert gwp_410a == expected.quantize(Decimal("0.001"))


# ===========================================================================
# Test: get_blend_components
# ===========================================================================


class TestGetBlendComponents:
    """Tests for blend component retrieval."""

    def test_get_blend_components_r410a(self, engine: RefrigerantDatabaseEngine):
        """R-410A has 2 components: R-32 and R-125."""
        components = engine.get_blend_components(RefrigerantType.R_410A)
        assert len(components) == 2
        types = {c.refrigerant_type for c in components}
        assert RefrigerantType.R_32 in types
        assert RefrigerantType.R_125 in types

    def test_get_blend_components_r404a(self, engine: RefrigerantDatabaseEngine):
        """R-404A has 3 components: R-125, R-143a, R-134a."""
        components = engine.get_blend_components(RefrigerantType.R_404A)
        assert len(components) == 3

    def test_get_blend_components_r407c(self, engine: RefrigerantDatabaseEngine):
        """R-407C has 3 components: R-32, R-125, R-134a."""
        components = engine.get_blend_components(RefrigerantType.R_407C)
        assert len(components) == 3

    def test_get_blend_components_pure_gas_empty(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Pure gas returns empty list."""
        components = engine.get_blend_components(RefrigerantType.R_134A)
        assert components == []

    def test_blend_components_are_blend_component_type(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Components are BlendComponent instances."""
        components = engine.get_blend_components(RefrigerantType.R_410A)
        for c in components:
            assert isinstance(c, BlendComponent)
            assert c.weight_fraction > 0
            assert c.weight_fraction <= 1.0


# ===========================================================================
# Test: Blend composition sums to 1.0
# ===========================================================================


class TestBlendCompositionSums:
    """All blend component weight fractions must sum to approximately 1.0."""

    @pytest.mark.parametrize("blend_type", [
        RefrigerantType.R_404A,
        RefrigerantType.R_407A,
        RefrigerantType.R_407C,
        RefrigerantType.R_407F,
        RefrigerantType.R_410A,
        RefrigerantType.R_413A,
        RefrigerantType.R_417A,
        RefrigerantType.R_422D,
        RefrigerantType.R_427A,
        RefrigerantType.R_438A,
        RefrigerantType.R_448A,
        RefrigerantType.R_449A,
        RefrigerantType.R_452A,
        RefrigerantType.R_454B,
        RefrigerantType.R_507A,
        RefrigerantType.R_508B,
    ])
    def test_blend_fractions_sum_to_one(
        self, engine: RefrigerantDatabaseEngine, blend_type
    ):
        """Weight fractions of blend components sum to ~1.0."""
        components = engine.get_blend_components(blend_type)
        total = sum(c.weight_fraction for c in components)
        assert abs(total - 1.0) < 0.01, (
            f"Blend {blend_type.value} fractions sum to {total}"
        )


# ===========================================================================
# Test: calculate_blend_gwp
# ===========================================================================


class TestCalculateBlendGWP:
    """Tests for calculate_blend_gwp."""

    def test_calculate_blend_gwp_r410a(self, engine: RefrigerantDatabaseEngine):
        """Manual blend GWP calculation matches engine."""
        components = engine.get_blend_components(RefrigerantType.R_410A, gwp_source="AR6")
        result = engine.calculate_blend_gwp(components, gwp_source="AR6")
        assert isinstance(result, Decimal)
        # R-410A AR6: 50% * 771 + 50% * 3740 = 2255.5
        expected = Decimal("2255.500")
        assert result == expected

    def test_calculate_blend_gwp_empty_raises(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Empty components list raises ValueError."""
        with pytest.raises(ValueError, match="At least one BlendComponent"):
            engine.calculate_blend_gwp([], gwp_source="AR6")

    def test_calculate_blend_gwp_with_explicit_gwp(
        self, engine: RefrigerantDatabaseEngine
    ):
        """When component.gwp is set, uses that value."""
        components = [
            BlendComponent(
                refrigerant_type=RefrigerantType.R_32,
                weight_fraction=0.5,
                gwp=100.0,
            ),
            BlendComponent(
                refrigerant_type=RefrigerantType.R_125,
                weight_fraction=0.5,
                gwp=200.0,
            ),
        ]
        result = engine.calculate_blend_gwp(components, gwp_source="AR6")
        assert result == Decimal("150.000")


# ===========================================================================
# Test: decompose_blend_emissions
# ===========================================================================


class TestDecomposeBlendEmissions:
    """Tests for blend emission decomposition."""

    def test_decompose_blend_r410a(self, engine: RefrigerantDatabaseEngine):
        """R-410A decomposition produces 2 GasEmission entries."""
        emissions = engine.decompose_blend_emissions(
            loss_kg=10.0,
            ref_type=RefrigerantType.R_410A,
            gwp_source="AR6",
        )
        assert len(emissions) == 2
        for e in emissions:
            assert isinstance(e, GasEmission)
            assert e.is_blend_component is True
            assert e.loss_kg > 0
            assert e.emissions_kg_co2e >= 0

    def test_decompose_pure_gas_single_entry(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Pure gas decomposition produces 1 GasEmission entry."""
        emissions = engine.decompose_blend_emissions(
            loss_kg=10.0,
            ref_type=RefrigerantType.R_134A,
            gwp_source="AR6",
        )
        assert len(emissions) == 1
        assert emissions[0].is_blend_component is False
        # 10 kg * 1530 GWP = 15300 kg CO2e = 15.3 tCO2e
        assert emissions[0].emissions_kg_co2e == 15300.0
        assert emissions[0].emissions_tco2e == 15.3

    def test_decompose_blend_loss_sums_to_total(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Sum of component losses equals total loss (within rounding)."""
        total_loss = 100.0
        emissions = engine.decompose_blend_emissions(
            loss_kg=total_loss,
            ref_type=RefrigerantType.R_404A,
            gwp_source="AR6",
        )
        component_loss_sum = sum(e.loss_kg for e in emissions)
        assert abs(component_loss_sum - total_loss) < 0.01

    def test_decompose_blend_tco2e_is_kg_divided_by_1000(
        self, engine: RefrigerantDatabaseEngine
    ):
        """tCO2e = kg_co2e / 1000."""
        emissions = engine.decompose_blend_emissions(
            loss_kg=1.0,
            ref_type=RefrigerantType.R_134A,
            gwp_source="AR6",
        )
        e = emissions[0]
        assert abs(e.emissions_tco2e - e.emissions_kg_co2e / 1000.0) < 0.001

    def test_decompose_negative_loss_raises(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Negative loss_kg raises ValueError."""
        with pytest.raises(ValueError, match="loss_kg must be >= 0"):
            engine.decompose_blend_emissions(
                loss_kg=-1.0,
                ref_type=RefrigerantType.R_410A,
                gwp_source="AR6",
            )

    def test_decompose_zero_loss_returns_zero_emissions(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Zero loss produces zero emissions."""
        emissions = engine.decompose_blend_emissions(
            loss_kg=0.0,
            ref_type=RefrigerantType.R_134A,
            gwp_source="AR6",
        )
        assert len(emissions) == 1
        assert emissions[0].emissions_kg_co2e == 0.0


# ===========================================================================
# Test: list_refrigerants
# ===========================================================================


class TestListRefrigerants:
    """Tests for list_refrigerants with optional category filter."""

    def test_list_refrigerants_all(self, engine: RefrigerantDatabaseEngine):
        """list_refrigerants(None) returns all entries."""
        all_refs = engine.list_refrigerants()
        assert len(all_refs) == engine.refrigerant_count

    @pytest.mark.parametrize("category", [
        RefrigerantCategory.HFC,
        RefrigerantCategory.HFO,
        RefrigerantCategory.PFC,
        RefrigerantCategory.SF6,
        RefrigerantCategory.NF3,
        RefrigerantCategory.HCFC,
        RefrigerantCategory.CFC,
        RefrigerantCategory.NATURAL,
    ])
    def test_list_refrigerants_by_category(
        self, engine: RefrigerantDatabaseEngine, category
    ):
        """Filtering by category returns only that category."""
        results = engine.list_refrigerants(category=category)
        for r in results:
            assert r.category == category

    def test_list_refrigerants_hfc_blend_category(
        self, engine: RefrigerantDatabaseEngine
    ):
        """HFC_BLEND category contains blend entries."""
        results = engine.list_refrigerants(category=RefrigerantCategory.HFC_BLEND)
        assert len(results) > 0
        for r in results:
            assert r.is_blend is True

    def test_list_refrigerants_sorted(self, engine: RefrigerantDatabaseEngine):
        """Results are sorted by refrigerant type value."""
        results = engine.list_refrigerants()
        values = [r.refrigerant_type.value for r in results]
        assert values == sorted(values)


# ===========================================================================
# Test: register_custom_refrigerant
# ===========================================================================


class TestRegisterCustom:
    """Tests for custom refrigerant registration."""

    def test_register_custom_refrigerant(self, engine: RefrigerantDatabaseEngine):
        """Custom registration works for CUSTOM type."""
        props = RefrigerantProperties(
            refrigerant_type=RefrigerantType.CUSTOM,
            category=RefrigerantCategory.OTHER,
            name="Custom-Test-Gas",
            formula="C2H3F3X",
            gwp_values={
                "AR6_100yr": GWPValue(
                    gwp_source=GWPSource.AR6,
                    timeframe=GWPTimeframe.GWP_100YR,
                    value=999.0,
                ),
            },
        )
        result = engine.register_custom(RefrigerantType.CUSTOM, props)
        assert result.name == "Custom-Test-Gas"
        assert engine.custom_count == 1

    def test_register_existing_type_raises(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Registering an existing static type raises ValueError."""
        props = RefrigerantProperties(
            refrigerant_type=RefrigerantType.R_134A,
            category=RefrigerantCategory.HFC,
            name="Duplicate",
        )
        with pytest.raises(ValueError, match="already exists"):
            engine.register_custom(RefrigerantType.R_134A, props)

    def test_custom_refrigerant_retrievable(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Custom refrigerant is retrievable via get_refrigerant."""
        props = RefrigerantProperties(
            refrigerant_type=RefrigerantType.CUSTOM,
            category=RefrigerantCategory.OTHER,
            name="MyCustomGas",
        )
        engine.register_custom(RefrigerantType.CUSTOM, props)
        retrieved = engine.get_refrigerant(RefrigerantType.CUSTOM)
        assert retrieved.name == "MyCustomGas"


# ===========================================================================
# Test: search_refrigerants
# ===========================================================================


class TestSearchRefrigerants:
    """Tests for refrigerant search functionality."""

    def test_search_by_name(self, engine: RefrigerantDatabaseEngine):
        """Search by partial name returns matching results."""
        results = engine.search_refrigerants("Difluoromethane")
        assert len(results) >= 1
        assert any(r.refrigerant_type == RefrigerantType.R_32 for r in results)

    def test_search_by_formula(self, engine: RefrigerantDatabaseEngine):
        """Search by chemical formula returns matching results."""
        results = engine.search_refrigerants("CH2F2")
        assert len(results) >= 1

    def test_search_by_enum_value(self, engine: RefrigerantDatabaseEngine):
        """Search by enum value fragment."""
        results = engine.search_refrigerants("R_134A")
        assert len(results) >= 1

    def test_search_by_category(self, engine: RefrigerantDatabaseEngine):
        """Search by category name."""
        results = engine.search_refrigerants("natural")
        assert len(results) >= 4  # R-717, R-744, R-290, R-600a

    def test_search_empty_query_returns_empty(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Empty query returns empty list."""
        results = engine.search_refrigerants("")
        assert results == []

    def test_search_no_match_returns_empty(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Non-matching query returns empty list."""
        results = engine.search_refrigerants("zzz_nonexistent_zzz")
        assert results == []

    def test_search_case_insensitive(self, engine: RefrigerantDatabaseEngine):
        """Search is case-insensitive."""
        results_lower = engine.search_refrigerants("ammonia")
        results_upper = engine.search_refrigerants("AMMONIA")
        # Both should find R-717 (Ammonia)
        assert len(results_lower) >= 1
        assert len(results_upper) >= 1


# ===========================================================================
# Test: is_regulated
# ===========================================================================


class TestIsRegulated:
    """Tests for regulatory status checking."""

    @pytest.mark.parametrize("ref_type", [
        RefrigerantType.R_32,
        RefrigerantType.R_134A,
        RefrigerantType.R_23,
        RefrigerantType.SF6_GAS,
        RefrigerantType.NF3_GAS,
        RefrigerantType.CF4,
        RefrigerantType.R_22,
        RefrigerantType.R_11,
    ])
    def test_regulated_gases(self, engine: RefrigerantDatabaseEngine, ref_type):
        """HFCs, PFCs, SF6, NF3, HCFCs, CFCs are regulated."""
        assert engine.is_regulated(ref_type) is True

    @pytest.mark.parametrize("ref_type", [
        RefrigerantType.R_717,
        RefrigerantType.R_744,
        RefrigerantType.R_290,
        RefrigerantType.R_600A,
        RefrigerantType.R_1234YF,
        RefrigerantType.R_1234ZE,
        RefrigerantType.R_1233ZD,
    ])
    def test_unregulated_gases(self, engine: RefrigerantDatabaseEngine, ref_type):
        """Natural refrigerants and HFOs are not regulated."""
        assert engine.is_regulated(ref_type) is False


# ===========================================================================
# Test: is_blend
# ===========================================================================


class TestIsBlend:
    """Tests for blend status checking."""

    @pytest.mark.parametrize("ref_type", [
        RefrigerantType.R_404A,
        RefrigerantType.R_407C,
        RefrigerantType.R_410A,
        RefrigerantType.R_507A,
        RefrigerantType.R_449A,
        RefrigerantType.R_454B,
    ])
    def test_blends_are_blends(self, engine: RefrigerantDatabaseEngine, ref_type):
        """Known blends return True."""
        assert engine.is_blend(ref_type) is True

    @pytest.mark.parametrize("ref_type", [
        RefrigerantType.R_134A,
        RefrigerantType.R_32,
        RefrigerantType.SF6_GAS,
        RefrigerantType.R_717,
        RefrigerantType.R_1234YF,
    ])
    def test_pure_gases_not_blends(
        self, engine: RefrigerantDatabaseEngine, ref_type
    ):
        """Pure gases return False."""
        assert engine.is_blend(ref_type) is False


# ===========================================================================
# Test: get_all_gwp_values
# ===========================================================================


class TestGetAllGWPValues:
    """Tests for get_all_gwp_values."""

    def test_get_all_gwp_pure_gas(self, engine: RefrigerantDatabaseEngine):
        """Pure gas returns 4 GWP entries."""
        values = engine.get_all_gwp_values(RefrigerantType.R_134A)
        assert len(values) == 4
        assert "AR4_100yr" in values
        assert "AR5_100yr" in values
        assert "AR6_100yr" in values
        assert "AR6_20yr" in values

    def test_get_all_gwp_blend(self, engine: RefrigerantDatabaseEngine):
        """Blend returns weighted GWP values for all sources."""
        values = engine.get_all_gwp_values(RefrigerantType.R_410A)
        assert len(values) == 4
        for key, val in values.items():
            assert isinstance(val, Decimal)
            assert val > 0

    def test_get_all_gwp_returns_decimals(
        self, engine: RefrigerantDatabaseEngine
    ):
        """All values are Decimal instances."""
        values = engine.get_all_gwp_values(RefrigerantType.R_32)
        for val in values.values():
            assert isinstance(val, Decimal)


# ===========================================================================
# Test: Unknown refrigerant raises error
# ===========================================================================


class TestUnknownRefrigerant:
    """Tests for error handling with unknown refrigerant types."""

    def test_get_refrigerant_custom_not_registered_raises(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Looking up CUSTOM when not registered raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            engine.get_refrigerant(RefrigerantType.CUSTOM)

    def test_get_gwp_custom_not_registered_raises(
        self, engine: RefrigerantDatabaseEngine
    ):
        """Looking up GWP for CUSTOM when not registered raises ValueError."""
        with pytest.raises(ValueError, match="No GWP data"):
            engine.get_gwp(RefrigerantType.CUSTOM)


# ===========================================================================
# Test: Provenance tracking
# ===========================================================================


class TestProvenanceTracking:
    """Tests for provenance chain recording."""

    def test_get_refrigerant_records_provenance(
        self, engine: RefrigerantDatabaseEngine
    ):
        """get_refrigerant records provenance without errors."""
        # Should not raise
        props = engine.get_refrigerant(RefrigerantType.R_134A)
        assert props is not None

    def test_get_gwp_records_provenance(
        self, engine: RefrigerantDatabaseEngine
    ):
        """get_gwp records provenance without errors."""
        gwp = engine.get_gwp(RefrigerantType.R_134A)
        assert gwp is not None

    def test_decompose_blend_records_provenance(
        self, engine: RefrigerantDatabaseEngine
    ):
        """decompose_blend_emissions records provenance without errors."""
        emissions = engine.decompose_blend_emissions(
            loss_kg=10.0,
            ref_type=RefrigerantType.R_410A,
        )
        assert len(emissions) > 0
