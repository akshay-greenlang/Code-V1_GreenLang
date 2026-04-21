# -*- coding: utf-8 -*-
"""GAP-7 Industry code taxonomy + crosswalk tests."""
from __future__ import annotations

import pytest

from greenlang.factors.mapping.base import MappingConfidence, MappingError
from greenlang.factors.mapping.classifications import (
    TradeCodeSystem,
    map_trade_code,
    parse_trade_code,
)
from greenlang.factors.mapping.industry_codes import (
    CodeLevel,
    IndustryCode,
    IndustryCodeSystem,
    children_of,
    count_codes,
    crosswalk_code,
    get_sector_default_ef,
    list_systems,
    load_codes,
    lookup_industry_code,
    lookup_industry_label,
    parent_of,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestIndustryCodeSystem:
    def test_values(self):
        vals = {s.value for s in IndustryCodeSystem}
        assert vals == {"isic", "nace", "naics", "anzsic", "jsic"}

    def test_from_str_happy(self):
        assert IndustryCodeSystem.from_str("isic") == IndustryCodeSystem.ISIC
        assert IndustryCodeSystem.from_str("NACE") == IndustryCodeSystem.NACE
        assert IndustryCodeSystem.from_str(" naics ") == IndustryCodeSystem.NAICS

    def test_from_str_empty(self):
        with pytest.raises(MappingError):
            IndustryCodeSystem.from_str("")

    def test_from_str_unknown(self):
        with pytest.raises(MappingError):
            IndustryCodeSystem.from_str("sic87")


class TestCodeLevel:
    def test_values(self):
        vals = {c.value for c in CodeLevel}
        assert vals == {"section", "division", "group", "class", "national"}


# ---------------------------------------------------------------------------
# Load + coverage
# ---------------------------------------------------------------------------


class TestLoaders:
    @pytest.mark.parametrize("system", ["isic", "nace", "naics", "anzsic", "jsic"])
    def test_loads(self, system):
        codes = load_codes(IndustryCodeSystem.from_str(system))
        assert isinstance(codes, dict)
        assert len(codes) > 0

    def test_isic_has_manufacturing_section(self):
        codes = load_codes(IndustryCodeSystem.ISIC)
        assert "C" in codes
        assert codes["C"].label.lower().startswith("manufacturing")

    def test_isic_has_steel_class(self):
        codes = load_codes(IndustryCodeSystem.ISIC)
        assert "2410" in codes
        assert "iron and steel" in codes["2410"].label.lower()

    def test_nace_steel_class(self):
        codes = load_codes(IndustryCodeSystem.NACE)
        assert "2410" in codes

    def test_naics_steel_class(self):
        codes = load_codes(IndustryCodeSystem.NAICS)
        assert "331110" in codes

    def test_counts_are_large(self):
        assert count_codes("isic") > 80
        assert count_codes("nace") > 80
        assert count_codes("naics") > 80

    def test_list_systems(self):
        systems = list_systems()
        assert set(systems) == {"isic", "nace", "naics", "anzsic", "jsic"}


# ---------------------------------------------------------------------------
# Code lookup
# ---------------------------------------------------------------------------


class TestLookupIndustryCode:
    def test_isic_division(self):
        r = lookup_industry_code("isic", "35")
        assert r.canonical["code"] == "35"
        assert "electricity" in r.canonical["label"].lower()

    def test_isic_class_numeric(self):
        r = lookup_industry_code("isic", "2410")
        assert r.canonical["code"] == "2410"
        assert r.canonical["ef_default"] is not None

    def test_isic_with_section_prefix(self):
        """'C2410' should strip leading 'C' and resolve to '2410'."""
        r = lookup_industry_code("isic", "C2410")
        assert r.canonical["code"] == "2410"

    def test_isic_section_letter(self):
        r = lookup_industry_code("isic", "C")
        assert r.canonical["code"] == "C"
        assert r.canonical["level"] == "section"

    def test_dots_tolerated(self):
        r = lookup_industry_code("naics", "331.110")
        assert r.canonical["code"] == "331110"

    def test_hyphens_tolerated(self):
        r = lookup_industry_code("naics", "331-110")
        assert r.canonical["code"] == "331110"

    def test_missing_code_returns_none(self):
        r = lookup_industry_code("isic", "9999")
        assert r.canonical is None
        assert r.band == MappingConfidence.UNKNOWN

    def test_invalid_system_raises(self):
        with pytest.raises(MappingError):
            lookup_industry_code("made-up", "1234")


class TestLookupLabel:
    def test_exact_label(self):
        r = lookup_industry_label("isic", "Mining of hard coal")
        assert r.canonical["code"] == "0510"

    def test_case_insensitive(self):
        r = lookup_industry_label("isic", "mining of hard coal")
        assert r.canonical["code"] == "0510"

    def test_punct_insensitive(self):
        r = lookup_industry_label("isic", "Mining of hard coal!!")
        assert r.canonical["code"] == "0510"

    def test_partial_token_overlap(self):
        r = lookup_industry_label("isic", "iron steel manufacture")
        assert r.canonical is not None
        assert r.band in (
            MappingConfidence.EXACT,
            MappingConfidence.HIGH,
            MappingConfidence.MEDIUM,
            MappingConfidence.LOW,
        )

    def test_empty_label(self):
        r = lookup_industry_label("isic", "")
        assert r.canonical is None

    def test_no_match(self):
        r = lookup_industry_label("isic", "alien probes and tractor beams")
        assert r.canonical is None


class TestHierarchyNavigation:
    def test_children_of_manufacturing(self):
        kids = children_of("isic", "C")
        # Every division in section C should be present
        codes = {k.code for k in kids}
        assert {"10", "20", "24", "33"} <= codes

    def test_children_of_mining_section(self):
        kids = children_of("isic", "B")
        codes = {k.code for k in kids}
        assert {"05", "06", "07", "08", "09"} <= codes

    def test_parent_of_class(self):
        parent = parent_of("isic", "2410")
        assert parent is not None
        assert parent.code == "24"

    def test_parent_of_division(self):
        parent = parent_of("isic", "24")
        assert parent is not None
        assert parent.code == "C"

    def test_parent_of_section_none(self):
        parent = parent_of("isic", "C")
        assert parent is None

    def test_parent_of_missing_code(self):
        parent = parent_of("isic", "9999")
        assert parent is None


class TestSectorDefaultEF:
    def test_ef_for_known_class(self):
        ef = get_sector_default_ef("isic", "2410")
        assert ef is not None
        assert 1.0 < ef < 3.0  # steel is carbon-intensive

    def test_ef_none_for_unknown(self):
        assert get_sector_default_ef("isic", "9999") is None

    def test_ef_for_naics(self):
        ef = get_sector_default_ef("naics", "327310")  # Cement
        assert ef is not None
        assert ef > 1.0


# ---------------------------------------------------------------------------
# Crosswalks
# ---------------------------------------------------------------------------


class TestCrosswalks:
    def test_isic_to_nace_identical_division(self):
        r = crosswalk_code("isic", "C", "nace")
        # Section letters not included in the crosswalk YAML; same-system
        # fallback kicks in only on src==dst; otherwise unknown.
        assert r.canonical is None or r.canonical["to_code"] == "C"

    def test_isic_to_nace_class_one_to_one(self):
        r = crosswalk_code("isic", "1062", "nace")
        assert r.canonical["to_code"] == "1062"
        assert r.canonical["match_quality"] == "one_to_one"

    def test_isic_to_naics_cement(self):
        r = crosswalk_code("isic", "2394", "naics")
        codes = r.canonical["to_code"]
        assert isinstance(codes, list)
        assert "327310" in codes  # Cement Manufacturing

    def test_naics_to_isic_inverse(self):
        r = crosswalk_code("naics", "327310", "isic")
        assert r.canonical is not None
        assert "2394" in (
            r.canonical["to_code"] if isinstance(r.canonical["to_code"], list)
            else [r.canonical["to_code"]]
        )

    def test_nace_to_isic_inverse(self):
        r = crosswalk_code("nace", "1062", "isic")
        assert r.canonical is not None

    def test_nace_to_naics_routes_via_isic(self):
        r = crosswalk_code("nace", "2351", "naics")
        # Cement - NACE 2351 -> ISIC 2394 -> NAICS 327310
        assert r.canonical is not None
        to = r.canonical["to_code"]
        flat = to if isinstance(to, list) else [to]
        assert "327310" in flat

    def test_crosswalk_same_system(self):
        r = crosswalk_code("isic", "2410", "isic")
        assert r.canonical["code"] == "2410"

    def test_crosswalk_unsupported_anzsic(self):
        r = crosswalk_code("anzsic", "2011", "naics")
        assert r.canonical is None

    def test_isic_to_nace_with_section_prefix(self):
        r = crosswalk_code("isic", "C2410", "nace")
        assert r.canonical is not None

    def test_crosswalk_unknown_source_code(self):
        r = crosswalk_code("isic", "9999", "nace")
        assert r.canonical is None

    def test_crosswalk_bidirectional_consistency(self):
        """ISIC -> NAICS -> ISIC should round-trip through the cement class."""
        forward = crosswalk_code("isic", "2394", "naics")
        assert forward.canonical is not None
        # take first NAICS target
        naics_target = forward.canonical["to_code"]
        naics_target = naics_target[0] if isinstance(naics_target, list) else naics_target
        back = crosswalk_code("naics", naics_target, "isic")
        assert back.canonical is not None


# ---------------------------------------------------------------------------
# Trade codes (HS / CN / CPC)
# ---------------------------------------------------------------------------


class TestTradeCodeSystem:
    def test_values(self):
        vals = {t.value for t in TradeCodeSystem}
        assert vals == {"hs", "cn", "cpc"}

    def test_from_str(self):
        assert TradeCodeSystem.from_str("hs") == TradeCodeSystem.HS
        assert TradeCodeSystem.from_str("CPC") == TradeCodeSystem.CPC

    def test_from_str_empty(self):
        with pytest.raises(MappingError):
            TradeCodeSystem.from_str("")

    def test_from_str_unknown(self):
        with pytest.raises(MappingError):
            TradeCodeSystem.from_str("siuxa")


class TestParseTradeCode:
    def test_parse_hs_6digit(self):
        p = parse_trade_code("hs", "720810")
        assert p["chapter"] == "72"
        assert p["heading"] == "7208"
        assert p["subheading"] == "720810"
        assert p["length"] == 6

    def test_parse_cn_8digit(self):
        p = parse_trade_code("cn", "76011000")
        assert p["chapter"] == "76"
        assert p["heading"] == "7601"

    def test_parse_cpc_5digit(self):
        p = parse_trade_code("cpc", "41111")
        assert p["chapter"] == "41"
        assert p["heading"] == "411"
        assert p["subheading"] == "41111"

    def test_dots_tolerated(self):
        p = parse_trade_code("hs", "7208.10")
        assert p["code"] == "720810"

    def test_hyphens_tolerated(self):
        p = parse_trade_code("hs", "7208-10")
        assert p["code"] == "720810"

    def test_non_numeric_rejected(self):
        with pytest.raises(MappingError):
            parse_trade_code("hs", "ABCDEF")

    def test_unknown_system_rejected(self):
        with pytest.raises(MappingError):
            parse_trade_code("blah", "1234")


class TestMapTradeCode:
    def test_hs_steel_prefix(self):
        r = map_trade_code("hs", "720810")
        assert r.canonical is not None
        assert r.canonical["factor_family"] == "material_embodied"
        assert r.canonical["parsed"]["chapter"] == "72"

    def test_cn_aluminium_prefix(self):
        r = map_trade_code("cn", "76011000")
        assert r.canonical is not None
        assert r.canonical["label"].lower().startswith("aluminium")

    def test_unknown_trade_code(self):
        r = map_trade_code("hs", "999999")
        assert r.canonical is None

    def test_non_numeric_returns_unknown(self):
        r = map_trade_code("hs", "ABCDEF")
        assert r.canonical is None
