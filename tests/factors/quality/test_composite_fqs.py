# -*- coding: utf-8 -*-
"""Tests for the 0-100 composite Factor Quality Score."""
from __future__ import annotations

import pytest

from greenlang.data.emission_factor_record import DataQualityScore
from greenlang.factors.quality.composite_fqs import (
    CTO_SPEC_ALIASES,
    DEFAULT_WEIGHTS,
    FORMULA_VERSION,
    RATING_BAND_CERTIFIED_MIN,
    RATING_BAND_PREVIEW_MIN,
    compute_fqs,
    compute_fqs_from_dict,
    promotion_eligibility,
    rating_label,
)


# --------------------------------------------------------------------------- #
# Rating + promotion bands.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fqs,expected",
    [
        (100.0, "excellent"),
        (85.0, "excellent"),
        (84.99, "good"),
        (70.0, "good"),
        (69.99, "fair"),
        (50.0, "fair"),
        (49.99, "poor"),
        (0.0, "poor"),
    ],
)
def test_rating_bands(fqs: float, expected: str) -> None:
    assert rating_label(fqs) == expected


@pytest.mark.parametrize(
    "fqs,expected",
    [
        (100.0, "certified"),
        (75.0, "certified"),
        (74.99, "preview"),
        (50.0, "preview"),
        (49.99, "connector_only"),
        (0.0, "connector_only"),
    ],
)
def test_promotion_eligibility_bands(fqs: float, expected: str) -> None:
    assert promotion_eligibility(fqs) == expected


def test_promotion_bands_are_monotonic() -> None:
    assert RATING_BAND_CERTIFIED_MIN > RATING_BAND_PREVIEW_MIN


# --------------------------------------------------------------------------- #
# compute_fqs — core computation.
# --------------------------------------------------------------------------- #


def _dqs(t=5, g=5, tech=5, rep=5, meth=5) -> DataQualityScore:
    return DataQualityScore(
        temporal=t,
        geographical=g,
        technological=tech,
        representativeness=rep,
        methodological=meth,
    )


def test_all_fives_is_100() -> None:
    result = compute_fqs(_dqs())
    assert result.composite_fqs == 100.0
    assert result.rating == "excellent"
    assert result.promotion_eligibility == "certified"
    assert len(result.components) == 5


def test_all_ones_is_20() -> None:
    # 1-5 DQS scale, linear mapping: 1 -> 20/100.
    result = compute_fqs(_dqs(t=1, g=1, tech=1, rep=1, meth=1))
    assert result.composite_fqs == 20.0
    assert result.rating == "poor"
    assert result.promotion_eligibility == "connector_only"


def test_mid_scores_produce_middle_composite() -> None:
    # All threes -> 60/100 composite.
    result = compute_fqs(_dqs(t=3, g=3, tech=3, rep=3, meth=3))
    assert result.composite_fqs == 60.0
    assert result.rating == "fair"
    assert result.promotion_eligibility == "preview"


def test_weighted_average_respects_weights() -> None:
    # Temporal=1 (weight 0.25), everything else 5 (weight 0.75).
    # Expected: 0.25*20 + 0.75*100 = 5 + 75 = 80.
    result = compute_fqs(_dqs(t=1, g=5, tech=5, rep=5, meth=5))
    assert result.composite_fqs == pytest.approx(80.0)
    assert result.rating == "good"


def test_component_scores_expose_both_names_and_both_scales() -> None:
    result = compute_fqs(_dqs(t=4, g=3, tech=2, rep=5, meth=1))
    by_name = {c.name: c for c in result.components}
    # Internal DQS names (GHG Protocol aligned).
    assert set(by_name) == {
        "temporal",
        "geographical",
        "technological",
        "representativeness",
        "methodological",
    }
    # CTO-spec aliases.
    for c in result.components:
        assert c.cto_alias == CTO_SPEC_ALIASES[c.name]
    # Scale conversions.
    assert by_name["temporal"].score_5 == 4.0
    assert by_name["temporal"].score_100 == 80.0
    assert by_name["methodological"].score_5 == 1.0
    assert by_name["methodological"].score_100 == 20.0


def test_formula_version_stable() -> None:
    result = compute_fqs(_dqs())
    assert result.formula_version == FORMULA_VERSION
    assert result.weights == DEFAULT_WEIGHTS


# --------------------------------------------------------------------------- #
# Custom weights + validation.
# --------------------------------------------------------------------------- #


def test_custom_weights_sum_validated() -> None:
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        compute_fqs(
            _dqs(),
            weights={
                "temporal": 0.5,
                "geographical": 0.5,
                "technological": 0.5,
                "representativeness": 0.0,
                "methodological": 0.0,
            },
        )


def test_custom_weights_used_in_computation() -> None:
    # All-temporal weighting: composite should equal temporal score_100.
    result = compute_fqs(
        _dqs(t=3, g=5, tech=5, rep=5, meth=5),
        weights={
            "temporal": 1.0,
            "geographical": 0.0,
            "technological": 0.0,
            "representativeness": 0.0,
            "methodological": 0.0,
        },
    )
    assert result.composite_fqs == 60.0


def test_missing_dimension_raises() -> None:
    class Broken:
        temporal = 3
        geographical = 3
        technological = 3
        representativeness = 3
        # methodological missing on purpose

    with pytest.raises(ValueError, match="missing required dimension 'methodological'"):
        compute_fqs(Broken())


# --------------------------------------------------------------------------- #
# Dict-input convenience.
# --------------------------------------------------------------------------- #


def test_from_dict_matches_from_dataclass() -> None:
    dqs = _dqs(t=4, g=4, tech=3, rep=2, meth=5)
    expected = compute_fqs(dqs)
    actual = compute_fqs_from_dict(
        {
            "temporal": 4,
            "geographical": 4,
            "technological": 3,
            "representativeness": 2,
            "methodological": 5,
        }
    )
    assert actual.composite_fqs == expected.composite_fqs
    assert actual.rating == expected.rating
    assert actual.promotion_eligibility == expected.promotion_eligibility


# --------------------------------------------------------------------------- #
# Serialization shape — this is the SDK/UI contract; if it changes the
# frontend breaks.
# --------------------------------------------------------------------------- #


def test_to_dict_contract() -> None:
    result = compute_fqs(_dqs(t=5, g=4, tech=3, rep=4, meth=5))
    d = result.to_dict()
    assert set(d) == {
        "composite_fqs",
        "rating",
        "promotion_eligibility",
        "components",
        "formula_version",
        "weights",
    }
    assert len(d["components"]) == 5
    for c in d["components"]:
        assert set(c) == {"name", "cto_alias", "score_5", "score_100"}
