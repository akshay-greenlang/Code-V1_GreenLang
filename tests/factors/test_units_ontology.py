# -*- coding: utf-8 -*-
"""Unit ontology conversion helpers (S2)."""

from __future__ import annotations

import pytest

from greenlang.factors.ontology.units import convert_energy, convert_energy_to_kwh, is_known_activity_unit


def test_is_known_activity_unit():
    assert is_known_activity_unit("kWh")
    assert not is_known_activity_unit("unknown_unit_xyz")


def test_convert_energy_round_trip():
    mwh = convert_energy(1.0, "mwh", "kwh")
    assert mwh == pytest.approx(1000.0)
    back = convert_energy(mwh, "kwh", "mwh")
    assert back == pytest.approx(1.0)


def test_convert_energy_to_kwh():
    assert convert_energy_to_kwh(2.0, "mwh") == pytest.approx(2000.0)


def test_unknown_unit_raises():
    with pytest.raises(ValueError):
        convert_energy_to_kwh(1.0, "not_a_unit")
