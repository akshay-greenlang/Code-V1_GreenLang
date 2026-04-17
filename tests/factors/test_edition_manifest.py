# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.edition_manifest."""

from __future__ import annotations

from greenlang.factors.edition_manifest import EditionManifest, build_manifest_for_factors


def test_build_manifest_deterministic(emission_db):
    factors = list(emission_db.factors.values())
    m1 = build_manifest_for_factors("test-edition", "stable", factors, changelog=["line a"])
    m2 = build_manifest_for_factors("test-edition", "stable", factors, changelog=["line a"])
    assert m1.manifest_fingerprint() == m2.manifest_fingerprint()


def test_manifest_fingerprint_changes_with_factors(emission_db):
    factors = list(emission_db.factors.values())
    m1 = build_manifest_for_factors("ed1", "stable", factors)
    m2 = build_manifest_for_factors("ed1", "stable", factors[:5])
    assert m1.manifest_fingerprint() != m2.manifest_fingerprint()


def test_manifest_fingerprint_changes_with_changelog(emission_db):
    factors = list(emission_db.factors.values())
    m1 = build_manifest_for_factors("ed1", "stable", factors, changelog=["v1"])
    m2 = build_manifest_for_factors("ed1", "stable", factors, changelog=["v2"])
    assert m1.manifest_fingerprint() != m2.manifest_fingerprint()


def test_manifest_contains_edition_id(emission_db):
    factors = list(emission_db.factors.values())
    m = build_manifest_for_factors("my-edition", "stable", factors)
    assert m.edition_id == "my-edition"
    d = m.to_dict()
    assert d["edition_id"] == "my-edition"


def test_manifest_contains_factor_count(emission_db):
    factors = list(emission_db.factors.values())
    m = build_manifest_for_factors("ed", "stable", factors)
    assert m.factor_count == len(factors)


def test_manifest_channel(emission_db):
    factors = list(emission_db.factors.values())
    m_stable = build_manifest_for_factors("ed", "stable", factors)
    assert m_stable.status == "stable"
    m_nightly = build_manifest_for_factors("ed", "pending", factors)
    assert m_nightly.status == "pending"


def test_empty_factors_manifest():
    m = build_manifest_for_factors("empty-ed", "stable", [])
    assert m.factor_count == 0
    assert m.manifest_fingerprint()  # still produces a valid hash


def test_manifest_iso_timestamp():
    m = EditionManifest(edition_id="ts-test", status="stable")
    assert "T" in m.created_at
    assert ":" in m.created_at
