# -*- coding: utf-8 -*-
"""
WS11-T1: tests for ``greenlang.factors.release_profile``.

Verifies:
  * ``current_profile()`` honors GL_FACTORS_RELEASE_PROFILE.
  * Feature ordering: each feature is OFF below its min profile and stays
    ON at every higher profile (including DEV).
  * Specific gates: resolve_endpoint=BETA, graphql=RC, billing/oem=GA.
  * Default fallback: production with no env var -> alpha; otherwise -> dev.
  * ``feature_enabled`` for unknown feature names returns False.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def rp(monkeypatch):
    """Yield a freshly-imported release_profile module with env scrubbed.

    ``current_profile()`` reads env vars at call time, so we just clear the
    relevant vars and return the module. Tests then ``monkeypatch.setenv``
    on demand.
    """
    for var in ("GL_FACTORS_RELEASE_PROFILE", "GL_ENV", "APP_ENV", "ENVIRONMENT"):
        monkeypatch.delenv(var, raising=False)
    module = importlib.import_module("greenlang.factors.release_profile")
    return module


# ---------------------------------------------------------------------------
# current_profile() resolution
# ---------------------------------------------------------------------------


def test_current_profile_honors_env_var_alpha(monkeypatch, rp):
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "alpha-v0.1")
    assert rp.current_profile() is rp.ReleaseProfile.ALPHA_V0_1
    assert rp.is_alpha() is True


def test_current_profile_honors_env_var_beta(monkeypatch, rp):
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "beta-v0.5")
    assert rp.current_profile() is rp.ReleaseProfile.BETA_V0_5
    assert rp.is_alpha() is False


def test_current_profile_honors_env_var_rc(monkeypatch, rp):
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "rc-v0.9")
    assert rp.current_profile() is rp.ReleaseProfile.RC_V0_9


def test_current_profile_honors_env_var_ga(monkeypatch, rp):
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "ga-v1.0")
    assert rp.current_profile() is rp.ReleaseProfile.GA_V1_0


def test_current_profile_honors_env_var_dev(monkeypatch, rp):
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "dev")
    assert rp.current_profile() is rp.ReleaseProfile.DEV


def test_current_profile_default_is_dev_when_not_production(rp):
    # No env vars set at all -> dev
    assert rp.current_profile() is rp.ReleaseProfile.DEV


def test_current_profile_default_is_alpha_in_production(monkeypatch, rp):
    monkeypatch.setenv("GL_ENV", "production")
    assert rp.current_profile() is rp.ReleaseProfile.ALPHA_V0_1


def test_current_profile_default_is_alpha_for_app_env_prod(monkeypatch, rp):
    monkeypatch.setenv("APP_ENV", "prod")
    assert rp.current_profile() is rp.ReleaseProfile.ALPHA_V0_1


def test_current_profile_unknown_value_falls_back(monkeypatch, rp):
    """Bogus env values must not crash; we fall back to defaults."""
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "nonsense-v999")
    # No production hint -> dev fallback.
    assert rp.current_profile() is rp.ReleaseProfile.DEV


def test_current_profile_alias_resolution(monkeypatch, rp):
    """Short aliases (``alpha``, ``beta``, ...) are normalized."""
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "alpha")
    assert rp.current_profile() is rp.ReleaseProfile.ALPHA_V0_1
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "BETA")
    assert rp.current_profile() is rp.ReleaseProfile.BETA_V0_5


# ---------------------------------------------------------------------------
# feature_enabled — per-feature gates
# ---------------------------------------------------------------------------


def test_resolve_endpoint_off_in_alpha(monkeypatch, rp):
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "alpha-v0.1")
    assert rp.feature_enabled("resolve_endpoint") is False
    assert rp.feature_enabled("explain_endpoint") is False
    assert rp.feature_enabled("batch_endpoint") is False
    assert rp.feature_enabled("coverage_endpoint") is False
    assert rp.feature_enabled("fqs_endpoint") is False
    assert rp.feature_enabled("edition_endpoint") is False
    assert rp.feature_enabled("signed_receipts") is False
    assert rp.feature_enabled("admin_console") is False
    assert rp.feature_enabled("graphql") is False
    assert rp.feature_enabled("billing") is False
    assert rp.feature_enabled("oem") is False
    assert rp.feature_enabled("ml_resolve") is False
    assert rp.feature_enabled("commercial_packs") is False
    assert rp.feature_enabled("real_time_grid") is False
    assert rp.feature_enabled("sql_over_http") is False
    assert rp.feature_enabled("ts_sdk") is False
    assert rp.feature_enabled("cli_extended") is False


def test_resolve_endpoint_on_in_beta(monkeypatch, rp):
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "beta-v0.5")
    assert rp.feature_enabled("resolve_endpoint") is True
    assert rp.feature_enabled("explain_endpoint") is True
    assert rp.feature_enabled("batch_endpoint") is True
    assert rp.feature_enabled("coverage_endpoint") is True
    assert rp.feature_enabled("fqs_endpoint") is True
    assert rp.feature_enabled("edition_endpoint") is True
    assert rp.feature_enabled("signed_receipts") is True
    assert rp.feature_enabled("admin_console") is True
    assert rp.feature_enabled("ts_sdk") is True
    assert rp.feature_enabled("cli_extended") is True
    # Not yet at higher tiers:
    assert rp.feature_enabled("graphql") is False
    assert rp.feature_enabled("ml_resolve") is False
    assert rp.feature_enabled("billing") is False
    assert rp.feature_enabled("oem") is False


def test_graphql_only_at_rc_or_higher(monkeypatch, rp):
    for profile, expected in (
        ("alpha-v0.1", False),
        ("beta-v0.5", False),
        ("rc-v0.9", True),
        ("ga-v1.0", True),
        ("dev", True),
    ):
        monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", profile)
        assert rp.feature_enabled("graphql") is expected, profile
        assert rp.feature_enabled("ml_resolve") is expected, profile


def test_billing_oem_only_at_ga(monkeypatch, rp):
    for profile, expected in (
        ("alpha-v0.1", False),
        ("beta-v0.5", False),
        ("rc-v0.9", False),
        ("ga-v1.0", True),
        ("dev", True),
    ):
        monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", profile)
        assert rp.feature_enabled("billing") is expected, profile
        assert rp.feature_enabled("oem") is expected, profile
        assert rp.feature_enabled("commercial_packs") is expected, profile
        assert rp.feature_enabled("real_time_grid") is expected, profile
        assert rp.feature_enabled("sql_over_http") is expected, profile


def test_unknown_feature_returns_false(monkeypatch, rp):
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "dev")
    assert rp.feature_enabled("nonexistent_feature_zzz") is False


# ---------------------------------------------------------------------------
# Ordering invariant: every feature must be ON at and above its min_profile.
# ---------------------------------------------------------------------------


def test_ordering_invariant_every_feature_reaches_its_min_profile(monkeypatch, rp):
    """For every feature F with min_profile P, F must be enabled at P
    and at every profile ranked higher than P (including DEV)."""
    profile_order = [
        rp.ReleaseProfile.ALPHA_V0_1,
        rp.ReleaseProfile.BETA_V0_5,
        rp.ReleaseProfile.RC_V0_9,
        rp.ReleaseProfile.GA_V1_0,
        rp.ReleaseProfile.DEV,  # dev = "all on"
    ]

    for feature, spec in rp.FEATURES.items():
        min_profile = spec["min_profile"]
        min_idx = profile_order.index(min_profile)

        # Below min_profile: must be OFF.
        for profile in profile_order[:min_idx]:
            if profile is rp.ReleaseProfile.DEV:
                continue  # dev sits at end anyway
            monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", profile.value)
            assert rp.feature_enabled(feature) is False, (
                f"Feature {feature!r} should be OFF at {profile.value}"
                f" (min_profile={min_profile.value})"
            )

        # At and above min_profile: must be ON.
        for profile in profile_order[min_idx:]:
            monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", profile.value)
            assert rp.feature_enabled(feature) is True, (
                f"Feature {feature!r} should be ON at {profile.value}"
                f" (min_profile={min_profile.value})"
            )

        # Plus: dev profile turns on everything.
        monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "dev")
        assert rp.feature_enabled(feature) is True, (
            f"Feature {feature!r} must be ON in dev profile"
        )


def test_features_table_has_expected_keys(rp):
    """Lock in the FEATURES table keys per the WS11-T1 spec."""
    expected = {
        "resolve_endpoint", "explain_endpoint", "batch_endpoint",
        "edition_endpoint", "coverage_endpoint", "fqs_endpoint",
        "signed_receipts", "graphql", "sql_over_http", "billing",
        "oem", "admin_console", "ts_sdk", "cli_extended",
        "ml_resolve", "commercial_packs", "real_time_grid",
        # method_packs is the additional alias used by factors_app.py
        "method_packs",
    }
    assert expected.issubset(set(rp.FEATURES.keys()))
