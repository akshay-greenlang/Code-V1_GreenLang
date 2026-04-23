# -*- coding: utf-8 -*-
"""
N1 — Certified factors MUST carry gas-level vectors (no CO2e-only).

This test file enforces CTO non-negotiable #1: a Certified-status
emission factor can never ship as a bare ``co2e_total`` number. For
combustion, electricity and refrigerant families we require the
relevant per-gas vectors (CO2 + CH4 + N2O for combustion/electricity;
at least one fluorinated gas for refrigerants).

Why it matters: a co2e-only record erases the ability to re-apply AR6
vs AR5 GWP tables, to distinguish biogenic from fossil CO2, and to
attribute methane risk. Regulators (CBAM, SEC, ISSB, CSRD) already
demand gas-level disclosure. If this gate is red, we have fake
reproducibility.

Run standalone::

    pytest tests/factors/gates/test_n1_no_co2e_only.py -v
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable, List

import pytest

# We intentionally import the production validator by path rather than a
# convenience wrapper so a future refactor does not hide the gate behind
# an indirection layer.
try:
    from greenlang.factors.quality.validators import (  # type: ignore
        validate_gas_vectors,
    )

    _HAS_PRODUCTION_VALIDATOR = True
except Exception:  # pragma: no cover — validator module exists, symbol may not
    _HAS_PRODUCTION_VALIDATOR = False


# Families that require non-zero per-gas decomposition when Certified.
# Mapped to the gases a source must report at minimum.
GAS_REQUIREMENTS = {
    "combustion": ("CO2", "CH4", "N2O"),
    "emissions": ("CO2", "CH4", "N2O"),          # legacy family alias
    "electricity": ("CO2", "CH4", "N2O"),
    "grid_intensity": ("CO2", "CH4", "N2O"),
    "refrigerant": ("HFCs", "PFCs", "SF6", "NF3"),
    "refrigerant_gwp": ("HFCs", "PFCs", "SF6", "NF3"),
}


def _has_any_gas(vectors: SimpleNamespace, gases: Iterable[str]) -> bool:
    for gas in gases:
        val = getattr(vectors, gas, None)
        if val is None:
            continue
        try:
            if float(val) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _reference_validator(record: SimpleNamespace) -> List[str]:
    """Lightweight spec-aligned validator used as the N1 oracle.

    Returns a list of violation messages (empty list = valid).
    Production code should provide the equivalent in
    ``greenlang/factors/quality/validators.py``.
    """
    failures: List[str] = []

    status = (getattr(record, "factor_status", "") or "").lower()
    if status != "certified":
        # N1 only applies to Certified-status rows.
        return failures

    family = (
        getattr(record, "factor_family", None)
        or getattr(record, "family", None)
        or ""
    )
    family = str(family).lower()

    vectors = getattr(record, "vectors", None)
    gwp = getattr(record, "gwp_100yr", None)
    co2e_total = float(getattr(gwp, "co2e_total", 0.0) or 0.0)

    if vectors is None:
        failures.append(
            f"N1 violation: {record.factor_id} Certified but has no `vectors` block."
        )
        return failures

    # Every Certified record must carry CO2 (even for refrigerants where
    # CO2 can be zero, the CO2 field itself must be present and numeric).
    co2_val = getattr(vectors, "CO2", None)
    if co2_val is None:
        failures.append(
            f"N1 violation: {record.factor_id} missing vectors.CO2 (required on every Certified record)."
        )

    # Family-specific: require at least one non-zero member of the required gas set.
    required = GAS_REQUIREMENTS.get(family)
    if required and not _has_any_gas(vectors, required):
        if co2e_total > 0.0:
            failures.append(
                f"N1 violation: {record.factor_id} is Certified with co2e_total={co2e_total} "
                f"but ALL per-gas vectors are zero. Family {family!r} requires at least one "
                f"of {required} to be populated. A co2e-only Certified row is forbidden."
            )

    return failures


def _run_validator(record):
    """Prefer the production validator when present; fall back to the reference."""
    if _HAS_PRODUCTION_VALIDATOR:
        try:
            return list(validate_gas_vectors(record))  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover
            return [f"N1 production validator raised: {exc!r}"]
    return _reference_validator(record)


# ---------------------------------------------------------------------------
# Happy path — every family fixture must PASS.
# ---------------------------------------------------------------------------


class TestN1ValidFactorsPass:
    """Every Certified family record with real gas vectors must validate."""

    def test_all_families_in_fixture_are_valid(self, certified_factor_records):
        failures = {}
        for family, record in certified_factor_records.items():
            viol = _run_validator(record)
            if viol:
                failures[family] = viol
        assert not failures, (
            "N1 (no CO2e-only): the following Certified factors failed gas-vector "
            "validation even though they were built with real per-gas values — "
            f"fix the fixture or the validator:\n{failures}"
        )


# ---------------------------------------------------------------------------
# The central gate — co2e-only Certified row must FAIL.
# ---------------------------------------------------------------------------


class TestN1Co2eOnlyIsRejected:
    """A Certified factor whose only numeric content is co2e_total must fail."""

    def test_co2e_only_record_is_rejected(self, co2e_only_record):
        violations = _run_validator(co2e_only_record)
        assert violations, (
            "N1 (no CO2e-only): a Certified factor with zero per-gas vectors "
            "but a non-zero co2e_total must FAIL validation. The validator "
            "accepted it — this means production will ship bare-CO2e records "
            "and we lose the ability to re-apply GWP tables (AR5 / AR6 / AR6-20yr), "
            "separate biogenic CO2, and attribute methane. "
            f"Record factor_id={co2e_only_record.factor_id}"
        )

    @pytest.mark.parametrize(
        "family,required_gases",
        [
            ("combustion", ("CO2", "CH4", "N2O")),
            ("electricity", ("CO2", "CH4", "N2O")),
            ("refrigerant", ("HFCs", "PFCs", "SF6", "NF3")),
        ],
    )
    def test_family_specific_gas_requirement(
        self, make_record, make_vectors, family, required_gases
    ):
        """For each gas-required family, a scrubbed Certified row must fail."""
        scrubbed = make_record(
            factor_id=f"EF:TEST:{family}:scrubbed:v1",
            family=family,
            vectors=make_vectors(),  # all zeros
            co2e_total=7.7,
        )
        violations = _run_validator(scrubbed)
        assert violations, (
            f"N1 violation: family={family!r} must require one of {required_gases} "
            "non-zero when Certified. The validator accepted an all-zero-gas record "
            f"(factor_id={scrubbed.factor_id})."
        )


# ---------------------------------------------------------------------------
# Preview / connector_only records are exempt — N1 is only for Certified.
# ---------------------------------------------------------------------------


class TestN1OnlyAppliesToCertified:
    """Preview + connector_only records are allowed to be co2e-only."""

    @pytest.mark.parametrize("status", ["preview", "connector_only", "deprecated"])
    def test_non_certified_skips_gate(self, make_record, make_vectors, status):
        rec = make_record(
            factor_id=f"EF:TEST:preview:{status}:v1",
            family="combustion",
            vectors=make_vectors(),
            co2e_total=1.0,
            factor_status=status,
        )
        violations = _run_validator(rec)
        assert not violations, (
            f"N1 should only gate factor_status='certified'. Status {status!r} "
            f"was incorrectly flagged: {violations}"
        )
