# -*- coding: utf-8 -*-
"""CLI contract tests for ``gl-factors resolve`` and ``gl-factors explain``.

Exercises :func:`greenlang.factors.cli.main` via argparse with ``capsys`` so
we don't need a HTTP server or a subprocess shell.  The tests rely on the
file-backed candidate source (built-in EmissionFactorDatabase), which
ships with US/EU/UK electricity + US fossil fuel factors — enough to
drive the 7-step cascade end-to-end.
"""

from __future__ import annotations

import json

import pytest

from greenlang.factors import cli


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _run(argv, capsys):
    """Run the CLI with ``argv`` and return (exit_code, parsed_json)."""
    exit_code = cli.main(argv)
    captured = capsys.readouterr()
    stdout = captured.out.strip()
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = None
    return exit_code, payload, stdout


# ----------------------------------------------------------------------------
# resolve
# ----------------------------------------------------------------------------


class TestResolveCommand:
    """``gl-factors resolve`` — mirrors POST /v1/resolve."""

    def test_resolve_returns_factor_with_explain_block(self, capsys):
        exit_code, payload, _ = _run(
            [
                "resolve",
                "--activity", "diesel combustion",
                "--method-profile", "corporate_scope1",
                "--country", "US",
                "--json",
            ],
            capsys,
        )
        assert exit_code == 0
        assert payload is not None

        # Core explain fields must be present.
        assert payload["chosen_factor_id"], "resolve must return a chosen_factor_id"
        assert payload["fallback_rank"] in {1, 2, 3, 4, 5, 6, 7}
        assert payload["step_label"]
        assert payload["why_chosen"]
        assert payload["method_profile"] == "corporate_scope1"
        assert "edition_id" in payload

        # The explain object mirrors the API payload.
        assert "explain" in payload
        explain = payload["explain"]
        assert "chosen" in explain
        assert "derivation" in explain
        assert "quality" in explain
        assert "emissions" in explain
        assert "alternates" in explain
        assert explain["derivation"]["fallback_rank"] == payload["fallback_rank"]

        # Alternates live at top level AND inside the explain block.
        assert isinstance(payload["alternates"], list)
        assert isinstance(explain["alternates"], list)

    def test_resolve_quantity_appends_co2e(self, capsys):
        """When --quantity is passed, payload exposes computed co2e."""
        exit_code, payload, _ = _run(
            [
                "resolve",
                "--activity", "diesel combustion",
                "--method-profile", "corporate_scope1",
                "--country", "US",
                "--quantity", "100",
                "--unit", "gallons",
                "--json",
            ],
            capsys,
        )
        assert exit_code == 0
        assert "co2e" in payload
        assert payload["co2e_unit"] == "kg"
        # 100 × per-unit co2e should equal co2e_total_kg × 100.
        per_unit = payload["gas_breakdown"]["co2e_total_kg"]
        assert payload["co2e"] == pytest.approx(per_unit * 100, rel=1e-6)

    def test_resolve_compact_mode_emits_single_line_json(self, capsys):
        exit_code, _payload, stdout = _run(
            [
                "resolve",
                "--activity", "diesel combustion",
                "--method-profile", "corporate_scope1",
                "--country", "US",
                "--compact",
            ],
            capsys,
        )
        assert exit_code == 0
        # Compact output is a single JSON line.
        assert "\n" not in stdout.strip()
        json.loads(stdout)  # must still be valid JSON

    def test_resolve_unknown_method_profile_exits_nonzero(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.main([
                "resolve",
                "--activity", "diesel",
                "--method-profile", "definitely_not_a_profile",
                "--json",
            ])
        # SystemExit with a string message is exit-code 1 when the
        # argument is a string (Python behaviour).
        assert exc_info.value.code != 0

    def test_resolve_no_match_emits_error_payload(self, capsys):
        """Activity that cannot pass any method-pack rule returns a
        structured error payload, not a stack trace."""
        exit_code, payload, _ = _run(
            [
                "resolve",
                "--activity", "purchased electricity",
                "--method-profile", "corporate_scope2_location_based",
                "--country", "IN",  # no IN factors in built-in DB
                "--json",
            ],
            capsys,
        )
        # Non-zero exit, structured error.
        assert exit_code != 0
        assert payload["error"] == "resolve_failed"
        assert "message" in payload


# ----------------------------------------------------------------------------
# explain
# ----------------------------------------------------------------------------


class TestExplainCommand:
    """``gl-factors explain <factor_id>`` — mirrors GET /v1/factors/{id}/explain."""

    def _pick_factor_id(self) -> str:
        from greenlang.data.emission_factor_database import EmissionFactorDatabase

        db = EmissionFactorDatabase(enable_cache=False)
        # Pick any factor whose family matches a method-pack selection rule.
        # Diesel scope-1 fits corporate_scope1's allowed families.
        for f in db.factors.values():
            if "diesel" in (getattr(f, "fuel_type", None) or "").lower():
                return f.factor_id
        # Fallback — first factor.
        return next(iter(db.factors.values())).factor_id

    def test_explain_returns_full_payload(self, capsys):
        factor_id = self._pick_factor_id()
        exit_code, payload, _ = _run(
            ["explain", factor_id, "--json"], capsys,
        )
        assert exit_code == 0
        assert payload is not None
        assert payload["chosen_factor_id"] == factor_id
        # Same explain shape as the API.
        assert "explain" in payload
        assert "chosen" in payload["explain"]
        assert "derivation" in payload["explain"]
        assert payload["explain"]["derivation"]["fallback_rank"] >= 1
        assert payload["explain"]["derivation"]["step_label"]
        assert "quality" in payload["explain"]
        assert "edition_id" in payload

    def test_explain_unknown_factor_returns_not_found(self, capsys):
        exit_code, payload, _ = _run(
            ["explain", "definitely-not-a-factor-id", "--json"], capsys,
        )
        assert exit_code == 1
        assert payload["error"] == "factor_not_found"
        assert payload["factor_id"] == "definitely-not-a-factor-id"

    def test_explain_compact_mode(self, capsys):
        factor_id = self._pick_factor_id()
        exit_code, _payload, stdout = _run(
            ["explain", factor_id, "--compact"], capsys,
        )
        assert exit_code == 0
        assert "\n" not in stdout.strip()
        json.loads(stdout)


# ----------------------------------------------------------------------------
# CLI surface smoke test
# ----------------------------------------------------------------------------


def test_cli_help_lists_resolve_and_explain(capsys):
    """The two CTO-critical commands must appear in --help."""
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "resolve" in out
    assert "explain" in out
