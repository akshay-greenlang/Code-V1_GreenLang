# -*- coding: utf-8 -*-
"""Tests for license compliance scanner (F025)."""

from __future__ import annotations

import pytest

from greenlang.factors.quality.license_scanner import (
    LicenseIssue,
    LicenseScanReport,
    LicenseSeverity,
    scan_license_compliance,
)
from greenlang.factors.source_registry import SourceRegistryEntry


def _entry(
    source_id="epa",
    connector_only=False,
    redistribution_allowed=True,
    attribution_required=False,
    citation_text="EPA citation",
    approval_required_for_certified=False,
    legal_signoff_artifact=None,
    **kw,
):
    return SourceRegistryEntry(
        source_id=source_id,
        display_name=source_id.upper(),
        connector_only=connector_only,
        license_class="public",
        redistribution_allowed=redistribution_allowed,
        derivative_works_allowed=True,
        commercial_use_allowed=True,
        attribution_required=attribution_required,
        citation_text=citation_text,
        cadence="annual",
        watch_mechanism="http",
        watch_url=None,
        watch_file_type=None,
        approval_required_for_certified=approval_required_for_certified,
        legal_signoff_artifact=legal_signoff_artifact,
        legal_signoff_version=None,
    )


def _factor(fid="EF:EPA:ng:US:2024:v1", source_id="epa", status="certified",
            redist=True, attribution_required=False, citation_text="", **kw):
    d = {
        "factor_id": fid,
        "source_id": source_id,
        "factor_status": status,
        "license_info": {
            "redistribution_allowed": redist,
            "attribution_required": attribution_required,
            "citation_text": citation_text,
        },
    }
    d.update(kw)
    return d


# ---- LicenseIssue ----

def test_license_issue_to_dict():
    issue = LicenseIssue(
        factor_id="EF:X:1", source_id="epa",
        severity="error", rule_id="L01", message="test",
    )
    d = issue.to_dict()
    assert d["rule_id"] == "L01"
    assert d["severity"] == "error"


# ---- LicenseScanReport ----

def test_report_compliant():
    r = LicenseScanReport(edition_id="test", total_factors=5)
    assert r.compliant
    assert not r.release_blocked


def test_report_not_compliant_on_error():
    r = LicenseScanReport(edition_id="test", total_factors=5, errors=1)
    assert not r.compliant


def test_report_blocked_on_block():
    r = LicenseScanReport(edition_id="test", total_factors=5, blocks=1)
    assert r.release_blocked


def test_report_to_dict():
    r = LicenseScanReport(edition_id="test", total_factors=10, total_issues=2, errors=1, warnings=1)
    d = r.to_dict()
    assert d["edition_id"] == "test"
    assert d["total_issues"] == 2
    assert d["compliant"] is False


# ---- scan_license_compliance ----

def test_clean_scan():
    """All factors compliant -> no issues."""
    reg = {"epa": _entry("epa", redistribution_allowed=True)}
    factors = [_factor("EF:EPA:ng:US:2024:v1", source_id="epa", redist=True)]
    report = scan_license_compliance(factors, edition_id="test", registry=reg)
    assert report.total_factors == 1
    assert report.compliant
    assert report.total_issues == 0


def test_l01_connector_only_redistribution():
    """L01: connector_only source + factor has redistribution_allowed."""
    reg = {"iea": _entry("iea", connector_only=True, redistribution_allowed=False)}
    factors = [_factor("EF:IEA:elec:DE:2024:v1", source_id="iea", redist=True)]
    report = scan_license_compliance(factors, edition_id="test", registry=reg)
    assert report.errors >= 1
    assert any(i.rule_id == "L01" for i in report.issues)


def test_l02_missing_citation():
    """L02: attribution_required but no citation text."""
    reg = {"defra": _entry("defra", attribution_required=True, citation_text="")}
    factors = [_factor("EF:DEFRA:ng:UK:2024:v1", source_id="defra", citation_text="")]
    report = scan_license_compliance(factors, edition_id="test", registry=reg)
    assert report.warnings >= 1
    assert any(i.rule_id == "L02" for i in report.issues)


def test_l02_citation_present_no_warning():
    """L02: citation_text present -> no warning."""
    reg = {"defra": _entry("defra", attribution_required=True, citation_text="DEFRA 2024")}
    factors = [_factor("EF:DEFRA:ng:UK:2024:v1", source_id="defra")]
    report = scan_license_compliance(factors, edition_id="test", registry=reg)
    assert not any(i.rule_id == "L02" for i in report.issues)


def test_l03_certified_without_legal_signoff():
    """L03: certified factor but source requires legal signoff (missing)."""
    reg = {"ecoinvent": _entry("ecoinvent", approval_required_for_certified=True, legal_signoff_artifact=None)}
    factors = [_factor("EF:ECOINVENT:cement:CH:2024:v1", source_id="ecoinvent", status="certified")]
    report = scan_license_compliance(factors, edition_id="test", registry=reg)
    assert report.blocks >= 1
    assert report.release_blocked
    assert any(i.rule_id == "L03" for i in report.issues)


def test_l03_certified_with_legal_signoff_ok():
    """L03: legal signoff present -> no block."""
    reg = {"ecoinvent": _entry("ecoinvent", approval_required_for_certified=True, legal_signoff_artifact="signoff_2024.pdf")}
    factors = [_factor("EF:ECOINVENT:cement:CH:2024:v1", source_id="ecoinvent", status="certified")]
    report = scan_license_compliance(factors, edition_id="test", registry=reg)
    assert not any(i.rule_id == "L03" for i in report.issues)


def test_l03_preview_not_blocked():
    """L03: preview status -> no block even without signoff."""
    reg = {"ecoinvent": _entry("ecoinvent", approval_required_for_certified=True, legal_signoff_artifact=None)}
    factors = [_factor("EF:ECOINVENT:cement:CH:2024:v1", source_id="ecoinvent", status="preview")]
    report = scan_license_compliance(factors, edition_id="test", registry=reg)
    assert not any(i.rule_id == "L03" for i in report.issues)


def test_l04_deprecated_redistribution():
    """L04: deprecated factor with redistribution_allowed."""
    reg = {"epa": _entry("epa")}
    factors = [_factor("EF:EPA:old:US:2020:v1", source_id="epa", status="deprecated", redist=True)]
    report = scan_license_compliance(factors, edition_id="test", registry=reg)
    assert report.warnings >= 1
    assert any(i.rule_id == "L04" for i in report.issues)


def test_l05_unknown_source():
    """L05: source not in registry."""
    reg = {"epa": _entry("epa")}
    factors = [_factor("EF:MYSTERY:ng:US:2024:v1", source_id="mystery")]
    report = scan_license_compliance(factors, edition_id="test", registry=reg)
    assert any(i.rule_id == "L05" for i in report.issues)


def test_l06_factor_redist_registry_no():
    """L06: factor says redistribution but registry says no."""
    reg = {"restricted": _entry("restricted", redistribution_allowed=False, connector_only=False)}
    factors = [_factor("EF:RESTRICTED:ng:US:2024:v1", source_id="restricted", redist=True)]
    report = scan_license_compliance(factors, edition_id="test", registry=reg)
    assert report.errors >= 1
    assert any(i.rule_id == "L06" for i in report.issues)


def test_multiple_issues():
    """Multiple factors with different issues."""
    reg = {
        "epa": _entry("epa"),
        "iea": _entry("iea", connector_only=True, redistribution_allowed=False),
    }
    factors = [
        _factor("EF:EPA:ng:US:2024:v1", source_id="epa"),  # clean
        _factor("EF:IEA:elec:DE:2024:v1", source_id="iea", redist=True),  # L01
    ]
    report = scan_license_compliance(factors, edition_id="test", registry=reg)
    assert report.total_factors == 2
    assert report.total_issues >= 1


def test_empty_factors():
    report = scan_license_compliance([], edition_id="test", registry={})
    assert report.total_factors == 0
    assert report.compliant


def test_source_id_from_factor_id():
    """Source ID extracted from factor_id when source_id field missing."""
    reg = {"epa": _entry("epa")}
    factors = [{"factor_id": "EF:EPA:ng:US:2024:v1", "factor_status": "certified",
                "license_info": {"redistribution_allowed": True}}]
    report = scan_license_compliance(factors, edition_id="test", registry=reg)
    # Should find the EPA entry
    assert not any(i.rule_id == "L05" for i in report.issues)


def test_severity_counts():
    reg = {
        "iea": _entry("iea", connector_only=True, redistribution_allowed=False),
        "eco": _entry("eco", approval_required_for_certified=True, legal_signoff_artifact=None),
        "defra": _entry("defra", attribution_required=True, citation_text=""),
    }
    factors = [
        _factor("EF:IEA:1", source_id="iea", redist=True),  # L01 error
        _factor("EF:ECO:1", source_id="eco", status="certified"),  # L03 block
        _factor("EF:DEFRA:1", source_id="defra", citation_text=""),  # L02 warning
    ]
    report = scan_license_compliance(factors, edition_id="test", registry=reg)
    assert report.errors >= 1
    assert report.blocks >= 1
    assert report.warnings >= 1
    assert not report.compliant
    assert report.release_blocked
