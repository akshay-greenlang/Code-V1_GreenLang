from pathlib import Path


def _read_policy(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_v2_authz_policy_semantics() -> None:
    body = _read_policy("greenlang/governance/policy/bundles/v2_authz.rego")
    assert "package greenlang.v2.authz" in body
    assert "default allow = false" in body
    assert "input.authz.approved == true" in body
    assert 'msg := "authz approval is required"' in body


def test_v2_pack_trust_policy_semantics() -> None:
    body = _read_policy("greenlang/governance/policy/bundles/v2_pack_trust.rego")
    assert "package greenlang.v2.pack_trust" in body
    assert "tier == \"supported\" or tier == \"regulated-critical\"" in body
    assert "not input.security.signed" in body
    assert "count(input.security.signatures) == 0" in body


def test_v2_egress_policy_semantics() -> None:
    body = _read_policy("greenlang/governance/policy/bundles/v2_egress_controls.rego")
    assert "package greenlang.v2.egress_controls" in body
    assert "input.workflow_tier == \"regulated-critical\"" in body
    assert "not input.egress.allowlist" in body
    assert "destination_allowed(dest, allowlist)" in body


def test_v2_data_controls_policy_semantics() -> None:
    body = _read_policy("greenlang/governance/policy/bundles/v2_data_controls.rego")
    assert "package greenlang.v2.data_controls" in body
    assert "default allow = false" in body
    assert "input.data_classification != \"\"" in body
    assert "input.retention_days > 0" in body
    assert 'msg := "data residency must be declared"' in body


def test_v2_pack_tier_policy_semantics() -> None:
    body = _read_policy("greenlang/governance/policy/bundles/v2_pack_tier_policy.rego")
    assert "package greenlang.v2.pack_tier_policy" in body
    assert "tier == \"supported\" or tier == \"regulated-critical\"" in body
    assert "not input.metadata.owner_team" in body
    assert "not input.evidence.security_scan" in body
    assert "input.metadata.quality_tier == \"regulated-critical\"" in body
