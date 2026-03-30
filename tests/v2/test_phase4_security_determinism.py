from pathlib import Path

from greenlang.v2.conformance import determinism_contract_checks, policy_bundle_checks


def test_phase4_policy_bundles_present() -> None:
    checks = policy_bundle_checks()
    assert checks, "expected policy bundle checks"
    failed = [check for check in checks if not check.ok]
    assert not failed, [f"{check.name}: {check.details}" for check in failed]


def test_phase4_determinism_contract_for_regulated_workflows(tmp_path: Path) -> None:
    checks = determinism_contract_checks(output_root=tmp_path / "determinism_runs")
    assert checks, "expected determinism checks"
    failed = [check for check in checks if not check.ok]
    assert not failed, [f"{check.name}: {check.details}" for check in failed]
