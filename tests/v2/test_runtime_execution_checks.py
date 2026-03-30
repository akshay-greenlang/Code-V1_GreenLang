from greenlang.v2.conformance import runtime_execution_checks


def test_runtime_execution_checks_regulated_profiles() -> None:
    checks = runtime_execution_checks()
    assert checks, "expected runtime execution checks"
    failed = [check for check in checks if not check.ok]
    assert not failed, [f"{check.name}: {check.details}" for check in failed]
