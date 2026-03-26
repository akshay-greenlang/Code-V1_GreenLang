from greenlang.v1.conformance import release_gate_checks


def test_release_gate_checks_all_pass() -> None:
    checks = release_gate_checks()
    failures = [check for check in checks if not check.ok]
    assert not failures, [f"{item.name}: {item.details}" for item in failures]

