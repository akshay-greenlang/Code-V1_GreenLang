from types import SimpleNamespace

from cbam_pack.audit.bundle import AuditBundleGenerator


def _fake_config() -> SimpleNamespace:
    quarter = SimpleNamespace(value="Q1")
    reporting_period = SimpleNamespace(quarter=quarter, year=2026)
    declarant = SimpleNamespace(eori_number="EU123456789")
    return SimpleNamespace(declarant=declarant, reporting_period=reporting_period)


def _fake_calc_result() -> SimpleNamespace:
    return SimpleNamespace(
        statistics={"total_lines": 2, "total_emissions_tco2e": 42.5}
    )


def test_run_context_is_deterministic_for_same_inputs() -> None:
    gen_one = AuditBundleGenerator(factor_library_version="2026.1")
    gen_two = AuditBundleGenerator(factor_library_version="2026.1")

    evidence = {
        "imports.csv": {"sha256": "abc123"},
        "config.yaml": {"sha256": "def456"},
    }

    gen_one._initialize_run_context(_fake_calc_result(), _fake_config(), evidence)
    gen_two._initialize_run_context(_fake_calc_result(), _fake_config(), evidence)

    assert gen_one._run_id == gen_two._run_id
    assert gen_one._generated_at == gen_two._generated_at


def test_stable_timestamp_changes_with_seed() -> None:
    gen = AuditBundleGenerator()
    first = gen._stable_timestamp("a" * 64)
    second = gen._stable_timestamp("b" * 64)

    assert first.endswith("Z")
    assert second.endswith("Z")
    assert first != second
