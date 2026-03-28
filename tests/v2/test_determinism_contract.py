from pathlib import Path

from greenlang.v2.standards import compare_artifact_hashes


def _write(root: Path, rel: str, content: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_determinism_contract_detects_identical_runs(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write(run_a, "audit/run_manifest.json", '{"status":"ok"}')
    _write(run_a, "audit/checksums.json", '{"x":"1"}')
    _write(run_a, "domain/output.json", '{"value":42}')

    _write(run_b, "audit/run_manifest.json", '{"status":"ok"}')
    _write(run_b, "audit/checksums.json", '{"x":"1"}')
    _write(run_b, "domain/output.json", '{"value":42}')

    result = compare_artifact_hashes(run_a, run_b)
    assert result.same_fileset
    assert result.diff_count == 0


def test_determinism_contract_detects_hash_diff(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write(run_a, "audit/run_manifest.json", '{"status":"ok"}')
    _write(run_a, "audit/checksums.json", '{"x":"1"}')
    _write(run_a, "domain/output.json", '{"value":42}')

    _write(run_b, "audit/run_manifest.json", '{"status":"ok"}')
    _write(run_b, "audit/checksums.json", '{"x":"1"}')
    _write(run_b, "domain/output.json", '{"value":43}')

    result = compare_artifact_hashes(run_a, run_b)
    assert result.same_fileset
    assert result.diff_count == 1
    assert result.diffs == ["domain/output.json"]

