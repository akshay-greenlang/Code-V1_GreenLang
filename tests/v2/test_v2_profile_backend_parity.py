from pathlib import Path

from greenlang.v2.backends import V2_BLOCKED_EXIT_CODE, run_v2_profile_backend


def _smoke_input(profile: str) -> Path:
    mapping = {
        "eudr": Path("applications/GL-EUDR-APP/v2/smoke_input.json"),
        "ghg": Path("applications/GL-GHG-APP/v2/smoke_input.json"),
        "iso14064": Path("applications/GL-ISO14064-APP/v2/smoke_input.json"),
    }
    return mapping[profile]


def test_v2_regulated_profiles_native_success(tmp_path: Path) -> None:
    for profile in ("eudr", "ghg", "iso14064"):
        result = run_v2_profile_backend(
            profile_key=profile,
            input_path=_smoke_input(profile),
            output_dir=tmp_path / profile,
            strict=True,
            allow_fallback=False,
        )
        assert result.success, f"{profile}: {result.errors}"
        assert result.exit_code == 0
        assert result.native_backend_used
        assert not result.fallback_used
        for artifact in result.artifacts:
            assert (tmp_path / profile / artifact).exists(), f"{profile} missing {artifact}"


def test_v2_regulated_profiles_blocked_exit_consistent(tmp_path: Path) -> None:
    payloads = {
        "eudr": '{"suppliers": [], "policy_block": true}',
        "ghg": '{"activities": [], "policy_block": true}',
        "iso14064": '{"controls": [], "policy_block": true}',
    }
    for profile, payload in payloads.items():
        input_path = tmp_path / f"{profile}_blocked.json"
        input_path.write_text(payload, encoding="utf-8")
        result = run_v2_profile_backend(
            profile_key=profile,
            input_path=input_path,
            output_dir=tmp_path / f"{profile}_out",
            strict=True,
            allow_fallback=False,
        )
        assert result.success
        assert result.exit_code == V2_BLOCKED_EXIT_CODE
