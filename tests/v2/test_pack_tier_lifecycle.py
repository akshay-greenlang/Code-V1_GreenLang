from pathlib import Path

import pytest
import yaml

from greenlang.ecosystem.packs.installer import PackInstaller
from greenlang.v2.pack_tiers import (
    evaluate_pack_path,
    evaluate_pack_tier,
    validate_tier_registry,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _v2_pack_payload(
    *,
    name: str,
    tier: str,
    signed: bool,
    signatures: list[str] | None = None,
) -> dict:
    return {
        "contract_version": "2.0",
        "name": name,
        "app_id": "GL-DEMO-APP",
        "version": "2.0.0",
        "kind": "pack",
        "runtime": "greenlang-v2",
        "entry_pipeline": "gl.yaml",
        "metadata": {
            "owner_team": "demo-team",
            "support_channel": "#demo",
            "lifecycle": tier,
            "quality_tier": tier,
        },
        "security": {
            "signed": signed,
            "signatures": signatures or [],
            "sbom": None,
            "provenance": None,
        },
    }


def test_validate_pilot_registry_passes_current_registry() -> None:
    registry = Path("greenlang/ecosystem/packs/v2_tier_registry.yaml")
    assert registry.exists()
    errors = validate_tier_registry(registry)
    assert errors == []


def test_evaluate_pack_tier_denies_supported_without_signature() -> None:
    result = evaluate_pack_tier(
        pack_slug="demo",
        tier="supported",
        owner_team="demo-team",
        support_channel="#demo",
        signed=False,
        signatures=[],
        evidence={
            "docs_contract": True,
            "security_scan": True,
            "determinism_report": False,
        },
    )
    assert not result.ok
    assert any("require signed=true" in error for error in result.errors)


def test_evaluate_pack_path_uses_registry_evidence_for_regulated_tier(tmp_path: Path) -> None:
    pack_dir = tmp_path / "boiler-solar"
    pack_dir.mkdir(parents=True, exist_ok=True)
    payload = _v2_pack_payload(
        name="boiler-solar",
        tier="regulated-critical",
        signed=True,
        signatures=["sig-1.sig"],
    )
    _write_yaml(pack_dir / "pack.yaml", payload)
    (pack_dir / "sig-1.sig").write_text("sig", encoding="utf-8")

    registry_payload = {
        "registry_version": "2.0",
        "pilot_packs": [
            {
                "pack_slug": "boiler-solar",
                "app_id": "GL-EUDR-APP",
                "tier": "regulated-critical",
                "owner_team": "eudr-domain",
                "support_channel": "#gl-eudr",
                "promotion_status": "regulated-approved",
                "evidence": {
                    "docs_contract": True,
                    "signed_artifact": True,
                    "security_scan": True,
                    "determinism_report": True,
                },
            }
        ],
    }
    registry = tmp_path / "registry.yaml"
    _write_yaml(registry, registry_payload)

    evaluation = evaluate_pack_path(pack_dir / "pack.yaml", registry)
    assert evaluation.ok


def test_runtime_enforcement_rejects_invalid_supported_pack(tmp_path: Path) -> None:
    pack_dir = tmp_path / "demo-pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(
        pack_dir / "pack.yaml",
        _v2_pack_payload(name="demo-pack", tier="supported", signed=False, signatures=[]),
    )
    registry_payload = {
        "registry_version": "2.0",
        "pilot_packs": [
            {
                "pack_slug": "demo-pack",
                "app_id": "GL-DEMO-APP",
                "tier": "supported",
                "owner_team": "demo-team",
                "support_channel": "#demo",
                "promotion_status": "supported-approved",
                "evidence": {
                    "docs_contract": True,
                    "signed_artifact": False,
                    "security_scan": True,
                    "determinism_report": False,
                },
            }
        ],
    }
    registry = tmp_path / "registry.yaml"
    _write_yaml(registry, registry_payload)

    installer = PackInstaller()
    installer.v2_tier_registry_path = registry

    with pytest.raises(ValueError) as exc:
        installer._enforce_v2_tier_lifecycle(pack_dir / "pack.yaml")
    assert "V2 tier lifecycle enforcement failed" in str(exc.value)
