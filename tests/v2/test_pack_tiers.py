from pathlib import Path

import yaml

from greenlang.v2.contracts import validate_v2_pack


def test_supported_tier_requires_signed_pack(tmp_path: Path) -> None:
    pack = {
        "contract_version": "2.0",
        "name": "demo-pack",
        "app_id": "GL-DEMO-APP",
        "version": "2.0.0",
        "kind": "pack",
        "runtime": "greenlang-v2",
        "entry_pipeline": "gl.yaml",
        "metadata": {
            "owner_team": "demo-team",
            "support_channel": "#demo",
            "lifecycle": "supported",
            "quality_tier": "supported",
        },
        "security": {
            "signed": False,
            "signatures": [],
            "sbom": None,
            "provenance": None,
        },
    }
    target = tmp_path / "pack.yaml"
    target.write_text(yaml.safe_dump(pack, sort_keys=False), encoding="utf-8")
    finding = validate_v2_pack(target)
    assert not finding.ok
    assert any("require signed=true" in err for err in finding.errors)

