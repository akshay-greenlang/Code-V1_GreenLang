import pytest
from pathlib import Path

def test_pack_structure():
    """Test that pack has required structure"""
    pack_dir = Path(__file__).parent.parent
    assert (pack_dir / "pack.yaml").exists()
    assert (pack_dir / "gl.yaml").exists()
    assert (pack_dir / "CARD.md").exists()

def test_manifest_valid():
    """Test that manifest is valid YAML"""
    import yaml
    pack_dir = Path(__file__).parent.parent
    with open(pack_dir / "pack.yaml") as f:
        manifest = yaml.safe_load(f)
    assert manifest["name"] == "tmp-pack"
    assert "version" in manifest
