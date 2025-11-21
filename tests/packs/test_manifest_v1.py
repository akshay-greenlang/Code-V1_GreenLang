# -*- coding: utf-8 -*-
"""
Tests for GreenLang Pack Manifest v1.0 Specification
"""

import pytest
import json
import yaml
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

from greenlang.packs.manifest import PackManifest, Contents, Compat, Security
from pydantic import ValidationError


class TestPackManifestV1:
    """Test suite for pack.yaml v1.0 specification"""
    
    def test_minimal_valid_manifest(self):
        """Test that minimal required fields create a valid manifest"""
        data = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "license": "MIT",
            "contents": {
                "pipelines": ["gl.yaml"]
            }
        }
        
        manifest = PackManifest(**data)
        assert manifest.name == "test-pack"
        assert manifest.version == "1.0.0"
        assert manifest.kind == "pack"
        assert manifest.license == "MIT"
        assert manifest.contents.pipelines == ["gl.yaml"]
    
    def test_full_valid_manifest(self):
        """Test full manifest with all optional fields"""
        data = {
            "name": "boiler-solar",
            "version": "2.3.1",
            "kind": "pack",
            "license": "Apache-2.0",
            "compat": {
                "greenlang": ">=0.3,<0.5",
                "python": ">=3.10"
            },
            "contents": {
                "pipelines": ["gl.yaml", "backup.yaml"],
                "agents": ["BoilerAgent", "SolarAgent"],
                "datasets": ["data/ef_2025.csv"],
                "reports": ["reports/summary.html.j2"]
            },
            "dependencies": [
                "pandas>=2.1",
                {"name": "ephem", "version": ">=4.1"}
            ],
            "card": "CARD.md",
            "policy": {
                "network": ["era5:*"],
                "data_residency": ["IN", "EU"],
                "ef_vintage_min": 2024
            },
            "security": {
                "sbom": "sbom.spdx.json",
                "signatures": ["pack.sig"]
            },
            "metadata": {
                "description": "Climate intelligence pack",
                "authors": ["John Doe"],
                "tags": ["climate", "solar"]
            }
        }
        
        manifest = PackManifest(**data)
        assert manifest.name == "boiler-solar"
        assert manifest.version == "2.3.1"
        assert manifest.compat.greenlang == ">=0.3,<0.5"
        assert len(manifest.contents.agents) == 2
        assert manifest.security.sbom == "sbom.spdx.json"
    
    def test_missing_required_field_name(self):
        """Test that missing 'name' field fails validation"""
        data = {
            "version": "1.0.0",
            "kind": "pack",
            "license": "MIT",
            "contents": {
                "pipelines": ["gl.yaml"]
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PackManifest(**data)
        
        assert "name" in str(exc_info.value)
    
    def test_missing_required_field_contents(self):
        """Test that missing 'contents' field fails validation"""
        data = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "license": "MIT"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PackManifest(**data)
        
        assert "contents" in str(exc_info.value)
    
    def test_empty_pipelines_fails(self):
        """Test that empty pipelines array fails validation"""
        data = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "license": "MIT",
            "contents": {
                "pipelines": []
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PackManifest(**data)
        
        assert "at least one pipeline" in str(exc_info.value).lower()
    
    def test_invalid_name_format(self):
        """Test that invalid pack names fail validation"""
        invalid_names = [
            "TestPack",  # uppercase
            "test_pack",  # underscore
            "test.pack",  # dot
            "123pack",  # starts with number
            "te",  # too short
            "a" * 65,  # too long
            "-test-pack",  # starts with hyphen
            "test-pack-"  # ends with hyphen
        ]
        
        for invalid_name in invalid_names:
            data = {
                "name": invalid_name,
                "version": "1.0.0",
                "kind": "pack",
                "license": "MIT",
                "contents": {
                    "pipelines": ["gl.yaml"]
                }
            }
            
            with pytest.raises(ValidationError) as exc_info:
                PackManifest(**data)
            
            assert "DNS-safe" in str(exc_info.value) or "name" in str(exc_info.value)
    
    def test_invalid_version_format(self):
        """Test that invalid semantic versions fail validation"""
        invalid_versions = [
            "1.0",  # missing patch
            "1",  # missing minor and patch
            "v1.0.0",  # has v prefix
            "1.0.0-alpha",  # has prerelease (not in v1.0 spec)
            "1.0.0.1",  # too many parts
            "1.a.0",  # non-numeric
        ]
        
        for invalid_version in invalid_versions:
            data = {
                "name": "test-pack",
                "version": invalid_version,
                "kind": "pack",
                "license": "MIT",
                "contents": {
                    "pipelines": ["gl.yaml"]
                }
            }
            
            with pytest.raises(ValidationError) as exc_info:
                PackManifest(**data)
            
            assert "MAJOR.MINOR.PATCH" in str(exc_info.value) or "version" in str(exc_info.value)
    
    def test_invalid_kind_enum(self):
        """Test that invalid kind values fail validation"""
        data = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "invalid",
            "license": "MIT",
            "contents": {
                "pipelines": ["gl.yaml"]
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PackManifest(**data)
        
        assert "pack" in str(exc_info.value) or "dataset" in str(exc_info.value)
    
    def test_file_existence_validation(self, tmp_path):
        """Test file existence validation method"""
        # Create a temporary pack structure
        pack_dir = tmp_path / "test-pack"
        pack_dir.mkdir()
        
        # Create manifest
        manifest = PackManifest(
            name="test-pack",
            version="1.0.0",
            kind="pack",
            license="MIT",
            contents=Contents(
                pipelines=["gl.yaml"],
                datasets=["data.csv"]
            ),
            card="CARD.md"
        )
        
        # Check missing files
        missing = manifest.validate_files_exist(pack_dir)
        assert len(missing) == 3  # gl.yaml, data.csv, CARD.md
        assert "gl.yaml" in str(missing)
        
        # Create the files
        (pack_dir / "gl.yaml").touch()
        (pack_dir / "data.csv").touch()
        (pack_dir / "CARD.md").touch()
        
        # Check again
        missing = manifest.validate_files_exist(pack_dir)
        assert len(missing) == 0
    
    def test_get_warnings(self):
        """Test warning generation for recommended fields"""
        # Minimal manifest
        manifest = PackManifest(
            name="test-pack",
            version="1.0.0",
            kind="pack",
            license="MIT",
            contents=Contents(pipelines=["gl.yaml"])
        )
        
        warnings = manifest.get_warnings()
        assert any("card" in w.lower() for w in warnings)
        assert any("compat" in w.lower() for w in warnings)
        assert any("description" in w.lower() for w in warnings)
        
        # Full manifest should have no warnings
        manifest_full = PackManifest(
            name="test-pack",
            version="1.0.0",
            kind="pack",
            license="MIT",
            contents=Contents(pipelines=["gl.yaml"]),
            card="CARD.md",
            compat=Compat(greenlang=">=0.3"),
            security=Security(sbom="sbom.json"),
            metadata={"description": "Test pack"}
        )
        
        warnings_full = manifest_full.get_warnings()
        assert len(warnings_full) == 0
    
    def test_yaml_roundtrip(self):
        """Test YAML serialization and deserialization"""
        manifest = PackManifest(
            name="test-pack",
            version="1.0.0",
            kind="pack",
            license="MIT",
            contents=Contents(pipelines=["gl.yaml"])
        )
        
        # Convert to YAML
        yaml_str = manifest.to_yaml()
        assert "name: test-pack" in yaml_str
        assert "version: 1.0.0" in yaml_str
        
        # Parse back from YAML
        manifest2 = PackManifest.from_yaml(yaml_str)
        assert manifest2.name == manifest.name
        assert manifest2.version == manifest.version
    
    def test_json_roundtrip(self):
        """Test JSON serialization and deserialization"""
        manifest = PackManifest(
            name="test-pack",
            version="1.0.0",
            kind="pack",
            license="MIT",
            contents=Contents(pipelines=["gl.yaml"])
        )
        
        # Convert to JSON
        json_str = manifest.to_json()
        data = json.loads(json_str)
        assert data["name"] == "test-pack"
        
        # Parse back from JSON
        manifest2 = PackManifest.from_json(json_str)
        assert manifest2.name == manifest.name
        assert manifest2.version == manifest.version
    
    def test_from_file_yaml(self, tmp_path):
        """Test loading manifest from YAML file"""
        # Create a YAML file
        yaml_file = tmp_path / "pack.yaml"
        yaml_content = """
name: test-pack
version: 1.0.0
kind: pack
license: MIT
contents:
  pipelines:
    - gl.yaml
"""
        yaml_file.write_text(yaml_content)
        
        # Load from file
        manifest = PackManifest.from_file(yaml_file)
        assert manifest.name == "test-pack"
        assert manifest.version == "1.0.0"
    
    def test_from_file_json(self, tmp_path):
        """Test loading manifest from JSON file"""
        # Create a JSON file
        json_file = tmp_path / "pack.json"
        json_content = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "license": "MIT",
            "contents": {
                "pipelines": ["gl.yaml"]
            }
        }
        json_file.write_text(json.dumps(json_content))
        
        # Load from file
        manifest = PackManifest.from_file(json_file)
        assert manifest.name == "test-pack"
        assert manifest.version == "1.0.0"
    
    def test_dependency_formats(self):
        """Test various dependency format support"""
        data = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "license": "MIT",
            "contents": {
                "pipelines": ["gl.yaml"]
            },
            "dependencies": [
                "pandas>=2.1",
                {"name": "numpy", "version": ">=1.24"},
                "requests",
                {"name": "scipy"}
            ]
        }
        
        manifest = PackManifest(**data)
        assert len(manifest.dependencies) == 4
        assert "pandas>=2.1" in manifest.dependencies
        assert {"name": "numpy", "version": ">=1.24"} in manifest.dependencies
    
    def test_common_licenses(self):
        """Test common SPDX license identifiers"""
        common_licenses = [
            "MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause",
            "ISC", "MPL-2.0", "Commercial", "Proprietary"
        ]
        
        for license_id in common_licenses:
            data = {
                "name": "test-pack",
                "version": "1.0.0",
                "kind": "pack",
                "license": license_id,
                "contents": {
                    "pipelines": ["gl.yaml"]
                }
            }
            
            manifest = PackManifest(**data)
            assert manifest.license == license_id
    
    def test_backward_compatibility(self):
        """Test that v1.0 parser handles future additions gracefully"""
        data = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "license": "MIT",
            "contents": {
                "pipelines": ["gl.yaml"]
            },
            # Future field that doesn't exist in v1.0
            "future_field": "some_value",
            "experimental": {
                "feature": "test"
            }
        }
        
        # Should not fail - unknown fields are preserved
        manifest = PackManifest(**data)
        assert manifest.name == "test-pack"
        
        # The model allows additional fields
        assert hasattr(manifest, "name")
        assert not hasattr(manifest, "future_field")  # Not in model, but allowed in data


class TestPackManifestIntegration:
    """Integration tests for pack manifest with real files"""
    
    def test_create_and_validate_pack(self, tmp_path):
        """Test creating a pack and validating it"""
        # Create pack directory
        pack_dir = tmp_path / "my-pack"
        pack_dir.mkdir()
        
        # Create pack.yaml
        pack_yaml = pack_dir / "pack.yaml"
        manifest_data = {
            "name": "my-pack",
            "version": "1.0.0",
            "kind": "pack",
            "license": "MIT",
            "contents": {
                "pipelines": ["gl.yaml"]
            }
        }
        pack_yaml.write_text(yaml.dump(manifest_data))
        
        # Create gl.yaml
        gl_yaml = pack_dir / "gl.yaml"
        gl_yaml.write_text("# Pipeline configuration\n")
        
        # Load and validate
        manifest = PackManifest.from_file(pack_yaml)
        missing = manifest.validate_files_exist(pack_dir)
        
        assert len(missing) == 0
        assert manifest.name == "my-pack"
        
        # Get warnings
        warnings = manifest.get_warnings()
        assert len(warnings) > 0  # Should have warnings for missing recommended fields


if __name__ == "__main__":
    pytest.main([__file__, "-v"])