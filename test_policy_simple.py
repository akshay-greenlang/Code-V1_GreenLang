#!/usr/bin/env python3
"""
Simple test script for GPL license policy enforcement
"""

import sys
import json
from pathlib import Path

# Add the core module to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

try:
    from greenlang.policy.enforcer import PolicyEnforcer, check_install
    from greenlang.packs.manifest import load_manifest, PackManifest
    from pydantic import BaseModel
    from typing import Optional, Dict, Any, List
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying with fallback imports...")
    
    # Simple fallback manifest class
    class PackManifest(BaseModel):
        name: str
        version: str
        kind: str = "pack"
        license: Optional[str] = None
        policy: Optional[Dict[str, Any]] = {}
        security: Optional[Dict[str, Any]] = {}
        
        def model_dump(self):
            return self.dict()
            
    def load_manifest(path):
        """Simple manifest loader"""
        import yaml
        pack_file = Path(path) / "pack.yaml" if Path(path).is_dir() else Path(path)
        
        if pack_file.name == "test-gpl-pack.yaml":
            pack_file = Path(path)
        
        with open(pack_file, 'r') as f:
            data = yaml.safe_load(f)
        
        return PackManifest(**data)


def test_gpl_denial():
    """Test that GPL pack is denied"""
    print("Testing GPL License Policy Denial")
    print("-" * 40)
    
    # Test with existing GPL pack file
    gpl_pack_path = Path("test-gpl-pack.yaml")
    
    if not gpl_pack_path.exists():
        print(f"ERROR: GPL test pack not found: {gpl_pack_path}")
        return False
    
    print(f"Loading pack from: {gpl_pack_path}")
    
    try:
        # Load the manifest
        manifest = load_manifest(gpl_pack_path)
        print(f"Pack loaded: {manifest.name} v{manifest.version}")
        print(f"License: {manifest.license}")
        print()
        
        # Test with the check_install function
        print("Testing with check_install function...")
        try:
            check_install(manifest, str(gpl_pack_path), "publish")
            print("FAIL: GPL pack was ALLOWED (should be denied)")
            return False
        except RuntimeError as e:
            print(f"OK: GPL pack was DENIED: {e}")
            return True
        
    except Exception as e:
        print(f"ERROR: Failed to test GPL pack: {e}")
        return False


def test_allowed_licenses():
    """Test that allowed licenses work"""
    print("\nTesting Allowed Licenses")
    print("-" * 40)
    
    allowed_licenses = ["MIT", "Apache-2.0", "Commercial"]
    results = []
    
    for license_type in allowed_licenses:
        print(f"Testing {license_type} license...")
        
        # Create test pack with allowed license
        test_pack = PackManifest(
            name="test-pack",
            version="1.0.0",
            kind="pack", 
            license=license_type,
            policy={"network": ["example.com"], "ef_vintage_min": 2024},
            security={"sbom": "sbom.json"}
        )
        
        try:
            check_install(test_pack, ".", "publish")
            print(f"OK: {license_type} pack allowed")
            results.append(True)
        except RuntimeError as e:
            print(f"Warning: {license_type} pack denied: {e}")
            results.append(False)
        except Exception as e:
            print(f"Error testing {license_type}: {e}")
            results.append(False)
    
    return all(results)


def create_gpl_test_pack():
    """Create a detailed GPL test pack"""
    print("\nCreating Detailed GPL Test Pack")
    print("-" * 40)
    
    gpl_pack_data = {
        "name": "gpl-test-detailed",
        "version": "1.0.0",
        "kind": "pack",
        "license": "GPL-3.0",
        "description": "A test pack with GPL license",
        "policy": {
            "network": ["github.com"],
            "ef_vintage_min": 2024
        },
        "security": {
            "sbom": "test-sbom.json"
        }
    }
    
    # Save to file
    gpl_pack_path = Path("test-gpl-detailed.yaml")
    
    import yaml
    with open(gpl_pack_path, 'w') as f:
        yaml.dump(gpl_pack_data, f, default_flow_style=False)
    
    print(f"Created detailed GPL test pack: {gpl_pack_path}")
    
    # Test it
    try:
        manifest = load_manifest(gpl_pack_path)
        check_install(manifest, str(gpl_pack_path), "publish") 
        print("FAIL: Detailed GPL pack was allowed")
        return False
    except RuntimeError as e:
        print(f"OK: Detailed GPL pack denied: {e}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_various_denied_licenses():
    """Test various licenses that should be denied"""
    print("\nTesting Various Denied Licenses")
    print("-" * 40)
    
    denied_licenses = ["GPL-2.0", "GPL-3.0", "LGPL-2.1", "AGPL-3.0"]
    
    for license_type in denied_licenses:
        test_pack = PackManifest(
            name="test-pack",
            version="1.0.0",
            license=license_type
        )
        
        try:
            check_install(test_pack, ".", "publish")
            print(f"FAIL: {license_type} was allowed (should be denied)")
        except RuntimeError as e:
            print(f"OK: {license_type} correctly denied")
        except Exception as e:
            print(f"Error with {license_type}: {e}")


def main():
    """Run all policy tests"""
    print("GreenLang Policy Check Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test GPL denial
    results.append(test_gpl_denial())
    
    # Test allowed licenses  
    results.append(test_allowed_licenses())
    
    # Create and test additional GPL pack
    results.append(create_gpl_test_pack())
    
    # Test various denied licenses
    test_various_denied_licenses()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("ALL TESTS PASSED - GPL license properly denied!")
    else:
        print("Some tests failed - review policy implementation")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)