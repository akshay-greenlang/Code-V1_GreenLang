#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for GPL license policy enforcement
"""

import sys
import json
from pathlib import Path

# Removed sys.path manipulation - using installed package

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


def test_gpl_pack_denial():
    """Test that GPL pack is properly denied"""
    print("=== Testing GPL License Policy Denial ===\n")
    
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
        
        # Test with the check_install function directly
        print("Testing with check_install function...")
        try:
            check_install(manifest, str(gpl_pack_path), "publish")
            print("❌ FAIL: GPL pack was ALLOWED (should be denied)")
            return False
        except RuntimeError as e:
            print(f"✅ OK: GPL pack was DENIED: {e}")
            
            # Check if the error mentions license
            if "license" in str(e).lower():
                print("✅ OK: Error correctly identified license issue")
            else:
                print("❓ Warning: Error doesn't mention license issue")
        
        print()
        
        # Test with PolicyEnforcer class
        print("Testing with PolicyEnforcer class...")
        try:
            enforcer = PolicyEnforcer()
            allowed, reasons = enforcer.check_install(manifest, str(gpl_pack_path), "publish")
            
            if not allowed:
                print(f"✅ OK: PolicyEnforcer correctly denied GPL pack")
                print(f"Reasons: {reasons}")
            else:
                print("❌ FAIL: PolicyEnforcer allowed GPL pack (should be denied)")
                return False
        except Exception as e:
            print(f"⚠️  PolicyEnforcer error (expected in fallback mode): {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to test GPL pack: {e}")
        return False


def test_allowed_licenses():
    """Test that allowed licenses work correctly"""
    print("\n=== Testing Allowed Licenses ===\n")
    
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
            print(f"✅ OK: {license_type} pack allowed")
            results.append(True)
        except RuntimeError as e:
            print(f"❓ Warning: {license_type} pack denied: {e}")
            results.append(False)
        except Exception as e:
            print(f"⚠️  Error testing {license_type}: {e}")
            results.append(False)
    
    return all(results)


def create_additional_gpl_test_pack():
    """Create an additional GPL test pack with more details"""
    print("\n=== Creating Additional GPL Test Pack ===\n")
    
    gpl_pack_data = {
        "name": "gpl-test-pack-detailed",
        "version": "1.0.0",
        "kind": "pack",
        "license": "GPL-3.0",
        "description": "A test pack with GPL license for policy testing",
        "author": "Test User",
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
        print("❌ FAIL: Detailed GPL pack was allowed")
        return False
    except RuntimeError as e:
        print(f"✅ OK: Detailed GPL pack denied: {e}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_policy_rules_completeness():
    """Test various aspects of policy rules"""
    print("\n=== Testing Policy Rules Completeness ===\n")
    
    # Test various denied licenses
    denied_licenses = ["GPL-2.0", "GPL-3.0", "LGPL-2.1", "AGPL-3.0", "WTFPL"]
    
    for license_type in denied_licenses:
        test_pack = PackManifest(
            name="test-pack",
            version="1.0.0",
            license=license_type
        )
        
        try:
            check_install(test_pack, ".", "publish")
            print(f"❌ FAIL: {license_type} was allowed (should be denied)")
        except RuntimeError as e:
            print(f"✅ OK: {license_type} correctly denied")
        except Exception as e:
            print(f"⚠️  Error with {license_type}: {e}")
    
    # Test missing license
    try:
        test_pack = PackManifest(
            name="test-pack",
            version="1.0.0"
            # No license specified
        )
        
        check_install(test_pack, ".", "publish")
        print("❌ FAIL: Pack with no license was allowed")
    except RuntimeError as e:
        print(f"✅ OK: Pack with no license denied: {e}")
    except Exception as e:
        print(f"⚠️  Error with missing license: {e}")


def main():
    """Run all policy tests"""
    print("GreenLang Policy Check Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test GPL denial
    results.append(test_gpl_pack_denial())
    
    # Test allowed licenses
    results.append(test_allowed_licenses())
    
    # Create and test additional GPL pack
    results.append(create_additional_gpl_test_pack())
    
    # Test policy completeness
    test_policy_rules_completeness()
    
    # Final summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - GPL license properly denied!")
    else:
        print("❌ Some tests failed - review policy implementation")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)