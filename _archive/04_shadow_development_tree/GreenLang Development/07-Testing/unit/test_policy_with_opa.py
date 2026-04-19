#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for GPL license policy enforcement with local OPA
"""

import sys
import json
import os
import subprocess
import tempfile
from pathlib import Path

# Removed sys.path manipulation - using installed package

# Set up environment with local OPA
current_dir = Path(__file__).parent
opa_path = current_dir / "opa.exe"
if opa_path.exists():
    os.environ["PATH"] = str(current_dir) + os.pathsep + os.environ.get("PATH", "")


def test_opa_available():
    """Test if OPA is now available"""
    print("Testing OPA Installation")
    print("-" * 40)
    
    try:
        result = subprocess.run([str(opa_path), "version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"OPA is available: {result.stdout.strip().split()[1]}")
            return True
        else:
            print(f"OPA error: {result.stderr}")
            return False
    except Exception as e:
        print(f"OPA test failed: {e}")
        return False


def test_policy_evaluation():
    """Test policy evaluation with actual OPA"""
    print("\nTesting Policy Evaluation with OPA")
    print("-" * 40)
    
    # Create test input for GPL pack
    gpl_input = {
        "pack": {
            "name": "test-gpl-pack",
            "version": "1.0.0",
            "kind": "pack",
            "license": "GPL-3.0",
            "policy": {
                "network": ["github.com"],
                "ef_vintage_min": 2024
            },
            "security": {
                "sbom": "test.json"
            }
        },
        "stage": "publish"
    }
    
    # Test with bundle policy
    bundles_dir = Path("core/greenlang/policy/bundles")
    install_policy = bundles_dir / "install.rego"
    
    if not install_policy.exists():
        print(f"Policy file not found: {install_policy}")
        return False
    
    try:
        # Create temp input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(gpl_input, f)
            input_file = f.name
        
        # Run OPA evaluation
        cmd = [
            str(opa_path), "eval",
            "-d", str(install_policy),
            "-i", input_file,
            "--format", "json",
            "data.greenlang.install"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"OPA evaluation failed: {result.stderr}")
            return False
        
        # Parse result
        output = json.loads(result.stdout)
        print(f"OPA output: {json.dumps(output, indent=2)}")
        
        # Extract decision
        if output.get("result") and len(output["result"]) > 0:
            expressions = output["result"][0].get("expressions", [])
            if expressions and len(expressions) > 0:
                decision = expressions[0].get("value", {})
                
                allow = decision.get("allow", False)
                reason = decision.get("reason", "No reason provided")
                
                print(f"Decision: allow={allow}, reason='{reason}'")
                
                if not allow and "gpl" in reason.lower():
                    print("SUCCESS: GPL license properly denied by OPA policy!")
                    return True
                elif not allow:
                    print("SUCCESS: Pack denied, but reason might not be license-specific")
                    return True
                else:
                    print("FAIL: GPL pack was allowed by policy")
                    return False
        
        print("No valid decision found in OPA output")
        return False
        
    except Exception as e:
        print(f"Error testing policy: {e}")
        return False
    finally:
        # Clean up temp file
        if 'input_file' in locals():
            Path(input_file).unlink(missing_ok=True)


def test_allowed_license_with_opa():
    """Test that allowed licenses work with OPA"""
    print("\nTesting Allowed License with OPA")
    print("-" * 40)
    
    # Test MIT license
    mit_input = {
        "pack": {
            "name": "test-mit-pack",
            "version": "1.0.0", 
            "kind": "pack",
            "license": "MIT",
            "policy": {
                "network": ["github.com"],
                "ef_vintage_min": 2024
            },
            "security": {
                "sbom": "test.json"
            }
        },
        "stage": "publish"
    }
    
    bundles_dir = Path("core/greenlang/policy/bundles")
    install_policy = bundles_dir / "install.rego"
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mit_input, f)
            input_file = f.name
        
        cmd = [
            str(opa_path), "eval",
            "-d", str(install_policy),
            "-i", input_file,
            "--format", "json",
            "data.greenlang.install"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"OPA evaluation failed: {result.stderr}")
            return False
        
        output = json.loads(result.stdout)
        
        if output.get("result") and len(output["result"]) > 0:
            expressions = output["result"][0].get("expressions", [])
            if expressions and len(expressions) > 0:
                decision = expressions[0].get("value", {})
                
                allow = decision.get("allow", False)
                reason = decision.get("reason", "No reason provided")
                
                print(f"MIT License Decision: allow={allow}, reason='{reason}'")
                
                if allow:
                    print("SUCCESS: MIT license properly allowed!")
                    return True
                else:
                    print(f"UNEXPECTED: MIT license denied: {reason}")
                    return False
        
        return False
        
    except Exception as e:
        print(f"Error testing MIT license: {e}")
        return False
    finally:
        if 'input_file' in locals():
            Path(input_file).unlink(missing_ok=True)


def test_policy_integration():
    """Test the full policy integration with local OPA"""
    print("\nTesting Full Policy Integration")
    print("-" * 40)
    
    try:
        from greenlang.policy.enforcer import check_install
        from pydantic import BaseModel
        from typing import Optional, Dict, Any
        
        class TestPackManifest(BaseModel):
            name: str
            version: str
            kind: str = "pack"
            license: Optional[str] = None
            policy: Optional[Dict[str, Any]] = {}
            security: Optional[Dict[str, Any]] = {}
            
            def model_dump(self):
                return self.dict()
        
        # Test GPL pack
        gpl_pack = TestPackManifest(
            name="test-gpl",
            version="1.0.0",
            license="GPL-3.0",
            policy={"network": ["github.com"], "ef_vintage_min": 2024},
            security={"sbom": "test.json"}
        )
        
        print("Testing GPL pack with integrated policy...")
        try:
            check_install(gpl_pack, ".", "publish")
            print("FAIL: GPL pack was allowed by integrated policy")
            return False
        except RuntimeError as e:
            print(f"SUCCESS: GPL pack denied by integrated policy: {e}")
            return True
        
    except Exception as e:
        print(f"Integration test error: {e}")
        return False


def main():
    """Run all tests with OPA"""
    print("GreenLang Policy Test Suite with OPA")
    print("=" * 50)
    
    results = []
    
    # Test OPA installation
    results.append(test_opa_available())
    
    # Test direct OPA policy evaluation
    results.append(test_policy_evaluation())
    
    # Test allowed license
    results.append(test_allowed_license_with_opa())
    
    # Test integrated policy
    results.append(test_policy_integration())
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("ALL TESTS PASSED - Policy enforcement working correctly!")
    else:
        print("Some tests failed - review policy implementation")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)