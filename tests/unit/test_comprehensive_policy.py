#!/usr/bin/env python3
"""
Comprehensive test of GreenLang policy enforcement at all points
"""

import sys
import json
import subprocess
import tempfile
from pathlib import Path

def test_install_time_enforcement():
    """Test policy enforcement during pack installation/publishing"""
    print("1. PACK INSTALLATION POLICY ENFORCEMENT")
    print("=" * 50)
    
    # Test GPL pack denial
    print("\n1.1 Testing GPL License Denial:")
    result = subprocess.run(
        [sys.executable, "gl", "policy", "check", "test-gpl-pack"],
        capture_output=True, text=True
    )
    
    if result.returncode != 0 and "gpl" in result.stderr.lower():
        print("PASS: GPL pack properly denied during install check")
        install_gpl = True
    else:
        print("FAIL: GPL pack was not properly denied")
        install_gpl = False
    
    # Test MIT pack approval
    print("\n1.2 Testing MIT License Approval:")
    result = subprocess.run(
        [sys.executable, "gl", "policy", "check", "test-mit-pack"],
        capture_output=True, text=True
    )
    
    if result.returncode == 0 and "ok" in result.stdout.lower():
        print("PASS: MIT pack properly approved during install check")
        install_mit = True
    else:
        print("FAIL: MIT pack was not properly approved")
        install_mit = False
    
    return install_gpl and install_mit


def test_runtime_enforcement():
    """Test policy enforcement during pipeline execution"""
    print("\n\n2. PIPELINE EXECUTION POLICY ENFORCEMENT")
    print("=" * 50)
    
    opa_path = Path.cwd() / "opa.exe"
    run_policy = Path("core/greenlang/policy/bundles/run.rego")
    
    # Test unauthorized network access blocking
    print("\n2.1 Testing Network Policy Enforcement:")
    bad_network_input = {
        "pipeline": {
            "policy": {"network": ["github.com"], "max_memory": 2048, "max_cpu": 4, "max_disk": 2048},
            "resources": {"memory": 1024, "cpu": 2, "disk": 1024}
        },
        "egress": ["malicious-site.com"],
        "stage": "production"
    }
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(bad_network_input, f)
            input_file = f.name
        
        result = subprocess.run([
            str(opa_path), "eval", "-d", str(run_policy), "-i", input_file,
            "--format", "json", "data.greenlang.decision"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            output = json.loads(result.stdout)
            decision = output["result"][0]["expressions"][0]["value"]
            
            if not decision.get("allow", True) and "egress" in decision.get("reason", "").lower():
                print("PASS: Unauthorized network access properly blocked")
                network_block = True
            else:
                print("FAIL: Unauthorized network access not blocked")
                network_block = False
        else:
            print("FAIL: Network policy evaluation failed")
            network_block = False
            
    except Exception as e:
        print(f"FAIL: Network test error: {e}")
        network_block = False
    finally:
        Path(input_file).unlink(missing_ok=True)
    
    # Test resource limit enforcement
    print("\n2.2 Testing Resource Limit Enforcement:")
    resource_limit_input = {
        "pipeline": {
            "policy": {"network": ["github.com"], "max_memory": 1024, "max_cpu": 2, "max_disk": 1024},
            "resources": {"memory": 2048, "cpu": 4, "disk": 512}  # Exceeds memory and CPU limits
        },
        "egress": [],
        "stage": "production"
    }
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(resource_limit_input, f)
            input_file = f.name
        
        result = subprocess.run([
            str(opa_path), "eval", "-d", str(run_policy), "-i", input_file,
            "--format", "json", "data.greenlang.decision"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            output = json.loads(result.stdout)
            decision = output["result"][0]["expressions"][0]["value"]
            
            if not decision.get("allow", True) and "resource" in decision.get("reason", "").lower():
                print("PASS: Resource limit enforcement working")
                resource_limit = True
            else:
                print("FAIL: Resource limit enforcement failed")
                resource_limit = False
        else:
            print("FAIL: Resource limit policy evaluation failed")
            resource_limit = False
            
    except Exception as e:
        print(f"FAIL: Resource limit test error: {e}")
        resource_limit = False
    finally:
        Path(input_file).unlink(missing_ok=True)
    
    return network_block and resource_limit


def test_opa_integration():
    """Test direct OPA integration"""
    print("\n\n3. OPA INTEGRATION VERIFICATION")
    print("=" * 50)
    
    # Test OPA availability
    opa_path = Path.cwd() / "opa.exe"
    print(f"\n3.1 OPA Executable: {opa_path.exists()}")
    
    if opa_path.exists():
        try:
            result = subprocess.run([str(opa_path), "version"], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip().split()[1]
                print(f"PASS: OPA Version: {version}")
                opa_available = True
            else:
                print("FAIL: OPA not working")
                opa_available = False
        except Exception:
            print("FAIL: OPA execution failed")
            opa_available = False
    else:
        print("FAIL: OPA executable not found")
        opa_available = False
    
    # Test policy files existence
    install_policy = Path("core/greenlang/policy/bundles/install.rego")
    run_policy = Path("core/greenlang/policy/bundles/run.rego")
    
    print(f"\n3.2 Install Policy: {install_policy.exists()}")
    print(f"3.3 Runtime Policy: {run_policy.exists()}")
    
    policy_files = install_policy.exists() and run_policy.exists()
    
    return opa_available and policy_files


def test_enforcement_scenarios():
    """Test various enforcement scenarios"""
    print("\n\n4. COMPREHENSIVE ENFORCEMENT SCENARIOS")
    print("=" * 50)
    
    scenarios_passed = 0
    total_scenarios = 6
    
    # Scenario 1: GPL pack with good network policy
    print("\n4.1 GPL pack with valid network policy (should be DENIED for license):")
    gpl_test_input = {
        "pack": {"license": "GPL-3.0", "policy": {"network": ["github.com"], "ef_vintage_min": 2024}},
        "stage": "publish"
    }
    
    if test_opa_scenario("core/greenlang/policy/bundles/install.rego", gpl_test_input, expect_allow=False):
        print("PASS: Scenario 1 passed")
        scenarios_passed += 1
    else:
        print("FAIL: Scenario 1 failed")
    
    # Scenario 2: MIT pack with missing network policy
    print("\n4.2 MIT pack with missing network policy (should be DENIED for network):")
    mit_no_network_input = {
        "pack": {"license": "MIT", "policy": {"network": [], "ef_vintage_min": 2024}},
        "stage": "publish"
    }
    
    if test_opa_scenario("core/greenlang/policy/bundles/install.rego", mit_no_network_input, expect_allow=False):
        print("PASS: Scenario 2 passed")
        scenarios_passed += 1
    else:
        print("FAIL: Scenario 2 failed")
    
    # Scenario 3: Pipeline with authorized network access
    print("\n4.3 Pipeline with authorized network access (should be ALLOWED):")
    good_pipeline_input = {
        "pipeline": {
            "policy": {"network": ["github.com"], "max_memory": 2048, "max_cpu": 4, "max_disk": 2048},
            "resources": {"memory": 1024, "cpu": 2, "disk": 1024}
        },
        "egress": ["github.com"],
        "stage": "production"
    }
    
    if test_opa_scenario("core/greenlang/policy/bundles/run.rego", good_pipeline_input, expect_allow=True):
        print("PASS: Scenario 3 passed")
        scenarios_passed += 1
    else:
        print("FAIL: Scenario 3 failed")
    
    # Scenario 4: Pipeline with mixed authorized/unauthorized domains
    print("\n4.4 Pipeline with mixed network domains (should be DENIED):")
    mixed_pipeline_input = {
        "pipeline": {
            "policy": {"network": ["github.com"], "max_memory": 2048, "max_cpu": 4, "max_disk": 2048},
            "resources": {"memory": 1024, "cpu": 2, "disk": 1024}
        },
        "egress": ["github.com", "bad-site.com"],  # One good, one bad
        "stage": "production"
    }
    
    if test_opa_scenario("core/greenlang/policy/bundles/run.rego", mixed_pipeline_input, expect_allow=False):
        print("PASS: Scenario 4 passed")
        scenarios_passed += 1
    else:
        print("FAIL: Scenario 4 failed")
    
    # Scenario 5: Development stage with relaxed rules
    print("\n4.5 Development stage execution (should be more permissive):")
    dev_pipeline_input = {
        "pipeline": {
            "policy": {"network": ["github.com"], "max_memory": 2048, "max_cpu": 4, "max_disk": 2048},
            "resources": {"memory": 1024, "cpu": 2, "disk": 1024}
        },
        "egress": ["github.com"],
        "stage": "dev"  # Development stage
    }
    
    if test_opa_scenario("core/greenlang/policy/bundles/run.rego", dev_pipeline_input, expect_allow=True):
        print("PASS: Scenario 5 passed")
        scenarios_passed += 1
    else:
        print("FAIL: Scenario 5 failed")
    
    # Scenario 6: Production stage with strict enforcement
    print("\n4.6 Production stage with policy violations (should be DENIED):")
    prod_violation_input = {
        "pipeline": {
            "policy": {"network": ["github.com"], "max_memory": 1024, "max_cpu": 2, "max_disk": 1024},
            "resources": {"memory": 2048, "cpu": 1, "disk": 512}  # Memory exceeds limit
        },
        "egress": [],
        "stage": "production"
    }
    
    if test_opa_scenario("core/greenlang/policy/bundles/run.rego", prod_violation_input, expect_allow=False):
        print("PASS: Scenario 6 passed")
        scenarios_passed += 1
    else:
        print("FAIL: Scenario 6 failed")
    
    print(f"\nScenario Results: {scenarios_passed}/{total_scenarios}")
    return scenarios_passed == total_scenarios


def test_opa_scenario(policy_file, input_data, expect_allow):
    """Helper to test an OPA scenario"""
    opa_path = Path.cwd() / "opa.exe"
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            input_file = f.name
        
        result = subprocess.run([
            str(opa_path), "eval", "-d", policy_file, "-i", input_file,
            "--format", "json", "data.greenlang.decision"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            output = json.loads(result.stdout)
            decision = output["result"][0]["expressions"][0]["value"]
            allow = decision.get("allow", False)
            reason = decision.get("reason", "No reason")
            
            success = (allow == expect_allow)
            status = "ALLOWED" if allow else "DENIED"
            expected_status = "ALLOWED" if expect_allow else "DENIED"
            
            print(f"   Result: {status} (expected {expected_status})")
            print(f"   Reason: {reason}")
            
            return success
        else:
            print(f"   Error: OPA evaluation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   Error: {e}")
        return False
    finally:
        if 'input_file' in locals():
            Path(input_file).unlink(missing_ok=True)


def main():
    """Run comprehensive policy enforcement verification"""
    print("GREENLANG POLICY GATES VERIFICATION WITH OPA")
    print("=" * 60)
    print("Testing third infrastructure verification check: Policy gates with OPA")
    print("=" * 60)
    
    results = []
    
    # Test all enforcement points
    results.append(test_install_time_enforcement())
    results.append(test_runtime_enforcement())
    results.append(test_opa_integration())
    results.append(test_enforcement_scenarios())
    
    # Final summary
    print("\n\n" + "=" * 60)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTest Categories Passed: {passed}/{total}")
    print("\nDetailed Results:")
    print("1. Pack Installation Policy Enforcement:", "PASS: PASS" if results[0] else "FAIL: FAIL")
    print("2. Pipeline Runtime Policy Enforcement:", "PASS: PASS" if results[1] else "FAIL: FAIL") 
    print("3. OPA Integration:", "PASS: PASS" if results[2] else "FAIL: FAIL")
    print("4. Comprehensive Scenarios:", "PASS: PASS" if results[3] else "FAIL: FAIL")
    
    if passed == total:
        print("\nSUCCESS: ALL POLICY GATES VERIFICATION TESTS PASSED!")
        print("\nGreenLang Policy Gates are properly implemented with OPA:")
        print("- GPL license denial works correctly")
        print("- Network policy enforcement blocks unauthorized domains") 
        print("- Resource limits are enforced")
        print("- OPA integration is functional")
        print("- Policy enforcement works at install and runtime")
        print("- Development vs production stage rules work")
        
        return True
    else:
        print(f"\nFAILED: {total - passed} test categories failed")
        print("Review policy implementation and OPA integration")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)