#!/usr/bin/env python3
"""
Test network policy enforcement during pipeline execution
"""

import sys
import json
import subprocess
import tempfile
from pathlib import Path

# Add the core module to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

def test_network_policy_enforcement():
    """Test that network policies prevent unauthorized egress"""
    print("Testing Network Policy Enforcement")
    print("-" * 40)
    
    # Create test input for pipeline with unauthorized network access
    pipeline_input = {
        "pipeline": {
            "name": "test-pipeline",
            "policy": {
                "network": ["github.com", "api.openai.com"],
                "max_memory": 2048,
                "max_cpu": 4,
                "max_disk": 2048
            },
            "resources": {
                "memory": 1024,
                "cpu": 2,
                "disk": 1024
            }
        },
        "egress": ["malicious-domain.com", "unauthorized-api.com"],  # Unauthorized domains
        "region": "us-west-2",
        "stage": "production"
    }
    
    # Use current directory OPA
    opa_path = Path.cwd() / "opa.exe"
    run_policy = Path("core/greenlang/policy/bundles/run.rego")
    
    if not run_policy.exists():
        print(f"Runtime policy not found: {run_policy}")
        return False
    
    try:
        # Create temp input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(pipeline_input, f)
            input_file = f.name
        
        # Run OPA evaluation
        cmd = [
            str(opa_path), "eval",
            "-d", str(run_policy),
            "-i", input_file,
            "--format", "json",
            "data.greenlang.decision"
        ]
        
        print(f"Testing pipeline with unauthorized domains: {pipeline_input['egress']}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"OPA evaluation failed: {result.stderr}")
            return False
        
        # Parse result
        output = json.loads(result.stdout)
        
        if output.get("result") and len(output["result"]) > 0:
            expressions = output["result"][0].get("expressions", [])
            if expressions and len(expressions) > 0:
                decision = expressions[0].get("value", {})
                
                allow = decision.get("allow", False)
                reason = decision.get("reason", "No reason provided")
                unauthorized = decision.get("unauthorized_egress", [])
                
                print(f"Decision: allow={allow}")
                print(f"Reason: {reason}")
                print(f"Unauthorized domains detected: {unauthorized}")
                
                if not allow and "egress" in reason.lower():
                    print("SUCCESS: Unauthorized network access properly blocked!")
                    return True
                elif not allow:
                    print(f"SUCCESS: Pipeline denied, reason: {reason}")
                    return True
                else:
                    print("FAIL: Unauthorized network access was allowed")
                    return False
        
        print("No valid decision found in OPA output")
        return False
        
    except Exception as e:
        print(f"Error testing network policy: {e}")
        return False
    finally:
        if 'input_file' in locals():
            Path(input_file).unlink(missing_ok=True)


def test_authorized_network_access():
    """Test that authorized network access is permitted"""
    print("\nTesting Authorized Network Access")
    print("-" * 40)
    
    # Create test input for pipeline with authorized network access
    pipeline_input = {
        "pipeline": {
            "name": "test-pipeline",
            "policy": {
                "network": ["github.com", "api.openai.com"],
                "max_memory": 2048,
                "max_cpu": 4,
                "max_disk": 2048
            },
            "resources": {
                "memory": 1024,
                "cpu": 2,
                "disk": 1024
            }
        },
        "egress": ["github.com"],  # Authorized domain
        "region": "us-west-2",
        "stage": "production"
    }
    
    opa_path = Path.cwd() / "opa.exe"
    run_policy = Path("core/greenlang/policy/bundles/run.rego")
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(pipeline_input, f)
            input_file = f.name
        
        cmd = [
            str(opa_path), "eval",
            "-d", str(run_policy),
            "-i", input_file,
            "--format", "json",
            "data.greenlang.decision"
        ]
        
        print(f"Testing pipeline with authorized domain: {pipeline_input['egress']}")
        
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
                
                print(f"Decision: allow={allow}")
                print(f"Reason: {reason}")
                
                if allow:
                    print("SUCCESS: Authorized network access permitted!")
                    return True
                else:
                    print(f"FAIL: Authorized network access denied: {reason}")
                    return False
        
        return False
        
    except Exception as e:
        print(f"Error testing authorized network: {e}")
        return False
    finally:
        if 'input_file' in locals():
            Path(input_file).unlink(missing_ok=True)


def test_resource_limits():
    """Test resource limit enforcement"""
    print("\nTesting Resource Limit Enforcement")
    print("-" * 40)
    
    # Create test input that exceeds resource limits
    pipeline_input = {
        "pipeline": {
            "name": "test-pipeline",
            "policy": {
                "network": ["github.com"],
                "max_memory": 1024,  # Limit: 1GB
                "max_cpu": 2,        # Limit: 2 cores
                "max_disk": 1024     # Limit: 1GB
            },
            "resources": {
                "memory": 2048,      # Request: 2GB (exceeds limit)
                "cpu": 4,           # Request: 4 cores (exceeds limit)
                "disk": 512         # Request: 512MB (within limit)
            }
        },
        "egress": [],
        "region": "us-west-2",
        "stage": "production"
    }
    
    opa_path = Path.cwd() / "opa.exe"
    run_policy = Path("core/greenlang/policy/bundles/run.rego")
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(pipeline_input, f)
            input_file = f.name
        
        cmd = [
            str(opa_path), "eval",
            "-d", str(run_policy),
            "-i", input_file,
            "--format", "json",
            "data.greenlang.decision"
        ]
        
        print("Testing pipeline with excessive resource requests...")
        
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
                
                print(f"Decision: allow={allow}")
                print(f"Reason: {reason}")
                
                if not allow and ("resource" in reason.lower() or "limit" in reason.lower()):
                    print("SUCCESS: Excessive resource usage properly blocked!")
                    return True
                elif not allow:
                    print(f"SUCCESS: Pipeline denied, reason: {reason}")
                    return True
                else:
                    print("FAIL: Excessive resource usage was allowed")
                    return False
        
        return False
        
    except Exception as e:
        print(f"Error testing resource limits: {e}")
        return False
    finally:
        if 'input_file' in locals():
            Path(input_file).unlink(missing_ok=True)


def main():
    """Run network policy enforcement tests"""
    print("GreenLang Network Policy Enforcement Tests")
    print("=" * 50)
    
    results = []
    
    # Test unauthorized network access blocking
    results.append(test_network_policy_enforcement())
    
    # Test authorized network access
    results.append(test_authorized_network_access())
    
    # Test resource limit enforcement
    results.append(test_resource_limits())
    
    # Summary
    print("\n" + "=" * 50)
    print("NETWORK POLICY TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("ALL NETWORK POLICY TESTS PASSED!")
        print("- Unauthorized network access is properly blocked")
        print("- Authorized network access is permitted")
        print("- Resource limits are enforced")
    else:
        print("Some network policy tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)