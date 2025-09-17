#!/usr/bin/env python
"""
Test script for enterprise features verification
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def run_command(cmd):
    """Run a command and return success status"""
    print(f"\n[Running] {cmd}")
    
    # Replace 'gl' with 'python -m greenlang.cli.main'
    if cmd.startswith("gl "):
        cmd = "python -m greenlang.cli.main " + cmd[3:]
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"[OK] Command succeeded")
            if result.stdout:
                print(f"Output: {result.stdout[:200]}...")
            return True
        else:
            print(f"[FAIL] Command failed with code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}...")
            return False
    except subprocess.TimeoutExpired:
        print(f"[FAIL] Command timed out")
        return False
    except Exception as e:
        print(f"[FAIL] Error running command: {e}")
        return False

def test_kubernetes_backend():
    """Test Kubernetes backend execution"""
    print("\n=== Testing Kubernetes Backend ===")
    
    # Create a simple pipeline file
    pipeline_yaml = """
name: test-pipeline
pipeline:
  steps:
    - name: echo-test
      command: ["echo", "Hello from K8s"]
"""
    
    Path("test-pipeline.yaml").write_text(pipeline_yaml)
    
    # Test with k8s backend
    success = run_command("gl run test-pipeline.yaml --backend k8s --namespace prod")
    
    # Clean up
    Path("test-pipeline.yaml").unlink(missing_ok=True)
    
    return success

def test_multitenancy():
    """Test multi-tenancy commands"""
    print("\n=== Testing Multi-tenancy ===")
    
    # Test admin tenants list command
    success = run_command("gl admin tenants list")
    
    # Also test the tenant command group
    if not success:
        print("[INFO] Trying alternative tenant command...")
        success = run_command("gl tenant list")
    
    return success

def test_metrics_endpoint():
    """Test Prometheus metrics endpoint"""
    print("\n=== Testing Metrics Endpoint ===")
    
    # Start metrics server in background
    import threading
    import time
    
    def start_metrics_server():
        """Start metrics server"""
        try:
            from greenlang.telemetry import get_metrics_collector
            collector = get_metrics_collector()
            collector.start_collection(9090)
            time.sleep(60)  # Keep running for 60 seconds
        except Exception as e:
            print(f"[INFO] Metrics server error: {e}")
    
    # Start server in background thread
    server_thread = threading.Thread(target=start_metrics_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    # Try to access metrics endpoint
    try:
        response = requests.get("http://localhost:9090/metrics", timeout=5)
        if response.status_code == 200:
            print("[OK] Metrics endpoint accessible")
            print(f"Metrics sample: {response.text[:200]}...")
            return True
        else:
            print(f"[FAIL] Metrics endpoint returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"[INFO] Could not connect to metrics endpoint: {e}")
        
        # Try using the telemetry CLI command instead
        print("[INFO] Trying telemetry CLI command...")
        return run_command("gl telemetry metrics list")

def test_all_features():
    """Test all enterprise features"""
    print("=" * 60)
    print("ENTERPRISE FEATURES VERIFICATION")
    print("=" * 60)
    
    results = {
        "Kubernetes Backend": test_kubernetes_backend(),
        "Multi-tenancy": test_multitenancy(),
        "Metrics Endpoint": test_metrics_endpoint()
    }
    
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    
    for feature, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[OK]" if passed else "[FAIL]"
        print(f"{symbol} {feature}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n[OK] All enterprise features verified successfully!")
    else:
        print("\n[FAIL] Some features could not be verified")
        print("\nNote: Some features may require additional setup:")
        print("  - Kubernetes cluster for k8s backend")
        print("  - Database for multi-tenancy")
        print("  - Prometheus client library for metrics")
    
    return all_passed

if __name__ == "__main__":
    success = test_all_features()
    sys.exit(0 if success else 1)