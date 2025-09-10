#!/usr/bin/env python
"""
Verify enterprise commands are available
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_kubernetes_backend():
    """Test K8s backend is available"""
    print("\n=== Testing Kubernetes Backend Support ===")
    try:
        from greenlang.runtime.backends import KubernetesBackend, BackendFactory
        print("[OK] Kubernetes backend module imported")
        
        # Check if k8s backend is registered
        backends = BackendFactory.list_backends()
        if "kubernetes" in backends:
            print(f"[OK] Kubernetes backend registered in factory")
            print(f"     Available backends: {backends}")
            return True
        else:
            print(f"[FAIL] Kubernetes backend not in factory")
            print(f"       Available: {backends}")
            return False
    except ImportError as e:
        print(f"[FAIL] Could not import Kubernetes backend: {e}")
        return False

def test_multitenancy():
    """Test multi-tenancy is available"""
    print("\n=== Testing Multi-tenancy Support ===")
    try:
        from greenlang.auth import TenantManager, RBACManager
        print("[OK] Multi-tenancy modules imported")
        
        # Create instances
        tenant_mgr = TenantManager()
        rbac_mgr = RBACManager()
        
        print(f"[OK] TenantManager created")
        print(f"[OK] RBACManager created with {len(rbac_mgr.roles)} default roles")
        
        # List default roles
        print("     Default roles:")
        for role_name in rbac_mgr.roles.keys():
            print(f"       - {role_name}")
        
        return True
    except ImportError as e:
        print(f"[FAIL] Could not import multi-tenancy: {e}")
        return False

def test_monitoring():
    """Test monitoring/metrics support"""
    print("\n=== Testing Monitoring & Observability ===")
    try:
        from greenlang.telemetry import (
            get_metrics_collector,
            get_health_checker,
            get_monitoring_service
        )
        print("[OK] Telemetry modules imported")
        
        # Create instances
        metrics = get_metrics_collector()
        health = get_health_checker()
        monitoring = get_monitoring_service()
        
        print(f"[OK] MetricsCollector created")
        print(f"[OK] HealthChecker created with {len(health.checks)} checks")
        print(f"[OK] MonitoringService created")
        
        # Get health status
        status = health.get_status()
        print(f"     Current health: {status.value}")
        
        return True
    except ImportError as e:
        print(f"[FAIL] Could not import monitoring: {e}")
        return False

def test_cli_commands():
    """Test CLI commands are registered"""
    print("\n=== Testing CLI Commands ===")
    try:
        from greenlang.cli.main import cli
        from click.testing import CliRunner
        
        runner = CliRunner()
        
        # Test main help
        result = runner.invoke(cli, ['--help'])
        if result.exit_code == 0:
            print("[OK] Main CLI help works")
        else:
            print(f"[FAIL] Main CLI help failed: {result.exit_code}")
            return False
        
        # Check for enterprise commands in help
        help_text = result.output
        
        commands_to_check = {
            'run': 'Run a workflow',
            'admin': 'Administrative commands',
            'tenant': 'Manage GreenLang tenants',
            'telemetry': 'Monitoring and observability'
        }
        
        found_commands = []
        missing_commands = []
        
        for cmd, desc in commands_to_check.items():
            if cmd in help_text:
                found_commands.append(cmd)
            else:
                missing_commands.append(cmd)
        
        if found_commands:
            print(f"[OK] Found commands: {', '.join(found_commands)}")
        
        if missing_commands:
            print(f"[INFO] Missing commands: {', '.join(missing_commands)}")
            print("      (May need enterprise features installed)")
        
        # Test run command with --help to check backend option
        result = runner.invoke(cli, ['run', '--help'])
        if '--backend' in result.output:
            print("[OK] Run command has --backend option")
        else:
            print("[INFO] Run command missing --backend option")
        
        return len(found_commands) > 0
        
    except Exception as e:
        print(f"[FAIL] CLI test error: {e}")
        return False

def main():
    """Run all verification tests"""
    print("=" * 60)
    print("ENTERPRISE FEATURES VERIFICATION")
    print("=" * 60)
    
    results = {
        "Kubernetes Backend": test_kubernetes_backend(),
        "Multi-tenancy": test_multitenancy(),
        "Monitoring": test_monitoring(),
        "CLI Commands": test_cli_commands()
    }
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for feature, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {feature}: {status}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\nResult: {passed}/{total} features verified")
    
    if passed == total:
        print("\nAll enterprise features are properly implemented!")
    elif passed > 0:
        print("\nSome enterprise features are available.")
        print("Check documentation for full setup requirements.")
    else:
        print("\nEnterprise features not detected.")
        print("This may be the community edition.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)