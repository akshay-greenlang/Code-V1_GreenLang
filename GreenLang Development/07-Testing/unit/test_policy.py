# -*- coding: utf-8 -*-
"""
Test policy enforcement
"""

from pathlib import Path
from greenlang.policy.enforcer import PolicyEnforcer, check_install, check_run
from greenlang.utils.net import http_get, policy_allow, add_allowed_domain
from greenlang.sdk.pipeline import Pipeline
from greenlang.sdk.context import Context


def test_network_policy():
    """Test network policy enforcement"""
    print("\n=== Testing Network Policy ===")
    
    # Test allowed domain
    try:
        policy_allow("https://github.com/test/repo", tag="test")
        print("OK: github.com allowed")
    except RuntimeError as e:
        print(f"FAIL: github.com blocked: {e}")
    
    # Test blocked domain  
    try:
        policy_allow("https://malicious-site.com/evil", tag="test")
        print("FAIL: malicious-site.com should be blocked!")
    except RuntimeError as e:
        print(f"OK: malicious-site.com blocked: {e}")
    
    # Test adding allowed domain
    add_allowed_domain("example.com")
    try:
        policy_allow("https://example.com/api", tag="test")
        print("OK: example.com allowed after adding")
    except RuntimeError as e:
        print(f"FAIL: example.com blocked: {e}")


def test_install_policy():
    """Test pack installation policy"""
    print("\n=== Testing Install Policy ===")
    
    enforcer = PolicyEnforcer()
    
    # Test pack with good metadata
    good_pack = {
        "name": "test-pack",
        "version": "1.0.0",
        "publisher": "greenlang",
        "source": "hub.greenlang.io",
        "size": 1024000,  # 1MB
        "security": {
            "sbom": "sbom.json",
            "signatures": True,
            "verified": True,
            "scan_status": "clean",
            "vulnerabilities": []
        },
        "metadata": {
            "ef_vintage": 2024
        }
    }
    
    # Test without OPA (will use fallback)
    allowed, reasons = enforcer.check_install(good_pack, ".", "add")
    if allowed:
        print("OK: Good pack allowed (fallback mode)")
    else:
        print(f"FAIL: Good pack denied: {reasons}")
    
    # Test pack with issues
    bad_pack = {
        "name": "bad-pack",
        "version": "1.0.0",
        "publisher": "unknown",
        "source": "random-site.com",
        "size": 200000000,  # 200MB - too large
        "security": {
            "sbom": None,
            "signatures": False,
            "vulnerabilities": [
                {"severity": "critical", "cve": "CVE-2024-0001"}
            ]
        },
        "metadata": {
            "ef_vintage": 2020  # Too old
        }
    }
    
    allowed, reasons = enforcer.check_install(bad_pack, ".", "add")
    if not allowed:
        print(f"OK: Bad pack denied (fallback mode): {reasons}")
    else:
        print("FAIL: Bad pack should be denied!")


def test_run_policy():
    """Test pipeline run policy"""
    print("\n=== Testing Run Policy ===")
    
    enforcer = PolicyEnforcer()
    
    # Create test pipeline
    pipeline = Pipeline(
        name="test-pipeline",
        version="1.0",
        description="Test pipeline",
        inputs={},
        steps=[],
        outputs={}
    )
    
    # Create test context
    context = Context(
        artifacts_dir=Path("out"),
        profile="dev",
        backend="local"
    )
    context.metadata = {"ef_vintage": 2024}
    
    # Test run (will use fallback without OPA)
    allowed, reasons = enforcer.check_run(pipeline, context)
    if allowed:
        print("OK: Pipeline run allowed (fallback mode)")
    else:
        print(f"FAIL: Pipeline run denied: {reasons}")
    
    # Test with production profile
    context.profile = "prod"
    allowed, reasons = enforcer.check_run(pipeline, context)
    print(f"Production profile: {'allowed' if allowed else f'denied: {reasons}'}")


def test_policy_files():
    """Test that policy files are created"""
    print("\n=== Testing Policy Files ===")
    
    bundles_dir = Path(__file__).parent / "core" / "greenlang" / "policy" / "bundles"
    
    # Check install.rego
    install_policy = bundles_dir / "install.rego"
    if install_policy.exists():
        print(f"OK: install.rego exists ({install_policy.stat().st_size} bytes)")
    else:
        print("FAIL: install.rego not found")
    
    # Check run.rego
    run_policy = bundles_dir / "run.rego"
    if run_policy.exists():
        print(f"OK: run.rego exists ({run_policy.stat().st_size} bytes)")
    else:
        print("FAIL: run.rego not found")
    
    # Try to validate with OPA if available
    from greenlang.policy.opa import _check_opa_installed, validate_policy
    
    if _check_opa_installed():
        print("\nOK: OPA is installed - validating policies...")
        
        if install_policy.exists():
            is_valid, errors = validate_policy(str(install_policy))
            if is_valid:
                print("  OK: install.rego is valid")
            else:
                print(f"  FAIL: install.rego has errors: {errors}")
        
        if run_policy.exists():
            is_valid, errors = validate_policy(str(run_policy))
            if is_valid:
                print("  OK: run.rego is valid")
            else:
                print(f"  FAIL: run.rego has errors: {errors}")
    else:
        print("\nWARNING: OPA not installed - cannot validate policy syntax")
        print("  Install OPA from: https://www.openpolicyagent.org/docs/latest/#running-opa")


if __name__ == "__main__":
    print("Testing GreenLang Policy Enforcement")
    print("=" * 40)
    
    test_network_policy()
    test_install_policy()
    test_run_policy()
    test_policy_files()
    
    print("\n" + "=" * 40)
    print("Policy enforcement test complete!")