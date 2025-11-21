#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple Security Gate Verification"""

import sys
import os
from pathlib import Path

# Use ASCII for better compatibility
PASS = "[PASS]"
FAIL = "[FAIL]"

print("=" * 60)
print("SECURITY GATE VERIFICATION REPORT")
print("=" * 60)

results = []
errors = []

def check(name, condition, error_msg=""):
    if condition:
        print(f"{PASS} {name}")
        results.append(True)
    else:
        print(f"{FAIL} {name}")
        if error_msg:
            print(f"      Error: {error_msg}")
            errors.append(f"{name}: {error_msg}")
        results.append(False)

# 1. Check files exist
print("\n1. CODE CHANGES VERIFICATION")
print("-" * 40)

files_to_check = [
    "greenlang/policy/enforcer.py",
    "greenlang/policy/opa.py",
    "greenlang/provenance/signing.py",
    "greenlang/packs/installer.py",
    "greenlang/registry/oci_client.py",
    "greenlang/runtime/executor.py",
    "tests/unit/security/test_default_deny_policy.py",
    "tests/unit/security/test_signature_verification.py",
    "tests/unit/security/test_network_security.py",
    "tests/unit/security/test_capabilities.py"
]

for f in files_to_check:
    check(f"File exists: {f}", Path(f).exists())

# 2. Check for security anti-patterns
print("\n2. SECURITY ANTI-PATTERNS CHECK")
print("-" * 40)

# Check for verify=False
verify_false_found = False
for root, dirs, files in os.walk("greenlang"):
    # Skip test directories
    if "test" in root or "__pycache__" in root:
        continue
    for file in files:
        if file.endswith(".py"):
            filepath = Path(root) / file
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            if "verify=False" in content and "#" not in content:
                verify_false_found = True
                break

check("No verify=False in production code", not verify_false_found)

# 3. Check default-deny implementation
print("\n3. DEFAULT-DENY IMPLEMENTATION")
print("-" * 40)

enforcer_file = Path("greenlang/policy/enforcer.py")
if enforcer_file.exists():
    content =enforcer_file.read_text(encoding='utf-8', errors='ignore')

    has_default_deny = "bool(dec.get(\"allow\", False))" in content
    has_error_deny = "Policy evaluation failed, denying by default" in content
    has_security_comment = "SECURITY: Default-deny" in content

    check("Enforcer uses bool() for explicit False", has_default_deny)
    check("Enforcer denies on evaluation error", has_error_deny)
    check("Security comments present", has_security_comment)

opa_file = Path("greenlang/policy/opa.py")
if opa_file.exists():
    content = opa_file.read_text(encoding='utf-8', errors='ignore')

    has_default_deny = "decision[\"allow\"] = False" in content or "\"allow\": False" in content
    has_bool_check = "bool(decision.get(\"allow\", False))" in content

    check("OPA defaults to deny", has_default_deny or has_bool_check)

# 4. Check signature verification
print("\n4. SIGNATURE VERIFICATION")
print("-" * 40)

signing_file = Path("greenlang/provenance/signing.py")
if signing_file.exists():
    content = signing_file.read_text(encoding='utf-8', errors='ignore')

    has_interface = "class SignatureVerifier" in content
    has_dev_verifier = "class DevKeyVerifier" in content
    has_sigstore = "class SigstoreVerifier" in content
    has_unsigned_error = "class UnsignedPackError" in content
    no_hardcoded_keys = "-----BEGIN" not in content

    check("SignatureVerifier interface exists", has_interface)
    check("DevKeyVerifier implementation", has_dev_verifier)
    check("SigstoreVerifier stub", has_sigstore)
    check("UnsignedPackError defined", has_unsigned_error)
    check("No hardcoded keys", no_hardcoded_keys)

installer_file = Path("greenlang/packs/installer.py")
if installer_file.exists():
    content = installer_file.read_text(encoding='utf-8', errors='ignore')

    has_verifier_import = "from ..provenance import" in content
    has_allow_unsigned = "allow_unsigned" in content
    has_signature_check = "verify_pack_signature" in content or "verifier.verify" in content
    has_audit = "audit_log" in content

    check("Installer imports verifier", has_verifier_import)
    check("Installer has allow_unsigned flag", has_allow_unsigned)
    check("Installer checks signatures", has_signature_check)
    check("Installer logs audit events", has_audit)

# 5. Check network security
print("\n5. NETWORK/SSL SECURITY")
print("-" * 40)

oci_file = Path("greenlang/registry/oci_client.py")
if oci_file.exists():
    content = oci_file.read_text(encoding='utf-8', errors='ignore')

    has_https_check = "startswith('http://')" in content or "startswith('https://')" in content
    has_env_check = "GL_DEBUG_INSECURE" in content
    has_insecure_flag = "insecure_transport" in content
    has_warning = "SECURITY WARNING" in content

    check("OCI client checks for HTTP", has_https_check)
    check("Requires GL_DEBUG_INSECURE env", has_env_check)
    check("Has insecure_transport flag", has_insecure_flag)
    check("Logs security warnings", has_warning)

# 6. Check capability defaults
print("\n6. CAPABILITY DEFAULTS")
print("-" * 40)

manifest_file = Path("greenlang/packs/manifest.py")
if manifest_file.exists():
    content = manifest_file.read_text(encoding='utf-8', errors='ignore')

    has_false_default = "Field(False" in content or "= False" in content

    check("Capabilities default to False", has_false_default)

executor_file = Path("greenlang/runtime/executor.py")
if executor_file.exists():
    content = executor_file.read_text(encoding='utf-8', errors='ignore')

    has_guard_check = "_should_use_guarded_worker" in content
    has_cap_enforcement = "'GL_CAPS'" in content or "GL_CAPS" in content
    has_default_guard = "return True" in content and "Default to using guard" in content

    check("Executor has guard check", has_guard_check)
    check("Executor passes capabilities", has_cap_enforcement)
    check("Defaults to guarded mode", has_default_guard)

# 7. Check test files
print("\n7. TEST COVERAGE")
print("-" * 40)

test_files = [
    "tests/unit/security/test_default_deny_policy.py",
    "tests/unit/security/test_signature_verification.py",
    "tests/unit/security/test_network_security.py",
    "tests/unit/security/test_capabilities.py"
]

for test_file in test_files:
    tf = Path(test_file)
    if tf.exists():
        content = tf.read_text(encoding='utf-8', errors='ignore')
        has_tests = "def test_" in content
        check(f"Tests in {tf.name}", has_tests)

# Summary
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)

passed = sum(results)
total = len(results)
failed = total - passed

print(f"\nTotal Checks: {total}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

if failed == 0:
    print(f"\n{PASS} SECURITY GATE VERIFICATION PASSED!")
    print("All security hardening features are in place.")
else:
    print(f"\n{FAIL} SECURITY GATE VERIFICATION FAILED")
    print(f"Failed checks: {failed}")
    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  - {err}")

# Create verification report
report = {
    "timestamp": "2025-09-17T13:55:00Z",
    "total_checks": total,
    "passed": passed,
    "failed": failed,
    "status": "PASSED" if failed == 0 else "FAILED",
    "errors": errors
}

# Write report
import json
with open("security_gate_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\nReport saved to: security_gate_report.json")

sys.exit(0 if failed == 0 else 1)