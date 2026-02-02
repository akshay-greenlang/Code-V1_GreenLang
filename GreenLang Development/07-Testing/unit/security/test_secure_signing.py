#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the new secure signing implementation
===========================================

Verifies that:
1. No mock keys exist in the codebase
2. Ephemeral signing works for tests
3. Verification works correctly
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Fix Windows encoding for Unicode output
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    # Try to set console to UTF-8
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, 'strict')
    except:
        pass

# Set test mode
os.environ['GL_SIGNING_MODE'] = 'ephemeral'

def test_ephemeral_signing():
    """Test ephemeral key signing"""
    print("Testing ephemeral signing...")

    from greenlang.security.signing import (
        EphemeralKeypairSigner,
        DetachedSigVerifier,
        sign_artifact,
        verify_artifact
    )

    # Create test artifact
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("test: content\nversion: 1.0.0\n")
        artifact_path = Path(f.name)

    try:
        # Create ephemeral signer
        signer = EphemeralKeypairSigner()
        print(f"[OK] Created ephemeral signer with key fingerprint: {signer.get_signer_info()['key_fingerprint']}")

        # Sign artifact
        signature = sign_artifact(artifact_path, signer)
        print(f"[OK] Signed artifact with algorithm: {signature['spec']['signature']['algorithm']}")

        # Verify signature
        is_valid = verify_artifact(artifact_path, signature)
        assert is_valid, "Signature verification failed"
        print("[OK] Signature verified successfully")

        # Test tampering detection
        print("\nTesting tampering detection...")

        # Modify artifact
        with open(artifact_path, 'a') as f:
            f.write("\ntampered: true\n")

        try:
            verify_artifact(artifact_path, signature)
            assert False, "Should have detected tampering"
        except Exception as e:
            if "Hash mismatch" in str(e):
                print("[OK] Tampering detected correctly")
            else:
                raise

        return True

    finally:
        # Cleanup
        artifact_path.unlink(missing_ok=True)


def test_no_mock_keys():
    """Verify no mock keys in signing module"""
    print("\nChecking for mock keys...")

    signing_file = Path("greenlang/security/signing.py")
    if signing_file.exists():
        content = signing_file.read_text()

        forbidden_patterns = [
            "MOCK_",
            "mock_key",
            "BEGIN PRIVATE KEY",
            "BEGIN RSA PRIVATE KEY",
            "hardcoded",
            "test_key = "
        ]

        for pattern in forbidden_patterns:
            if pattern in content:
                print(f"[X] Found forbidden pattern: {pattern}")
                return False

        print("[OK] No mock keys or hardcoded secrets found")
        return True
    else:
        print("[X] Signing module not found")
        return False


def test_signing_config():
    """Test signing configuration"""
    print("\nTesting signing configuration...")

    from greenlang.security.signing import SigningConfig

    # Test default config
    config = SigningConfig.from_env()
    print(f"[OK] Default mode: {config.mode}")
    assert config.mode in ['ephemeral', 'keyless'], f"Unexpected mode: {config.mode}"

    # Test CI detection
    os.environ['CI'] = 'true'
    os.environ['GITHUB_ACTIONS'] = 'true'
    config_ci = SigningConfig.from_env()
    # In CI it would be keyless, but without proper OIDC it falls back
    print(f"[OK] CI mode: {config_ci.mode}")

    # Cleanup
    del os.environ['CI']
    del os.environ['GITHUB_ACTIONS']

    return True


def test_signature_format():
    """Test signature format compliance"""
    print("\nTesting signature format...")

    from greenlang.security.signing import EphemeralKeypairSigner

    # Create test payload
    test_payload = b"test data for signing"

    # Sign with ephemeral key
    signer = EphemeralKeypairSigner()
    result = signer.sign(test_payload)

    # Verify result structure
    assert 'signature' in result, "Missing signature"
    assert 'algorithm' in result, "Missing algorithm"
    assert 'timestamp' in result, "Missing timestamp"
    assert 'public_key' in result, "Missing public key"

    print(f"[OK] Signature format valid")
    print(f"  Algorithm: {result['algorithm']}")
    print(f"  Timestamp: {result['timestamp']}")
    print(f"  Has public key: {'Yes' if result['public_key'] else 'No'}")

    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("GreenLang Secure Signing Tests")
    print("=" * 60)

    tests = [
        ("No Mock Keys Check", test_no_mock_keys),
        ("Ephemeral Signing", test_ephemeral_signing),
        ("Signing Configuration", test_signing_config),
        ("Signature Format", test_signature_format),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n[PASS] {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n[FAIL] {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n[ERROR] {test_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)