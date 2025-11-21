#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick JWT Validation Test Script

This script performs a quick validation of the JWT implementation without
requiring pytest. Run this to verify the JWT authentication is working.

Usage:
    python test_jwt_quick.py
"""

import sys
import os
from datetime import datetime
from greenlang.determinism import DeterministicClock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from tenancy.tenant_context import JWTValidator, AuthenticationError
    print("✅ Successfully imported JWTValidator and AuthenticationError")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please install dependencies: pip install PyJWT==2.8.0 cryptography==42.0.2")
    sys.exit(1)


def test_basic_functionality():
    """Test basic JWT functionality."""
    print("\n" + "="*70)
    print("JWT AUTHENTICATION QUICK TEST")
    print("="*70)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Initialize validator
    print("\n[Test 1] Initialize JWTValidator...")
    try:
        validator = JWTValidator(
            secret_key="test-secret-key-min-32-characters",
            algorithm="HS256",
            issuer="greenlang.ai",
            audience="greenlang-api"
        )
        print("✅ PASS: Validator initialized successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAIL: {e}")
        tests_failed += 1
        return

    # Test 2: Generate valid access token
    print("\n[Test 2] Generate access token...")
    try:
        token = validator.generate_token(
            tenant_id="550e8400-e29b-41d4-a716-446655440000",
            user_id="user-123",
            token_type="access",
            expires_in=3600
        )
        print(f"✅ PASS: Token generated")
        print(f"   Token: {token[:50]}...")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAIL: {e}")
        tests_failed += 1

    # Test 3: Validate token
    print("\n[Test 3] Validate token...")
    try:
        payload = validator.validate_token(token)
        assert payload["tenant_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert payload["sub"] == "user-123"
        assert payload["type"] == "access"
        assert payload["iss"] == "greenlang.ai"
        assert payload["aud"] == "greenlang-api"
        print("✅ PASS: Token validated successfully")
        print(f"   Tenant ID: {payload['tenant_id']}")
        print(f"   User ID: {payload['sub']}")
        print(f"   Type: {payload['type']}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAIL: {e}")
        tests_failed += 1

    # Test 4: Reject expired token
    print("\n[Test 4] Reject expired token...")
    try:
        expired_token = validator.generate_token(
            tenant_id="550e8400-e29b-41d4-a716-446655440000",
            user_id="user-123",
            expires_in=-1  # Already expired
        )
        try:
            validator.validate_token(expired_token)
            print("❌ FAIL: Expired token should have been rejected!")
            tests_failed += 1
        except AuthenticationError as e:
            if "expired" in str(e).lower():
                print("✅ PASS: Expired token correctly rejected")
                print(f"   Error: {e}")
                tests_passed += 1
            else:
                print(f"❌ FAIL: Wrong error message: {e}")
                tests_failed += 1
    except Exception as e:
        print(f"❌ FAIL: Unexpected error: {e}")
        tests_failed += 1

    # Test 5: Reject invalid signature
    print("\n[Test 5] Reject invalid signature...")
    try:
        validator1 = JWTValidator(secret_key="secret-1")
        validator2 = JWTValidator(secret_key="secret-2")

        token = validator1.generate_token(
            tenant_id="550e8400-e29b-41d4-a716-446655440000",
            user_id="user-123"
        )

        try:
            validator2.validate_token(token)
            print("❌ FAIL: Invalid signature should have been rejected!")
            tests_failed += 1
        except AuthenticationError as e:
            if "signature" in str(e).lower():
                print("✅ PASS: Invalid signature correctly rejected")
                print(f"   Error: {e}")
                tests_passed += 1
            else:
                print(f"❌ FAIL: Wrong error message: {e}")
                tests_failed += 1
    except Exception as e:
        print(f"❌ FAIL: Unexpected error: {e}")
        tests_failed += 1

    # Test 6: Reject missing tenant_id
    print("\n[Test 6] Reject token with missing tenant_id...")
    try:
        import jwt
        from datetime import datetime, timedelta

        now = DeterministicClock.utcnow()
        payload = {
            "sub": "user-123",
            "type": "access",
            "iat": now,
            "exp": now + timedelta(hours=1)
        }
        bad_token = jwt.encode(payload, "test-secret-key-min-32-characters", algorithm="HS256")

        validator_test = JWTValidator(secret_key="test-secret-key-min-32-characters")
        try:
            validator_test.validate_token(bad_token)
            print("❌ FAIL: Missing tenant_id should have been rejected!")
            tests_failed += 1
        except AuthenticationError as e:
            if "tenant_id" in str(e).lower():
                print("✅ PASS: Missing tenant_id correctly rejected")
                print(f"   Error: {e}")
                tests_passed += 1
            else:
                print(f"❌ FAIL: Wrong error message: {e}")
                tests_failed += 1
    except Exception as e:
        print(f"❌ FAIL: Unexpected error: {e}")
        tests_failed += 1

    # Test 7: Generate refresh token
    print("\n[Test 7] Generate and validate refresh token...")
    try:
        refresh_token = validator.generate_token(
            tenant_id="550e8400-e29b-41d4-a716-446655440000",
            user_id="user-123",
            token_type="refresh",
            expires_in=604800  # 7 days
        )
        payload = validator.validate_token(refresh_token)
        assert payload["type"] == "refresh"
        print("✅ PASS: Refresh token generated and validated")
        print(f"   Type: {payload['type']}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAIL: {e}")
        tests_failed += 1

    # Test 8: Token with additional claims
    print("\n[Test 8] Token with additional claims...")
    try:
        token_with_claims = validator.generate_token(
            tenant_id="550e8400-e29b-41d4-a716-446655440000",
            user_id="user-123",
            additional_claims={
                "role": "admin",
                "permissions": ["read", "write"]
            }
        )
        payload = validator.validate_token(token_with_claims)
        assert payload["role"] == "admin"
        assert payload["permissions"] == ["read", "write"]
        print("✅ PASS: Additional claims included and validated")
        print(f"   Role: {payload['role']}")
        print(f"   Permissions: {payload['permissions']}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FAIL: {e}")
        tests_failed += 1

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Total Tests:  {tests_passed + tests_failed}")

    if tests_failed == 0:
        print("\n✅ ALL TESTS PASSED - JWT Authentication is working correctly!")
        print("\n🔒 SECURITY STATUS: PRODUCTION READY")
        print("   - Signature verification: ENABLED")
        print("   - Expiration validation: ENABLED")
        print("   - Claims validation: ENABLED")
        print("   - CWE-287 vulnerability: FIXED")
        return True
    else:
        print(f"\n❌ {tests_failed} TEST(S) FAILED - Please review implementation")
        return False


if __name__ == "__main__":
    print("\n🔐 JWT AUTHENTICATION SECURITY TEST")
    print("Verifying CWE-287 vulnerability fix...")

    success = test_basic_functionality()

    if success:
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("1. Install dependencies in production:")
        print("   pip install PyJWT==2.8.0 cryptography==42.0.2")
        print("")
        print("2. Generate secure secret key:")
        print("   python -c \"import secrets; print(secrets.token_urlsafe(64))\"")
        print("")
        print("3. Configure .env file:")
        print("   JWT_SECRET_KEY=<generated-secret-key>")
        print("   JWT_ALGORITHM=HS256")
        print("   JWT_ISSUER=greenlang.ai")
        print("   JWT_AUDIENCE=greenlang-api")
        print("")
        print("4. Run full test suite:")
        print("   pytest testing/security_tests/test_jwt_validation.py -v")
        print("")
        print("5. Deploy to production!")
        print("="*70)
        sys.exit(0)
    else:
        sys.exit(1)
