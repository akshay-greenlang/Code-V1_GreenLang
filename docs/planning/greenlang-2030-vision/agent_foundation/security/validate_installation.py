# -*- coding: utf-8 -*-
"""
Quick validation script to test input validation framework.

Run this to verify the framework is working correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_basic_validation():
    """Test basic validation functions."""
    from security.input_validation import InputValidator

    print("=" * 80)
    print("INPUT VALIDATION FRAMEWORK - INSTALLATION TEST")
    print("=" * 80)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Alphanumeric validation
    print("\n1. Testing alphanumeric validation...")
    try:
        result = InputValidator.validate_alphanumeric("test-123", "field")
        assert result == "test-123"
        print("   ✓ Valid input accepted")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        tests_failed += 1

    try:
        InputValidator.validate_alphanumeric("test@123", "field")
        print("   ✗ Invalid input not rejected!")
        tests_failed += 1
    except ValueError:
        print("   ✓ Invalid input rejected")
        tests_passed += 1

    # Test 2: UUID validation
    print("\n2. Testing UUID validation...")
    try:
        result = InputValidator.validate_uuid(
            "123e4567-e89b-12d3-a456-426614174000", "user_id"
        )
        assert "123e4567" in result
        print("   ✓ Valid UUID accepted")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        tests_failed += 1

    try:
        InputValidator.validate_uuid("not-a-uuid", "user_id")
        print("   ✗ Invalid UUID not rejected!")
        tests_failed += 1
    except ValueError:
        print("   ✓ Invalid UUID rejected")
        tests_passed += 1

    # Test 3: Email validation
    print("\n3. Testing email validation...")
    try:
        result = InputValidator.validate_email("user@example.com")
        assert result == "user@example.com"
        print("   ✓ Valid email accepted")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        tests_failed += 1

    try:
        InputValidator.validate_email("not-an-email")
        print("   ✗ Invalid email not rejected!")
        tests_failed += 1
    except ValueError:
        print("   ✓ Invalid email rejected")
        tests_passed += 1

    # Test 4: SQL injection detection
    print("\n4. Testing SQL injection detection...")
    try:
        InputValidator.validate_no_sql_injection("test' OR '1'='1", "field")
        print("   ✗ SQL injection not detected!")
        tests_failed += 1
    except ValueError:
        print("   ✓ SQL injection detected")
        tests_passed += 1

    # Test 5: Command injection detection
    print("\n5. Testing command injection detection...")
    try:
        InputValidator.validate_no_command_injection("test; rm -rf /", "field")
        print("   ✗ Command injection not detected!")
        tests_failed += 1
    except ValueError:
        print("   ✓ Command injection detected")
        tests_passed += 1

    # Test 6: Field name whitelisting
    print("\n6. Testing field name whitelisting...")
    try:
        result = InputValidator.validate_field_name("tenant_id")
        assert result == "tenant_id"
        print("   ✓ Whitelisted field accepted")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        tests_failed += 1

    try:
        InputValidator.validate_field_name("malicious_field")
        print("   ✗ Non-whitelisted field not rejected!")
        tests_failed += 1
    except ValueError:
        print("   ✓ Non-whitelisted field rejected")
        tests_passed += 1

    # Test 7: Pydantic models
    print("\n7. Testing Pydantic models...")
    try:
        from security.input_validation import TenantIdModel
        model = TenantIdModel(tenant_id="tenant-123")
        assert model.tenant_id == "tenant-123"
        print("   ✓ TenantIdModel works")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        tests_failed += 1

    # Test 8: SafeQueryInput
    print("\n8. Testing SafeQueryInput...")
    try:
        from security.input_validation import SafeQueryInput
        query = SafeQueryInput(
            field="tenant_id",
            value="tenant-123",
            operator="="
        )
        assert query.field == "tenant_id"
        print("   ✓ SafeQueryInput works")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        tests_failed += 1

    # Test 9: SecureQueryBuilder
    print("\n9. Testing SecureQueryBuilder...")
    try:
        from database.postgres_manager_secure import SecureQueryBuilder
        from security.input_validation import SafeQueryInput

        builder = SecureQueryBuilder("agents")
        filters = [SafeQueryInput(field="tenant_id", value="tenant-123", operator="=")]
        query, params = builder.build_select(filters=filters, limit=10, offset=0)

        assert "$1" in query
        assert params == ["tenant-123"]
        print("   ✓ SecureQueryBuilder works")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        tests_failed += 1

    # Test 10: SecureCommandExecutor
    print("\n10. Testing SecureCommandExecutor...")
    try:
        from factory.deployment_secure import SecureCommandExecutor

        executor = SecureCommandExecutor()

        # Test command validation
        assert "kubectl" in executor.ALLOWED_KUBECTL_COMMANDS
        assert "get" in executor.ALLOWED_KUBECTL_COMMANDS

        print("   ✓ SecureCommandExecutor initialized")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Success Rate: {(tests_passed / (tests_passed + tests_failed) * 100):.1f}%")

    if tests_failed == 0:
        print("\n✓ ALL TESTS PASSED - Framework installed correctly!")
        print("=" * 80)
        return 0
    else:
        print(f"\n✗ {tests_failed} TESTS FAILED - Please review errors above")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(test_basic_validation())
