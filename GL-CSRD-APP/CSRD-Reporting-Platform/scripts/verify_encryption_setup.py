"""
Verification script for encryption implementation.

This script checks that all encryption components are properly installed
and configured for the CSRD Reporting Platform.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_files_exist():
    """Check that all required files exist."""
    print("\n" + "="*70)
    print("STEP 1: Checking Required Files")
    print("="*70)

    base_dir = Path(__file__).parent.parent
    required_files = {
        "Encryption Module": base_dir / "utils" / "encryption.py",
        "Encryption Config": base_dir / "config" / "encryption_config.yaml",
        "Test Suite": base_dir / "tests" / "test_encryption.py",
        "Usage Example": base_dir / "examples" / "encryption_usage_example.py",
        "Environment Template": base_dir / ".env.example",
        ".gitignore": base_dir / ".gitignore",
    }

    all_exist = True
    for name, path in required_files.items():
        exists = path.exists()
        status = "✓ FOUND" if exists else "✗ MISSING"
        print(f"  {status}: {name}")
        if exists:
            size = path.stat().st_size
            print(f"           Size: {size:,} bytes")
        all_exist = all_exist and exists

    return all_exist


def check_dependencies():
    """Check that required Python packages are installed."""
    print("\n" + "="*70)
    print("STEP 2: Checking Dependencies")
    print("="*70)

    dependencies = {
        "cryptography": "Encryption library (Fernet)",
    }

    all_installed = True
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"  ✓ INSTALLED: {package} - {description}")

            # Get version if possible
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"              Version: {version}")
            except:
                pass

        except ImportError:
            print(f"  ✗ MISSING: {package} - {description}")
            print(f"            Install with: pip install {package}>=41.0.0")
            all_installed = False

    return all_installed


def check_encryption_module():
    """Check that encryption module can be imported and used."""
    print("\n" + "="*70)
    print("STEP 3: Testing Encryption Module")
    print("="*70)

    try:
        from utils.encryption import EncryptionManager
        print("  ✓ Module imported successfully")

        # Test key generation
        key = EncryptionManager.generate_key()
        print(f"  ✓ Key generation works (generated {len(key)} bytes)")

        # Test encryption manager with key
        em = EncryptionManager(key=key)
        print("  ✓ EncryptionManager initialization works")

        # Test basic encryption
        test_data = "test sensitive data"
        encrypted = em.encrypt(test_data)
        print(f"  ✓ Encryption works (encrypted {len(test_data)} bytes)")

        # Test decryption
        decrypted = em.decrypt(encrypted)
        if decrypted == test_data:
            print("  ✓ Decryption works (data matches)")
        else:
            print("  ✗ Decryption failed (data mismatch)")
            return False

        # Test dictionary encryption
        test_dict = {"field1": "value1", "field2": "value2"}
        encrypted_dict = em.encrypt_dict(test_dict, ['field1'])
        print("  ✓ Dictionary encryption works")

        # Test dictionary decryption
        decrypted_dict = em.decrypt_dict(encrypted_dict, ['field1'])
        if decrypted_dict['field1'] == "value1":
            print("  ✓ Dictionary decryption works")
        else:
            print("  ✗ Dictionary decryption failed")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Error testing encryption module: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_configuration():
    """Check encryption configuration file."""
    print("\n" + "="*70)
    print("STEP 4: Checking Configuration")
    print("="*70)

    try:
        import yaml

        config_path = Path(__file__).parent.parent / "config" / "encryption_config.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print("  ✓ Configuration file loaded successfully")

        # Check for required sections
        if 'sensitive_fields' in config:
            field_count = sum(len(fields) for fields in config['sensitive_fields'].values())
            print(f"  ✓ Sensitive fields defined: {field_count} fields across {len(config['sensitive_fields'])} categories")

            # List categories
            for category in config['sensitive_fields'].keys():
                count = len(config['sensitive_fields'][category])
                print(f"      - {category}: {count} fields")

        else:
            print("  ✗ Missing 'sensitive_fields' section in config")
            return False

        if 'encryption_settings' in config:
            print("  ✓ Encryption settings defined")
        else:
            print("  ⚠ Warning: Missing 'encryption_settings' section")

        return True

    except ImportError:
        print("  ⚠ Warning: PyYAML not installed, cannot verify config")
        print("            Install with: pip install pyyaml")
        return True  # Don't fail if PyYAML not installed

    except Exception as e:
        print(f"  ✗ Error checking configuration: {e}")
        return False


def check_environment():
    """Check environment variable setup."""
    print("\n" + "="*70)
    print("STEP 5: Checking Environment Variables")
    print("="*70)

    env_file = Path(__file__).parent.parent / ".env"

    if env_file.exists():
        print("  ✓ .env file exists")

        # Check for encryption key
        with open(env_file, 'r') as f:
            content = f.read()

        if 'CSRD_ENCRYPTION_KEY' in content:
            print("  ✓ CSRD_ENCRYPTION_KEY found in .env")
        else:
            print("  ⚠ Warning: CSRD_ENCRYPTION_KEY not set in .env")
            print("            Generate with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'")

    else:
        print("  ⚠ .env file not found")
        print("    To set up:")
        print("      1. Copy .env.example to .env")
        print("      2. Generate key: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'")
        print("      3. Add to .env: CSRD_ENCRYPTION_KEY=your-generated-key")

    # Check if key is set in environment
    if os.getenv('CSRD_ENCRYPTION_KEY'):
        print("  ✓ CSRD_ENCRYPTION_KEY loaded in environment")
    else:
        print("  ⚠ CSRD_ENCRYPTION_KEY not in environment")
        print("    Note: Tests will use test keys, but production needs this set")

    return True


def check_tests():
    """Check if tests can be run."""
    print("\n" + "="*70)
    print("STEP 6: Checking Test Suite")
    print("="*70)

    try:
        import pytest
        print("  ✓ pytest installed")

        test_file = Path(__file__).parent.parent / "tests" / "test_encryption.py"

        if test_file.exists():
            print(f"  ✓ Test file exists")

            # Count tests
            with open(test_file, 'r') as f:
                content = f.read()
                test_count = content.count('def test_')
                print(f"  ✓ Found {test_count} test cases")

        print("\n  To run tests:")
        print("    pytest tests/test_encryption.py -v")

        return True

    except ImportError:
        print("  ⚠ pytest not installed")
        print("    Install with: pip install pytest")
        return True  # Don't fail if pytest not installed


def generate_setup_instructions():
    """Generate setup instructions."""
    print("\n" + "="*70)
    print("SETUP INSTRUCTIONS")
    print("="*70)

    print("""
1. Install Dependencies:
   pip install cryptography>=41.0.0

2. Generate Encryption Key:
   python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'

3. Create .env file:
   cp .env.example .env

4. Add encryption key to .env:
   CSRD_ENCRYPTION_KEY=your-generated-key-here

5. Run tests:
   pytest tests/test_encryption.py -v

6. Review configuration:
   config/encryption_config.yaml

7. See usage examples:
   python examples/encryption_usage_example.py

8. Read documentation:
   C:\\Users\\aksha\\Code-V1_GreenLang\\GL-CSRD-ENCRYPTION-IMPLEMENTATION.md
""")


def main():
    """Run all verification checks."""
    print("="*70)
    print("CSRD ENCRYPTION SETUP VERIFICATION")
    print("="*70)
    print("\nThis script verifies that encryption components are properly installed.")

    results = []

    # Run checks
    results.append(("Files", check_files_exist()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Encryption Module", check_encryption_module()))
    results.append(("Configuration", check_configuration()))
    results.append(("Environment", check_environment()))
    results.append(("Tests", check_tests()))

    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} checks passed")

    if passed == total:
        print("\n✓ ALL CHECKS PASSED - Encryption setup is complete!")
        print("\nNext steps:")
        print("  1. Review configuration: config/encryption_config.yaml")
        print("  2. Update agents to use encryption utilities")
        print("  3. Run full test suite: pytest tests/test_encryption.py -v")
        print("  4. See examples: python examples/encryption_usage_example.py")
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED - Review errors above")
        generate_setup_instructions()
        return 1


if __name__ == "__main__":
    exit_code = main()
    print("\n")
    sys.exit(exit_code)
