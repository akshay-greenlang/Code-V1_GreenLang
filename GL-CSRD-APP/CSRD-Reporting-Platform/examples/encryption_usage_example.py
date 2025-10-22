"""
Example: How to integrate encryption into CSRD agents.

This demonstrates the proper usage of the encryption utilities
for protecting sensitive ESG data in the CSRD Reporting Platform.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.encryption import get_encryption_manager, EncryptionManager


def example_reporting_agent():
    """Example: Reporting Agent with encryption."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Reporting Agent - Save and Load Encrypted Report")
    print("="*70)

    # Get encryption manager
    em = get_encryption_manager()

    # Sample ESG report data
    report_data = {
        "company_name": "Green Manufacturing GmbH",
        "reporting_year": 2024,
        "country": "DE",
        "revenue_eur": 250000000,
        "lei_code": "549300ABCDEF1234567890",
        "tax_id": "DE987654321",
        "employees": 1200,
        "ghg_emissions_scope1": 5000,
        "ghg_emissions_scope2": 8000,
        "renewable_energy_percentage": 45.5
    }

    # Define sensitive fields
    sensitive_fields = [
        'revenue_eur',
        'lei_code',
        'tax_id'
    ]

    print("\nOriginal Report Data:")
    for key, value in report_data.items():
        print(f"  {key}: {value}")

    # ENCRYPT before saving
    encrypted_report = em.encrypt_dict(report_data, sensitive_fields)

    print("\nEncrypted Report Data (what gets stored):")
    for key, value in encrypted_report.items():
        if key in sensitive_fields:
            print(f"  {key}: {value[:50]}... [ENCRYPTED]")
        elif '_encrypted' in key:
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    # Simulate saving to database/file
    # save_to_database(encrypted_report)

    # DECRYPT when loading
    decrypted_report = em.decrypt_dict(encrypted_report, sensitive_fields)

    print("\nDecrypted Report Data (after loading):")
    for key, value in decrypted_report.items():
        if '_encrypted' not in key:
            print(f"  {key}: {value}")

    print("\n✓ Sensitive data protected in storage, readable in application")


def example_intake_agent():
    """Example: Intake Agent with encryption."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Intake Agent - Process Company Profile")
    print("="*70)

    em = get_encryption_manager()

    # Company profile from user input
    company_profile = {
        "company_name": "Sustainable Tech AG",
        "lei_code": "529900DEMO00000000001",
        "tax_id": "CH123456789",
        "iban": "CH9300762011623852957",
        "industry": "Technology",
        "country": "Switzerland",
        "founded_year": 2015
    }

    sensitive_fields = ['lei_code', 'tax_id', 'iban']

    print("\nInput Company Profile:")
    for key, value in company_profile.items():
        print(f"  {key}: {value}")

    # Encrypt sensitive fields before storage
    encrypted_profile = em.encrypt_dict(company_profile, sensitive_fields)

    print("\nStored Company Profile (encrypted):")
    for key, value in encrypted_profile.items():
        if key in sensitive_fields:
            print(f"  {key}: {value[:40]}... [ENCRYPTED]")
        elif '_encrypted' not in key:
            print(f"  {key}: {value}")

    print("\n✓ Regulatory identifiers encrypted for GDPR compliance")


def example_calculator_agent():
    """Example: Calculator Agent with encryption."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Calculator Agent - Financial Metrics")
    print("="*70)

    em = get_encryption_manager()

    # Financial metrics calculation results
    financial_metrics = {
        "reporting_period": "2024-Q4",
        "revenue_eur": 125000000,
        "cost_of_goods_sold": 75000000,
        "gross_profit": 50000000,
        "operating_expenses": 30000000,
        "ebitda": 20000000,
        "net_income": 15000000,
        "profit_margin_pct": 12.0,
        "roi_pct": 18.5
    }

    sensitive_fields = [
        'revenue_eur',
        'cost_of_goods_sold',
        'gross_profit',
        'net_income',
        'profit_margin_pct'
    ]

    print("\nCalculated Financial Metrics:")
    for key, value in financial_metrics.items():
        print(f"  {key}: {value}")

    # Encrypt before storage
    encrypted_metrics = em.encrypt_dict(financial_metrics, sensitive_fields)

    print("\nEncrypted Financial Metrics (for storage):")
    for key, value in encrypted_metrics.items():
        if key in sensitive_fields:
            print(f"  {key}: [ENCRYPTED]")
        elif '_encrypted' not in key:
            print(f"  {key}: {value}")

    print("\n✓ Financial data encrypted for SOX compliance")


def example_materiality_agent():
    """Example: Materiality Agent with encryption."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Materiality Agent - Stakeholder Assessment")
    print("="*70)

    em = get_encryption_manager()

    # Materiality assessment data
    materiality_data = {
        "topic": "Climate Change Adaptation",
        "esrs_reference": "E1",
        "financial_materiality_score": 8.5,
        "impact_materiality_score": 9.2,
        "stakeholder_feedback": "Confidential: Board expressed concerns about transition risks in coal portfolio",
        "confidential_impact_data": "Internal analysis shows €50M potential stranded assets by 2030",
        "public_statement": "Company is committed to net-zero by 2050",
        "material": True
    }

    sensitive_fields = [
        'stakeholder_feedback',
        'confidential_impact_data'
    ]

    print("\nMateriality Assessment Data:")
    for key, value in materiality_data.items():
        if key in sensitive_fields:
            print(f"  {key}: [CONTAINS CONFIDENTIAL INFO]")
        else:
            print(f"  {key}: {value}")

    # Encrypt confidential assessments
    encrypted_assessment = em.encrypt_dict(materiality_data, sensitive_fields)

    print("\nStored Assessment (confidential data encrypted):")
    for key in encrypted_assessment:
        if key in sensitive_fields:
            print(f"  {key}: [ENCRYPTED]")
        elif '_encrypted' in key:
            continue
        else:
            print(f"  {key}: {materiality_data[key]}")

    print("\n✓ Confidential stakeholder feedback protected")


def example_key_generation():
    """Example: Generate encryption key."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Generate Encryption Key")
    print("="*70)

    # Generate a new key
    new_key = EncryptionManager.generate_key()

    print("\nGenerated Encryption Key:")
    print(f"  Key (base64): {new_key.decode()}")
    print(f"  Length: {len(new_key)} bytes")

    print("\nAdd to .env file:")
    print(f"  CSRD_ENCRYPTION_KEY={new_key.decode()}")

    print("\n✓ Use this key for CSRD_ENCRYPTION_KEY environment variable")


def example_error_handling():
    """Example: Error handling for encryption."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Error Handling")
    print("="*70)

    em = get_encryption_manager()

    # Example 1: Decrypting with wrong key
    print("\n1. Wrong Key Error:")
    try:
        # Create manager with different key
        different_key = EncryptionManager.generate_key()
        em2 = EncryptionManager(key=different_key)

        # Encrypt with first key
        encrypted = em.encrypt("sensitive data")

        # Try to decrypt with second key (will fail)
        em2.decrypt(encrypted)
    except Exception as e:
        print(f"  ✓ Error caught: {type(e).__name__}")
        print(f"  Message: Data encrypted with different key cannot be decrypted")

    # Example 2: Handling None values
    print("\n2. None Value Handling:")
    data_with_none = {
        "field1": "value1",
        "field2": None,
        "field3": "value3"
    }
    encrypted = em.encrypt_dict(data_with_none, ['field1', 'field2', 'field3'])
    print(f"  field1 encrypted: {encrypted.get('field1_encrypted', False)}")
    print(f"  field2 encrypted: {encrypted.get('field2_encrypted', False)} (None values not encrypted)")
    print(f"  field3 encrypted: {encrypted.get('field3_encrypted', False)}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CSRD ENCRYPTION USAGE EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates how to integrate encryption into CSRD agents.")
    print("All examples use a test encryption key generated on-the-fly.")
    print("\nNOTE: In production, use CSRD_ENCRYPTION_KEY from environment variable.")

    # Generate a test key for examples
    test_key = EncryptionManager.generate_key()

    # Override global instance with test key for examples
    import utils.encryption
    utils.encryption._encryption_manager = EncryptionManager(key=test_key)

    # Run examples
    example_reporting_agent()
    example_intake_agent()
    example_calculator_agent()
    example_materiality_agent()
    example_key_generation()
    example_error_handling()

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Generate production encryption key")
    print("  2. Set CSRD_ENCRYPTION_KEY environment variable")
    print("  3. Update agents to use encryption for sensitive fields")
    print("  4. Run tests: pytest tests/test_encryption.py -v")
    print("  5. Review encryption_config.yaml for field definitions")
    print("\n")
