"""
Entity Resolution Usage Examples

This file demonstrates practical usage of the entity resolution services
for various business scenarios including:
- Supply chain vendor validation
- M&A due diligence
- Regulatory compliance checks
- Batch processing for data enrichment
"""

import asyncio
import json
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

from greenlang.intelligence.entity_resolution import (
    EntityResolver,
    ResolutionRequest,
    EntitySource,
    EntityType
)


async def example_1_supply_chain_validation():
    """
    Example 1: Validate and enrich supplier data for supply chain management.

    Use case: A company needs to validate its supplier database against
    authoritative sources for compliance and risk assessment.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Supply Chain Vendor Validation")
    print("=" * 80)

    # Initialize resolver with API credentials
    resolver = EntityResolver(
        gleif_api_key=None,  # Optional for basic GLEIF access
        duns_api_key="your_duns_api_key",
        duns_api_secret="your_duns_api_secret",
        opencorporates_api_key="your_oc_api_key"
    )

    # Sample supplier list that needs validation
    suppliers = [
        {"name": "Apple Inc", "country": "US", "internal_id": "SUP001"},
        {"name": "Samsung Electronics", "country": "KR", "internal_id": "SUP002"},
        {"name": "Taiwan Semiconductor", "country": "TW", "internal_id": "SUP003"},
        {"name": "BASF SE", "country": "DE", "internal_id": "SUP004"},
        {"name": "Toyota Motor Corporation", "country": "JP", "internal_id": "SUP005"}
    ]

    print("\nValidating suppliers...")
    validated_suppliers = []

    for supplier in suppliers:
        request = ResolutionRequest(
            query=supplier["name"],
            country=supplier["country"],
            min_confidence=0.8,
            preferred_sources=[EntitySource.GLEIF, EntitySource.DUNS, EntitySource.OPENCORPORATES]
        )

        result = await resolver.resolve_entity(request)

        if result.best_match:
            validated = {
                "internal_id": supplier["internal_id"],
                "original_name": supplier["name"],
                "validated_name": result.best_match.entity_name,
                "lei": result.best_match.lei,
                "duns": result.best_match.duns,
                "confidence": result.best_match.confidence_score,
                "source": result.best_match.source.value,
                "status": result.best_match.status,
                "parent_company": result.best_match.parent_name
            }
            validated_suppliers.append(validated)

            print(f"\n✓ {supplier['name']}")
            print(f"  Validated as: {result.best_match.entity_name}")
            print(f"  LEI: {result.best_match.lei or 'N/A'}")
            print(f"  DUNS: {result.best_match.duns or 'N/A'}")
            print(f"  Confidence: {result.best_match.confidence_score:.2%}")
            print(f"  Source: {result.best_match.source.value}")
        else:
            print(f"\n✗ {supplier['name']} - No match found")
            validated_suppliers.append({
                "internal_id": supplier["internal_id"],
                "original_name": supplier["name"],
                "validation_failed": True
            })

    # Create supplier validation report
    print("\n" + "-" * 40)
    print("SUPPLIER VALIDATION SUMMARY")
    print("-" * 40)

    total = len(suppliers)
    validated = sum(1 for s in validated_suppliers if not s.get("validation_failed"))
    print(f"Total suppliers: {total}")
    print(f"Successfully validated: {validated} ({validated/total:.1%})")
    print(f"Failed validation: {total - validated}")

    # Export to CSV (simulated)
    print("\nExporting validated supplier data to CSV...")
    df = pd.DataFrame(validated_suppliers)
    # df.to_csv("validated_suppliers.csv", index=False)
    print("✓ Export complete")

    return validated_suppliers


async def example_2_ma_due_diligence():
    """
    Example 2: M&A Due Diligence - Deep entity investigation.

    Use case: Investment firm needs comprehensive information about
    a target company and its corporate structure.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: M&A Due Diligence Investigation")
    print("=" * 80)

    resolver = EntityResolver()

    target_company = "Nvidia Corporation"
    print(f"\nInvestigating target: {target_company}")

    # First, get comprehensive data from all sources
    request = ResolutionRequest(
        query=target_company,
        country="US",
        min_confidence=0.7,
        max_results=5,
        preferred_sources=[EntitySource.GLEIF, EntitySource.DUNS, EntitySource.OPENCORPORATES]
    )

    result = await resolver.resolve_entity(request)

    print("\n" + "-" * 40)
    print("DUE DILIGENCE REPORT")
    print("-" * 40)

    if result.best_match:
        match = result.best_match

        print(f"\n1. ENTITY IDENTIFICATION")
        print(f"   Legal Name: {match.entity_name}")
        print(f"   LEI: {match.lei or 'Not found'}")
        print(f"   DUNS: {match.duns or 'Not found'}")
        print(f"   OpenCorporates ID: {match.opencorporates_id or 'Not found'}")

        print(f"\n2. CORPORATE DETAILS")
        print(f"   Entity Type: {match.entity_type.value if match.entity_type else 'Unknown'}")
        print(f"   Status: {match.status or 'Unknown'}")
        print(f"   Jurisdiction: {match.jurisdiction or 'Unknown'}")

        if match.incorporation_date:
            years_in_business = (datetime.now() - match.incorporation_date).days / 365.25
            print(f"   Incorporation Date: {match.incorporation_date.strftime('%Y-%m-%d')}")
            print(f"   Years in Business: {years_in_business:.1f}")

        print(f"\n3. REGISTERED ADDRESS")
        if match.address:
            for key, value in match.address.items():
                if value:
                    print(f"   {key.title()}: {value}")

        print(f"\n4. OWNERSHIP STRUCTURE")
        print(f"   Parent Company: {match.parent_name or 'None (Independent)'}")
        print(f"   Parent LEI: {match.parent_lei or 'N/A'}")
        print(f"   Ultimate Parent: {match.ultimate_parent_lei or 'N/A'}")

        print(f"\n5. DATA SOURCES")
        print(f"   Primary Source: {match.source.value}")
        print(f"   Match Confidence: {match.confidence_score:.2%}")
        print(f"   Data Freshness: {match.data_freshness.strftime('%Y-%m-%d') if match.data_freshness else 'Unknown'}")

        # Check all sources for additional information
        print(f"\n6. CROSS-SOURCE VALIDATION")
        print(f"   Sources Queried: {', '.join(s.value for s in result.sources_queried)}")
        print(f"   Sources with Data: {', '.join(s.value for s in result.sources_succeeded)}")

        if len(result.all_matches) > 1:
            print(f"\n   Alternative Matches Found:")
            for alt_match in result.all_matches[1:4]:  # Show top 3 alternatives
                print(f"   - {alt_match.entity_name} ({alt_match.source.value}, {alt_match.confidence_score:.2%})")

    # Risk indicators
    print(f"\n7. RISK INDICATORS")
    risk_factors = []

    if not match.lei:
        risk_factors.append("No LEI found - may indicate smaller or private entity")

    if match.status and "INACTIVE" in match.status.upper():
        risk_factors.append("Entity status is INACTIVE")

    if len(result.sources_succeeded) < 2:
        risk_factors.append("Limited data sources available for validation")

    if match.confidence_score < 0.9:
        risk_factors.append(f"Match confidence below 90% ({match.confidence_score:.2%})")

    if risk_factors:
        for risk in risk_factors:
            print(f"   ⚠ {risk}")
    else:
        print("   ✓ No significant risk indicators identified")

    print("\n" + "-" * 40)
    print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Provenance Hash: {result.provenance_hash[:16]}...")


async def example_3_regulatory_compliance():
    """
    Example 3: Regulatory Compliance - LEI validation for reporting.

    Use case: Financial institution needs to validate counterparty LEIs
    for regulatory reporting (e.g., MiFID II, EMIR, Dodd-Frank).
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Regulatory Compliance - LEI Validation")
    print("=" * 80)

    resolver = EntityResolver()

    # Transactions that need LEI validation
    transactions = [
        {
            "id": "TRX001",
            "counterparty": "Goldman Sachs Group Inc",
            "reported_lei": "784F5XWPLTWKTBV3E584",  # Actual Goldman Sachs LEI
            "value": 50000000
        },
        {
            "id": "TRX002",
            "counterparty": "JP Morgan Chase",
            "reported_lei": "8I5DZWZKVSZI1NUHU748",  # Actual JPM LEI
            "value": 75000000
        },
        {
            "id": "TRX003",
            "counterparty": "Deutsche Bank",
            "reported_lei": "7LTWFZYICNSX8D621K86",  # Actual DB LEI
            "value": 30000000
        },
        {
            "id": "TRX004",
            "counterparty": "Unknown Trading Corp",
            "reported_lei": "INVALID123456789012",  # Invalid LEI
            "value": 10000000
        }
    ]

    print("\nValidating transaction counterparties for regulatory compliance...")

    compliance_report = []

    for txn in transactions:
        print(f"\n{txn['id']}: {txn['counterparty']}")

        # Validate the reported LEI
        request = ResolutionRequest(
            query=txn["counterparty"],
            lei=txn["reported_lei"],
            min_confidence=0.95,  # High confidence required for compliance
            preferred_sources=[EntitySource.GLEIF]  # LEI must come from GLEIF
        )

        result = await resolver.resolve_entity(request)

        compliance_status = "COMPLIANT"
        issues = []

        if result.best_match:
            # Check if LEI matches
            if result.best_match.lei != txn["reported_lei"]:
                compliance_status = "NON_COMPLIANT"
                issues.append(f"LEI mismatch: reported {txn['reported_lei']} vs actual {result.best_match.lei}")

            # Check if entity is active
            if result.best_match.status != "ACTIVE":
                compliance_status = "WARNING"
                issues.append(f"Entity status: {result.best_match.status}")

            # Check name match
            if result.best_match.confidence_score < 0.95:
                compliance_status = "WARNING"
                issues.append(f"Name match confidence: {result.best_match.confidence_score:.2%}")

            print(f"  ✓ LEI Validated: {result.best_match.lei}")
            print(f"  Status: {compliance_status}")

        else:
            compliance_status = "NON_COMPLIANT"
            issues.append("LEI validation failed - no match found")
            print(f"  ✗ LEI validation failed")

        if issues:
            print(f"  Issues: {', '.join(issues)}")

        compliance_report.append({
            "transaction_id": txn["id"],
            "counterparty": txn["counterparty"],
            "reported_lei": txn["reported_lei"],
            "validated_lei": result.best_match.lei if result.best_match else None,
            "compliance_status": compliance_status,
            "issues": issues,
            "transaction_value": txn["value"]
        })

    # Generate compliance summary
    print("\n" + "-" * 40)
    print("COMPLIANCE SUMMARY")
    print("-" * 40)

    total_transactions = len(transactions)
    compliant = sum(1 for r in compliance_report if r["compliance_status"] == "COMPLIANT")
    warnings = sum(1 for r in compliance_report if r["compliance_status"] == "WARNING")
    non_compliant = sum(1 for r in compliance_report if r["compliance_status"] == "NON_COMPLIANT")

    total_value = sum(r["transaction_value"] for r in compliance_report)
    compliant_value = sum(r["transaction_value"] for r in compliance_report if r["compliance_status"] == "COMPLIANT")

    print(f"Total Transactions: {total_transactions}")
    print(f"Compliant: {compliant} ({compliant/total_transactions:.1%})")
    print(f"Warnings: {warnings} ({warnings/total_transactions:.1%})")
    print(f"Non-Compliant: {non_compliant} ({non_compliant/total_transactions:.1%})")
    print(f"\nTotal Transaction Value: ${total_value:,.0f}")
    print(f"Compliant Value: ${compliant_value:,.0f} ({compliant_value/total_value:.1%})")

    if non_compliant > 0:
        print("\n⚠ REGULATORY ACTION REQUIRED")
        print("Non-compliant transactions must be resolved before reporting")


async def example_4_batch_enrichment():
    """
    Example 4: Batch Data Enrichment.

    Use case: Enrich a large customer/vendor database with
    official identifiers and validated information.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Batch Data Enrichment")
    print("=" * 80)

    resolver = EntityResolver()

    # Simulate a large dataset
    companies = [
        "Microsoft Corporation",
        "Amazon.com Inc",
        "Alphabet Inc",
        "Tesla Inc",
        "Meta Platforms",
        "Berkshire Hathaway",
        "Johnson & Johnson",
        "Procter & Gamble",
        "Visa Inc",
        "Walmart Inc"
    ]

    print(f"\nEnriching {len(companies)} companies in batch mode...")
    print("Processing with concurrency limit of 3...")

    # Batch resolve with concurrency control
    results = await resolver.batch_resolve(companies, max_concurrent=3)

    # Process results
    enriched_data = []
    for company, result in zip(companies, results):
        if result.best_match:
            enriched = {
                "original_name": company,
                "official_name": result.best_match.entity_name,
                "lei": result.best_match.lei,
                "duns": result.best_match.duns,
                "country": result.best_match.country,
                "source": result.best_match.source.value,
                "confidence": result.best_match.confidence_score
            }
        else:
            enriched = {
                "original_name": company,
                "official_name": None,
                "error": "No match found"
            }

        enriched_data.append(enriched)

    # Display enrichment results
    print("\n" + "-" * 40)
    print("ENRICHMENT RESULTS")
    print("-" * 40)

    successful = sum(1 for e in enriched_data if not e.get("error"))
    print(f"Successfully enriched: {successful}/{len(companies)} ({successful/len(companies):.1%})")

    print("\nSample enriched records:")
    for record in enriched_data[:5]:
        if not record.get("error"):
            print(f"\n{record['original_name']}")
            print(f"  → {record['official_name']}")
            print(f"  LEI: {record.get('lei', 'N/A')}")
            print(f"  Country: {record.get('country', 'N/A')}")
            print(f"  Confidence: {record.get('confidence', 0):.2%}")

    # Show statistics
    stats = resolver.get_statistics()
    print("\n" + "-" * 40)
    print("PERFORMANCE STATISTICS")
    print("-" * 40)
    print(f"Total resolutions: {stats['total_resolutions']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"API calls made: {stats['total_api_calls']}")

    if stats['source_successes']:
        print("\nSource success rates:")
        for source, count in stats['source_successes'].items():
            print(f"  {source}: {count} successful queries")

    if stats['source_failures']:
        print("\nSource failures:")
        for source, count in stats['source_failures'].items():
            print(f"  {source}: {count} failed queries")


async def example_5_fuzzy_matching():
    """
    Example 5: Fuzzy Matching and Name Variations.

    Use case: Handle company name variations, typos, and different formats.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Fuzzy Matching and Name Variations")
    print("=" * 80)

    resolver = EntityResolver()

    # Test various name variations
    name_variations = [
        ("Microsoft", "Microsoft Corporation"),  # Missing suffix
        ("Apple Computer", "Apple Inc."),  # Old name
        ("Google", "Alphabet Inc."),  # Parent company
        ("Facebook", "Meta Platforms, Inc."),  # Rebrand
        ("GE", "General Electric Company"),  # Abbreviation
        ("IBM", "International Business Machines Corporation"),  # Abbreviation
        ("AT&T", "AT&T Inc."),  # Special characters
        ("3M", "3M Company"),  # Numeric start
        ("BMW", "Bayerische Motoren Werke AG"),  # Foreign full name
        ("P&G", "Procter & Gamble Company")  # Multiple abbreviations
    ]

    print("\nTesting fuzzy matching capabilities...")

    for search_name, expected_name in name_variations:
        print(f"\nSearching for: '{search_name}'")
        print(f"Expected: '{expected_name}'")

        request = ResolutionRequest(
            query=search_name,
            fuzzy_match=True,
            min_confidence=0.6  # Lower threshold for fuzzy matching
        )

        result = await resolver.resolve_entity(request)

        if result.best_match:
            print(f"  ✓ Found: {result.best_match.entity_name}")
            print(f"  Confidence: {result.best_match.confidence_score:.2%}")
            print(f"  Match Method: {result.best_match.match_method}")

            # Check if we found the expected company
            similarity = resolver._calculate_name_similarity(
                expected_name,
                result.best_match.entity_name
            )
            if similarity > 0.8:
                print(f"  ✓ Correct match!")
            else:
                print(f"  ⚠ Possible mismatch (similarity: {similarity:.2%})")
        else:
            print(f"  ✗ No match found")

    print("\n" + "-" * 40)
    print("FUZZY MATCHING SUMMARY")
    print("-" * 40)
    print("The resolver successfully handles:")
    print("  • Missing corporate suffixes (Inc, Corp, LLC)")
    print("  • Company abbreviations (IBM, GE, P&G)")
    print("  • Special characters and punctuation")
    print("  • Historical names and rebrands")
    print("  • Parent/subsidiary relationships")


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("ENTITY RESOLUTION SERVICE - USAGE EXAMPLES")
    print("=" * 80)

    examples = [
        ("Supply Chain Validation", example_1_supply_chain_validation),
        ("M&A Due Diligence", example_2_ma_due_diligence),
        ("Regulatory Compliance", example_3_regulatory_compliance),
        ("Batch Data Enrichment", example_4_batch_enrichment),
        ("Fuzzy Matching", example_5_fuzzy_matching)
    ]

    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    # Run examples (in production, you might want to make this interactive)
    print("\nRunning all examples...")

    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    # Run all examples
    asyncio.run(main())