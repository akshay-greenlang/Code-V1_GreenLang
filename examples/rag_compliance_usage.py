"""
Example: Using RAG Regulatory Compliance Components (INTL-104)

This example demonstrates all 4 regulatory compliance components:
1. DocumentVersionManager - Version tracking for historical compliance
2. RAGGovernance - CSRB approval workflow and allowlist enforcement
3. ClimateTableExtractor - Extract emission factor tables from PDFs
4. SectionPathExtractor - Extract section hierarchies for citations

For climate/GHG accounting regulatory compliance.
"""

from datetime import date, datetime
from pathlib import Path

from greenlang.intelligence.rag import (
    DocumentVersionManager,
    RAGGovernance,
    ClimateTableExtractor,
    SectionPathExtractor,
    DocMeta,
    RAGConfig,
)


def example_1_version_management():
    """
    Example 1: Document Version Management

    Track multiple versions of GHG Protocol for historical compliance.
    """
    print("=" * 80)
    print("EXAMPLE 1: Document Version Management")
    print("=" * 80)

    # Initialize version manager
    vm = DocumentVersionManager()

    # Register GHG Protocol v1.00 (2004)
    ghg_v1_00 = DocMeta(
        doc_id="ghg-protocol-corp-v1.00",
        title="GHG Protocol Corporate Accounting and Reporting Standard",
        collection="ghg_protocol_corp",
        publisher="WRI/WBCSD",
        publication_date=date(2004, 9, 1),
        version="1.00",
        content_hash="a3f5b2c8d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4",
        doc_hash="b2c8d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2"
    )
    vm.register_version(ghg_v1_00, standard_id="ghg_protocol_corp")

    # Register GHG Protocol v1.05 (2015 revision)
    ghg_v1_05 = DocMeta(
        doc_id="ghg-protocol-corp-v1.05",
        title="GHG Protocol Corporate Accounting and Reporting Standard",
        collection="ghg_protocol_corp",
        publisher="WRI/WBCSD",
        publication_date=date(2015, 3, 24),
        revision_date=date(2015, 3, 24),
        version="1.05",
        content_hash="c8d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5",
        doc_hash="d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8"
    )
    vm.register_version(ghg_v1_05, standard_id="ghg_protocol_corp")

    # Historical compliance: Retrieve version for 2010 report
    print("\n1. Historical compliance for 2010 report:")
    doc_2010 = vm.retrieve_by_date("ghg_protocol_corp", date(2010, 1, 1))
    print(f"   Version for 2010: {doc_2010.version} (published {doc_2010.publication_date})")

    # Current compliance: Retrieve version for 2023 report
    print("\n2. Current compliance for 2023 report:")
    doc_2023 = vm.retrieve_by_date("ghg_protocol_corp", date(2023, 1, 1))
    print(f"   Version for 2023: {doc_2023.version} (published {doc_2023.publication_date})")

    # List all versions
    print("\n3. All versions of GHG Protocol Corporate Standard:")
    versions = vm.list_versions("ghg_protocol_corp")
    for v in versions:
        print(f"   - v{v.version} published {v.publication_date}")

    # Track errata
    print("\n4. Track errata:")
    vm.apply_errata(
        doc_id="ghg-protocol-corp-v1.05",
        errata_date=date(2016, 6, 1),
        description="Corrected Table 7.3 natural gas emission factor",
        sections_affected=["Chapter 7 > Table 7.3"]
    )
    errata_list = vm.get_errata("ghg-protocol-corp-v1.05")
    for e in errata_list:
        print(f"   - {e['errata_date']}: {e['description']}")

    # Mark v1.00 as deprecated
    print("\n5. Deprecation tracking:")
    vm.mark_deprecated(
        standard_id="ghg_protocol_corp_v1.00",
        deprecation_date=date(2015, 3, 24),
        replacement="ghg_protocol_corp_v1.05",
        reason="Superseded by revised edition"
    )
    is_dep, info = vm.is_deprecated("ghg_protocol_corp_v1.00", date(2020, 1, 1))
    if is_dep:
        print(f"   v1.00 is deprecated: {info['reason']}")
        print(f"   Use instead: {info['replacement']}")

    print()


def example_2_governance():
    """
    Example 2: CSRB Governance and Approval Workflow

    Demonstrate Climate Science Review Board approval process.
    """
    print("=" * 80)
    print("EXAMPLE 2: CSRB Governance and Approval Workflow")
    print("=" * 80)

    # Initialize governance
    config = RAGConfig(
        allowlist=['ghg_protocol_corp', 'ipcc_ar6_wg3'],
        require_approval=True,
        verify_checksums=True
    )
    gov = RAGGovernance(config, audit_dir=Path("./audit_example"))

    # Create document metadata for new standard
    new_standard = DocMeta(
        doc_id="iso-14064-1-2018",
        title="ISO 14064-1:2018 Greenhouse gases",
        collection="iso_14064_1",
        publisher="ISO",
        publication_date=date(2018, 12, 1),
        version="2018",
        content_hash="e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1",
        doc_hash="f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4"
    )

    # Submit for CSRB approval
    print("\n1. Submit ISO 14064-1 for CSRB approval:")
    approvers = [
        "climate_scientist_1",
        "climate_scientist_2",
        "audit_lead"
    ]

    # Create a dummy file for the example
    dummy_file = Path("./audit_example/iso_14064_1.pdf")
    dummy_file.parent.mkdir(parents=True, exist_ok=True)
    dummy_file.write_bytes(b"dummy pdf content for example")

    # Note: For this example, we'll skip actual file verification
    success = gov.submit_for_approval(
        doc_path=dummy_file,
        metadata=new_standard,
        approvers=approvers,
        requested_by="data_team",
        verify_checksum=False  # Skip for example
    )
    print(f"   Submission successful: {success}")

    # CSRB members vote
    print("\n2. CSRB voting process (2/3 majority required):")
    gov.vote_approval(
        "iso_14064_1",
        approver="climate_scientist_1",
        approve=True,
        comment="ISO standard verified, emission factors accurate"
    )
    print("   [+] climate_scientist_1: APPROVE")

    gov.vote_approval(
        "iso_14064_1",
        approver="climate_scientist_2",
        approve=True,
        comment="Reviewed methodology, aligns with IPCC guidelines"
    )
    print("   [+] climate_scientist_2: APPROVE")

    # Check approval status
    req = gov.get_approval_request("iso_14064_1")
    print(f"\n   Status: {req.status.upper()}")
    print(f"   Votes: {req.votes_approve}/{len(req.approvers_required)} approve")

    # Verify allowlist enforcement
    print("\n3. Runtime allowlist enforcement:")
    try:
        gov.enforce_allowlist(['ghg_protocol_corp', 'iso_14064_1'])
        print("   [+] All collections allowed")
    except ValueError as e:
        print(f"   [-] Allowlist violation: {e}")

    # Audit trail
    print("\n4. Audit trail:")
    audit = gov.get_audit_trail("iso_14064_1")
    if audit:
        print(f"   Requested by: {audit['requested_by']}")
        print(f"   Requested at: {audit['requested_at']}")
        print(f"   Votes: {audit['votes_approve']} approve, {audit['votes_reject']} reject")
        print(f"   Comments: {len(audit['comments'])}")

    # Cleanup dummy file
    if dummy_file.exists():
        dummy_file.unlink()

    print()


def example_3_table_extraction():
    """
    Example 3: Climate Table Extraction

    Extract emission factor tables from PDFs.
    """
    print("=" * 80)
    print("EXAMPLE 3: Climate Table Extraction")
    print("=" * 80)

    # Initialize extractor
    extractor = ClimateTableExtractor(prefer_camelot=True, flavor='lattice')

    # Demonstrate table structure (simulated)
    print("\n1. Simulated emission factor table structure:")
    simulated_table = {
        'caption': 'Table 7.3: Emission Factors for Stationary Combustion',
        'headers': ['Fuel Type', 'CO2 (kg/GJ)', 'CH4 (kg/GJ)', 'N2O (kg/GJ)'],
        'rows': [
            ['Natural Gas', '56.1', '0.001', '0.0001'],
            ['Coal (bituminous)', '94.6', '0.001', '0.0015'],
            ['Diesel', '74.1', '0.003', '0.0006']
        ],
        'units': {
            'CO2': 'kg/GJ',
            'CH4': 'kg/GJ',
            'N2O': 'kg/GJ'
        },
        'notes': [
            '* Values from IPCC 2006 Guidelines',
            '† Natural gas includes pipeline natural gas'
        ],
        'page': 45,
        'bbox': (72.0, 200.0, 540.0, 400.0)
    }

    # Convert to markdown
    print("\n2. Table in Markdown format:")
    md = extractor.table_to_markdown(simulated_table)
    print(md)

    # Show JSON structure
    print("\n3. Table in JSON format (for chunk metadata):")
    json_str = extractor.table_to_json(simulated_table)
    print(json_str[:200] + "...")

    print("\n4. Usage in PDF extraction:")
    print("   # Extract tables from PDF")
    print("   tables = extractor.extract_tables(")
    print("       pdf_path=Path('ghg_protocol.pdf'),")
    print("       page_num=45")
    print("   )")
    print("   ")
    print("   # Store in chunk metadata")
    print("   chunk.extra['table_data'] = tables[0]")

    print()


def example_4_section_extraction():
    """
    Example 4: Section Hierarchy Extraction

    Extract section paths for proper citations.
    """
    print("=" * 80)
    print("EXAMPLE 4: Section Hierarchy Extraction")
    print("=" * 80)

    # Initialize extractor
    extractor = SectionPathExtractor()

    # IPCC AR6 example
    print("\n1. IPCC AR6 section extraction:")
    ipcc_text = """
    IPCC AR6 WG3 Chapter 6: Energy Systems

    As shown in Box 6.2, renewable energy deployment has accelerated.
    Figure 6.3a illustrates global renewable capacity trends from 2010-2020.
    """
    path = extractor.extract_path(ipcc_text, doc_type='ipcc')
    print(f"   Text: {ipcc_text.strip()[:60]}...")
    print(f"   Section path: {path}")

    # GHG Protocol example
    print("\n2. GHG Protocol section extraction:")
    ghg_text = """
    See Appendix E for default emission factors.
    Table E.1 provides natural gas combustion factors.
    """
    path = extractor.extract_path(ghg_text, doc_type='ghg_protocol')
    print(f"   Text: {ghg_text.strip()[:60]}...")
    print(f"   Section path: {path}")

    # ISO standard example
    print("\n3. ISO standard section extraction:")
    iso_text = """
    5.2.3 Quantification of GHG emissions

    Organizations shall quantify emissions using approved methodologies.
    See Table 5.1 for calculation approaches.
    """
    path = extractor.extract_path(iso_text, doc_type='iso')
    print(f"   Text: {iso_text.strip()[:60]}...")
    print(f"   Section path: {path}")

    # Auto-detect document type
    print("\n4. Auto-detect document type:")
    for test_text, expected_type in [
        ("IPCC AR6 WG3 Chapter 6", "ipcc"),
        ("GHG Protocol Corporate Standard", "ghg_protocol"),
        ("ISO 14064-1:2018", "iso"),
    ]:
        detected = extractor.detect_section_type(test_text)
        match = "[+]" if detected == expected_type else "[-]"
        print(f"   '{test_text}' -> {detected} ({match})")

    # Generate citation anchor
    print("\n5. Generate URL anchor for citation:")
    section_path = "Chapter 6 > Box 6.2 > Figure 6.3a"
    anchor = extractor.get_section_anchor(section_path)
    print(f"   Section path: {section_path}")
    print(f"   URL anchor: #{anchor}")
    print(f"   Full citation URL: https://ipcc.ch/report.pdf#{anchor}")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("RAG REGULATORY COMPLIANCE COMPONENTS (INTL-104)")
    print("Climate/GHG Accounting Regulatory Compliance Examples")
    print("=" * 80 + "\n")

    example_1_version_management()
    example_2_governance()
    example_3_table_extraction()
    example_4_section_extraction()

    print("=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Install optional dependencies:")
    print("   pip install camelot-py[cv] tabula-py pdfplumber PyPDF2")
    print("2. Integrate with RAG ingestion pipeline")
    print("3. Configure CSRB approval workflow")
    print("4. Set up version tracking for climate standards")
    print()


if __name__ == "__main__":
    main()
