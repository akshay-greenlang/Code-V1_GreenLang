# -*- coding: utf-8 -*-
"""
Tests for GL-FOUND-X-005: Citations & Evidence Agent

Tests cover:
    - Citation registration and retrieval
    - Citation versioning and validity checking
    - Source verification
    - Evidence packaging
    - BibTeX and JSON export
    - Methodology references
    - Regulatory requirements
    - Provenance hash calculation
"""

import json
import sys
import pytest
from datetime import date, datetime, timedelta
from typing import Any, Dict

# Import directly from the module file to avoid __init__.py import chain issues
# This is necessary because other modules in the foundation package have
# Pydantic v1/v2 compatibility issues that cause import errors
from greenlang.agents.base import AgentConfig, AgentResult

# Direct import from citations_agent module
import importlib.util
import os

_module_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    'greenlang', 'agents', 'foundation', 'citations_agent.py'
)

_spec = importlib.util.spec_from_file_location('citations_agent_direct', _module_path)
_citations_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_citations_module)

# Extract all needed classes from the dynamically loaded module
CitationsEvidenceAgent = _citations_module.CitationsEvidenceAgent
Citation = _citations_module.Citation
CitationMetadata = _citations_module.CitationMetadata
CitationType = _citations_module.CitationType
SourceAuthority = _citations_module.SourceAuthority
VerificationStatus = _citations_module.VerificationStatus
RegulatoryFramework = _citations_module.RegulatoryFramework
EvidenceItem = _citations_module.EvidenceItem
EvidenceType = _citations_module.EvidenceType
EvidencePackage = _citations_module.EvidencePackage
MethodologyReference = _citations_module.MethodologyReference
RegulatoryRequirement = _citations_module.RegulatoryRequirement
DataSourceAttribution = _citations_module.DataSourceAttribution
CitationsAgentInput = _citations_module.CitationsAgentInput
CitationsAgentOutput = _citations_module.CitationsAgentOutput


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def agent() -> CitationsEvidenceAgent:
    """Create a Citations & Evidence Agent instance."""
    return CitationsEvidenceAgent()


@pytest.fixture
def sample_defra_citation() -> Citation:
    """Create a sample DEFRA emission factor citation."""
    return Citation(
        citation_id="defra-2024-ghg-factors",
        citation_type=CitationType.EMISSION_FACTOR,
        source_authority=SourceAuthority.DEFRA,
        metadata=CitationMetadata(
            title="UK Government GHG Conversion Factors for Company Reporting",
            authors=["Department for Energy Security and Net Zero"],
            publication_date=date(2024, 6, 1),
            version="2024",
            publisher="UK Government",
            url="https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting",
        ),
        effective_date=date(2024, 1, 1),
        expiration_date=date(2024, 12, 31),
        verification_status=VerificationStatus.VERIFIED,
        regulatory_frameworks=[RegulatoryFramework.CSRD],
        abstract="Annual GHG conversion factors for UK company reporting",
        key_values={
            "electricity_uk_kwh_kgco2e": 0.207074,
            "natural_gas_kwh_kgco2e": 0.18293,
            "diesel_litre_kgco2e": 2.51233,
        },
    )


@pytest.fixture
def sample_scientific_citation() -> Citation:
    """Create a sample scientific paper citation with DOI."""
    return Citation(
        citation_id="ipcc-ar6-wg3",
        citation_type=CitationType.SCIENTIFIC,
        source_authority=SourceAuthority.IPCC,
        metadata=CitationMetadata(
            title="Climate Change 2022: Mitigation of Climate Change",
            authors=[
                "IPCC Working Group III",
                "Shukla, P.R.",
                "Skea, J.",
            ],
            publication_date=date(2022, 4, 4),
            doi="10.1017/9781009157926",
            publisher="Cambridge University Press",
        ),
        effective_date=date(2022, 4, 4),
        verification_status=VerificationStatus.VERIFIED,
        regulatory_frameworks=[RegulatoryFramework.CSRD, RegulatoryFramework.TCFD],
        abstract="IPCC AR6 Working Group III report on climate change mitigation",
    )


@pytest.fixture
def sample_regulatory_citation() -> Citation:
    """Create a sample regulatory citation for CSRD."""
    return Citation(
        citation_id="csrd-directive-2022-2464",
        citation_type=CitationType.REGULATORY,
        source_authority=SourceAuthority.EU_COMMISSION,
        metadata=CitationMetadata(
            title="Directive (EU) 2022/2464 - Corporate Sustainability Reporting Directive",
            publication_date=date(2022, 12, 14),
            publisher="European Parliament and Council",
            url="https://eur-lex.europa.eu/eli/dir/2022/2464",
        ),
        effective_date=date(2024, 1, 1),
        verification_status=VerificationStatus.VERIFIED,
        regulatory_frameworks=[RegulatoryFramework.CSRD],
        abstract="Directive amending Regulation (EU) No 537/2014 and Directives 2004/109/EC, 2006/43/EC and 2013/34/EU regarding corporate sustainability reporting",
    )


@pytest.fixture
def sample_evidence_item() -> EvidenceItem:
    """Create a sample evidence item."""
    return EvidenceItem(
        evidence_type=EvidenceType.CALCULATION,
        description="Scope 2 electricity emissions calculation",
        data={
            "activity_data": {
                "electricity_kwh": 1000000,
                "location": "UK",
            },
            "emission_factor": {
                "value": 0.207074,
                "unit": "kgCO2e/kWh",
                "source": "DEFRA 2024",
            },
            "result": {
                "emissions_kgco2e": 207074,
                "emissions_tco2e": 207.074,
            },
        },
        citation_ids=["defra-2024-ghg-factors"],
        source_agent="GL-CALC-X-001",
    )


# =============================================================================
# Test Classes
# =============================================================================

class TestCitationModel:
    """Tests for the Citation model."""

    def test_citation_creation(self, sample_defra_citation: Citation):
        """Test creating a citation."""
        citation = sample_defra_citation

        assert citation.citation_id == "defra-2024-ghg-factors"
        assert citation.citation_type == CitationType.EMISSION_FACTOR
        assert citation.source_authority == SourceAuthority.DEFRA
        assert citation.metadata.title is not None
        assert citation.metadata.version == "2024"

    def test_citation_content_hash(self, sample_defra_citation: Citation):
        """Test content hash calculation."""
        hash1 = sample_defra_citation.calculate_content_hash()

        # Hash should be deterministic
        hash2 = sample_defra_citation.calculate_content_hash()
        assert hash1 == hash2

        # Hash should be 64 characters (SHA-256)
        assert len(hash1) == 64

    def test_citation_validity_current(self, sample_defra_citation: Citation):
        """Test citation validity checking - current citation."""
        # Set dates to ensure validity
        sample_defra_citation.effective_date = date.today() - timedelta(days=30)
        sample_defra_citation.expiration_date = date.today() + timedelta(days=30)
        sample_defra_citation.verification_status = VerificationStatus.VERIFIED

        assert sample_defra_citation.is_valid() is True

    def test_citation_validity_expired(self, sample_defra_citation: Citation):
        """Test citation validity checking - expired citation."""
        sample_defra_citation.expiration_date = date.today() - timedelta(days=1)

        assert sample_defra_citation.is_valid() is False

    def test_citation_validity_future(self, sample_defra_citation: Citation):
        """Test citation validity checking - future citation."""
        sample_defra_citation.effective_date = date.today() + timedelta(days=30)

        assert sample_defra_citation.is_valid() is False

    def test_citation_validity_invalid_status(self, sample_defra_citation: Citation):
        """Test citation validity checking - invalid status."""
        sample_defra_citation.verification_status = VerificationStatus.INVALID

        assert sample_defra_citation.is_valid() is False

    def test_citation_to_bibtex(self, sample_defra_citation: Citation):
        """Test BibTeX export."""
        bibtex = sample_defra_citation.to_bibtex()

        assert '@techreport{' in bibtex
        assert 'title = {' in bibtex
        assert 'UK Government GHG Conversion Factors' in bibtex
        assert 'year = {2024}' in bibtex

    def test_scientific_citation_bibtex(self, sample_scientific_citation: Citation):
        """Test BibTeX export for scientific citation."""
        bibtex = sample_scientific_citation.to_bibtex()

        assert '@article{' in bibtex
        assert 'doi = {10.1017/9781009157926}' in bibtex

    def test_doi_validation_valid(self):
        """Test DOI validation with valid DOI."""
        metadata = CitationMetadata(
            title="Test Paper",
            doi="10.1017/9781009157926"
        )
        assert metadata.doi == "10.1017/9781009157926"

    def test_doi_validation_invalid(self):
        """Test DOI validation with invalid DOI."""
        with pytest.raises(ValueError, match="Invalid DOI format"):
            CitationMetadata(
                title="Test Paper",
                doi="invalid-doi"
            )


class TestEvidenceModel:
    """Tests for Evidence models."""

    def test_evidence_item_creation(self, sample_evidence_item: EvidenceItem):
        """Test creating an evidence item."""
        evidence = sample_evidence_item

        assert evidence.evidence_type == EvidenceType.CALCULATION
        assert "electricity_kwh" in str(evidence.data)
        assert len(evidence.citation_ids) == 1

    def test_evidence_item_content_hash(self, sample_evidence_item: EvidenceItem):
        """Test evidence content hash calculation."""
        hash1 = sample_evidence_item.calculate_content_hash()
        hash2 = sample_evidence_item.calculate_content_hash()

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_evidence_package_creation(self):
        """Test creating an evidence package."""
        package = EvidencePackage(
            name="Scope 2 Emissions Calculation",
            description="Complete evidence for Scope 2 electricity emissions",
            calculation_context={
                "period": "2024",
                "entity": "UK Operations",
            },
            calculation_result={
                "total_emissions_tco2e": 207.074,
            },
        )

        assert package.name == "Scope 2 Emissions Calculation"
        assert package.package_id is not None

    def test_evidence_package_add_items(
        self,
        sample_defra_citation: Citation,
        sample_evidence_item: EvidenceItem
    ):
        """Test adding items to evidence package."""
        package = EvidencePackage(name="Test Package")

        package.add_citation(sample_defra_citation)
        package.add_evidence(sample_evidence_item)

        assert len(package.citations) == 1
        assert len(package.evidence_items) == 1
        assert package.citations[0].content_hash is not None
        assert package.evidence_items[0].content_hash is not None

    def test_evidence_package_finalize(
        self,
        sample_defra_citation: Citation,
        sample_evidence_item: EvidenceItem
    ):
        """Test finalizing an evidence package."""
        package = EvidencePackage(name="Test Package")
        package.add_citation(sample_defra_citation)
        package.add_evidence(sample_evidence_item)

        package_hash = package.finalize()

        assert package.package_hash is not None
        assert package.package_hash == package_hash
        assert len(package_hash) == 64

    def test_evidence_package_to_json(self, sample_defra_citation: Citation):
        """Test JSON export of evidence package."""
        package = EvidencePackage(name="Test Package")
        package.add_citation(sample_defra_citation)
        package.finalize()

        json_str = package.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["name"] == "Test Package"
        assert len(data["citations"]) == 1


class TestCitationsEvidenceAgent:
    """Tests for the Citations & Evidence Agent."""

    def test_agent_initialization(self, agent: CitationsEvidenceAgent):
        """Test agent initialization."""
        assert agent.AGENT_ID == "GL-FOUND-X-005"
        assert agent.AGENT_NAME == "Citations & Evidence Agent"
        assert len(agent._methodologies) > 0  # Should have standard methodologies

    def test_register_citation_via_execute(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation
    ):
        """Test registering a citation through execute."""
        result = agent.run({
            'action': 'register_citation',
            'citation': sample_defra_citation.model_dump(),
        })

        assert result.success is True
        assert 'citation' in result.data
        assert result.data['citation']['citation_id'] == sample_defra_citation.citation_id

    def test_register_citation_direct(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation
    ):
        """Test registering a citation directly."""
        citation_id = agent.register_citation(sample_defra_citation)

        assert citation_id == sample_defra_citation.citation_id

        # Should be retrievable
        retrieved = agent.get_citation(citation_id)
        assert retrieved is not None
        assert retrieved.metadata.title == sample_defra_citation.metadata.title

    def test_lookup_citation(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation
    ):
        """Test looking up a citation."""
        agent.register_citation(sample_defra_citation)

        result = agent.run({
            'action': 'lookup_citation',
            'citation_id': sample_defra_citation.citation_id,
        })

        assert result.success is True
        assert result.data['citation']['citation_id'] == sample_defra_citation.citation_id

    def test_lookup_citation_not_found(self, agent: CitationsEvidenceAgent):
        """Test looking up a non-existent citation."""
        result = agent.run({
            'action': 'lookup_citation',
            'citation_id': 'non-existent-id',
        })

        assert result.success is False
        assert 'not found' in result.error.lower()

    def test_lookup_multiple_citations(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation,
        sample_scientific_citation: Citation
    ):
        """Test looking up multiple citations."""
        agent.register_citation(sample_defra_citation)
        agent.register_citation(sample_scientific_citation)

        result = agent.run({
            'action': 'lookup_multiple',
            'citation_ids': [
                sample_defra_citation.citation_id,
                sample_scientific_citation.citation_id,
                'non-existent',
            ],
        })

        assert result.success is True
        assert len(result.data['citations']) == 2
        assert len(result.data['warnings']) == 1  # One not found

    def test_verify_citation(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation
    ):
        """Test verifying a citation."""
        # Ensure the citation is not expired by setting future expiration
        sample_defra_citation.effective_date = date.today() - timedelta(days=30)
        sample_defra_citation.expiration_date = date.today() + timedelta(days=300)
        sample_defra_citation.verification_status = VerificationStatus.UNVERIFIED
        agent.register_citation(sample_defra_citation)

        result = agent.run({
            'action': 'verify_citation',
            'citation_id': sample_defra_citation.citation_id,
            'user_id': 'test_user',
        })

        assert result.success is True
        assert 'verification_results' in result.data
        # Should be verified (has version and not expired)
        assert result.data['verification_results'][sample_defra_citation.citation_id] == 'verified'

    def test_verify_citation_expired(self, agent: CitationsEvidenceAgent):
        """Test verifying an expired citation."""
        citation = Citation(
            citation_id="expired-citation",
            citation_type=CitationType.EMISSION_FACTOR,
            source_authority=SourceAuthority.DEFRA,
            metadata=CitationMetadata(
                title="Old Factors",
                version="2020",
            ),
            effective_date=date(2020, 1, 1),
            expiration_date=date(2020, 12, 31),  # Expired
        )
        agent.register_citation(citation)

        result = agent.run({
            'action': 'verify_citation',
            'citation_id': citation.citation_id,
        })

        assert result.success is True
        assert result.data['verification_results'][citation.citation_id] == 'expired'

    def test_create_evidence_package(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation
    ):
        """Test creating an evidence package."""
        agent.register_citation(sample_defra_citation)

        result = agent.run({
            'action': 'create_package',
            'package_name': 'Scope 2 Calculation Evidence',
            'citation_ids': [sample_defra_citation.citation_id],
            'calculation_context': {
                'electricity_kwh': 1000000,
            },
            'calculation_result': {
                'emissions_tco2e': 207.074,
            },
            'user_id': 'test_user',
        })

        assert result.success is True
        assert 'evidence_package' in result.data
        assert result.data['evidence_package']['name'] == 'Scope 2 Calculation Evidence'
        assert len(result.data['evidence_package']['citations']) == 1

    def test_create_evidence_package_direct(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation
    ):
        """Test creating an evidence package directly."""
        agent.register_citation(sample_defra_citation)

        package = agent.create_evidence_package(
            name="Direct Package Test",
            calculation_context={'input': 'test'},
            calculation_result={'output': 'result'},
            citation_ids=[sample_defra_citation.citation_id],
            user_id='test_user',
        )

        assert package is not None
        assert package.package_hash is not None
        assert len(package.citations) == 1
        assert len(package.evidence_items) == 1

    def test_add_evidence_to_package(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation,
        sample_evidence_item: EvidenceItem
    ):
        """Test adding evidence to an existing package."""
        agent.register_citation(sample_defra_citation)

        # Create package
        create_result = agent.run({
            'action': 'create_package',
            'package_name': 'Test Package',
        })
        package_id = create_result.data['evidence_package']['package_id']

        # Add evidence
        result = agent.run({
            'action': 'add_evidence',
            'evidence': sample_evidence_item.model_dump(),
            'query_filters': {'package_id': package_id},
        })

        assert result.success is True
        assert len(result.data['evidence_package']['evidence_items']) == 1

    def test_finalize_package(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation
    ):
        """Test finalizing an evidence package."""
        agent.register_citation(sample_defra_citation)

        # Create and finalize
        create_result = agent.run({
            'action': 'create_package',
            'package_name': 'Finalize Test',
            'citation_ids': [sample_defra_citation.citation_id],
        })
        package_id = create_result.data['evidence_package']['package_id']

        result = agent.run({
            'action': 'finalize_package',
            'query_filters': {'package_id': package_id},
        })

        assert result.success is True
        assert result.data['evidence_package']['package_hash'] is not None
        assert result.data['provenance_hash'] is not None

    def test_export_citations_bibtex(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation,
        sample_scientific_citation: Citation
    ):
        """Test exporting citations to BibTeX format."""
        agent.register_citation(sample_defra_citation)
        agent.register_citation(sample_scientific_citation)

        result = agent.run({
            'action': 'export_citations',
            'export_format': 'bibtex',
            'citation_ids': [
                sample_defra_citation.citation_id,
                sample_scientific_citation.citation_id,
            ],
        })

        assert result.success is True
        assert '@techreport{' in result.data['exported_content']
        assert '@article{' in result.data['exported_content']

    def test_export_citations_json(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation
    ):
        """Test exporting citations to JSON format."""
        agent.register_citation(sample_defra_citation)

        result = agent.run({
            'action': 'export_citations',
            'export_format': 'json',
        })

        assert result.success is True

        # Should be valid JSON
        data = json.loads(result.data['exported_content'])
        assert isinstance(data, list)
        assert len(data) == 1

    def test_query_citations_by_type(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation,
        sample_scientific_citation: Citation,
        sample_regulatory_citation: Citation
    ):
        """Test querying citations by type."""
        agent.register_citation(sample_defra_citation)
        agent.register_citation(sample_scientific_citation)
        agent.register_citation(sample_regulatory_citation)

        result = agent.run({
            'action': 'query_citations',
            'query_filters': {
                'citation_type': 'emission_factor',
            },
        })

        assert result.success is True
        assert len(result.data['citations']) == 1
        assert result.data['citations'][0]['citation_type'] == 'emission_factor'

    def test_query_citations_by_source(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation,
        sample_scientific_citation: Citation
    ):
        """Test querying citations by source authority."""
        agent.register_citation(sample_defra_citation)
        agent.register_citation(sample_scientific_citation)

        result = agent.run({
            'action': 'query_citations',
            'query_filters': {
                'source_authority': 'defra',
            },
        })

        assert result.success is True
        assert len(result.data['citations']) == 1
        assert result.data['citations'][0]['source_authority'] == 'defra'

    def test_query_citations_by_framework(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation,
        sample_regulatory_citation: Citation
    ):
        """Test querying citations by regulatory framework."""
        agent.register_citation(sample_defra_citation)
        agent.register_citation(sample_regulatory_citation)

        result = agent.run({
            'action': 'query_citations',
            'query_filters': {
                'regulatory_framework': 'csrd',
            },
        })

        assert result.success is True
        # Both have CSRD
        assert len(result.data['citations']) == 2

    def test_query_citations_valid_only(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation
    ):
        """Test querying only valid citations."""
        # Create expired citation
        expired = Citation(
            citation_id="expired",
            citation_type=CitationType.EMISSION_FACTOR,
            source_authority=SourceAuthority.DEFRA,
            metadata=CitationMetadata(title="Expired"),
            effective_date=date(2020, 1, 1),
            expiration_date=date(2020, 12, 31),
        )

        # Valid citation
        sample_defra_citation.effective_date = date.today() - timedelta(days=30)
        sample_defra_citation.expiration_date = date.today() + timedelta(days=30)

        agent.register_citation(expired)
        agent.register_citation(sample_defra_citation)

        result = agent.run({
            'action': 'query_citations',
            'query_filters': {
                'valid_only': True,
            },
        })

        assert result.success is True
        assert len(result.data['citations']) == 1

    def test_get_methodology(self, agent: CitationsEvidenceAgent):
        """Test getting methodology references."""
        result = agent.run({
            'action': 'get_methodology',
            'query_filters': {
                'methodology_id': 'ghg-protocol-corporate',
            },
        })

        assert result.success is True
        assert len(result.data['citations']) == 1
        assert 'GHG Protocol Corporate Standard' in result.data['citations'][0]['metadata']['title']

    def test_get_all_methodologies(self, agent: CitationsEvidenceAgent):
        """Test getting all methodology references."""
        result = agent.run({
            'action': 'get_methodology',
            'query_filters': {},
        })

        assert result.success is True
        assert len(result.data['citations']) >= 3  # At least 3 standard methodologies

    def test_check_validity(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation
    ):
        """Test checking citation validity."""
        sample_defra_citation.effective_date = date.today() - timedelta(days=30)
        sample_defra_citation.expiration_date = date.today() + timedelta(days=30)
        agent.register_citation(sample_defra_citation)

        result = agent.run({
            'action': 'check_validity',
            'citation_ids': [sample_defra_citation.citation_id],
        })

        assert result.success is True
        assert result.data['validity_results'][sample_defra_citation.citation_id] is True

    def test_check_validity_with_reference_date(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation
    ):
        """Test checking citation validity at a specific date."""
        sample_defra_citation.effective_date = date(2024, 1, 1)
        sample_defra_citation.expiration_date = date(2024, 12, 31)
        agent.register_citation(sample_defra_citation)

        # Check at a date when valid
        result = agent.run({
            'action': 'check_validity',
            'citation_ids': [sample_defra_citation.citation_id],
            'query_filters': {
                'reference_date': '2024-06-15',
            },
        })

        assert result.success is True
        assert result.data['validity_results'][sample_defra_citation.citation_id] is True

        # Check at a date when expired
        result2 = agent.run({
            'action': 'check_validity',
            'citation_ids': [sample_defra_citation.citation_id],
            'query_filters': {
                'reference_date': '2025-06-15',
            },
        })

        assert result2.success is True
        assert result2.data['validity_results'][sample_defra_citation.citation_id] is False

    def test_get_citations_for_framework(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation,
        sample_regulatory_citation: Citation
    ):
        """Test getting citations for a regulatory framework."""
        agent.register_citation(sample_defra_citation)
        agent.register_citation(sample_regulatory_citation)

        citations = agent.get_citations_for_framework(RegulatoryFramework.CSRD)

        assert len(citations) == 2

    def test_get_valid_citations(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation
    ):
        """Test getting all valid citations."""
        # Valid
        sample_defra_citation.effective_date = date.today() - timedelta(days=30)
        sample_defra_citation.expiration_date = date.today() + timedelta(days=30)
        agent.register_citation(sample_defra_citation)

        # Expired
        expired = Citation(
            citation_id="expired",
            citation_type=CitationType.EMISSION_FACTOR,
            source_authority=SourceAuthority.DEFRA,
            metadata=CitationMetadata(title="Expired"),
            effective_date=date(2020, 1, 1),
            expiration_date=date(2020, 12, 31),
        )
        agent.register_citation(expired)

        valid_citations = agent.get_valid_citations()

        assert len(valid_citations) == 1
        assert valid_citations[0].citation_id == sample_defra_citation.citation_id

    def test_agent_stats(
        self,
        agent: CitationsEvidenceAgent,
        sample_defra_citation: Citation,
        sample_scientific_citation: Citation
    ):
        """Test agent statistics."""
        agent.register_citation(sample_defra_citation)
        agent.register_citation(sample_scientific_citation)

        stats = agent.get_stats()

        assert stats['total_citations'] == 2
        assert 'citations_by_type' in stats
        assert 'citations_by_status' in stats
        assert stats['total_methodologies'] >= 3

    def test_invalid_action(self, agent: CitationsEvidenceAgent):
        """Test handling of invalid action."""
        result = agent.run({
            'action': 'invalid_action',
        })

        assert result.success is False
        assert 'Invalid action' in result.error

    def test_missing_required_data(self, agent: CitationsEvidenceAgent):
        """Test handling of missing required data."""
        result = agent.run({
            'action': 'register_citation',
            # Missing citation
        })

        assert result.success is False
        assert 'No citation provided' in result.data.get('error', result.error or '')


class TestMethodologyReference:
    """Tests for MethodologyReference model."""

    def test_methodology_creation(self):
        """Test creating a methodology reference."""
        methodology = MethodologyReference(
            reference_id="test-methodology",
            name="Test Methodology",
            standard="GHG Protocol",
            version="2024",
            description="Test methodology for unit tests",
            scope_1_applicable=True,
            scope_2_applicable=True,
            scope_3_applicable=True,
        )

        assert methodology.reference_id == "test-methodology"
        assert methodology.scope_1_applicable is True


class TestRegulatoryRequirement:
    """Tests for RegulatoryRequirement model."""

    def test_requirement_creation(self):
        """Test creating a regulatory requirement."""
        requirement = RegulatoryRequirement(
            requirement_id="csrd-art-29b",
            framework=RegulatoryFramework.CSRD,
            article="Article 29b",
            requirement_text="Companies shall disclose information about scope 1, 2, and 3 emissions",
            effective_date=date(2024, 1, 1),
            compliance_deadline=date(2025, 1, 1),
            applies_to_scope_1=True,
            applies_to_scope_2=True,
            applies_to_scope_3=True,
        )

        assert requirement.framework == RegulatoryFramework.CSRD
        assert requirement.applies_to_scope_3 is True


class TestDataSourceAttribution:
    """Tests for DataSourceAttribution model."""

    def test_attribution_creation(self):
        """Test creating a data source attribution."""
        attribution = DataSourceAttribution(
            attribution_id="defra-2024-electricity",
            source_authority=SourceAuthority.DEFRA,
            dataset_name="UK Government GHG Conversion Factors",
            dataset_version="2024",
            extracted_values={
                "electricity_uk_kwh_kgco2e": 0.207074,
            },
            valid_from=date(2024, 1, 1),
            valid_until=date(2024, 12, 31),
        )

        assert attribution.source_authority == SourceAuthority.DEFRA
        assert attribution.extracted_values["electricity_uk_kwh_kgco2e"] == 0.207074


class TestCitationsAgentInput:
    """Tests for CitationsAgentInput model."""

    def test_valid_actions(self):
        """Test valid action validation."""
        valid_actions = [
            'register_citation',
            'lookup_citation',
            'verify_citation',
            'create_package',
            'export_citations',
        ]

        for action in valid_actions:
            input_data = CitationsAgentInput(action=action)
            assert input_data.action == action

    def test_invalid_action(self):
        """Test invalid action validation."""
        with pytest.raises(ValueError, match="Invalid action"):
            CitationsAgentInput(action="not_a_valid_action")


class TestIntegration:
    """Integration tests for the Citations & Evidence Agent."""

    def test_full_workflow(self, agent: CitationsEvidenceAgent):
        """Test a complete citation and evidence workflow."""
        # 1. Register emission factor citation
        defra_citation = Citation(
            citation_id="defra-2024",
            citation_type=CitationType.EMISSION_FACTOR,
            source_authority=SourceAuthority.DEFRA,
            metadata=CitationMetadata(
                title="DEFRA GHG Conversion Factors 2024",
                version="2024",
            ),
            effective_date=date.today() - timedelta(days=30),
            expiration_date=date.today() + timedelta(days=300),
        )
        agent.register_citation(defra_citation)

        # 2. Register methodology citation
        methodology_citation = Citation(
            citation_id="ghg-protocol-scope2",
            citation_type=CitationType.METHODOLOGY,
            source_authority=SourceAuthority.GHG_PROTOCOL,
            metadata=CitationMetadata(
                title="GHG Protocol Scope 2 Guidance",
                version="2015",
            ),
            effective_date=date(2015, 1, 1),
        )
        agent.register_citation(methodology_citation)

        # 3. Verify citations
        verify_result = agent.run({
            'action': 'verify_sources',
            'citation_ids': [defra_citation.citation_id, methodology_citation.citation_id],
        })
        assert verify_result.success is True

        # 4. Create evidence package
        package = agent.create_evidence_package(
            name="Q1 2024 Scope 2 Emissions",
            calculation_context={
                "period": "Q1 2024",
                "electricity_kwh": 500000,
                "emission_factor": 0.207074,
            },
            calculation_result={
                "emissions_kgco2e": 103537,
                "emissions_tco2e": 103.537,
            },
            citation_ids=[defra_citation.citation_id, methodology_citation.citation_id],
            user_id="test_user",
        )

        assert package is not None
        assert package.package_hash is not None
        assert len(package.citations) == 2

        # 5. Export to BibTeX
        export_result = agent.run({
            'action': 'export_citations',
            'export_format': 'bibtex',
        })
        assert export_result.success is True
        assert '@' in export_result.data['exported_content']

        # 6. Get stats
        stats = agent.get_stats()
        assert stats['total_citations'] == 2
        assert stats['total_packages'] == 1

    def test_regulatory_compliance_workflow(self, agent: CitationsEvidenceAgent):
        """Test regulatory compliance tracking workflow."""
        # 1. Register CSRD regulatory citation
        csrd_citation = Citation(
            citation_id="csrd-2022",
            citation_type=CitationType.REGULATORY,
            source_authority=SourceAuthority.EU_COMMISSION,
            metadata=CitationMetadata(
                title="Corporate Sustainability Reporting Directive (CSRD)",
            ),
            effective_date=date(2024, 1, 1),
            regulatory_frameworks=[RegulatoryFramework.CSRD],
        )
        agent.register_citation(csrd_citation)

        # 2. Query by framework
        query_result = agent.run({
            'action': 'query_citations',
            'query_filters': {
                'regulatory_framework': 'csrd',
            },
        })

        assert query_result.success is True
        assert len(query_result.data['citations']) >= 1

        # 3. Get citations for framework
        framework_citations = agent.get_citations_for_framework(RegulatoryFramework.CSRD)
        assert len(framework_citations) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
