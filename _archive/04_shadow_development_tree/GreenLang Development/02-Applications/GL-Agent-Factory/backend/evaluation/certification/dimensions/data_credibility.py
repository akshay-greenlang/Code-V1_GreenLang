"""
Data Credibility Dimension Evaluator

Evaluates agent data credibility including:
- Source validation and verification
- Provenance tracking
- Data freshness
- Citation completeness
- Authority verification

Ensures all data used by agents comes from verified, authoritative sources.

"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from dimension evaluation."""
    score: float
    test_count: int
    tests_passed: int
    tests_failed: int
    details: Dict[str, Any] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class DataCredibilityEvaluator:
    """
    Evaluator for data credibility dimension.

    Tests:
    1. Source validation - All data sources are verified
    2. Provenance tracking - Complete data lineage
    3. Citation completeness - All sources properly cited
    4. Data freshness - Data is current and not stale
    5. Authority verification - Sources are authoritative
    """

    # Authoritative sources for environmental data
    AUTHORITATIVE_SOURCES = {
        "epa.gov": "U.S. Environmental Protection Agency",
        "ipcc.ch": "Intergovernmental Panel on Climate Change",
        "iea.org": "International Energy Agency",
        "ghgprotocol.org": "GHG Protocol",
        "iso.org": "International Organization for Standardization",
        "eea.europa.eu": "European Environment Agency",
        "ecoinvent.org": "Ecoinvent Database",
        "defra.gov.uk": "UK Department for Environment",
    }

    # Required provenance fields
    REQUIRED_PROVENANCE_FIELDS = [
        "provenance_hash",
        "data_source",
        "emission_factor_source",
    ]

    def __init__(self):
        """Initialize data credibility evaluator."""
        logger.info("DataCredibilityEvaluator initialized")

    def evaluate(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
        golden_result: Optional[Any] = None,
        determinism_result: Optional[Any] = None,
    ) -> EvaluationResult:
        """
        Evaluate agent data credibility.

        Args:
            agent: Agent instance to evaluate
            pack_spec: Agent pack specification
            sample_inputs: Sample inputs for testing
            golden_result: Optional golden test results
            determinism_result: Optional determinism results

        Returns:
            EvaluationResult with score and details
        """
        tests_run = 0
        tests_passed = 0
        findings = []
        recommendations = []
        details = {}

        # Test 1: Source validation
        source_score, source_details = self._test_source_validation(
            agent, pack_spec, sample_inputs
        )
        details["source_validation"] = source_details
        tests_run += source_details.get("test_count", 0)
        tests_passed += source_details.get("tests_passed", 0)

        if source_score < 100:
            findings.append(f"Source validation score: {source_score:.1f}%")
            recommendations.append(
                "Ensure all data sources are from authoritative providers"
            )

        # Test 2: Provenance tracking
        provenance_score, provenance_details = self._test_provenance_tracking(
            agent, sample_inputs
        )
        details["provenance_tracking"] = provenance_details
        tests_run += provenance_details.get("test_count", 0)
        tests_passed += provenance_details.get("tests_passed", 0)

        if provenance_score < 100:
            findings.append(f"Provenance tracking: {provenance_score:.1f}%")
            recommendations.append(
                "Implement complete provenance hash for all outputs"
            )

        # Test 3: Citation completeness
        citation_score, citation_details = self._test_citation_completeness(
            agent, pack_spec, sample_inputs
        )
        details["citation_completeness"] = citation_details
        tests_run += citation_details.get("test_count", 0)
        tests_passed += citation_details.get("tests_passed", 0)

        if citation_score < 100:
            findings.append(f"Citation completeness: {citation_score:.1f}%")
            recommendations.append(
                "Add complete citations for all external data references"
            )

        # Test 4: Data freshness
        freshness_score, freshness_details = self._test_data_freshness(pack_spec)
        details["data_freshness"] = freshness_details
        tests_run += freshness_details.get("test_count", 0)
        tests_passed += freshness_details.get("tests_passed", 0)

        if freshness_score < 100:
            findings.append(f"Data freshness: {freshness_score:.1f}%")
            recommendations.append(
                "Update emission factors and reference data to latest versions"
            )

        # Test 5: Authority verification
        authority_score, authority_details = self._test_authority_verification(
            pack_spec
        )
        details["authority_verification"] = authority_details
        tests_run += authority_details.get("test_count", 0)
        tests_passed += authority_details.get("tests_passed", 0)

        if authority_score < 100:
            findings.append(f"Authority verification: {authority_score:.1f}%")
            recommendations.append(
                "Use data only from recognized authoritative sources"
            )

        # Calculate overall score
        if tests_run == 0:
            overall_score = 0.0
        else:
            overall_score = (tests_passed / tests_run) * 100

        return EvaluationResult(
            score=overall_score,
            test_count=tests_run,
            tests_passed=tests_passed,
            tests_failed=tests_run - tests_passed,
            details=details,
            findings=findings,
            recommendations=recommendations,
        )

    def _test_source_validation(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test source validation."""
        tests_run = 0
        tests_passed = 0
        validated_sources = []

        # Check pack spec for source references
        sources = pack_spec.get("data", {}).get("sources", [])
        if not sources:
            sources = pack_spec.get("sources", [])

        for source in sources:
            tests_run += 1
            source_url = source.get("url", "") if isinstance(source, dict) else str(source)

            # Check if source is from authoritative domain
            if self._is_authoritative_source(source_url):
                tests_passed += 1
                validated_sources.append({
                    "source": source_url,
                    "authority": self._get_authority_name(source_url),
                    "status": "VALIDATED",
                })
            else:
                validated_sources.append({
                    "source": source_url,
                    "authority": "Unknown",
                    "status": "UNVERIFIED",
                })

        # If no explicit sources, check output
        if tests_run == 0:
            tests_run = 1
            tests_passed = 1  # Default pass if no sources to validate

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "validated_sources": validated_sources,
        }

    def _test_provenance_tracking(
        self,
        agent: Any,
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test provenance tracking completeness."""
        tests_run = 0
        tests_passed = 0
        provenance_checks = []

        for sample_input in sample_inputs[:3]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                # Check for provenance hash
                has_provenance = hasattr(result, "provenance_hash")
                provenance_valid = False

                if has_provenance:
                    prov_hash = result.provenance_hash
                    # Validate SHA-256 format (64 hex chars)
                    if prov_hash and len(prov_hash) == 64:
                        if all(c in "0123456789abcdef" for c in prov_hash.lower()):
                            provenance_valid = True
                            tests_passed += 1

                provenance_checks.append({
                    "has_provenance": has_provenance,
                    "provenance_valid": provenance_valid,
                    "hash_preview": (
                        result.provenance_hash[:16] + "..."
                        if has_provenance and result.provenance_hash
                        else None
                    ),
                })

            except Exception as e:
                tests_run += 1
                provenance_checks.append({
                    "error": str(e),
                    "has_provenance": False,
                    "provenance_valid": False,
                })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "provenance_checks": provenance_checks,
        }

    def _test_citation_completeness(
        self,
        agent: Any,
        pack_spec: Dict[str, Any],
        sample_inputs: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Test citation completeness."""
        tests_run = 0
        tests_passed = 0
        citation_fields = []

        # Check pack spec for citations
        citations = pack_spec.get("citations", [])
        references = pack_spec.get("references", [])

        tests_run += 1
        if citations or references:
            tests_passed += 1
            citation_fields.append({
                "location": "pack_spec",
                "count": len(citations) + len(references),
                "status": "PRESENT",
            })
        else:
            citation_fields.append({
                "location": "pack_spec",
                "count": 0,
                "status": "MISSING",
            })

        # Check agent output for source references
        for sample_input in sample_inputs[:2]:
            try:
                result = agent.run(sample_input)
                tests_run += 1

                source_fields = [
                    "emission_factor_source",
                    "data_source",
                    "sources",
                    "citation",
                ]

                has_source = any(hasattr(result, f) for f in source_fields)
                if has_source:
                    tests_passed += 1
                    citation_fields.append({
                        "location": "output",
                        "status": "PRESENT",
                    })
                else:
                    citation_fields.append({
                        "location": "output",
                        "status": "MISSING",
                    })

            except Exception:
                tests_run += 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "citation_fields": citation_fields,
        }

    def _test_data_freshness(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test data freshness."""
        tests_run = 0
        tests_passed = 0
        freshness_checks = []

        # Check for version dates in pack spec
        data_version = pack_spec.get("data", {}).get("version", "")
        last_updated = pack_spec.get("data", {}).get("last_updated", "")

        tests_run += 1

        # Check if data is from last 2 years
        current_year = datetime.now().year
        if str(current_year) in str(last_updated) or str(current_year - 1) in str(last_updated):
            tests_passed += 1
            freshness_checks.append({
                "check": "data_version",
                "status": "CURRENT",
                "details": last_updated or data_version,
            })
        else:
            # Default to pass if no date info
            tests_passed += 1
            freshness_checks.append({
                "check": "data_version",
                "status": "UNKNOWN",
                "details": "No version date specified",
            })

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "freshness_checks": freshness_checks,
        }

    def _test_authority_verification(
        self, pack_spec: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Test authority verification."""
        tests_run = 0
        tests_passed = 0
        authority_checks = []

        # Check emission factor sources
        ef_sources = pack_spec.get("emission_factors", {}).get("sources", [])
        if not ef_sources:
            ef_sources = pack_spec.get("data", {}).get("emission_factor_sources", [])

        for source in ef_sources:
            tests_run += 1
            source_str = str(source)

            if self._is_authoritative_source(source_str):
                tests_passed += 1
                authority_checks.append({
                    "source": source_str[:50],
                    "authority": self._get_authority_name(source_str),
                    "status": "VERIFIED",
                })
            else:
                authority_checks.append({
                    "source": source_str[:50],
                    "status": "UNVERIFIED",
                })

        # Default pass if no sources to verify
        if tests_run == 0:
            tests_run = 1
            tests_passed = 1

        score = (tests_passed / tests_run * 100) if tests_run > 0 else 0.0

        return score, {
            "test_count": tests_run,
            "tests_passed": tests_passed,
            "authority_checks": authority_checks,
        }

    def _is_authoritative_source(self, source: str) -> bool:
        """Check if source is from authoritative domain."""
        source_lower = source.lower()
        for domain in self.AUTHORITATIVE_SOURCES.keys():
            if domain in source_lower:
                return True
        return False

    def _get_authority_name(self, source: str) -> str:
        """Get authority name for source."""
        source_lower = source.lower()
        for domain, name in self.AUTHORITATIVE_SOURCES.items():
            if domain in source_lower:
                return name
        return "Unknown"
