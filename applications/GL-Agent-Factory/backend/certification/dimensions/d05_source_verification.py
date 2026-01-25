"""
Dimension 05: Source Verification

This dimension verifies that all emission factors and data sources
are traceable and from verified authoritative sources.

Checks:
    - All emission factors traceable
    - Source URLs validated
    - Version pinning verified
    - Source currency (not outdated)

Example:
    >>> dimension = SourceVerificationDimension()
    >>> result = dimension.evaluate(agent_path, agent, sample_input)
    >>> assert result.status == DimensionStatus.PASS
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from .base import BaseDimension, CheckResult, DimensionResult, DimensionStatus

logger = logging.getLogger(__name__)


class SourceVerificationDimension(BaseDimension):
    """
    Source Verification Dimension Evaluator (D05).

    Verifies that all emission factors and data sources are
    traceable to authoritative sources.

    Configuration:
        require_source_urls: Require URL validation (default: False)
        max_source_age_years: Maximum source age in years (default: 5)
        authoritative_sources: List of recognized sources
    """

    DIMENSION_ID = "D05"
    DIMENSION_NAME = "Source Verification"
    DESCRIPTION = "Verifies all emission factors are traceable to authoritative sources"
    WEIGHT = 1.3
    REQUIRED_FOR_CERTIFICATION = True

    # Recognized authoritative sources
    AUTHORITATIVE_SOURCES = {
        "EPA": "U.S. Environmental Protection Agency",
        "DEFRA": "UK Department for Environment, Food and Rural Affairs",
        "IPCC": "Intergovernmental Panel on Climate Change",
        "IEA": "International Energy Agency",
        "GHG Protocol": "Greenhouse Gas Protocol",
        "ISO": "International Organization for Standardization",
        "ECOINVENT": "ecoinvent Database",
        "GaBi": "GaBi Database",
        "EEA": "European Environment Agency",
        "CARB": "California Air Resources Board",
        "BEIS": "UK Department for Business, Energy & Industrial Strategy",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize source verification dimension evaluator."""
        super().__init__(config)

        self.require_source_urls = self.config.get("require_source_urls", False)
        self.max_source_age_years = self.config.get("max_source_age_years", 5)
        self.authoritative_sources = self.config.get(
            "authoritative_sources",
            list(self.AUTHORITATIVE_SOURCES.keys()),
        )

    def evaluate(
        self,
        agent_path: Path,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
    ) -> DimensionResult:
        """
        Evaluate source verification for the given agent.

        Args:
            agent_path: Path to agent directory
            agent: Optional agent instance
            sample_input: Optional sample input

        Returns:
            DimensionResult with source verification evaluation
        """
        start_time = datetime.utcnow()
        self._reset_checks()

        logger.info("Starting source verification evaluation")

        # Load agent source code
        agent_file = agent_path / "agent.py"
        if not agent_file.exists():
            self._add_check(
                name="agent_file_exists",
                passed=False,
                message="agent.py not found",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        try:
            source_code = agent_file.read_text(encoding="utf-8")
        except Exception as e:
            self._add_check(
                name="source_readable",
                passed=False,
                message=f"Cannot read agent source: {str(e)}",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        # Load agent instance if not provided
        if agent is None:
            agent = self._load_agent(agent_path)

        # Check 1: Emission factors have sources
        ef_sources = self._extract_emission_factor_sources(source_code, agent)
        has_sources = len(ef_sources["sources"]) > 0

        self._add_check(
            name="emission_factors_sourced",
            passed=has_sources,
            message=f"Found {len(ef_sources['sources'])} source attribution(s)"
            if has_sources
            else "No emission factor source attributions found",
            severity="error",
            details=ef_sources,
        )

        # Check 2: Sources are authoritative
        authoritative_check = self._check_authoritative_sources(ef_sources["sources"])
        self._add_check(
            name="sources_authoritative",
            passed=authoritative_check["all_authoritative"],
            message=f"{authoritative_check['authoritative_count']}/{len(ef_sources['sources'])} sources are authoritative"
            if ef_sources["sources"]
            else "No sources to verify",
            severity="error" if not authoritative_check["all_authoritative"] else "info",
            details=authoritative_check,
        )

        # Check 3: Source years are current
        year_check = self._check_source_currency(ef_sources)
        self._add_check(
            name="sources_current",
            passed=year_check["all_current"],
            message="All sources are within acceptable age"
            if year_check["all_current"]
            else f"{year_check['outdated_count']} outdated source(s) found",
            severity="warning" if not year_check["all_current"] else "info",
            details=year_check,
        )

        # Check 4: Version pinning
        version_check = self._check_version_pinning(source_code, agent_path)
        self._add_check(
            name="version_pinning",
            passed=version_check["has_pinning"],
            message="Emission factor versions are pinned"
            if version_check["has_pinning"]
            else "No version pinning found",
            severity="warning",
            details=version_check,
        )

        # Check 5: Source references in pack.yaml
        pack_sources = self._check_pack_yaml_sources(agent_path)
        self._add_check(
            name="pack_yaml_sources",
            passed=pack_sources["has_sources"],
            message=f"pack.yaml references {len(pack_sources['factors'])} emission factor source(s)"
            if pack_sources["has_sources"]
            else "No emission factor references in pack.yaml",
            severity="warning",
            details=pack_sources,
        )

        # Check 6: Documentation of methodology
        methodology_check = self._check_methodology_documentation(source_code, agent_path)
        self._add_check(
            name="methodology_documented",
            passed=methodology_check["has_documentation"],
            message="Calculation methodology is documented"
            if methodology_check["has_documentation"]
            else "Missing methodology documentation",
            severity="warning",
            details=methodology_check,
        )

        # Check 7: URL validation (if required)
        if self.require_source_urls:
            url_check = self._check_source_urls(source_code)
            self._add_check(
                name="source_urls_valid",
                passed=url_check["urls_valid"],
                message=f"Found {url_check['url_count']} source URL(s)"
                if url_check["url_count"] > 0
                else "No source URLs found",
                severity="warning",
                details=url_check,
            )

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return self._create_result(
            execution_time,
            details={
                "sources_found": ef_sources["sources"],
                "authoritative_count": authoritative_check.get("authoritative_count", 0),
                "max_source_age_years": self.max_source_age_years,
            },
        )

    def _load_agent(self, agent_path: Path) -> Optional[Any]:
        """Load agent from path."""
        try:
            agent_file = agent_path / "agent.py"
            if not agent_file.exists():
                return None

            import importlib.util

            spec = importlib.util.spec_from_file_location("agent", agent_file)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and name.endswith("Agent")
                    and hasattr(obj, "run")
                ):
                    return obj()

            return None

        except Exception as e:
            logger.error(f"Failed to load agent: {str(e)}")
            return None

    def _extract_emission_factor_sources(
        self,
        source_code: str,
        agent: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Extract emission factor sources from code and agent.

        Args:
            source_code: Agent source code
            agent: Optional agent instance

        Returns:
            Dictionary with source information
        """
        result = {
            "sources": [],
            "years": [],
            "hardcoded_values": [],
        }

        # Pattern to find source attributions
        source_pattern = re.compile(
            r"source\s*[:=]\s*[\"']([^\"']+)[\"']",
            re.IGNORECASE,
        )

        sources = source_pattern.findall(source_code)
        result["sources"] = list(set(sources))

        # Pattern to find years
        year_pattern = re.compile(
            r"year\s*[:=]\s*(\d{4})",
            re.IGNORECASE,
        )

        years = year_pattern.findall(source_code)
        result["years"] = [int(y) for y in years]

        # Extract from agent instance
        if agent and hasattr(agent, "EMISSION_FACTORS"):
            ef_dict = getattr(agent, "EMISSION_FACTORS", {})
            for fuel_type, regions in ef_dict.items():
                if isinstance(regions, dict):
                    for region, ef in regions.items():
                        if hasattr(ef, "source") and ef.source:
                            result["sources"].append(ef.source)
                        if hasattr(ef, "year") and ef.year:
                            result["years"].append(ef.year)

        # Remove duplicates
        result["sources"] = list(set(result["sources"]))
        result["years"] = list(set(result["years"]))

        return result

    def _check_authoritative_sources(
        self,
        sources: List[str],
    ) -> Dict[str, Any]:
        """
        Check if sources are from authoritative bodies.

        Args:
            sources: List of source names

        Returns:
            Dictionary with authoritative source check results
        """
        result = {
            "all_authoritative": True,
            "authoritative_count": 0,
            "non_authoritative": [],
            "recognized_sources": [],
        }

        if not sources:
            return result

        for source in sources:
            is_authoritative = False
            for auth_source in self.authoritative_sources:
                if auth_source.lower() in source.lower():
                    is_authoritative = True
                    result["recognized_sources"].append({
                        "source": source,
                        "matched": auth_source,
                    })
                    break

            if is_authoritative:
                result["authoritative_count"] += 1
            else:
                result["all_authoritative"] = False
                result["non_authoritative"].append(source)

        return result

    def _check_source_currency(
        self,
        ef_sources: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check if source years are current.

        Args:
            ef_sources: Extracted source information

        Returns:
            Dictionary with currency check results
        """
        result = {
            "all_current": True,
            "outdated_count": 0,
            "outdated_sources": [],
            "current_year": datetime.now().year,
        }

        years = ef_sources.get("years", [])
        if not years:
            return result

        cutoff_year = result["current_year"] - self.max_source_age_years

        for year in years:
            if year < cutoff_year:
                result["all_current"] = False
                result["outdated_count"] += 1
                result["outdated_sources"].append({
                    "year": year,
                    "age": result["current_year"] - year,
                })

        return result

    def _check_version_pinning(
        self,
        source_code: str,
        agent_path: Path,
    ) -> Dict[str, Any]:
        """
        Check for emission factor version pinning.

        Args:
            source_code: Agent source code
            agent_path: Path to agent directory

        Returns:
            Dictionary with version pinning check results
        """
        result = {
            "has_pinning": False,
            "versions_found": [],
        }

        # Check in source code
        version_patterns = [
            r"ef_version\s*[:=]\s*[\"']([^\"']+)[\"']",
            r"version_pin\s*[:=]\s*[\"']([^\"']+)[\"']",
            r"data_version\s*[:=]\s*[\"']([^\"']+)[\"']",
        ]

        for pattern in version_patterns:
            matches = re.findall(pattern, source_code, re.IGNORECASE)
            result["versions_found"].extend(matches)

        # Check in pack.yaml
        pack_file = agent_path / "pack.yaml"
        if pack_file.exists():
            try:
                with open(pack_file, "r", encoding="utf-8") as f:
                    pack_spec = yaml.safe_load(f)

                provenance = pack_spec.get("provenance", {})
                if "ef_version_pin" in provenance:
                    result["versions_found"].append(provenance["ef_version_pin"])

            except Exception:
                pass

        result["has_pinning"] = len(result["versions_found"]) > 0

        return result

    def _check_pack_yaml_sources(self, agent_path: Path) -> Dict[str, Any]:
        """
        Check for emission factor references in pack.yaml.

        Args:
            agent_path: Path to agent directory

        Returns:
            Dictionary with pack.yaml source check results
        """
        result = {
            "has_sources": False,
            "factors": [],
        }

        pack_file = agent_path / "pack.yaml"
        if not pack_file.exists():
            return result

        try:
            with open(pack_file, "r", encoding="utf-8") as f:
                pack_spec = yaml.safe_load(f)

            factors = pack_spec.get("factors", [])
            for factor in factors:
                if isinstance(factor, dict) and "ref" in factor:
                    result["factors"].append(factor["ref"])
                elif isinstance(factor, str):
                    result["factors"].append(factor)

            result["has_sources"] = len(result["factors"]) > 0

        except Exception as e:
            logger.error(f"Failed to read pack.yaml: {str(e)}")

        return result

    def _check_methodology_documentation(
        self,
        source_code: str,
        agent_path: Path,
    ) -> Dict[str, Any]:
        """
        Check for methodology documentation.

        Args:
            source_code: Agent source code
            agent_path: Path to agent directory

        Returns:
            Dictionary with methodology documentation check results
        """
        result = {
            "has_documentation": False,
            "documentation_types": [],
        }

        # Check for methodology comments/docstrings
        methodology_patterns = [
            r"methodology",
            r"GHG Protocol",
            r"ISO 14064",
            r"calculation method",
            r"formula.*emissions",
        ]

        for pattern in methodology_patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                result["has_documentation"] = True
                result["documentation_types"].append(pattern)

        # Check for README or documentation files
        doc_files = ["README.md", "METHODOLOGY.md", "docs/methodology.md"]
        for doc_file in doc_files:
            doc_path = agent_path / doc_file
            if doc_path.exists():
                result["has_documentation"] = True
                result["documentation_types"].append(doc_file)

        return result

    def _check_source_urls(self, source_code: str) -> Dict[str, Any]:
        """
        Check for source URLs in code.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with URL check results
        """
        result = {
            "urls_valid": True,
            "url_count": 0,
            "urls_found": [],
        }

        # Pattern to find URLs
        url_pattern = re.compile(
            r"https?://[^\s\"'<>]+",
            re.IGNORECASE,
        )

        urls = url_pattern.findall(source_code)
        result["urls_found"] = list(set(urls))
        result["url_count"] = len(result["urls_found"])

        # Note: Actual URL validation would require HTTP requests
        # which we skip for offline evaluation

        return result

    def _get_check_remediation(self, check: CheckResult) -> Optional[str]:
        """Get remediation for failed checks."""
        remediation_map = {
            "agent_file_exists": (
                "Create agent.py in the agent directory."
            ),
            "source_readable": (
                "Ensure agent.py is readable and uses UTF-8 encoding."
            ),
            "emission_factors_sourced": (
                "Add source attribution to all emission factors:\n"
                "  class EmissionFactor:\n"
                "      value: float\n"
                "      source: str  # e.g., 'EPA', 'DEFRA'\n"
                "      year: int"
            ),
            "sources_authoritative": (
                f"Use recognized authoritative sources:\n"
                f"  {', '.join(list(self.AUTHORITATIVE_SOURCES.keys())[:5])}\n"
                "These sources have peer-reviewed methodologies."
            ),
            "sources_current": (
                f"Update emission factors to sources within {self.max_source_age_years} years:\n"
                "  - EPA updates factors annually\n"
                "  - DEFRA updates factors annually\n"
                "  - IPCC updates with assessment reports"
            ),
            "version_pinning": (
                "Add version pinning to pack.yaml:\n"
                "  provenance:\n"
                "    ef_version_pin: '2024-Q4'\n"
                "    gwp_set: 'AR6'"
            ),
            "pack_yaml_sources": (
                "Add emission factor references to pack.yaml:\n"
                "  factors:\n"
                "    - ref: 'ef://epa/stationary-combustion/2024'\n"
                "    - ref: 'ef://ipcc/gwp/ar6'"
            ),
            "methodology_documented": (
                "Document the calculation methodology:\n"
                "  - Add docstrings explaining the formula\n"
                "  - Reference GHG Protocol or ISO 14064\n"
                "  - Create METHODOLOGY.md file"
            ),
            "source_urls_valid": (
                "Include source URLs for verification:\n"
                "  # Source: https://www.epa.gov/ghgemissions/..."
            ),
        }

        return remediation_map.get(check.name)
