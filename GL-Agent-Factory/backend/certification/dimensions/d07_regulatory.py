"""
Dimension 07: Regulatory Compliance Verification

This dimension verifies that agents align with regulatory standards
and include required fields for compliance reporting.

Checks:
    - Standard alignment (GHG Protocol, ISO 14064)
    - Required fields present
    - Methodology documentation
    - Scope classification

Example:
    >>> dimension = RegulatoryComplianceDimension()
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


class RegulatoryComplianceDimension(BaseDimension):
    """
    Regulatory Compliance Dimension Evaluator (D07).

    Verifies that agents comply with regulatory standards like
    GHG Protocol, ISO 14064, and sector-specific requirements.

    Configuration:
        required_standards: List of standards to check (default: GHG Protocol)
        sector: Sector for sector-specific requirements (default: None)
    """

    DIMENSION_ID = "D07"
    DIMENSION_NAME = "Regulatory Compliance"
    DESCRIPTION = "Verifies alignment with GHG Protocol, ISO 14064, and regulatory requirements"
    WEIGHT = 1.4
    REQUIRED_FOR_CERTIFICATION = True

    # GHG Protocol required fields
    GHG_PROTOCOL_FIELDS = {
        "scope": "GHG Protocol scope (1, 2, or 3)",
        "category": "Emission category",
        "methodology": "Calculation methodology",
        "emission_factor_source": "Source of emission factors",
        "data_quality": "Data quality indicator",
    }

    # ISO 14064 required fields
    ISO_14064_FIELDS = {
        "organizational_boundary": "Organizational boundary definition",
        "operational_boundary": "Operational boundary definition",
        "base_year": "Base year for comparison",
        "recalculation_policy": "Policy for recalculation",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize regulatory compliance dimension evaluator."""
        super().__init__(config)

        self.required_standards = self.config.get(
            "required_standards",
            ["ghg_protocol"],
        )
        self.sector = self.config.get("sector")

    def evaluate(
        self,
        agent_path: Path,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
    ) -> DimensionResult:
        """
        Evaluate regulatory compliance for the given agent.

        Args:
            agent_path: Path to agent directory
            agent: Optional agent instance
            sample_input: Optional sample input

        Returns:
            DimensionResult with regulatory compliance evaluation
        """
        start_time = datetime.utcnow()
        self._reset_checks()

        logger.info("Starting regulatory compliance evaluation")

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

        # Check 1: GHG Protocol alignment
        ghg_check = self._check_ghg_protocol_alignment(source_code, agent)
        self._add_check(
            name="ghg_protocol_alignment",
            passed=ghg_check["aligned"],
            message=f"GHG Protocol: {ghg_check['found_count']}/{len(self.GHG_PROTOCOL_FIELDS)} required fields"
            if ghg_check["found_count"] > 0
            else "No GHG Protocol alignment found",
            severity="error" if "ghg_protocol" in self.required_standards else "warning",
            details=ghg_check,
        )

        # Check 2: Scope classification
        scope_check = self._check_scope_classification(source_code, agent)
        self._add_check(
            name="scope_classification",
            passed=scope_check["has_scope"],
            message=f"Scope classification present: Scope {scope_check['scopes_found']}"
            if scope_check["has_scope"]
            else "No scope classification found",
            severity="error",
            details=scope_check,
        )

        # Check 3: Methodology documentation
        methodology_check = self._check_methodology_documentation(source_code, agent_path)
        self._add_check(
            name="methodology_documented",
            passed=methodology_check["has_methodology"],
            message="Calculation methodology documented"
            if methodology_check["has_methodology"]
            else "Methodology documentation missing",
            severity="error",
            details=methodology_check,
        )

        # Check 4: ISO 14064 alignment (if required)
        if "iso_14064" in self.required_standards:
            iso_check = self._check_iso_14064_alignment(source_code, agent_path)
            self._add_check(
                name="iso_14064_alignment",
                passed=iso_check["aligned"],
                message=f"ISO 14064: {iso_check['found_count']}/{len(self.ISO_14064_FIELDS)} requirements"
                if iso_check["found_count"] > 0
                else "No ISO 14064 alignment found",
                severity="warning",
                details=iso_check,
            )

        # Check 5: Data quality indicators
        quality_check = self._check_data_quality_indicators(source_code, agent)
        self._add_check(
            name="data_quality_indicators",
            passed=quality_check["has_quality"],
            message="Data quality indicators present"
            if quality_check["has_quality"]
            else "No data quality indicators found",
            severity="warning",
            details=quality_check,
        )

        # Check 6: Uncertainty quantification
        uncertainty_check = self._check_uncertainty_quantification(source_code, agent)
        self._add_check(
            name="uncertainty_quantification",
            passed=uncertainty_check["has_uncertainty"],
            message="Uncertainty quantification present"
            if uncertainty_check["has_uncertainty"]
            else "No uncertainty quantification found",
            severity="warning",
            details=uncertainty_check,
        )

        # Check 7: Reporting period handling
        period_check = self._check_reporting_period(source_code)
        self._add_check(
            name="reporting_period",
            passed=period_check["has_period"],
            message="Reporting period handling present"
            if period_check["has_period"]
            else "No reporting period handling found",
            severity="warning",
            details=period_check,
        )

        # Check 8: Audit trail support
        audit_check = self._check_audit_trail_support(source_code, agent)
        self._add_check(
            name="audit_trail_support",
            passed=audit_check["has_audit"],
            message="Audit trail support present"
            if audit_check["has_audit"]
            else "No audit trail support found",
            severity="warning",
            details=audit_check,
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return self._create_result(
            execution_time,
            details={
                "required_standards": self.required_standards,
                "ghg_protocol_fields_found": ghg_check.get("fields_found", []),
                "scopes_supported": scope_check.get("scopes_found", []),
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

    def _check_ghg_protocol_alignment(
        self,
        source_code: str,
        agent: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Check GHG Protocol alignment.

        Args:
            source_code: Agent source code
            agent: Optional agent instance

        Returns:
            Dictionary with GHG Protocol check results
        """
        result = {
            "aligned": False,
            "found_count": 0,
            "fields_found": [],
            "fields_missing": [],
        }

        for field, description in self.GHG_PROTOCOL_FIELDS.items():
            # Check in source code
            pattern = re.compile(rf"\b{field}\b", re.IGNORECASE)
            if pattern.search(source_code):
                result["fields_found"].append(field)
                result["found_count"] += 1
            else:
                result["fields_missing"].append(field)

        # Check for GHG Protocol mention
        if re.search(r"GHG\s*Protocol|Greenhouse\s*Gas\s*Protocol", source_code, re.IGNORECASE):
            result["ghg_protocol_mentioned"] = True

        # Aligned if at least 3 fields present
        result["aligned"] = result["found_count"] >= 3

        return result

    def _check_scope_classification(
        self,
        source_code: str,
        agent: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Check for GHG scope classification.

        Args:
            source_code: Agent source code
            agent: Optional agent instance

        Returns:
            Dictionary with scope classification check results
        """
        result = {
            "has_scope": False,
            "scopes_found": [],
        }

        # Look for Scope enum or field
        scope_patterns = [
            r"class\s+Scope.*Enum",
            r"SCOPE_[123]",
            r"scope\s*[:=]\s*[123]",
            r"Scope\s*\.\s*SCOPE_[123]",
            r"scope\s*=\s*[\"']?[123][\"']?",
        ]

        for pattern in scope_patterns:
            matches = re.findall(pattern, source_code, re.IGNORECASE)
            if matches:
                result["has_scope"] = True
                # Extract scope numbers
                for match in matches:
                    for num in ["1", "2", "3"]:
                        if num in str(match):
                            result["scopes_found"].append(num)

        result["scopes_found"] = sorted(set(result["scopes_found"]))

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
            "has_methodology": False,
            "methodology_types": [],
        }

        # Check in source code
        methodology_patterns = [
            (r"methodology\s*[:=]", "methodology field"),
            (r"calculation_method", "calculation_method field"),
            (r"def\s+.*calculate.*:", "calculate method"),
            (r"#.*formula|#.*methodology", "methodology comment"),
            (r'""".*(?:formula|methodology|calculation).*"""', "methodology docstring"),
        ]

        for pattern, mtype in methodology_patterns:
            if re.search(pattern, source_code, re.IGNORECASE | re.DOTALL):
                result["has_methodology"] = True
                result["methodology_types"].append(mtype)

        # Check for documentation files
        doc_files = ["METHODOLOGY.md", "README.md", "docs/methodology.md"]
        for doc_file in doc_files:
            doc_path = agent_path / doc_file
            if doc_path.exists():
                try:
                    content = doc_path.read_text(encoding="utf-8")
                    if "methodology" in content.lower() or "calculation" in content.lower():
                        result["has_methodology"] = True
                        result["methodology_types"].append(f"{doc_file} documentation")
                except Exception:
                    pass

        return result

    def _check_iso_14064_alignment(
        self,
        source_code: str,
        agent_path: Path,
    ) -> Dict[str, Any]:
        """
        Check ISO 14064 alignment.

        Args:
            source_code: Agent source code
            agent_path: Path to agent directory

        Returns:
            Dictionary with ISO 14064 check results
        """
        result = {
            "aligned": False,
            "found_count": 0,
            "fields_found": [],
            "fields_missing": [],
        }

        for field, description in self.ISO_14064_FIELDS.items():
            pattern = re.compile(rf"\b{field}\b", re.IGNORECASE)
            if pattern.search(source_code):
                result["fields_found"].append(field)
                result["found_count"] += 1
            else:
                result["fields_missing"].append(field)

        # Check for ISO 14064 mention
        if re.search(r"ISO\s*14064", source_code, re.IGNORECASE):
            result["iso_mentioned"] = True

        result["aligned"] = result["found_count"] >= 2

        return result

    def _check_data_quality_indicators(
        self,
        source_code: str,
        agent: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Check for data quality indicators.

        Args:
            source_code: Agent source code
            agent: Optional agent instance

        Returns:
            Dictionary with data quality check results
        """
        result = {
            "has_quality": False,
            "quality_indicators": [],
        }

        quality_patterns = [
            (r"data_quality", "data_quality field"),
            (r"confidence", "confidence score"),
            (r"reliability", "reliability indicator"),
            (r"accuracy", "accuracy indicator"),
            (r"completeness", "completeness indicator"),
        ]

        for pattern, indicator in quality_patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                result["has_quality"] = True
                result["quality_indicators"].append(indicator)

        return result

    def _check_uncertainty_quantification(
        self,
        source_code: str,
        agent: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Check for uncertainty quantification.

        Args:
            source_code: Agent source code
            agent: Optional agent instance

        Returns:
            Dictionary with uncertainty check results
        """
        result = {
            "has_uncertainty": False,
            "uncertainty_types": [],
        }

        uncertainty_patterns = [
            (r"uncertainty", "uncertainty field"),
            (r"confidence_interval", "confidence interval"),
            (r"error_margin", "error margin"),
            (r"uncertainty_pct", "uncertainty percentage"),
            (r"uncertainty_lower|uncertainty_upper", "uncertainty bounds"),
        ]

        for pattern, utype in uncertainty_patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                result["has_uncertainty"] = True
                result["uncertainty_types"].append(utype)

        return result

    def _check_reporting_period(self, source_code: str) -> Dict[str, Any]:
        """
        Check for reporting period handling.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with reporting period check results
        """
        result = {
            "has_period": False,
            "period_types": [],
        }

        period_patterns = [
            (r"reporting_period", "reporting_period field"),
            (r"fiscal_year", "fiscal year"),
            (r"calendar_year", "calendar year"),
            (r"start_date.*end_date", "date range"),
            (r"period\s*[:=]", "period field"),
        ]

        for pattern, ptype in period_patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                result["has_period"] = True
                result["period_types"].append(ptype)

        return result

    def _check_audit_trail_support(
        self,
        source_code: str,
        agent: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Check for audit trail support.

        Args:
            source_code: Agent source code
            agent: Optional agent instance

        Returns:
            Dictionary with audit trail check results
        """
        result = {
            "has_audit": False,
            "audit_features": [],
        }

        audit_patterns = [
            (r"provenance", "provenance tracking"),
            (r"audit_trail", "audit trail"),
            (r"_track_step", "step tracking"),
            (r"logging\.|logger\.", "logging"),
            (r"sha256|hash", "hash verification"),
        ]

        for pattern, feature in audit_patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                result["has_audit"] = True
                result["audit_features"].append(feature)

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
            "ghg_protocol_alignment": (
                "Add GHG Protocol required fields:\n"
                "  - scope: Scope enum (1, 2, or 3)\n"
                "  - category: Emission category\n"
                "  - methodology: 'GHG Protocol' or specific approach\n"
                "  - emission_factor_source: Source attribution"
            ),
            "scope_classification": (
                "Add scope classification:\n"
                "  class Scope(int, Enum):\n"
                "      SCOPE_1 = 1  # Direct emissions\n"
                "      SCOPE_2 = 2  # Indirect (energy)\n"
                "      SCOPE_3 = 3  # Other indirect"
            ),
            "methodology_documented": (
                "Document calculation methodology:\n"
                "  - Add calculation docstrings\n"
                "  - Reference GHG Protocol chapter\n"
                "  - Create METHODOLOGY.md file"
            ),
            "iso_14064_alignment": (
                "Add ISO 14064 required elements:\n"
                "  - organizational_boundary: str\n"
                "  - operational_boundary: str\n"
                "  - base_year: int"
            ),
            "data_quality_indicators": (
                "Add data quality indicators:\n"
                "  data_quality: str = Field(..., description='high/medium/low')\n"
                "  confidence: float = Field(..., ge=0, le=1)"
            ),
            "uncertainty_quantification": (
                "Add uncertainty quantification:\n"
                "  uncertainty_pct: Optional[float] = Field(None, ge=0, le=100)"
            ),
            "reporting_period": (
                "Add reporting period handling:\n"
                "  reporting_period: str = Field(..., description='YYYY or YYYY-MM')"
            ),
            "audit_trail_support": (
                "Add audit trail support:\n"
                "  - Generate provenance_hash\n"
                "  - Track calculation steps\n"
                "  - Add logging statements"
            ),
        }

        return remediation_map.get(check.name)
