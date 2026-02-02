"""
Dimension 03: Zero-Hallucination Verification

This dimension verifies that agents do not use LLM calls in their
calculation paths and all values come from verified sources.

Checks:
    - No LLM in calculation path
    - All values from verified sources
    - No interpolation or estimation without disclosure
    - Deterministic calculation methods

Example:
    >>> dimension = ZeroHallucinationDimension()
    >>> result = dimension.evaluate(agent_path, agent, sample_input)
    >>> assert result.status == DimensionStatus.PASS
"""

import ast
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base import BaseDimension, CheckResult, DimensionResult, DimensionStatus

logger = logging.getLogger(__name__)


class ZeroHallucinationDimension(BaseDimension):
    """
    Zero-Hallucination Dimension Evaluator (D03).

    Verifies that agents follow zero-hallucination principles:
    - No LLM calls in numeric calculation paths
    - All emission factors from verified sources
    - No interpolation without disclosure
    - Deterministic formulas only

    Configuration:
        strict_mode: Disallow any LLM usage (default: False)
        allow_llm_classification: Allow LLM for non-numeric tasks (default: True)
    """

    DIMENSION_ID = "D03"
    DIMENSION_NAME = "Zero-Hallucination"
    DESCRIPTION = "Verifies no LLM in calculation path, all values from verified sources"
    WEIGHT = 2.0  # Highest weight - fundamental requirement
    REQUIRED_FOR_CERTIFICATION = True

    # Patterns that indicate LLM usage
    LLM_PATTERNS = [
        r"openai\.",
        r"anthropic\.",
        r"llm\.",
        r"gpt[_-]?[34]",
        r"claude",
        r"chat_completion",
        r"completion\(",
        r"generate\(",
        r"\.ask\(",
        r"\.query\(",
        r"langchain",
        r"llama",
        r"gemini",
        r"palm",
        r"cohere",
    ]

    # Patterns that indicate calculation methods
    CALCULATION_PATTERNS = [
        r"def\s+_?calculate",
        r"def\s+_?compute",
        r"def\s+_?process",
        r"def\s+run\(",
        r"emissions\s*=",
        r"result\s*=.*\*",
        r"total\s*=",
    ]

    # Patterns that indicate interpolation/estimation
    ESTIMATION_PATTERNS = [
        r"interpolate",
        r"estimate",
        r"approximate",
        r"extrapolate",
        r"predict\(",
        r"forecast\(",
        r"\.fit\(",
        r"\.predict\(",
        r"ml_model",
        r"model\.predict",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize zero-hallucination dimension evaluator."""
        super().__init__(config)

        self.strict_mode = self.config.get("strict_mode", False)
        self.allow_llm_classification = self.config.get("allow_llm_classification", True)

    def evaluate(
        self,
        agent_path: Path,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
    ) -> DimensionResult:
        """
        Evaluate zero-hallucination compliance for the given agent.

        Args:
            agent_path: Path to agent directory
            agent: Optional agent instance
            sample_input: Optional sample input

        Returns:
            DimensionResult with zero-hallucination evaluation
        """
        start_time = datetime.utcnow()
        self._reset_checks()

        logger.info("Starting zero-hallucination evaluation")

        # Check 1: Find agent source file
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

        # Read source code
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

        # Check 2: No LLM in calculation path
        llm_issues = self._check_llm_in_calculations(source_code)
        self._add_check(
            name="no_llm_in_calculations",
            passed=len(llm_issues) == 0,
            message="No LLM usage found in calculation paths"
            if len(llm_issues) == 0
            else f"Found {len(llm_issues)} LLM usage(s) in calculation paths",
            severity="error",
            details={"issues": llm_issues[:10]},
        )

        # Check 3: No estimation/interpolation without disclosure
        estimation_issues = self._check_estimation_usage(source_code)
        self._add_check(
            name="no_undisclosed_estimation",
            passed=len(estimation_issues) == 0,
            message="No undisclosed estimation/interpolation found"
            if len(estimation_issues) == 0
            else f"Found {len(estimation_issues)} estimation pattern(s)",
            severity="error" if estimation_issues else "info",
            details={"issues": estimation_issues[:10]},
        )

        # Check 4: Emission factors from verified sources
        ef_issues = self._check_emission_factor_sources(source_code, agent)
        self._add_check(
            name="verified_emission_factors",
            passed=ef_issues["has_sources"],
            message="Emission factors have source attribution"
            if ef_issues["has_sources"]
            else "Emission factors missing source attribution",
            severity="error",
            details=ef_issues,
        )

        # Check 5: Deterministic formulas
        formula_issues = self._check_deterministic_formulas(source_code)
        self._add_check(
            name="deterministic_formulas",
            passed=formula_issues["is_deterministic"],
            message="Calculation formulas are deterministic"
            if formula_issues["is_deterministic"]
            else "Non-deterministic patterns found in formulas",
            severity="error",
            details=formula_issues,
        )

        # Check 6: No random values in calculations
        random_issues = self._check_random_usage(source_code)
        self._add_check(
            name="no_random_in_calculations",
            passed=len(random_issues) == 0,
            message="No random value generation in calculations"
            if len(random_issues) == 0
            else f"Found {len(random_issues)} random usage(s)",
            severity="error",
            details={"issues": random_issues[:10]},
        )

        # Check 7: Calculation path traceability
        trace_check = self._check_calculation_traceability(source_code)
        self._add_check(
            name="calculation_traceability",
            passed=trace_check["traceable"],
            message="Calculation path is traceable"
            if trace_check["traceable"]
            else "Cannot trace calculation path",
            severity="warning",
            details=trace_check,
        )

        # Check 8: Data validation before calculation
        validation_check = self._check_input_validation(source_code)
        self._add_check(
            name="input_validation",
            passed=validation_check["has_validation"],
            message="Input data is validated before calculation"
            if validation_check["has_validation"]
            else "Missing input validation",
            severity="warning",
            details=validation_check,
        )

        # Check 9: Output verification
        output_check = self._check_output_verification(source_code)
        self._add_check(
            name="output_verification",
            passed=output_check["has_verification"],
            message="Output values are verified"
            if output_check["has_verification"]
            else "Missing output verification",
            severity="warning",
            details=output_check,
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return self._create_result(
            execution_time,
            details={
                "llm_issues_found": len(llm_issues),
                "estimation_issues_found": len(estimation_issues),
                "random_issues_found": len(random_issues),
                "strict_mode": self.strict_mode,
            },
        )

    def _check_llm_in_calculations(self, source_code: str) -> List[Dict[str, Any]]:
        """
        Check for LLM usage in calculation paths.

        Args:
            source_code: Agent source code

        Returns:
            List of issues found
        """
        issues = []

        # Find calculation methods
        calc_method_pattern = re.compile(
            r"def\s+(\w*(?:calculate|compute|process|run)\w*)\s*\([^)]*\):\s*(?:\"\"\"[^\"]*\"\"\")?([^def]*?)(?=\n\s*def|\Z)",
            re.DOTALL | re.IGNORECASE,
        )

        calc_methods = calc_method_pattern.findall(source_code)

        for method_name, method_body in calc_methods:
            for pattern in self.LLM_PATTERNS:
                matches = re.finditer(pattern, method_body, re.IGNORECASE)
                for match in matches:
                    issues.append({
                        "method": method_name,
                        "pattern": pattern,
                        "match": match.group(),
                        "severity": "critical",
                    })

        # Also check imports
        import_pattern = re.compile(r"^(?:from|import)\s+(\S+)", re.MULTILINE)
        imports = import_pattern.findall(source_code)

        for imp in imports:
            for pattern in self.LLM_PATTERNS:
                if re.search(pattern, imp, re.IGNORECASE):
                    issues.append({
                        "method": "imports",
                        "pattern": pattern,
                        "match": imp,
                        "severity": "warning",
                    })

        return issues

    def _check_estimation_usage(self, source_code: str) -> List[Dict[str, Any]]:
        """
        Check for estimation/interpolation without proper disclosure.

        Args:
            source_code: Agent source code

        Returns:
            List of issues found
        """
        issues = []

        for pattern in self.ESTIMATION_PATTERNS:
            matches = re.finditer(pattern, source_code, re.IGNORECASE)
            for match in matches:
                # Check if there's a disclosure nearby
                context_start = max(0, match.start() - 200)
                context_end = min(len(source_code), match.end() + 200)
                context = source_code[context_start:context_end]

                has_disclosure = any(
                    word in context.lower()
                    for word in ["uncertainty", "confidence", "estimated", "approximate"]
                )

                if not has_disclosure:
                    issues.append({
                        "pattern": pattern,
                        "match": match.group(),
                        "context": context[:100],
                        "has_disclosure": has_disclosure,
                    })

        return issues

    def _check_emission_factor_sources(
        self,
        source_code: str,
        agent: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Check that emission factors have source attribution.

        Args:
            source_code: Agent source code
            agent: Optional agent instance

        Returns:
            Dictionary with source check results
        """
        result = {
            "has_sources": False,
            "sources_found": [],
            "hardcoded_values": [],
        }

        # Check for source fields in EmissionFactor class or similar
        ef_pattern = re.compile(
            r"(?:emission_factor|ef|factor).*?(?:source|reference|citation)\s*[:=]\s*[\"']([^\"']+)[\"']",
            re.IGNORECASE | re.DOTALL,
        )

        sources = ef_pattern.findall(source_code)
        result["sources_found"] = list(set(sources))
        result["has_sources"] = len(sources) > 0

        # Check for hardcoded numeric values without source
        hardcoded_pattern = re.compile(
            r"(?:emission_factor|ef)\s*[:=]\s*(\d+\.?\d*)\b(?![^{]*source)",
            re.IGNORECASE,
        )

        hardcoded = hardcoded_pattern.findall(source_code)
        result["hardcoded_values"] = hardcoded

        # If agent instance available, check EMISSION_FACTORS dict
        if agent and hasattr(agent, "EMISSION_FACTORS"):
            ef_dict = getattr(agent, "EMISSION_FACTORS", {})
            for fuel_type, regions in ef_dict.items():
                if isinstance(regions, dict):
                    for region, ef in regions.items():
                        if hasattr(ef, "source") and ef.source:
                            result["has_sources"] = True
                            result["sources_found"].append(ef.source)

        return result

    def _check_deterministic_formulas(self, source_code: str) -> Dict[str, Any]:
        """
        Check that calculation formulas are deterministic.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with determinism check results
        """
        result = {
            "is_deterministic": True,
            "non_deterministic_patterns": [],
        }

        # Patterns that indicate non-determinism
        non_det_patterns = [
            (r"random\.", "random module usage"),
            (r"uuid\.", "UUID generation"),
            (r"time\.time\(\)", "time-based values"),
            (r"datetime\.now\(\).*(?:calculation|result|emissions)", "datetime in calculation"),
            (r"os\.urandom", "OS random"),
            (r"numpy\.random", "numpy random"),
        ]

        for pattern, description in non_det_patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                result["is_deterministic"] = False
                result["non_deterministic_patterns"].append(description)

        return result

    def _check_random_usage(self, source_code: str) -> List[Dict[str, Any]]:
        """
        Check for random value usage in calculations.

        Args:
            source_code: Agent source code

        Returns:
            List of random usage issues
        """
        issues = []

        random_patterns = [
            r"random\.\w+\(",
            r"np\.random\.\w+\(",
            r"numpy\.random\.\w+\(",
            r"secrets\.\w+\(",
            r"os\.urandom\(",
        ]

        for pattern in random_patterns:
            matches = re.finditer(pattern, source_code)
            for match in matches:
                # Check if it's setting a seed
                context_start = max(0, match.start() - 50)
                context = source_code[context_start:match.start()]

                if "seed" not in context.lower():
                    issues.append({
                        "pattern": pattern,
                        "match": match.group(),
                        "severity": "error",
                    })

        return issues

    def _check_calculation_traceability(self, source_code: str) -> Dict[str, Any]:
        """
        Check that calculation path is traceable.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with traceability check results
        """
        result = {
            "traceable": False,
            "has_provenance_tracking": False,
            "has_step_logging": False,
        }

        # Check for provenance tracking
        if re.search(r"provenance|_track_step|audit_trail", source_code, re.IGNORECASE):
            result["has_provenance_tracking"] = True

        # Check for step logging
        if re.search(r"logger\.(info|debug).*(?:step|calculating|processing)", source_code, re.IGNORECASE):
            result["has_step_logging"] = True

        result["traceable"] = result["has_provenance_tracking"] or result["has_step_logging"]

        return result

    def _check_input_validation(self, source_code: str) -> Dict[str, Any]:
        """
        Check for input validation before calculation.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with validation check results
        """
        result = {
            "has_validation": False,
            "validation_patterns": [],
        }

        validation_patterns = [
            (r"@validator", "Pydantic validator"),
            (r"validate_", "validation method"),
            (r"if\s+not\s+\w+:", "null check"),
            (r"isinstance\(", "type check"),
            (r"assert\s+", "assertion"),
            (r"raise\s+ValueError", "value error"),
            (r"Field\(.*ge=|le=|gt=|lt=", "Pydantic constraints"),
        ]

        for pattern, description in validation_patterns:
            if re.search(pattern, source_code):
                result["has_validation"] = True
                result["validation_patterns"].append(description)

        return result

    def _check_output_verification(self, source_code: str) -> Dict[str, Any]:
        """
        Check for output verification.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with verification check results
        """
        result = {
            "has_verification": False,
            "verification_patterns": [],
        }

        verification_patterns = [
            (r"_validate_output", "output validation method"),
            (r"if.*emissions.*[<>=]", "bounds checking"),
            (r"assert.*result", "result assertion"),
            (r"Output\(", "typed output model"),
        ]

        for pattern, description in verification_patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                result["has_verification"] = True
                result["verification_patterns"].append(description)

        return result

    def _get_check_remediation(self, check: CheckResult) -> Optional[str]:
        """Get remediation for failed checks."""
        remediation_map = {
            "agent_file_exists": (
                "Create agent.py in the agent directory with the agent implementation."
            ),
            "source_readable": (
                "Ensure agent.py is readable and uses UTF-8 encoding."
            ),
            "no_llm_in_calculations": (
                "Remove LLM calls from calculation paths:\n"
                "  - NEVER use LLM for numeric calculations\n"
                "  - Use database lookups for emission factors\n"
                "  - Use deterministic formulas only\n"
                "  - LLM is OK for classification/categorization only"
            ),
            "no_undisclosed_estimation": (
                "If using estimation/interpolation:\n"
                "  - Add uncertainty_pct field to output\n"
                "  - Document estimation methodology\n"
                "  - Include confidence bounds"
            ),
            "verified_emission_factors": (
                "Add source attribution to emission factors:\n"
                "  class EmissionFactor:\n"
                "      value: float\n"
                "      source: str  # e.g., 'EPA', 'DEFRA', 'IPCC'\n"
                "      year: int"
            ),
            "deterministic_formulas": (
                "Remove non-deterministic elements:\n"
                "  - Don't use random() without fixed seed\n"
                "  - Don't include timestamps in calculation results\n"
                "  - Use consistent rounding (e.g., round(x, 6))"
            ),
            "no_random_in_calculations": (
                "If random is needed, set a fixed seed:\n"
                "  random.seed(42)\n"
                "  Or better: remove randomness entirely"
            ),
            "calculation_traceability": (
                "Add calculation tracing:\n"
                "  self._provenance_steps = []\n"
                "  self._track_step('calculation', {...})"
            ),
            "input_validation": (
                "Add input validation:\n"
                "  - Use Pydantic validators\n"
                "  - Check for None/null values\n"
                "  - Validate ranges (ge=0, le=100)"
            ),
            "output_verification": (
                "Add output verification:\n"
                "  - Validate result is within expected bounds\n"
                "  - Check for NaN/Inf values\n"
                "  - Use typed output models"
            ),
        }

        return remediation_map.get(check.name)
