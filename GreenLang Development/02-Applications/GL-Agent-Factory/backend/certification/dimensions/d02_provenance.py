"""
Dimension 02: Provenance Verification

This dimension verifies that agents generate proper SHA-256 provenance
hashes and maintain complete audit trails.

Checks:
    - SHA-256 hash generation
    - Complete audit trail
    - Input to output chain validation
    - Provenance data completeness

Example:
    >>> dimension = ProvenanceDimension()
    >>> result = dimension.evaluate(agent_path, agent, sample_input)
    >>> assert result.status == DimensionStatus.PASS
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseDimension, CheckResult, DimensionResult, DimensionStatus

logger = logging.getLogger(__name__)


class ProvenanceDimension(BaseDimension):
    """
    Provenance Dimension Evaluator (D02).

    Verifies that agents generate proper provenance tracking
    for regulatory audit compliance.

    Configuration:
        require_provenance_hash: Require provenance_hash field (default: True)
        require_source_tracking: Require source field tracking (default: True)
        hash_algorithm: Expected hash algorithm (default: "sha256")
    """

    DIMENSION_ID = "D02"
    DIMENSION_NAME = "Provenance"
    DESCRIPTION = "Verifies SHA-256 provenance hash generation and audit trail"
    WEIGHT = 1.5  # Higher weight - critical for audit compliance
    REQUIRED_FOR_CERTIFICATION = True

    # SHA-256 hash pattern (64 hex characters)
    SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$", re.IGNORECASE)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize provenance dimension evaluator."""
        super().__init__(config)

        self.require_provenance_hash = self.config.get("require_provenance_hash", True)
        self.require_source_tracking = self.config.get("require_source_tracking", True)
        self.hash_algorithm = self.config.get("hash_algorithm", "sha256")

    def evaluate(
        self,
        agent_path: Path,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
    ) -> DimensionResult:
        """
        Evaluate provenance tracking for the given agent.

        Args:
            agent_path: Path to agent directory
            agent: Agent instance with run() method
            sample_input: Sample input for testing

        Returns:
            DimensionResult with provenance evaluation
        """
        start_time = datetime.utcnow()
        self._reset_checks()

        logger.info("Starting provenance evaluation")

        if agent is None:
            agent = self._load_agent(agent_path)

        if agent is None:
            self._add_check(
                name="agent_load",
                passed=False,
                message="Failed to load agent instance",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        if sample_input is None:
            sample_input = self._get_sample_input(agent_path)

        # Run agent to get output
        try:
            result = agent.run(sample_input) if sample_input else None
        except Exception as e:
            self._add_check(
                name="agent_execution",
                passed=False,
                message=f"Agent execution failed: {str(e)}",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        # Check 1: Provenance hash existence
        has_provenance_hash = hasattr(result, "provenance_hash") and result.provenance_hash
        self._add_check(
            name="provenance_hash_exists",
            passed=has_provenance_hash or not self.require_provenance_hash,
            message="Output contains provenance_hash field"
            if has_provenance_hash
            else "Missing provenance_hash field in output",
            severity="error" if self.require_provenance_hash else "warning",
        )

        # Check 2: SHA-256 hash format
        if has_provenance_hash:
            prov_hash = result.provenance_hash
            valid_format = bool(self.SHA256_PATTERN.match(prov_hash))
            self._add_check(
                name="hash_format",
                passed=valid_format,
                message=f"Valid SHA-256 hash format ({prov_hash[:16]}...)"
                if valid_format
                else f"Invalid hash format (length: {len(prov_hash)}, expected: 64)",
                severity="error",
                details={
                    "hash_length": len(prov_hash),
                    "expected_length": 64,
                    "hash_preview": prov_hash[:32] if prov_hash else None,
                },
            )

        # Check 3: Source tracking
        source_fields = self._check_source_fields(result)
        has_sources = len(source_fields) > 0
        self._add_check(
            name="source_tracking",
            passed=has_sources or not self.require_source_tracking,
            message=f"Found {len(source_fields)} source tracking field(s): {', '.join(source_fields)}"
            if has_sources
            else "No source tracking fields found",
            severity="error" if self.require_source_tracking else "warning",
            details={"source_fields": source_fields},
        )

        # Check 4: Source values populated
        if has_sources:
            empty_sources = self._check_empty_sources(result, source_fields)
            all_populated = len(empty_sources) == 0
            self._add_check(
                name="source_values",
                passed=all_populated,
                message="All source fields have values"
                if all_populated
                else f"Empty source fields: {', '.join(empty_sources)}",
                severity="error",
                details={"empty_fields": empty_sources},
            )

        # Check 5: Input-output chain
        chain_valid = self._verify_input_output_chain(sample_input, result, agent)
        self._add_check(
            name="input_output_chain",
            passed=chain_valid,
            message="Input-to-output chain verified"
            if chain_valid
            else "Cannot verify input-output chain",
            severity="warning",
        )

        # Check 6: Agent provenance metadata
        has_agent_metadata = self._check_agent_provenance_metadata(agent)
        self._add_check(
            name="agent_metadata",
            passed=has_agent_metadata,
            message="Agent has provenance metadata (ID, version)"
            if has_agent_metadata
            else "Missing agent provenance metadata",
            severity="warning",
        )

        # Check 7: Timestamp tracking
        has_timestamp = self._check_timestamp_tracking(result)
        self._add_check(
            name="timestamp_tracking",
            passed=has_timestamp,
            message="Output includes calculation timestamp"
            if has_timestamp
            else "Missing timestamp in output",
            severity="warning",
        )

        # Check 8: Reproducible hash calculation
        if has_provenance_hash and hasattr(agent, "_calculate_provenance_hash"):
            # Run twice to verify reproducibility
            try:
                result2 = agent.run(sample_input) if sample_input else None
                if result2 and hasattr(result2, "provenance_hash"):
                    # Note: Hashes may differ due to timestamps
                    # This is a soft check
                    self._add_check(
                        name="hash_method_exists",
                        passed=True,
                        message="Agent has provenance hash calculation method",
                        severity="info",
                    )
            except Exception:
                pass

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return self._create_result(
            execution_time,
            details={
                "has_provenance_hash": has_provenance_hash,
                "source_fields_found": source_fields,
                "hash_preview": (
                    result.provenance_hash[:32] if has_provenance_hash else None
                ),
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

    def _get_sample_input(self, agent_path: Path) -> Optional[Any]:
        """Get sample input from agent path."""
        try:
            import yaml

            pack_file = agent_path / "pack.yaml"
            if pack_file.exists():
                with open(pack_file, "r", encoding="utf-8") as f:
                    pack_spec = yaml.safe_load(f)

                tests = pack_spec.get("tests", {}).get("golden", [])
                if tests:
                    return tests[0].get("input")

            return None

        except Exception as e:
            logger.error(f"Failed to get sample input: {str(e)}")
            return None

    def _check_source_fields(self, result: Any) -> List[str]:
        """
        Check for source tracking fields in output.

        Args:
            result: Agent output

        Returns:
            List of source field names found
        """
        source_patterns = [
            "source",
            "data_source",
            "emission_factor_source",
            "methodology_source",
            "reference",
            "citation",
            "ef_source",
        ]

        found_fields = []

        if hasattr(result, "dict"):
            result_dict = result.dict()
        elif hasattr(result, "model_dump"):
            result_dict = result.model_dump()
        elif hasattr(result, "__dict__"):
            result_dict = result.__dict__
        else:
            return found_fields

        for field_name in result_dict.keys():
            field_lower = field_name.lower()
            if any(pattern in field_lower for pattern in source_patterns):
                found_fields.append(field_name)

        return found_fields

    def _check_empty_sources(self, result: Any, source_fields: List[str]) -> List[str]:
        """
        Check which source fields are empty.

        Args:
            result: Agent output
            source_fields: List of source field names

        Returns:
            List of empty source field names
        """
        empty_fields = []

        if hasattr(result, "dict"):
            result_dict = result.dict()
        elif hasattr(result, "model_dump"):
            result_dict = result.model_dump()
        elif hasattr(result, "__dict__"):
            result_dict = result.__dict__
        else:
            return empty_fields

        for field_name in source_fields:
            value = result_dict.get(field_name)
            if not value or (isinstance(value, str) and not value.strip()):
                empty_fields.append(field_name)

        return empty_fields

    def _verify_input_output_chain(
        self,
        input_data: Any,
        output: Any,
        agent: Any,
    ) -> bool:
        """
        Verify input-to-output chain can be reconstructed.

        Args:
            input_data: Agent input
            output: Agent output
            agent: Agent instance

        Returns:
            True if chain can be verified
        """
        # Check if agent tracks provenance steps
        if hasattr(agent, "_provenance_steps"):
            return True

        # Check if output contains enough information to trace back
        if output is None:
            return False

        if hasattr(output, "dict"):
            output_dict = output.dict()
        elif hasattr(output, "model_dump"):
            output_dict = output.model_dump()
        elif hasattr(output, "__dict__"):
            output_dict = output.__dict__
        else:
            return False

        # Look for calculation traceability fields
        trace_fields = [
            "emission_factor_used",
            "calculation_method",
            "methodology",
            "formula",
        ]

        return any(field in output_dict for field in trace_fields)

    def _check_agent_provenance_metadata(self, agent: Any) -> bool:
        """
        Check if agent has provenance metadata.

        Args:
            agent: Agent instance

        Returns:
            True if agent has required metadata
        """
        has_id = hasattr(agent, "AGENT_ID") or hasattr(agent, "agent_id")
        has_version = hasattr(agent, "VERSION") or hasattr(agent, "version")

        return has_id and has_version

    def _check_timestamp_tracking(self, result: Any) -> bool:
        """
        Check if output includes timestamp.

        Args:
            result: Agent output

        Returns:
            True if timestamp is present
        """
        if result is None:
            return False

        timestamp_fields = [
            "timestamp",
            "calculated_at",
            "created_at",
            "processed_at",
        ]

        if hasattr(result, "dict"):
            result_dict = result.dict()
        elif hasattr(result, "model_dump"):
            result_dict = result.model_dump()
        elif hasattr(result, "__dict__"):
            result_dict = result.__dict__
        else:
            return False

        return any(field in result_dict for field in timestamp_fields)

    def _get_check_remediation(self, check: CheckResult) -> Optional[str]:
        """Get remediation for failed checks."""
        remediation_map = {
            "agent_load": (
                "Ensure agent.py exists and contains a class ending with 'Agent' "
                "that has a run() method."
            ),
            "agent_execution": (
                "Fix agent execution errors. Check sample input format and "
                "agent initialization."
            ),
            "provenance_hash_exists": (
                "Add provenance_hash field to agent output:\n"
                "  1. Create a _calculate_provenance_hash() method\n"
                "  2. Include it in your output model\n"
                "  3. Use hashlib.sha256() for the hash calculation"
            ),
            "hash_format": (
                "Ensure provenance hash is a valid SHA-256 hex string:\n"
                "  hash = hashlib.sha256(data.encode()).hexdigest()\n"
                "  # Returns 64 lowercase hex characters"
            ),
            "source_tracking": (
                "Add source tracking to output:\n"
                "  - emission_factor_source: str (e.g., 'EPA', 'DEFRA')\n"
                "  - data_source: str (for input data origin)\n"
                "  - methodology_source: str (e.g., 'GHG Protocol')"
            ),
            "source_values": (
                "Ensure all source fields have values. Empty sources cannot "
                "be audited. Use 'N/A' if source is not applicable."
            ),
            "input_output_chain": (
                "Add calculation traceability:\n"
                "  - Track _provenance_steps list in agent\n"
                "  - Include emission_factor_used in output\n"
                "  - Include calculation_method in output"
            ),
            "agent_metadata": (
                "Add agent identification:\n"
                "  AGENT_ID = 'category/agent_name_v1'\n"
                "  VERSION = '1.0.0'"
            ),
            "timestamp_tracking": (
                "Add timestamp to output:\n"
                "  calculated_at: datetime = Field(default_factory=datetime.utcnow)"
            ),
        }

        return remediation_map.get(check.name)
