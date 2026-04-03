# -*- coding: utf-8 -*-
"""
Batch Code Generator Engine - AGENT-EUDR-014 QR Code Generator (Engine 4)

Production-grade batch code generation engine for EUDR supply chain
tracking. Generates unique, verifiable batch codes with configurable
prefix format, check digit algorithms (Luhn, ISO 7064 Mod 11,10,
CRC-8), and zero-padded sequential numbering.

Batch Code Format:
    {operator}-{commodity}-{year}-{sequence}{check_digit}

    Example: OP001-coffee-2026-00001-7

    Where:
        - operator:   EUDR operator code (from prefix template)
        - commodity:  EUDR commodity type
        - year:       Production or import year
        - sequence:   Zero-padded sequential number
        - check_digit: Computed check character(s)

Capabilities:
    - Batch code generation with operator-commodity-year prefix
    - Sub-batch and unit-level code generation for granular tracking
    - Range generation for bulk code allocation
    - Range reservation with thread-safe sequence management
    - Three check digit algorithms:
        * Luhn (Mod 10): detects single-digit and most transposition errors
        * ISO 7064 Mod 11,10: detects all single and adjacent transpositions
        * CRC-8: 8-bit cyclic redundancy check for longer codes
    - Check digit validation for all three algorithms
    - Flexible prefix format with {operator}, {commodity}, {year} tokens
    - Code-to-entity association tracking (DDS, supply chain node, anchor)
    - SHA-256 provenance for every generated code

Zero-Hallucination Guarantees:
    - All check digit algorithms are fully implemented per their standards
    - Luhn algorithm per ISO/IEC 7812-1
    - ISO 7064 per ISO 7064:2003
    - CRC-8 with standard polynomial 0x07
    - No LLM calls in any code generation path
    - All sequence numbers are deterministic

PRD: PRD-AGENT-EUDR-014 Feature F4 (Batch Code Generation)
Agent ID: GL-EUDR-QRG-014
Regulation: EU 2023/1115 (EUDR) Articles 4, 10, 14

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.qr_code_generator.config import get_config
from greenlang.agents.eudr.qr_code_generator.metrics import (
    record_api_error,
    record_batch_code_generated,
)
from greenlang.agents.eudr.qr_code_generator.models import (
    BatchCode,
    CheckDigitAlgorithm,
    CodeAssociation,
    CodeStatus,
)
from greenlang.agents.eudr.qr_code_generator.provenance import (
    get_provenance_tracker,
)
from greenlang.utilities.exceptions.compliance import ComplianceException

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class BatchCodeError(ComplianceException):
    """Base exception for batch code generation operations."""


class BatchCodeFormatError(BatchCodeError):
    """Raised when a batch code format is invalid."""


class CheckDigitError(BatchCodeError):
    """Raised when check digit computation or validation fails."""


class SequenceExhaustedError(BatchCodeError):
    """Raised when the sequence range is exhausted."""


class CodeReservationError(BatchCodeError):
    """Raised when code range reservation fails."""


# ---------------------------------------------------------------------------
# CRC-8 lookup table (polynomial 0x07)
# ---------------------------------------------------------------------------

_CRC8_TABLE: List[int] = []


def _init_crc8_table() -> None:
    """Initialise the CRC-8 lookup table using polynomial 0x07.

    The polynomial x^8 + x^2 + x^1 + x^0 (0x07) is the standard
    CRC-8 polynomial used in ATM HEC, I.432.1, and ISDN.
    """
    polynomial = 0x07
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ polynomial) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
        _CRC8_TABLE.append(crc)


_init_crc8_table()


# ---------------------------------------------------------------------------
# Sequence manager (thread-safe)
# ---------------------------------------------------------------------------


class _SequenceManager:
    """Thread-safe sequence number manager for batch code generation.

    Maintains per-prefix sequence counters and reserved ranges.
    Uses reentrant locks for thread safety.

    Attributes:
        _sequences: Current sequence number per prefix key.
        _reservations: Reserved ranges per prefix key.
        _lock: Reentrant lock for thread-safe access.
    """

    def __init__(self, start_sequence: int = 1) -> None:
        """Initialize with a default starting sequence number.

        Args:
            start_sequence: Default sequence start (from config).
        """
        self._default_start = start_sequence
        self._sequences: Dict[str, int] = {}
        self._reservations: Dict[str, List[Tuple[int, int]]] = {}
        self._lock = threading.RLock()

    def next_sequence(self, prefix_key: str) -> int:
        """Get the next sequence number for a prefix key.

        Args:
            prefix_key: Unique key combining operator+commodity+year.

        Returns:
            Next available sequence number.
        """
        with self._lock:
            if prefix_key not in self._sequences:
                self._sequences[prefix_key] = self._default_start
            seq = self._sequences[prefix_key]
            self._sequences[prefix_key] = seq + 1
            return seq

    def next_sequence_batch(
        self, prefix_key: str, count: int,
    ) -> List[int]:
        """Get a batch of sequential numbers for a prefix key.

        Args:
            prefix_key: Unique key combining operator+commodity+year.
            count: Number of sequences to allocate.

        Returns:
            List of count sequential numbers.
        """
        with self._lock:
            if prefix_key not in self._sequences:
                self._sequences[prefix_key] = self._default_start
            start = self._sequences[prefix_key]
            self._sequences[prefix_key] = start + count
            return list(range(start, start + count))

    def reserve_range(
        self, prefix_key: str, count: int,
    ) -> Tuple[int, int]:
        """Reserve a range of sequence numbers for future use.

        Args:
            prefix_key: Unique key combining operator+commodity+year.
            count: Number of sequences to reserve.

        Returns:
            Tuple of (start, end) inclusive sequence numbers.
        """
        with self._lock:
            if prefix_key not in self._sequences:
                self._sequences[prefix_key] = self._default_start
            start = self._sequences[prefix_key]
            end = start + count - 1
            self._sequences[prefix_key] = end + 1

            if prefix_key not in self._reservations:
                self._reservations[prefix_key] = []
            self._reservations[prefix_key].append((start, end))

            return (start, end)

    def get_current(self, prefix_key: str) -> int:
        """Get the current sequence number without advancing.

        Args:
            prefix_key: Unique key.

        Returns:
            Current sequence number.
        """
        with self._lock:
            return self._sequences.get(prefix_key, self._default_start)

    def get_reservations(
        self, prefix_key: str,
    ) -> List[Tuple[int, int]]:
        """Get all reservations for a prefix key.

        Args:
            prefix_key: Unique key.

        Returns:
            List of (start, end) tuples.
        """
        with self._lock:
            return list(self._reservations.get(prefix_key, []))


# ===========================================================================
# BatchCodeGenerator
# ===========================================================================


class BatchCodeGenerator:
    """Batch code generation engine for EUDR supply chain tracking.

    Generates unique batch codes with configurable prefix format,
    check digit algorithms, and sequential numbering. All code
    generation is fully deterministic.

    Thread-safe: uses a per-instance _SequenceManager with locks
    for concurrent code generation.

    Attributes:
        _cfg: Reference to the QR Code Generator configuration singleton.
        _seq_mgr: Thread-safe sequence number manager.

    Example:
        >>> gen = BatchCodeGenerator()
        >>> code = gen.generate_batch_code(
        ...     operator_code="OP001",
        ...     commodity_code="coffee",
        ...     year=2026,
        ... )
        >>> assert code.code_value.startswith("OP001-coffee-2026-")
        >>> assert gen.validate_check_digit(code.code_value, "luhn")
    """

    def __init__(self) -> None:
        """Initialize BatchCodeGenerator with config and sequence manager."""
        self._cfg = get_config()
        self._seq_mgr = _SequenceManager(self._cfg.start_sequence)
        logger.info(
            "BatchCodeGenerator initialized: prefix=%s, "
            "check_digit=%s, padding=%d, start=%d",
            self._cfg.default_prefix_format,
            self._cfg.check_digit_algorithm,
            self._cfg.code_padding,
            self._cfg.start_sequence,
        )

    # ------------------------------------------------------------------
    # Public API: generate_batch_code
    # ------------------------------------------------------------------

    def generate_batch_code(
        self,
        operator_code: str,
        commodity_code: str,
        year: int,
        sequence: Optional[int] = None,
        check_digit_algo: Optional[CheckDigitAlgorithm] = None,
        facility_id: Optional[str] = None,
        quantity: Optional[float] = None,
        quantity_unit: Optional[str] = None,
        operator_id: Optional[str] = None,
    ) -> BatchCode:
        """Generate a single batch code with check digit.

        Args:
            operator_code: Operator identifier code for the prefix.
            commodity_code: Commodity type code for the prefix.
            year: Production or import year (2020-2100).
            sequence: Optional explicit sequence number. If None,
                auto-increments from the sequence manager.
            check_digit_algo: Check digit algorithm override.
                Defaults to config.
            facility_id: Optional facility identifier.
            quantity: Optional batch quantity.
            quantity_unit: Optional unit of measurement.
            operator_id: EUDR operator identifier for provenance.
                Defaults to operator_code.

        Returns:
            BatchCode with generated code value and metadata.

        Raises:
            BatchCodeFormatError: If input parameters are invalid.
            CheckDigitError: If check digit computation fails.
        """
        start_time = time.monotonic()

        # Validate inputs
        self._validate_inputs(operator_code, commodity_code, year)

        # Resolve check digit algorithm
        algo = self._resolve_algorithm(check_digit_algo)

        # Build prefix
        prefix = self._build_prefix(operator_code, commodity_code, year)
        prefix_key = f"{operator_code}:{commodity_code}:{year}"

        # Get sequence number
        if sequence is None:
            sequence = self._seq_mgr.next_sequence(prefix_key)

        # Format sequence with padding
        padded_seq = str(sequence).zfill(self._cfg.code_padding)

        # Build the code string (without check digit)
        code_base = f"{prefix}-{padded_seq}"

        # Compute check digit
        check_digit = self._compute_check_digit(code_base, algo)

        # Full code value
        code_value = f"{code_base}-{check_digit}"

        # Resolve operator_id
        op_id = operator_id or operator_code

        # Build BatchCode model
        batch_code = BatchCode(
            code_value=code_value,
            prefix=prefix,
            sequence_number=sequence,
            check_digit=check_digit,
            check_digit_algorithm=algo,
            operator_id=op_id,
            commodity=commodity_code,
            year=year,
            facility_id=facility_id,
            quantity=quantity,
            quantity_unit=quantity_unit,
        )

        # Provenance tracking
        tracker = get_provenance_tracker()
        prov_entry = tracker.record(
            entity_type="batch_code",
            action="generate",
            entity_id=batch_code.batch_code_id,
            data={
                "code_value": code_value,
                "prefix": prefix,
                "sequence": sequence,
                "check_digit": check_digit,
                "algorithm": algo.value if isinstance(algo, CheckDigitAlgorithm) else str(algo),
            },
            metadata={
                "operator_id": op_id,
                "commodity": commodity_code,
                "year": year,
            },
        )
        batch_code.provenance_hash = prov_entry.hash_value

        # Metrics
        record_batch_code_generated(commodity_code)

        elapsed = time.monotonic() - start_time
        logger.info(
            "Batch code generated: %s (algo=%s, seq=%d) in %.3fs",
            code_value, algo, sequence, elapsed,
        )

        return batch_code

    # ------------------------------------------------------------------
    # Public API: generate_sub_batch_code
    # ------------------------------------------------------------------

    def generate_sub_batch_code(
        self,
        batch_code: str,
        sub_batch_number: int,
    ) -> str:
        """Generate a sub-batch code from a parent batch code.

        Appends a sub-batch suffix to the parent batch code for
        tracking sub-lots within a batch (e.g., different processing
        lines or packaging runs).

        Format: {batch_code}/S{sub_batch_number:02d}

        Args:
            batch_code: Parent batch code string.
            sub_batch_number: Sub-batch number (1-99).

        Returns:
            Sub-batch code string.

        Raises:
            BatchCodeFormatError: If inputs are invalid.
        """
        if not batch_code:
            raise BatchCodeFormatError("batch_code must not be empty")
        if sub_batch_number < 1 or sub_batch_number > 99:
            raise BatchCodeFormatError(
                f"sub_batch_number must be 1-99, got {sub_batch_number}"
            )

        sub_code = f"{batch_code}/S{sub_batch_number:02d}"

        logger.debug(
            "Sub-batch code generated: %s -> %s",
            batch_code, sub_code,
        )
        return sub_code

    # ------------------------------------------------------------------
    # Public API: generate_unit_code
    # ------------------------------------------------------------------

    def generate_unit_code(
        self,
        batch_code: str,
        sub_batch_number: int,
        unit_number: int,
    ) -> str:
        """Generate a unit-level code from a batch and sub-batch.

        Creates the most granular code level for individual product
        unit tracking within a sub-batch.

        Format: {batch_code}/S{sub_batch:02d}/U{unit:04d}

        Args:
            batch_code: Parent batch code string.
            sub_batch_number: Sub-batch number (1-99).
            unit_number: Unit number (1-9999).

        Returns:
            Unit code string.

        Raises:
            BatchCodeFormatError: If inputs are invalid.
        """
        if not batch_code:
            raise BatchCodeFormatError("batch_code must not be empty")
        if sub_batch_number < 1 or sub_batch_number > 99:
            raise BatchCodeFormatError(
                f"sub_batch_number must be 1-99, got {sub_batch_number}"
            )
        if unit_number < 1 or unit_number > 9999:
            raise BatchCodeFormatError(
                f"unit_number must be 1-9999, got {unit_number}"
            )

        unit_code = (
            f"{batch_code}/S{sub_batch_number:02d}/U{unit_number:04d}"
        )

        logger.debug(
            "Unit code generated: %s/S%02d/U%04d",
            batch_code, sub_batch_number, unit_number,
        )
        return unit_code

    # ------------------------------------------------------------------
    # Public API: generate_code_range
    # ------------------------------------------------------------------

    def generate_code_range(
        self,
        operator_code: str,
        commodity_code: str,
        year: int,
        start: Optional[int] = None,
        count: int = 10,
        check_digit_algo: Optional[CheckDigitAlgorithm] = None,
        operator_id: Optional[str] = None,
    ) -> List[BatchCode]:
        """Generate a range of sequential batch codes.

        Args:
            operator_code: Operator code for the prefix.
            commodity_code: Commodity code for the prefix.
            year: Production year.
            start: Optional starting sequence. If None, continues from
                current sequence.
            count: Number of codes to generate (1-10000).
            check_digit_algo: Check digit algorithm override.
            operator_id: EUDR operator identifier.

        Returns:
            List of BatchCode objects.

        Raises:
            BatchCodeFormatError: If count is out of range.
        """
        start_time = time.monotonic()

        if count < 1 or count > 10000:
            raise BatchCodeFormatError(
                f"count must be 1-10000, got {count}"
            )

        self._validate_inputs(operator_code, commodity_code, year)

        prefix_key = f"{operator_code}:{commodity_code}:{year}"
        algo = self._resolve_algorithm(check_digit_algo)

        # Allocate sequence range
        if start is not None:
            sequences = list(range(start, start + count))
        else:
            sequences = self._seq_mgr.next_sequence_batch(
                prefix_key, count,
            )

        # Generate codes
        codes: List[BatchCode] = []
        for seq in sequences:
            code = self.generate_batch_code(
                operator_code=operator_code,
                commodity_code=commodity_code,
                year=year,
                sequence=seq,
                check_digit_algo=algo,
                operator_id=operator_id,
            )
            codes.append(code)

        elapsed = time.monotonic() - start_time
        logger.info(
            "Batch code range generated: %d codes for %s/%s/%d "
            "in %.3fs",
            count, operator_code, commodity_code, year, elapsed,
        )

        return codes

    # ------------------------------------------------------------------
    # Public API: reserve_code_range
    # ------------------------------------------------------------------

    def reserve_code_range(
        self,
        operator_code: str,
        commodity_code: str,
        year: int,
        count: int,
    ) -> Dict[str, Any]:
        """Reserve a range of sequence numbers for future code generation.

        Reserves sequential numbers without generating the codes,
        ensuring no other concurrent generator will use these sequences.

        Args:
            operator_code: Operator code.
            commodity_code: Commodity code.
            year: Production year.
            count: Number of sequences to reserve.

        Returns:
            Dictionary with reservation details:
                - prefix_key: The prefix key used.
                - start_sequence: First reserved sequence.
                - end_sequence: Last reserved sequence.
                - count: Number of sequences reserved.
                - reserved_at: UTC timestamp.

        Raises:
            CodeReservationError: If reservation fails.
        """
        if count < 1 or count > 100000:
            raise CodeReservationError(
                f"count must be 1-100000, got {count}"
            )

        self._validate_inputs(operator_code, commodity_code, year)

        prefix_key = f"{operator_code}:{commodity_code}:{year}"
        start, end = self._seq_mgr.reserve_range(prefix_key, count)

        reservation = {
            "prefix_key": prefix_key,
            "start_sequence": start,
            "end_sequence": end,
            "count": count,
            "reserved_at": datetime.now(timezone.utc).isoformat(),
        }

        # Provenance
        tracker = get_provenance_tracker()
        tracker.record(
            entity_type="batch_code",
            action="generate",
            entity_id=f"reservation:{prefix_key}:{start}-{end}",
            data=reservation,
            metadata={
                "operator_code": operator_code,
                "commodity_code": commodity_code,
                "year": year,
            },
        )

        logger.info(
            "Code range reserved: %s sequences %d-%d (%d codes)",
            prefix_key, start, end, count,
        )

        return reservation

    # ------------------------------------------------------------------
    # Public API: Check digit algorithms
    # ------------------------------------------------------------------

    def calculate_luhn_check_digit(self, code_string: str) -> str:
        """Calculate the Luhn (Mod 10) check digit for a code string.

        Implementation per ISO/IEC 7812-1. Processes only the numeric
        characters in the code string.

        Algorithm:
            1. Extract digits from the input string.
            2. Starting from the rightmost digit, double every second digit.
            3. If doubling results in a value > 9, subtract 9.
            4. Sum all digits.
            5. Check digit = (10 - (sum mod 10)) mod 10.

        Args:
            code_string: Code string (non-digit characters are ignored).

        Returns:
            Single check digit character ('0'-'9').

        Raises:
            CheckDigitError: If code_string contains no digits.
        """
        digits = [int(ch) for ch in code_string if ch.isdigit()]
        if not digits:
            raise CheckDigitError(
                f"No numeric digits found in code string: '{code_string}'"
            )

        # Luhn algorithm
        total = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 0:
                # Double even-positioned digits (0-indexed from right)
                doubled = d * 2
                if doubled > 9:
                    doubled -= 9
                total += doubled
            else:
                total += d

        check = (10 - (total % 10)) % 10
        return str(check)

    def calculate_iso7064_check_digit(self, code_string: str) -> str:
        """Calculate the ISO 7064 Mod 11,10 hybrid check digit.

        Implementation per ISO 7064:2003 Section 8. This is the same
        algorithm used in GS1 identifiers.

        Algorithm:
            1. Extract digits from the input string.
            2. Initialize P = 10.
            3. For each digit d:
               a. S = (P + d) mod 10; if S == 0, S = 10
               b. P = (2 * S) mod 11
            4. Check digit = (11 - P) mod 10

        Args:
            code_string: Code string (non-digit characters are ignored).

        Returns:
            Single check digit character ('0'-'9').

        Raises:
            CheckDigitError: If code_string contains no digits.
        """
        digits = [int(ch) for ch in code_string if ch.isdigit()]
        if not digits:
            raise CheckDigitError(
                f"No numeric digits found in code string: '{code_string}'"
            )

        p = 10
        for d in digits:
            s = (p + d) % 10
            if s == 0:
                s = 10
            p = (2 * s) % 11

        check = (11 - p) % 10
        return str(check)

    def calculate_crc8_check_digit(self, code_bytes: bytes) -> str:
        """Calculate the CRC-8 check digit for binary data.

        Uses the standard CRC-8 polynomial 0x07 (x^8 + x^2 + x^1 + 1).
        The result is a two-character hexadecimal string.

        Args:
            code_bytes: Input bytes for CRC computation.

        Returns:
            Two-character uppercase hexadecimal CRC-8 string.

        Raises:
            CheckDigitError: If code_bytes is empty.
        """
        if not code_bytes:
            raise CheckDigitError("code_bytes must not be empty")

        crc = 0x00
        for byte in code_bytes:
            crc = _CRC8_TABLE[crc ^ byte]

        return f"{crc:02X}"

    # ------------------------------------------------------------------
    # Public API: validate_check_digit
    # ------------------------------------------------------------------

    def validate_check_digit(
        self,
        code_string: str,
        algorithm: str,
    ) -> bool:
        """Validate the check digit of a batch code string.

        Parses the code string to extract the check digit (characters
        after the last hyphen) and recomputes it to verify correctness.

        Args:
            code_string: Full batch code string including check digit.
            algorithm: Algorithm name ("luhn", "iso7064_mod11_10", "crc8").

        Returns:
            True if the check digit is valid.

        Raises:
            CheckDigitError: If the algorithm is unknown or code is
                malformed.
        """
        if not code_string:
            raise CheckDigitError("code_string must not be empty")

        # Split at last hyphen to separate code base from check digit
        parts = code_string.rsplit("-", 1)
        if len(parts) != 2:
            raise CheckDigitError(
                f"Cannot parse check digit from code: '{code_string}'"
            )

        code_base = parts[0]
        provided_check = parts[1]

        algo_lower = algorithm.lower().strip()

        if algo_lower == "luhn":
            expected = self.calculate_luhn_check_digit(code_base)
        elif algo_lower == "iso7064_mod11_10":
            expected = self.calculate_iso7064_check_digit(code_base)
        elif algo_lower == "crc8":
            expected = self.calculate_crc8_check_digit(
                code_base.encode("utf-8"),
            )
        else:
            raise CheckDigitError(
                f"Unknown check digit algorithm: '{algorithm}'; "
                f"valid: luhn, iso7064_mod11_10, crc8"
            )

        is_valid = provided_check == expected

        logger.debug(
            "Check digit validation: code=%s algo=%s "
            "provided=%s expected=%s valid=%s",
            code_string, algo_lower,
            provided_check, expected, is_valid,
        )

        return is_valid

    # ------------------------------------------------------------------
    # Public API: format_code
    # ------------------------------------------------------------------

    def format_code(
        self,
        prefix_template: str,
        sequence: int,
        padding: Optional[int] = None,
        check_digit: Optional[str] = None,
        **kwargs: str,
    ) -> str:
        """Format a code string from a prefix template and sequence.

        The prefix_template may contain {operator}, {commodity}, {year}
        tokens that are replaced from kwargs.

        Args:
            prefix_template: Template string with {} tokens.
            sequence: Sequence number.
            padding: Zero-pad width. Defaults to config.
            check_digit: Pre-computed check digit to append.
            **kwargs: Token values for template substitution.

        Returns:
            Formatted code string.
        """
        pad = padding if padding is not None else self._cfg.code_padding
        padded_seq = str(sequence).zfill(pad)

        # Substitute tokens in prefix template
        prefix = prefix_template
        for key, value in kwargs.items():
            prefix = prefix.replace(f"{{{key}}}", str(value))

        code = f"{prefix}-{padded_seq}"

        if check_digit:
            code = f"{code}-{check_digit}"

        return code

    # ------------------------------------------------------------------
    # Public API: associate_code
    # ------------------------------------------------------------------

    def associate_code(
        self,
        batch_code: str,
        dds_reference: Optional[str] = None,
        supply_chain_node_id: Optional[str] = None,
        anchor_id: Optional[str] = None,
        operator_id: Optional[str] = None,
    ) -> CodeAssociation:
        """Create an association between a batch code and external entities.

        Links a batch code to a DDS reference number, supply chain node,
        or blockchain anchor for complete traceability.

        Args:
            batch_code: Batch code string to associate.
            dds_reference: Due Diligence Statement reference.
            supply_chain_node_id: Supply chain node identifier.
            anchor_id: Blockchain anchor identifier.
            operator_id: EUDR operator identifier.

        Returns:
            CodeAssociation record.

        Raises:
            BatchCodeError: If batch_code is empty.
        """
        if not batch_code:
            raise BatchCodeError("batch_code must not be empty")

        # Determine entity type and ID for the association
        if dds_reference:
            entity_type = "dds_statement"
            entity_id = dds_reference
        elif supply_chain_node_id:
            entity_type = "supply_chain_node"
            entity_id = supply_chain_node_id
        elif anchor_id:
            entity_type = "blockchain_anchor"
            entity_id = anchor_id
        else:
            entity_type = "batch_code"
            entity_id = batch_code

        metadata: Dict[str, Any] = {
            "batch_code": batch_code,
        }
        if dds_reference:
            metadata["dds_reference"] = dds_reference
        if supply_chain_node_id:
            metadata["supply_chain_node_id"] = supply_chain_node_id
        if anchor_id:
            metadata["anchor_id"] = anchor_id

        association = CodeAssociation(
            code_id=batch_code,
            entity_type=entity_type,
            entity_id=entity_id,
            relationship="primary",
            metadata=metadata,
        )

        # Provenance
        tracker = get_provenance_tracker()
        prov_entry = tracker.record(
            entity_type="association",
            action="generate",
            entity_id=association.association_id,
            data={
                "batch_code": batch_code,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "dds_reference": dds_reference,
                "supply_chain_node_id": supply_chain_node_id,
                "anchor_id": anchor_id,
            },
            metadata={"operator_id": operator_id or "system"},
        )
        association.provenance_hash = prov_entry.hash_value

        logger.info(
            "Code association created: batch=%s -> %s:%s",
            batch_code, entity_type, entity_id,
        )

        return association

    # ==================================================================
    # Internal: input validation
    # ==================================================================

    def _validate_inputs(
        self,
        operator_code: str,
        commodity_code: str,
        year: int,
    ) -> None:
        """Validate batch code generation inputs.

        Raises:
            BatchCodeFormatError: If any input is invalid.
        """
        if not operator_code or not operator_code.strip():
            raise BatchCodeFormatError(
                "operator_code must not be empty"
            )
        if not commodity_code or not commodity_code.strip():
            raise BatchCodeFormatError(
                "commodity_code must not be empty"
            )
        if year < 2020 or year > 2100:
            raise BatchCodeFormatError(
                f"year must be 2020-2100, got {year}"
            )

        # Validate no separator characters in codes
        for char in ["-", "/", " "]:
            if char in operator_code:
                raise BatchCodeFormatError(
                    f"operator_code must not contain '{char}'"
                )
            if char in commodity_code:
                raise BatchCodeFormatError(
                    f"commodity_code must not contain '{char}'"
                )

    # ==================================================================
    # Internal: prefix construction
    # ==================================================================

    def _build_prefix(
        self,
        operator_code: str,
        commodity_code: str,
        year: int,
    ) -> str:
        """Build the batch code prefix from the config template.

        Replaces {operator}, {commodity}, {year} tokens in the
        default_prefix_format config field.

        Args:
            operator_code: Operator code.
            commodity_code: Commodity code.
            year: Year value.

        Returns:
            Formatted prefix string.
        """
        prefix = self._cfg.default_prefix_format
        prefix = prefix.replace("{operator}", operator_code)
        prefix = prefix.replace("{commodity}", commodity_code)
        prefix = prefix.replace("{year}", str(year))
        return prefix

    # ==================================================================
    # Internal: algorithm resolution
    # ==================================================================

    def _resolve_algorithm(
        self,
        algo: Optional[CheckDigitAlgorithm],
    ) -> CheckDigitAlgorithm:
        """Resolve check digit algorithm from argument or config."""
        if algo is not None:
            return algo
        cfg_algo = self._cfg.check_digit_algorithm.lower().strip()
        return CheckDigitAlgorithm(cfg_algo)

    # ==================================================================
    # Internal: check digit computation
    # ==================================================================

    def _compute_check_digit(
        self,
        code_base: str,
        algorithm: CheckDigitAlgorithm,
    ) -> str:
        """Compute check digit for a code string using the given algorithm.

        Args:
            code_base: Code string without check digit.
            algorithm: Check digit algorithm to use.

        Returns:
            Check digit string.

        Raises:
            CheckDigitError: If computation fails.
        """
        algo_str = (
            algorithm.value
            if isinstance(algorithm, CheckDigitAlgorithm)
            else str(algorithm)
        ).lower()

        try:
            if algo_str == "luhn":
                return self.calculate_luhn_check_digit(code_base)
            elif algo_str == "iso7064_mod11_10":
                return self.calculate_iso7064_check_digit(code_base)
            elif algo_str == "crc8":
                return self.calculate_crc8_check_digit(
                    code_base.encode("utf-8"),
                )
            else:
                raise CheckDigitError(
                    f"Unknown check digit algorithm: '{algo_str}'"
                )
        except CheckDigitError:
            raise
        except Exception as exc:
            raise CheckDigitError(
                f"Check digit computation failed for algorithm "
                f"'{algo_str}': {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "BatchCodeGenerator",
    "BatchCodeError",
    "BatchCodeFormatError",
    "CheckDigitError",
    "SequenceExhaustedError",
    "CodeReservationError",
]
