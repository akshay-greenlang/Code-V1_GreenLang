# -*- coding: utf-8 -*-
"""
GS1 Digital Link Specifications - AGENT-EUDR-014

Reference data for GS1 Digital Link URI formatting per GS1 General
Specifications Release 22.0.  Provides Application Identifier (AI)
definitions, GTIN validation, check digit calculation, and Digital
Link URI construction for EUDR QR code compliance labels.

GS1 Digital Links encode product identification (GTIN) plus
additional data attributes into web-resolvable URIs that can be
embedded in QR codes for supply chain traceability.

Includes:
    - GS1_APPLICATION_IDENTIFIERS: Mapping of AI codes to data types
    - GS1_DIGITAL_LINK_BASE_URL: GS1 resolver base URL
    - build_gs1_digital_link_uri: Construct a GS1 Digital Link URI
    - validate_gtin: Validate GTIN-14 including check digit
    - calculate_gtin_check_digit: Compute GTIN check digit
    - AI_DESCRIPTIONS: Human-readable AI descriptions
    - EUDR_RELEVANT_AIS: Application Identifiers relevant to EUDR
    - GS1_URI_SYNTAX_REGEX: Regex pattern for URI validation

Data Sources:
    - GS1 General Specifications, Release 22.0, Section 7.8
    - GS1 Digital Link Standard, Release 1.3
    - ISO/IEC 15459 (unique identification of transport units)
    - ISO/IEC 15418 (GS1 Application Identifiers and ASC MH10 DIs)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
Status: Production Ready
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GS1 Digital Link base URL
# ---------------------------------------------------------------------------

GS1_DIGITAL_LINK_BASE_URL: str = "https://id.gs1.org"


# ---------------------------------------------------------------------------
# GS1 Application Identifiers (AI)
# ---------------------------------------------------------------------------

GS1_APPLICATION_IDENTIFIERS: Dict[str, Dict[str, Any]] = {
    # -- Identification AIs --
    "01": {
        "name": "GTIN",
        "description": "Global Trade Item Number (GTIN-14)",
        "format": "N14",
        "length": 14,
        "is_fixed_length": True,
        "fnc1_required": False,
        "data_title": "GTIN",
        "gs1_key": True,
    },
    "02": {
        "name": "CONTENT",
        "description": "GTIN of contained trade items",
        "format": "N14",
        "length": 14,
        "is_fixed_length": True,
        "fnc1_required": False,
        "data_title": "CONTENT",
        "gs1_key": False,
    },
    # -- Batch / Lot --
    "10": {
        "name": "BATCH/LOT",
        "description": "Batch or lot number",
        "format": "X..20",
        "max_length": 20,
        "is_fixed_length": False,
        "fnc1_required": True,
        "data_title": "BATCH/LOT",
        "gs1_key": False,
    },
    # -- Dates --
    "11": {
        "name": "PROD DATE",
        "description": "Production date (YYMMDD)",
        "format": "N6",
        "length": 6,
        "is_fixed_length": True,
        "fnc1_required": False,
        "data_title": "PROD DATE",
        "gs1_key": False,
    },
    "13": {
        "name": "PACK DATE",
        "description": "Packaging date (YYMMDD)",
        "format": "N6",
        "length": 6,
        "is_fixed_length": True,
        "fnc1_required": False,
        "data_title": "PACK DATE",
        "gs1_key": False,
    },
    "15": {
        "name": "BEST BEFORE",
        "description": "Best before date (YYMMDD)",
        "format": "N6",
        "length": 6,
        "is_fixed_length": True,
        "fnc1_required": False,
        "data_title": "BEST BEFORE or BEST BY",
        "gs1_key": False,
    },
    "17": {
        "name": "USE BY",
        "description": "Expiration date (YYMMDD)",
        "format": "N6",
        "length": 6,
        "is_fixed_length": True,
        "fnc1_required": False,
        "data_title": "USE BY OR EXPIRY",
        "gs1_key": False,
    },
    # -- Serial number --
    "21": {
        "name": "SERIAL",
        "description": "Serial number",
        "format": "X..20",
        "max_length": 20,
        "is_fixed_length": False,
        "fnc1_required": True,
        "data_title": "SERIAL",
        "gs1_key": False,
    },
    # -- Quantity --
    "30": {
        "name": "VAR COUNT",
        "description": "Variable count of items",
        "format": "N..8",
        "max_length": 8,
        "is_fixed_length": False,
        "fnc1_required": True,
        "data_title": "VAR. COUNT",
        "gs1_key": False,
    },
    "37": {
        "name": "COUNT",
        "description": "Count of trade items contained",
        "format": "N..8",
        "max_length": 8,
        "is_fixed_length": False,
        "fnc1_required": True,
        "data_title": "COUNT",
        "gs1_key": False,
    },
    # -- Weight --
    "3100": {
        "name": "NET WEIGHT (kg)",
        "description": "Net weight, kilograms (0 decimal places)",
        "format": "N6",
        "length": 6,
        "is_fixed_length": True,
        "fnc1_required": False,
        "data_title": "NET WEIGHT (kg)",
        "gs1_key": False,
    },
    "3101": {
        "name": "NET WEIGHT (kg)",
        "description": "Net weight, kilograms (1 decimal place)",
        "format": "N6",
        "length": 6,
        "is_fixed_length": True,
        "fnc1_required": False,
        "data_title": "NET WEIGHT (kg)",
        "gs1_key": False,
    },
    "3102": {
        "name": "NET WEIGHT (kg)",
        "description": "Net weight, kilograms (2 decimal places)",
        "format": "N6",
        "length": 6,
        "is_fixed_length": True,
        "fnc1_required": False,
        "data_title": "NET WEIGHT (kg)",
        "gs1_key": False,
    },
    "3103": {
        "name": "NET WEIGHT (kg)",
        "description": "Net weight, kilograms (3 decimal places)",
        "format": "N6",
        "length": 6,
        "is_fixed_length": True,
        "fnc1_required": False,
        "data_title": "NET WEIGHT (kg)",
        "gs1_key": False,
    },
    # -- Country --
    "422": {
        "name": "ORIGIN",
        "description": "Country of origin (ISO 3166 numeric)",
        "format": "N3",
        "length": 3,
        "is_fixed_length": True,
        "fnc1_required": True,
        "data_title": "ORIGIN",
        "gs1_key": False,
    },
    "423": {
        "name": "COUNTRY - INITIAL PROCESS",
        "description": "Country of initial processing",
        "format": "N3..15",
        "max_length": 15,
        "is_fixed_length": False,
        "fnc1_required": True,
        "data_title": "COUNTRY - INITIAL PROCESS.",
        "gs1_key": False,
    },
    "424": {
        "name": "COUNTRY - PROCESS",
        "description": "Country of processing",
        "format": "N3",
        "length": 3,
        "is_fixed_length": True,
        "fnc1_required": True,
        "data_title": "COUNTRY - PROCESS.",
        "gs1_key": False,
    },
    "426": {
        "name": "COUNTRY - FULL PROCESS",
        "description": "Country covering full process chain",
        "format": "N3",
        "length": 3,
        "is_fixed_length": True,
        "fnc1_required": True,
        "data_title": "COUNTRY - FULL PROCESS",
        "gs1_key": False,
    },
    # -- SSCC (shipping) --
    "00": {
        "name": "SSCC",
        "description": "Serial Shipping Container Code",
        "format": "N18",
        "length": 18,
        "is_fixed_length": True,
        "fnc1_required": False,
        "data_title": "SSCC",
        "gs1_key": True,
    },
    # -- GLN (location) --
    "410": {
        "name": "SHIP TO LOC",
        "description": "Ship-to location (GLN)",
        "format": "N13",
        "length": 13,
        "is_fixed_length": True,
        "fnc1_required": True,
        "data_title": "SHIP TO LOC",
        "gs1_key": False,
    },
    "411": {
        "name": "BILL TO",
        "description": "Bill-to / invoice-to (GLN)",
        "format": "N13",
        "length": 13,
        "is_fixed_length": True,
        "fnc1_required": True,
        "data_title": "BILL TO",
        "gs1_key": False,
    },
    "414": {
        "name": "LOC No",
        "description": "Identification of a physical location (GLN)",
        "format": "N13",
        "length": 13,
        "is_fixed_length": True,
        "fnc1_required": True,
        "data_title": "LOC No",
        "gs1_key": True,
    },
    # -- Certification reference --
    "7023": {
        "name": "CERT",
        "description": "Certification reference",
        "format": "X..30",
        "max_length": 30,
        "is_fixed_length": False,
        "fnc1_required": True,
        "data_title": "CERT #",
        "gs1_key": False,
    },
    # -- Internal company reference --
    "240": {
        "name": "ADDITIONAL ID",
        "description": "Additional product identification",
        "format": "X..30",
        "max_length": 30,
        "is_fixed_length": False,
        "fnc1_required": True,
        "data_title": "ADDITIONAL ID",
        "gs1_key": False,
    },
    # -- URL / Digital Link --
    "8200": {
        "name": "PRODUCT URL",
        "description": "Extended packaging URL",
        "format": "X..70",
        "max_length": 70,
        "is_fixed_length": False,
        "fnc1_required": True,
        "data_title": "PRODUCT URL",
        "gs1_key": False,
    },
    # -- Company internal info --
    "91": {
        "name": "INTERNAL",
        "description": "Company internal information",
        "format": "X..90",
        "max_length": 90,
        "is_fixed_length": False,
        "fnc1_required": True,
        "data_title": "INTERNAL",
        "gs1_key": False,
    },
    # -- Mutual agreed info --
    "90": {
        "name": "INTERNAL",
        "description": "Information mutually agreed between partners",
        "format": "X..30",
        "max_length": 30,
        "is_fixed_length": False,
        "fnc1_required": True,
        "data_title": "INTERNAL",
        "gs1_key": False,
    },
}


# ---------------------------------------------------------------------------
# Human-readable AI descriptions
# ---------------------------------------------------------------------------

AI_DESCRIPTIONS: Dict[str, str] = {
    "00": "Serial Shipping Container Code (SSCC-18)",
    "01": "Global Trade Item Number (GTIN-14)",
    "02": "GTIN of contained trade items",
    "10": "Batch or lot number",
    "11": "Production date (YYMMDD)",
    "13": "Packaging date (YYMMDD)",
    "15": "Best before or best by date (YYMMDD)",
    "17": "Expiration or use-by date (YYMMDD)",
    "21": "Serial number",
    "30": "Variable count of items",
    "37": "Count of trade items contained in a logistics unit",
    "3100": "Net weight in kg (0 decimals)",
    "3101": "Net weight in kg (1 decimal)",
    "3102": "Net weight in kg (2 decimals)",
    "3103": "Net weight in kg (3 decimals)",
    "240": "Additional product identification",
    "410": "Ship-to location Global Location Number (GLN)",
    "411": "Bill-to / invoice-to Global Location Number (GLN)",
    "414": "Physical location Global Location Number (GLN)",
    "422": "Country of origin of a trade item (ISO 3166 numeric)",
    "423": "Country of initial processing",
    "424": "Country of processing",
    "426": "Country covering full process chain",
    "7023": "Certification reference number",
    "8200": "Extended packaging URL (GS1 Digital Link)",
    "90": "Information mutually agreed between trading partners",
    "91": "Company internal information (not to be used in open systems)",
}


# ---------------------------------------------------------------------------
# EUDR-relevant Application Identifiers
# ---------------------------------------------------------------------------

EUDR_RELEVANT_AIS: Dict[str, str] = {
    "01": "GTIN - Product identification for traceability",
    "10": "Batch/Lot - Batch tracking per EUDR Article 4(2)",
    "21": "Serial - Individual item serialization",
    "3103": "Net weight (kg) - Quantity tracking",
    "422": "Country of origin - Origin verification per Article 9",
    "7023": "Certification reference - DDS and compliance certificates",
    "11": "Production date - Production date tracking",
    "240": "Additional ID - Operator or DDS reference number",
    "91": "Internal - EUDR compliance status encoding",
}


# ---------------------------------------------------------------------------
# GS1 URI syntax regex
# ---------------------------------------------------------------------------

GS1_URI_SYNTAX_REGEX: str = (
    r"^https?://(?:[a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,}"
    r"/(?:01|gtin)/(\d{14})"
    r"(?:/(?:10|lot)/([^\s/]+))?"
    r"(?:/(?:21|cpv|ser)/([^\s/]+))?"
    r"(?:\?.*)?$"
)

_URI_PATTERN = re.compile(GS1_URI_SYNTAX_REGEX)


# ---------------------------------------------------------------------------
# GS1 Digital Link path segment mappings
# ---------------------------------------------------------------------------

_AI_TO_PATH_SEGMENT: Dict[str, str] = {
    "00": "00",
    "01": "01",
    "10": "10",
    "21": "21",
    "22": "cpv",
    "414": "414",
}

_AI_TO_QUERY_PARAM: Dict[str, str] = {
    "11": "11",
    "13": "13",
    "15": "15",
    "17": "17",
    "30": "30",
    "37": "37",
    "3100": "3100",
    "3101": "3101",
    "3102": "3102",
    "3103": "3103",
    "240": "240",
    "422": "422",
    "423": "423",
    "424": "424",
    "426": "426",
    "7023": "7023",
    "8200": "8200",
    "90": "90",
    "91": "91",
}


# ---------------------------------------------------------------------------
# GTIN validation and check digit calculation
# ---------------------------------------------------------------------------


def calculate_gtin_check_digit(gtin_without_check: str) -> str:
    """Calculate the GTIN check digit using the GS1 modulo 10 algorithm.

    Supports GTIN-8 (7 digits + check), GTIN-12 (11 + check),
    GTIN-13 (12 + check), and GTIN-14 (13 + check).

    The algorithm:
        1. Starting from the rightmost digit, multiply alternating
           digits by 3 and 1.
        2. Sum all products.
        3. Check digit = (10 - (sum mod 10)) mod 10.

    Args:
        gtin_without_check: GTIN string without the check digit.
            Must be 7, 11, 12, or 13 digits.

    Returns:
        Single check digit character ('0'-'9').

    Raises:
        ValueError: If the input length is invalid or contains
            non-digit characters.

    Example:
        >>> calculate_gtin_check_digit("0614141000050")
        '7'
    """
    cleaned = gtin_without_check.strip()
    if not cleaned.isdigit():
        raise ValueError(
            f"GTIN must contain only digits, got '{cleaned}'"
        )
    if len(cleaned) not in (7, 11, 12, 13):
        raise ValueError(
            f"GTIN without check digit must be 7, 11, 12, or 13 "
            f"digits, got {len(cleaned)} digits"
        )

    digits = [int(d) for d in cleaned]
    # Multipliers alternate 3, 1 starting from rightmost position
    total = 0
    for i, digit in enumerate(reversed(digits)):
        multiplier = 3 if i % 2 == 0 else 1
        total += digit * multiplier

    check = (10 - (total % 10)) % 10
    return str(check)


def validate_gtin(gtin: str) -> bool:
    """Validate a GTIN string including its check digit.

    Supports GTIN-8, GTIN-12, GTIN-13, and GTIN-14.

    Args:
        gtin: Complete GTIN string including check digit.

    Returns:
        True if the GTIN is valid (correct length and check digit).

    Example:
        >>> validate_gtin("06141410000507")
        True
        >>> validate_gtin("06141410000508")
        False
    """
    cleaned = re.sub(r"[\s\-]", "", gtin.strip())
    if not cleaned.isdigit():
        return False
    if len(cleaned) not in (8, 12, 13, 14):
        return False

    payload = cleaned[:-1]
    expected_check = calculate_gtin_check_digit(payload)
    return cleaned[-1] == expected_check


def normalize_to_gtin14(gtin: str) -> str:
    """Normalize a GTIN-8, GTIN-12, or GTIN-13 to GTIN-14 by zero-padding.

    Args:
        gtin: GTIN string (8, 12, 13, or 14 digits).

    Returns:
        14-digit GTIN string.

    Raises:
        ValueError: If the input is not a valid GTIN.

    Example:
        >>> normalize_to_gtin14("4006381333931")
        '04006381333931'
    """
    cleaned = re.sub(r"[\s\-]", "", gtin.strip())
    if not cleaned.isdigit():
        raise ValueError(f"GTIN must contain only digits, got '{gtin}'")
    if len(cleaned) not in (8, 12, 13, 14):
        raise ValueError(
            f"GTIN must be 8, 12, 13, or 14 digits, "
            f"got {len(cleaned)} digits"
        )
    return cleaned.zfill(14)


# ---------------------------------------------------------------------------
# GS1 Digital Link URI construction
# ---------------------------------------------------------------------------


def build_gs1_digital_link_uri(
    gtin: str,
    ais: Optional[Dict[str, str]] = None,
    base_url: Optional[str] = None,
) -> str:
    """Build a GS1 Digital Link URI from a GTIN and optional AIs.

    Constructs a web-resolvable URI per GS1 Digital Link Standard 1.3.
    The GTIN forms the primary path element; batch/lot (AI 10) and
    serial (AI 21) are added as path segments; all other AIs become
    query parameters.

    Args:
        gtin: GTIN string (8-14 digits). Will be normalized to
            GTIN-14 internally.
        ais: Optional dictionary of Application Identifier code ->
            value pairs to include in the URI.
        base_url: Optional base URL override. Defaults to
            ``GS1_DIGITAL_LINK_BASE_URL``.

    Returns:
        Complete GS1 Digital Link URI string.

    Raises:
        ValueError: If the GTIN is invalid or an AI value exceeds
            its maximum length.

    Example:
        >>> build_gs1_digital_link_uri(
        ...     "06141410000507",
        ...     {"10": "BATCH001", "422": "076"},
        ... )
        'https://id.gs1.org/01/06141410000507/10/BATCH001?422=076'
    """
    resolved_base = base_url or GS1_DIGITAL_LINK_BASE_URL
    resolved_base = resolved_base.rstrip("/")

    gtin14 = normalize_to_gtin14(gtin)
    if not validate_gtin(gtin14):
        raise ValueError(f"Invalid GTIN check digit: {gtin14}")

    # Build path: /01/{gtin14}
    path_parts = [f"/01/{gtin14}"]

    # Optional path segments for batch/lot and serial
    query_params: List[str] = []
    extra_ais = dict(ais) if ais else {}

    # AI 10 (batch/lot) as path segment
    if "10" in extra_ais:
        batch_val = extra_ais.pop("10")
        _validate_ai_value("10", batch_val)
        path_parts.append(f"/10/{_uri_encode(batch_val)}")

    # AI 21 (serial) as path segment
    if "21" in extra_ais:
        serial_val = extra_ais.pop("21")
        _validate_ai_value("21", serial_val)
        path_parts.append(f"/21/{_uri_encode(serial_val)}")

    # Remaining AIs as query parameters
    for ai_code in sorted(extra_ais.keys()):
        ai_value = extra_ais[ai_code]
        _validate_ai_value(ai_code, ai_value)
        query_params.append(
            f"{ai_code}={_uri_encode(ai_value)}"
        )

    uri = resolved_base + "".join(path_parts)
    if query_params:
        uri += "?" + "&".join(query_params)

    return uri


def parse_gs1_digital_link_uri(
    uri: str,
) -> Optional[Dict[str, str]]:
    """Parse a GS1 Digital Link URI and extract AI values.

    Args:
        uri: GS1 Digital Link URI string.

    Returns:
        Dictionary mapping AI codes to their values, or None if
        the URI does not match the expected format.

    Example:
        >>> result = parse_gs1_digital_link_uri(
        ...     "https://id.gs1.org/01/06141410000507/10/BATCH001"
        ... )
        >>> result["01"]
        '06141410000507'
        >>> result["10"]
        'BATCH001'
    """
    match = _URI_PATTERN.match(uri)
    if not match:
        return None

    result: Dict[str, str] = {}
    gtin = match.group(1)
    if gtin:
        result["01"] = gtin
    batch = match.group(2)
    if batch:
        result["10"] = batch
    serial = match.group(3)
    if serial:
        result["21"] = serial

    # Parse query parameters
    if "?" in uri:
        query_string = uri.split("?", 1)[1]
        for param in query_string.split("&"):
            if "=" in param:
                key, value = param.split("=", 1)
                if key in GS1_APPLICATION_IDENTIFIERS:
                    result[key] = value

    return result


def validate_gs1_digital_link_uri(uri: str) -> bool:
    """Validate a GS1 Digital Link URI format.

    Args:
        uri: URI string to validate.

    Returns:
        True if the URI matches the GS1 Digital Link format with
        a valid GTIN.

    Example:
        >>> validate_gs1_digital_link_uri(
        ...     "https://id.gs1.org/01/06141410000507"
        ... )
        True
    """
    parsed = parse_gs1_digital_link_uri(uri)
    if parsed is None or "01" not in parsed:
        return False
    return validate_gtin(parsed["01"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uri_encode(value: str) -> str:
    """Percent-encode a value for use in a URI path or query.

    Only encodes characters that are not safe in URI paths:
    spaces, forward slashes, and other reserved characters.
    """
    # Minimal encoding for GS1 Digital Link compatibility
    encoded = value.replace("%", "%25")
    encoded = encoded.replace(" ", "%20")
    encoded = encoded.replace("/", "%2F")
    encoded = encoded.replace("?", "%3F")
    encoded = encoded.replace("#", "%23")
    encoded = encoded.replace("&", "%26")
    encoded = encoded.replace("=", "%3D")
    return encoded


def _validate_ai_value(ai_code: str, value: str) -> None:
    """Validate an AI value against its specification.

    Args:
        ai_code: GS1 Application Identifier code.
        value: Value string to validate.

    Raises:
        ValueError: If the value exceeds the maximum length
            or is otherwise invalid for the AI.
    """
    spec = GS1_APPLICATION_IDENTIFIERS.get(ai_code)
    if spec is None:
        # Unknown AI codes are allowed (future compatibility)
        return

    max_len = spec.get("max_length") or spec.get("length")
    if max_len and len(value) > max_len:
        raise ValueError(
            f"AI ({ai_code}) value exceeds max length {max_len}: "
            f"got {len(value)} characters"
        )


def get_ai_description(ai_code: str) -> str:
    """Get a human-readable description for an Application Identifier.

    Args:
        ai_code: GS1 Application Identifier code.

    Returns:
        Description string, or "Unknown AI" if not found.

    Example:
        >>> get_ai_description("01")
        'Global Trade Item Number (GTIN-14)'
    """
    return AI_DESCRIPTIONS.get(ai_code, "Unknown AI")


def get_eudr_relevant_ais() -> Dict[str, str]:
    """Return the subset of AIs relevant to EUDR compliance.

    Returns:
        Dictionary mapping AI codes to their EUDR-specific
        descriptions.
    """
    return dict(EUDR_RELEVANT_AIS)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "GS1_DIGITAL_LINK_BASE_URL",
    "GS1_APPLICATION_IDENTIFIERS",
    "AI_DESCRIPTIONS",
    "EUDR_RELEVANT_AIS",
    "GS1_URI_SYNTAX_REGEX",
    "calculate_gtin_check_digit",
    "validate_gtin",
    "normalize_to_gtin14",
    "build_gs1_digital_link_uri",
    "parse_gs1_digital_link_uri",
    "validate_gs1_digital_link_uri",
    "get_ai_description",
    "get_eudr_relevant_ais",
]
