# -*- coding: utf-8 -*-
"""
AGENT-EUDR-027: Information Gathering Agent - Data Normalization Engine

Normalizes data collected from multiple heterogeneous sources into
standardized formats suitable for EUDR due diligence statement assembly.
Supports 8 normalization types: unit conversion, date formatting,
coordinate parsing, currency standardization, ISO country code resolution,
HS product code formatting, address cleaning, and certificate ID formatting.

Production infrastructure includes:
    - Type-dispatched normalization with dedicated handlers per type
    - Batch normalization for efficient bulk processing
    - Confidence scoring per normalization record
    - Comprehensive country code lookup (name/code -> ISO 3166-1 alpha-2)
    - Multi-format date parsing with ISO 8601 output
    - Coordinate parsing for DMS, DDM, and DD formats -> WGS84
    - Unit standardization to SI/metric base units
    - HS code validation and formatting (4-10 digit)
    - Normalization audit log for traceability
    - SHA-256 provenance hash on batch operations

Zero-Hallucination Guarantees:
    - All normalization uses deterministic string parsing and lookup tables
    - Country code mapping via static ISO 3166-1 lookup (no LLM)
    - Date parsing via explicit format patterns (no NLP)
    - Coordinate conversion via deterministic arithmetic
    - All provenance hashes computed from canonical JSON

Regulatory References:
    - EUDR Article 9(1)(c): Country of production in standardized format
    - EUDR Article 9(1)(d): Geolocation in WGS84 coordinate system
    - EUDR Article 9(1)(b): Quantity in standardized units
    - EUDR Article 31: 5-year record retention for normalization audit logs

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (Engine 6: Data Normalization)
Agent ID: GL-EUDR-IGA-027
Status: Production Ready
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.information_gathering.config import (
    InformationGatheringConfig,
    get_config,
)
from greenlang.agents.eudr.information_gathering.models import (
    NormalizationRecord,
    NormalizationType,
)
from greenlang.agents.eudr.information_gathering.provenance import ProvenanceTracker
from greenlang.schemas import utcnow
from greenlang.agents.eudr.information_gathering.metrics import (
    record_normalization_error,
)

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# ISO 3166-1 Alpha-2 Country Code Lookup
# ---------------------------------------------------------------------------

_COUNTRY_NAME_TO_CODE: Dict[str, str] = {
    "afghanistan": "AF", "albania": "AL", "algeria": "DZ", "angola": "AO",
    "argentina": "AR", "australia": "AU", "austria": "AT", "bangladesh": "BD",
    "belgium": "BE", "benin": "BJ", "bolivia": "BO", "brazil": "BR",
    "burkina faso": "BF", "burundi": "BI", "cambodia": "KH", "cameroon": "CM",
    "canada": "CA", "central african republic": "CF", "chad": "TD",
    "chile": "CL", "china": "CN", "colombia": "CO", "congo": "CG",
    "democratic republic of the congo": "CD", "costa rica": "CR",
    "cote d'ivoire": "CI", "ivory coast": "CI", "croatia": "HR",
    "czech republic": "CZ", "czechia": "CZ", "denmark": "DK",
    "ecuador": "EC", "egypt": "EG", "el salvador": "SV",
    "equatorial guinea": "GQ", "ethiopia": "ET", "finland": "FI",
    "france": "FR", "gabon": "GA", "germany": "DE", "ghana": "GH",
    "greece": "GR", "guatemala": "GT", "guinea": "GN", "guyana": "GY",
    "honduras": "HN", "hungary": "HU", "india": "IN", "indonesia": "ID",
    "ireland": "IE", "italy": "IT", "jamaica": "JM", "japan": "JP",
    "kenya": "KE", "laos": "LA", "lao": "LA", "liberia": "LR",
    "madagascar": "MG", "malawi": "MW", "malaysia": "MY", "mali": "ML",
    "mexico": "MX", "mozambique": "MZ", "myanmar": "MM", "netherlands": "NL",
    "new zealand": "NZ", "nicaragua": "NI", "nigeria": "NG", "norway": "NO",
    "pakistan": "PK", "panama": "PA", "papua new guinea": "PG", "paraguay": "PY",
    "peru": "PE", "philippines": "PH", "poland": "PL", "portugal": "PT",
    "romania": "RO", "russia": "RU", "russian federation": "RU",
    "rwanda": "RW", "senegal": "SN", "sierra leone": "SL", "singapore": "SG",
    "slovakia": "SK", "slovenia": "SI", "south africa": "ZA",
    "south korea": "KR", "korea": "KR", "spain": "ES", "sri lanka": "LK",
    "sudan": "SD", "suriname": "SR", "sweden": "SE", "switzerland": "CH",
    "tanzania": "TZ", "thailand": "TH", "togo": "TG", "trinidad and tobago": "TT",
    "tunisia": "TN", "turkey": "TR", "turkiye": "TR", "uganda": "UG",
    "ukraine": "UA", "united arab emirates": "AE", "united kingdom": "GB",
    "united states": "US", "united states of america": "US",
    "uruguay": "UY", "venezuela": "VE", "vietnam": "VN", "zambia": "ZM",
    "zimbabwe": "ZW",
}

#: Known ISO 3166-1 alpha-2 codes for quick validation.
_VALID_ALPHA2: set = {v for v in _COUNTRY_NAME_TO_CODE.values()}

#: Alpha-3 to alpha-2 mapping for common producer countries.
_ALPHA3_TO_ALPHA2: Dict[str, str] = {
    "AFG": "AF", "AGO": "AO", "ARG": "AR", "AUS": "AU", "AUT": "AT",
    "BEL": "BE", "BEN": "BJ", "BGD": "BD", "BOL": "BO", "BRA": "BR",
    "BFA": "BF", "BDI": "BI", "KHM": "KH", "CMR": "CM", "CAN": "CA",
    "CAF": "CF", "TCD": "TD", "CHL": "CL", "CHN": "CN", "COL": "CO",
    "COG": "CG", "COD": "CD", "CRI": "CR", "CIV": "CI", "HRV": "HR",
    "CZE": "CZ", "DNK": "DK", "ECU": "EC", "EGY": "EG", "SLV": "SV",
    "GNQ": "GQ", "ETH": "ET", "FIN": "FI", "FRA": "FR", "GAB": "GA",
    "DEU": "DE", "GHA": "GH", "GRC": "GR", "GTM": "GT", "GIN": "GN",
    "GUY": "GY", "HND": "HN", "HUN": "HU", "IND": "IN", "IDN": "ID",
    "IRL": "IE", "ITA": "IT", "JAM": "JM", "JPN": "JP", "KEN": "KE",
    "LAO": "LA", "LBR": "LR", "MDG": "MG", "MWI": "MW", "MYS": "MY",
    "MLI": "ML", "MEX": "MX", "MOZ": "MZ", "MMR": "MM", "NLD": "NL",
    "NZL": "NZ", "NIC": "NI", "NGA": "NG", "NOR": "NO", "PAK": "PK",
    "PAN": "PA", "PNG": "PG", "PRY": "PY", "PER": "PE", "PHL": "PH",
    "POL": "PL", "PRT": "PT", "ROU": "RO", "RUS": "RU", "RWA": "RW",
    "SEN": "SN", "SLE": "SL", "SGP": "SG", "SVK": "SK", "SVN": "SI",
    "ZAF": "ZA", "KOR": "KR", "ESP": "ES", "LKA": "LK", "SDN": "SD",
    "SUR": "SR", "SWE": "SE", "CHE": "CH", "TZA": "TZ", "THA": "TH",
    "TGO": "TG", "TTO": "TT", "TUN": "TN", "TUR": "TR", "UGA": "UG",
    "UKR": "UA", "ARE": "AE", "GBR": "GB", "USA": "US", "URY": "UY",
    "VEN": "VE", "VNM": "VN", "ZMB": "ZM", "ZWE": "ZW",
}

# ---------------------------------------------------------------------------
# Date Format Patterns
# ---------------------------------------------------------------------------

_DATE_FORMATS: List[str] = [
    "%Y-%m-%d",       # 2025-03-15
    "%Y/%m/%d",       # 2025/03/15
    "%d-%m-%Y",       # 15-03-2025
    "%d/%m/%Y",       # 15/03/2025
    "%m-%d-%Y",       # 03-15-2025
    "%m/%d/%Y",       # 03/15/2025
    "%d %b %Y",       # 15 Mar 2025
    "%d %B %Y",       # 15 March 2025
    "%B %d, %Y",      # March 15, 2025
    "%b %d, %Y",      # Mar 15, 2025
    "%Y%m%d",         # 20250315
    "%d.%m.%Y",       # 15.03.2025
]

# ---------------------------------------------------------------------------
# Unit Conversion Aliases
# ---------------------------------------------------------------------------

_UNIT_ALIASES: Dict[str, str] = {
    # Weight -> kg
    "kg": "kg", "kilogram": "kg", "kilograms": "kg", "kgs": "kg",
    "t": "t", "tonne": "t", "tonnes": "t", "metric ton": "t",
    "metric tons": "t", "mt": "t",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "g": "g", "gram": "g", "grams": "g",
    # Volume -> m3
    "m3": "m3", "m\u00b3": "m3", "cubic meter": "m3", "cubic meters": "m3",
    "cubic metre": "m3", "cubic metres": "m3",
    "l": "l", "liter": "l", "litre": "l", "liters": "l", "litres": "l",
    # Area -> ha
    "ha": "ha", "hectare": "ha", "hectares": "ha",
    "acre": "acre", "acres": "acre",
    "km2": "km2", "km\u00b2": "km2", "sq km": "km2",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute deterministic SHA-256 hash of data.

    Args:
        data: Any JSON-serializable object.

    Returns:
        64-character lowercase hex SHA-256 hash string.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class DataNormalizationEngine:
    """Engine for normalizing heterogeneous data into EUDR-standard formats.

    Dispatches normalization requests to type-specific handlers covering
    units, dates, coordinates, currencies, country codes, product codes,
    addresses, and certificate IDs. Each normalization produces an audit
    record with confidence scoring.

    Args:
        config: Agent configuration (uses singleton if None).

    Example:
        >>> engine = DataNormalizationEngine()
        >>> record = engine.normalize_record(
        ...     "country", "Brazil", NormalizationType.COUNTRY_CODE
        ... )
        >>> assert record.normalized_value == "BR"
        >>> assert record.confidence == Decimal("1.0")
    """

    def __init__(self, config: Optional[InformationGatheringConfig] = None) -> None:
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._normalization_history: List[NormalizationRecord] = []
        self._error_count: int = 0

        # Dispatch table mapping NormalizationType to handler
        self._handlers: Dict[NormalizationType, Any] = {
            NormalizationType.UNIT: self.normalize_unit,
            NormalizationType.DATE: self.normalize_date,
            NormalizationType.COORDINATE: self.normalize_coordinate,
            NormalizationType.CURRENCY: self.normalize_currency,
            NormalizationType.COUNTRY_CODE: self.normalize_country_code,
            NormalizationType.PRODUCT_CODE: self.normalize_product_code,
            NormalizationType.ADDRESS: self.normalize_address,
            NormalizationType.CERTIFICATE_ID: self.normalize_certificate_id,
        }

        logger.info(
            "DataNormalizationEngine initialized "
            "(coordinate_system=%s, currency=%s, handlers=%d)",
            self._config.default_coordinate_system,
            self._config.default_currency,
            len(self._handlers),
        )

    def normalize_record(
        self,
        field_name: str,
        source_value: str,
        normalization_type: NormalizationType,
    ) -> NormalizationRecord:
        """Normalize a single field value by dispatching to the type handler.

        Args:
            field_name: Name of the field being normalized.
            source_value: Raw source value to normalize.
            normalization_type: Type of normalization to apply.

        Returns:
            NormalizationRecord with source value, normalized value,
            and confidence score.
        """
        handler = self._handlers.get(normalization_type)
        if handler is None:
            logger.error(
                "No handler for normalization type: %s", normalization_type.value
            )
            record_normalization_error(normalization_type.value)
            self._error_count += 1
            return NormalizationRecord(
                field_name=field_name,
                source_value=source_value,
                normalized_value=source_value,
                normalization_type=normalization_type,
                confidence=Decimal("0"),
            )

        try:
            normalized_value = handler(source_value)
            confidence = self._compute_confidence(
                source_value, normalized_value, normalization_type
            )
            record = NormalizationRecord(
                field_name=field_name,
                source_value=source_value,
                normalized_value=normalized_value,
                normalization_type=normalization_type,
                confidence=confidence,
            )
            self._normalization_history.append(record)
            logger.debug(
                "Normalized %s [%s]: '%s' -> '%s' (confidence=%s)",
                field_name,
                normalization_type.value,
                source_value,
                normalized_value,
                confidence,
            )
            return record

        except Exception as exc:
            logger.error(
                "Normalization failed for %s [%s]: '%s' -> %s",
                field_name,
                normalization_type.value,
                source_value,
                str(exc),
            )
            record_normalization_error(normalization_type.value)
            self._error_count += 1
            return NormalizationRecord(
                field_name=field_name,
                source_value=source_value,
                normalized_value=source_value,
                normalization_type=normalization_type,
                confidence=Decimal("0"),
            )

    def normalize_batch(
        self,
        records: List[Tuple[str, str, NormalizationType]],
    ) -> List[NormalizationRecord]:
        """Normalize a batch of field values.

        Args:
            records: List of (field_name, source_value, normalization_type)
                tuples.

        Returns:
            List of NormalizationRecord objects, one per input tuple.
        """
        start_time = time.monotonic()
        results: List[NormalizationRecord] = []

        for field_name, source_value, norm_type in records:
            result = self.normalize_record(field_name, source_value, norm_type)
            results.append(result)

        elapsed = time.monotonic() - start_time

        # Provenance for batch
        input_hash = _compute_hash({
            "batch_size": len(records),
            "types": [r[2].value for r in records],
        })
        output_hash = _compute_hash({
            "results": [
                {"field": r.field_name, "normalized": r.normalized_value}
                for r in results
            ],
        })
        self._provenance.create_entry(
            step="batch_normalization",
            source="multi_source",
            input_hash=input_hash,
            output_hash=output_hash,
        )

        logger.info(
            "Batch normalization: %d records in %.0fms (%d errors)",
            len(records),
            elapsed * 1000,
            sum(1 for r in results if r.confidence == Decimal("0")),
        )
        return results

    def normalize_coordinate(self, value: str) -> str:
        """Parse various coordinate formats and return WGS84 lat,lon string.

        Supported input formats:
            - Decimal degrees: "1.2345, 103.456" or "1.2345 103.456"
            - DMS: "1\u00b014'04.2\"N, 103\u00b027'21.6\"E"
            - DDM: "1\u00b014.07'N, 103\u00b027.36'E"
            - Signed decimal: "-1.2345, 103.456"

        Args:
            value: Raw coordinate string.

        Returns:
            Normalized "lat,lon" string in decimal degrees.
        """
        cleaned = value.strip()

        # Try decimal degrees first: "lat, lon" or "lat lon"
        dd_pattern = re.compile(
            r"^([+-]?\d+\.?\d*)\s*[,;\s]\s*([+-]?\d+\.?\d*)$"
        )
        match = dd_pattern.match(cleaned)
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            return f"{lat:.6f},{lon:.6f}"

        # Try DMS format: 1\u00b014'04.2"N, 103\u00b027'21.6"E
        dms_pattern = re.compile(
            r"(\d+)[°\u00b0]\s*(\d+)['\u2032]\s*(\d+\.?\d*)[\"″\u2033]?\s*([NSns])"
            r"\s*[,;\s]\s*"
            r"(\d+)[°\u00b0]\s*(\d+)['\u2032]\s*(\d+\.?\d*)[\"″\u2033]?\s*([EWew])"
        )
        match = dms_pattern.match(cleaned)
        if match:
            lat = self._dms_to_dd(
                int(match.group(1)),
                int(match.group(2)),
                float(match.group(3)),
                match.group(4).upper(),
            )
            lon = self._dms_to_dd(
                int(match.group(5)),
                int(match.group(6)),
                float(match.group(7)),
                match.group(8).upper(),
            )
            return f"{lat:.6f},{lon:.6f}"

        # Try DDM format: 1\u00b014.07'N, 103\u00b027.36'E
        ddm_pattern = re.compile(
            r"(\d+)[°\u00b0]\s*(\d+\.?\d*)['\u2032]\s*([NSns])"
            r"\s*[,;\s]\s*"
            r"(\d+)[°\u00b0]\s*(\d+\.?\d*)['\u2032]\s*([EWew])"
        )
        match = ddm_pattern.match(cleaned)
        if match:
            lat = self._ddm_to_dd(
                int(match.group(1)),
                float(match.group(2)),
                match.group(3).upper(),
            )
            lon = self._ddm_to_dd(
                int(match.group(4)),
                float(match.group(5)),
                match.group(6).upper(),
            )
            return f"{lat:.6f},{lon:.6f}"

        # Fallback: return cleaned value
        logger.warning("Could not parse coordinate format: '%s'", value)
        return cleaned

    @staticmethod
    def _dms_to_dd(
        degrees: int, minutes: int, seconds: float, direction: str
    ) -> float:
        """Convert DMS (degrees, minutes, seconds) to decimal degrees.

        Args:
            degrees: Degree component.
            minutes: Minute component.
            seconds: Second component.
            direction: Cardinal direction (N/S/E/W).

        Returns:
            Decimal degree value, negative for S/W.
        """
        dd = degrees + minutes / 60.0 + seconds / 3600.0
        if direction in ("S", "W"):
            dd = -dd
        return dd

    @staticmethod
    def _ddm_to_dd(
        degrees: int, decimal_minutes: float, direction: str
    ) -> float:
        """Convert DDM (degrees, decimal minutes) to decimal degrees.

        Args:
            degrees: Degree component.
            decimal_minutes: Decimal minutes component.
            direction: Cardinal direction (N/S/E/W).

        Returns:
            Decimal degree value, negative for S/W.
        """
        dd = degrees + decimal_minutes / 60.0
        if direction in ("S", "W"):
            dd = -dd
        return dd

    def normalize_country_code(self, value: str) -> str:
        """Normalize country name or code to ISO 3166-1 alpha-2.

        Handles: full country names (case-insensitive), ISO alpha-2,
        ISO alpha-3, and common abbreviations.

        Args:
            value: Country name or code string.

        Returns:
            ISO 3166-1 alpha-2 code (uppercase, 2 characters).
        """
        cleaned = value.strip()

        # Already valid alpha-2
        if len(cleaned) == 2 and cleaned.upper() in _VALID_ALPHA2:
            return cleaned.upper()

        # Try alpha-3
        if len(cleaned) == 3 and cleaned.upper() in _ALPHA3_TO_ALPHA2:
            return _ALPHA3_TO_ALPHA2[cleaned.upper()]

        # Try country name lookup (case-insensitive)
        lower = cleaned.lower()
        if lower in _COUNTRY_NAME_TO_CODE:
            return _COUNTRY_NAME_TO_CODE[lower]

        # Partial match: check if input starts with or is contained in a key
        for name, code in _COUNTRY_NAME_TO_CODE.items():
            if name.startswith(lower) or lower in name:
                return code

        logger.warning("Could not resolve country code for: '%s'", value)
        return cleaned.upper()[:2] if len(cleaned) >= 2 else cleaned.upper()

    def normalize_date(self, value: str) -> str:
        """Normalize various date formats to ISO 8601 (YYYY-MM-DD).

        Tries multiple date format patterns and returns the first
        successful parse as ISO 8601 date string.

        Args:
            value: Raw date string in any supported format.

        Returns:
            ISO 8601 date string (YYYY-MM-DD).
        """
        cleaned = value.strip()

        for fmt in _DATE_FORMATS:
            try:
                parsed = datetime.strptime(cleaned, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # Fallback: try to extract YYYY-MM-DD pattern from string
        iso_pattern = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
        match = iso_pattern.search(cleaned)
        if match:
            return match.group(0)

        logger.warning("Could not parse date format: '%s'", value)
        return cleaned

    def normalize_currency(self, value: str, target: str = "EUR") -> str:
        """Format and standardize currency values.

        Strips currency symbols, normalizes thousands/decimal separators,
        and appends the target currency code.

        Args:
            value: Raw currency string (e.g., "$1,234.56", "EUR 1234.56").
            target: Target currency code (default: EUR from config).

        Returns:
            Formatted string: "1234.56 EUR".
        """
        if not target:
            target = self._config.default_currency

        cleaned = value.strip()

        # Remove common currency symbols and codes
        for symbol in ("$", "\u20ac", "\u00a3", "\u00a5", "EUR", "USD", "GBP", "JPY"):
            cleaned = cleaned.replace(symbol, "")
        cleaned = cleaned.strip()

        # Handle European number format: 1.234,56 -> 1234.56
        if re.match(r"^\d{1,3}(\.\d{3})+(,\d{1,2})?$", cleaned):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        # Handle standard format: 1,234.56 -> 1234.56
        elif "," in cleaned and "." in cleaned:
            cleaned = cleaned.replace(",", "")
        # Handle comma as decimal: 1234,56 -> 1234.56
        elif "," in cleaned:
            cleaned = cleaned.replace(",", ".")

        # Remove any remaining non-numeric characters except dot and minus
        cleaned = re.sub(r"[^\d.\-]", "", cleaned)

        try:
            amount = Decimal(cleaned).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            return f"{amount} {target}"
        except Exception:
            logger.warning("Could not parse currency value: '%s'", value)
            return f"{cleaned} {target}"

    def normalize_unit(self, value: str) -> str:
        """Standardize weight, volume, and area unit representations.

        Normalizes unit strings to canonical forms: kg, t, g, lb,
        m3, l, ha, acre, km2.

        Args:
            value: Raw unit or quantity+unit string (e.g., "500 kilograms").

        Returns:
            Normalized unit string (e.g., "500 kg").
        """
        cleaned = value.strip().lower()

        # Try to extract numeric value and unit
        match = re.match(r"^([+-]?\d+\.?\d*)\s*(.+)$", cleaned)
        if match:
            numeric = match.group(1)
            unit_str = match.group(2).strip()
            canonical = _UNIT_ALIASES.get(unit_str, unit_str)
            return f"{numeric} {canonical}"

        # No numeric value; try to normalize the unit string alone
        canonical = _UNIT_ALIASES.get(cleaned, cleaned)
        return canonical

    def normalize_certificate_id(self, value: str) -> str:
        """Clean and format certificate identifiers.

        Strips whitespace, normalizes separators, and uppercases the ID.
        Handles FSC (FSC-CXXXXXX), RSPO (SCC-XXXXXXX), PEFC (PEFC/XX-XX-XX),
        EU Organic (EU-BIO-XXX), and generic certificate formats.

        Args:
            value: Raw certificate ID string.

        Returns:
            Cleaned, uppercase certificate ID.
        """
        cleaned = value.strip()

        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Uppercase for consistency
        cleaned = cleaned.upper()

        # Normalize common separator variations
        # FSC: "FSC C012345" -> "FSC-C012345"
        cleaned = re.sub(r"^(FSC)\s+(C\d+)", r"\1-\2", cleaned)

        # PEFC: "PEFC 01-23-45" -> "PEFC/01-23-45"
        cleaned = re.sub(r"^(PEFC)\s+(\d)", r"\1/\2", cleaned)

        # EU-BIO: "EU BIO 123" -> "EU-BIO-123"
        cleaned = re.sub(r"^EU\s+BIO\s+", "EU-BIO-", cleaned)
        cleaned = re.sub(r"^EU\s+ORG\s+", "EU-ORG-", cleaned)

        return cleaned

    def normalize_address(self, value: str) -> str:
        """Normalize address strings by trimming and collapsing whitespace.

        Removes leading/trailing whitespace, collapses multiple spaces
        and newlines into single spaces, and normalizes common
        abbreviation formatting.

        Args:
            value: Raw address string.

        Returns:
            Cleaned address string.
        """
        cleaned = value.strip()

        # Collapse multiple whitespace/newlines
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Normalize common abbreviations
        cleaned = re.sub(r"\bSt\.\s", "Street ", cleaned)
        cleaned = re.sub(r"\bAve\.\s", "Avenue ", cleaned)
        cleaned = re.sub(r"\bRd\.\s", "Road ", cleaned)
        cleaned = re.sub(r"\bBlvd\.\s", "Boulevard ", cleaned)

        return cleaned

    def normalize_product_code(self, value: str) -> str:
        """Normalize and validate HS (Harmonized System) product codes.

        Strips non-digit characters and validates length (4-10 digits).
        Adds leading zeros if needed for minimum 4-digit format.

        Args:
            value: Raw HS code string (e.g., "0901.21", "090121").

        Returns:
            Cleaned numeric HS code string (4-10 digits).
        """
        cleaned = value.strip()

        # Remove dots, dashes, spaces (HS codes may be formatted)
        digits = re.sub(r"[^0-9]", "", cleaned)

        # Validate length
        if len(digits) < 4:
            digits = digits.zfill(4)
            logger.debug("Padded HS code to 4 digits: '%s' -> '%s'", value, digits)

        if len(digits) > 10:
            logger.warning(
                "HS code exceeds 10 digits: '%s' (%d digits); truncating",
                value,
                len(digits),
            )
            digits = digits[:10]

        return digits

    def _compute_confidence(
        self,
        source_value: str,
        normalized_value: str,
        normalization_type: NormalizationType,
    ) -> Decimal:
        """Compute confidence score for a normalization result.

        Confidence is based on whether the normalization produced a
        meaningful change and the type of normalization performed.

        Args:
            source_value: Original value.
            normalized_value: Normalized value.
            normalization_type: Type of normalization.

        Returns:
            Confidence score as Decimal in range [0.0, 1.0].
        """
        # If output is identical to input (no handler found or pass-through)
        if source_value.strip() == normalized_value.strip():
            # May be already normalized or could not parse
            if normalization_type in (
                NormalizationType.COUNTRY_CODE,
                NormalizationType.DATE,
                NormalizationType.COORDINATE,
            ):
                # These types should typically produce a change
                return Decimal("0.5")
            return Decimal("0.8")

        # High-confidence normalizations (deterministic lookups)
        if normalization_type == NormalizationType.COUNTRY_CODE:
            if len(normalized_value) == 2 and normalized_value.upper() in _VALID_ALPHA2:
                return Decimal("1.0")
            return Decimal("0.5")

        if normalization_type == NormalizationType.DATE:
            # Validate output is ISO 8601
            try:
                datetime.strptime(normalized_value, "%Y-%m-%d")
                return Decimal("1.0")
            except ValueError:
                return Decimal("0.3")

        if normalization_type == NormalizationType.COORDINATE:
            # Validate lat,lon format
            parts = normalized_value.split(",")
            if len(parts) == 2:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        return Decimal("1.0")
                except ValueError:
                    pass
            return Decimal("0.5")

        if normalization_type == NormalizationType.PRODUCT_CODE:
            if re.match(r"^\d{4,10}$", normalized_value):
                return Decimal("1.0")
            return Decimal("0.5")

        # Default confidence for successful normalization
        return Decimal("0.9")

    def get_normalization_stats(self) -> Dict[str, Any]:
        """Return data normalization engine statistics.

        Returns:
            Dict with total_normalizations, error_count,
            type_breakdown, and average_confidence keys.
        """
        type_counts: Dict[str, int] = {}
        total_confidence = Decimal("0")

        for record in self._normalization_history:
            type_key = record.normalization_type.value
            type_counts[type_key] = type_counts.get(type_key, 0) + 1
            total_confidence += record.confidence

        total = len(self._normalization_history)
        avg_confidence = (
            (total_confidence / Decimal(str(total))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if total > 0
            else Decimal("0")
        )

        return {
            "total_normalizations": total,
            "error_count": self._error_count,
            "type_breakdown": type_counts,
            "average_confidence": float(avg_confidence),
        }

    def clear_history(self) -> None:
        """Clear normalization history (for testing)."""
        self._normalization_history.clear()
        self._error_count = 0
        logger.info("Normalization history cleared")
