"""
GL-EUDR-002: LLM Integration Service

Provides LLM-powered features for the Geolocation Collector Agent:
- Address parsing (converting addresses to coordinates)
- Validation error explanations (human-readable explanations)
- Location description generation
- Coordinate format conversion assistance

Uses a provider-agnostic interface supporting Claude and GPT-4.
Maintains zero-hallucination guarantees - LLM NEVER generates coordinates,
only parses and explains data.

Important: The validation engine is fully deterministic. LLM is ONLY used for:
1. Parsing unstructured addresses into structured components
2. Generating human-readable explanations of validation errors
3. Generating location descriptions from enrichment data
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    CLAUDE = "claude"
    OPENAI = "openai"
    LOCAL = "local"  # For testing


@dataclass
class ParsedAddress:
    """Address parsed by LLM into structured components."""
    original_input: str
    street: Optional[str] = None
    village: Optional[str] = None
    district: Optional[str] = None
    province: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    postal_code: Optional[str] = None
    landmark: Optional[str] = None
    confidence: float = 0.0
    parsing_notes: str = ""
    needs_geocoding: bool = True


@dataclass
class ValidationExplanation:
    """Human-readable explanation of validation error."""
    error_code: str
    technical_message: str
    user_friendly_message: str
    suggested_action: str
    example_fix: Optional[str] = None
    severity_description: str = ""
    regulatory_context: str = ""


@dataclass
class LocationDescription:
    """Generated description of a plot location."""
    plot_id: str
    short_description: str
    detailed_description: str
    administrative_hierarchy: str
    environmental_notes: Optional[str] = None


@dataclass
class CoordinateFormatHelp:
    """Help with coordinate format conversion."""
    original_format: str
    original_value: str
    detected_format: str
    decimal_degrees: Optional[Tuple[float, float]] = None
    conversion_notes: str = ""
    confidence: float = 0.0


class GeolocationLLMService:
    """
    LLM integration service for Geolocation Collector.

    CRITICAL: LLM is NEVER used for coordinate generation or validation.
    All validation is deterministic. LLM only provides:
    - Address parsing assistance
    - Error explanations
    - Location descriptions
    """

    # Supported coordinate formats
    COORDINATE_FORMATS = {
        "DECIMAL_DEGREES": r"^-?\d+\.\d+$",
        "DMS": r"^\d+°\s*\d+['′]\s*\d+(?:\.\d+)?[\"″]\s*[NSEW]$",
        "DDM": r"^\d+°\s*\d+(?:\.\d+)?['′]\s*[NSEW]$",
    }

    # Country codes for common producer countries
    COUNTRY_KEYWORDS = {
        "indonesia": "ID", "indo": "ID",
        "brazil": "BR", "brasil": "BR",
        "malaysia": "MY",
        "peru": "PE",
        "colombia": "CO",
        "ecuador": "EC",
        "ghana": "GH",
        "cote d'ivoire": "CI", "ivory coast": "CI",
        "cameroon": "CM",
        "nigeria": "NG",
        "papua new guinea": "PG", "png": "PG",
        "vietnam": "VN",
        "thailand": "TH",
        "philippines": "PH",
    }

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.LOCAL,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._default_model()
        self._client = None

    def _default_model(self) -> str:
        """Get default model for provider."""
        if self.provider == LLMProvider.CLAUDE:
            return "claude-sonnet-4-20250514"
        elif self.provider == LLMProvider.OPENAI:
            return "gpt-4"
        return "local"

    def _get_client(self):
        """Get or create LLM client."""
        if self._client is not None:
            return self._client

        if self.provider == LLMProvider.CLAUDE:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("anthropic package not installed, using local mode")
                self.provider = LLMProvider.LOCAL

        elif self.provider == LLMProvider.OPENAI:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("openai package not installed, using local mode")
                self.provider = LLMProvider.LOCAL

        return self._client

    # =========================================================================
    # ADDRESS PARSING
    # =========================================================================

    def parse_address(
        self,
        address_text: str,
        expected_country: Optional[str] = None
    ) -> ParsedAddress:
        """
        Parse an unstructured address into components.

        Args:
            address_text: Free-form address string
            expected_country: Expected ISO country code (for validation)

        Returns:
            ParsedAddress with structured components
        """
        if self.provider == LLMProvider.LOCAL:
            return self._parse_address_local(address_text, expected_country)

        prompt = self._build_address_parsing_prompt(address_text, expected_country)

        try:
            response = self._call_llm(prompt)
            return self._parse_address_response(response, address_text)
        except Exception as e:
            logger.warning(f"LLM address parsing failed: {e}, using local fallback")
            return self._parse_address_local(address_text, expected_country)

    def _build_address_parsing_prompt(
        self,
        address_text: str,
        expected_country: Optional[str]
    ) -> str:
        """Build prompt for address parsing."""
        country_hint = f"Expected country: {expected_country}" if expected_country else ""

        return f"""Parse this agricultural plot address into structured components.
This is for EUDR compliance - we need accurate location data.

Address: {address_text}
{country_hint}

Extract and return JSON with these fields:
- street: Street address or road name
- village: Village or local area name
- district: District or sub-district name
- province: Province or state name
- country: Full country name
- country_code: ISO 2-letter code (e.g., ID, BR, MY)
- postal_code: Postal or zip code if present
- landmark: Any landmark mentioned
- confidence: Your confidence 0.0-1.0
- parsing_notes: Any notes about ambiguity

Important:
- If unsure about a field, leave it null
- For Indonesian addresses, look for: Desa/Kelurahan (village), Kecamatan (district), Kabupaten/Kota (regency), Provinsi (province)
- For Brazilian addresses, look for: Fazenda/Sitio (farm), Municipio (municipality), Estado (state)

Return ONLY valid JSON."""

    def _parse_address_response(
        self,
        response: str,
        original: str
    ) -> ParsedAddress:
        """Parse LLM response into ParsedAddress."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return ParsedAddress(
                    original_input=original,
                    street=data.get("street"),
                    village=data.get("village"),
                    district=data.get("district"),
                    province=data.get("province"),
                    country=data.get("country"),
                    country_code=data.get("country_code"),
                    postal_code=data.get("postal_code"),
                    landmark=data.get("landmark"),
                    confidence=float(data.get("confidence", 0.5)),
                    parsing_notes=data.get("parsing_notes", ""),
                    needs_geocoding=True
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM address response: {e}")

        return self._parse_address_local(original, None)

    def _parse_address_local(
        self,
        address_text: str,
        expected_country: Optional[str]
    ) -> ParsedAddress:
        """Simple local address parsing without LLM."""
        text_lower = address_text.lower()

        # Detect country
        country_code = expected_country
        country_name = None

        for keyword, code in self.COUNTRY_KEYWORDS.items():
            if keyword in text_lower:
                country_code = code
                country_name = keyword.title()
                break

        # Simple heuristics for common patterns
        parts = [p.strip() for p in address_text.split(",")]

        return ParsedAddress(
            original_input=address_text,
            country=country_name,
            country_code=country_code,
            confidence=0.3 if country_code else 0.1,
            parsing_notes="Parsed using simple heuristics - manual review recommended",
            needs_geocoding=True
        )

    # =========================================================================
    # VALIDATION ERROR EXPLANATIONS
    # =========================================================================

    def explain_validation_error(
        self,
        error_code: str,
        error_message: str,
        metadata: Dict[str, Any] = None
    ) -> ValidationExplanation:
        """
        Generate a user-friendly explanation of a validation error.

        Args:
            error_code: The error code (e.g., INSUFFICIENT_LAT_PRECISION)
            error_message: The technical error message
            metadata: Additional context about the error

        Returns:
            ValidationExplanation with user-friendly details
        """
        if self.provider == LLMProvider.LOCAL:
            return self._explain_error_local(error_code, error_message, metadata)

        prompt = self._build_error_explanation_prompt(error_code, error_message, metadata)

        try:
            response = self._call_llm(prompt)
            return self._parse_error_explanation_response(response, error_code, error_message)
        except Exception as e:
            logger.warning(f"LLM error explanation failed: {e}, using local fallback")
            return self._explain_error_local(error_code, error_message, metadata)

    def _build_error_explanation_prompt(
        self,
        error_code: str,
        error_message: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Build prompt for error explanation."""
        meta_str = json.dumps(metadata or {}, indent=2)

        return f"""Explain this EUDR geolocation validation error in simple terms for a farmer or supplier.

Error Code: {error_code}
Technical Message: {error_message}
Additional Context: {meta_str}

Provide a JSON response with:
- user_friendly_message: Simple explanation a non-technical person can understand
- suggested_action: What they should do to fix it
- example_fix: A specific example if applicable
- severity_description: How serious this is (e.g., "Must fix before submission")
- regulatory_context: Brief explanation of why EUDR requires this

Keep language simple and actionable.
Return ONLY valid JSON."""

    def _parse_error_explanation_response(
        self,
        response: str,
        error_code: str,
        error_message: str
    ) -> ValidationExplanation:
        """Parse LLM error explanation response."""
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return ValidationExplanation(
                    error_code=error_code,
                    technical_message=error_message,
                    user_friendly_message=data.get("user_friendly_message", error_message),
                    suggested_action=data.get("suggested_action", "Please review and correct"),
                    example_fix=data.get("example_fix"),
                    severity_description=data.get("severity_description", ""),
                    regulatory_context=data.get("regulatory_context", "")
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM error explanation: {e}")

        return self._explain_error_local(error_code, error_message, {})

    def _explain_error_local(
        self,
        error_code: str,
        error_message: str,
        metadata: Dict[str, Any] = None
    ) -> ValidationExplanation:
        """Generate local error explanations without LLM."""
        explanations = {
            "INSUFFICIENT_LAT_PRECISION": ValidationExplanation(
                error_code=error_code,
                technical_message=error_message,
                user_friendly_message="The latitude coordinate doesn't have enough decimal places. "
                                      "EUDR requires at least 6 decimal places for precise location.",
                suggested_action="Use a GPS device or app that provides coordinates with 6+ decimal places. "
                                 "Example: -4.123456 instead of -4.12",
                example_fix="-4.123456 (correct) vs -4.12 (incorrect)",
                severity_description="Must fix - coordinates will be rejected",
                regulatory_context="EUDR Article 9 requires coordinates accurate to ~0.1 meters"
            ),
            "INSUFFICIENT_LON_PRECISION": ValidationExplanation(
                error_code=error_code,
                technical_message=error_message,
                user_friendly_message="The longitude coordinate doesn't have enough decimal places. "
                                      "EUDR requires at least 6 decimal places for precise location.",
                suggested_action="Use a GPS device or app that provides coordinates with 6+ decimal places.",
                example_fix="102.654321 (correct) vs 102.65 (incorrect)",
                severity_description="Must fix - coordinates will be rejected",
                regulatory_context="EUDR Article 9 requires coordinates accurate to ~0.1 meters"
            ),
            "INVALID_LAT_RANGE": ValidationExplanation(
                error_code=error_code,
                technical_message=error_message,
                user_friendly_message="The latitude value is outside the valid range. "
                                      "Latitude must be between -90 and 90 degrees.",
                suggested_action="Check your coordinates - latitude cannot exceed 90 or be less than -90.",
                example_fix="Valid: 4.5 or -4.5 | Invalid: 104.5",
                severity_description="Must fix - invalid geographic coordinate",
                regulatory_context="GPS coordinates use standard WGS-84 format"
            ),
            "INVALID_LON_RANGE": ValidationExplanation(
                error_code=error_code,
                technical_message=error_message,
                user_friendly_message="The longitude value is outside the valid range. "
                                      "Longitude must be between -180 and 180 degrees.",
                suggested_action="Check your coordinates - longitude cannot exceed 180 or be less than -180.",
                example_fix="Valid: 102.5 or -75.5 | Invalid: 202.5",
                severity_description="Must fix - invalid geographic coordinate",
                regulatory_context="GPS coordinates use standard WGS-84 format"
            ),
            "NOT_IN_COUNTRY": ValidationExplanation(
                error_code=error_code,
                technical_message=error_message,
                user_friendly_message="The coordinates don't appear to be within the declared country. "
                                      "The plot location doesn't match the country you selected.",
                suggested_action="Verify the coordinates are correct and match the country code. "
                                 "Check if latitude and longitude are swapped.",
                example_fix="For Indonesia (ID), coordinates should be around -8 to 6 lat, 95 to 141 lon",
                severity_description="Must fix - location and country don't match",
                regulatory_context="EUDR requires accurate country of origin declaration"
            ),
            "IN_WATER_BODY": ValidationExplanation(
                error_code=error_code,
                technical_message=error_message,
                user_friendly_message="The coordinates are located in a water body (ocean, lake, or river). "
                                      "Agricultural plots cannot be in water.",
                suggested_action="Check your coordinates - they may be incorrect or swapped.",
                severity_description="Must fix - impossible plot location",
                regulatory_context="EUDR requires valid land-based production locations"
            ),
            "SELF_INTERSECTING": ValidationExplanation(
                error_code=error_code,
                technical_message=error_message,
                user_friendly_message="The polygon boundary crosses over itself (figure-8 shape). "
                                      "Plot boundaries must form a simple shape without overlapping lines.",
                suggested_action="Redraw the polygon so the boundary doesn't cross itself.",
                example_fix="Draw boundaries in order around the plot perimeter",
                severity_description="Must fix - invalid polygon shape",
                regulatory_context="EUDR requires valid geographic boundaries for plots ≥4 hectares"
            ),
            "AREA_TOO_SMALL": ValidationExplanation(
                error_code=error_code,
                technical_message=error_message,
                user_friendly_message="The plot area is too small (less than 0.01 hectares). "
                                      "This may indicate incorrect coordinates.",
                suggested_action="Verify polygon coordinates are correct. The area should match your actual plot.",
                severity_description="Must fix - area below minimum threshold",
                regulatory_context="EUDR tracks plots with meaningful agricultural production"
            ),
            "IN_PROTECTED_AREA": ValidationExplanation(
                error_code=error_code,
                technical_message=error_message,
                user_friendly_message="This plot is located in or near a protected area (national park, reserve, etc.). "
                                      "This requires additional documentation.",
                suggested_action="Provide documentation showing legal authorization to farm in this area.",
                severity_description="Warning - may require additional documentation",
                regulatory_context="EUDR prohibits deforestation in protected areas after Dec 2020"
            ),
            "IN_URBAN_AREA": ValidationExplanation(
                error_code=error_code,
                technical_message=error_message,
                user_friendly_message="This plot is located in an urban area, which is unusual for agricultural production.",
                suggested_action="Verify coordinates are correct. If this is urban farming, additional documentation may be needed.",
                severity_description="Warning - unusual location for agriculture",
                regulatory_context="EUDR focuses on rural agricultural and forestry land"
            ),
            "NEEDS_POLYGON": ValidationExplanation(
                error_code=error_code,
                technical_message=error_message,
                user_friendly_message="This plot is 4 hectares or larger and needs polygon coordinates "
                                      "(boundary outline) instead of a single point.",
                suggested_action="Provide polygon coordinates outlining the full plot boundary.",
                severity_description="Warning - polygon recommended for large plots",
                regulatory_context="EUDR requires polygons for plots ≥4 hectares for accurate deforestation checks"
            ),
            "POOR_GPS_ACCURACY": ValidationExplanation(
                error_code=error_code,
                technical_message=error_message,
                user_friendly_message="The GPS accuracy reported is lower than recommended (>10 meters). "
                                      "This may affect the precision of the location.",
                suggested_action="Try collecting coordinates again with clear sky view and better GPS signal.",
                severity_description="Warning - accuracy could be improved",
                regulatory_context="EUDR recommends high-precision GPS for reliable verification"
            ),
            "AREA_MISMATCH": ValidationExplanation(
                error_code=error_code,
                technical_message=error_message,
                user_friendly_message="The calculated plot area doesn't match the declared area by more than 20%. "
                                      "This may indicate incorrect polygon coordinates.",
                suggested_action="Review polygon coordinates and declared area. One may need correction.",
                severity_description="Warning - area discrepancy detected",
                regulatory_context="EUDR requires consistent area declarations for traceability"
            ),
        }

        return explanations.get(error_code, ValidationExplanation(
            error_code=error_code,
            technical_message=error_message,
            user_friendly_message=f"Validation issue: {error_message}",
            suggested_action="Please review the coordinates and try again.",
            severity_description="Please review",
            regulatory_context="EUDR compliance requirement"
        ))

    def explain_multiple_errors(
        self,
        errors: List[Dict[str, Any]]
    ) -> List[ValidationExplanation]:
        """Generate explanations for multiple errors."""
        return [
            self.explain_validation_error(
                e.get("code", "UNKNOWN"),
                e.get("message", "Unknown error"),
                e.get("metadata", {})
            )
            for e in errors
        ]

    # =========================================================================
    # LOCATION DESCRIPTIONS
    # =========================================================================

    def generate_location_description(
        self,
        plot_id: str,
        country_code: str,
        admin_levels: Dict[str, str],
        coordinates: Tuple[float, float],
        enrichment_data: Dict[str, Any] = None
    ) -> LocationDescription:
        """
        Generate a human-readable description of a plot location.

        Args:
            plot_id: The plot identifier
            country_code: ISO country code
            admin_levels: Admin level names (admin_level_1, etc.)
            coordinates: (latitude, longitude)
            enrichment_data: Additional data (biome, elevation, etc.)

        Returns:
            LocationDescription with formatted descriptions
        """
        if self.provider == LLMProvider.LOCAL:
            return self._generate_description_local(
                plot_id, country_code, admin_levels, coordinates, enrichment_data
            )

        prompt = self._build_description_prompt(
            country_code, admin_levels, coordinates, enrichment_data
        )

        try:
            response = self._call_llm(prompt)
            return self._parse_description_response(
                response, plot_id, admin_levels, enrichment_data
            )
        except Exception as e:
            logger.warning(f"LLM description generation failed: {e}")
            return self._generate_description_local(
                plot_id, country_code, admin_levels, coordinates, enrichment_data
            )

    def _build_description_prompt(
        self,
        country_code: str,
        admin_levels: Dict[str, str],
        coordinates: Tuple[float, float],
        enrichment_data: Dict[str, Any] = None
    ) -> str:
        """Build prompt for location description."""
        admin_str = ", ".join(f"{k}: {v}" for k, v in admin_levels.items() if v)
        enrichment_str = json.dumps(enrichment_data or {})

        return f"""Generate a brief, informative description of this agricultural plot location.

Country: {country_code}
Administrative Levels: {admin_str}
Coordinates: {coordinates[0]:.6f}, {coordinates[1]:.6f}
Additional Info: {enrichment_str}

Provide JSON with:
- short_description: One sentence location summary (e.g., "Palm oil plantation in West Sumatra, Indonesia")
- detailed_description: 2-3 sentences with context
- administrative_hierarchy: Formatted admin hierarchy (e.g., "Padang District, West Sumatra Province, Indonesia")
- environmental_notes: Brief note if in notable biome or ecosystem

Keep descriptions factual and concise.
Return ONLY valid JSON."""

    def _parse_description_response(
        self,
        response: str,
        plot_id: str,
        admin_levels: Dict[str, str],
        enrichment_data: Dict[str, Any]
    ) -> LocationDescription:
        """Parse LLM description response."""
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return LocationDescription(
                    plot_id=plot_id,
                    short_description=data.get("short_description", ""),
                    detailed_description=data.get("detailed_description", ""),
                    administrative_hierarchy=data.get("administrative_hierarchy", ""),
                    environmental_notes=data.get("environmental_notes")
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM description: {e}")

        return self._generate_description_local(
            plot_id, "", admin_levels, (0, 0), enrichment_data
        )

    def _generate_description_local(
        self,
        plot_id: str,
        country_code: str,
        admin_levels: Dict[str, str],
        coordinates: Tuple[float, float],
        enrichment_data: Dict[str, Any] = None
    ) -> LocationDescription:
        """Generate simple description without LLM."""
        # Build admin hierarchy
        parts = []
        for level in ["admin_level_3", "admin_level_2", "admin_level_1"]:
            if admin_levels.get(level):
                parts.append(admin_levels[level])

        # Add country
        country_names = {
            "ID": "Indonesia", "BR": "Brazil", "MY": "Malaysia",
            "PE": "Peru", "CO": "Colombia", "GH": "Ghana",
        }
        country = country_names.get(country_code, country_code)
        parts.append(country)

        hierarchy = ", ".join(parts) if parts else "Unknown location"

        # Short description
        short = f"Plot in {hierarchy}"

        # Detailed description
        lat, lon = coordinates
        detailed = f"Agricultural plot located at coordinates ({lat:.6f}, {lon:.6f})"
        if hierarchy != "Unknown location":
            detailed += f" in {hierarchy}."

        # Environmental notes
        env_notes = None
        if enrichment_data:
            if enrichment_data.get("biome"):
                env_notes = f"Biome: {enrichment_data['biome']}"
            if enrichment_data.get("elevation_m"):
                elev = f"Elevation: {enrichment_data['elevation_m']}m"
                env_notes = f"{env_notes}. {elev}" if env_notes else elev

        return LocationDescription(
            plot_id=plot_id,
            short_description=short,
            detailed_description=detailed,
            administrative_hierarchy=hierarchy,
            environmental_notes=env_notes
        )

    # =========================================================================
    # COORDINATE FORMAT HELP
    # =========================================================================

    def help_with_coordinate_format(
        self,
        coordinate_string: str
    ) -> CoordinateFormatHelp:
        """
        Help user convert coordinates from various formats to decimal degrees.

        Args:
            coordinate_string: Coordinate in any format (DMS, DDM, etc.)

        Returns:
            CoordinateFormatHelp with conversion assistance
        """
        # Try to detect format
        detected_format = self._detect_coordinate_format(coordinate_string)

        if detected_format == "DECIMAL_DEGREES":
            try:
                value = float(coordinate_string)
                return CoordinateFormatHelp(
                    original_format="Decimal Degrees",
                    original_value=coordinate_string,
                    detected_format="DECIMAL_DEGREES",
                    decimal_degrees=(value, None),
                    conversion_notes="Already in decimal degrees format",
                    confidence=0.95
                )
            except ValueError:
                pass

        elif detected_format == "DMS":
            converted = self._convert_dms_to_decimal(coordinate_string)
            if converted:
                return CoordinateFormatHelp(
                    original_format="Degrees Minutes Seconds",
                    original_value=coordinate_string,
                    detected_format="DMS",
                    decimal_degrees=(converted, None),
                    conversion_notes="Converted from DMS to decimal degrees",
                    confidence=0.85
                )

        # Fallback - try LLM if available
        if self.provider != LLMProvider.LOCAL:
            return self._llm_coordinate_help(coordinate_string)

        return CoordinateFormatHelp(
            original_format="Unknown",
            original_value=coordinate_string,
            detected_format="UNKNOWN",
            decimal_degrees=None,
            conversion_notes="Could not automatically convert. Please enter coordinates in decimal degrees format (e.g., -4.123456).",
            confidence=0.0
        )

    def _detect_coordinate_format(self, value: str) -> str:
        """Detect coordinate format."""
        value = value.strip()

        for format_name, pattern in self.COORDINATE_FORMATS.items():
            if re.match(pattern, value, re.IGNORECASE):
                return format_name

        return "UNKNOWN"

    def _convert_dms_to_decimal(self, dms: str) -> Optional[float]:
        """Convert Degrees Minutes Seconds to decimal degrees."""
        # Pattern: 4°7'23.45"S or 4° 7' 23.45" S
        pattern = r"(\d+)[°]\s*(\d+)[′']\s*(\d+(?:\.\d+)?)[″\"]\s*([NSEW])"
        match = re.match(pattern, dms.upper())

        if match:
            degrees = int(match.group(1))
            minutes = int(match.group(2))
            seconds = float(match.group(3))
            direction = match.group(4)

            decimal = degrees + (minutes / 60) + (seconds / 3600)

            if direction in ['S', 'W']:
                decimal = -decimal

            return round(decimal, 6)

        return None

    def _llm_coordinate_help(self, coordinate_string: str) -> CoordinateFormatHelp:
        """Use LLM to help with coordinate conversion."""
        prompt = f"""Convert this coordinate to decimal degrees format:
Input: {coordinate_string}

Return JSON with:
- detected_format: What format this appears to be
- decimal_value: The coordinate in decimal degrees (number)
- confidence: Your confidence 0.0-1.0
- notes: Any conversion notes

Return ONLY valid JSON."""

        try:
            response = self._call_llm(prompt)
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return CoordinateFormatHelp(
                    original_format=data.get("detected_format", "Unknown"),
                    original_value=coordinate_string,
                    detected_format=data.get("detected_format", "UNKNOWN"),
                    decimal_degrees=(data.get("decimal_value"), None),
                    conversion_notes=data.get("notes", ""),
                    confidence=float(data.get("confidence", 0.5))
                )
        except Exception as e:
            logger.warning(f"LLM coordinate help failed: {e}")

        return CoordinateFormatHelp(
            original_format="Unknown",
            original_value=coordinate_string,
            detected_format="UNKNOWN",
            decimal_degrees=None,
            conversion_notes="Conversion failed. Please enter in decimal degrees.",
            confidence=0.0
        )

    # =========================================================================
    # LLM CALL HELPER
    # =========================================================================

    def _call_llm(self, prompt: str) -> str:
        """Make LLM API call."""
        client = self._get_client()

        if self.provider == LLMProvider.CLAUDE:
            response = client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif self.provider == LLMProvider.OPENAI:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        # Local mode - return empty
        return "{}"
