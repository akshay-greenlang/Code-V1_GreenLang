# -*- coding: utf-8 -*-
"""
Hallucination Detection for Climate Data

CRITICAL: Prevents LLM from fabricating numbers without tool backing.

Climate calculations require extreme accuracy - a hallucinated number could
lead to incorrect carbon credits, wrong emissions reporting, or flawed policy
decisions. This module ensures every numeric claim is backed by tool output.

Features:
- Extract numeric claims from LLM responses (handles scientific notation, units)
- Verify each claim has a valid tool citation
- Fuzzy match claimed numbers with tool responses (±1% tolerance for rounding)
- Unit normalization (kg vs g, kWh vs MWh)
- Raise exception if numbers don't match tool results

Usage:
    detector = HallucinationDetector(tolerance=0.01)

    # Verify response after LLM call
    detector.verify_response(
        response_text="Grid intensity is 450 gCO2/kWh [tool:get_grid_intensity]",
        tool_calls=[{"name": "get_grid_intensity", ...}],
        tool_responses=[{"result": {"intensity": 450.3, "unit": "gCO2/kWh"}}]
    )

    # Raises HallucinationDetected if numbers fabricated
"""

from __future__ import annotations
import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from greenlang.utilities.determinism import DeterministicClock


class HallucinationDetected(Exception):
    """
    Raised when LLM fabricates numbers without tool backing

    Attributes:
        message: Description of what was hallucinated
        claim: The numeric claim that couldn't be verified
        tool_response: What the tool actually returned (if any)
        expected_citation: Which tool should have been cited
    """

    def __init__(
        self,
        message: str,
        claim: Optional["NumericClaim"] = None,
        tool_response: Optional[Dict[str, Any]] = None,
        expected_citation: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.claim = claim
        self.tool_response = tool_response
        self.expected_citation = expected_citation

    def __str__(self) -> str:
        parts = [self.message]
        if self.claim:
            parts.append(f"Claimed: {self.claim.value} {self.claim.unit}")
        if self.tool_response:
            parts.append(f"Tool returned: {self.tool_response}")
        if self.expected_citation:
            parts.append(f"Expected citation: {self.expected_citation}")
        return " | ".join(parts)


class NumericClaim(BaseModel):
    """
    A numeric claim extracted from LLM response

    Examples:
        "450 gCO2/kWh" -> NumericClaim(value=450, unit="gCO2/kWh")
        "1.5e3 kg" -> NumericClaim(value=1500, unit="kg")
        "The value is 42.7 MWh [tool:energy_calc]" ->
            NumericClaim(value=42.7, unit="MWh", citation_tool="energy_calc")
    """

    value: float = Field(description="Numeric value")
    unit: str = Field(description="Unit of measurement (empty string if unitless)")
    citation_tool: Optional[str] = Field(
        default=None, description="Tool cited for this claim (if any)"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence in extraction (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    context: str = Field(
        default="",
        description="Surrounding text for debugging"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "value": 450.0,
                    "unit": "gCO2/kWh",
                    "citation_tool": "get_grid_intensity",
                    "confidence": 1.0,
                    "context": "Grid intensity is 450 gCO2/kWh [tool:get_grid_intensity]"
                },
                {
                    "value": 1500.0,
                    "unit": "kg",
                    "citation_tool": None,
                    "confidence": 0.9,
                    "context": "approximately 1.5e3 kg of CO2"
                }
            ]
        }


class Citation(BaseModel):
    """
    A verified citation linking a claim to tool output

    Created when a numeric claim is successfully matched to tool response.
    Used for audit trails and provenance tracking.
    """

    tool: str = Field(description="Tool name that provided the value")
    value: float = Field(description="Value from tool output")
    unit: str = Field(description="Unit from tool output")
    source: str = Field(description="JSON path or key where value was found")
    timestamp: str = Field(
        default_factory=lambda: DeterministicClock.utcnow().isoformat(),
        description="When verification occurred (UTC)"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "tool": "get_grid_intensity",
                    "value": 450.3,
                    "unit": "gCO2/kWh",
                    "source": "result.intensity",
                    "timestamp": "2025-10-01T12:00:00Z"
                }
            ]
        }


class HallucinationDetector:
    """
    Detects when LLM fabricates numbers without tool backing

    Prevents climate data hallucinations by:
    1. Extracting all numeric claims from response text
    2. Verifying each claim has a tool citation
    3. Checking claimed numbers exist in tool responses (with tolerance)
    4. Raising HallucinationDetected if verification fails

    Args:
        tolerance: Fuzzy match tolerance for floating-point comparison (default 0.01 = ±1%)
        require_citations: If True, all numeric claims must have tool citations (default True)
        unit_normalizations: Custom unit conversion mappings

    Example:
        detector = HallucinationDetector(tolerance=0.01)

        # This will pass (450 matches 450.3 within 1% tolerance)
        detector.verify_response(
            response_text="Grid is 450 gCO2/kWh [tool:grid]",
            tool_calls=[{"name": "grid", "arguments": {}}],
            tool_responses=[{"result": {"intensity": 450.3, "unit": "gCO2/kWh"}}]
        )

        # This will raise HallucinationDetected (no tool output for 999)
        detector.verify_response(
            response_text="Grid is 999 gCO2/kWh",
            tool_calls=[],
            tool_responses=[]
        )
    """

    # Regex patterns for extracting numbers and citations
    # Matches: 123, 1,234.5, 1.5e3, 42.7, -3.14, etc.
    # Handles comma-separated thousands
    # Uses negative lookahead to avoid matching numbers in units (e.g., CO2, H2O)
    NUMBER_PATTERN = re.compile(
        r'(?<![a-zA-Z])(-?(?:\d{1,3}(?:,\d{3})*\.?\d*|\d+\.\d+|\d+)(?:[eE][+-]?\d+)?)(?![a-zA-Z])'
    )

    # Matches: [tool:name], (tool:name), {tool:name}, <tool:name>
    CITATION_PATTERN = re.compile(
        r'[\[\(\{<]tool:(\w+)[\]\)\}>]'
    )

    # Common unit patterns
    UNIT_PATTERN = re.compile(
        r'([a-zA-Z]+(?:/[a-zA-Z]+)?)'
    )

    # Unit normalization (convert everything to base units for comparison)
    DEFAULT_UNIT_NORMALIZATIONS = {
        # Mass
        'g': ('kg', 0.001),
        'kg': ('kg', 1.0),
        't': ('kg', 1000.0),
        'ton': ('kg', 1000.0),
        'tonne': ('kg', 1000.0),
        'mt': ('kg', 1000.0),
        'lb': ('kg', 0.453592),

        # Energy
        'wh': ('kwh', 0.001),
        'kwh': ('kwh', 1.0),
        'mwh': ('kwh', 1000.0),
        'gwh': ('kwh', 1000000.0),
        'j': ('kwh', 2.77778e-7),
        'kj': ('kwh', 2.77778e-4),
        'mj': ('kwh', 0.277778),
        'gj': ('kwh', 277.778),

        # Power
        'w': ('kw', 0.001),
        'kw': ('kw', 1.0),
        'mw': ('kw', 1000.0),
        'gw': ('kw', 1000000.0),

        # CO2 intensity (grid)
        'gco2/kwh': ('gco2/kwh', 1.0),
        'kgco2/kwh': ('gco2/kwh', 1000.0),
        'gco2/mwh': ('gco2/kwh', 0.001),
        'kgco2/mwh': ('gco2/kwh', 1.0),

        # Emissions
        'gco2': ('kgco2', 0.001),
        'gco2e': ('kgco2e', 0.001),
        'kgco2': ('kgco2', 1.0),
        'kgco2e': ('kgco2e', 1.0),
        'tco2': ('kgco2', 1000.0),
        'tco2e': ('kgco2e', 1000.0),
        'mtco2e': ('kgco2e', 1000.0),
    }

    def __init__(
        self,
        tolerance: float = 0.01,
        require_citations: bool = True,
        unit_normalizations: Optional[Dict[str, tuple]] = None
    ):
        """
        Initialize hallucination detector

        Args:
            tolerance: Fuzzy match tolerance (0.01 = ±1%, 0.05 = ±5%)
            require_citations: If True, all numeric claims must cite tools
            unit_normalizations: Custom unit conversions (merges with defaults)
        """
        self.tolerance = tolerance
        self.require_citations = require_citations

        # Merge custom normalizations with defaults
        self.unit_normalizations = self.DEFAULT_UNIT_NORMALIZATIONS.copy()
        if unit_normalizations:
            self.unit_normalizations.update(unit_normalizations)

    def extract_numeric_claims(self, text: str) -> List[NumericClaim]:
        """
        Extract all numeric claims from text

        Finds patterns like:
        - "450 gCO2/kWh" -> NumericClaim(value=450, unit="gCO2/kWh")
        - "1.5e3 kg [tool:calc]" -> NumericClaim(value=1500, unit="kg", citation_tool="calc")
        - "The result is 42.7 MWh" -> NumericClaim(value=42.7, unit="MWh")

        Args:
            text: LLM response text to analyze

        Returns:
            List of extracted numeric claims
        """
        claims = []

        # Find all numbers in text
        for match in self.NUMBER_PATTERN.finditer(text):
            value_str = match.group(1)
            try:
                # Remove commas before converting to float
                value = float(value_str.replace(',', ''))
            except ValueError:
                continue

            # Get context around the number (±50 chars)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]

            # Look for unit immediately after number
            unit = ""
            rest_of_text = text[match.end():match.end()+30]
            # Match unit that starts with space or directly after number
            # Handles: kg, kWh, gCO2/kWh, kgCO2e, etc.
            unit_match = re.match(r'\s*([a-zA-Z]+[0-9]*[a-zA-Z]*(?:/[a-zA-Z]+[0-9]*[a-zA-Z]*)?)', rest_of_text)
            if unit_match:
                unit = unit_match.group(1)

            # Look for citation - prefer same line after number, then before
            citation_tool = None

            # First, look for citation on same line after number (most common pattern)
            line_end = text.find('\n', match.end())
            if line_end == -1:
                line_end = len(text)
            after_context = text[match.end():min(line_end, match.end() + 100)]
            citation_match = self.CITATION_PATTERN.search(after_context)

            # If not found, look in broader context (±100 chars)
            if not citation_match:
                citation_start = max(0, match.start() - 100)
                citation_end = min(len(text), match.end() + 100)
                citation_context = text[citation_start:citation_end]
                citation_match = self.CITATION_PATTERN.search(citation_context)

            if citation_match:
                citation_tool = citation_match.group(1)

            # Determine confidence (lower if ambiguous)
            confidence = 1.0
            if not unit:
                confidence *= 0.9  # Lower confidence if no unit
            if not citation_tool and self.require_citations:
                confidence *= 0.8  # Lower confidence if no citation

            claims.append(NumericClaim(
                value=value,
                unit=unit,
                citation_tool=citation_tool,
                confidence=confidence,
                context=context.strip()
            ))

        return claims

    def normalize_value(self, value: float, unit: str) -> tuple[float, str]:
        """
        Normalize value to base unit for comparison

        Examples:
            normalize_value(1500, "g") -> (1.5, "kg")
            normalize_value(2.5, "MWh") -> (2500, "kWh")
            normalize_value(450, "gCO2/kWh") -> (450, "gCO2/kWh")

        Args:
            value: Numeric value
            unit: Unit string (case-insensitive)

        Returns:
            Tuple of (normalized_value, base_unit)
        """
        unit_lower = unit.lower().replace(' ', '')

        if unit_lower in self.unit_normalizations:
            base_unit, multiplier = self.unit_normalizations[unit_lower]
            return (value * multiplier, base_unit)

        # No normalization available - return as-is
        return (value, unit)

    def fuzzy_match(self, claimed: float, actual: float, tolerance: float = None) -> bool:
        """
        Check if claimed value matches actual within tolerance

        Uses relative tolerance for flexibility with floating-point numbers.

        Examples:
            fuzzy_match(450, 450.3, 0.01) -> True  (within 1%)
            fuzzy_match(100, 105, 0.01) -> False  (5% difference)
            fuzzy_match(0, 0, 0.01) -> True

        Args:
            claimed: Value claimed by LLM
            actual: Value from tool response
            tolerance: Relative tolerance (default: use self.tolerance)

        Returns:
            True if values match within tolerance
        """
        if tolerance is None:
            tolerance = self.tolerance

        # Handle exact matches (including both zero)
        if claimed == actual:
            return True

        # Handle zero cases
        if actual == 0:
            return abs(claimed) <= tolerance

        # Relative difference
        rel_diff = abs(claimed - actual) / abs(actual)
        return rel_diff <= tolerance

    def find_value_in_response(
        self,
        response: Dict[str, Any],
        target_value: float,
        target_unit: str,
        tolerance: float = None
    ) -> Optional[str]:
        """
        Recursively search for value in tool response

        Searches nested dictionaries and lists for numeric values that match
        the target (within tolerance). Returns JSON path if found.

        Args:
            response: Tool response dictionary
            target_value: Value to find
            target_unit: Expected unit
            tolerance: Match tolerance (default: use self.tolerance)

        Returns:
            JSON path where value was found (e.g., "result.intensity") or None
        """
        if tolerance is None:
            tolerance = self.tolerance

        def search(obj: Any, path: str = "") -> Optional[str]:
            # Check if this is a numeric value
            if isinstance(obj, (int, float)):
                if self.fuzzy_match(target_value, float(obj), tolerance):
                    return path

            # Check if this is a dict with value and unit
            elif isinstance(obj, dict):
                # Direct value/unit pattern
                if 'value' in obj and 'unit' in obj:
                    try:
                        val = float(obj['value'])
                        unit = str(obj['unit'])
                        # Normalize both for comparison
                        norm_val, norm_unit = self.normalize_value(val, unit)
                        norm_target, norm_target_unit = self.normalize_value(target_value, target_unit)

                        if norm_target_unit == norm_unit and self.fuzzy_match(norm_target, norm_val, tolerance):
                            return f"{path}.value" if path else "value"
                    except (ValueError, TypeError):
                        pass

                # Recursively search dict values
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    result = search(value, new_path)
                    if result:
                        return result

            # Recursively search lists
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]"
                    result = search(item, new_path)
                    if result:
                        return result

            return None

        return search(response)

    def verify_citation(
        self,
        claim: NumericClaim,
        tool_response: Dict[str, Any]
    ) -> Optional[Citation]:
        """
        Verify claim matches tool response (with tolerance)

        Searches tool response for a value matching the claim. Handles:
        - Direct numeric values
        - Nested objects with value/unit fields
        - Unit normalization
        - Fuzzy matching within tolerance

        Args:
            claim: Numeric claim to verify
            tool_response: Tool response to search

        Returns:
            Citation if verified, None if not found
        """
        # Normalize claim value and unit
        norm_value, norm_unit = self.normalize_value(claim.value, claim.unit)

        # Search for value in response
        source_path = self.find_value_in_response(
            tool_response,
            norm_value,
            norm_unit,
            self.tolerance
        )

        if source_path:
            return Citation(
                tool=claim.citation_tool or "unknown",
                value=claim.value,
                unit=claim.unit,
                source=source_path
            )

        return None

    def verify_response(
        self,
        response_text: str,
        tool_calls: List[Dict[str, Any]],
        tool_responses: List[Dict[str, Any]]
    ) -> List[Citation]:
        """
        Verify entire response (raises HallucinationDetected if invalid)

        Comprehensive verification:
        1. Extract all numeric claims from response text
        2. For each claim, check if it has a tool citation
        3. Verify citation matches actual tool response
        4. Raise exception if any claim can't be verified

        Args:
            response_text: LLM response text
            tool_calls: List of tool calls made (dicts with 'name', 'arguments')
            tool_responses: List of tool responses (dicts with results)

        Returns:
            List of verified citations

        Raises:
            HallucinationDetected: If any numeric claim can't be verified

        Example:
            detector.verify_response(
                response_text="Grid is 450 gCO2/kWh [tool:grid], emissions are 1,021 kg [tool:calc]",
                tool_calls=[
                    {"name": "grid", "arguments": {"region": "CA"}},
                    {"name": "calc", "arguments": {"kwh": 2000}}
                ],
                tool_responses=[
                    {"result": {"intensity": 450.3, "unit": "gCO2/kWh"}},
                    {"result": {"emissions": 1021.5, "unit": "kg"}}
                ]
            )
            # Returns list of 2 verified citations
        """
        # Extract all numeric claims
        claims = self.extract_numeric_claims(response_text)

        if not claims:
            # No numeric claims - nothing to verify
            return []

        # Build tool response lookup by name
        tool_response_map = {}
        for i, call in enumerate(tool_calls):
            tool_name = call.get('name', f'tool_{i}')
            if i < len(tool_responses):
                tool_response_map[tool_name] = tool_responses[i]

        # Verify each claim
        verified_citations = []

        for claim in claims:
            # Check if claim has citation
            if not claim.citation_tool:
                if self.require_citations:
                    raise HallucinationDetected(
                        message=f"Numeric claim without tool citation",
                        claim=claim,
                        tool_response=None,
                        expected_citation="any_tool"
                    )
                else:
                    # Citations not required - skip verification
                    continue

            # Check if cited tool exists
            if claim.citation_tool not in tool_response_map:
                raise HallucinationDetected(
                    message=f"Citation to non-existent tool",
                    claim=claim,
                    tool_response=None,
                    expected_citation=claim.citation_tool
                )

            # Verify claim against tool response
            tool_response = tool_response_map[claim.citation_tool]
            citation = self.verify_citation(claim, tool_response)

            if not citation:
                raise HallucinationDetected(
                    message=f"Claim does not match tool output",
                    claim=claim,
                    tool_response=tool_response,
                    expected_citation=claim.citation_tool
                )

            verified_citations.append(citation)

        return verified_citations
