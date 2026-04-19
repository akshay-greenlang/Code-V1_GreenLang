# -*- coding: utf-8 -*-
"""
Entity Extractors for Knowledge Graph
======================================

Extract entities from various data sources including:
- Equipment tags (P-101, E-201, B-301, etc.)
- Process parameters (temperature, pressure, flow)
- Safety interlocks (LLWC, HHP, flame failure)
- Standards references (ASME, API, NFPA)

This module follows GreenLang's zero-hallucination principle by using
deterministic pattern matching and lookup tables for entity extraction.

Example:
    >>> extractor = EntityExtractor()
    >>> entities = extractor.extract_all("Boiler B-101 with LLWC per NFPA 85")
    >>> for entity in entities:
    ...     print(f"{entity.type}: {entity.value}")
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    EQUIPMENT = "equipment"
    EQUIPMENT_TAG = "equipment_tag"
    PROCESS_PARAMETER = "process_parameter"
    MEASUREMENT = "measurement"
    SAFETY_INTERLOCK = "safety_interlock"
    HAZARD = "hazard"
    STANDARD = "standard"
    STANDARD_SECTION = "standard_section"
    MATERIAL = "material"
    CHEMICAL = "chemical"
    LOCATION = "location"
    MANUFACTURER = "manufacturer"
    MODEL_NUMBER = "model_number"
    UNIT = "unit"
    NUMERIC_VALUE = "numeric_value"


class EquipmentPrefix(str, Enum):
    """Standard equipment tag prefixes per ISA 5.1."""
    # Rotating equipment
    P = "pump"
    C = "compressor"
    B = "blower"
    AG = "agitator"

    # Vessels and tanks
    V = "vessel"
    T = "tank"
    D = "drum"
    TK = "storage_tank"

    # Heat transfer
    E = "heat_exchanger"
    H = "heater"
    F = "furnace"
    AC = "air_cooler"
    CW = "cooling_water"

    # Steam/boiler
    BL = "boiler"
    ST = "steam_trap"
    DA = "deaerator"
    FW = "feedwater"

    # Separation
    COL = "column"
    SEP = "separator"
    FLT = "filter"

    # Control/instrumentation
    FCV = "flow_control_valve"
    PCV = "pressure_control_valve"
    TCV = "temperature_control_valve"
    LCV = "level_control_valve"
    PSV = "pressure_safety_valve"
    PRV = "pressure_reducing_valve"

    # General
    M = "mixer"
    R = "reactor"
    DR = "dryer"
    K = "kiln"


# =============================================================================
# Data Models
# =============================================================================

class TagPattern(BaseModel):
    """Pattern definition for equipment tag extraction."""

    pattern: str = Field(..., description="Regex pattern for matching")
    equipment_type: str = Field(..., description="Equipment type this pattern matches")
    prefix: str = Field(..., description="Tag prefix (e.g., 'P', 'E', 'B')")
    description: str = Field(default="", description="Pattern description")
    examples: List[str] = Field(default_factory=list, description="Example tags")

    @property
    def compiled_pattern(self) -> Pattern:
        """Return compiled regex pattern."""
        return re.compile(self.pattern, re.IGNORECASE)


class ExtractedEntity(BaseModel):
    """Extracted entity with metadata."""

    id: str = Field(..., description="Unique entity ID")
    type: EntityType = Field(..., description="Entity type")
    value: str = Field(..., description="Extracted value")
    normalized_value: str = Field(..., description="Normalized/canonical value")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score"
    )
    source_text: str = Field(default="", description="Original source text")
    start_position: int = Field(default=-1, description="Start position in source")
    end_position: int = Field(default=-1, description="End position in source")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def provenance_hash(self) -> str:
        """Generate provenance hash for this entity."""
        data = f"{self.type}:{self.value}:{self.normalized_value}"
        return hashlib.sha256(data.encode()).hexdigest()

    class Config:
        """Pydantic config."""
        use_enum_values = True


class ExtractionResult(BaseModel):
    """Result of entity extraction operation."""

    source_text: str = Field(..., description="Source text that was processed")
    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="Extracted entities"
    )
    total_count: int = Field(default=0, description="Total entities extracted")
    by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count by entity type"
    )
    extraction_time_ms: float = Field(default=0.0, description="Extraction time")
    warnings: List[str] = Field(default_factory=list, description="Extraction warnings")


# =============================================================================
# Base Extractor Interface
# =============================================================================

class BaseExtractor(ABC):
    """Abstract base class for entity extractors."""

    @abstractmethod
    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities from text.

        Args:
            text: Source text to extract from

        Returns:
            List of extracted entities
        """
        pass

    @abstractmethod
    def get_supported_types(self) -> List[EntityType]:
        """Get entity types this extractor supports."""
        pass


# =============================================================================
# Equipment Tag Extractor
# =============================================================================

class EquipmentTagExtractor(BaseExtractor):
    """
    Extract equipment tags from text.

    Recognizes standard ISA-style equipment tags:
    - P-101, P-101A/B (pumps)
    - E-201, E-201A (heat exchangers)
    - B-301, BL-301 (boilers)
    - V-401, D-402 (vessels/drums)
    - F-501, H-502 (furnaces/heaters)

    Also handles variations like:
    - P101 (no dash)
    - 101-P (reversed)
    - PUMP-101 (spelled out)

    Example:
        >>> extractor = EquipmentTagExtractor()
        >>> entities = extractor.extract("Pump P-101A feeds to HX E-201")
        >>> print([e.value for e in entities])
        ['P-101A', 'E-201']
    """

    # Standard tag patterns
    TAG_PATTERNS: List[TagPattern] = [
        # Standard ISA format: PREFIX-NUMBER or PREFIX-NUMBER-SUFFIX
        TagPattern(
            pattern=r"\b(P|C|B|AG)-(\d{3,4})([A-Z]?(?:/[A-Z])?)\b",
            equipment_type="rotating_equipment",
            prefix="P/C/B/AG",
            description="Pumps, compressors, blowers, agitators",
            examples=["P-101", "P-101A", "P-101A/B", "C-201"],
        ),
        TagPattern(
            pattern=r"\b(V|T|D|TK)-(\d{3,4})([A-Z]?)\b",
            equipment_type="vessel",
            prefix="V/T/D/TK",
            description="Vessels, tanks, drums",
            examples=["V-101", "T-201", "D-301", "TK-401"],
        ),
        TagPattern(
            pattern=r"\b(E|H|AC|CW)-(\d{3,4})([A-Z]?(?:/[A-Z])?)\b",
            equipment_type="heat_transfer",
            prefix="E/H/AC/CW",
            description="Heat exchangers, heaters, air coolers",
            examples=["E-101", "E-101A/B", "H-201", "AC-301"],
        ),
        TagPattern(
            pattern=r"\b(BL|B)-(\d{3,4})([A-Z]?)\b",
            equipment_type="boiler",
            prefix="BL/B",
            description="Boilers",
            examples=["BL-101", "B-101", "BL-201A"],
        ),
        TagPattern(
            pattern=r"\b(F|FH|PF)-(\d{3,4})([A-Z]?)\b",
            equipment_type="furnace",
            prefix="F/FH/PF",
            description="Furnaces, fired heaters, process furnaces",
            examples=["F-101", "FH-201", "PF-301"],
        ),
        TagPattern(
            pattern=r"\b(ST|DA|FW)-(\d{3,4})([A-Z]?)\b",
            equipment_type="steam_system",
            prefix="ST/DA/FW",
            description="Steam traps, deaerators, feedwater equipment",
            examples=["ST-101", "DA-201", "FW-301"],
        ),
        TagPattern(
            pattern=r"\b(COL|SEP|FLT)-(\d{3,4})([A-Z]?)\b",
            equipment_type="separation",
            prefix="COL/SEP/FLT",
            description="Columns, separators, filters",
            examples=["COL-101", "SEP-201", "FLT-301"],
        ),
        TagPattern(
            pattern=r"\b(FCV|PCV|TCV|LCV|PSV|PRV)-(\d{3,4})([A-Z]?)\b",
            equipment_type="valve",
            prefix="FCV/PCV/TCV/LCV/PSV/PRV",
            description="Control and safety valves",
            examples=["FCV-101", "PSV-201", "PRV-301"],
        ),
        TagPattern(
            pattern=r"\b(M|R|DR|K)-(\d{3,4})([A-Z]?)\b",
            equipment_type="general",
            prefix="M/R/DR/K",
            description="Mixers, reactors, dryers, kilns",
            examples=["M-101", "R-201", "DR-301", "K-401"],
        ),
        # Instrument tags
        TagPattern(
            pattern=r"\b([PFLT]I[CRAT]?)-(\d{3,5})([A-Z]?)\b",
            equipment_type="instrument",
            prefix="PI/FI/LI/TI",
            description="Pressure, flow, level, temperature instruments",
            examples=["PI-101", "FIC-201", "LIT-301", "TICA-401"],
        ),
    ]

    # Alternative patterns (spelled out, reversed, etc.)
    ALTERNATIVE_PATTERNS = [
        # Spelled out equipment type
        (r"\b(PUMP|COMPRESSOR|BLOWER|AGITATOR)-(\d{3,4})([A-Z]?)\b", "rotating_equipment"),
        (r"\b(VESSEL|TANK|DRUM)-(\d{3,4})([A-Z]?)\b", "vessel"),
        (r"\b(EXCHANGER|HEATER|COOLER)-(\d{3,4})([A-Z]?)\b", "heat_transfer"),
        (r"\b(BOILER|STEAM\s*GENERATOR)-(\d{3,4})([A-Z]?)\b", "boiler"),
        (r"\b(FURNACE|KILN|OVEN|DRYER)-(\d{3,4})([A-Z]?)\b", "furnace"),
    ]

    def __init__(self):
        """Initialize equipment tag extractor."""
        self._compiled_patterns: List[Tuple[Pattern, str, str]] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile all regex patterns."""
        for tp in self.TAG_PATTERNS:
            self._compiled_patterns.append((
                tp.compiled_pattern,
                tp.equipment_type,
                tp.prefix,
            ))

        for pattern, eq_type in self.ALTERNATIVE_PATTERNS:
            self._compiled_patterns.append((
                re.compile(pattern, re.IGNORECASE),
                eq_type,
                "alternative",
            ))

    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract equipment tags from text.

        Args:
            text: Source text to extract from

        Returns:
            List of extracted equipment tag entities
        """
        entities = []
        seen_tags = set()

        for pattern, equipment_type, prefix in self._compiled_patterns:
            for match in pattern.finditer(text):
                tag = match.group(0).upper()

                # Skip duplicates
                if tag in seen_tags:
                    continue
                seen_tags.add(tag)

                # Normalize tag format
                normalized = self._normalize_tag(tag)

                entity = ExtractedEntity(
                    id=f"eq_{hashlib.md5(tag.encode()).hexdigest()[:12]}",
                    type=EntityType.EQUIPMENT_TAG,
                    value=tag,
                    normalized_value=normalized,
                    confidence=1.0,
                    source_text=text,
                    start_position=match.start(),
                    end_position=match.end(),
                    metadata={
                        "equipment_type": equipment_type,
                        "prefix": prefix,
                        "groups": match.groups(),
                    },
                )
                entities.append(entity)

        logger.debug(f"Extracted {len(entities)} equipment tags from text")
        return entities

    def _normalize_tag(self, tag: str) -> str:
        """Normalize equipment tag to standard format."""
        # Remove extra spaces
        tag = re.sub(r"\s+", "", tag)

        # Ensure uppercase
        tag = tag.upper()

        # Standardize separator (prefer dash)
        # Handle cases like P101 -> P-101
        match = re.match(r"([A-Z]+)(\d+)([A-Z]*)", tag)
        if match and "-" not in tag:
            prefix, number, suffix = match.groups()
            tag = f"{prefix}-{number}{suffix}"

        return tag

    def get_supported_types(self) -> List[EntityType]:
        """Get entity types this extractor supports."""
        return [EntityType.EQUIPMENT_TAG, EntityType.EQUIPMENT]

    def parse_tag(self, tag: str) -> Dict[str, Any]:
        """
        Parse equipment tag into components.

        Args:
            tag: Equipment tag string

        Returns:
            Dictionary with tag components

        Example:
            >>> extractor.parse_tag("P-101A/B")
            {'prefix': 'P', 'number': '101', 'suffix': 'A/B', 'type': 'pump'}
        """
        normalized = self._normalize_tag(tag)

        # Extract components
        match = re.match(r"([A-Z]+)-(\d+)([A-Z/]*)", normalized)
        if not match:
            return {"raw": tag, "valid": False}

        prefix, number, suffix = match.groups()

        # Determine equipment type
        eq_type = "unknown"
        for tp in self.TAG_PATTERNS:
            if any(p == prefix for p in tp.prefix.split("/")):
                eq_type = tp.equipment_type
                break

        return {
            "prefix": prefix,
            "number": number,
            "suffix": suffix or None,
            "equipment_type": eq_type,
            "normalized": normalized,
            "valid": True,
        }


# =============================================================================
# Process Parameter Extractor
# =============================================================================

class ProcessParameterExtractor(BaseExtractor):
    """
    Extract process parameters and measurements from text.

    Recognizes:
    - Temperature values (100 degC, 500 F, 373 K)
    - Pressure values (10 bar, 150 psig, 1.5 MPa)
    - Flow rates (100 kg/h, 50 m3/h, 200 gpm)
    - Concentrations (5%, 100 ppm, 50 mg/L)
    - Energy/power values (100 kW, 50 MW, 1000 MMBtu/h)

    Example:
        >>> extractor = ProcessParameterExtractor()
        >>> entities = extractor.extract("Operating at 350 degC and 25 bar")
        >>> for e in entities:
        ...     print(f"{e.metadata['parameter']}: {e.value}")
    """

    # Parameter patterns: (pattern, parameter_type, unit_type)
    PARAMETER_PATTERNS = [
        # Temperature
        (r"(\d+(?:\.\d+)?)\s*(deg\s*[CF]|[°]?[CF]|K(?:elvin)?)\b", "temperature", "temperature"),
        (r"(\d+(?:\.\d+)?)\s*(degC|degF|deg\s*C|deg\s*F)", "temperature", "temperature"),

        # Pressure
        (r"(\d+(?:\.\d+)?)\s*(bar|barg|bara|psig?|psia|kPa|MPa|atm|mmHg)\b", "pressure", "pressure"),

        # Flow rates - mass
        (r"(\d+(?:\.\d+)?)\s*(kg/h|kg/s|t/h|lb/h|klb/h|tph)\b", "mass_flow", "mass_flow"),

        # Flow rates - volumetric
        (r"(\d+(?:\.\d+)?)\s*(m3/h|m3/s|L/min|L/s|gpm|scfm|Nm3/h|cfm)\b", "volumetric_flow", "volume_flow"),

        # Energy/Power
        (r"(\d+(?:\.\d+)?)\s*(kW|MW|GW|hp|Btu/h|MMBtu/h|GJ/h|MJ/h)\b", "power", "power"),
        (r"(\d+(?:\.\d+)?)\s*(kJ|MJ|GJ|kWh|MWh|Btu|MMBtu|therm)\b", "energy", "energy"),

        # Concentration
        (r"(\d+(?:\.\d+)?)\s*(%|pct|percent)\b", "concentration", "percentage"),
        (r"(\d+(?:\.\d+)?)\s*(ppm|ppb|mg/L|mg/Nm3|g/L)\b", "concentration", "concentration"),

        # Heat transfer
        (r"(\d+(?:\.\d+)?)\s*(W/m2K|Btu/h\.ft2\.F|kW/m2)\b", "heat_transfer_coefficient", "htc"),

        # Efficiency
        (r"(\d+(?:\.\d+)?)\s*%\s*(?:efficiency|eff\.?)", "efficiency", "percentage"),

        # Specific enthalpy
        (r"(\d+(?:\.\d+)?)\s*(kJ/kg|Btu/lb|J/g)\b", "specific_enthalpy", "specific_enthalpy"),

        # Area
        (r"(\d+(?:\.\d+)?)\s*(m2|ft2|cm2|in2)\b", "area", "area"),

        # Length/thickness
        (r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft)\b", "length", "length"),

        # Time
        (r"(\d+(?:\.\d+)?)\s*(s|sec|min|h|hr|hour|day)\b", "time", "time"),
    ]

    def __init__(self):
        """Initialize process parameter extractor."""
        self._compiled_patterns: List[Tuple[Pattern, str, str]] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile all regex patterns."""
        for pattern, param_type, unit_type in self.PARAMETER_PATTERNS:
            self._compiled_patterns.append((
                re.compile(pattern, re.IGNORECASE),
                param_type,
                unit_type,
            ))

    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract process parameters from text.

        Args:
            text: Source text to extract from

        Returns:
            List of extracted parameter entities
        """
        entities = []
        seen_values = set()

        for pattern, param_type, unit_type in self._compiled_patterns:
            for match in pattern.finditer(text):
                value_str = match.group(0)
                numeric_value = match.group(1)
                unit = match.group(2)

                # Create unique key to avoid duplicates
                key = f"{numeric_value}_{unit}"
                if key in seen_values:
                    continue
                seen_values.add(key)

                # Normalize unit
                normalized_unit = self._normalize_unit(unit)

                entity = ExtractedEntity(
                    id=f"param_{hashlib.md5(value_str.encode()).hexdigest()[:12]}",
                    type=EntityType.PROCESS_PARAMETER,
                    value=value_str,
                    normalized_value=f"{numeric_value} {normalized_unit}",
                    confidence=1.0,
                    source_text=text,
                    start_position=match.start(),
                    end_position=match.end(),
                    metadata={
                        "parameter_type": param_type,
                        "unit_type": unit_type,
                        "numeric_value": float(numeric_value),
                        "unit": normalized_unit,
                        "original_unit": unit,
                    },
                )
                entities.append(entity)

        logger.debug(f"Extracted {len(entities)} process parameters from text")
        return entities

    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit to standard form."""
        # Standard unit mappings
        mappings = {
            "degc": "degC",
            "deg c": "degC",
            "deg_c": "degC",
            "celsius": "degC",
            "degf": "degF",
            "deg f": "degF",
            "fahrenheit": "degF",
            "kelvin": "K",
            "psig": "psig",
            "psi": "psi",
            "barg": "barg",
            "bara": "bara",
            "percent": "%",
            "pct": "%",
        }

        unit_lower = unit.lower().strip()
        return mappings.get(unit_lower, unit)

    def get_supported_types(self) -> List[EntityType]:
        """Get entity types this extractor supports."""
        return [EntityType.PROCESS_PARAMETER, EntityType.MEASUREMENT, EntityType.UNIT]


# =============================================================================
# Safety Interlock Extractor
# =============================================================================

class SafetyInterlockExtractor(BaseExtractor):
    """
    Extract safety interlock references from text.

    Recognizes:
    - Level interlocks (LLWC, LL, HH, L, H)
    - Pressure interlocks (HHP, LLP)
    - Temperature interlocks (HHT, LLT)
    - Flow interlocks (LLF, HHF)
    - Flame interlocks (FFS, flame failure)
    - SIL levels (SIL 1, SIL 2, SIL 3)

    Example:
        >>> extractor = SafetyInterlockExtractor()
        >>> entities = extractor.extract("LLWC trips fuel, SIL 2 required per NFPA 85")
        >>> for e in entities:
        ...     print(f"{e.metadata['interlock_type']}: {e.value}")
    """

    # Interlock patterns
    INTERLOCK_PATTERNS = [
        # Level interlocks
        (r"\b(LLWC|LL[LH]?L?|HH[LH]?L?|LAHH|LALL)\b", "level", "Low-Low Water Cutoff / Level"),
        (r"\b(low[\s-]*low[\s-]*(?:water[\s-]*)?(?:cutoff|level|alarm))\b", "level", "Level interlock"),

        # Pressure interlocks
        (r"\b(HHP|LLP|PAHH|PALL|PSH|PSL|PSHH|PSLL)\b", "pressure", "Pressure interlock"),
        (r"\b(high[\s-]*high[\s-]*pressure|low[\s-]*low[\s-]*pressure)\b", "pressure", "Pressure interlock"),

        # Temperature interlocks
        (r"\b(HHT|LLT|TAHH|TALL|TSH|TSL|TSHH|TSLL)\b", "temperature", "Temperature interlock"),
        (r"\b(high[\s-]*high[\s-]*temp(?:erature)?|low[\s-]*low[\s-]*temp(?:erature)?)\b", "temperature", "Temperature interlock"),

        # Flow interlocks
        (r"\b(LLF|HHF|FALL|FAHH|FSH|FSL|FSHH|FSLL)\b", "flow", "Flow interlock"),
        (r"\b(low[\s-]*low[\s-]*flow|high[\s-]*high[\s-]*flow)\b", "flow", "Flow interlock"),

        # Flame interlocks
        (r"\b(FFS|flame[\s-]*failure|loss[\s-]*of[\s-]*flame|flame[\s-]*detector)\b", "flame", "Flame failure"),
        (r"\b(BMS|burner[\s-]*management[\s-]*system)\b", "flame", "Burner management"),

        # Combustion interlocks
        (r"\b(purge[\s-]*interlock|pre[\s-]*purge|post[\s-]*purge)\b", "combustion", "Purge interlock"),
        (r"\b(combustion[\s-]*air[\s-]*(?:proving|interlock)|air[\s-]*flow[\s-]*proving)\b", "combustion", "Combustion air"),

        # Emergency shutdown
        (r"\b(ESD|E[\s-]*?S[\s-]*?D|emergency[\s-]*shut[\s-]*down)\b", "emergency", "Emergency shutdown"),
        (r"\b(SIS|safety[\s-]*instrumented[\s-]*system)\b", "safety", "Safety instrumented system"),

        # SIL levels
        (r"\b(SIL[\s-]*[1234]|SIL[\s]*level[\s]*[1234])\b", "sil", "Safety Integrity Level"),

        # General trip/shutdown
        (r"\b(trip|shutdown|cutoff|interlock)\b", "general", "General safety"),
    ]

    # SIL-specific patterns
    SIL_PATTERN = re.compile(r"\bSIL[\s-]*([1234])\b", re.IGNORECASE)

    def __init__(self):
        """Initialize safety interlock extractor."""
        self._compiled_patterns: List[Tuple[Pattern, str, str]] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile all regex patterns."""
        for pattern, interlock_type, description in self.INTERLOCK_PATTERNS:
            self._compiled_patterns.append((
                re.compile(pattern, re.IGNORECASE),
                interlock_type,
                description,
            ))

    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract safety interlock references from text.

        Args:
            text: Source text to extract from

        Returns:
            List of extracted safety interlock entities
        """
        entities = []
        seen_values = set()

        for pattern, interlock_type, description in self._compiled_patterns:
            for match in pattern.finditer(text):
                value = match.group(0).upper()

                # Skip generic terms without context
                if value.lower() in {"trip", "shutdown", "cutoff", "interlock"}:
                    # Check for context
                    context_start = max(0, match.start() - 20)
                    context = text[context_start:match.end()].lower()
                    if not any(kw in context for kw in ["safety", "emergency", "sil", "interlock"]):
                        continue

                # Skip duplicates
                key = f"{value}_{interlock_type}"
                if key in seen_values:
                    continue
                seen_values.add(key)

                # Check for SIL level
                sil_level = None
                sil_match = self.SIL_PATTERN.search(text)
                if sil_match:
                    sil_level = int(sil_match.group(1))

                entity = ExtractedEntity(
                    id=f"safety_{hashlib.md5(value.encode()).hexdigest()[:12]}",
                    type=EntityType.SAFETY_INTERLOCK,
                    value=value,
                    normalized_value=self._normalize_interlock(value),
                    confidence=0.9 if interlock_type != "general" else 0.7,
                    source_text=text,
                    start_position=match.start(),
                    end_position=match.end(),
                    metadata={
                        "interlock_type": interlock_type,
                        "description": description,
                        "sil_level": sil_level,
                    },
                )
                entities.append(entity)

        logger.debug(f"Extracted {len(entities)} safety interlocks from text")
        return entities

    def _normalize_interlock(self, value: str) -> str:
        """Normalize interlock name."""
        # Standard abbreviations
        mappings = {
            "LLWC": "Low-Low Water Cutoff",
            "HHP": "High-High Pressure",
            "LLP": "Low-Low Pressure",
            "HHT": "High-High Temperature",
            "LLT": "Low-Low Temperature",
            "LLF": "Low-Low Flow",
            "HHF": "High-High Flow",
            "FFS": "Flame Failure",
            "ESD": "Emergency Shutdown",
            "SIS": "Safety Instrumented System",
            "BMS": "Burner Management System",
        }

        return mappings.get(value.upper(), value)

    def get_supported_types(self) -> List[EntityType]:
        """Get entity types this extractor supports."""
        return [EntityType.SAFETY_INTERLOCK, EntityType.HAZARD]


# =============================================================================
# Standards Reference Extractor
# =============================================================================

class StandardsReferenceExtractor(BaseExtractor):
    """
    Extract standards references from text.

    Recognizes:
    - ASME standards (ASME BPVC Section I, ASME B31.1)
    - API standards (API 560, API 530, API 579)
    - NFPA standards (NFPA 85, NFPA 86)
    - IEC/ISA standards (IEC 61511, ISA 84)
    - ISO standards (ISO 9001, ISO 14001)
    - Section references (Section 4.3.2, Clause 5.1)

    Example:
        >>> extractor = StandardsReferenceExtractor()
        >>> entities = extractor.extract("Per NFPA 85 Section 4.6.2 and API 560")
        >>> for e in entities:
        ...     print(f"{e.metadata['standards_body']}: {e.value}")
    """

    # Standards patterns
    STANDARDS_PATTERNS = [
        # ASME standards
        (r"\b(ASME[\s-]*BPVC[\s-]*(?:Section[\s-]*)?[IVX]+(?:[\s-]*Division[\s-]*\d)?)\b", "ASME", "Boiler and Pressure Vessel Code"),
        (r"\b(ASME[\s-]*B31\.\d+)\b", "ASME", "Piping Code"),
        (r"\b(ASME[\s-]*PTC[\s-]*\d+(?:\.\d+)?)\b", "ASME", "Performance Test Code"),
        (r"\b(ASME[\s-]*CSD-\d+)\b", "ASME", "Controls and Safety Devices"),

        # API standards
        (r"\b(API[\s-]*\d{2,3}(?:-\d)?(?:[\s-]*(?:1st|2nd|3rd|\d+th)?[\s-]*Ed(?:ition)?)?)\b", "API", "API Standard"),
        (r"\b(API[\s-]*RP[\s-]*\d{2,3}[A-Z]?)\b", "API", "API Recommended Practice"),

        # NFPA standards
        (r"\b(NFPA[\s-]*\d{1,3}[A-Z]?)\b", "NFPA", "NFPA Standard"),

        # IEC standards
        (r"\b(IEC[\s-]*\d{4,5}(?:-\d+)?(?:-\d+)?)\b", "IEC", "IEC Standard"),

        # ISA standards
        (r"\b(ISA[\s-]*\d{1,3}(?:\.\d+)?)\b", "ISA", "ISA Standard"),
        (r"\b(ISA[\s-]*(?:S|TR)\d{1,3}(?:\.\d+)?)\b", "ISA", "ISA Standard/Technical Report"),

        # ISO standards
        (r"\b(ISO[\s-]*\d{4,5}(?:-\d+)?(?::\d{4})?)\b", "ISO", "ISO Standard"),

        # EN standards
        (r"\b(EN[\s-]*\d{3,5}(?:-\d+)?)\b", "EN", "European Standard"),

        # OSHA regulations
        (r"\b(OSHA[\s-]*\d{4}\.\d{2,3})\b", "OSHA", "OSHA Regulation"),
        (r"\b(29[\s-]*CFR[\s-]*\d{4}\.\d{2,3})\b", "OSHA", "CFR Regulation"),

        # TEMA (Heat exchanger)
        (r"\b(TEMA[\s-]*(?:Class[\s-]*)?[A-Z])\b", "TEMA", "TEMA Standard"),
    ]

    # Section/clause reference pattern
    SECTION_PATTERN = re.compile(
        r"\b((?:Section|Sect\.|Clause|Para(?:graph)?\.?|Art(?:icle)?\.?)[\s-]*\d+(?:\.\d+)*(?:\.\d+)?)\b",
        re.IGNORECASE
    )

    def __init__(self):
        """Initialize standards reference extractor."""
        self._compiled_patterns: List[Tuple[Pattern, str, str]] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile all regex patterns."""
        for pattern, body, description in self.STANDARDS_PATTERNS:
            self._compiled_patterns.append((
                re.compile(pattern, re.IGNORECASE),
                body,
                description,
            ))

    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract standards references from text.

        Args:
            text: Source text to extract from

        Returns:
            List of extracted standards reference entities
        """
        entities = []
        seen_standards = set()

        # Extract main standard references
        for pattern, body, description in self._compiled_patterns:
            for match in pattern.finditer(text):
                value = match.group(0)
                normalized = self._normalize_standard(value, body)

                # Skip duplicates
                if normalized in seen_standards:
                    continue
                seen_standards.add(normalized)

                # Look for associated section reference
                section = self._find_associated_section(text, match.end())

                entity = ExtractedEntity(
                    id=f"std_{hashlib.md5(normalized.encode()).hexdigest()[:12]}",
                    type=EntityType.STANDARD,
                    value=value,
                    normalized_value=normalized,
                    confidence=1.0,
                    source_text=text,
                    start_position=match.start(),
                    end_position=match.end(),
                    metadata={
                        "standards_body": body,
                        "description": description,
                        "section": section,
                    },
                )
                entities.append(entity)

        # Extract section references that might be standalone
        for match in self.SECTION_PATTERN.finditer(text):
            # Check if this section is already associated with a standard
            if any(match.start() < e.end_position + 20 and match.start() > e.start_position
                   for e in entities):
                continue

            value = match.group(0)
            entity = ExtractedEntity(
                id=f"sect_{hashlib.md5(value.encode()).hexdigest()[:12]}",
                type=EntityType.STANDARD_SECTION,
                value=value,
                normalized_value=value,
                confidence=0.8,  # Lower confidence for standalone sections
                source_text=text,
                start_position=match.start(),
                end_position=match.end(),
                metadata={
                    "standards_body": "unknown",
                    "description": "Section reference",
                },
            )
            entities.append(entity)

        logger.debug(f"Extracted {len(entities)} standards references from text")
        return entities

    def _normalize_standard(self, value: str, body: str) -> str:
        """Normalize standard reference."""
        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", value).strip()

        # Standardize format
        normalized = normalized.upper()

        # Add body prefix if missing
        if not normalized.startswith(body):
            normalized = f"{body} {normalized}"

        return normalized

    def _find_associated_section(self, text: str, start_pos: int) -> Optional[str]:
        """Find section reference near a standard reference."""
        # Look within 30 characters after the standard reference
        search_text = text[start_pos:start_pos + 30]
        match = self.SECTION_PATTERN.search(search_text)
        if match:
            return match.group(0)
        return None

    def get_supported_types(self) -> List[EntityType]:
        """Get entity types this extractor supports."""
        return [EntityType.STANDARD, EntityType.STANDARD_SECTION]


# =============================================================================
# Main Entity Extractor (Composite)
# =============================================================================

class EntityExtractor:
    """
    Composite entity extractor combining all specialized extractors.

    Provides a single interface for extracting all entity types from text.

    Example:
        >>> extractor = EntityExtractor()
        >>> result = extractor.extract_all(
        ...     "Boiler B-101 operates at 350 degC with LLWC per NFPA 85"
        ... )
        >>> print(f"Found {result.total_count} entities")
        >>> for entity in result.entities:
        ...     print(f"  {entity.type}: {entity.value}")
    """

    def __init__(
        self,
        enable_equipment: bool = True,
        enable_parameters: bool = True,
        enable_safety: bool = True,
        enable_standards: bool = True,
    ):
        """
        Initialize composite entity extractor.

        Args:
            enable_equipment: Enable equipment tag extraction
            enable_parameters: Enable process parameter extraction
            enable_safety: Enable safety interlock extraction
            enable_standards: Enable standards reference extraction
        """
        self.extractors: List[BaseExtractor] = []

        if enable_equipment:
            self.extractors.append(EquipmentTagExtractor())
        if enable_parameters:
            self.extractors.append(ProcessParameterExtractor())
        if enable_safety:
            self.extractors.append(SafetyInterlockExtractor())
        if enable_standards:
            self.extractors.append(StandardsReferenceExtractor())

        logger.info(f"EntityExtractor initialized with {len(self.extractors)} extractors")

    def extract_all(self, text: str) -> ExtractionResult:
        """
        Extract all entity types from text.

        Args:
            text: Source text to extract from

        Returns:
            ExtractionResult with all extracted entities

        Example:
            >>> result = extractor.extract_all("P-101 operates at 100 degC")
            >>> print(result.total_count)
        """
        start_time = datetime.utcnow()
        all_entities = []
        warnings = []

        for extractor in self.extractors:
            try:
                entities = extractor.extract(text)
                all_entities.extend(entities)
            except Exception as e:
                warning = f"Extractor {type(extractor).__name__} failed: {str(e)}"
                warnings.append(warning)
                logger.warning(warning)

        # Count by type
        by_type: Dict[str, int] = {}
        for entity in all_entities:
            entity_type = entity.type if isinstance(entity.type, str) else entity.type.value
            by_type[entity_type] = by_type.get(entity_type, 0) + 1

        extraction_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ExtractionResult(
            source_text=text,
            entities=all_entities,
            total_count=len(all_entities),
            by_type=by_type,
            extraction_time_ms=extraction_time,
            warnings=warnings,
        )

    def extract_by_type(
        self,
        text: str,
        entity_type: EntityType,
    ) -> List[ExtractedEntity]:
        """
        Extract entities of a specific type.

        Args:
            text: Source text to extract from
            entity_type: Type of entities to extract

        Returns:
            List of extracted entities of the specified type
        """
        for extractor in self.extractors:
            if entity_type in extractor.get_supported_types():
                return extractor.extract(text)
        return []

    def get_supported_types(self) -> Set[EntityType]:
        """Get all supported entity types."""
        types: Set[EntityType] = set()
        for extractor in self.extractors:
            types.update(extractor.get_supported_types())
        return types


# =============================================================================
# Module-level convenience functions
# =============================================================================

_extractor_instance: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Get or create the global entity extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = EntityExtractor()
    return _extractor_instance


def extract_entities(text: str) -> ExtractionResult:
    """
    Convenience function to extract all entities from text.

    Args:
        text: Source text to extract from

    Returns:
        ExtractionResult with all extracted entities

    Example:
        >>> result = extract_entities("Boiler B-101 at 350 degC per NFPA 85")
        >>> print(result.total_count)
    """
    return get_entity_extractor().extract_all(text)
