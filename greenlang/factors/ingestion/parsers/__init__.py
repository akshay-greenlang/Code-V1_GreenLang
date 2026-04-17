# -*- coding: utf-8 -*-
"""
Source parser package for GreenLang Factors catalog (F018).

Provides:
- ``BaseSourceParser``: Abstract base class for all source parsers.
- ``ParserRegistry``: Plugin registry for dynamic parser lookup by source_id.
- All 8 built-in parsers (EPA, eGRID, DESNZ, IPCC, CBAM, GHG Protocol, TCR, Green-e).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class BaseSourceParser(ABC):
    """Abstract base class for source-specific emission factor parsers.

    Subclasses must implement ``parse()`` and ``validate_schema()``.
    """

    source_id: str = ""
    parser_id: str = ""
    parser_version: str = "1.0"
    supported_formats: List[str] = []

    @abstractmethod
    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse source data into normalized factor dicts.

        Args:
            data: JSON-decoded dict from the source file.

        Returns:
            List of factor dicts ready for QA validation.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate source data matches expected schema.

        Returns:
            (ok, issues) tuple.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} source_id={self.source_id!r} v{self.parser_version}>"


class ParserRegistry:
    """Plugin registry for source parsers.

    Parsers register themselves by source_id. The registry supports
    versioned parsers and lookup by source_id.
    """

    def __init__(self) -> None:
        self._parsers: Dict[str, BaseSourceParser] = {}

    def register(self, parser: BaseSourceParser) -> None:
        """Register a parser instance."""
        key = parser.source_id
        if key in self._parsers:
            logger.warning(
                "Overwriting parser for source_id=%s (old=%s new=%s)",
                key, self._parsers[key].parser_id, parser.parser_id,
            )
        self._parsers[key] = parser
        logger.debug("Registered parser %s for source_id=%s", parser.parser_id, key)

    def get(self, source_id: str) -> Optional[BaseSourceParser]:
        """Get parser by source_id, or None if not registered."""
        return self._parsers.get(source_id)

    def list_source_ids(self) -> List[str]:
        """List all registered source_ids."""
        return sorted(self._parsers.keys())

    def list_parsers(self) -> List[BaseSourceParser]:
        """List all registered parsers."""
        return list(self._parsers.values())

    def __len__(self) -> int:
        return len(self._parsers)

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._parsers


# ---- Built-in parser adapters ----

class EPAGHGHubParser(BaseSourceParser):
    source_id = "epa_hub"
    parser_id = "epa_ghg_hub_v1"
    parser_version = "1.0"
    supported_formats = ["json"]

    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        from greenlang.factors.ingestion.parsers.epa_ghg_hub import parse_epa_ghg_hub
        return parse_epa_ghg_hub(data)

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        issues = []
        if "metadata" not in data:
            issues.append("missing 'metadata' key")
        sections = ["stationary_combustion", "mobile_combustion", "electricity"]
        if not any(k in data for k in sections):
            issues.append(f"expected at least one of {sections}")
        return (len(issues) == 0, issues)


class EGridParser(BaseSourceParser):
    source_id = "egrid"
    parser_id = "egrid_v1"
    parser_version = "1.0"
    supported_formats = ["json"]

    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        from greenlang.factors.ingestion.parsers.egrid import parse_egrid
        return parse_egrid(data)

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        issues = []
        if not any(k in data for k in ("subregions", "states", "national")):
            issues.append("expected 'subregions', 'states', or 'national' key")
        return (len(issues) == 0, issues)


class DESNZUKParser(BaseSourceParser):
    source_id = "desnz_ghg_conversion"
    parser_id = "desnz_uk_v1"
    parser_version = "1.0"
    supported_formats = ["json"]

    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        from greenlang.factors.ingestion.parsers.desnz_uk import parse_desnz_uk
        return parse_desnz_uk(data)

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        issues = []
        sections = ["scope1_fuels", "scope2_electricity", "scope3_wtt"]
        if not any(k in data for k in sections):
            issues.append(f"expected at least one of {sections}")
        return (len(issues) == 0, issues)


class DEFRAParser(BaseSourceParser):
    """Alias parser for defra_conversion source_id (same parser as DESNZ)."""
    source_id = "defra_conversion"
    parser_id = "desnz_uk_v1"
    parser_version = "1.0"
    supported_formats = ["json"]

    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        from greenlang.factors.ingestion.parsers.desnz_uk import parse_desnz_uk
        return parse_desnz_uk(data)

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        return DESNZUKParser().validate_schema(data)


class IPCCDefaultsParser(BaseSourceParser):
    source_id = "ipcc_defaults"
    parser_id = "ipcc_defaults_v1"
    parser_version = "1.0"
    supported_formats = ["json"]

    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        from greenlang.factors.ingestion.parsers.ipcc_defaults import parse_ipcc_defaults
        return parse_ipcc_defaults(data)

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        issues = []
        sections = ["energy_stationary", "energy_mobile", "industrial_processes", "agriculture", "waste"]
        if not any(k in data for k in sections):
            issues.append(f"expected at least one of {sections}")
        return (len(issues) == 0, issues)


class CBAMFullParser(BaseSourceParser):
    source_id = "eu_cbam"
    parser_id = "cbam_full_v1"
    parser_version = "1.0"
    supported_formats = ["json"]

    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        from greenlang.factors.ingestion.parsers.cbam_full import parse_cbam_full
        return parse_cbam_full(data)

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        issues = []
        if "products" not in data:
            issues.append("missing 'products' key")
        return (len(issues) == 0, issues)


class GHGProtocolParser(BaseSourceParser):
    source_id = "ghgp_method_refs"
    parser_id = "ghg_protocol_v1"
    parser_version = "1.0"
    supported_formats = ["json"]

    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        from greenlang.factors.ingestion.parsers.ghg_protocol import parse_ghg_protocol
        return parse_ghg_protocol(data)

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        issues = []
        cats = [k for k in data if k.startswith("cat")]
        if not cats:
            issues.append("expected at least one 'catN_*' section")
        return (len(issues) == 0, issues)


class TCRParser(BaseSourceParser):
    source_id = "tcr_grp_defaults"
    parser_id = "tcr_v1"
    parser_version = "1.0"
    supported_formats = ["json"]

    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        from greenlang.factors.ingestion.parsers.tcr import parse_tcr
        return parse_tcr(data)

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        issues = []
        sections = ["stationary_combustion", "mobile_combustion", "electricity"]
        if not any(k in data for k in sections):
            issues.append(f"expected at least one of {sections}")
        return (len(issues) == 0, issues)


class GreenEParser(BaseSourceParser):
    source_id = "green_e_residual"
    parser_id = "green_e_v1"
    parser_version = "1.0"
    supported_formats = ["json"]

    def parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        from greenlang.factors.ingestion.parsers.green_e import parse_green_e
        return parse_green_e(data)

    def validate_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        issues = []
        if "residual_mix" not in data:
            issues.append("missing 'residual_mix' key")
        return (len(issues) == 0, issues)


# ---- Default registry ----

def build_default_registry() -> ParserRegistry:
    """Build a ParserRegistry pre-loaded with all built-in parsers."""
    registry = ParserRegistry()
    for cls in (
        EPAGHGHubParser,
        EGridParser,
        DESNZUKParser,
        DEFRAParser,
        IPCCDefaultsParser,
        CBAMFullParser,
        GHGProtocolParser,
        TCRParser,
        GreenEParser,
    ):
        registry.register(cls())
    logger.info("Built default parser registry with %d parsers", len(registry))
    return registry
