"""
Citation Data Structures for AI Agents

Provides standardized citation tracking for emission factors, data sources,
and calculations used by AI agents. Ensures transparency and auditability.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
import hashlib


class EmissionFactorCitation(BaseModel):
    """
    Citation for an emission factor used in calculations.

    Provides complete traceability for emission factors including source,
    version, confidence, and content identifier (CID) for verification.

    Attributes:
        source: Data source name (e.g., "EPA eGRID 2025", "IPCC AR6")
        factor_name: Descriptive name of the factor (e.g., "Natural Gas Combustion")
        value: The numeric value of the emission factor
        unit: Unit of measurement (e.g., "kgCO2e/m3", "gCO2e/kWh")
        ef_cid: Content identifier for the emission factor (for verification)
        version: Version of the data source (e.g., "2025.1", "AR6")
        last_updated: When the data was last updated
        confidence: Confidence level (high/medium/low)
        region: Geographic region if applicable (e.g., "US-WECC", "EU")
        gwp_set: Global warming potential set used (e.g., "AR6GWP100")
        metadata: Additional context (method, assumptions, etc.)

    Example:
        >>> citation = EmissionFactorCitation(
        ...     source="EPA eGRID 2025",
        ...     factor_name="US Grid Average",
        ...     value=0.385,
        ...     unit="kgCO2e/kWh",
        ...     ef_cid="ef_abc123...",
        ...     version="2025.1",
        ...     last_updated=datetime(2025, 1, 15),
        ...     confidence="high",
        ...     region="US"
        ... )
    """

    source: str = Field(..., description="Data source name")
    factor_name: str = Field(..., description="Descriptive name of emission factor")
    value: float = Field(..., description="Numeric value of emission factor")
    unit: str = Field(..., description="Unit of measurement")
    ef_cid: str = Field(..., description="Content identifier for emission factor")
    version: Optional[str] = Field(None, description="Version of data source")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    confidence: str = Field(default="medium", description="Confidence level: high/medium/low")
    region: Optional[str] = Field(None, description="Geographic region if applicable")
    gwp_set: Optional[str] = Field(None, description="GWP set used (e.g., AR6GWP100)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    def formatted(self) -> str:
        """
        Human-readable citation string.

        Returns:
            Formatted citation string for display
        """
        parts = [f"{self.source}"]
        if self.version:
            parts.append(f"v{self.version}")
        if self.region:
            parts.append(f"({self.region})")

        result = " ".join(parts)
        result += f": {self.factor_name} = {self.value} {self.unit}"

        if self.last_updated:
            result += f" [Updated: {self.last_updated.strftime('%Y-%m-%d')}]"

        if self.confidence:
            result += f" (Confidence: {self.confidence})"

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        return {
            "source": self.source,
            "factor_name": self.factor_name,
            "value": self.value,
            "unit": self.unit,
            "ef_cid": self.ef_cid,
            "version": self.version,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "confidence": self.confidence,
            "region": self.region,
            "gwp_set": self.gwp_set,
            "metadata": self.metadata,
            "formatted": self.formatted(),
        }


class CalculationCitation(BaseModel):
    """
    Citation for a calculation step in an AI agent.

    Tracks intermediate calculations and their provenance, allowing
    complete audit trail from inputs to outputs.

    Attributes:
        step_name: Name of calculation step (e.g., "calculate_emissions")
        formula: Formula or method used
        inputs: Input values and their sources
        output: Output value and unit
        tool_call_id: Tool call ID from runtime (if applicable)
        timestamp: When calculation was performed
        metadata: Additional context
    """

    step_name: str = Field(..., description="Name of calculation step")
    formula: Optional[str] = Field(None, description="Formula or method used")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input values")
    output: Dict[str, Any] = Field(default_factory=dict, description="Output value and unit")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID from runtime")
    timestamp: datetime = Field(default_factory=datetime.now, description="Calculation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    def formatted(self) -> str:
        """
        Human-readable calculation citation.

        Returns:
            Formatted string for display
        """
        result = f"{self.step_name}"
        if self.formula:
            result += f": {self.formula}"

        if self.output:
            if "value" in self.output and "unit" in self.output:
                result += f" = {self.output['value']} {self.output['unit']}"

        return result


class DataSourceCitation(BaseModel):
    """
    Citation for external data sources used by agents.

    Tracks data retrieved from external sources like weather APIs,
    grid intensity APIs, or databases.

    Attributes:
        source_name: Name of data source
        source_type: Type (api/database/file/connector)
        query: Query or request parameters
        timestamp: When data was retrieved
        checksum: SHA-256 checksum of retrieved data
        url: URL if applicable
        metadata: Additional context
    """

    source_name: str = Field(..., description="Name of data source")
    source_type: str = Field(..., description="Type: api/database/file/connector")
    query: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    timestamp: datetime = Field(default_factory=datetime.now, description="Retrieval timestamp")
    checksum: Optional[str] = Field(None, description="SHA-256 checksum of data")
    url: Optional[str] = Field(None, description="URL if applicable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    def formatted(self) -> str:
        """
        Human-readable data source citation.

        Returns:
            Formatted string for display
        """
        result = f"{self.source_name} ({self.source_type})"
        if self.url:
            result += f": {self.url}"
        result += f" [Retrieved: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}]"
        return result


def generate_ef_cid(source: str, factor_name: str, value: float, unit: str, version: Optional[str] = None) -> str:
    """
    Generate a content identifier (CID) for an emission factor.

    Creates a deterministic hash of the emission factor's key attributes
    to enable verification and deduplication.

    Args:
        source: Data source name
        factor_name: Descriptive name
        value: Numeric value
        unit: Unit of measurement
        version: Version (optional)

    Returns:
        Content identifier string (ef_<16-char-hex>)

    Example:
        >>> cid = generate_ef_cid("EPA eGRID", "US Average", 0.385, "kgCO2e/kWh", "2025.1")
        >>> print(cid)
        ef_a1b2c3d4e5f6g7h8
    """
    # Create canonical string representation
    version_str = version if version else "unknown"
    canonical = f"{source}|{factor_name}|{value}|{unit}|{version_str}"

    # Generate SHA-256 hash
    hash_bytes = hashlib.sha256(canonical.encode("utf-8")).digest()

    # Take first 8 bytes (16 hex chars)
    short_hash = hash_bytes[:8].hex()

    return f"ef_{short_hash}"


def create_emission_factor_citation(
    source: str,
    factor_name: str,
    value: float,
    unit: str,
    version: Optional[str] = None,
    last_updated: Optional[datetime] = None,
    confidence: str = "medium",
    region: Optional[str] = None,
    gwp_set: Optional[str] = None,
    **kwargs: Any
) -> EmissionFactorCitation:
    """
    Convenience function to create emission factor citations.

    Automatically generates EF CID and populates citation structure.

    Args:
        source: Data source name
        factor_name: Descriptive name
        value: Numeric value
        unit: Unit of measurement
        version: Version (optional)
        last_updated: Last update timestamp (optional)
        confidence: Confidence level (default: "medium")
        region: Geographic region (optional)
        gwp_set: GWP set used (optional)
        **kwargs: Additional metadata

    Returns:
        EmissionFactorCitation instance

    Example:
        >>> citation = create_emission_factor_citation(
        ...     source="EPA eGRID 2025",
        ...     factor_name="Natural Gas Combustion",
        ...     value=5.31,
        ...     unit="kgCO2e/therm",
        ...     version="2025.1",
        ...     confidence="high"
        ... )
    """
    ef_cid = generate_ef_cid(source, factor_name, value, unit, version)

    return EmissionFactorCitation(
        source=source,
        factor_name=factor_name,
        value=value,
        unit=unit,
        ef_cid=ef_cid,
        version=version,
        last_updated=last_updated,
        confidence=confidence,
        region=region,
        gwp_set=gwp_set,
        metadata=kwargs
    )


class CitationBundle(BaseModel):
    """
    Bundle of all citations for an agent execution.

    Aggregates emission factors, calculations, and data sources
    used during an agent execution for complete traceability.

    Attributes:
        emission_factors: List of emission factor citations
        calculations: List of calculation citations
        data_sources: List of data source citations
        agent_id: ID of agent that generated citations
        execution_timestamp: When agent was executed
    """

    emission_factors: List[EmissionFactorCitation] = Field(default_factory=list)
    calculations: List[CalculationCitation] = Field(default_factory=list)
    data_sources: List[DataSourceCitation] = Field(default_factory=list)
    agent_id: str = Field(..., description="Agent that generated citations")
    execution_timestamp: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert bundle to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        return {
            "agent_id": self.agent_id,
            "execution_timestamp": self.execution_timestamp.isoformat(),
            "emission_factors": [ef.to_dict() for ef in self.emission_factors],
            "calculations": [calc.dict() for calc in self.calculations],
            "data_sources": [ds.dict() for ds in self.data_sources],
            "total_citations": len(self.emission_factors) + len(self.calculations) + len(self.data_sources),
        }

    def formatted_summary(self) -> str:
        """
        Human-readable summary of all citations.

        Returns:
            Formatted multi-line string
        """
        lines = [
            f"Citations for {self.agent_id} (Executed: {self.execution_timestamp.strftime('%Y-%m-%d %H:%M:%S')})",
            "",
            f"Emission Factors ({len(self.emission_factors)}):",
        ]

        for i, ef in enumerate(self.emission_factors, 1):
            lines.append(f"  {i}. {ef.formatted()}")

        if self.calculations:
            lines.append("")
            lines.append(f"Calculations ({len(self.calculations)}):")
            for i, calc in enumerate(self.calculations, 1):
                lines.append(f"  {i}. {calc.formatted()}")

        if self.data_sources:
            lines.append("")
            lines.append(f"Data Sources ({len(self.data_sources)}):")
            for i, ds in enumerate(self.data_sources, 1):
                lines.append(f"  {i}. {ds.formatted()}")

        return "\n".join(lines)


__all__ = [
    "EmissionFactorCitation",
    "CalculationCitation",
    "DataSourceCitation",
    "CitationBundle",
    "generate_ef_cid",
    "create_emission_factor_citation",
]
